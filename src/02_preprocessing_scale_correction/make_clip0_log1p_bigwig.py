#!/usr/bin/env python
"""Create a clip0+log1p transformed bigWig without densifying coverage."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyBigWig


@dataclass
class IntervalStats:
    interval_count: int = 0
    bases: int = 0
    finite_interval_count: int = 0
    finite_bases: int = 0
    nonfinite_interval_count: int = 0
    nonfinite_bases: int = 0
    negative_interval_count: int = 0
    negative_bases: int = 0
    min_value: float | None = None
    max_value: float | None = None
    weighted_mean: float | None = None


class RunningStats:
    def __init__(self) -> None:
        self.interval_count = 0
        self.bases = 0
        self.finite_interval_count = 0
        self.finite_bases = 0
        self.nonfinite_interval_count = 0
        self.nonfinite_bases = 0
        self.negative_interval_count = 0
        self.negative_bases = 0
        self.min_value: float | None = None
        self.max_value: float | None = None
        self.weighted_sum = 0.0

    def add(self, value: float, length: int) -> None:
        self.interval_count += 1
        self.bases += length
        if not math.isfinite(value):
            self.nonfinite_interval_count += 1
            self.nonfinite_bases += length
            return

        self.finite_interval_count += 1
        self.finite_bases += length
        if value < 0:
            self.negative_interval_count += 1
            self.negative_bases += length
        self.min_value = value if self.min_value is None else min(self.min_value, value)
        self.max_value = value if self.max_value is None else max(self.max_value, value)
        self.weighted_sum += value * length

    def snapshot(self) -> IntervalStats:
        mean = None
        if self.finite_bases:
            mean = self.weighted_sum / self.finite_bases
        return IntervalStats(
            interval_count=self.interval_count,
            bases=self.bases,
            finite_interval_count=self.finite_interval_count,
            finite_bases=self.finite_bases,
            nonfinite_interval_count=self.nonfinite_interval_count,
            nonfinite_bases=self.nonfinite_bases,
            negative_interval_count=self.negative_interval_count,
            negative_bases=self.negative_bases,
            min_value=self.min_value,
            max_value=self.max_value,
            weighted_mean=mean,
        )


def flush_entries(out_bw: pyBigWig.pyBigWig, chrom: str, starts: list[int], ends: list[int], values: list[float]) -> None:
    if not starts:
        return
    out_bw.addEntries([chrom] * len(starts), starts, ends=ends, values=values)
    starts.clear()
    ends.clear()
    values.clear()


def transform_bigwig(input_bw: Path, output_bw: Path, metadata_json: Path, force: bool) -> dict:
    if output_bw.exists() and not force:
        raise FileExistsError(f"Output exists; use --force to replace: {output_bw}")
    if metadata_json.exists() and not force:
        raise FileExistsError(f"Metadata exists; use --force to replace: {metadata_json}")

    output_bw.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)

    tmp_bw = output_bw.with_name(f".{output_bw.stem}.tmp.{os.getpid()}.bw")
    tmp_metadata = metadata_json.with_name(f".{metadata_json.name}.tmp.{os.getpid()}")

    input_stats = RunningStats()
    output_stats = RunningStats()
    skipped_nonfinite = RunningStats()

    in_bw = pyBigWig.open(str(input_bw))
    if in_bw is None:
        raise OSError(f"Could not open input bigWig: {input_bw}")

    chrom_sizes = list(in_bw.chroms().items())
    source_header = in_bw.header()

    out_bw = pyBigWig.open(str(tmp_bw), "w")
    try:
        out_bw.addHeader(chrom_sizes)
        for chrom, _size in chrom_sizes:
            intervals = in_bw.intervals(chrom)
            if not intervals:
                continue

            starts: list[int] = []
            ends: list[int] = []
            values: list[float] = []
            for start, end, raw_value in intervals:
                value = float(raw_value)
                length = int(end) - int(start)
                input_stats.add(value, length)

                if not math.isfinite(value):
                    skipped_nonfinite.add(value, length)
                    continue

                transformed = float(np.log1p(np.maximum(value, 0.0)))
                if not math.isfinite(transformed):
                    skipped_nonfinite.add(transformed, length)
                    continue

                output_stats.add(transformed, length)
                starts.append(int(start))
                ends.append(int(end))
                values.append(transformed)
                if len(starts) >= 100_000:
                    flush_entries(out_bw, chrom, starts, ends, values)
            flush_entries(out_bw, chrom, starts, ends, values)
    except Exception:
        out_bw.close()
        in_bw.close()
        if tmp_bw.exists():
            tmp_bw.unlink()
        raise
    else:
        out_bw.close()
        in_bw.close()

    os.replace(tmp_bw, output_bw)

    verify_bw = pyBigWig.open(str(output_bw))
    if verify_bw is None:
        raise OSError(f"Could not reopen output bigWig: {output_bw}")
    output_header = verify_bw.header()
    output_chroms = verify_bw.chroms()
    verify_bw.close()

    metadata = {
        "source_path": str(input_bw),
        "output_path": str(output_bw),
        "operation_order": ["clip0", "log1p"],
        "formula": "np.log1p(np.maximum(values, 0))",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interval_handling": (
            "Transforms explicit bigWig intervals with their original coordinates. "
            "Absent intervals remain absent; non-finite source intervals are omitted as no-data."
        ),
        "chromosome_count": len(chrom_sizes),
        "chromosomes": dict(chrom_sizes),
        "summary_stats": {
            "source_explicit_intervals": asdict(input_stats.snapshot()),
            "output_written_intervals": asdict(output_stats.snapshot()),
            "skipped_nonfinite_intervals": asdict(skipped_nonfinite.snapshot()),
        },
        "source_bigwig_header": source_header,
        "output_bigwig_header": output_header,
        "verification": {
            "reopened_output": True,
            "output_chromosome_count": len(output_chroms),
        },
    }

    with tmp_metadata.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_metadata, metadata_json)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream explicit bigWig intervals through clip0 then log1p."
    )
    parser.add_argument("--input-bw", required=True, type=Path)
    parser.add_argument("--output-bw", required=True, type=Path)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_json = args.metadata_json
    if metadata_json is None:
        metadata_json = args.output_bw.with_suffix(".metadata.json")
    metadata = transform_bigwig(
        input_bw=args.input_bw,
        output_bw=args.output_bw,
        metadata_json=metadata_json,
        force=args.force,
    )
    print(json.dumps(metadata["summary_stats"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
