#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REGION = "chr2:60005424-61005424"
DEFAULT_BIN_SIZES = (50, 100, 200, 1000)


@dataclass(frozen=True)
class TrackSpec:
    label: str
    prediction_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BCL11A Pearson for one or more prediction bigWigs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ground-truth", required=True, help="Ground-truth bigWig.")
    parser.add_argument(
        "--track",
        action="append",
        required=True,
        help="Prediction track as label=/path/to/prediction.bw. Can be repeated.",
    )
    parser.add_argument("--region", default=DEFAULT_REGION, help="Region as chr:start-end.")
    parser.add_argument(
        "--bin-sizes",
        default=",".join(str(size) for size in DEFAULT_BIN_SIZES),
        help="Comma- or space-separated bin sizes in bp.",
    )
    parser.add_argument("--output-csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--strict", action="store_true", help="Fail if a prediction is missing.")
    return parser.parse_args()


def parse_region(region: str) -> tuple[str, int, int]:
    match = re.fullmatch(r"([^:]+):([0-9,]+)-([0-9,]+)", region.strip())
    if match is None:
        raise ValueError(f"Expected region in chr:start-end format, got {region!r}.")
    chrom = match.group(1)
    start = int(match.group(2).replace(",", ""))
    end = int(match.group(3).replace(",", ""))
    if end <= start:
        raise ValueError(f"Region end must be greater than start: {region}")
    return chrom, start, end


def parse_bin_sizes(raw_bin_sizes: str) -> tuple[int, ...]:
    sizes: list[int] = []
    for token in re.split(r"[\s,]+", raw_bin_sizes.strip()):
        if not token:
            continue
        size = int(token)
        if size <= 0:
            raise ValueError(f"Bin sizes must be positive, got {size}.")
        sizes.append(size)
    if not sizes:
        raise ValueError("--bin-sizes must include at least one positive integer.")
    return tuple(sizes)


def parse_track(raw_track: str) -> TrackSpec:
    if "=" not in raw_track:
        raise ValueError(
            f"Expected --track as label=/path/to/prediction.bw, got {raw_track!r}."
        )
    label, raw_path = raw_track.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Track label is empty in {raw_track!r}.")
    return TrackSpec(label=label, prediction_path=Path(raw_path).expanduser())


def read_binned_bigwig(path: Path, chrom: str, start: int, end: int, bin_size: int):
    import numpy as np
    import pyBigWig

    length = end - start
    if length % bin_size != 0:
        raise ValueError(
            f"Region length {length} bp is not divisible by bin size {bin_size} bp."
        )

    handle = pyBigWig.open(str(path))
    try:
        if handle is None:
            raise OSError(f"Could not open bigWig: {path}")
        chroms = handle.chroms()
        if chrom not in chroms:
            raise ValueError(f"{path} does not contain chromosome {chrom}.")
        stats = handle.stats(chrom, start, end, nBins=length // bin_size, type="mean", exact=True)
    finally:
        if handle is not None:
            handle.close()

    return np.asarray([np.nan if value is None else float(value) for value in stats], dtype=float)


def pearson_for_values(ground_truth, prediction) -> tuple[str, str, int]:
    import numpy as np

    finite_mask = np.isfinite(ground_truth) & np.isfinite(prediction)
    n_finite = int(finite_mask.sum())
    if n_finite < 2:
        return "", "insufficient_finite_pairs", n_finite

    ground_truth_values = ground_truth[finite_mask]
    prediction_values = prediction[finite_mask]
    if float(np.std(ground_truth_values)) == 0.0 or float(np.std(prediction_values)) == 0.0:
        return "", "constant_values", n_finite

    pearson = float(np.corrcoef(ground_truth_values, prediction_values)[0, 1])
    if not np.isfinite(pearson):
        return "", "nonfinite_pearson", n_finite
    return f"{pearson:.10g}", "ok", n_finite


def missing_rows(
    track: TrackSpec,
    ground_truth_path: Path,
    region: str,
    bin_sizes: tuple[int, ...],
    status: str,
    message: str = "",
) -> list[dict[str, str]]:
    return [
        {
            "label": track.label,
            "status": status,
            "region": region,
            "bin_size_bp": str(bin_size),
            "n_bins": "",
            "n_finite_pairs": "",
            "pearson": "",
            "prediction_path": str(track.prediction_path),
            "ground_truth_path": str(ground_truth_path),
            "message": message,
        }
        for bin_size in bin_sizes
    ]


def evaluate_track(
    track: TrackSpec,
    ground_truth_by_bin: dict[int, object],
    ground_truth_path: Path,
    chrom: str,
    start: int,
    end: int,
    region: str,
    bin_sizes: tuple[int, ...],
    *,
    strict: bool,
) -> list[dict[str, str]]:
    if not track.prediction_path.is_file():
        if strict:
            raise FileNotFoundError(
                f"Missing prediction bigWig for {track.label}: {track.prediction_path}"
            )
        return missing_rows(track, ground_truth_path, region, bin_sizes, "missing")

    rows: list[dict[str, str]] = []
    for bin_size in bin_sizes:
        try:
            prediction = read_binned_bigwig(track.prediction_path, chrom, start, end, bin_size)
            pearson, status, n_finite = pearson_for_values(
                ground_truth_by_bin[bin_size],
                prediction,
            )
        except Exception as exc:
            if strict:
                raise
            rows.append(
                {
                    "label": track.label,
                    "status": "error",
                    "region": region,
                    "bin_size_bp": str(bin_size),
                    "n_bins": "",
                    "n_finite_pairs": "",
                    "pearson": "",
                    "prediction_path": str(track.prediction_path),
                    "ground_truth_path": str(ground_truth_path),
                    "message": str(exc),
                }
            )
            continue

        rows.append(
            {
                "label": track.label,
                "status": status,
                "region": region,
                "bin_size_bp": str(bin_size),
                "n_bins": str(len(ground_truth_by_bin[bin_size])),
                "n_finite_pairs": str(n_finite),
                "pearson": pearson,
                "prediction_path": str(track.prediction_path),
                "ground_truth_path": str(ground_truth_path),
                "message": "",
            }
        )
    return rows


def write_table(rows: list[dict[str, str]], handle, *, delimiter: str) -> None:
    fieldnames = [
        "label",
        "status",
        "region",
        "bin_size_bp",
        "n_bins",
        "n_finite_pairs",
        "pearson",
        "prediction_path",
        "ground_truth_path",
        "message",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def main() -> int:
    args = parse_args()
    chrom, start, end = parse_region(args.region)
    bin_sizes = parse_bin_sizes(args.bin_sizes)
    ground_truth_path = Path(args.ground_truth).expanduser()
    tracks = [parse_track(raw_track) for raw_track in args.track]

    if not ground_truth_path.is_file():
        raise FileNotFoundError(f"Missing ground-truth bigWig: {ground_truth_path}")

    ground_truth_by_bin = {
        bin_size: read_binned_bigwig(ground_truth_path, chrom, start, end, bin_size)
        for bin_size in bin_sizes
    }

    rows: list[dict[str, str]] = []
    for track in tracks:
        rows.extend(
            evaluate_track(
                track,
                ground_truth_by_bin,
                ground_truth_path,
                chrom,
                start,
                end,
                args.region,
                bin_sizes,
                strict=args.strict,
            )
        )

    write_table(rows, sys.stdout, delimiter="\t")

    if args.output_csv is not None:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            write_table(rows, handle, delimiter=",")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
