#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_GROUND_TRUTH = Path(
    "/broad/boxialab/sihengtao/projects/check4hic2_anewPaper/"
    "chrom2vec_output/SRR21983756_SRR21983758/s9_bigwig/genrich_normalized.bw"
)
DEFAULT_REGION = "chr2:60005424-61005424"
DEFAULT_BIN_SIZES = (50, 100, 200, 1000)
CELLTYPE = "check4hic2_anewPaper_single_sample"
CAP = "GATA1"
TRANSFER_OUTPUT_NAME = "hic2_1mb_bigwig_gata1_transfer_1mb"
OFFICIAL_GATA1_CONTROL = Path(
    "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
    "gata1_official_bcl11a_1mb/check4hic2_anewPaper_single_sample/"
    "GATA1/processed/data.bigwig"
)
BASE_ONLY_GATA1_CONTROL = Path(
    "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
    "gata1_base_only_bcl11a_1mb/check4hic2_anewPaper_single_sample/"
    "GATA1/processed/gata1_base.bigwig"
)


@dataclass(frozen=True)
class TrackSpec:
    label: str
    source: str
    prediction_path: Path
    run_dir: Path | None = None


def transfer_prediction_path(run_dir: Path) -> Path:
    return (
        run_dir
        / TRANSFER_OUTPUT_NAME
        / "prediction"
        / "prediction"
        / CELLTYPE
        / CAP
        / "bigwigs"
        / f"{CAP}_{CELLTYPE}_prediction.bw"
    )


def transfer_track(label: str, run_dir: str) -> TrackSpec:
    run_path = Path(run_dir)
    return TrackSpec(
        label=label,
        source="hic2_to_gata1_transfer",
        run_dir=run_path,
        prediction_path=transfer_prediction_path(run_path),
    )


DEFAULT_TRANSFER_TRACKS = (
    transfer_track(
        "hic2_target_A_log1p_clip_mse_r4_base_8ep",
        "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
        "hic2_target_A_log1p_clip_mse_r4_base_lr3em4_8ep/20260426-145616",
    ),
    transfer_track(
        "hic2_warmstart_no_atac_dropout",
        "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
        "hic2_warmstart_no_atac_dropout/20260329-164943",
    ),
    transfer_track(
        "hic2_warmstart_with_atac_dropout",
        "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
        "hic2_warmstart_no_atac_dropout/hic2_warmstart_with_atac_dropout/20260328-230921",
    ),
    transfer_track(
        "hic2_loss_weighted_multiscale_pearsonon_r4_base_5ep",
        "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
        "hic2_loss_weighted_multiscale_pearsonon_r4_base_lr3em4_5ep/20260425-184209",
    ),
    transfer_track(
        "hic2_loss_weighted_multiscale_pearsonon_r4_fullws_5ep",
        "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
        "hic2_loss_weighted_multiscale_pearsonon_r4_fullws_lr3em4_5ep/20260426-020616",
    ),
    transfer_track(
        "hic2_loss_weighted_multiscale_pearsonoff_r4_base_5ep",
        "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
        "hic2_loss_weighted_multiscale_pearsonoff_r4_base_lr3em4_5ep/20260425-160431",
    ),
    transfer_track(
        "hic2_loss_weighted_multiscale_pearsonoff_r4_fullws_5ep",
        "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
        "hic2_loss_weighted_multiscale_pearsonoff_r4_fullws_lr3em4_5ep/20260426-020622",
    ),
    transfer_track(
        "hic2_loss_mse_pearsonoff_r4_base_5ep",
        "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
        "hic2_loss_mse_pearsonoff_r4_base_lr3em4_5ep/20260425-160431",
    ),
    transfer_track(
        "hic2_loss_mse_pearsonoff_r4_fullws_5ep",
        "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
        "hic2_loss_mse_pearsonoff_r4_fullws_lr3em4_5ep/20260426-020622",
    ),
    transfer_track(
        "hic2_loss_mse_pearsonoff_r4_fullws_5ep_v2",
        "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
        "hic2_loss_mse_pearsonoff_r4_fullws_lr3em4_5ep/20260427-162712",
    ),
    transfer_track(
        "hic2_target_A_log1p_clip_mse_r8_base_8ep",
        "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
        "hic2_target_A_log1p_clip_mse_r8_base_lr1em3_8ep/20260427-035224",
    ),
    transfer_track(
        "hic2_target_B_clip_mse_r4_base_8ep",
        "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
        "hic2_target_B_clip_mse_r4_base_lr3em4_8ep/20260426-145616",
    ),
    transfer_track(
        "hic2_target_B_clip_mse_r8_base_8ep",
        "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
        "hic2_target_B_clip_mse_r8_base_lr1em3_8ep/20260427-035224",
    ),
    transfer_track(
        "hic2_target_D_gamma2_mse_r4_base_8ep",
        "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
        "hic2_target_D_gamma2_mse_r4_base_lr3em4_8ep/20260426-145616",
    ),
    transfer_track(
        "hic2_target_D_gamma2_mse_r8_base_8ep",
        "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
        "hic2_target_D_gamma2_mse_r8_base_lr1em3_8ep/20260427-035224",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate BCL11A Pearson between corrected GATA1 ground truth and "
            "HIC2 adapters exported with GATA1 CAP/protein."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ground-truth",
        default=str(DEFAULT_GROUND_TRUTH),
        help="Corrected GATA1 ground-truth bigWig.",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="Evaluation region as chr:start-end.",
    )
    parser.add_argument(
        "--bin-sizes",
        default=",".join(str(size) for size in DEFAULT_BIN_SIZES),
        help="Comma- or space-separated bin sizes in bp.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV output path. TSV is always printed to stdout.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any default HIC2 transfer prediction is missing or unreadable.",
    )
    parser.add_argument(
        "--no-controls",
        action="store_true",
        help="Do not append official GATA1 and base-only GATA1 controls.",
    )
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
        raise ValueError("--bin-sizes must contain at least one positive integer.")
    return tuple(sizes)


def control_tracks() -> list[TrackSpec]:
    tracks: list[TrackSpec] = []
    if OFFICIAL_GATA1_CONTROL.is_file():
        tracks.append(
            TrackSpec(
                label="official_gata1",
                source="control",
                prediction_path=OFFICIAL_GATA1_CONTROL,
            )
        )
    if BASE_ONLY_GATA1_CONTROL.is_file():
        tracks.append(
            TrackSpec(
                label="base_only_gata1",
                source="control",
                prediction_path=BASE_ONLY_GATA1_CONTROL,
            )
        )
    return tracks


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

    gt_values = ground_truth[finite_mask]
    pred_values = prediction[finite_mask]
    if float(np.std(gt_values)) == 0.0 or float(np.std(pred_values)) == 0.0:
        return "", "constant_values", n_finite

    pearson = float(np.corrcoef(gt_values, pred_values)[0, 1])
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
            "source": track.source,
            "status": status,
            "region": region,
            "bin_size_bp": str(bin_size),
            "n_bins": "",
            "n_finite_pairs": "",
            "pearson": "",
            "prediction_path": str(track.prediction_path),
            "ground_truth_path": str(ground_truth_path),
            "run_dir": "" if track.run_dir is None else str(track.run_dir),
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
        if strict and track.source == "hic2_to_gata1_transfer":
            raise FileNotFoundError(f"Missing prediction bigWig for {track.label}: {track.prediction_path}")
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
            if strict and track.source == "hic2_to_gata1_transfer":
                raise
            rows.append(
                {
                    "label": track.label,
                    "source": track.source,
                    "status": "error",
                    "region": region,
                    "bin_size_bp": str(bin_size),
                    "n_bins": "",
                    "n_finite_pairs": "",
                    "pearson": "",
                    "prediction_path": str(track.prediction_path),
                    "ground_truth_path": str(ground_truth_path),
                    "run_dir": "" if track.run_dir is None else str(track.run_dir),
                    "message": str(exc),
                }
            )
            continue

        rows.append(
            {
                "label": track.label,
                "source": track.source,
                "status": status,
                "region": region,
                "bin_size_bp": str(bin_size),
                "n_bins": str(len(ground_truth_by_bin[bin_size])),
                "n_finite_pairs": str(n_finite),
                "pearson": pearson,
                "prediction_path": str(track.prediction_path),
                "ground_truth_path": str(ground_truth_path),
                "run_dir": "" if track.run_dir is None else str(track.run_dir),
                "message": "",
            }
        )
    return rows


def write_table(rows: list[dict[str, str]], handle, *, delimiter: str) -> None:
    fieldnames = [
        "label",
        "source",
        "status",
        "region",
        "bin_size_bp",
        "n_bins",
        "n_finite_pairs",
        "pearson",
        "prediction_path",
        "ground_truth_path",
        "run_dir",
        "message",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def main() -> int:
    args = parse_args()
    chrom, start, end = parse_region(args.region)
    bin_sizes = parse_bin_sizes(args.bin_sizes)
    ground_truth_path = Path(args.ground_truth)

    if not ground_truth_path.is_file():
        raise FileNotFoundError(f"Missing ground-truth bigWig: {ground_truth_path}")

    tracks = list(DEFAULT_TRANSFER_TRACKS)
    if not args.no_controls:
        tracks.extend(control_tracks())

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
