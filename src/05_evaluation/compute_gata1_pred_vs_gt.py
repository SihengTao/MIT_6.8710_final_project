#!/usr/bin/env python
"""Compute GATA1 prediction and ATAC baseline correlations against GATA1 ChIP."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig


OUT_DIR = Path(__file__).resolve().parent

GT_CHIP_BW = Path(
    "/broad/boxialab/sihengtao/projects/check4hic2_anewPaper/"
    "chrom2vec_output/SRR21983756_SRR21983758/s9_bigwig/genrich_normalized.bw"
)
ATAC_BW = Path(
    "/broad/boxialab/sihengtao/projects/check4hic2_anewPaper/"
    "chrom2vec_output/SRR21983447_SRR21983446/s9_bigwig/genrich_normalized.bw"
)

LOCI = {
    "BCL11A": {
        "chrom": "chr2",
        "start": 60005424,
        "end": 61005424,
        "prediction_bw": Path(
            "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
            "gata1_from_baseonly_scratch_r4_lr1e4_ep10_2024paper/20260502-154138/"
            "hic2_1mb_bigwig/prediction/prediction/"
            "check4hic2_anewPaper_single_sample/GATA1/bigwigs/"
            "GATA1_check4hic2_anewPaper_single_sample_prediction.bw"
        ),
    },
    "ASXL1": {
        "chrom": "chr20",
        "start": 31898825,
        "end": 32898825,
        "prediction_bw": Path(
            "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
            "gata1_from_baseonly_scratch_r4_lr1e4_ep10_2024paper/20260502-154138/"
            "hic2_1mb_bigwig_chr20_asxl1/prediction/prediction/"
            "check4hic2_anewPaper_single_sample/GATA1/bigwigs/"
            "GATA1_check4hic2_anewPaper_single_sample_prediction.bw"
        ),
    },
}

BIN_SIZES = (100, 200, 1000)
MAIN_BIN_SIZE = 1000
SOURCE_LABEL = "check4hic2_anewPaper_single_sample"
SOURCE_NOTE = (
    "GATA1 prediction/ChIP/ATAC here is from check4hic2_anewPaper_single_sample "
    "and check4hic2_anewPaper chrom2vec raw inputs, not 2022paper GATA1."
)


def read_bigwig(path: Path, chrom: str, start: int, end: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)

    bw = pyBigWig.open(str(path))
    try:
        chrom_sizes = bw.chroms()
        if chrom not in chrom_sizes:
            raise ValueError(f"{chrom} is absent from {path}")
        if end > chrom_sizes[chrom]:
            raise ValueError(f"{chrom}:{start}-{end} exceeds chromosome length in {path}")
        values = bw.values(chrom, start, end, numpy=True)
    finally:
        bw.close()

    values = np.asarray(values, dtype=np.float64)
    expected_len = end - start
    if values.size != expected_len:
        raise ValueError(f"{path} returned {values.size} values, expected {expected_len}")
    return values


def preprocess_prediction(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    values[~np.isfinite(values)] = np.nan
    return values


def preprocess_chrom2vec_raw(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    values[~np.isfinite(values)] = 0.0
    values = np.clip(values, 0.0, None)
    return np.log1p(values)


def bin_mean(values: np.ndarray, bin_size: int) -> np.ndarray:
    n_bins = values.size // bin_size
    if n_bins == 0:
        raise ValueError(f"Cannot bin {values.size} values at bin_size={bin_size}")
    trimmed = values[: n_bins * bin_size].reshape(n_bins, bin_size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(trimmed, axis=1)


def pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 2:
        return math.nan, n
    x_finite = x[mask]
    y_finite = y[mask]
    if np.std(x_finite) == 0 or np.std(y_finite) == 0:
        return math.nan, n
    return float(np.corrcoef(x_finite, y_finite)[0, 1]), n


def compute_metrics() -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int], dict[str, np.ndarray]]]:
    metrics_rows = []
    track_rows = []
    binned_by_locus = {}

    for locus, cfg in LOCI.items():
        chrom = cfg["chrom"]
        start = cfg["start"]
        end = cfg["end"]

        pred_bp = preprocess_prediction(read_bigwig(cfg["prediction_bw"], chrom, start, end))
        gt_bp = preprocess_chrom2vec_raw(read_bigwig(GT_CHIP_BW, chrom, start, end))
        atac_bp = preprocess_chrom2vec_raw(read_bigwig(ATAC_BW, chrom, start, end))

        for bin_size in BIN_SIZES:
            pred_bin = bin_mean(pred_bp, bin_size)
            gt_bin = bin_mean(gt_bp, bin_size)
            atac_bin = bin_mean(atac_bp, bin_size)
            binned_by_locus[(locus, bin_size)] = {
                "prediction": pred_bin,
                "gata1_chip_gt": gt_bin,
                "atac_baseline": atac_bin,
            }

            for comparison, signal in (
                ("prediction_vs_gt", pred_bin),
                ("atac_baseline_vs_gt", atac_bin),
            ):
                r, n_finite = pearson(signal, gt_bin)
                metrics_rows.append(
                    {
                        "locus": locus,
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "bin_size": bin_size,
                        "comparison": comparison,
                        "pearson": r,
                        "n_bins": int(gt_bin.size),
                        "n_finite_pairs": n_finite,
                        "source_label": SOURCE_LABEL,
                    }
                )

            if bin_size == MAIN_BIN_SIZE:
                bin_starts = start + np.arange(gt_bin.size) * bin_size
                for idx, bin_start in enumerate(bin_starts):
                    track_rows.append(
                        {
                            "locus": locus,
                            "chrom": chrom,
                            "bin_index": int(idx),
                            "bin_start": int(bin_start),
                            "bin_end": int(bin_start + bin_size),
                            "prediction_1kb": pred_bin[idx],
                            "gata1_chip_gt_log1p_1kb": gt_bin[idx],
                            "atac_baseline_log1p_1kb": atac_bin[idx],
                        }
                    )

    metrics = pd.DataFrame(metrics_rows)
    tracks = pd.DataFrame(track_rows)
    return metrics, tracks, binned_by_locus


def write_manifest() -> None:
    rows = []
    for locus, cfg in LOCI.items():
        rows.append(
            {
                "role": "prediction",
                "locus": locus,
                "source_label": SOURCE_LABEL,
                "path": str(cfg["prediction_bw"]),
                "transform": "as-is; finite values averaged into non-overlapping bins",
                "notes": "Do not log1p prediction again.",
            }
        )
    for role, path in (
        ("gata1_chip_ground_truth", GT_CHIP_BW),
        ("matched_atac_baseline", ATAC_BW),
    ):
        rows.append(
            {
                "role": role,
                "locus": "BCL11A;ASXL1",
                "source_label": "check4hic2_anewPaper chrom2vec raw",
                "path": str(path),
                "transform": "missing/NaN/inf to 0; negative clipped to 0; log1p at bp level; non-overlapping bin mean",
                "notes": "Read-only input under /broad/boxialab/sihengtao/projects.",
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "source_manifest.tsv", sep="\t", index=False)


def write_figure(metrics: pd.DataFrame, binned_by_locus: dict[tuple[str, int], dict[str, np.ndarray]]) -> None:
    main_metrics = metrics[metrics["bin_size"] == MAIN_BIN_SIZE].copy()
    metric_lookup = {
        (row.locus, row.comparison): row.pearson for row in main_metrics.itertuples(index=False)
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.0), constrained_layout=True)
    comparisons = [
        ("prediction_vs_gt", "prediction", "Prediction vs GATA1 ChIP", "#0072B2"),
        ("atac_baseline_vs_gt", "atac_baseline", "ATAC baseline vs GATA1 ChIP", "#6A6A6A"),
    ]

    for row_idx, locus in enumerate(LOCI):
        binned = binned_by_locus[(locus, MAIN_BIN_SIZE)]
        gt = binned["gata1_chip_gt"]
        for col_idx, (comparison, signal_key, title, color) in enumerate(comparisons):
            ax = axes[row_idx, col_idx]
            y = binned[signal_key]
            mask = np.isfinite(gt) & np.isfinite(y)
            ax.scatter(gt[mask], y[mask], s=12, color=color, alpha=0.75, linewidths=0)
            if int(mask.sum()) >= 2 and np.std(gt[mask]) > 0 and np.std(y[mask]) > 0:
                slope, intercept = np.polyfit(gt[mask], y[mask], deg=1)
                x_line = np.linspace(float(np.min(gt[mask])), float(np.max(gt[mask])), 100)
                ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=0.9, alpha=0.75)
            r = metric_lookup[(locus, comparison)]
            ax.text(
                0.03,
                0.95,
                f"r = {r:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )
            ax.set_title(f"{locus}: {title}", fontsize=10)
            ax.set_xlabel("GATA1 ChIP log1p mean, 1kb", fontsize=9)
            ylabel = "Prediction mean, 1kb" if signal_key == "prediction" else "ATAC log1p mean, 1kb"
            ax.set_ylabel(ylabel, fontsize=9)
            ax.tick_params(axis="both", labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle("GATA1 1kb Pearson against ChIP ground truth", fontsize=12)
    fig.text(0.5, 0.005, SOURCE_NOTE, ha="center", va="bottom", fontsize=8)
    fig.savefig(OUT_DIR / "figure_gata1_pred_vs_gt_1kb.png", dpi=300)
    fig.savefig(OUT_DIR / "figure_gata1_pred_vs_gt_1kb.pdf")
    plt.close(fig)


def format_pearson_table(metrics: pd.DataFrame) -> str:
    main = metrics[metrics["bin_size"] == MAIN_BIN_SIZE]
    rows = ["| Locus | Prediction vs GATA1 ChIP | ATAC baseline vs GATA1 ChIP |", "| --- | ---: | ---: |"]
    for locus in LOCI:
        pred_r = main[(main["locus"] == locus) & (main["comparison"] == "prediction_vs_gt")][
            "pearson"
        ].iloc[0]
        atac_r = main[(main["locus"] == locus) & (main["comparison"] == "atac_baseline_vs_gt")][
            "pearson"
        ].iloc[0]
        rows.append(f"| {locus} | {pred_r:.6f} | {atac_r:.6f} |")
    return "\n".join(rows)


def write_readme(metrics: pd.DataFrame) -> None:
    pearson_table = format_pearson_table(metrics)
    readme = f"""# GATA1 Prediction vs ChIP Ground Truth, 1kb

This folder computes Pearson correlations between GATA1 predictions, matched ATAC baseline, and GATA1 ChIP ground truth over two 1 Mb loci.

Important source label: this is not 2022paper GATA1. The prediction label is `{SOURCE_LABEL}`, and the ChIP/ATAC ground truth tracks come from `check4hic2_anewPaper` chrom2vec raw outputs.

## Main 1kb Pearson

{pearson_table}

## Compute rule

- Prediction: values are used as stored, with no additional log1p; non-overlapping bin mean.
- GATA1 ChIP and matched ATAC chrom2vec raw: missing, NaN, and inf values are set to 0; negative values are clipped to 0; log1p is applied at bp level; non-overlapping bin mean.
- Pearson is computed on finite paired bins only.
- Main result uses 1000 bp bins. A supplemental resolution sweep is included for 100, 200, and 1000 bp bins.

## Files

- `compute_gata1_pred_vs_gt.py`: reproducible script.
- `gata1_pred_vs_gt_1kb_metrics.tsv`: main 1kb metrics.
- `gata1_resolution_sweep_metrics.tsv`: supplemental 100/200/1000 bp metrics.
- `gata1_pred_vs_gt_1kb_tracks.tsv`: 1kb binned tracks used for plotting.
- `figure_gata1_pred_vs_gt_1kb.png` and `.pdf`: compact scatter panels.
- `source_manifest.tsv`: input paths and transforms.
- `caption.md`: figure caption.
"""
    (OUT_DIR / "README.md").write_text(readme)


def write_caption() -> None:
    caption = f"""Figure: GATA1 1kb Pearson against ChIP ground truth.

Each point is a non-overlapping 1kb bin within the indicated 1 Mb locus. Prediction values are used as stored. GATA1 ChIP and matched ATAC chrom2vec raw signals are cleaned, clipped at zero, log1p transformed at bp level, and averaged into 1kb bins. Pearson r is computed on finite paired bins only. Source label: `{SOURCE_LABEL}`; this is not a 2022paper GATA1 result.
"""
    (OUT_DIR / "caption.md").write_text(caption)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics, tracks, binned_by_locus = compute_metrics()

    main_metrics = metrics[metrics["bin_size"] == MAIN_BIN_SIZE].copy()
    main_metrics.to_csv(OUT_DIR / "gata1_pred_vs_gt_1kb_metrics.tsv", sep="\t", index=False)
    metrics.to_csv(OUT_DIR / "gata1_resolution_sweep_metrics.tsv", sep="\t", index=False)
    tracks.to_csv(OUT_DIR / "gata1_pred_vs_gt_1kb_tracks.tsv", sep="\t", index=False)

    write_manifest()
    write_figure(metrics, binned_by_locus)
    write_readme(metrics)
    write_caption()

    print(format_pearson_table(metrics))


if __name__ == "__main__":
    main()
