#!/usr/bin/env python3
"""Generate the clean 100 bp GATA1 report figure."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
ONE_KB_METRICS = OUT_DIR / "gata1_pred_vs_gt_1kb_metrics.tsv"
SWEEP_METRICS = OUT_DIR / "gata1_resolution_sweep_metrics.tsv"
MANIFEST = OUT_DIR / "source_manifest.tsv"
METRICS = OUT_DIR / "gata1_pred_vs_gt_100bp_metrics.tsv"
SOURCE = OUT_DIR / "figure5_gata1_pred_vs_gt_100bp_clean_source.tsv"
BIN_SIZE = 100
LOCI = ["BCL11A", "ASXL1"]

COMPARISON_LABELS = {
    "prediction_vs_gt": "Prediction",
    "atac_baseline_vs_gt": "ATAC baseline",
}


def _load_existing_100bp_metrics() -> Optional[pd.DataFrame]:
    if not SWEEP_METRICS.exists():
        return None

    metrics = pd.read_csv(SWEEP_METRICS, sep="\t")
    source = metrics[
        (metrics["bin_size"] == BIN_SIZE)
        & (metrics["locus"].isin(LOCI))
        & (metrics["comparison"].isin(COMPARISON_LABELS))
    ].copy()
    if len(source) != 4:
        return None
    return source


def _regions_from_1kb_metrics() -> dict[str, tuple[str, int, int]]:
    metrics = pd.read_csv(ONE_KB_METRICS, sep="\t")
    regions: dict[str, tuple[str, int, int]] = {}
    for locus in LOCI:
        locus_rows = metrics[metrics["locus"] == locus][["chrom", "start", "end"]].drop_duplicates()
        if len(locus_rows) != 1:
            raise ValueError(f"Expected one region for {locus}, found {len(locus_rows)}")
        row = locus_rows.iloc[0]
        regions[locus] = (row["chrom"], int(row["start"]), int(row["end"]))
    return regions


def _manifest_paths() -> dict[tuple[str, str], str]:
    manifest = pd.read_csv(MANIFEST, sep="\t")
    paths: dict[tuple[str, str], str] = {}
    for _, row in manifest.iterrows():
        for locus in str(row["locus"]).split(";"):
            paths[(str(row["role"]), locus)] = str(row["path"])
    return paths


def _load_bigwig_values(path: str, chrom: str, start: int, end: int, clean_log1p: bool) -> np.ndarray:
    import pyBigWig

    with pyBigWig.open(path) as bw:
        values = np.asarray(bw.values(chrom, start, end, numpy=True), dtype=np.float64)

    if not clean_log1p:
        return values

    values = np.where(np.isfinite(values), values, 0.0)
    values[values < 0] = 0.0
    return np.log1p(values)


def _bin_mean(values: np.ndarray, bin_size: int, finite_only: bool) -> np.ndarray:
    n_bins = values.size // bin_size
    binned = values[: n_bins * bin_size].reshape(n_bins, bin_size)
    if not finite_only:
        return binned.mean(axis=1)

    finite = np.isfinite(binned)
    counts = finite.sum(axis=1)
    out = np.full(n_bins, np.nan, dtype=np.float64)
    valid = counts > 0
    out[valid] = np.where(finite[valid], binned[valid], 0.0).sum(axis=1) / counts[valid]
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    finite = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[finite], y[finite])[0, 1]), int(finite.sum())


def _compute_100bp_metrics() -> pd.DataFrame:
    regions = _regions_from_1kb_metrics()
    paths = _manifest_paths()
    rows = []

    for locus in LOCI:
        chrom, start, end = regions[locus]
        prediction = _bin_mean(
            _load_bigwig_values(paths[("prediction", locus)], chrom, start, end, clean_log1p=False),
            BIN_SIZE,
            finite_only=True,
        )
        gt_chip = _bin_mean(
            _load_bigwig_values(paths[("gata1_chip_ground_truth", locus)], chrom, start, end, clean_log1p=True),
            BIN_SIZE,
            finite_only=False,
        )
        atac = _bin_mean(
            _load_bigwig_values(paths[("matched_atac_baseline", locus)], chrom, start, end, clean_log1p=True),
            BIN_SIZE,
            finite_only=False,
        )
        source_label = pd.read_csv(MANIFEST, sep="\t")
        source_label = source_label[
            (source_label["role"] == "prediction") & (source_label["locus"] == locus)
        ]["source_label"].iloc[0]

        for comparison, signal in [
            ("prediction_vs_gt", prediction),
            ("atac_baseline_vs_gt", atac),
        ]:
            r, n_finite = _pearson(signal, gt_chip)
            rows.append(
                {
                    "locus": locus,
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "bin_size": BIN_SIZE,
                    "comparison": comparison,
                    "pearson": r,
                    "n_bins": len(gt_chip),
                    "n_finite_pairs": n_finite,
                    "source_label": source_label,
                }
            )

    return pd.DataFrame(rows)


def collect_metrics() -> pd.DataFrame:
    metrics = _load_existing_100bp_metrics()
    if metrics is None:
        metrics = _compute_100bp_metrics()
    metrics = metrics.sort_values(["locus", "comparison"]).copy()
    metrics["plot_order"] = metrics["locus"].map({"BCL11A": 0, "ASXL1": 1}) * 2
    metrics["plot_order"] += metrics["comparison"].map({"prediction_vs_gt": 0, "atac_baseline_vs_gt": 1})
    return metrics.sort_values("plot_order").drop(columns="plot_order")


def collect_source(metrics: pd.DataFrame) -> pd.DataFrame:
    source = metrics.copy()
    source["display_label"] = source["comparison"].map(COMPARISON_LABELS)
    source["plot_label"] = source["locus"] + " - " + source["display_label"]
    source["source_table"] = str(METRICS)
    source["source_note"] = "GATA1 matched sample: check4hic2_anewPaper_single_sample"
    source["plot_order"] = source["locus"].map({"BCL11A": 0, "ASXL1": 1}) * 2
    source["plot_order"] += source["comparison"].map({"prediction_vs_gt": 0, "atac_baseline_vs_gt": 1})
    return source.sort_values("plot_order")


def plot(source: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = {
        "prediction_vs_gt": "#287c79",
        "atac_baseline_vs_gt": "#9aa0a6",
    }
    plot_df = source.sort_values("plot_order").reset_index(drop=True)
    y_pos = list(range(len(plot_df)))[::-1]

    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    ax.barh(
        y_pos,
        plot_df["pearson"],
        height=0.62,
        color=[colors[v] for v in plot_df["comparison"]],
        edgecolor="white",
        linewidth=0.9,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["plot_label"])
    ax.set_xlabel("Pearson vs GATA1 ChIP at 100 bp")
    ax.set_xlim(0, 1.0)
    ax.set_title(
        "Figure 5. GATA1 prediction outperforms ATAC baseline\n"
        "GATA1 matched sample: check4hic2_anewPaper",
        loc="left",
        fontsize=11,
        pad=10,
    )

    for y, value in zip(y_pos, plot_df["pearson"]):
        ax.text(value + 0.018, y, f"{value:.3f}", va="center", ha="left", fontsize=8.6)

    ax.axhline(1.5, color="#d5d9de", linewidth=0.8)
    fig.tight_layout(pad=1.1)
    fig.savefig(OUT_DIR / "figure5_gata1_pred_vs_gt_100bp_clean.png", dpi=450, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure5_gata1_pred_vs_gt_100bp_clean.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics = collect_metrics()
    metrics.to_csv(METRICS, sep="\t", index=False)
    source = collect_source(metrics)
    source.to_csv(SOURCE, sep="\t", index=False)
    plot(source)


if __name__ == "__main__":
    main()
