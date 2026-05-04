#!/usr/bin/env python3
"""Generate the clean report version of Figure 4 from corrected 100 bp metrics."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/broad/boxialab/sihengtao/projects/chromnitron_finetune")
CORRECTED_DIR = PROJECT_ROOT / "20260503_analysis/figure4_bcl11a_100bp_corrected"
SUMMARY = CORRECTED_DIR / "figure4_bcl11a_pearson_100bp_corrected_summary.tsv"
DETAIL = CORRECTED_DIR / "figure4_bcl11a_pearson_100bp_corrected.tsv"
INITIAL_GRID_SEARCH_LABEL = "Initial grid-search fine-tune"
CLEAN_PNG = OUT_DIR / "figure4_finetuning_hyperparameter_selection_clean.png"
CLEAN_PDF = OUT_DIR / "figure4_finetuning_hyperparameter_selection_clean.pdf"
SIMPLE_PNG = OUT_DIR / "figure4.png"
SIMPLE_PDF = OUT_DIR / "figure4.pdf"
LEGACY_PNG = OUT_DIR / "figure4_finetuning_hyperparameter_selection.png"
LEGACY_PDF = OUT_DIR / "figure4_finetuning_hyperparameter_selection.pdf"
CLEAN_SOURCE_TSV = OUT_DIR / "figure4_finetuning_hyperparameter_selection_clean_source.tsv"
LEGACY_SOURCE_TSV = OUT_DIR / "figure4_finetuning_hyperparameter_selection_source.tsv"
MAPPING_TSV = OUT_DIR / "figure4_candidate_mapping.tsv"
BCL11A_REGION = "chr2:60005424-61005424"
PLOT_BIN_SIZE_BP = 100
PLOT_METRIC = "mean_pearson_100bp"
SCALE_RULE = "prediction as-is; ground-truth ChIP clip0 + log1p"


def parse_config(config: str) -> dict[str, str]:
    config_lower = config.lower()
    rank_match = re.search(r"(?:^|_)r([0-9]+)(?:_|$)", config)
    lr_match = re.search(r"_lr([0-9]+)em([0-9]+)_", config)
    epoch_match = re.search(r"_([0-9]+)ep$", config)

    if "target_A" in config:
        family = "target A log1p+clip"
        loss = "MSE"
    elif "target_B" in config:
        family = "target B clip"
        loss = "MSE"
    elif "target_D" in config:
        family = "target D gamma2"
        loss = "MSE"
    elif "loss_weighted_multiscale" in config:
        family = "weighted multiscale"
        loss = "weighted multiscale"
    elif "loss_mse" in config:
        family = "standard target"
        loss = "MSE"
    else:
        family = "other"
        loss = "other"

    pearson_term = "on" if "pearsonon" in config else "off"
    if "_fullws_" in config_lower or "fullws" in config_lower:
        init_or_warm_start = "full warm-start"
    elif "from_baseonly" in config_lower:
        init_or_warm_start = "from base-only"
    elif "baseonly" in config_lower:
        init_or_warm_start = "base-only"
    elif "_base_" in config_lower:
        init_or_warm_start = "base warm-start"
    elif "scratch" in config_lower:
        init_or_warm_start = "scratch"
    elif "warmstart" in config_lower or "warm_start" in config_lower:
        init_or_warm_start = "warm-start"
    else:
        init_or_warm_start = "init=?"

    rank = f"r{rank_match.group(1)}" if rank_match else "rank=?"
    learning_rate = f"{lr_match.group(1)}e-{lr_match.group(2)}" if lr_match else "lr unknown"
    epochs = f"{epoch_match.group(1)}" if epoch_match else "unknown"
    description = (
        f"{family}; {loss}; Pearson term {pearson_term}; "
        f"{rank}; {init_or_warm_start}; lr {learning_rate}; {epochs} epochs"
    )
    return {
        "family": family,
        "loss": loss,
        "pearson_term": pearson_term,
        "parsed_rank": rank,
        "parsed_init_or_warm_start": init_or_warm_start,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "description": description,
    }


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def collect_source() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SUMMARY, sep="\t")
    detail = pd.read_csv(DETAIL, sep="\t")
    detail = detail[
        (detail["status"] == "ok")
        & (detail["bin_size_bp"] == PLOT_BIN_SIZE_BP)
        & (detail["region"] == BCL11A_REGION)
    ].copy()
    if detail.empty:
        raise ValueError(f"No {PLOT_BIN_SIZE_BP} bp ok rows found for {BCL11A_REGION} in {DETAIL}")

    summary = summary.rename(
        columns={
            "rank_mean_100": "candidate_order",
            "GATA1_100": "gata1_pearson_100bp",
            "HIC2_100": "hic2_pearson_100bp",
            "mean_100": "mean_pearson_100bp",
        }
    )
    summary = summary.sort_values("mean_pearson_100bp", ascending=False).reset_index(drop=True)
    summary["candidate_order"] = range(1, len(summary) + 1)

    parsed = pd.DataFrame([parse_config(str(config)) for config in summary["config"]])
    parsed["candidate_id"] = summary["candidate_id"].to_numpy()

    details = (
        detail.sort_values(["candidate_id", "target"])
        .groupby("candidate_id", as_index=False)
        .agg(
            region=("region", "first"),
            bin_size_bp=("bin_size_bp", "first"),
            n_bins=("n_bins", "first"),
            n_finite_pairs=("n_finite_pairs", "first"),
            example_source_label=("label", "first"),
            representative_run_dir=("run_dir", "first"),
            representative_export_dir=("export_dir", "first"),
        )
    )
    mapping = summary.merge(details, on="candidate_id", how="left").merge(parsed, on="candidate_id", how="left")

    mapping["original_label"] = mapping["config"]
    mapping["source_summary_path"] = str(SUMMARY)
    mapping["source_detail_path"] = str(DETAIL)
    mapping["plot_metric"] = PLOT_METRIC
    mapping["scale_rule"] = SCALE_RULE
    mapping["selected_final_recipe"] = bool_series(mapping["selected_final_recipe"])
    mapping["initial_grid_search_recipe"] = bool_series(mapping["initial_grid_search_recipe"])
    mapping["candidate_note"] = mapping["candidate_note"].fillna("")
    mapping.loc[mapping["initial_grid_search_recipe"], "candidate_note"] = INITIAL_GRID_SEARCH_LABEL
    mapping["rank"] = mapping["rank"].fillna(mapping["parsed_rank"])
    mapping["init_or_warm_start"] = mapping["init_or_warm_start"].fillna(mapping["parsed_init_or_warm_start"])
    mapping["candidate_label"] = mapping.apply(
        lambda row: f"{row['candidate_id']}  {row['rank']}  {row['init_or_warm_start']}",
        axis=1,
    )

    expected_pairs = len(mapping) * 2
    if len(detail) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} 100 bp target rows, found {len(detail)}")

    source = detail.merge(
        mapping[
            [
                "candidate_id",
                "candidate_order",
                "mean_pearson_100bp",
                "plot_metric",
                "source_summary_path",
                "source_detail_path",
            ]
        ],
        on="candidate_id",
        how="left",
    )
    source["pearson_100bp"] = source["pearson"]
    source["scale_rule"] = SCALE_RULE
    source["source_table"] = str(CLEAN_SOURCE_TSV)
    source = source.sort_values(["candidate_order", "target"]).drop(columns=["candidate_order"])

    source_details = (
        source[["candidate_id", "target", "prediction_bw", "ground_truth_bw"]]
        .sort_values(["candidate_id", "target"])
        .pivot(index="candidate_id", columns="target", values=["prediction_bw", "ground_truth_bw"])
    )
    source_details.columns = ["_".join(col).lower() for col in source_details.columns]
    mapping = mapping.merge(source_details.reset_index(), on="candidate_id", how="left")
    mapping = mapping.sort_values("candidate_order").reset_index(drop=True)
    mapping = mapping[
        [
            "candidate_id",
            "candidate_label",
            "rank",
            "init_or_warm_start",
            "original_label",
            "candidate_order",
            "gata1_pearson_100bp",
            "hic2_pearson_100bp",
            "mean_pearson_100bp",
            "n_targets",
            "all_rows_ok",
            "selected_final_recipe",
            "initial_grid_search_recipe",
            "candidate_note",
            "family",
            "loss",
            "pearson_term",
            "learning_rate",
            "epochs",
            "description",
            "region",
            "bin_size_bp",
            "n_bins",
            "n_finite_pairs",
            "plot_metric",
            "scale_rule",
            "source_summary_path",
            "source_detail_path",
            "example_source_label",
            "representative_run_dir",
            "representative_export_dir",
            "prediction_bw_gata1",
            "prediction_bw_hic2",
            "ground_truth_bw_gata1",
            "ground_truth_bw_hic2",
            "config",
        ]
    ]
    return source, mapping


def plot(mapping: pd.DataFrame) -> None:
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

    plot_df = mapping.sort_values("mean_pearson_100bp", ascending=True).reset_index(drop=True)
    y_pos = range(len(plot_df))
    colors = ["#e6862f" if selected else "#8fa3ad" for selected in plot_df["selected_final_recipe"]]
    edges = ["#2b2b2b" if selected else "white" for selected in plot_df["selected_final_recipe"]]

    fig, ax = plt.subplots(figsize=(8.4, 6.1))
    ax.barh(
        y_pos,
        plot_df["mean_pearson_100bp"],
        height=0.62,
        color=colors,
        edgecolor=edges,
        linewidth=1.2,
        label="Mean",
        zorder=2,
    )
    ax.scatter(
        plot_df["gata1_pearson_100bp"],
        [y - 0.18 for y in y_pos],
        s=28,
        marker="o",
        color="#2d6a9f",
        edgecolor="white",
        linewidth=0.5,
        label="GATA1",
        zorder=4,
    )
    ax.scatter(
        plot_df["hic2_pearson_100bp"],
        [y + 0.18 for y in y_pos],
        s=28,
        marker="D",
        color="#6b4c9a",
        edgecolor="white",
        linewidth=0.5,
        label="HIC2",
        zorder=4,
    )
    plot_df["plot_label"] = plot_df.apply(
        lambda row: f"{row['candidate_id']}\n{row['rank']}, {row['init_or_warm_start']}",
        axis=1,
    )
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(plot_df["plot_label"])
    for tick in ax.get_yticklabels():
        tick.set_linespacing(1.05)
    ax.set_xlabel("Pred-vs-ground-truth ChIP Pearson (100 bp non-overlapping bins)")
    ax.set_ylabel("")
    max_value = float(
        plot_df[
            ["mean_pearson_100bp", "gata1_pearson_100bp", "hic2_pearson_100bp"]
        ].max().max()
    )
    ax.set_xlim(0, max_value * 1.18)
    ax.set_title(
        "Figure 4. Initial hyperparameter sweep candidate selection\n"
        f"Predicted ChIP targets at BCL11A {BCL11A_REGION}",
        loc="left",
        fontsize=11,
        pad=10,
    )
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8.3)

    for y, value, selected in zip(y_pos, plot_df["mean_pearson_100bp"], plot_df["selected_final_recipe"]):
        label = f"{value:.3f}"
        if selected:
            label += "  initial pick"
        ax.text(value + 0.003, y, label, va="center", ha="left", fontsize=8.3)

    fig.text(
        0.01,
        0.012,
        f"Scale: {SCALE_RULE}. Candidate-selection panel, not the final 10-epoch result.",
        ha="left",
        va="bottom",
        fontsize=7.8,
    )
    fig.tight_layout(pad=1.1, rect=(0, 0.08, 1, 1))
    fig.savefig(CLEAN_PNG, dpi=450, bbox_inches="tight")
    fig.savefig(CLEAN_PDF, bbox_inches="tight")
    plt.close(fig)
    for target in [SIMPLE_PNG, LEGACY_PNG]:
        shutil.copy2(CLEAN_PNG, target)
    for target in [SIMPLE_PDF, LEGACY_PDF]:
        shutil.copy2(CLEAN_PDF, target)


def write_manifest() -> None:
    manifest = pd.DataFrame(
        [
            {
                "source_id": "corrected_100bp_summary",
                "path": str(SUMMARY),
                "source_type": "tsv",
                "used_for": "candidate ordering and corrected 100 bp mean Pearson bar values",
                "notes": f"Figure 4 uses mean_100 across GATA1 and HIC2 at {BCL11A_REGION}.",
            },
            {
                "source_id": "corrected_100bp_detail",
                "path": str(DETAIL),
                "source_type": "tsv",
                "used_for": "individual GATA1/HIC2 corrected 100 bp Pearson points and source paths",
                "notes": f"Pearson scale rule: {SCALE_RULE}.",
            },
        ]
    )
    manifest.to_csv(OUT_DIR / "source_manifest.tsv", sep="\t", index=False)


def write_docs(mapping: pd.DataFrame) -> None:
    top = mapping.sort_values("mean_pearson_100bp", ascending=False).iloc[0]
    selected_text = (
        f"{top.candidate_id} ({top.config}; GATA1 {top.gata1_pearson_100bp:.3f}, "
        f"HIC2 {top.hic2_pearson_100bp:.3f}, mean {top.mean_pearson_100bp:.3f})"
    )
    caption = f"""# Figure 4 caption

Clean report version: `figure4.png` / `.pdf`, duplicated from `figure4_finetuning_hyperparameter_selection_clean.png` / `.pdf`.

Figure 4 shows the initial hyperparameter sweep / candidate-selection result, not the final 10-epoch result. Fine-tuning candidates are compared at the BCL11A locus `{BCL11A_REGION}` using corrected prediction-vs-ground-truth ChIP Pearson values in 100 bp non-overlapping bins. GATA1 and HIC2 are predicted ChIP targets, not genomic loci. Bars show the mean corrected 100 bp Pearson across GATA1 and HIC2, and overlaid points show the individual target Pearson values. Each y label is `candidate ID + LoRA rank + initialization/warm-start status`; full run mappings and source paths are in `figure4_candidate_mapping.tsv` and `figure4_finetuning_hyperparameter_selection_clean_source.tsv`.

Pearson was computed after applying the scale rule `{SCALE_RULE}`. The top candidate in this corrected 100 bp summary is {selected_text}. Dropout-containing rows are excluded by design and handled separately in Figure 3.
"""
    (OUT_DIR / "caption.md").write_text(caption)

    readme = f"""# Figure 4 Fine-tuning Hyperparameter Selection

Use the clean report version:

- `figure4.png`
- `figure4.pdf`
- `figure4_finetuning_hyperparameter_selection_clean.png`
- `figure4_finetuning_hyperparameter_selection_clean.pdf`
- `figure4_finetuning_hyperparameter_selection_clean_source.tsv`
- `figure4_candidate_mapping.tsv`
- `source_manifest.tsv`
- `make_figure4_finetuning_hyperparameter_selection_clean.py`

Question: which initial fine-tuning candidate was selected from the hyperparameter sweep?

Answer: Figure 4 uses corrected 100 bp Pearson values at the BCL11A locus `{BCL11A_REGION}`. This is the initial hyperparameter sweep / candidate-selection panel, not the final 10-epoch result. Candidate labels keep the established `candidate ID + LoRA rank + initialization/warm-start status` format, for example `C1  r4  base warm-start`. The bar height is `mean_pearson_100bp = (gata1_pearson_100bp + hic2_pearson_100bp) / 2`, and the overlaid points show the individual GATA1 and HIC2 corrected 100 bp Pearson values. In this corrected 100 bp summary, the top candidate is {selected_text}.

GATA1 and HIC2 are predicted ChIP targets, not genomic loci. The genomic interval for every plotted Pearson value is the BCL11A locus `{BCL11A_REGION}`.

## How Pearson was computed

Figure 4 reads existing corrected metrics from:

- `{SUMMARY}`
- `{DETAIL}`

The script keeps `status == ok`, `region == {BCL11A_REGION}`, and `bin_size_bp == 100`. Each Pearson value is prediction bigWig versus matched ground-truth ChIP bigWig after applying the scale rule `{SCALE_RULE}`. The corrected 100 bp rows contain 10,000 bins and 10,000 finite prediction/ground-truth pairs per target/candidate.

Per-candidate prediction and ground-truth bigWig paths are retained in `figure4_finetuning_hyperparameter_selection_clean_source.tsv`; the compact plotted table is retained in `figure4_candidate_mapping.tsv`.
"""
    (OUT_DIR / "README.md").write_text(readme)


def main() -> None:
    source, mapping = collect_source()
    source.to_csv(CLEAN_SOURCE_TSV, sep="\t", index=False)
    source.to_csv(LEGACY_SOURCE_TSV, sep="\t", index=False)
    mapping.to_csv(MAPPING_TSV, sep="\t", index=False)
    write_manifest()
    write_docs(mapping)
    plot(mapping)


if __name__ == "__main__":
    main()
