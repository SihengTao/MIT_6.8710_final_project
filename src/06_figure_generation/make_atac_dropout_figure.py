#!/usr/bin/env python3
"""Generate the clean report version of Figure 3."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/broad/boxialab/sihengtao/projects/chromnitron_finetune")
WIDE = PROJECT_ROOT / "20260503_analysis/finetune_prediction_bcl11a_pearson_all/all_finetune_prediction_bcl11a_pearson_wide.tsv"

RESOLUTIONS = [50, 100, 200, 1000]
COMPARISONS = [
    ("early_no_atac_dropout", "Early no-dropout"),
    ("with_atac_dropout", "ATAC dropout"),
    ("best_no_dropout", "Best non-dropout"),
]


def _base_label(label: str) -> str:
    return label.split("|", maxsplit=1)[0]


def _pick_one(target_rows: pd.DataFrame, comparison: str) -> pd.Series:
    labels = target_rows["label"].astype(str)
    if comparison == "early_no_atac_dropout":
        mask = labels.str.contains("hic2_warmstart_no_atac_dropout/20260329", case=False, na=False)
    elif comparison == "with_atac_dropout":
        mask = labels.str.contains("with_atac_dropout", case=False, na=False)
    elif comparison == "best_no_dropout":
        mask = ~labels.str.contains("atac_dropout|dropout", case=False, na=False)
    else:
        raise ValueError(f"Unknown comparison: {comparison}")

    selected = target_rows[mask].sort_values("pearson_100bp", ascending=False).head(1)
    if selected.empty:
        raise ValueError(f"No row found for {comparison}")
    return selected.iloc[0]


def collect_source() -> pd.DataFrame:
    wide = pd.read_csv(WIDE, sep="\t")
    records: list[dict[str, object]] = []

    for target in ["GATA1", "HIC2"]:
        target_rows = wide[wide["target"] == target].copy()
        for order, (comparison, display_label) in enumerate(COMPARISONS):
            row = _pick_one(target_rows, comparison)
            record: dict[str, object] = {
                "target": target,
                "comparison": comparison,
                "display_label": display_label,
                "plot_label": f"{target} - {display_label}",
                "plot_order": order,
                "source_table": str(WIDE),
                "label": _base_label(str(row["label"])),
            }
            for resolution in RESOLUTIONS:
                record[f"pearson_{resolution}bp"] = row[f"pearson_{resolution}bp"]
            records.append(record)

    source = pd.DataFrame.from_records(records)
    return source.sort_values(["target", "plot_order"])


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
        "early_no_atac_dropout": "#9aa0a6",
        "with_atac_dropout": "#3b7cb8",
        "best_no_dropout": "#2e8b57",
    }
    plot_df = source.sort_values(["target", "plot_order"]).reset_index(drop=True)
    y_pos = list(range(len(plot_df)))[::-1]

    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.barh(
        y_pos,
        plot_df["pearson_100bp"],
        height=0.62,
        color=[colors[v] for v in plot_df["comparison"]],
        edgecolor="white",
        linewidth=0.9,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["plot_label"])
    ax.set_xlabel("Prediction vs GATA1/HIC2 ChIP Pearson at 100 bp")
    ax.set_xlim(0, 0.21)
    ax.set_title(
        "Figure 3. ATAC dropout: no consistent recipe-level benefit\n"
        "BCL11A transfer locus; higher Pearson is better",
        loc="left",
        fontsize=11,
        pad=10,
    )

    for y, value, comparison in zip(y_pos, plot_df["pearson_100bp"], plot_df["comparison"]):
        suffix = "  * selected" if comparison == "best_no_dropout" else ""
        ax.text(value + 0.004, y, f"{value:.3f}{suffix}", va="center", ha="left", fontsize=8.4)

    for boundary in [2.5]:
        ax.axhline(boundary, color="#d5d9de", linewidth=0.8)

    fig.tight_layout(pad=1.1)
    fig.savefig(OUT_DIR / "figure3_atac_dropout_ablation_clean.png", dpi=450, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure3_atac_dropout_ablation_clean.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    source = collect_source()
    source.to_csv(OUT_DIR / "figure3_atac_dropout_ablation_clean_source.tsv", sep="\t", index=False)
    plot(source)


if __name__ == "__main__":
    main()
