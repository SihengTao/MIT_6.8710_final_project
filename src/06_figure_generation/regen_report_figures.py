#!/usr/bin/env python3
"""Regenerate Figure 2 (ATAC dropout ablation) and Figure 3 (hyperparameter
sweep) for the final report using the locally cached source TSVs. The titles
embedded in the milestone PNGs include literal "Figure 3." / "Figure 4." which
collides with the final paper's figure numbering. This script strips the
hard-coded figure-number prefix and uses a descriptive title only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DROP_SRC = ROOT / "final_support/03_atac_dropout_ablation/figure3_atac_dropout_ablation_clean_source.tsv"
SWEEP_SRC = ROOT / "final_support/04_finetuning_hyperparameter_selection/figure4_candidate_mapping.tsv"
OUT_DIR = ROOT / "final"


def make_dropout_figure() -> Path:
    src = pd.read_csv(DROP_SRC, sep="\t")
    src = src.sort_values(["target", "plot_order"]).reset_index(drop=True)

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
    y_pos = list(range(len(src)))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.barh(
        y_pos,
        src["pearson_100bp"],
        height=0.62,
        color=[colors[v] for v in src["comparison"]],
        edgecolor="white",
        linewidth=0.9,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(src["plot_label"])
    ax.set_xlabel("Prediction vs ChIP Pearson at 100 bp (BCL11A locus)")
    ax.set_xlim(0, 0.21)
    ax.set_title(
        "ATAC dropout does not exceed best non-dropout candidate\n"
        "(higher Pearson is better)",
        loc="left",
        fontsize=10.5,
        pad=10,
    )
    for y, value, comparison in zip(y_pos, src["pearson_100bp"], src["comparison"]):
        suffix = "  selected" if comparison == "best_no_dropout" else ""
        ax.text(value + 0.004, y, f"{value:.3f}{suffix}", va="center", ha="left", fontsize=8.4)
    ax.axhline(2.5, color="#d5d9de", linewidth=0.8)
    fig.tight_layout(pad=1.0)
    out = OUT_DIR / "figure_dropout.png"
    fig.savefig(out, dpi=450, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def make_sweep_figure() -> Path:
    df = pd.read_csv(SWEEP_SRC, sep="\t")
    df = df.sort_values("candidate_order").reset_index(drop=True)

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

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    y_pos = list(range(len(df)))[::-1]

    bar_color = "#7d8aa3"
    pick_color = "#e08a3c"
    colors = [pick_color if cid == "C1" else bar_color for cid in df["candidate_id"]]
    ax.barh(y_pos, df["mean_pearson_100bp"], height=0.62, color=colors, edgecolor="white", linewidth=0.9)

    ax.scatter(df["gata1_pearson_100bp"], y_pos, color="#1f77b4", s=22, label="GATA1", zorder=3)
    ax.scatter(df["hic2_pearson_100bp"], y_pos, color="#9b59b6", marker="D", s=22, label="HIC2", zorder=3)

    labels = [f"{cid}  r{r}  {ws}" for cid, r, ws in zip(df["candidate_id"], df["rank"].astype(str), df["init_or_warm_start"])]
    labels = [lbl.replace("rrank", "r") for lbl in labels]
    labels = [
        f"{cid}  r{rank.lstrip('r')}  {ws}"
        for cid, rank, ws in zip(df["candidate_id"], df["rank"].astype(str), df["init_or_warm_start"])
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)

    for y, value in zip(y_pos, df["mean_pearson_100bp"]):
        ax.text(value + 0.005, y, f"{value:.3f}", va="center", ha="left", fontsize=8.4)
    top_idx = df.index[df["candidate_id"] == "C1"][0]
    ax.text(
        df.loc[top_idx, "mean_pearson_100bp"] + 0.06,
        y_pos[top_idx],
        "selected",
        va="center",
        ha="left",
        fontsize=8.4,
        color=pick_color,
    )

    ax.set_xlabel("Pred-vs-ChIP Pearson at 100 bp (mean of GATA1 and HIC2)")
    ax.set_xlim(0, 0.62)
    ax.set_title(
        "Initial hyperparameter sweep at BCL11A locus\n"
        "bars: mean Pearson;  points: per-target Pearson",
        loc="left",
        fontsize=10.5,
        pad=10,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout(pad=1.0)
    out = OUT_DIR / "figure_sweep.png"
    fig.savefig(out, dpi=450, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p1 = make_dropout_figure()
    p2 = make_sweep_figure()
    print("wrote:", p1, p2)
