#!/usr/bin/env python3
"""Generate 10-epoch fine-tuning loss trajectory support figure."""

from __future__ import annotations

import csv
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


OUTPUT_DIR = Path(__file__).resolve().parent

INPUTS = [
    {
        "target": "GATA1",
        "panel_title": "GATA1 fine-tuning",
        "path": Path(
            "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
            "gata1_from_baseonly_scratch_r4_lr1e4_ep10_2024paper/"
            "20260502-154138/metrics.csv"
        ),
    },
    {
        "target": "HIC2",
        "panel_title": "HIC2 fine-tuning",
        "path": Path(
            "/broad/boxialab/sihengtao/projects/chromnitron_finetune/"
            "hic2_from_baseonly_scratch_r4_lr1e4_ep10_2024paper/"
            "20260502-154138/metrics.csv"
        ),
    },
]

FIGURE_TITLE = "Fine-tuning loss decreases over 10 epochs"
PNG_PATH = OUTPUT_DIR / "loss_trajectories_10epoch.png"
PDF_PATH = OUTPUT_DIR / "loss_trajectories_10epoch.pdf"
SOURCE_TSV_PATH = OUTPUT_DIR / "loss_trajectories_10epoch_source.tsv"
MANIFEST_PATH = OUTPUT_DIR / "source_manifest.tsv"
CAPTION_PATH = OUTPUT_DIR / "caption.md"
README_PATH = OUTPUT_DIR / "README.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metrics(path: Path, target: str) -> tuple[list[dict[str, object]], list[str]]:
    required = {"epoch", "train_loss", "val_loss"}
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

        rows: list[dict[str, object]] = []
        for row in reader:
            epoch = int(row["epoch"])
            train_loss = float(row["train_loss"])
            val_loss = float(row["val_loss"])
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise ValueError(f"Non-finite train/val loss at epoch {epoch} in {path}")
            rows.append(
                {
                    "target": target,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )

    rows.sort(key=lambda item: int(item["epoch"]))
    epochs = [int(item["epoch"]) for item in rows]
    if epochs != list(range(1, 11)):
        raise ValueError(f"Expected epochs 1-10 for {target}; observed {epochs}")
    return rows, reader.fieldnames


def write_source_tsv(all_rows: list[dict[str, object]]) -> None:
    with SOURCE_TSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["target", "epoch", "train_loss", "val_loss"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


def write_manifest(records: list[dict[str, object]]) -> None:
    generated_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with MANIFEST_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target",
                "source_metrics_csv",
                "sha256",
                "mtime_utc",
                "n_rows",
                "epochs",
                "columns_used",
                "generated_utc",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            source_path = Path(str(record["source_metrics_csv"]))
            stat = source_path.stat()
            writer.writerow(
                {
                    **record,
                    "sha256": sha256_file(source_path),
                    "mtime_utc": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(timespec="seconds"),
                    "generated_utc": generated_utc,
                }
            )


def y_limits(rows: list[dict[str, object]]) -> tuple[float, float]:
    values = [float(row["train_loss"]) for row in rows] + [
        float(row["val_loss"]) for row in rows
    ]
    lo = min(values)
    hi = max(values)
    span = hi - lo
    pad = max(span * 0.18, 0.00025)
    return lo - pad, hi + pad


def make_figure(grouped_rows: dict[str, list[dict[str, object]]]) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titlecolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = {"train_loss": "#2563eb", "val_loss": "#d97706"}
    labels = {"train_loss": "Train loss", "val_loss": "Validation loss"}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=300)
    fig.patch.set_facecolor("white")

    handles = []
    panel_labels = ["A", "B"]
    for index, config in enumerate(INPUTS):
        ax = axes[index]
        target = config["target"]
        rows = grouped_rows[target]
        epochs = [int(row["epoch"]) for row in rows]

        for key in ("train_loss", "val_loss"):
            line = ax.plot(
                epochs,
                [float(row[key]) for row in rows],
                color=colors[key],
                linewidth=2.0,
                marker="o",
                markersize=3.7,
                markeredgewidth=0,
                label=labels[key],
            )[0]
            if index == 0:
                handles.append(line)

        ax.set_title(str(config["panel_title"]), pad=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_xlim(0.75, 10.25)
        ax.set_ylim(*y_limits(rows))
        ax.set_xticks(range(1, 11))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.75)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.12,
            1.08,
            panel_labels[index],
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    fig.suptitle(FIGURE_TITLE, fontsize=12, fontweight="bold", y=0.98)
    fig.legend(
        handles,
        [labels["train_loss"], labels["val_loss"]],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.7,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.16, top=0.76, wspace=0.35)

    for path in (PNG_PATH, PDF_PATH):
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_caption_and_readme() -> None:
    caption = f"""# Caption

**Figure. {FIGURE_TITLE}.** Training and validation loss trajectories are shown for GATA1 and HIC2 fine-tuning runs over epochs 1-10. Each panel uses the corresponding `train_loss` and `val_loss` columns from the run-level `metrics.csv`; `test_loss` is intentionally excluded. The y-axis limits are set separately by panel to emphasize within-run loss trajectories.

Source metrics are listed in `source_manifest.tsv`; plotted values are in `loss_trajectories_10epoch_source.tsv`.
"""
    CAPTION_PATH.write_text(caption, encoding="utf-8")

    readme = f"""# 10-epoch fine-tuning loss trajectories

This directory contains a report-support figure for 10-epoch GATA1 and HIC2 fine-tuning loss trajectories.

## Outputs

- `loss_trajectories_10epoch.png`: raster figure for report placement.
- `loss_trajectories_10epoch.pdf`: vector figure for publication/report editing.
- `loss_trajectories_10epoch_source.tsv`: plotted epoch, target, train_loss, and val_loss values.
- `source_manifest.tsv`: source metrics paths, hashes, timestamps, and columns used.
- `caption.md`: draft report caption.
- `make_loss_trajectories_10epoch.py`: reproducible source script.

## Rebuild

Run from this directory or anywhere with the chromnitron Python/matplotlib environment available:

```bash
python make_loss_trajectories_10epoch.py
```

The script reads the two fixed metrics CSV files under `/broad/boxialab/sihengtao/projects/chromnitron_finetune/` and writes only to this directory.
"""
    README_PATH.write_text(readme, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grouped_rows: dict[str, list[dict[str, object]]] = {}
    all_rows: list[dict[str, object]] = []
    manifest_records: list[dict[str, object]] = []

    for config in INPUTS:
        target = str(config["target"])
        source_path = Path(config["path"])
        rows, fieldnames = read_metrics(source_path, target)
        grouped_rows[target] = rows
        all_rows.extend(rows)
        manifest_records.append(
            {
                "target": target,
                "source_metrics_csv": str(source_path),
                "n_rows": len(rows),
                "epochs": "1-10",
                "columns_used": "epoch,train_loss,val_loss",
            }
        )

    write_source_tsv(all_rows)
    write_manifest(manifest_records)
    make_figure(grouped_rows)
    write_caption_and_readme()

    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {SOURCE_TSV_PATH}")
    print(f"Wrote {MANIFEST_PATH}")
    print(f"Wrote {CAPTION_PATH}")
    print(f"Wrote {README_PATH}")


if __name__ == "__main__":
    main()
