# 05 External 2022 Adult HIC2 Case Study

Purpose: hold Figure 5 support for the external 2022 adult HIC2 prediction-vs-ground-truth ChIP case study.

Canonical Figure 5 uses 100bp non-overlapping bins, not 1 kb. The metric is Pearson between Chromnitron prediction and ground-truth HIC2 ChIP over the BCL11A and ASXL1 1 Mb regions. Prediction values are used as-is; ground-truth ChIP is missing/NaN/inf-to-zero, clip0, then log1p before binning.

Canonical top-level copies:

- `figure5.png`
- `figure5.pdf`
- `figure5_metrics.tsv`
- `figure5_tracks.tsv`
- `figure5_source_manifest.tsv`
- `figure5_caption.md`

Canonical source folder:

- `figure5_pred_vs_gt_100bp/`

Canonical 100 bp Pearson values:

| Locus | Region | Pearson vs HIC2 ChIP |
| --- | --- | ---: |
| BCL11A | `chr2:60005424-61005424` | 0.019897 |
| ASXL1 | `chr20:31898825-32898825` | 0.010928 |

Legacy/provenance folders retained, but not final canonical Figure 5:

- `figure5_pred_vs_gt_1kb/`
- `gata1_pred_vs_gt_1kb/`
