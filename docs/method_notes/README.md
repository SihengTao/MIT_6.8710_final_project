# Final Support Materials

This directory holds final-report support assets. All generated files here are local support artifacts; source bigWigs and large project outputs are referenced by absolute path under `/broad/boxialab/sihengtao/projects/` and are not copied here.

Current figure plan:

| Figure | Current support location | Status |
| --- | --- | --- |
| Figure 1 | `../milestone_support/` | Use the milestone figure; not regenerated here. |
| Figure 2 | `02_atac_shortcut_scale_diagnosis/` | Note only; use original milestone figure(s), no generated Figure 2. |
| Figure 3 | `03_atac_dropout_ablation/` | Generated ATAC dropout ablation. |
| Figure 4 | `04_finetuning_hyperparameter_selection/` | Generated non-dropout hyperparameter selection. |
| Figure 5 | `05_external_2022_adult_case_study/gata1_pred_vs_gt_1kb/` | Clean GATA1 1 kb prediction and ATAC baseline comparison. |

Deprecated placeholders:

- `03_preprocessing_finetune_preparation/` now only points to the new Figure 3 folder.
- `04_finetuning_strategy_comparison/` now only points to the new Figure 4 folder.
- The old Figure-ready/Figure 6-style support folder was removed because Figure 6 is not needed.

Plan and caption notes:

- Current plan: `FIGURE_PLAN_CURRENT.md`
- Legacy story file: `FIGURE_2_TO_5_STORY.md` now points to the current plan.
- Scale handling: `SCALE_RULES.md`

Important Figure 5 scale rule:

- Chromnitron prediction bigWigs are already on the model/log1p scale and are used as stored.
- chrom2vec-derived GATA1 ChIP and matched ATAC ground truth are transformed with missing=0, clip negative values to 0, then log1p.
- Pearson is computed after 1 kb non-overlapping bin means over each 1 Mb region.
- The Figure 5 source label is `check4hic2_anewPaper_single_sample`, not 2022paper.
