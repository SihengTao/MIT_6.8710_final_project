# Paper Module to Code Map

This repository is organized around the modules that appear in
`paper/final_report_icml.tex`. Server paths are preserved where they document
the actual Broad/local runs. Large tracks, checkpoints, zarr stores, and logs
are intentionally excluded.

## 1. Chromnitron Runtime Modules

Paper context: the base Chromnitron model, DNA/ATAC/protein inputs, and the
LoRA adaptation target.

Relevant files:

- `chromnitron/training/finetuning/model/v4_5/chromnitron_models.py`
- `chromnitron/training/finetuning/model/v4_5/chromnitron_blocks.py`
- `chromnitron/training/pretraining/data/dataset.py`
- `chromnitron/training/pretraining/data/transforms.py`
- `chromnitron/training/pretraining/utils.py`
- `chromnitron/training/infra/origami/base/`
- `chromnitron/training/infra/origami/track/`

These are the minimal Chromnitron modules imported by the project scripts.

## 2. External Erythroid Data and Manifests

Paper context: the 2024 erythroid sample, matched ATAC/GATA1/HIC2 tracks,
chr10 validation, chr20 test, and BCL11A/ASXL1 1 Mb regions.

Relevant files:

- `manifests/gata1_2024paper_single_sample.csv`
- `manifests/hic2_2024paper_single_sample.csv`
- `manifests/hic2_single_sample.csv`
- `data/final_gata1_case_study/prediction_ground_truth_package_manifests/`
- `configs/prediction_export/*/inputs/locus.bed`
- `configs/prediction_export/*/inputs/cap.txt`
- `configs/prediction_export/*/inputs/celltype.txt`

## 3. Zero-Shot HIC2 Diagnostic

Paper context: zero-shot held-out HIC2 behavior, ATAC-vs-prediction,
prediction-vs-ChIP, and signal-rich-bin diagnostics.

Relevant files:

- `src/01_zero_shot_diagnostic/eval_hic2_zero_shot.py`
- `data/zero_shot/new_heldout_midpoint_100bp_summary.tsv`
- `data/zero_shot/new_heldout_pairwise_summary.tsv`
- `data/zero_shot/old_heldout_pairwise_summary.tsv`
- `data/zero_shot/train_chr2_pairwise_summary_100pct.tsv`
- `data/zero_shot/training_reconstruction_summary.md`
- `data/zero_shot/zero_shot_readme.md`

## 4. Signal-Scale Correction

Paper context: prediction tracks are already log1p-scaled; chrom2vec-derived
ground-truth ATAC/ChIP tracks are clipped at zero and transformed with
`log1p` before Pearson evaluation.

Relevant files:

- `src/02_preprocessing_scale_correction/make_clip0_log1p_bigwig.py`
- `data/preprocessing_scale_correction/hic2_anewpaper_adult_transform_stats.tsv`
- `data/preprocessing_scale_correction/hic2_2022paper_transform_stats.tsv`
- `docs/method_notes/SCALE_RULES.md`
- `docs/method_notes/preprocessing_finetune_preparation.md`

## 5. ATAC Dropout Ablation

Paper context: span-based ATAC dropout with 32-position spans and target mask
fraction 0.20, compared against a no-dropout control.

Relevant files:

- `configs/atac_dropout/hic2_no_atac_dropout.yaml`
- `configs/atac_dropout/hic2_with_atac_dropout.yaml`
- `qsub/atac_dropout/sub_hic2_finetune_no_atac_dropout.qsub`
- `qsub/atac_dropout/sub_hic2_finetune_with_atac_dropout.qsub`
- `data/atac_dropout/figure3_atac_dropout_ablation_source.tsv`
- `data/atac_dropout/figure3_atac_dropout_ablation_clean_source.tsv`
- `data/atac_dropout/source_manifest.tsv`
- `src/06_figure_generation/make_atac_dropout_figure.py`
- `docs/method_notes/atac_shortcut_scale_diagnosis.md`

## 6. LoRA Fine-Tuning Core

Paper context: rank-4/rank-8 LoRA fine-tuning on top of `chromnitron_base`,
AdamW optimization, multiscale weighted MSE and Pearson losses, train/val/test
chromosome splits, and target-specific adaptation.

Relevant files:

- `src/03_finetuning/run_hic2_finetune.py`
- `src/03_finetuning/finetune_dataset.py`
- `src/03_finetuning/finetune_losses.py`
- `requirements.txt`

## 7. Hyperparameter Sweep and C1 Selection

Paper context: 12-candidate C1-C12 sweep over loss family, Pearson term, LoRA
rank, warm start, learning rate, and epoch count, evaluated at BCL11A for GATA1
and HIC2.

Relevant files:

- `src/07_config_generation/make_hic2_trial_config.py`
- `configs/hyperparameter_sweep/generated_hic2_trials/`
- `configs/hyperparameter_sweep/generated_hic2_loss_trials/`
- `configs/hyperparameter_sweep/generated_hic2_loss_compare_3task/`
- `configs/hyperparameter_sweep/generated_hic2_target_transform_abd_8ep/`
- `qsub/hyperparameter_sweeps/`
- `data/hyperparameter_sweep/figure4_candidate_mapping.tsv`
- `data/hyperparameter_sweep/figure4_finetuning_hyperparameter_selection_source.tsv`
- `data/hyperparameter_sweep/figure4_finetuning_hyperparameter_selection_clean_source.tsv`
- `data/hyperparameter_sweep/figure4_source_manifest.tsv`
- `src/06_figure_generation/make_hyperparameter_sweep_figure.py`
- `docs/method_notes/finetuning_hyperparameter_selection.md`

## 8. Final 10-Epoch Target-Specific Fine-Tuning

Paper context: final GATA1 and HIC2 10-epoch fine-tunes using the selected C1
structural recipe.

Relevant files:

- `configs/final_10epoch/gata1_scratch_r4_lr1e4_10ep_2024paper.yaml`
- `configs/final_10epoch/hic2_scratch_r4_lr1e4_10ep_2024paper.yaml`
- `qsub/final_10epoch_and_exports/sub_5.2finetuneidea_scratch_lora_2024paper.qsub`
- `qsub/final_10epoch_and_exports/sub_5.2finetuneidea_bcl11a_pearson_2024paper.qsub`
- `qsub/final_10epoch_and_exports/sub_5.2_direct_scratch_export_bcl11a_1mb.qsub`
- `qsub/final_10epoch_and_exports/sub_5.2_direct_scratch_export_chr20_asxl1_1mb.qsub`
- `data/hyperparameter_sweep/loss_trajectories_10epoch_source.tsv`
- `data/hyperparameter_sweep/loss_trajectories_source_manifest.tsv`
- `src/06_figure_generation/make_loss_trajectories_10epoch.py`
- `docs/method_notes/final_10epoch_strategy.md`

## 9. Prediction Export

Paper context: exporting base, official-adapter, and fine-tuned-adapter
predictions to 1 Mb BCL11A/ASXL1 windows for downstream Pearson and track-view
analysis.

Relevant files:

- `src/04_prediction_export/export_hic2_best_adapter_bigwig_1mb.py`
- `configs/prediction_export/hic2_bcl11a_export/`
- `configs/prediction_export/hic2_bcl11a_export_2022paper/`
- `configs/prediction_export/hic2_base_only_bcl11a_1mb/`
- `configs/prediction_export/hic2_base_only_chr20_asxl1_1mb/`
- `configs/prediction_export/gata1_base_only_bcl11a_1mb/`
- `configs/prediction_export/gata1_base_only_chr20_asxl1_1mb/`
- `configs/prediction_export/gata1_official_bcl11a_1mb/`
- `qsub/final_10epoch_and_exports/sub_hic2_export_bcl11a_1mb_selected.qsub`
- `qsub/final_10epoch_and_exports/sub_hic2_export_recent3_8ep_chr20_asxl1_1mb.qsub`
- `qsub/final_10epoch_and_exports/sub_gata1_base_only_bcl11a_1mb.qsub`
- `qsub/final_10epoch_and_exports/sub_gata1_official_bcl11a_1mb.qsub`
- `qsub/final_10epoch_and_exports/sub_hic2_base_only_bcl11a_1mb.qsub`

## 10. Evaluation and Track Metrics

Paper context: 100 bp Pearson at BCL11A/ASXL1, ATAC-vs-ChIP baselines,
prediction-vs-ChIP comparisons, HIC2-to-GATA1 transfer checks, and sparse-track
case-study interpretation.

Relevant files:

- `src/05_evaluation/evaluate_bcl11a_pearson_tracks.py`
- `src/05_evaluation/evaluate_hic2_to_gata1_transfer_bcl11a.py`
- `src/05_evaluation/compute_gata1_pred_vs_gt.py`
- `data/final_gata1_case_study/figure5_metrics.tsv`
- `data/final_gata1_case_study/figure5_tracks.tsv`
- `data/final_gata1_case_study/figure5_source_manifest.tsv`
- `data/final_gata1_case_study/gata1_pred_vs_gt_100bp_metrics.tsv`
- `data/final_gata1_case_study/gata1_pred_vs_gt_1kb_metrics.tsv`
- `data/final_gata1_case_study/gata1_pred_vs_gt_1kb_tracks.tsv`
- `data/final_gata1_case_study/gata1_resolution_sweep_metrics.tsv`
- `docs/method_notes/external_case_study.md`

## 11. Report Figure Generation

Paper context: regenerating the figure source panels used in the final report.
Generated images are excluded from git; the source TSVs and plotting scripts
are included.

Relevant files:

- `src/06_figure_generation/regen_report_figures.py`
- `src/06_figure_generation/make_atac_dropout_figure.py`
- `src/06_figure_generation/make_hyperparameter_sweep_figure.py`
- `src/06_figure_generation/make_loss_trajectories_10epoch.py`
- `src/06_figure_generation/make_final_gata1_case_study_figure.py`
- `data/atac_dropout/`
- `data/hyperparameter_sweep/`
- `data/final_gata1_case_study/`

## 12. Paper Source

Paper context: final report source, bibliography, and local ICML style files.

Relevant files:

- `paper/final_report_icml.tex`
- `paper/references.bib`
- `paper/icml2022.bst`
- `paper/icml2022.sty`
- `paper/fancyhdr.sty`
