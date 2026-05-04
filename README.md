# MIT 6.8710 Final Project: Chromnitron Fine-Tuning

This repository contains the code used for the final report, **Improving Chromnitron Predictions in Unseen Erythroid Cell Types with Target-Specific Fine-Tuning**.

The repository is organized by the modules that appear in the paper. Broad/server paths are intentionally preserved in configs, qsub files, manifests, and source tables so that the submitted code maps back to the actual runs. Large data and generated artifacts are excluded: bigWig tracks, zarr stores, checkpoints, BAM/FASTQ files, figures, PDFs, DOCX files, and cluster logs are not committed.

## Repository Layout

```text
src/
  01_zero_shot_diagnostic/          # zero-shot HIC2 diagnostic evaluation
  02_preprocessing_scale_correction/# clip0+log1p preprocessing utilities
  03_finetuning/                    # LoRA fine-tuning entrypoint, dataset, losses
  04_prediction_export/             # export trained adapters to 1 Mb prediction bigWigs
  05_evaluation/                    # Pearson/track evaluation scripts
  06_figure_generation/             # scripts that regenerate report figures from source TSVs
  07_config_generation/             # generated sweep-config helper

configs/
  atac_dropout/                     # no-dropout vs ATAC-dropout configs
  hyperparameter_sweep/             # C1-C12 and related sweep configs
  final_10epoch/                    # final GATA1/HIC2 10-epoch fine-tune configs
  prediction_export/                # 1 Mb prediction/export configs

qsub/                               # Broad cluster launch scripts
manifests/                          # sample manifests used by fine-tuning
data/                               # small source TSV/CSV tables used by evaluation/figures
docs/method_notes/                  # project notes that document run choices
paper/                              # final LaTeX source and bibliography
chromnitron/                        # minimal Chromnitron modules imported by this project
```

## Main Entry Points

- Fine-tuning:
  - `src/03_finetuning/run_hic2_finetune.py`
  - `src/03_finetuning/finetune_dataset.py`
  - `src/03_finetuning/finetune_losses.py`

- Zero-shot diagnostic:
  - `src/01_zero_shot_diagnostic/eval_hic2_zero_shot.py`

- Scale correction:
  - `src/02_preprocessing_scale_correction/make_clip0_log1p_bigwig.py`

- Prediction export:
  - `src/04_prediction_export/export_hic2_best_adapter_bigwig_1mb.py`

- Evaluation:
  - `src/05_evaluation/evaluate_bcl11a_pearson_tracks.py`
  - `src/05_evaluation/evaluate_hic2_to_gata1_transfer_bcl11a.py`
  - `src/05_evaluation/compute_gata1_pred_vs_gt.py`

- Figure generation:
  - `src/06_figure_generation/regen_report_figures.py`
  - `src/06_figure_generation/make_atac_dropout_figure.py`
  - `src/06_figure_generation/make_hyperparameter_sweep_figure.py`
  - `src/06_figure_generation/make_loss_trajectories_10epoch.py`
  - `src/06_figure_generation/make_final_gata1_case_study_figure.py`

## Reproducibility Notes

The code was written for the original Broad/local environment. The configs preserve absolute paths to:

- Chromnitron base checkpoint and LoRA adapters
- chrom2vec-derived ATAC/ChIP zarr and bigWig tracks
- CAP protein embeddings
- Broad project output directories

To rerun from a different environment, update those paths in `configs/`, `manifests/`, and `qsub/`. The scripts under `src/` were lightly adjusted so that imports work from this GitHub layout, but the scientific paths and run labels were not rewritten.

## Paper-to-Code Map

See `MODULE_MAP.md` for the mapping from report sections to code/config/data files.
