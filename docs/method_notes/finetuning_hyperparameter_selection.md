# Figure 4 Fine-tuning Hyperparameter Selection

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

Answer: Figure 4 uses corrected 100 bp Pearson values at the BCL11A locus `chr2:60005424-61005424`. This is the initial hyperparameter sweep / candidate-selection panel, not the final 10-epoch result. Candidate labels keep the established `candidate ID + LoRA rank + initialization/warm-start status` format, for example `C1  r4  base warm-start`. The bar height is `mean_pearson_100bp = (gata1_pearson_100bp + hic2_pearson_100bp) / 2`, and the overlaid points show the individual GATA1 and HIC2 corrected 100 bp Pearson values. In this corrected 100 bp summary, the top candidate is C1 (hic2_loss_weighted_multiscale_pearsonon_r4_base_lr3em4_5ep; GATA1 0.435, HIC2 0.508, mean 0.471).

GATA1 and HIC2 are predicted ChIP targets, not genomic loci. The genomic interval for every plotted Pearson value is the BCL11A locus `chr2:60005424-61005424`.

## How Pearson was computed

Figure 4 reads existing corrected metrics from:

- `/broad/boxialab/sihengtao/projects/chromnitron_finetune/20260503_analysis/figure4_bcl11a_100bp_corrected/figure4_bcl11a_pearson_100bp_corrected_summary.tsv`
- `/broad/boxialab/sihengtao/projects/chromnitron_finetune/20260503_analysis/figure4_bcl11a_100bp_corrected/figure4_bcl11a_pearson_100bp_corrected.tsv`

The script keeps `status == ok`, `region == chr2:60005424-61005424`, and `bin_size_bp == 100`. Each Pearson value is prediction bigWig versus matched ground-truth ChIP bigWig after applying the scale rule `prediction as-is; ground-truth ChIP clip0 + log1p`. The corrected 100 bp rows contain 10,000 bins and 10,000 finite prediction/ground-truth pairs per target/candidate.

Per-candidate prediction and ground-truth bigWig paths are retained in `figure4_finetuning_hyperparameter_selection_clean_source.tsv`; the compact plotted table is retained in `figure4_candidate_mapping.tsv`.
