# GATA1 final10 prediction and processed ground-truth BW package

Created: 2026-05-04 UTC

Purpose: browser-ready bigWig files for showing that the 10-epoch GATA1 fine-tuned Chromnitron prediction matches GATA1 ChIP-seq ground truth at BCL11A and ASXL1 loci.

## Contents

- `predictions_gata1_final10/`: GATA1 final 10-epoch prediction bigWigs copied unchanged from the Chromnitron output. These predictions are already on the model's log1p-scale output.
- `ground_truth_gata1_chip_clip0_log1p/`: locus-specific GATA1 ChIP ground-truth bigWigs generated from chrom2vec `genrich_normalized.bw` after `missing/nonfinite -> 0`, `negative -> 0`, then `log1p`.
- `regions/`: BED intervals for the two loci.
- `manifest.tsv`: labels, source paths, transform state, and SHA256 checksums.
- `ground_truth_transform_stats.tsv`: transform QC counts for the processed ground-truth files.

## Loci

- BCL11A: `chr2:60005424-61005424`
- ASXL1: `chr20:31898825-32898825`

## Source files

- BCL11A GATA1 prediction: `/broad/boxialab/sihengtao/projects/chromnitron_finetune/gata1_from_baseonly_scratch_r4_lr1e4_ep10_2024paper/20260502-154138/hic2_1mb_bigwig/prediction/prediction/check4hic2_anewPaper_single_sample/GATA1/bigwigs/GATA1_check4hic2_anewPaper_single_sample_prediction.bw`
- ASXL1 GATA1 prediction: `/broad/boxialab/sihengtao/projects/chromnitron_finetune/gata1_from_baseonly_scratch_r4_lr1e4_ep10_2024paper/20260502-154138/hic2_1mb_bigwig_chr20_asxl1/prediction/prediction/check4hic2_anewPaper_single_sample/GATA1/bigwigs/GATA1_check4hic2_anewPaper_single_sample_prediction.bw`
- Raw GATA1 ChIP ground truth: `/broad/boxialab/sihengtao/projects/check4hic2_anewPaper/chrom2vec_output/SRR21983756_SRR21983758/s9_bigwig/genrich_normalized.bw`

## Important scale note

Use the packaged `*_ground_truth_ChIP_clip0_log1p.bw` files for visual comparison with prediction. The original ground-truth source is chrom2vec-normalized and was not assumed to be pre-log1p.
