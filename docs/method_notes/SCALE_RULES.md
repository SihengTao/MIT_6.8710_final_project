# Evaluation Scale Rules

These rules apply when preparing final Pearson or related metrics for chrom2vec-derived tracks and Chromnitron prediction tracks.

## chrom2vec-derived ATAC/ChIP ground truth

Use this transform before Pearson or other signal metrics:

1. Convert missing or non-finite values to `0`.
2. Clip negative values to `0`.
3. Apply `log1p` to the resulting values.

Recommended manifest value: `chrom2vec_clip0_missing0_log1p`.

## Chromnitron prediction tracks

Chromnitron prediction tracks are already on a `log1p` scale. Use the stored prediction values as-is for Pearson or related metrics.

Do not apply an additional `clip0+log1p` transform to Chromnitron prediction tracks.

Recommended manifest value: `chromnitron_prediction_as_is`.

## Existing Pearson values

Existing Pearson values computed directly on raw chrom2vec ATAC/ChIP tracks should be treated as provisional. Recompute them with the chrom2vec scale rule above before using them in final figures, tables, or claims.

Recommended manifest value for legacy values: `provisional_recompute_required`.
