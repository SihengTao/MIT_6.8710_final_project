# 01 Generalization Gap Zero Shot

Purpose: organize evidence for the generalization gap.

Supported claim: the base, official, or direct-transfer Chromnitron-like model does not robustly zero-shot transfer to unseen or new cell type/dataset settings.

Intended contents:

- Final metric panels comparing direct transfer against target-domain or related-domain alternatives.
- Held-out cell type or dataset summaries.
- Representative track visualizations showing where zero-shot prediction succeeds or fails.
- Source path manifest rows for metrics tables and bigWigs.

Known source paths from context:

- Data and result paths should be referenced from `/broad/boxialab/sihengtao/projects/` after exact outputs are selected.

Scale rule: any Pearson based on chrom2vec-derived ATAC/ChIP tracks must use `chrom2vec_clip0_missing0_log1p`. Raw-track Pearson values are provisional until recomputed.
