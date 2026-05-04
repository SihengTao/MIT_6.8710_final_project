# 5.2finetuneidea

Date: 2026-05-02

## Summary

- Add two new scratch LoRA finetune experiments from base model only, with no warm-start.
- GATA1: use 2024 paper GATA1 ATAC plus GATA1 ground truth; rank 4; learning rate 1e-4; 10 epochs.
- HIC2: use 2024 paper HIC2 ATAC plus HIC2 ground truth; rank 4; learning rate 1e-4; 10 epochs.
- During training, record parameters, loss, and Pearson at 50/100/200/1000 bp resolution.
- After both finish, run cross-transfer: treat each adapter as an ATAC-context adapter rather than a protein-specific adapter. Use the GATA1 finetune adapter with HIC2 CAP/protein to predict HIC2, and use the HIC2 finetune adapter with GATA1 CAP/protein to predict GATA1.
- Important pending confirmation: exact 2024 paper SRR mapping for GATA1/HIC2 ground truth and corresponding ATAC.
- Current immediate task before doing this new finetune: test whether prior HIC2 finetune adapters can already transfer to GATA1 at BCL11A.
