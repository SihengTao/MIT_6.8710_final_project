#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

"${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  "${ROOT_DIR}/run_hic2_finetune.py" \
  "${ROOT_DIR}/configs/hic2_no_atac_dropout.yaml"

"${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  "${ROOT_DIR}/run_hic2_finetune.py" \
  "${ROOT_DIR}/configs/hic2_with_atac_dropout.yaml"
