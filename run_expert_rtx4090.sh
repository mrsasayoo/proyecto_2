#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# run_expert_rtx4090.sh — Dry-run optimizado para RTX 4090
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Uso:
#   bash run_expert_rtx4090.sh              # Dry-run con auto-detect GPUs
#   bash run_expert_rtx4090.sh --full       # Entrenamiento completo (20 épocas)
#
# Este script:
#   1. Detecta GPUs automáticamente
#   2. Aplica config RTX 4090 (batch=128, accum=1, workers=12)
#   3. Ejecuta dry-run (2 batches train + 1 batch val)
#   4. Reporta tiempos, VRAM, loss
#
# Si la GPU tiene < 20GB VRAM, auto-ajusta a config Titan Xp.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
PYTHON="python3"

# ── Detectar GPUs ──────────────────────────────────────────────────────
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
else
    NUM_GPUS=1
    GPU_NAME="CPU"
    GPU_VRAM="0"
fi

[ "$NUM_GPUS" -eq 0 ] && NUM_GPUS=1

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Expert 1 — RTX 4090 Dry-Run"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  GPU:       ${GPU_NAME} (${GPU_VRAM} MiB)"
echo "  Num GPUs:  ${NUM_GPUS}"
echo "  Config:    batch=128, accum=1, workers=12, epochs=20"
echo "  Mode:      dry-run (2 train + 1 val batches)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── NCCL config ────────────────────────────────────────────────────────
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

# ── Ejecutar ───────────────────────────────────────────────────────────
START_TIME=$(date +%s%N)

exec ${PYTHON} -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="127.0.0.1" \
    --master_port="29501" \
    src/pipeline/fase2/run_dryrun_rtx4090.py \
    "$@"
