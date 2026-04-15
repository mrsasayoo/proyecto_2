#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# run_expert.sh — Lanzador DDP para expertos del sistema MoE médico
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Uso:
#   bash run_expert.sh 1              # Expert 1 (ConvNeXt-Tiny / ChestXray14)
#   bash run_expert.sh 2              # Expert 2 (ConvNeXt-Small / ISIC 2019)
#   bash run_expert.sh 3              # Expert 3 (DenseNet3D / LUNA16)
#   bash run_expert.sh 4              # Expert 4 (ResNet3D)
#   bash run_expert.sh 5              # Expert 5 (Res-U-Net Autoencoder / OOD)
#   bash run_expert.sh oa             # Expert OA (EfficientNet-B3)
#
#   bash run_expert.sh 1 --dry-run    # Dry-run de Expert 1
#   bash run_expert.sh 1 --batch-per-gpu 24  # Expert 1 con batch custom
#
# Detección automática de GPUs:
#   El script detecta cuántas GPUs NVIDIA hay disponibles con nvidia-smi
#   y configura torchrun con --nproc_per_node acorde.
#   Si hay 1 GPU o no hay GPUs, DDP se desactiva transparentemente.
#
# Nota térmica (2× NVIDIA Titan Xp, 12 GB VRAM cada una):
#   Con la configuración original en 1 GPU:
#     - GPU 0: 100% carga, ~84°C, batch_size=32
#     - GPU 1: idle
#
#   Con DDP y batch_per_gpu = batch_total // num_gpus = 32 // 2 = 16:
#     - GPU 0: ~50% carga, ~65-70°C estimado
#     - GPU 1: ~50% carga, ~65-70°C estimado
#     - Batch efectivo total idéntico: 16 × 2 GPUs × 4 accum = 128
#
#   Con VRAM de 12 GB y ConvNeXt-Tiny FP16 en 224×224 RGB, batch=16
#   consume ~500 MiB. Se puede subir a batch_per_gpu=24-28 para
#   aprovechar los ~11 GB libres sin saturar temperatura.
#   Usar --batch-per-gpu 24 y monitorizar con nvidia-smi -l 2.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail

# ── Colores para output ────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ── Directorio del proyecto (donde vive este script) ───────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# ── Python del entorno virtual ─────────────────────────────────────────
PYTHON="python3"

# ── Validar argumentos ─────────────────────────────────────────────────
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: se requiere el ID del experto.${NC}"
    echo ""
    echo "Uso: bash run_expert.sh <expert_id> [opciones...]"
    echo ""
    echo "  expert_id:  1, 2, 3, 4, 5, oa"
    echo "  opciones:   --dry-run, --batch-per-gpu N, --data-root /ruta"
    echo ""
    echo "Ejemplos:"
    echo "  bash run_expert.sh 1                     # Expert 1 multi-GPU"
    echo "  bash run_expert.sh 1 --dry-run            # Verificación rápida"
    echo "  bash run_expert.sh 1 --batch-per-gpu 24   # Batch custom por GPU"
    exit 1
fi

EXPERT_ID="$1"
shift  # Remover expert_id de los argumentos, el resto pasa al script Python

# ── Mapeo de expert_id → script ────────────────────────────────────────
declare -A EXPERT_SCRIPTS
EXPERT_SCRIPTS=(
    ["1"]="src/pipeline/fase2/train_expert1_ddp.py"
    ["2"]="src/pipeline/fase2/train_expert2_ddp.py"
    ["3"]="src/pipeline/fase2/train_expert3_ddp.py"
    ["4"]="src/pipeline/fase2/train_expert4_ddp.py"
    ["5"]="src/pipeline/fase3/train_expert5_ddp.py"
    ["oa"]="src/pipeline/fase2/train_expert_oa_ddp.py"
)

SCRIPT_PATH="${EXPERT_SCRIPTS[$EXPERT_ID]:-}"

if [ -z "$SCRIPT_PATH" ]; then
    echo -e "${RED}Error: expert_id='${EXPERT_ID}' no reconocido.${NC}"
    echo "IDs válidos: 1, 2, 3, 4, 5, oa"
    exit 1
fi

FULL_SCRIPT_PATH="${PROJECT_ROOT}/${SCRIPT_PATH}"

if [ ! -f "$FULL_SCRIPT_PATH" ]; then
    echo -e "${RED}Error: script no encontrado: ${FULL_SCRIPT_PATH}${NC}"
    echo -e "${YELLOW}Solo Expert 1 DDP está implementado actualmente.${NC}"
    echo "Archivo esperado: ${SCRIPT_PATH}"
    exit 1
fi

# ── Detectar GPUs disponibles ──────────────────────────────────────────
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
else
    NUM_GPUS=0
fi

# Fallback: al menos 1 proceso
if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=1
    echo -e "${YELLOW}[WARN] nvidia-smi no detectó GPUs. Ejecutando con 1 proceso (CPU).${NC}"
fi

# ── Información de lanzamiento ─────────────────────────────────────────
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  MoE Médico — Lanzador DDP${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Expert:     ${GREEN}${EXPERT_ID}${NC}"
echo -e "  Script:     ${SCRIPT_PATH}"
echo -e "  Python:     ${PYTHON}"
echo -e "  GPUs:       ${GREEN}${NUM_GPUS}${NC}"
echo -e "  Argumentos: ${*:-<ninguno>}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ── Configurar NCCL para comunicación multi-GPU ────────────────────────
# NCCL_P2P_DISABLE=0: habilitar P2P si las GPUs lo soportan
# NCCL_IB_DISABLE=1: deshabilitar InfiniBand (no hay en este servidor)
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

# ── Lanzar con torchrun ────────────────────────────────────────────────
exec ${PYTHON} -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="127.0.0.1" \
    --master_port="29500" \
    "${FULL_SCRIPT_PATH}" \
    "$@"
