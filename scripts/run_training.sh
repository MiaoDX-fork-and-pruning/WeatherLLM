#!/bin/bash
# ==============================================================================
# PhyDiff-Net Training Launch Script
# ==============================================================================
# Usage:
#   bash scripts/run_training.sh                          # Default: single GPU, pretrain stage
#   bash scripts/run_training.sh --gpu 1 --seed 123       # Custom GPU and seed
#   bash scripts/run_training.sh --resume checkpoints/xxx # Resume from checkpoint
#   bash scripts/run_training.sh --distributed             # Multi-GPU DDP
#
# Environment Variables (for distributed training):
#   MASTER_ADDR  - Master node address (default: localhost)
#   MASTER_PORT  - Master node port (default: 29500)
#   WORLD_SIZE   - Number of GPUs
#   RANK         - Current process rank
# ==============================================================================

set -euo pipefail

# ---- Project root ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "============================================================"
echo "  PhyDiff-Net Training"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Date: $(date)"
echo "============================================================"

# ---- Default configuration ----
MODEL_CONFIG="src/configs/model_config.yaml"
DATA_CONFIG="src/configs/data_config.yaml"
TRAINING_CONFIG="src/configs/training_config.yaml"
GPU=0
SEED=42
RESUME=""
START_STAGE="pretrain"
DISTRIBUTED=false

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        --resume)
            RESUME="$2"; shift 2 ;;
        --start_stage)
            START_STAGE="$2"; shift 2 ;;
        --distributed)
            DISTRIBUTED=true; shift ;;
        --model_config)
            MODEL_CONFIG="$2"; shift 2 ;;
        --data_config)
            DATA_CONFIG="$2"; shift 2 ;;
        --training_config)
            TRAINING_CONFIG="$2"; shift 2 ;;
        --master_addr)
            export MASTER_ADDR="$2"; shift 2 ;;
        --master_port)
            export MASTER_PORT="$2"; shift 2 ;;
        --world_size)
            export WORLD_SIZE="$2"; shift 2 ;;
        --rank)
            export RANK="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu GPU_ID          GPU device id (default: 0)"
            echo "  --seed SEED           Random seed (default: 42)"
            echo "  --resume PATH         Resume from checkpoint"
            echo "  --start_stage STAGE   Training stage: pretrain|fusion_pretrain|finetune|extreme_enhance"
            echo "  --distributed         Enable distributed training (DDP)"
            echo "  --model_config PATH   Model config file"
            echo "  --data_config PATH    Data config file"
            echo "  --training_config PATH Training config file"
            echo "  --master_addr ADDR    DDP master address (default: localhost)"
            echo "  --master_port PORT    DDP master port (default: 29500)"
            echo "  --world_size N        Number of GPUs for DDP"
            echo "  --rank RANK           Process rank for DDP"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ---- Environment info ----
echo ""
echo "Configuration:"
echo "  Model config:    ${MODEL_CONFIG}"
echo "  Data config:     ${DATA_CONFIG}"
echo "  Training config: ${TRAINING_CONFIG}"
echo "  GPU:             ${GPU}"
echo "  Seed:            ${SEED}"
echo "  Start stage:     ${START_STAGE}"
echo "  Distributed:     ${DISTRIBUTED}"
if [[ -n "${RESUME}" ]]; then
    echo "  Resume from:     ${RESUME}"
fi
echo ""

# ---- GPU check ----
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{print "  Device: "$1", Memory: "$2"MB (Free: "$3"MB)"}'
    echo ""
else
    echo "WARNING: nvidia-smi not found. GPU availability unknown."
    echo ""
fi

# ---- Create directories ----
mkdir -p checkpoints
mkdir -p logs/tensorboard

# ---- Launch training ----
if [[ "${DISTRIBUTED}" == "true" ]]; then
    echo "Launching distributed training..."
    export MASTER_ADDR="${MASTER_ADDR:-localhost}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
    export WORLD_SIZE="${WORLD_SIZE:-1}"
    export RANK="${RANK:-0}"

    torchrun \
        --nproc_per_node="${WORLD_SIZE}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        src/training/train.py \
        --config "${TRAINING_CONFIG}" \
        --data_config "${DATA_CONFIG}" \
        --model_config "${MODEL_CONFIG}" \
        --gpu "${GPU}" \
        --seed "${SEED}" \
        --start_stage "${START_STAGE}" \
        --distributed
else
    echo "Launching single-GPU training..."
    python src/training/train.py \
        --config "${TRAINING_CONFIG}" \
        --data_config "${DATA_CONFIG}" \
        --model_config "${MODEL_CONFIG}" \
        --gpu "${GPU}" \
        --seed "${SEED}" \
        --start_stage "${START_STAGE}"
fi

echo ""
echo "============================================================"
echo "  Training completed at $(date)"
echo "============================================================"
