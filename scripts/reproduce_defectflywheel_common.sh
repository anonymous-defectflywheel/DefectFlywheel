#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <dataset_key> <dataset_label> <dataset_dirname> <ofa_dataset_name>" >&2
  exit 2
fi

DATASET_KEY="$1"
DATASET_LABEL="$2"
DATASET_DIRNAME="$3"
OFA_DATASET_NAME="$4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/datasets}"
PRETRAINED_DIR="${PRETRAINED_DIR:-$REPO_ROOT/pretrained}"
EXP_ROOT="${EXP_ROOT:-$REPO_ROOT/experiments}"
SHOT="${SHOT:-2}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-10}"
DRY_RUN="${DRY_RUN:-0}"

case "$GPU_ID" in
  0|1|2|3|4|5|6|7) ;;
  *) echo "ERROR: GPU_ID must be a non-negative single GPU id; got '$GPU_ID'" >&2; exit 2 ;;
esac

DATA_PATH="$DATA_ROOT/$DATASET_DIRNAME"
BLIP_MODEL_PATH="$PRETRAINED_DIR/blipdiffusion_model"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$EXP_ROOT/reproduce_defectflywheel_${DATASET_KEY}_${SHOT}shot_seed${SEED}_${EPOCHS}epoch_${TS}_gpu${GPU_ID}"
METHOD_OUTPUT_DIR="$RUN_DIR/method_outputs/DefectFlywheel"
EXP_NAME="$METHOD_OUTPUT_DIR/inner"

CMD=(python scripts/run_experiments.py
  --data_path "$DATA_PATH"
  --dataset "$OFA_DATASET_NAME"
  --exp_name "$EXP_NAME"
  --shot "$SHOT"
  --seed "$SEED"
  --macro_epochs "$EPOCHS"
  --epochs 1
  --context_backend blipdiffusion_qformer
  --blip_model_path "$BLIP_MODEL_PATH"
  --mine_method top_k
  --mine_top_k_ratio 0.10
  --heatmap_source baseline_chain_legacy
  --eval_policy final_only
  --baseline_eval_policy final_only
)

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY RUN] repo_root=$REPO_ROOT"
  echo "[DRY RUN] dataset=$DATASET_LABEL"
  echo "[DRY RUN] data_path=$DATA_PATH"
  echo "[DRY RUN] blip_model_path=$BLIP_MODEL_PATH"
  echo "[DRY RUN] run_dir=$RUN_DIR"
  echo "[DRY RUN] command: CUDA_VISIBLE_DEVICES=$GPU_ID ${CMD[*]}"
  exit 0
fi

if [[ ! -d "$DATA_PATH" ]]; then
  echo "ERROR: dataset directory not found: $DATA_PATH" >&2
  echo "Set DATA_ROOT=/path/to/datasets or create: $DATA_PATH" >&2
  exit 1
fi

if [[ ! -d "$BLIP_MODEL_PATH" ]]; then
  echo "ERROR: BLIP-Diffusion directory not found: $BLIP_MODEL_PATH" >&2
  echo "Download Salesforce/blipdiffusion to pretrained/blipdiffusion_model or set PRETRAINED_DIR=/path/to/pretrained" >&2
  exit 1
fi

if [[ ! -f "$BLIP_MODEL_PATH/model_index.json" ]]; then
  if ! find "$BLIP_MODEL_PATH" -name model_index.json -type f -print -quit | grep -q .; then
    echo "ERROR: no model_index.json found under: $BLIP_MODEL_PATH" >&2
    echo "The directory must be a BLIP-Diffusion model directory or HuggingFace snapshot tree." >&2
    exit 1
  fi
fi

mkdir -p "$RUN_DIR" "$RUN_DIR/logs" "$RUN_DIR/figures" "$RUN_DIR/heatmaps" "$RUN_DIR/anomaly_maps" "$RUN_DIR/checkpoints" "$RUN_DIR/method_outputs"
cp "$0" "$RUN_DIR/command.sh"
if [[ -f "$SCRIPT_DIR/reproduce_defectflywheel_common.sh" ]]; then
  cp "$SCRIPT_DIR/reproduce_defectflywheel_common.sh" "$RUN_DIR/reproduce_defectflywheel_common.sh"
fi

cat > "$RUN_DIR/run_config.json" <<JSON
{
  "method": "DefectFlywheel",
  "dataset": "$DATASET_LABEL",
  "dataset_key": "$DATASET_KEY",
  "data_path": "$DATA_PATH",
  "shot": $SHOT,
  "seed": $SEED,
  "train_epochs": $EPOCHS,
  "formal_epoch_budget": 10,
  "checkpoint_selection": "fixed_epoch_${EPOCHS}",
  "context_backend": "blipdiffusion_qformer",
  "blip_model_path": "$BLIP_MODEL_PATH",
  "mine_method": "top_k",
  "mine_top_k_ratio": 0.10,
  "heatmap_source": "baseline_chain_legacy",
  "eval_policy": "final_only",
  "baseline_eval_policy": "final_only",
  "gpu_id": "$GPU_ID",
  "source_repo": "$REPO_ROOT",
  "result_eligibility": "reproduction"
}
JSON

cat > "$RUN_DIR/RUN_INFO.txt" <<EOF_INFO
run_dir=$RUN_DIR
repo_root=$REPO_ROOT
dataset=$DATASET_LABEL
dataset_key=$DATASET_KEY
data_path=$DATA_PATH
shot=$SHOT
seed=$SEED
epochs=$EPOCHS
gpu_id=$GPU_ID
blip_model_path=$BLIP_MODEL_PATH
started_at=$(date -Is)
EOF_INFO

{
  echo "[DefectFlywheel reproduction] repo_root=$REPO_ROOT"
  echo "[DefectFlywheel reproduction] run_dir=$RUN_DIR"
  echo "[DefectFlywheel reproduction] dataset=$DATASET_LABEL"
  echo "[DefectFlywheel reproduction] command: CUDA_VISIBLE_DEVICES=$GPU_ID ${CMD[*]}"
} | tee "$RUN_DIR/bootstrap.log"

cd "$REPO_ROOT"
nvidia-smi > "$RUN_DIR/nvidia_smi_before.txt" 2>/dev/null || true
python -V > "$RUN_DIR/env_snapshot.txt" 2>&1 || true
python - <<'PYENV' >> "$RUN_DIR/env_snapshot.txt" 2>&1 || true
try:
    import torch
    print('torch', torch.__version__, 'cuda', getattr(torch.version, 'cuda', None), 'cuda_available', torch.cuda.is_available())
except Exception as exc:
    print('torch_import_error', repr(exc))
PYENV
python -m pip freeze > "$RUN_DIR/pip_freeze.txt" 2>/dev/null || true

set +e
CUDA_VISIBLE_DEVICES="$GPU_ID" "${CMD[@]}" 2>&1 | tee "$RUN_DIR/stdout_stderr.log"
STATUS=${PIPESTATUS[0]}
set -e

nvidia-smi > "$RUN_DIR/nvidia_smi_after.txt" 2>/dev/null || true
find "$RUN_DIR" -mindepth 2 -maxdepth 8 -name raw_metrics.json -print -quit | xargs -r -I{} cp {} "$RUN_DIR/raw_metrics.json"
find "$RUN_DIR" -mindepth 2 -maxdepth 8 -name metrics_summary.csv -print -quit | xargs -r -I{} cp {} "$RUN_DIR/metrics_summary.csv"
find "$RUN_DIR" -mindepth 2 -maxdepth 8 -name metrics_summary.jsonl -print -quit | xargs -r -I{} cp {} "$RUN_DIR/metrics_summary.jsonl"
find "$RUN_DIR" -mindepth 2 -maxdepth 8 -name checkpoint_manifest.json -print -quit | xargs -r -I{} cp {} "$RUN_DIR/checkpoint_manifest.json"

echo "finished_at=$(date -Is)" >> "$RUN_DIR/RUN_INFO.txt"
echo "status=$STATUS" >> "$RUN_DIR/RUN_INFO.txt"
echo "output_dir=$RUN_DIR" | tee -a "$RUN_DIR/bootstrap.log"
exit "$STATUS"
