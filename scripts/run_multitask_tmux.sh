#!/usr/bin/env bash
set -euo pipefail

CONFIG_ARG="${1:?Usage: bash scripts/run_multitask_tmux.sh <config_stem_or_path> [fold]}"
FOLD="${2:-0}"
GPUS="${GPUS:-1,2,3,4}"
NPROC="${NPROC:-4}"
DATA_ROOT="${DATA_ROOT:-/workspace/landmark-assistant-model/Dataset}"
WANDB_PROJECT="${WANDB_PROJECT:-landmark-assistant-sprint2}"

if [[ "$CONFIG_ARG" == *.yaml ]]; then
  CONFIG="$CONFIG_ARG"
  CONFIG_STEM="$(basename "$CONFIG_ARG" .yaml)"
else
  CONFIG="configs/experiments/${CONFIG_ARG}.yaml"
  CONFIG_STEM="$CONFIG_ARG"
fi

SESSION="${SESSION:-landmark-${CONFIG_STEM}-fold${FOLD}}"

if [ ! -f "$CONFIG" ]; then
  echo "Missing multitask config: $CONFIG" >&2
  exit 1
fi

mkdir -p runs logs splits

tmux new-session -d -s "$SESSION" "cd '$PWD' && source .venv/bin/activate && export PYTHONPATH='$PWD/src:\${PYTHONPATH:-}' DATA_ROOT='$DATA_ROOT' WANDB_PROJECT='$WANDB_PROJECT' CUDA_VISIBLE_DEVICES='$GPUS' && python -m landmark_candidate.split_data --data-root '$DATA_ROOT' --out splits/kfold_seed20260513.json --seed 20260513 --folds 5 --test-ratio 0.15 && torchrun --nproc_per_node=$NPROC -m landmark_candidate.train_multitask --config '$CONFIG' --data-root '$DATA_ROOT' --split splits/kfold_seed20260513.json --fold '$FOLD' 2>&1 | tee logs/${SESSION}.log"

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
