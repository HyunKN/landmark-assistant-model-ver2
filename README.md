# landmark-assistant-model-ver2

Unified Landmark Assistant model training repo.

This is the preferred GitHub repository layout. It keeps one shared training/evaluation codebase and separates model candidates by config.

## Project Docs

Shared project documentation is managed in the separate docs workspace:

```text
C:\Users\hi\Downloads\종설_작업중\landmark-assistant\docs
```

Published docs:

```text
https://landmark-assistant-sprint1.vercel.app/
```

Keep this repository focused on training code, configs, scripts, and run outputs. Do not add the Vercel docs site or demo app copy here.

## Candidates

- `mobileclip2_s4` -> `configs/candidates/mobileclip2_s4.yaml`
- `mobileclip2_s3` -> `configs/candidates/mobileclip2_s3.yaml`
- `mobileclip2_b` -> `configs/candidates/mobileclip2_b.yaml`
- `mobilenetv4_hybrid_large` -> `configs/candidates/mobilenetv4_hybrid_large.yaml`
- `mobilenetv4_conv_aa_large_in12k` -> `configs/candidates/mobilenetv4_conv_aa_large_in12k.yaml`

## Why One Repo

One repo is better than five repos for this project because:

- the dataset split code stays identical
- the training loop stays identical
- W&B run naming and metrics stay comparable
- bug fixes apply to every candidate at once
- GitHub history is much cleaner

## Data Strategy

The dataset is still growing. This repo scans the server dataset at execution time:

```text
/workspace/landmark-assistant-model/Dataset/
  <landmark_id>/
    labels.json
    images/
```

As long as the folder and JSON format stay the same, add more landmark folders or records, then regenerate splits:

```bash
export DATA_ROOT=/workspace/landmark-assistant-model/Dataset
bash scripts/make_splits.sh
```

The split manifest records `dataset_fingerprint`, total records, class counts, and confirmed/holdout counts. Do not compare W&B runs across different fingerprints as if they used the same dataset.

## Server Quickstart

```bash
git clone https://github.com/HyunKN/landmark-assistant-model-ver2.git
cd landmark-assistant-model-ver2

bash scripts/setup_venv.sh
source .venv/bin/activate

export DATA_ROOT=/workspace/landmark-assistant-model/Dataset
export WANDB_PROJECT=landmark-assistant-sprint1
export GDRIVE_BACKUP_ROOT=gdrive:landmark-assistant/runs

bash scripts/make_splits.sh
GPUS=1,2,3,4 NPROC=4 EXPORT_ONNX=0 bash scripts/run_candidate_tmux.sh mobileclip2_s4 0
```

## Sprint 2 Multi-Task Training

The Sprint 2 training path uses MobileCLIP2-S4 with classification, image-text contrastive learning, and `confusing_with` hard-negative loss.

```bash
export DATA_ROOT=/workspace/landmark-assistant-model/Dataset
GPUS=1,2,3,4 NPROC=4 bash scripts/run_multitask_tmux.sh mobileclip2_s4_partial_unfreeze_ce_hardneg 0
```

Implementation notes and output files are documented in the central docs:

```text
https://landmark-assistant-sprint1.vercel.app/operations/multitask-model-training-implementation-2026-05-21.html
```

After a multi-task run finishes, mine review candidates for the next `confusing_with` update:

```bash
python scripts/mine_hard_negative_candidates.py --run-dir runs/<run_name> --low-margin-threshold 0.05
```

Run all candidates/folds sequentially:

```bash
bash scripts/run_all_candidates_tmux.sh
```

## Ranking

Primary ranking after the checkpoint-selection fix:

1. mean validation Top-1 accuracy across folds
2. mean validation Top-3 accuracy across folds
3. locked test Top-1
4. locked test Top-3
5. out-of-scope AUROC when a true negative set exists
6. ONNX export success
7. latency and file size as secondary checks
