# MobileCLIP2-S4 Multi-Task Training Implementation

Date: 2026-05-21

This document records the Sprint 2 model-training implementation added to `landmark-assistant-model-ver2`.

## Why This Was Added

Sprint 1 used an image classification/retrieval path that could recognize the 13 supported landmarks, but it did not fully use the improved caption labels:

- Korean and English captions were not directly used during training.
- `confusing_with` hard-negative labels were recorded, but not used by the loss.
- Similar landmarks such as palace gates and palace halls needed a stronger separation signal.
- Text search needed the image encoder and text encoder to remain aligned after fine-tuning.

The new training path keeps MobileCLIP2-S4 as the base model, then fine-tunes it with three training signals at once.

## Implemented Training Signals

1. **Classification loss**
   - Selectable by config:
     - `cross_entropy`
     - `cosface`
     - `arcface`
   - First recommended run uses `cross_entropy`.
   - ArcFace/CosFace configs are available for second-stage comparison.

2. **Image-text contrastive loss**
   - Images and captions with the same `landmark_id` are treated as positives.
   - Different landmarks in the batch become implicit negatives.
   - This keeps the image encoder and text encoder useful for natural-language retrieval.

3. **Hard-negative margin loss**
   - Reads `confusing_with` from each record in `labels.json`.
   - Penalizes cases where a confusing class logit is too close to or higher than the true class logit.
   - Can be disabled by config.

## Added Files

```text
src/landmark_candidate/losses.py
src/landmark_candidate/multitask_dataset.py
src/landmark_candidate/multitask_model.py
src/landmark_candidate/train_multitask.py
scripts/run_multitask_tmux.sh
scripts/mine_hard_negative_candidates.py
configs/experiments/mobileclip2_s4_partial_unfreeze_ce_hardneg.yaml
configs/experiments/mobileclip2_s4_lora_ce_hardneg.yaml
configs/experiments/mobileclip2_s4_partial_unfreeze_arcface_hardneg.yaml
```

## Primary Experiment

Use this first:

```bash
export DATA_ROOT=/workspace/landmark-assistant-model/Dataset
GPUS=1,2,3,4 NPROC=4 bash scripts/run_multitask_tmux.sh mobileclip2_s4_partial_unfreeze_ce_hardneg 0
```

This runs:

```bash
torchrun --nproc_per_node=4 \
  -m landmark_candidate.train_multitask \
  --config configs/experiments/mobileclip2_s4_partial_unfreeze_ce_hardneg.yaml \
  --data-root "$DATA_ROOT" \
  --split splits/kfold_seed20260513.json \
  --fold 0
```

## Comparison Experiments

LoRA comparison:

```bash
GPUS=1,2,3,4 NPROC=4 bash scripts/run_multitask_tmux.sh mobileclip2_s4_lora_ce_hardneg 0
```

ArcFace comparison:

```bash
GPUS=1,2,3,4 NPROC=4 bash scripts/run_multitask_tmux.sh mobileclip2_s4_partial_unfreeze_arcface_hardneg 0
```

## Output Files

Each run writes to:

```text
runs/<run_name>/
  best.pt
  config.yaml
  classes.json
  split_summary.json
  metrics.json
  predictions_val.jsonl
  predictions_test.jsonl
  predictions_text_queries.jsonl
  low_margin_val.csv
  low_margin_test.csv
```

Important fields:

- `val.top1_accuracy`: main model-selection metric.
- `val.top3_accuracy`: tie-breaker.
- `val.hard_case_top1_accuracy`: accuracy on records with `confusing_with`.
- `val.low_margin_count`: number of cases where Top-1 and Top-2 are too close.
- `text_query_retrieval`: text-query retrieval check using `landmark_text_catalog_v2.json` when available.

## Hard-negative Candidate Mining

After a run finishes, mine the model's actual confusion and low-margin cases:

```bash
python scripts/mine_hard_negative_candidates.py \
  --run-dir runs/<run_name> \
  --low-margin-threshold 0.05
```

This reads:

```text
runs/<run_name>/
  predictions_val.jsonl
  predictions_test.jsonl
  low_margin_val.csv
  low_margin_test.csv
```

It writes:

```text
runs/<run_name>/
  hard_negative_candidates.json
  hard_negative_candidates.csv
```

The miner combines three signals:

- **confusion matrix evidence**: the model predicts another landmark as Top-1.
- **low-margin evidence**: Top-1 and Top-2 are close, so the result is fragile.
- **nearest negative evidence**: the nearest non-target Top-3 class is repeatedly close.

The output is a review list, not an automatic label mutation. A human should inspect the examples, then decide whether to add the suggested pair to `confusing_with` in `labels.json`.

## Dataset Requirements

The expected dataset layout is:

```text
Dataset/
  <landmark_id>/
    labels.json
    images/
```

Each training record should use the current multi-task labeling guide:

```json
{
  "landmark_id": "deoksugung",
  "label_status": "confirmed",
  "training_role": "train_positive",
  "confusing_with": ["changgyeonggung"],
  "caption_set": [
    {
      "caption_type": "name_anchor",
      "target": "deoksugung",
      "text_ko": "덕수궁의 전통 궁궐 전각이 정면에서 크게 보인다.",
      "text_en": "A traditional palace hall of Deoksugung is prominently shown from the front."
    }
  ]
}
```

## Validation Performed Locally

Local validation was limited to checks that do not require the full training environment:

```bash
python -m compileall src
```

Result: passed.

```bash
python -c "<loss smoke test>"
```

Result: `cross_entropy`, `cosface`, `arcface`, contrastive loss, and hard-negative loss produced valid tensors.

The full training entrypoint requires the server environment with the project dependencies installed, especially `scikit-learn`, `torch`, `open_clip`, and `mobileclip`.

## Design Notes

- The original `landmark_candidate.train` Sprint 1 training path remains unchanged.
- The new path is `landmark_candidate.train_multitask`.
- Partial-unfreeze and LoRA are config-level experiment choices, not separate repositories.
- The first run should use CE because it is easier to debug and interpret.
- ArcFace/CosFace are implemented now so they can be compared later without redesigning the repo.
