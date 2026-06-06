# RTX 2060 6GB S3/S4 Multi-task Runbook

## Goal

RTX 2060 6GB에서 MobileCLIP2-S3와 MobileCLIP2-S4를 모두 실제로 돌려보고, 어느 지점에서 가능한지 또는 OOM이 나는지 확인한다.

이 runbook은 최종 성능을 단정하기 위한 문서가 아니라, 로컬 6GB GPU에서 다음을 확인하기 위한 실행 절차다.

- S3/S4 모델 로드 가능 여부
- forward/backward 가능 여부
- CE + image-text contrastive + hard-negative loss 동작 여부
- 1 epoch probe의 loss/validation 산출 여부
- full 30 epoch 실행 가능 여부
- OOM 발생 시 어떤 config에서 실패했는지

## Dataset

기본 데이터셋 경로:

```powershell
D:\Dataset_0601
```

현재 기대 구조:

```text
Dataset_0601/
  confusion_prior_v1.json
  <landmark_id>/
    catalog.json
    labels.json
    images/
```

`confusion_prior_v1.json`은 root에 하나만 둔다. 각 landmark 폴더 안의 `confusion_prior.json`은 legacy 파일이므로 학습 입력으로 쓰지 않는다.

## Added Configs

Probe configs:

- `configs/experiments/mobileclip2_s3_rtx2060_probe.yaml`
- `configs/experiments/mobileclip2_s4_rtx2060_probe.yaml`

Full configs:

- `configs/experiments/mobileclip2_s3_rtx2060_full.yaml`
- `configs/experiments/mobileclip2_s4_rtx2060_full.yaml`

공통 학습 목표:

- classification: cross entropy
- retrieval alignment: image-text contrastive
- hard case separation: hard-negative margin loss from `confusing_with`

## Why Gradient Accumulation Was Added

RTX 2060 6GB에서는 batch size를 크게 잡기 어렵다. Contrastive loss는 batch 안의 image-caption 관계를 보므로 batch가 작으면 신호가 약해질 수 있다.

따라서 실제 GPU batch는 작게 유지하고, `runtime.grad_accum_steps`로 여러 micro-batch를 누적해 optimizer step을 수행한다.

예:

```yaml
training:
  batch_size_per_gpu: 1
runtime:
  grad_accum_steps: 32
```

이 설정은 한 번에 1장씩 올리되, 32번 누적 후 optimizer step을 수행한다.

## Recommended Execution

PowerShell에서 repo root로 이동:

```powershell
cd C:\Users\hi\Downloads\종설_작업중\landmark-assistant-model-ver2
```

가상환경 활성화 후 실행한다.

Probe만 실행:

```powershell
.\scripts\run_rtx2060_multitask.ps1 -DataRoot D:\Dataset_0601 -ContinueOnError
```

Probe + full run까지 모두 실행:

```powershell
.\scripts\run_rtx2060_multitask.ps1 -DataRoot D:\Dataset_0601 -Full -ContinueOnError
```

`-ContinueOnError`를 주면 S4에서 OOM이 나도 다음 config로 넘어간다.

## Interpreting Results

성공으로 볼 수 있는 최소 기준:

- split 생성 성공
- model load 성공
- 첫 train batch 통과
- 1 epoch 종료
- `runs/<run_name>/metrics.json` 생성
- `runs/<run_name>/best.pt` 생성

OOM으로 볼 수 있는 로그:

```text
CUDA out of memory
```

OOM이 나면 먼저 다음 순서로 낮춘다.

1. `training.batch_size_per_gpu`
2. `data.max_captions_per_image`
3. `model.text_unfreeze_ratio`
4. `model.image_unfreeze_ratio`
5. `loss.image_text_contrastive.enabled`

단, contrastive를 끄면 multi-task 비교 의미가 약해지므로 마지막 수단으로 둔다.

## Expected Reading Order

1. `logs/*.log`에서 모델 로드와 OOM 여부 확인
2. `runs/*/config.yaml`에서 실제 사용 config 확인
3. `runs/*/metrics.json`에서 val/test/text retrieval 확인
4. `runs/*/low_margin_val.csv`에서 hard/confusing case 확인
5. S3와 S4의 `val_top1_accuracy`, `hard_case_top1_accuracy`, GPU 가능 여부를 비교

## Caveat

RTX 2060 6GB 결과는 최종 성능 판단보다 local feasibility 판단에 가깝다. 최종 비교는 같은 dataset fingerprint, 같은 fold, 같은 config 계열로 서버 GPU에서 재실행해 확인하는 것이 안전하다.
