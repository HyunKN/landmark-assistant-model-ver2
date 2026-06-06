param(
  [string]$DataRoot = "D:\Dataset_0601",
  [string]$Fold = "0",
  [switch]$Full,
  [switch]$ContinueOnError
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$env:PYTHONPATH = "$repoRoot\src;$env:PYTHONPATH"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:WANDB_PROJECT = if ($env:WANDB_PROJECT) { $env:WANDB_PROJECT } else { "landmark-assistant-sprint2" }

New-Item -ItemType Directory -Force -Path "splits", "logs", "runs" | Out-Null

Write-Host "[split] data root: $DataRoot"
python -m landmark_candidate.split_data `
  --data-root $DataRoot `
  --out "splits\kfold_seed20260513.json" `
  --seed 20260513 `
  --folds 5 `
  --test-ratio 0.15

$configs = @(
  "configs\experiments\mobileclip2_s3_rtx2060_probe.yaml",
  "configs\experiments\mobileclip2_s4_rtx2060_probe.yaml"
)

if ($Full) {
  $configs += @(
    "configs\experiments\mobileclip2_s3_rtx2060_full.yaml",
    "configs\experiments\mobileclip2_s4_rtx2060_full.yaml"
  )
}

foreach ($config in $configs) {
  $stem = [System.IO.Path]::GetFileNameWithoutExtension($config)
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $logPath = "logs\$stem-fold$Fold-$stamp.log"
  Write-Host "[train] $config"
  Write-Host "[log] $logPath"
  try {
    python -m landmark_candidate.train_multitask `
      --config $config `
      --data-root $DataRoot `
      --split "splits\kfold_seed20260513.json" `
      --fold $Fold 2>&1 | Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) {
      throw "train command exited with code $LASTEXITCODE"
    }
  } catch {
    Write-Host "[failed] $config"
    Write-Host $_
    if (-not $ContinueOnError) {
      throw
    }
  }
}

Write-Host "[done] finished requested RTX 2060 multitask runs"
