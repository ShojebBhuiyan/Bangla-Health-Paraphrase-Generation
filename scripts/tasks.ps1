#Requires -Version 5.1
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("setup", "preprocess", "eda", "augment", "train-mt5-lora", "train-baselines", "evaluate", "test")]
    [string]$Task
)

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

switch ($Task) {
    "setup" {
        pip install -r requirements.txt
        & "$Root\scripts\setup_spark_windows.ps1"
    }
    "preprocess" { python scripts/run_preprocess.py }
    "eda" { python scripts/run_eda.py }
    "augment" { python scripts/run_augment.py }
    "train-mt5-lora" { python scripts/run_train.py --model mt5_lora }
    "train-baselines" {
        python scripts/run_train.py --model mt5_baseline
    }
    "evaluate" { python scripts/run_evaluate.py }
    "test" { pytest tests/ -v }
}
