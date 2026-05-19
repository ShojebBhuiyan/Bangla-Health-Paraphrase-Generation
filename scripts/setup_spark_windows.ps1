#Requires -Version 5.1
<#
.SYNOPSIS
    Configure Spark/Hadoop environment on Windows for local PySpark runs.
#>
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$hadoopDir = Join-Path $ProjectRoot "hadoop"
$binDir = Join-Path $hadoopDir "bin"
$winutilsPath = Join-Path $binDir "winutils.exe"
$sentinelPath = Join-Path $ProjectRoot "outputs\.spark_ready"

New-Item -ItemType Directory -Force -Path $binDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "outputs") | Out-Null

if (-not (Test-Path $winutilsPath)) {
    Write-Host "Downloading winutils.exe for Hadoop 3.3..."
    $urls = @(
        "https://github.com/kontext-tech/winutils/raw/master/hadoop-3.3.5/bin/winutils.exe",
        "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.5/bin/winutils.exe"
    )
    $downloaded = $false
    foreach ($url in $urls) {
        try {
            Invoke-WebRequest -Uri $url -OutFile $winutilsPath -UseBasicParsing -TimeoutSec 60
            if ((Get-Item $winutilsPath).Length -gt 0) {
                $downloaded = $true
                break
            }
        } catch {
            Write-Warning "Failed to download from $url : $_"
        }
    }
    if (-not $downloaded) {
        Write-Warning "Could not download winutils.exe. Spark checkpointing may fail on Windows."
        Write-Warning "Manually place winutils.exe at: $winutilsPath"
    }
}

$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonExe = $venvPython
} else {
    $pythonExe = (Get-Command python).Source
}

if (-not $env:SPARK_HOME -and (Test-Path "C:\Spark\bin\spark-submit.cmd")) {
    $env:SPARK_HOME = "C:\Spark"
}

$env:HADOOP_HOME = $hadoopDir
$env:PATH = "$binDir;$env:PATH"
$env:PYSPARK_PYTHON = $pythonExe
$env:PYSPARK_DRIVER_PYTHON = $pythonExe
$env:SPARK_LOCAL_IP = "127.0.0.1"

$lines = @(
    "HADOOP_HOME=$hadoopDir"
    "SPARK_HOME=$($env:SPARK_HOME)"
    "PYSPARK_PYTHON=$pythonExe"
    "Configured at $(Get-Date -Format o)"
)
Set-Content -Path $sentinelPath -Value ($lines -join "`n")
Write-Host "Spark Windows setup complete."
Write-Host "  HADOOP_HOME=$hadoopDir"
Write-Host "  SPARK_HOME=$($env:SPARK_HOME)"
Write-Host "  PYSPARK_PYTHON=$pythonExe"
