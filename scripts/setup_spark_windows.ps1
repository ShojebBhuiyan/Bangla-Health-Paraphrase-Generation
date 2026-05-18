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
    $url = "https://github.com/kontext-tech/winutils/raw/master/hadoop-3.3.5/bin/winutils.exe"
    Invoke-WebRequest -Uri $url -OutFile $winutilsPath
}

$env:HADOOP_HOME = $hadoopDir
$env:PATH = "$binDir;$env:PATH"
$env:PYSPARK_PYTHON = (Get-Command python).Source
$env:PYSPARK_DRIVER_PYTHON = $env:PYSPARK_PYTHON
$env:SPARK_LOCAL_IP = "127.0.0.1"

Set-Content -Path $sentinelPath -Value "HADOOP_HOME=$hadoopDir`nConfigured at $(Get-Date -Format o)"
Write-Host "Spark Windows setup complete. HADOOP_HOME=$hadoopDir"
