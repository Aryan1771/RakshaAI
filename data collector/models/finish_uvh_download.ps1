param([Parameter(Mandatory = $true)][int]$DownloadProcessId)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$datasetRoot = Join-Path $projectRoot 'dataset\UVH-26'
$log = Join-Path $datasetRoot 'download-finalize.log'
Add-Content -LiteralPath $log -Value "[$(Get-Date -Format o)] waiting for Hugging Face download PID $DownloadProcessId"
while (Get-Process -Id $DownloadProcessId -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 5
& (Join-Path $projectRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'finalize_uvh_manifest.py') *>> $log
if ($LASTEXITCODE -ne 0) {
    Add-Content -LiteralPath $log -Value "[$(Get-Date -Format o)] source manifest finalization failed with exit $LASTEXITCODE"
    exit $LASTEXITCODE
}
$manifest = Get-Content -LiteralPath (Join-Path $datasetRoot 'source_manifest.json') -Raw | ConvertFrom-Json
if ($manifest.download_status -ne 'complete') {
    Add-Content -LiteralPath $log -Value "[$(Get-Date -Format o)] download remains incomplete; kept partial data and did not prepare training split"
    exit 2
}
& (Join-Path $projectRoot '.venv\Scripts\python.exe') -m models.raksha_training annotate-uvh-available --config models/configs/phase2.full.json *>> $log
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $projectRoot '.venv\Scripts\python.exe') -m models.raksha_training prepare-uvh --config models/configs/phase2.full.json *>> $log
Add-Content -LiteralPath $log -Value "[$(Get-Date -Format o)] annotation and split preparation exit $LASTEXITCODE"
exit $LASTEXITCODE
