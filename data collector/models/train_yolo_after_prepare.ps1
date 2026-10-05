$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$prepared = Join-Path $projectRoot 'dataset\UVH-26\derived\prepared\split_manifest.json'
$log = Join-Path $projectRoot 'models\runs\phase2\yolo\background_training.log'
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
Add-Content -LiteralPath $log -Value "[$(Get-Date -Format o)] waiting for verified UVH split preparation at $prepared"
while (-not (Test-Path -LiteralPath $prepared)) {
    Start-Sleep -Seconds 30
}
$python = Join-Path $projectRoot '.venv-phase2\Scripts\python.exe'
$cudaReady = $false
for ($attempt = 1; $attempt -le 60; $attempt++) {
    if (Test-Path -LiteralPath $python) {
        $cuda = & $python -c "import torch; print(torch.cuda.is_available())" 2>> $log
        if ($LASTEXITCODE -eq 0 -and $cuda -contains 'True') {
            $cudaReady = $true
            break
        }
    }
    Start-Sleep -Seconds 60
}
if (-not $cudaReady) {
    Add-Content -LiteralPath $log -Value "[$(Get-Date -Format o)] CUDA-enabled PyTorch was not ready; detector training was not started."
    exit 3
}
Add-Content -LiteralPath $log -Value "[$(Get-Date -Format o)] starting YOLOv8-S train/selection and official UVH validation benchmark"
& $python -m models.raksha_training train-yolo --config models/configs/phase2.full.json *>> $log
$exitCode = $LASTEXITCODE
Add-Content -LiteralPath $log -Value "[$(Get-Date -Format o)] YOLO training/evaluation exit $exitCode"
if ($exitCode -eq 0) {
    # Remove UVH source and derived data only after successful training and benchmark evaluation.
    $datasetRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot 'dataset'))
    $uvhRoot = [IO.Path]::GetFullPath((Join-Path $datasetRoot 'UVH-26'))
    $expectedUvhRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot 'dataset\UVH-26'))
    if (-not [string]::Equals($uvhRoot, $expectedUvhRoot, [StringComparison]::OrdinalIgnoreCase) -or
        -not [string]::Equals((Split-Path -Parent $uvhRoot), $datasetRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing unsafe UVH cleanup target: $uvhRoot"
    }
    if (Test-Path -LiteralPath $uvhRoot) {
        Add-Content -LiteralPath $log -Value "[$(Get-Date -Format o)] successful YOLO run; deleting only $uvhRoot as requested"
        Remove-Item -LiteralPath $uvhRoot -Recurse -Force
        Add-Content -LiteralPath $log -Value "[$(Get-Date -Format o)] UVH-26 local dataset removed; model artifacts retained under models/runs"
    }

    # Remove the exploratory partial-download checkpoint and logs as UVH-derived artifacts.
    $phase2Root = [IO.Path]::GetFullPath((Join-Path $projectRoot 'models\runs\phase2'))
    $partialRun = [IO.Path]::GetFullPath((Join-Path $phase2Root 'uvh26_partial'))
    if (-not [string]::Equals((Split-Path -Parent $partialRun), $phase2Root, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing unsafe UVH partial-run cleanup target: $partialRun"
    }
    if (Test-Path -LiteralPath $partialRun) {
        Add-Content -LiteralPath $log -Value "[$(Get-Date -Format o)] removing exploratory UVH-derived partial run $partialRun"
        Remove-Item -LiteralPath $partialRun -Recurse -Force
    }

    # Delete only cache entries belonging to this dataset; leave unrelated HF caches untouched.
    $hf = Join-Path $projectRoot '.venv\Scripts\hf.exe'
    if (Test-Path -LiteralPath $hf) {
        foreach ($cacheId in @('dataset/iisc-aim/UVH-26', 'datasets/iisc-aim/UVH-26')) {
            & $hf cache rm $cacheId --yes *>> $log
        }
    }
}
exit $exitCode
