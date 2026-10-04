param(
    [string]$ComsolRoot = "C:\Program Files\COMSOL\COMSOL64\Multiphysics_copy1",
    [string]$DatasetRoot = "",
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$SolverRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$RepositoryRoot = (Resolve-Path (Join-Path $SolverRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
    $DatasetRoot = Join-Path $RepositoryRoot "model_dataset\cf4_o2_etch_caseA_nonlinear_sass"
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $SolverRoot "evidence\m3v"
}

$Compiler = Join-Path $ComsolRoot "bin\win64\comsolcompile.exe"
$Batch = Join-Path $ComsolRoot "bin\win64\comsolbatch.exe"
$JavaSource = Join-Path $PSScriptRoot "comsol\InspectM3VModels.java"
$TheoryModel = Join-Path $DatasetRoot "model\icp_rf_bias_cf4_o2_si_etching_caseP_caseA_SASS_formal_iondrag_theory_consistent_10_30_100nm.mph"
$ImageModel = Join-Path $DatasetRoot "model\icp_rf_bias_cf4_o2_si_etching_caseP_caseA_SASS_formal_iondrag_image_minimal_corrected_10_30_100nm.mph"

foreach ($RequiredPath in @($Compiler, $Batch, $JavaSource, $TheoryModel, $ImageModel)) {
    if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
        throw "Required M3-V input does not exist: $RequiredPath"
    }
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$GeneratedArtifacts = @(
    "README.md",
    "case_metrics.csv",
    "comparison_manifest.json",
    "comsol_launcher.log",
    "comsol_launcher.tmp.log",
    "comsol_model_audit.log",
    "comsol_model_audit.txt",
    "comsol_model_inventory.jsonl",
    "comsol_process_output.log",
    "comsol_process_output.tmp.log",
    "comsol_version.txt",
    "force_relevance.csv",
    "gates.csv",
    "model_inventory.json",
    "mph_hashes.csv",
    "priority_decision.csv",
    "variant_config_diff.csv",
    "variant_sensitivity.csv",
    "variant_time_history.csv"
)
foreach ($Artifact in $GeneratedArtifacts) {
    Remove-Item -LiteralPath (Join-Path $OutputDirectory $Artifact) -ErrorAction SilentlyContinue
}
$TheoryHashBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $TheoryModel).Hash
$ImageHashBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $ImageModel).Hash
$VersionText = & $Batch -version 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read COMSOL version; exit code $LASTEXITCODE"
}
$VersionText | Set-Content -LiteralPath (Join-Path $OutputDirectory "comsol_version.txt") -Encoding utf8

Push-Location (Split-Path -Parent $JavaSource)
try {
    & $Compiler $JavaSource
    if ($LASTEXITCODE -ne 0) {
        throw "COMSOL Java compilation failed with exit code $LASTEXITCODE"
    }

    $LauncherLog = Join-Path $OutputDirectory "comsol_launcher.tmp.log"
    $ProcessOutputLog = Join-Path $OutputDirectory "comsol_process_output.tmp.log"
    $AuditLog = Join-Path $OutputDirectory "comsol_model_audit.txt"
    $InventoryLog = Join-Path $OutputDirectory "comsol_model_inventory.jsonl"
    $ComsolLogRoot = Join-Path $env:USERPROFILE ".comsol\v64\logs"
    $AuditStart = Get-Date
    $ClassFile = Join-Path (Split-Path -Parent $JavaSource) "InspectM3VModels.class"
    Push-Location (Join-Path $DatasetRoot "model")
    try {
        & $Batch -inputfile $ClassFile -nosave -np 1 -batchlog $LauncherLog *> $ProcessOutputLog
        if ($LASTEXITCODE -ne 0) {
            throw "COMSOL read-only model audit failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
    $ServerLog = Get-ChildItem -LiteralPath $ComsolLogRoot -Filter "batch*.log" -File |
        Where-Object { $_.Name -notmatch "_render" -and $_.LastWriteTime -ge $AuditStart.AddSeconds(-1) } |
        Sort-Object LastWriteTime -Descending |
        Where-Object { Select-String -LiteralPath $_.FullName -SimpleMatch `
            'M3V_JSON|{"type":"audit_pass"' -Quiet } |
        Select-Object -First 1
    if ($null -eq $ServerLog) {
        throw "COMSOL read-only model audit did not emit its completion record"
    }
    Copy-Item -LiteralPath $ServerLog.FullName -Destination $AuditLog -Force
    Get-Content -LiteralPath $AuditLog | ForEach-Object {
        if ($_ -match 'M3V_JSON\|(\{.*\}) \[com\.comsol\.util\]$') {
            "M3V_JSON|$($Matches[1])"
        }
    } | Set-Content -LiteralPath $InventoryLog -Encoding utf8
    Remove-Item -LiteralPath $LauncherLog -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $ProcessOutputLog -ErrorAction SilentlyContinue

    $TheoryHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $TheoryModel).Hash
    $ImageHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $ImageModel).Hash
    if ($TheoryHashAfter -ne $TheoryHashBefore -or $ImageHashAfter -ne $ImageHashBefore) {
        throw "A source MPH hash changed during the read-only M3-V audit"
    }
    @(
        "variant,path,sha256_before,sha256_after,unchanged"
        "relative_flow_screened_collection_orbital_v1,`"$TheoryModel`",$TheoryHashBefore,$TheoryHashAfter,true"
        "electric_field_aligned_image_sensitivity_v1,`"$ImageModel`",$ImageHashBefore,$ImageHashAfter,true"
    ) | Set-Content -LiteralPath (Join-Path $OutputDirectory "mph_hashes.csv") -Encoding utf8
} finally {
    @("InspectM3VModels.class", "InspectM3VModels.class.status") | ForEach-Object {
        Remove-Item -LiteralPath (Join-Path (Split-Path -Parent $JavaSource) $_) `
            -ErrorAction SilentlyContinue
    }
    Pop-Location
}

Push-Location $SolverRoot
try {
    & uv run --locked python tools/vv/comsol/evaluate_dataset.py `
        --dataset-root $DatasetRoot `
        --output-dir $OutputDirectory `
        --model-audit-log (Join-Path $OutputDirectory "comsol_model_inventory.jsonl") `
        --strict
    if ($LASTEXITCODE -ne 0) {
        throw "M3-V dataset evaluation failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Output "M3-V completed: $OutputDirectory"
