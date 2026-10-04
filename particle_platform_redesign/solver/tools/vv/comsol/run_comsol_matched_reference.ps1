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
    $OutputDirectory = Join-Path $SolverRoot "_out_m3v_matched\caseA_100nm_common_v1"
}

$Compiler = Join-Path $ComsolRoot "bin\win64\comsolcompile.exe"
$Batch = Join-Path $ComsolRoot "bin\win64\comsolbatch.exe"
$JavaSource = Join-Path $PSScriptRoot "comsol\RunM3VMatchedCaseA100.java"
$SourceModel = Join-Path $DatasetRoot `
    "model\icp_rf_bias_cf4_o2_si_etching_caseP_caseA_SASS_formal_iondrag_theory_consistent_10_30_100nm.mph"
$Normalizer = Join-Path $PSScriptRoot "normalize_matched_reference.py"
foreach ($Required in @($Compiler, $Batch, $JavaSource, $SourceModel, $Normalizer)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required matched-reference input does not exist: $Required"
    }
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "Matched-reference output already exists: $OutputDirectory"
}
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
foreach ($Name in @("dt_10us", "dt_5us", "dt_2p5us")) {
    New-Item -ItemType Directory -Path (Join-Path $OutputDirectory $Name) | Out-Null
}

$SourceHashBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel).Hash.ToLowerInvariant()
$SourceCopy = Join-Path $OutputDirectory "source_copy.mph"
Copy-Item -LiteralPath $SourceModel -Destination $SourceCopy
$CopyHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceCopy).Hash.ToLowerInvariant()
if ($CopyHash -ne $SourceHashBefore) {
    throw "The isolated source copy hash does not match the audited source MPH"
}
$JavaHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $JavaSource).Hash.ToLowerInvariant()
$VersionText = (& $Batch -version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read COMSOL version; exit code $LASTEXITCODE"
}

Push-Location (Split-Path -Parent $JavaSource)
try {
    & $Compiler $JavaSource
    if ($LASTEXITCODE -ne 0) {
        throw "COMSOL Java compilation failed with exit code $LASTEXITCODE"
    }

    $ClassFile = Join-Path (Split-Path -Parent $JavaSource) "RunM3VMatchedCaseA100.class"
    $BatchLog = Join-Path $OutputDirectory "comsol_batch.log"
    $ProcessLog = Join-Path $OutputDirectory "comsol_process.log"
    Push-Location $OutputDirectory
    try {
        & $Batch -inputfile $ClassFile -nosave -np 1 -batchlog $BatchLog *> $ProcessLog
        if ($LASTEXITCODE -ne 0) {
            throw "COMSOL matched-reference run failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
} finally {
    foreach ($Name in @("RunM3VMatchedCaseA100.class", "RunM3VMatchedCaseA100.class.status")) {
        Remove-Item -LiteralPath (Join-Path (Split-Path -Parent $JavaSource) $Name) `
            -ErrorAction SilentlyContinue
    }
    Pop-Location
}

$ExpectedHistory = Join-Path $OutputDirectory "dt_2p5us\history_raw_wide.csv"
if (-not (Test-Path -LiteralPath $ExpectedHistory -PathType Leaf)) {
    throw "COMSOL did not emit the matched-reference completion artifact: $ExpectedHistory"
}
$SourceHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel).Hash.ToLowerInvariant()
if ($SourceHashAfter -ne $SourceHashBefore) {
    throw "The audited source MPH changed during the loadCopy/no-save run"
}

Push-Location $SolverRoot
try {
    & uv run --locked python $Normalizer $OutputDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "Matched-reference normalization failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

$Provenance = [ordered]@{
    classification = "external_comsol_reference"
    source_model = [System.IO.Path]::GetFullPath($SourceModel)
    source_sha256_before = $SourceHashBefore
    source_sha256_after = $SourceHashAfter
    source_unchanged = $true
    source_load_mode = "ModelUtil.loadCopy"
    model_saved = $false
    isolated_source_copy_sha256 = $CopyHash
    isolated_source_copy_retained = $false
    java_source = [System.IO.Path]::GetFullPath($JavaSource)
    java_sha256 = $JavaHash
    comsol_root = [System.IO.Path]::GetFullPath($ComsolRoot)
    comsol_version = $VersionText
    process_count = 1
    generated_utc = [DateTime]::UtcNow.ToString("o")
}
$Provenance | ConvertTo-Json -Depth 4 | Set-Content `
    -LiteralPath (Join-Path $OutputDirectory "provenance.json") -Encoding utf8
Remove-Item -LiteralPath $SourceCopy
$OutputRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
$ArtifactHashes = Get-ChildItem -LiteralPath $OutputRoot -Recurse -File |
    Where-Object { $_.Name -ne "artifact_hashes.csv" } |
    Sort-Object FullName |
    ForEach-Object {
        [ordered]@{
            path = [System.IO.Path]::GetRelativePath($OutputRoot, $_.FullName).Replace("\", "/")
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            bytes = $_.Length
        }
    }
$ArtifactHashes | Export-Csv -LiteralPath (Join-Path $OutputRoot "artifact_hashes.csv") `
    -NoTypeInformation -Encoding utf8

Write-Output "COMSOL matched reference completed: $OutputDirectory"
