param(
    [string]$ComsolRoot = "C:\Program Files\COMSOL\COMSOL64\Multiphysics_copy1",
    [string]$DatasetRoot = "",
    [string]$CandidateInput = "",
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$SolverRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$RepositoryRoot = (Resolve-Path (Join-Path $SolverRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
    $DatasetRoot = Join-Path $RepositoryRoot "model_dataset\cf4_o2_etch_caseA_nonlinear_sass"
}
if ([string]::IsNullOrWhiteSpace($CandidateInput)) {
    $CandidateInput = Join-Path $SolverRoot `
        "_out_m3v_matched\candidate_caseA_100nm_v1\candidate_input.h5"
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $SolverRoot `
        "_out_m3v_matched\caseA_100nm_canonical_p1_sectionwise_v2"
}
if (-not [System.IO.Path]::IsPathRooted($DatasetRoot)) {
    $DatasetRoot = Join-Path $RepositoryRoot $DatasetRoot
}
if (-not [System.IO.Path]::IsPathRooted($CandidateInput)) {
    $CandidateInput = Join-Path $SolverRoot $CandidateInput
}
if (-not [System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $SolverRoot $OutputDirectory
}
$DatasetRoot = [System.IO.Path]::GetFullPath($DatasetRoot)
$CandidateInput = [System.IO.Path]::GetFullPath($CandidateInput)
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)

$Compiler = Join-Path $ComsolRoot "bin\win64\comsolcompile.exe"
$Batch = Join-Path $ComsolRoot "bin\win64\comsolbatch.exe"
$JavaSource = Join-Path $PSScriptRoot "comsol\RunM3VMatchedCaseA100P1.java"
$SourceModel = Join-Path $DatasetRoot `
    "model\icp_rf_bias_cf4_o2_si_etching_caseP_caseA_SASS_formal_iondrag_theory_consistent_10_30_100nm.mph"
$Preparer = Join-Path $PSScriptRoot "prepare_comsol_p1_tables.py"
$Normalizer = Join-Path $PSScriptRoot "normalize_matched_reference.py"
foreach ($Required in @(
        $Compiler, $Batch, $JavaSource, $SourceModel,
        $CandidateInput, $Preparer, $Normalizer
    )) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required sectionwise-reference input does not exist: $Required"
    }
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "Sectionwise-reference output already exists: $OutputDirectory"
}

$SourceHashBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel).Hash.ToLowerInvariant()
$CandidateHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $CandidateInput).Hash.ToLowerInvariant()
$JavaHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $JavaSource).Hash.ToLowerInvariant()
$PreparerHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Preparer).Hash.ToLowerInvariant()
$NormalizerHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Normalizer).Hash.ToLowerInvariant()
$RunnerHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath).Hash.ToLowerInvariant()
$VersionText = (& $Batch -version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read COMSOL version; exit code $LASTEXITCODE"
}

Push-Location $SolverRoot
try {
    & uv run --locked python $Preparer $CandidateInput $OutputDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "Canonical P1 table preparation failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
foreach ($Name in @("dt_10us", "dt_5us", "dt_2p5us")) {
    New-Item -ItemType Directory -Path (Join-Path $OutputDirectory $Name) | Out-Null
}

$SourceCopy = Join-Path $OutputDirectory "source_copy.mph"
$ClassDirectory = Split-Path -Parent $JavaSource
$ClassFile = Join-Path $ClassDirectory "RunM3VMatchedCaseA100P1.class"
$CopyHash = ""
$SourceHashAfter = ""
try {
    Copy-Item -LiteralPath $SourceModel -Destination $SourceCopy
    $CopyHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceCopy).Hash.ToLowerInvariant()
    if ($CopyHash -ne $SourceHashBefore) {
        throw "The isolated source copy hash does not match the audited source MPH"
    }

    Push-Location $ClassDirectory
    try {
        & $Compiler $JavaSource
        if ($LASTEXITCODE -ne 0) {
            throw "COMSOL Java compilation failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }

    $BatchLog = Join-Path $OutputDirectory "comsol_batch.log"
    $ProcessLog = Join-Path $OutputDirectory "comsol_process.log"
    Push-Location $OutputDirectory
    try {
        & $Batch -inputfile $ClassFile -nosave -np 1 -batchlog $BatchLog *> $ProcessLog
        if ($LASTEXITCODE -ne 0) {
            throw "COMSOL sectionwise-reference run failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }

    $ExpectedHistory = Join-Path $OutputDirectory "dt_2p5us\history_raw_wide.csv"
    if (-not (Test-Path -LiteralPath $ExpectedHistory -PathType Leaf)) {
        throw "COMSOL did not emit the sectionwise-reference completion artifact"
    }
    Push-Location $SolverRoot
    try {
        & uv run --locked python $Normalizer $OutputDirectory
        if ($LASTEXITCODE -ne 0) {
            throw "Sectionwise-reference normalization failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }

} finally {
    foreach ($Name in @(
            "RunM3VMatchedCaseA100P1.class",
            "RunM3VMatchedCaseA100P1.class.status"
        )) {
        $GeneratedClass = Join-Path $ClassDirectory $Name
        if (Test-Path -LiteralPath $GeneratedClass) {
            Remove-Item -LiteralPath $GeneratedClass -Force
        }
    }
    if (Test-Path -LiteralPath $SourceCopy) {
        Remove-Item -LiteralPath $SourceCopy -Force
    }
}

if (Test-Path -LiteralPath $SourceCopy) {
    throw "The isolated source copy was not deleted: $SourceCopy"
}
$SourceHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel).Hash.ToLowerInvariant()
if ($SourceHashAfter -ne $SourceHashBefore) {
    throw "The audited source MPH changed during the loadCopy/no-save run"
}

$Provenance = [ordered]@{
    classification = "external_comsol_reference_canonical_p1_sectionwise"
    source_model = [System.IO.Path]::GetFullPath($SourceModel)
    source_sha256_before = $SourceHashBefore
    source_sha256_after = $SourceHashAfter
    source_unchanged = $true
    source_load_mode = "ModelUtil.loadCopy"
    model_saved = $false
    isolated_source_copy_sha256 = $CopyHash
    isolated_source_copy_retained = $false
    candidate_input = [System.IO.Path]::GetFullPath($CandidateInput)
    candidate_input_sha256 = $CandidateHash
    canonical_field_interpolation = "P1 sectionwise exact triangle connectivity"
    epstein_pressure_expression = "rho_P1*k_B_const*T_P1/(Mmix/N_A_const)"
    runner_sha256 = $RunnerHash
    java_source = [System.IO.Path]::GetFullPath($JavaSource)
    java_sha256 = $JavaHash
    preparer_sha256 = $PreparerHash
    normalizer_sha256 = $NormalizerHash
    comsol_root = [System.IO.Path]::GetFullPath($ComsolRoot)
    comsol_version = $VersionText
    process_count = 1
    generated_utc = [DateTime]::UtcNow.ToString("o")
}
$Provenance | ConvertTo-Json -Depth 4 | Set-Content `
    -LiteralPath (Join-Path $OutputDirectory "provenance_p1_sectionwise.json") `
    -Encoding utf8

$OutputRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
$ArtifactHashes = Get-ChildItem -LiteralPath $OutputRoot -Recurse -File |
    Where-Object { $_.Name -ne "artifact_hashes.csv" } |
    Sort-Object FullName |
    ForEach-Object {
        $RelativePath = $_.FullName.Substring($OutputRoot.Length)
        if ($RelativePath.StartsWith("\")) {
            $RelativePath = $RelativePath.Substring(1)
        }
        [pscustomobject][ordered]@{
            path = $RelativePath.Replace("\", "/")
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            bytes = $_.Length
        }
    }
$ArtifactHashes | Export-Csv -LiteralPath (Join-Path $OutputRoot "artifact_hashes.csv") `
    -NoTypeInformation -Encoding utf8

Write-Output "COMSOL canonical-P1 sectionwise reference completed: $OutputDirectory"
