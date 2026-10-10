param(
    [string]$ComsolRoot = "C:\Program Files\COMSOL\COMSOL64\Multiphysics_copy1",
    [string]$CandidateInput = "",
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$ExpectedCandidateSha256,
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$InvocationDirectory = (Get-Location).ProviderPath
$ComsolRoot = [IO.Path]::GetFullPath($ComsolRoot, $InvocationDirectory)
$SolverRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$RepositoryRoot = (Resolve-Path (Join-Path $SolverRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($CandidateInput)) {
    $CandidateInput = Join-Path $SolverRoot `
        "_out_m3c1\theory_100nm_30ms_candidate_v3\caseP\candidate_input.h5"
} else {
    $CandidateInput = [IO.Path]::GetFullPath($CandidateInput, $InvocationDirectory)
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $SolverRoot "_out_m3c3\caseP_negative_ion_primitives_v1"
} else {
    $OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory, $InvocationDirectory)
}

$Compiler = Join-Path $ComsolRoot "bin\win64\comsolcompile.exe"
$Batch = Join-Path $ComsolRoot "bin\win64\comsolbatch.exe"
$ClassName = "ExportM3C3CasePNegativeIonPrimitives"
$JavaSource = Join-Path $PSScriptRoot "comsol\$ClassName.java"
$Normalizer = Join-Path $PSScriptRoot "normalize_m3c3_negative_ion_primitives.py"
$SourceModel = Join-Path $RepositoryRoot (
    "model_dataset\cf4_o2_etch_caseA_nonlinear_sass\model\" +
    "icp_rf_bias_cf4_o2_si_etching_caseP_caseA_SASS_" +
    "formal_iondrag_theory_consistent_10_30_100nm.mph"
)
$ExpectedSourceSha256 = "3BBF08E3469758313EAC5DE473A7A0DD4CC9A6F72C9722393229F0B856E9B524"
$ExpectedComsolVersion = "COMSOL Multiphysics 6.4.0.429"

foreach ($Required in @(
    $Compiler, $Batch, $JavaSource, $Normalizer, $SourceModel, $CandidateInput
)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required M3-C3 primitive-export input does not exist: $Required"
    }
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "M3-C3 primitive-export output already exists: $OutputDirectory"
}
$SourceHashBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel).Hash
$CandidateHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $CandidateInput).Hash
if ($SourceHashBefore -ne $ExpectedSourceSha256) {
    throw "Locked source MPH hash mismatch: expected $ExpectedSourceSha256, found $SourceHashBefore"
}
if ($CandidateHash -ne $ExpectedCandidateSha256) {
    throw "Locked common-P1 H5 hash mismatch: expected $ExpectedCandidateSha256, found $CandidateHash"
}
$VersionText = (& $Batch -version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $VersionText -notmatch [regex]::Escape($ExpectedComsolVersion)) {
    throw "COMSOL version mismatch or version query failed: $VersionText"
}

New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$SourceCopy = Join-Path $OutputDirectory "source_copy.mph"
$StagedJava = Join-Path $OutputDirectory "$ClassName.java"
$ClassFile = Join-Path $OutputDirectory "$ClassName.class"
$ClassStatus = "$ClassFile.status"
$BatchLog = Join-Path $OutputDirectory "comsol_batch.log"
$ProcessLog = Join-Path $OutputDirectory "comsol_process.log"
$DomainCsv = Join-Path $OutputDirectory "domain_cache.csv"
$BoundaryCsv = Join-Path $OutputDirectory "boundary_cache.csv"
$CanonicalOutput = Join-Path $OutputDirectory "candidate_input_three_current.h5"
$Receipt = Join-Path $OutputDirectory "primitive_receipt.json"
$Succeeded = $false
try {
    Copy-Item -LiteralPath $SourceModel -Destination $SourceCopy
    Copy-Item -LiteralPath $JavaSource -Destination $StagedJava
    Push-Location $OutputDirectory
    try {
        & $Compiler $StagedJava
        if ($LASTEXITCODE -ne 0) {
            throw "COMSOL Java compilation failed with exit code $LASTEXITCODE"
        }
        & $Batch -inputfile $ClassFile -nosave -np 1 -batchlog $BatchLog *> $ProcessLog
        if ($LASTEXITCODE -ne 0) {
            throw "COMSOL M3-C3 primitive export failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
    if (-not (Select-String -LiteralPath $ProcessLog -SimpleMatch "M3C3_PRIMITIVE_EXPORT|pass|")) {
        throw "M3-C3 primitive exporter did not emit its completion record"
    }
    if (Select-String -LiteralPath $ProcessLog, $BatchLog -Pattern `
            'Error running java class\.|/\*+Error\*+/') {
        throw "COMSOL primitive exporter reported a native execution error"
    }
    foreach ($Csv in @($DomainCsv, $BoundaryCsv)) {
        if (-not (Test-Path -LiteralPath $Csv -PathType Leaf)) {
            throw "M3-C3 primitive exporter did not create $Csv"
        }
    }
    $SourceHashAfterComsol = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel).Hash
    if ($SourceHashAfterComsol -ne $SourceHashBefore) {
        throw "The locked source MPH changed during COMSOL export"
    }
    Remove-Item -LiteralPath $SourceCopy, $ClassFile, $ClassStatus -ErrorAction SilentlyContinue

    Push-Location $SolverRoot
    try {
        & uv run --locked python -m tools.vv.comsol.normalize_m3c3_negative_ion_primitives `
            --canonical-input $CandidateInput `
            --expected-input-sha256 $ExpectedCandidateSha256.ToLowerInvariant() `
            --domain-csv $DomainCsv `
            --boundary-csv $BoundaryCsv `
            --source-mph $SourceModel `
            --java-source $JavaSource `
            --runner-script $PSCommandPath `
            --output $CanonicalOutput `
            --receipt $Receipt
        if ($LASTEXITCODE -ne 0) {
            throw "M3-C3 primitive normalization failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
    $ReceiptData = Get-Content -LiteralPath $Receipt -Raw | ConvertFrom-Json
    if ([string]$ReceiptData.status -ne "PASS" -or -not [bool]$ReceiptData.source_unchanged) {
        throw "M3-C3 primitive receipt did not pass"
    }
    $Succeeded = $true
} finally {
    Remove-Item -LiteralPath $SourceCopy -ErrorAction SilentlyContinue
    $SourceHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel).Hash
    if ($SourceHashAfter -ne $SourceHashBefore) {
        throw "The locked source MPH changed during M3-C3 primitive export"
    }
}

if (-not $Succeeded) {
    throw "M3-C3 Case-P negative-ion primitive export did not complete"
}
Write-Output "M3-C3 Case-P negative-ion primitives passed: $OutputDirectory"
