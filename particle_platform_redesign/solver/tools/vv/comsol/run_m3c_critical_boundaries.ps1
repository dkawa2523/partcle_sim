param(
    [string]$ComsolRoot = "C:\Program Files\COMSOL\COMSOL64\Multiphysics_copy1",
    [string]$OutputDirectory = "",
    [string]$EvidenceDirectory = ""
)

$ErrorActionPreference = "Stop"
$SolverRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$ConfigPath = Join-Path $PSScriptRoot "cases\m3c_critical_boundaries_v1.json"
$JavaSource = Join-Path $PSScriptRoot "comsol\RunM3CCriticalBoundaries.java"
$Evaluator = Join-Path $PSScriptRoot "evaluate_m3c_critical_boundaries.py"
$Runner = $PSCommandPath
$Compiler = Join-Path $ComsolRoot "bin\win64\comsolcompile.exe"
$Batch = Join-Path $ComsolRoot "bin\win64\comsolbatch.exe"

foreach ($Required in @($ConfigPath, $JavaSource, $Evaluator, $Runner, $Compiler, $Batch)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required critical-boundary input does not exist: $Required"
    }
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required to run the public-API comparison"
}

$Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $SolverRoot "_out_m3c\critical_boundaries_v1"
}
if ([string]::IsNullOrWhiteSpace($EvidenceDirectory)) {
    $EvidenceDirectory = Join-Path $SolverRoot "evidence\m3c0\critical_boundaries_v1"
}
$OutputDirectory = [System.IO.Path]::GetFullPath(
    $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDirectory)
)
$EvidenceDirectory = [System.IO.Path]::GetFullPath(
    $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($EvidenceDirectory)
)
foreach ($Destination in @($OutputDirectory, $EvidenceDirectory)) {
    if (Test-Path -LiteralPath $Destination) {
        throw "Critical-boundary destination already exists: $Destination"
    }
}

$VersionText = (& $Batch -version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read COMSOL version; exit code $LASTEXITCODE"
}
if ($VersionText -notmatch [regex]::Escape($Config.expected_comsol_version)) {
    throw "COMSOL version does not match the locked configuration: $VersionText"
}

New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
foreach ($Step in $Config.case.fixed_rk4_steps) {
    New-Item -ItemType Directory -Path (Join-Path $OutputDirectory $Step.label) | Out-Null
}

$StagedJava = Join-Path $OutputDirectory "RunM3CCriticalBoundaries.java"
$StagedConfig = Join-Path $OutputDirectory "m3c_critical_boundaries_v1.json"
$StagedEvaluator = Join-Path $OutputDirectory "evaluate_m3c_critical_boundaries.py"
$StagedRunner = Join-Path $OutputDirectory "run_m3c_critical_boundaries.ps1"
$ClassFile = Join-Path $OutputDirectory "RunM3CCriticalBoundaries.class"
$StatusFile = Join-Path $OutputDirectory "run_status.json"
$Succeeded = $false
$FailureText = ""

try {
    Copy-Item -LiteralPath $JavaSource -Destination $StagedJava
    Copy-Item -LiteralPath $ConfigPath -Destination $StagedConfig
    Copy-Item -LiteralPath $Evaluator -Destination $StagedEvaluator
    Copy-Item -LiteralPath $Runner -Destination $StagedRunner

    $StagedHashes = [ordered]@{}
    foreach ($Pair in @(
        @($JavaSource, $StagedJava, "java"),
        @($ConfigPath, $StagedConfig, "config"),
        @($Evaluator, $StagedEvaluator, "evaluator"),
        @($Runner, $StagedRunner, "runner")
    )) {
        $SourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Pair[0]).Hash.ToLowerInvariant()
        $StagedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Pair[1]).Hash.ToLowerInvariant()
        if ($SourceHash -ne $StagedHash) {
            throw "Staged $($Pair[2]) does not match its source"
        }
        $StagedHashes[$Pair[2]] = $StagedHash
    }

    Push-Location $OutputDirectory
    try {
        & $Compiler $StagedJava
        if ($LASTEXITCODE -ne 0) {
            throw "COMSOL Java compilation failed with exit code $LASTEXITCODE"
        }
        $BatchLog = Join-Path $OutputDirectory "comsol_batch.log"
        $ProcessLog = Join-Path $OutputDirectory "comsol_process.log"
        & $Batch -inputfile $ClassFile -nosave -np 1 -batchlog $BatchLog *> $ProcessLog
        if ($LASTEXITCODE -ne 0) {
            throw "COMSOL critical-boundary run failed with exit code $LASTEXITCODE"
        }
        $ProcessText = Get-Content -LiteralPath $ProcessLog -Raw
        if ($ProcessText -notmatch [regex]::Escape("M3CCB|run_pass|")) {
            throw "COMSOL method did not emit its complete-run receipt"
        }
    } finally {
        Pop-Location
    }

    foreach ($Step in $Config.case.fixed_rk4_steps) {
        $History = Join-Path $OutputDirectory (Join-Path $Step.label "state_raw_wide.csv")
        if (-not (Test-Path -LiteralPath $History -PathType Leaf) -or
            (Get-Item -LiteralPath $History).Length -eq 0) {
            throw "COMSOL did not emit the expected boundary table: $History"
        }
    }

    $Provenance = [ordered]@{
        classification = $Config.classification
        evaluation_id = $Config.evaluation_id
        evaluation_revision = $Config.evaluation_revision
        model_source = $Config.model.source
        model_saved = $false
        staged_sources = $StagedHashes
        comsol_root = [System.IO.Path]::GetFullPath($ComsolRoot)
        comsol_version = $VersionText
        process_count = 1
        generated_utc = [DateTime]::UtcNow.ToString("o")
    }
    $Provenance | ConvertTo-Json -Depth 5 | Set-Content `
        -LiteralPath (Join-Path $OutputDirectory "provenance.json") -Encoding utf8

    Push-Location $SolverRoot
    try {
        & uv run --locked python $StagedEvaluator $OutputDirectory `
            --config $StagedConfig --evidence $EvidenceDirectory
        if ($LASTEXITCODE -ne 0) {
            throw "Critical-boundary comparison failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
    $Succeeded = $true
} catch {
    $FailureText = $_.Exception.Message
    throw
} finally {
    $CleanupFailures = @()
    foreach ($Temporary in @(
        $ClassFile,
        (Join-Path $OutputDirectory "RunM3CCriticalBoundaries.class.status")
    )) {
        if (Test-Path -LiteralPath $Temporary) {
            try {
                Remove-Item -LiteralPath $Temporary -Force -ErrorAction Stop
            } catch {
                $CleanupFailures += "$Temporary`: $($_.Exception.Message)"
            }
        }
    }
    $ClassRetained = Test-Path -LiteralPath $ClassFile
    if ($CleanupFailures.Count -gt 0 -or $ClassRetained) {
        $Succeeded = $false
        $CleanupText = "cleanup failed or retained compiled class: " + ($CleanupFailures -join "; ")
        $FailureText = $(if ($FailureText) { "$FailureText; $CleanupText" } else { $CleanupText })
    }
    if ($Succeeded) {
        try {
            $ArtifactHashes = Get-ChildItem -LiteralPath $OutputDirectory -Recurse -File |
                Where-Object { $_.Name -notin @("artifact_hashes.csv", "run_status.json") } |
                Sort-Object FullName |
                ForEach-Object {
                    [ordered]@{
                        path = [System.IO.Path]::GetRelativePath(
                            $OutputDirectory, $_.FullName
                        ).Replace("\", "/")
                        sha256 = (
                            Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName
                        ).Hash.ToLowerInvariant()
                        bytes = $_.Length
                    }
                }
            $ArtifactHashes | Export-Csv `
                -LiteralPath (Join-Path $OutputDirectory "artifact_hashes.csv") `
                -NoTypeInformation -Encoding utf8
        } catch {
            $Succeeded = $false
            $HashText = "artifact hashing failed: $($_.Exception.Message)"
            $FailureText = $(if ($FailureText) { "$FailureText; $HashText" } else { $HashText })
        }
    }
    $Status = [ordered]@{
        status = $(if ($Succeeded) { "COMPLETE" } else { "INCOMPLETE" })
        failure = $FailureText
        compiled_class_retained = $ClassRetained
        evidence_directory = $EvidenceDirectory
        generated_utc = [DateTime]::UtcNow.ToString("o")
    }
    $Status | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $StatusFile -Encoding utf8
}

if (-not $Succeeded) {
    throw "M3-C critical-boundary comparison did not complete"
}

Write-Output "M3-C critical-boundary comparison completed: $EvidenceDirectory"
