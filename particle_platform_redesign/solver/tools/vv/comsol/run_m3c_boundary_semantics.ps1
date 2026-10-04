param(
    [string]$ComsolRoot = "C:\Program Files\COMSOL\COMSOL64\Multiphysics_copy1",
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$SolverRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$RepositoryRoot = (Resolve-Path (Join-Path $SolverRoot "..\..")).Path
$ConfigPath = Join-Path $PSScriptRoot "cases\m3c_boundary_semantics_v2.json"
$JavaSource = Join-Path $PSScriptRoot "comsol\RunM3CBoundarySemantics.java"
$Normalizer = Join-Path $PSScriptRoot "normalize_m3c_boundary_semantics.py"
$Runner = $PSCommandPath
$Compiler = Join-Path $ComsolRoot "bin\win64\comsolcompile.exe"
$Batch = Join-Path $ComsolRoot "bin\win64\comsolbatch.exe"

foreach ($Required in @($ConfigPath, $JavaSource, $Normalizer, $Runner, $Compiler, $Batch)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required M3-C boundary-semantics input does not exist: $Required"
    }
}

$Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$SourceModel = Join-Path $RepositoryRoot $Config.source_model.relative_path
if (-not (Test-Path -LiteralPath $SourceModel -PathType Leaf)) {
    throw "Audited source MPH does not exist: $SourceModel"
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $SolverRoot "_out_m3c\boundary_semantics_v2"
}
$OutputDirectory = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath(
    $OutputDirectory
)
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "M3-C boundary-semantics output already exists: $OutputDirectory"
}

$SourceHashBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel).Hash.ToLowerInvariant()
if ($SourceHashBefore -ne $Config.source_model.sha256) {
    throw "Source MPH hash does not match the locked boundary-semantics configuration"
}
$VersionText = (& $Batch -version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read COMSOL version; exit code $LASTEXITCODE"
}
if ($VersionText -notmatch [regex]::Escape($Config.expected_comsol_version)) {
    throw "COMSOL version does not match the locked configuration: $VersionText"
}

New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
foreach ($Scenario in $Config.scenarios) {
    foreach ($Step in $Config.case.fixed_rk4_steps) {
        New-Item -ItemType Directory `
            -Path (Join-Path $OutputDirectory (Join-Path $Scenario.id $Step.label)) `
            -Force | Out-Null
    }
}

$SourceCopy = Join-Path $OutputDirectory "source_copy.mph"
$StagedJava = Join-Path $OutputDirectory "RunM3CBoundarySemantics.java"
$StagedConfig = Join-Path $OutputDirectory "m3c_boundary_semantics_v2.json"
$StagedNormalizer = Join-Path $OutputDirectory "normalize_m3c_boundary_semantics.py"
$StagedRunner = Join-Path $OutputDirectory "run_m3c_boundary_semantics.ps1"
$ClassFile = Join-Path $OutputDirectory "RunM3CBoundarySemantics.class"
$StatusFile = Join-Path $OutputDirectory "run_status.json"
$Succeeded = $false
$FailureText = ""

try {
    Copy-Item -LiteralPath $SourceModel -Destination $SourceCopy
    $CopyHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceCopy).Hash.ToLowerInvariant()
    if ($CopyHash -ne $SourceHashBefore) {
        throw "The isolated source copy hash does not match the audited source MPH"
    }
    Copy-Item -LiteralPath $JavaSource -Destination $StagedJava
    Copy-Item -LiteralPath $ConfigPath -Destination $StagedConfig
    Copy-Item -LiteralPath $Normalizer -Destination $StagedNormalizer
    Copy-Item -LiteralPath $Runner -Destination $StagedRunner

    $StagedHashes = [ordered]@{}
    foreach ($Pair in @(
        @($JavaSource, $StagedJava, "java"),
        @($ConfigPath, $StagedConfig, "config"),
        @($Normalizer, $StagedNormalizer, "normalizer"),
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
            throw "COMSOL boundary-semantics run failed with exit code $LASTEXITCODE"
        }
        $ProcessText = Get-Content -LiteralPath $ProcessLog -Raw
        if ($ProcessText -notmatch [regex]::Escape("M3CB|run_pass|")) {
            throw "COMSOL method did not emit its complete-run receipt"
        }
    } finally {
        Pop-Location
    }

    foreach ($Scenario in $Config.scenarios) {
        foreach ($Step in $Config.case.fixed_rk4_steps) {
            $History = Join-Path $OutputDirectory `
                (Join-Path $Scenario.id (Join-Path $Step.label "state_raw_wide.csv"))
            if (-not (Test-Path -LiteralPath $History -PathType Leaf) -or
                (Get-Item -LiteralPath $History).Length -eq 0) {
                throw "COMSOL did not emit the expected boundary table: $History"
            }
        }
    }

    $SourceHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel).Hash.ToLowerInvariant()
    if ($SourceHashAfter -ne $SourceHashBefore) {
        throw "The audited source MPH changed during the loadCopy/no-save run"
    }

    $Provenance = [ordered]@{
        classification = $Config.classification
        evaluation_id = $Config.evaluation_id
        evaluation_revision = $Config.evaluation_revision
        source_model = [System.IO.Path]::GetFullPath($SourceModel)
        source_sha256_before = $SourceHashBefore
        source_sha256_after = $SourceHashAfter
        source_unchanged = $true
        source_load_mode = $Config.source_model.load_mode
        model_saved = $false
        isolated_source_copy_sha256 = $CopyHash
        isolated_source_copy_retained = $false
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
        & uv run --locked python $StagedNormalizer $OutputDirectory --config $StagedConfig
        if ($LASTEXITCODE -ne 0) {
            throw "Boundary-semantics normalization failed with exit code $LASTEXITCODE"
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
        $SourceCopy,
        $ClassFile,
        (Join-Path $OutputDirectory "RunM3CBoundarySemantics.class.status")
    )) {
        if (Test-Path -LiteralPath $Temporary) {
            try {
                Remove-Item -LiteralPath $Temporary -Force -ErrorAction Stop
            } catch {
                $CleanupFailures += "$Temporary`: $($_.Exception.Message)"
            }
        }
    }
    $SourceRetained = Test-Path -LiteralPath $SourceCopy
    $ClassRetained = Test-Path -LiteralPath $ClassFile
    if ($CleanupFailures.Count -gt 0 -or $SourceRetained -or $ClassRetained) {
        $Succeeded = $false
        $CleanupText = "cleanup failed or retained transient files: " + ($CleanupFailures -join "; ")
        $FailureText = $(if ($FailureText) { "$FailureText; $CleanupText" } else { $CleanupText })
    }
    if ($Succeeded) {
        try {
            $OutputRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
            $ArtifactHashes = Get-ChildItem -LiteralPath $OutputRoot -Recurse -File |
                Where-Object { $_.Name -notin @("artifact_hashes.csv", "run_status.json") } |
                Sort-Object FullName |
                ForEach-Object {
                    [ordered]@{
                        path = [System.IO.Path]::GetRelativePath(
                            $OutputRoot, $_.FullName
                        ).Replace("\", "/")
                        sha256 = (
                            Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName
                        ).Hash.ToLowerInvariant()
                        bytes = $_.Length
                    }
                }
            $ArtifactHashes | Export-Csv `
                -LiteralPath (Join-Path $OutputRoot "artifact_hashes.csv") `
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
        source_copy_retained = $SourceRetained
        compiled_class_retained = $ClassRetained
        generated_utc = [DateTime]::UtcNow.ToString("o")
    }
    $Status | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $StatusFile -Encoding utf8
}

if (-not $Succeeded) {
    throw "M3-C boundary-semantics probe did not complete"
}

Write-Output "M3-C boundary-semantics probe completed: $OutputDirectory"
