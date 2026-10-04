param(
    [string]$ComsolRoot = "C:\Program Files\COMSOL\COMSOL64\Multiphysics_copy1",
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$SolverRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$RepositoryRoot = (Resolve-Path (Join-Path $SolverRoot "..\..")).Path
$ConfigPath = Join-Path $PSScriptRoot "cases\m3c1_caseA_100nm_thermophoresis_ppr_v1.json"
$JavaSource = Join-Path $PSScriptRoot "comsol\RunM3C1CaseA100ThermophoresisPpr.java"
$Compiler = Join-Path $ComsolRoot "bin\win64\comsolcompile.exe"
$Batch = Join-Path $ComsolRoot "bin\win64\comsolbatch.exe"

foreach ($Required in @($ConfigPath, $JavaSource, $Compiler, $Batch)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required M3-C1 PPR export input does not exist: $Required"
    }
}

$Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$SourceModel = Join-Path $RepositoryRoot $Config.source_model.relative_path
if (-not (Test-Path -LiteralPath $SourceModel -PathType Leaf)) {
    throw "Audited source MPH does not exist: $SourceModel"
}
$PreservedRoot = Join-Path $SolverRoot $Config.preserved_reference.root
$PreservedStep = Join-Path $PreservedRoot $Config.preserved_reference.step_directory
$PreservedState = Join-Path $PreservedStep "state_raw_wide.csv"
$PreservedForce = Join-Path $PreservedStep "force_raw_wide.csv"
foreach ($Preserved in @($PreservedState, $PreservedForce)) {
    if (-not (Test-Path -LiteralPath $Preserved -PathType Leaf)) {
        throw "Locked M3-C0b v6 evidence does not exist: $Preserved"
    }
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path `
        $SolverRoot "_out_m3c1\caseA_100nm_thermophoresis_ppr_v1"
}
$OutputFullPath = [System.IO.Path]::GetFullPath($OutputDirectory)
$PreservedFullPath = [System.IO.Path]::GetFullPath($PreservedRoot)
$PreservedPrefix = $PreservedFullPath.TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
) + [System.IO.Path]::DirectorySeparatorChar
if ($OutputFullPath.Equals(
        $PreservedFullPath,
        [System.StringComparison]::OrdinalIgnoreCase
    ) -or $OutputFullPath.StartsWith(
        $PreservedPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
    throw "M3-C1 PPR output must remain outside the locked M3-C0b v6 evidence tree"
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "M3-C1 PPR output already exists: $OutputDirectory"
}

$SourceHashBefore = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel
).Hash.ToLowerInvariant()
if ($SourceHashBefore -ne $Config.source_model.sha256) {
    throw "Source MPH hash does not match the locked M3-C1 PPR configuration"
}
$V6StateHashBefore = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $PreservedState
).Hash.ToLowerInvariant()
$V6ForceHashBefore = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $PreservedForce
).Hash.ToLowerInvariant()
if ($V6StateHashBefore -ne $Config.preserved_reference.state_raw_sha256 -or
    $V6ForceHashBefore -ne $Config.preserved_reference.force_raw_sha256) {
    throw "M3-C0b v6 evidence hashes do not match the locked M3-C1 PPR configuration"
}
$VersionText = (& $Batch -version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read COMSOL version; exit code $LASTEXITCODE"
}
if ($VersionText -notmatch [regex]::Escape($Config.expected_comsol_version)) {
    throw "COMSOL version does not match the locked configuration: $VersionText"
}

New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$StepDirectory = Join-Path $OutputDirectory $Config.case.step_directory
New-Item -ItemType Directory -Path $StepDirectory | Out-Null

$SourceCopy = Join-Path $OutputDirectory "source_copy.mph"
$StagedJava = Join-Path $OutputDirectory "RunM3C1CaseA100ThermophoresisPpr.java"
$StagedConfig = Join-Path $OutputDirectory "m3c1_caseA_100nm_thermophoresis_ppr_v1.json"
$ClassFile = Join-Path $OutputDirectory "RunM3C1CaseA100ThermophoresisPpr.class"
$StatusFile = Join-Path $OutputDirectory "run_status.json"
$Succeeded = $false
$FailureText = ""

try {
    Copy-Item -LiteralPath $SourceModel -Destination $SourceCopy
    $CopyHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $SourceCopy
    ).Hash.ToLowerInvariant()
    if ($CopyHash -ne $SourceHashBefore) {
        throw "The isolated source copy hash does not match the audited source MPH"
    }
    Copy-Item -LiteralPath $JavaSource -Destination $StagedJava
    Copy-Item -LiteralPath $ConfigPath -Destination $StagedConfig
    $JavaHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $StagedJava
    ).Hash.ToLowerInvariant()
    $ConfigHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $StagedConfig
    ).Hash.ToLowerInvariant()
    $SourceJavaHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $JavaSource
    ).Hash.ToLowerInvariant()
    $SourceConfigHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $ConfigPath
    ).Hash.ToLowerInvariant()
    if ($JavaHash -ne $SourceJavaHash -or $ConfigHash -ne $SourceConfigHash) {
        throw "A staged M3-C1 PPR input does not match its source"
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
            throw "COMSOL M3-C1 PPR export failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }

    foreach ($Table in @(
        "thermophoresis_ppr_particle_raw_wide.csv",
        "thermophoresis_ppr_native_mesh_nodes.csv"
    )) {
        $Export = Join-Path $StepDirectory $Table
        if (-not (Test-Path -LiteralPath $Export -PathType Leaf) -or
            (Get-Item -LiteralPath $Export).Length -eq 0) {
            throw "COMSOL did not emit the expected nonempty PPR table: $Export"
        }
    }
    $ProcessText = Get-Content -LiteralPath $ProcessLog -Raw
    if ($ProcessText -notmatch [regex]::Escape("M3C1PPR|feature_authority") -or
        $ProcessText -notmatch [regex]::Escape("UsePPR=true") -or
        $ProcessText -notmatch [regex]::Escape("ppr_gradient_r=ppr(d(root.comp1.AS_Tg,r))") -or
        $ProcessText -notmatch [regex]::Escape("steps_run=1")) {
        throw "COMSOL process log is missing an exact M3-C1 PPR configuration receipt"
    }

    $SourceHashAfter = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel
    ).Hash.ToLowerInvariant()
    $V6StateHashAfter = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $PreservedState
    ).Hash.ToLowerInvariant()
    $V6ForceHashAfter = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $PreservedForce
    ).Hash.ToLowerInvariant()
    if ($SourceHashAfter -ne $SourceHashBefore) {
        throw "The audited source MPH changed during the loadCopy/no-save run"
    }
    if ($V6StateHashAfter -ne $V6StateHashBefore -or
        $V6ForceHashAfter -ne $V6ForceHashBefore) {
        throw "Locked M3-C0b v6 evidence changed during the M3-C1 PPR run"
    }

    $Provenance = [ordered]@{
        classification = $Config.classification
        evaluation_id = $Config.evaluation_id
        evaluation_revision = $Config.evaluation_revision
        config = [System.IO.Path]::GetFullPath($ConfigPath)
        config_sha256 = $ConfigHash
        staged_config = [System.IO.Path]::GetFileName($StagedConfig)
        source_model = [System.IO.Path]::GetFullPath($SourceModel)
        source_sha256_before = $SourceHashBefore
        source_sha256_after = $SourceHashAfter
        source_unchanged = $true
        source_load_mode = $Config.source_model.load_mode
        model_saved = $false
        isolated_source_copy_sha256 = $CopyHash
        isolated_source_copy_retained = $false
        preserved_m3c0b_v6_root = [System.IO.Path]::GetFullPath($PreservedRoot)
        preserved_m3c0b_v6_state_sha256_before = $V6StateHashBefore
        preserved_m3c0b_v6_state_sha256_after = $V6StateHashAfter
        preserved_m3c0b_v6_force_sha256_before = $V6ForceHashBefore
        preserved_m3c0b_v6_force_sha256_after = $V6ForceHashAfter
        preserved_m3c0b_v6_unchanged = $true
        java_source = [System.IO.Path]::GetFullPath($JavaSource)
        java_sha256 = $JavaHash
        staged_java = [System.IO.Path]::GetFileName($StagedJava)
        comsol_root = [System.IO.Path]::GetFullPath($ComsolRoot)
        comsol_version = $VersionText
        process_count = 1
        steps_run = 1
        export_status = "COMPLETE"
        thermophoresis_closure_status = "NOT_EVALUATED"
        closure_evaluator_required = $true
        generated_utc = [DateTime]::UtcNow.ToString("o")
    }
    $Provenance | ConvertTo-Json -Depth 5 | Set-Content `
        -LiteralPath (Join-Path $OutputDirectory "provenance.json") -Encoding utf8
    $Succeeded = $true
} catch {
    $FailureText = $_.Exception.Message
    throw
} finally {
    $CleanupFailures = @()
    foreach ($Temporary in @(
        $SourceCopy,
        $ClassFile,
        (Join-Path $OutputDirectory "RunM3C1CaseA100ThermophoresisPpr.class.status")
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
        $CleanupText = "cleanup failed or retained transient files: " + (
            $CleanupFailures -join "; "
        )
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
        status_scope = "raw PPR export only"
        thermophoresis_closure_status = "NOT_EVALUATED"
        failure = $FailureText
        source_copy_retained = $SourceRetained
        compiled_class_retained = $ClassRetained
        generated_utc = [DateTime]::UtcNow.ToString("o")
    }
    $Status | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $StatusFile -Encoding utf8
}

if (-not $Succeeded) {
    throw "M3-C1 thermophoresis PPR export did not complete"
}

Write-Output "M3-C1 thermophoresis PPR export completed: $OutputDirectory"
