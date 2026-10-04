param(
    [string]$ComsolRoot = "C:\Program Files\COMSOL\COMSOL64\Multiphysics_copy1",
    [string]$CandidateInput = "",
    [string]$OutputDirectory = "",
    [ValidateSet("pre_event", "material_event")]
    [string]$RunProfile = "pre_event"
)

$ErrorActionPreference = "Stop"

function Confirm-PreparedTableArtifacts {
    param(
        [string]$ReceiptPath,
        [string]$ArtifactRoot
    )

    $Receipt = Get-Content -LiteralPath $ReceiptPath -Raw | ConvertFrom-Json
    if ([int]$Receipt.schema_version -ne 1 -or
        [string]$Receipt.tool_revision -ne "m3c1_full_physics_common_p1_tables_v1") {
        throw "Prepared-table receipt schema or tool revision is not supported"
    }
    $Entries = @($Receipt.artifacts.PSObject.Properties)
    $ExpectedCount = [int]$Receipt.component_count + @($Receipt.release.functions).Count + 1
    if ($Entries.Count -ne $ExpectedCount) {
        throw "Prepared-table receipt artifact count does not cover every field, release function, and probe"
    }

    $Verified = @()
    [long]$TotalSize = 0
    foreach ($Entry in $Entries) {
        $Name = [string]$Entry.Name
        if ([string]::IsNullOrWhiteSpace($Name) -or $Name -in @(".", "..") -or
            [System.IO.Path]::IsPathRooted($Name) -or
            [System.IO.Path]::GetFileName($Name) -ne $Name) {
            throw "Prepared-table receipt contains an unsafe artifact name: $Name"
        }
        $Path = Join-Path $ArtifactRoot $Name
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "Prepared-table artifact is missing: $Name"
        }
        $ActualSize = (Get-Item -LiteralPath $Path).Length
        $ActualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
        if ($ActualSize -ne [long]$Entry.Value.size_bytes -or
            $ActualHash -ne [string]$Entry.Value.sha256) {
            throw "Prepared-table artifact differs from its receipt: $Name"
        }
        $TotalSize += $ActualSize
        $Verified += [ordered]@{
            path = $Name
            sha256 = $ActualHash
            size_bytes = $ActualSize
        }
    }
    return [ordered]@{
        status = "PASS"
        receipt_sha256 = (
            Get-FileHash -Algorithm SHA256 -LiteralPath $ReceiptPath
        ).Hash.ToLowerInvariant()
        artifact_count = $Entries.Count
        total_size_bytes = $TotalSize
        artifacts = $Verified
    }
}

$SolverRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$RepositoryRoot = (Resolve-Path (Join-Path $SolverRoot "..\..")).Path
$IsMaterialEvent = $RunProfile -eq "material_event"
$ConfigPath = Join-Path $PSScriptRoot $(
    if ($IsMaterialEvent) {
        "cases\m3c1_common_p1_material_event_v1.json"
    } else {
        "cases\m3c1_common_p1_reference_run_v1.json"
    }
)
$JavaSource = Join-Path $PSScriptRoot "comsol\RunM3C1CaseA100CommonP1.java"
$MaterialEventJavaSource = Join-Path `
    $PSScriptRoot "comsol\RunM3C1CaseA100CommonP1MaterialEvent.java"
$Preparer = Join-Path $PSScriptRoot "prepare_m3c1_common_p1_tables.py"
$Postprocessor = Join-Path $PSScriptRoot $(
    if ($IsMaterialEvent) {
        "validate_m3c1_common_p1_material_event_run.py"
    } else {
        "normalize_m3c1_common_p1_reference.py"
    }
)
$StepNames = $(
    if ($IsMaterialEvent) {
        @("dt_0p15625us")
    } else {
        @("dt_0p625us", "dt_0p3125us", "dt_0p15625us")
    }
)
$Compiler = Join-Path $ComsolRoot "bin\win64\comsolcompile.exe"
$Batch = Join-Path $ComsolRoot "bin\win64\comsolbatch.exe"

$RequiredInputs = @($ConfigPath, $JavaSource, $Preparer, $Postprocessor, $Compiler, $Batch)
if ($IsMaterialEvent) {
    $RequiredInputs += $MaterialEventJavaSource
}
foreach ($Required in $RequiredInputs) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required M3-C1 common-P1 input does not exist: $Required"
    }
}

$Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$SourceModel = Join-Path $RepositoryRoot $Config.source_model.relative_path
if ([string]::IsNullOrWhiteSpace($CandidateInput)) {
    $CandidateInput = $(
        if ($IsMaterialEvent) {
            Join-Path (Join-Path $SolverRoot $Config.candidate.parent_root_relative_path) `
                $Config.candidate.input_filename
        } else {
            Join-Path $SolverRoot $Config.candidate_input.relative_path
        }
    )
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $SolverRoot $(
        if ($IsMaterialEvent) {
            "_out_m3c1\caseA_100nm_common_p1_material_event_v1"
        } else {
            "_out_m3c1_common_p1\caseA_100nm_v1"
        }
    )
}
if (-not [System.IO.Path]::IsPathRooted($CandidateInput)) {
    $CandidateInput = Join-Path $SolverRoot $CandidateInput
}
if (-not [System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $SolverRoot $OutputDirectory
}
$CandidateInput = [System.IO.Path]::GetFullPath($CandidateInput)
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)

foreach ($Required in @($SourceModel, $CandidateInput)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Locked M3-C1 common-P1 input does not exist: $Required"
    }
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "M3-C1 common-P1 output already exists: $OutputDirectory"
}

$SourceHashBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel).Hash.ToLowerInvariant()
$CandidateHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $CandidateInput).Hash.ToLowerInvariant()
$ExpectedCandidateHash = $(
    if ($IsMaterialEvent) { $Config.candidate.input_sha256 } else { $Config.candidate_input.sha256 }
)
$CandidateContentHash = $(
    if ($IsMaterialEvent) {
        $Config.candidate.input_content_hash
    } else {
        $Config.candidate_input.content_hash
    }
)
if ($SourceHashBefore -ne $Config.source_model.sha256) {
    throw "Source MPH hash does not match the locked common-P1 run configuration"
}
if ($CandidateHash -ne $ExpectedCandidateHash) {
    throw "Candidate HDF5 hash does not match the locked common-P1 run configuration"
}
$VersionText = (& $Batch -version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read COMSOL version; exit code $LASTEXITCODE"
}
if ($VersionText -notmatch [regex]::Escape($Config.expected_comsol_version)) {
    throw "COMSOL version does not match the locked configuration: $VersionText"
}

$ConfigSourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ConfigPath).Hash.ToLowerInvariant()
$JavaSourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $JavaSource).Hash.ToLowerInvariant()
$MaterialEventJavaSourceHash = $(
    if ($IsMaterialEvent) {
        (Get-FileHash -Algorithm SHA256 -LiteralPath $MaterialEventJavaSource).Hash.ToLowerInvariant()
    } else {
        ""
    }
)
$PreparerSourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Preparer).Hash.ToLowerInvariant()
$PostprocessorSourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Postprocessor).Hash.ToLowerInvariant()
$RunnerHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath).Hash.ToLowerInvariant()

Push-Location $SolverRoot
try {
    & uv run --locked python $Preparer $CandidateInput $OutputDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "M3-C1 common-P1 table preparation failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

foreach ($Name in $StepNames) {
    New-Item -ItemType Directory -Path (Join-Path $OutputDirectory $Name) | Out-Null
}

$SourceCopy = Join-Path $OutputDirectory "source_copy.mph"
$StagedJava = Join-Path $OutputDirectory "RunM3C1CaseA100CommonP1.java"
$StagedMaterialEventJava = Join-Path `
    $OutputDirectory "RunM3C1CaseA100CommonP1MaterialEvent.java"
$StagedConfig = Join-Path $OutputDirectory ([System.IO.Path]::GetFileName($ConfigPath))
$StagedPreparer = Join-Path $OutputDirectory "prepare_m3c1_common_p1_tables.py"
$StagedPostprocessor = Join-Path $OutputDirectory ([System.IO.Path]::GetFileName($Postprocessor))
$SharedClassFile = Join-Path $OutputDirectory "RunM3C1CaseA100CommonP1.class"
$MaterialEventClassFile = Join-Path `
    $OutputDirectory "RunM3C1CaseA100CommonP1MaterialEvent.class"
$ClassFile = $(if ($IsMaterialEvent) { $MaterialEventClassFile } else { $SharedClassFile })
$ClassStatusFile = "${ClassFile}.status"
$CompiledClassFiles = @($SharedClassFile)
if ($IsMaterialEvent) {
    $CompiledClassFiles += $MaterialEventClassFile
}
$ClassStatusFiles = @($CompiledClassFiles | ForEach-Object { "${_}.status" })
$StatusFile = Join-Path $OutputDirectory "run_status.json"
$TableReceipt = Join-Path $OutputDirectory "common_p1_table_receipt.json"
$TableValidationPath = Join-Path $OutputDirectory "prepared_table_validation.json"
$RunReceiptFilename = $(
    if ($IsMaterialEvent) {
        "common_p1_material_event_run_receipt.json"
    } else {
        "common_p1_run_receipt.json"
    }
)
$Succeeded = $false
$FailureText = ""
$CopyHash = ""
$SourceHashAfter = ""
$TableReceiptHash = ""
$ArtifactHashesHash = ""

try {
    foreach ($Required in @($TableReceipt, (Join-Path $OutputDirectory "common_p1_release_probes.csv"))) {
        if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
            throw "P1 preparer did not emit required artifact: $Required"
        }
    }
    $TableReceiptHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $TableReceipt).Hash.ToLowerInvariant()
    $TableValidationBefore = Confirm-PreparedTableArtifacts $TableReceipt $OutputDirectory

    Copy-Item -LiteralPath $SourceModel -Destination $SourceCopy
    $CopyHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceCopy).Hash.ToLowerInvariant()
    if ($CopyHash -ne $SourceHashBefore) {
        throw "The isolated source copy hash does not match the audited source MPH"
    }
    Copy-Item -LiteralPath $JavaSource -Destination $StagedJava
    if ($IsMaterialEvent) {
        Copy-Item -LiteralPath $MaterialEventJavaSource -Destination $StagedMaterialEventJava
    }
    Copy-Item -LiteralPath $ConfigPath -Destination $StagedConfig
    Copy-Item -LiteralPath $Preparer -Destination $StagedPreparer
    Copy-Item -LiteralPath $Postprocessor -Destination $StagedPostprocessor

    $StagedLocks = [ordered]@{
        java = @($StagedJava, $JavaSourceHash)
        config = @($StagedConfig, $ConfigSourceHash)
        preparer = @($StagedPreparer, $PreparerSourceHash)
        postprocessor = @($StagedPostprocessor, $PostprocessorSourceHash)
    }
    if ($IsMaterialEvent) {
        $StagedLocks["material_event_entry_java"] = @(
            $StagedMaterialEventJava,
            $MaterialEventJavaSourceHash
        )
    }
    foreach ($Name in $StagedLocks.Keys) {
        $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $StagedLocks[$Name][0]).Hash.ToLowerInvariant()
        if ($Actual -ne $StagedLocks[$Name][1]) {
            throw "Staged $Name does not match its locked source"
        }
    }
    Push-Location $OutputDirectory
    try {
        & $Compiler $StagedJava
        if ($LASTEXITCODE -ne 0) {
            throw "COMSOL shared Java compilation failed with exit code $LASTEXITCODE"
        }
        if (-not (Test-Path -LiteralPath $SharedClassFile -PathType Leaf)) {
            throw "COMSOL Java compiler did not emit: $SharedClassFile"
        }
        if ($IsMaterialEvent) {
            & $Compiler -classpathadd $OutputDirectory $StagedMaterialEventJava
            if ($LASTEXITCODE -ne 0) {
                throw "COMSOL material-event Java compilation failed with exit code $LASTEXITCODE"
            }
        }
        $MissingClasses = @(
            $CompiledClassFiles | Where-Object {
                -not (Test-Path -LiteralPath $_ -PathType Leaf)
            }
        )
        if ($MissingClasses.Count -gt 0) {
            throw "COMSOL Java compiler did not emit: $($MissingClasses -join ', ')"
        }
        $BatchLog = Join-Path $OutputDirectory "comsol_batch.log"
        $ProcessLog = Join-Path $OutputDirectory "comsol_process.log"
        & $Batch -inputfile $ClassFile -nosave -np 1 -batchlog $BatchLog *> $ProcessLog
        if ($LASTEXITCODE -ne 0) {
            throw "COMSOL M3-C1 common-P1 run failed with exit code $LASTEXITCODE"
        }
        $ClassStatusText = $(
            if (Test-Path -LiteralPath $ClassStatusFile -PathType Leaf) {
                (Get-Content -LiteralPath $ClassStatusFile -Raw).Trim()
            } else {
                "not emitted"
            }
        )
        if (-not (Test-Path -LiteralPath $ProcessLog -PathType Leaf) -or
            (Get-Item -LiteralPath $ProcessLog).Length -eq 0) {
            throw "COMSOL returned without process output; class status: $ClassStatusText"
        }
        if (Select-String -LiteralPath $ProcessLog -SimpleMatch "M3C1_COMMON_P1|fatal|") {
            throw "COMSOL Java program reported a fatal exception; inspect $ProcessLog"
        }
    } finally {
        Pop-Location
    }

    foreach ($Name in $StepNames) {
        foreach ($Table in @("state_raw_wide.csv", "force_raw_wide.csv", "primitive_raw_wide.csv")) {
            $History = Join-Path (Join-Path $OutputDirectory $Name) $Table
            if (-not (Test-Path -LiteralPath $History -PathType Leaf) -or
                (Get-Item -LiteralPath $History).Length -eq 0) {
                throw "COMSOL did not emit expected nonempty common-P1 table: $History"
            }
        }
    }
    $SourceHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel).Hash.ToLowerInvariant()
    if ($SourceHashAfter -ne $SourceHashBefore) {
        throw "The audited source MPH changed during the loadCopy/no-save run"
    }
    $TableValidationAfter = Confirm-PreparedTableArtifacts $TableReceipt $OutputDirectory
    if ($TableValidationBefore.receipt_sha256 -ne $TableValidationAfter.receipt_sha256 -or
        $TableValidationBefore.artifact_count -ne $TableValidationAfter.artifact_count -or
        $TableValidationBefore.total_size_bytes -ne $TableValidationAfter.total_size_bytes) {
        throw "Prepared-table validation changed across the COMSOL run"
    }
    $TableValidation = [ordered]@{
        schema_version = 1
        status = "PASS"
        criterion = "every common_p1_table_receipt artifact matches receipt SHA-256 and byte size immediately before and after COMSOL"
        receipt_sha256 = $TableValidationAfter.receipt_sha256
        pre_comsol = [ordered]@{
            status = $TableValidationBefore.status
            artifact_count = $TableValidationBefore.artifact_count
            total_size_bytes = $TableValidationBefore.total_size_bytes
        }
        post_comsol = [ordered]@{
            status = $TableValidationAfter.status
            artifact_count = $TableValidationAfter.artifact_count
            total_size_bytes = $TableValidationAfter.total_size_bytes
        }
        artifacts = $TableValidationAfter.artifacts
    }
    $TableValidation | ConvertTo-Json -Depth 6 | Set-Content `
        -LiteralPath $TableValidationPath -Encoding utf8

    $Provenance = [ordered]@{
        classification = $Config.classification
        evaluation_id = $Config.evaluation_id
        evaluation_revision = $Config.evaluation_revision
        config = [System.IO.Path]::GetFullPath($ConfigPath)
        config_sha256 = $ConfigSourceHash
        staged_config = [System.IO.Path]::GetFileName($StagedConfig)
        source_model = [System.IO.Path]::GetFullPath($SourceModel)
        source_sha256_before = $SourceHashBefore
        source_sha256_after = $SourceHashAfter
        source_unchanged = $true
        source_load_mode = $Config.source_model.load_mode
        model_saved = $false
        isolated_source_copy_sha256 = $CopyHash
        isolated_source_copy_retained = $false
        candidate_input = $CandidateInput
        candidate_input_sha256 = $CandidateHash
        candidate_content_hash = $CandidateContentHash
        table_receipt_sha256 = $TableReceiptHash
        prepared_table_validation = [System.IO.Path]::GetFileName($TableValidationPath)
        prepared_table_validation_status = $TableValidation.status
        runner_sha256 = $RunnerHash
        java_source_sha256 = $JavaSourceHash
        preparer_source_sha256 = $PreparerSourceHash
        postprocessor_source_sha256 = $PostprocessorSourceHash
        staged_java = [System.IO.Path]::GetFileName($StagedJava)
        staged_preparer = [System.IO.Path]::GetFileName($StagedPreparer)
        staged_postprocessor = [System.IO.Path]::GetFileName($StagedPostprocessor)
        run_profile = $RunProfile
        material_event_entry_java = $(
            if ($IsMaterialEvent) {
                [System.IO.Path]::GetFileName($StagedMaterialEventJava)
            } else {
                $null
            }
        )
        material_event_entry_java_sha256 = $(
            if ($IsMaterialEvent) { $MaterialEventJavaSourceHash } else { $null }
        )
        comsol_root = [System.IO.Path]::GetFullPath($ComsolRoot)
        comsol_version = $VersionText
        process_count = 1
        generated_utc = [DateTime]::UtcNow.ToString("o")
    }
    $Provenance | ConvertTo-Json -Depth 6 | Set-Content `
        -LiteralPath (Join-Path $OutputDirectory "provenance.json") -Encoding utf8

    Push-Location $SolverRoot
    try {
        & uv run --locked python $StagedPostprocessor $OutputDirectory --config $StagedConfig
        if ($LASTEXITCODE -ne 0) {
            throw "M3-C1 common-P1 postprocessing failed with exit code $LASTEXITCODE"
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
    $CleanupTargets = @($SourceCopy) + $CompiledClassFiles
    if ($Succeeded) {
        $CleanupTargets += $ClassStatusFiles
    }
    foreach ($Temporary in $CleanupTargets) {
        if (Test-Path -LiteralPath $Temporary) {
            try {
                Remove-Item -LiteralPath $Temporary -Force -ErrorAction Stop
            } catch {
                $CleanupFailures += "$Temporary`: $($_.Exception.Message)"
            }
        }
    }
    $SourceRetained = Test-Path -LiteralPath $SourceCopy
    $ClassRetained = @(
        $CompiledClassFiles | Where-Object { Test-Path -LiteralPath $_ }
    ).Count -gt 0
    $ClassStatusRetained = @(
        $ClassStatusFiles | Where-Object { Test-Path -LiteralPath $_ }
    ).Count -gt 0
    if ($CleanupFailures.Count -gt 0 -or $SourceRetained -or $ClassRetained -or
        $ClassStatusRetained) {
        $Succeeded = $false
        $CleanupText = "cleanup failed or retained transient files: " + ($CleanupFailures -join "; ")
        $FailureText = $(if ($FailureText) { "$FailureText; $CleanupText" } else { $CleanupText })
    }

    if ($Succeeded) {
        try {
            $OutputRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
            $ArtifactHashesPath = Join-Path $OutputRoot "artifact_hashes.csv"
            $ArtifactHashes = Get-ChildItem -LiteralPath $OutputRoot -Recurse -File |
                Where-Object {
                    $_.Name -notin @(
                        "artifact_hashes.csv",
                        "run_status.json",
                        $RunReceiptFilename
                    )
                } |
                Sort-Object FullName |
                ForEach-Object {
                    $RootPrefix = $OutputRoot + [System.IO.Path]::DirectorySeparatorChar
                    if (-not $_.FullName.StartsWith(
                        $RootPrefix,
                        [System.StringComparison]::OrdinalIgnoreCase
                    )) {
                        throw "Artifact path escaped the output root: $($_.FullName)"
                    }
                    [pscustomobject][ordered]@{
                        path = $_.FullName.Substring($RootPrefix.Length).Replace("\", "/")
                        sha256 = (
                            Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName
                        ).Hash.ToLowerInvariant()
                        bytes = $_.Length
                    }
                }
            $ArtifactHashes | Export-Csv -LiteralPath $ArtifactHashesPath `
                -NoTypeInformation -Encoding utf8
            $ArtifactHashesHash = (
                Get-FileHash -Algorithm SHA256 -LiteralPath $ArtifactHashesPath
            ).Hash.ToLowerInvariant()

            if ($IsMaterialEvent) {
                $SummaryPath = Join-Path $OutputRoot "material_event_raw_validation.json"
                $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
                $RunReceipt = [ordered]@{
                    schema_version = 1
                    tool_revision = "m3c1_common_p1_material_event_comsol_v1"
                    classification = $Config.classification
                    status = "COMPLETE"
                    config_sha256 = $ConfigSourceHash
                    candidate_input_sha256 = $CandidateHash
                    table_receipt_sha256 = $TableReceiptHash
                    source_sha256_before = $SourceHashBefore
                    source_sha256_after = $SourceHashAfter
                    comsol_version = $VersionText
                    scope = [ordered]@{
                        particles = [int]$Config.case.particle_count
                        frames = @($Config.case.output_times_s).Count
                        time_window_s = @(0.0, [double]$Config.case.event_window_end_s)
                        fixed_rk4_step_s = [double]$Config.case.fixed_rk4_step_s
                    }
                    configuration_receipt = $Summary.configuration_receipt
                    raw_tables = $Summary.raw_tables
                    raw_validation_sha256 = (
                        Get-FileHash -Algorithm SHA256 -LiteralPath $SummaryPath
                    ).Hash.ToLowerInvariant()
                    artifact_hashes_sha256 = $ArtifactHashesHash
                    prepared_table_validation = $TableValidation
                    claim_policy = $Summary.claim_policy
                }
            } else {
                $SummaryPath = Join-Path $OutputRoot "normalization_summary.json"
                $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
                $RunRecords = [ordered]@{}
                foreach ($Name in $StepNames) {
                    $Run = $Summary.runs.$Name
                    $RunRecords[$Name] = [ordered]@{
                        dt_s = [double]$Run.dt_s
                        trajectory_path = [string]$Run.trajectory_path
                        trajectory_sha256 = [string]$Run.trajectory_sha256
                        rows = [int]$Run.rows
                        all_active = [bool]$Run.all_active
                        event_count = [int]$Run.event_count
                        initial_primitive_validation = [ordered]@{
                            status = [string]$Run.initial_primitives.status
                            checked_particle_count = [int]$Run.initial_primitives.checked_particle_count
                            checked_component_count = [int]$Run.initial_primitives.checked_component_count
                            checked_value_count = [int]$Run.initial_primitives.checked_value_count
                            roundoff_multiplier = [double]$Run.initial_primitives.roundoff_multiplier
                            maximum_difference_in_component_scale_ulp = [double]$Run.initial_primitives.maximum_difference_in_component_scale_ulp
                        }
                    }
                }
                $RunReceipt = [ordered]@{
                    schema_version = 1
                    tool_revision = "m3c1_full_physics_common_p1_reference_v1"
                    classification = "external_comsol_full_physics_common_p1_reference"
                    status = "COMPLETE"
                    candidate_input_sha256 = $CandidateHash
                    table_receipt_sha256 = $TableReceiptHash
                    source_sha256_before = $SourceHashBefore
                    source_sha256_after = $SourceHashAfter
                    comsol_version = $VersionText
                    scope = [ordered]@{
                        particles = 287
                        frames = 46
                        time_window_s = @(0.0, 0.00045)
                        output_interval_s = 0.00001
                    }
                    runs = $RunRecords
                    artifact_hashes_sha256 = $ArtifactHashesHash
                    prepared_table_validation = $Summary.prepared_table_artifact_validation
                }
            }
            $RunReceipt | ConvertTo-Json -Depth 6 | Set-Content `
                -LiteralPath (Join-Path $OutputRoot $RunReceiptFilename) `
                -Encoding utf8
        } catch {
            $Succeeded = $false
            $ReceiptText = "artifact receipt construction failed: $($_.Exception.Message)"
            $FailureText = $(if ($FailureText) { "$FailureText; $ReceiptText" } else { $ReceiptText })
        }
    }

    $Status = [ordered]@{
        status = $(if ($Succeeded) { "COMPLETE" } else { "INCOMPLETE" })
        failure = $FailureText
        source_copy_retained = $SourceRetained
        compiled_class_retained = $ClassRetained
        class_status_retained = $ClassStatusRetained
        generated_utc = [DateTime]::UtcNow.ToString("o")
    }
    $Status | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $StatusFile -Encoding utf8
}

if (-not $Succeeded) {
    throw "M3-C1 common-P1 $RunProfile run did not complete"
}

Write-Output "M3-C1 full-physics common-P1 $RunProfile run completed: $OutputDirectory"
