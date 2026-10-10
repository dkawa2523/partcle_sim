param(
    [string]$ComsolRoot = "C:\Program Files\COMSOL\COMSOL64\Multiphysics_copy1",
    [string]$PreparedRoot = "",
    [string]$OutputDirectory = "",
    [double]$FixedStepS = 1.0e-5,
    [switch]$DiagnosticForceExport
)

$ErrorActionPreference = "Stop"
$InvocationDirectory = (Get-Location).ProviderPath
$ComsolRoot = [IO.Path]::GetFullPath($ComsolRoot, $InvocationDirectory)
$SolverRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$RepositoryRoot = (Resolve-Path (Join-Path $SolverRoot "..\..")).Path
$FixedStepText = $(
    if ($FixedStepS -eq 1.0e-5) {
        "1e-5"
    } elseif ($FixedStepS -eq 5.0e-6) {
        "5e-6"
    } elseif ($FixedStepS -eq 2.5e-6) {
        "2.5e-6"
    } elseif ($FixedStepS -eq 1.25e-6) {
        "1.25e-6"
    } else {
        throw "M3-C3 Case-P fixed RK4 step must be 1e-5, 5e-6, 2.5e-6, or 1.25e-6 s"
    }
)
$NumericalRunRole = $(
    if ($FixedStepS -eq 1.0e-5) { "baseline" } else { "time_step_refinement" }
)
if ($DiagnosticForceExport -and $FixedStepS -ne 1.25e-6) {
    throw "M3-C3 force diagnostic is restricted to the 1.25 us fine rerun"
}
$PreparedRoot = $(
    if ([string]::IsNullOrWhiteSpace($PreparedRoot)) {
        Join-Path $SolverRoot "_out_m3c3_casep_three_current_v1"
    } else {
        [IO.Path]::GetFullPath($PreparedRoot, $InvocationDirectory)
    }
)
$OutputDirectory = $(
    if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
        $DefaultOutputName = $(
            if ($DiagnosticForceExport) {
                "diagnostic_dt_1p25us"
            } elseif ($NumericalRunRole -ceq "baseline") {
                "reference"
            } elseif ($FixedStepS -eq 5.0e-6) {
                "reference_dt_5us"
            } elseif ($FixedStepS -eq 2.5e-6) {
                "reference_dt_2p5us"
            } else {
                "reference_dt_1p25us"
            }
        )
        Join-Path $PreparedRoot $DefaultOutputName
    } else {
        [IO.Path]::GetFullPath($OutputDirectory, $InvocationDirectory)
    }
)

$ReferenceConfigPath = Join-Path $PreparedRoot "reference_run_config.json"
$CampaignConfigPath = Join-Path $PreparedRoot "campaign_config.json"
$CandidateInput = Join-Path $PreparedRoot "candidate_input_three_current_z0.h5"
$ReleaseState = Join-Path $PreparedRoot "three_current_release_state.csv"
$ReleaseReceiptPath = Join-Path $PreparedRoot "three_current_release_receipt.json"
$PrimitiveReceiptPath = Join-Path $PreparedRoot "primitive_receipt.json"
$JavaSource = Join-Path $PSScriptRoot "comsol\RunM3C3CasePThreeCurrent.java"
$DiagnosticJavaSource = Join-Path `
    $PSScriptRoot "comsol\RunM3C3CasePThreeCurrentDiagnostic.java"
$ReadbackJavaSource = Join-Path $PSScriptRoot "comsol\ParticleRunReadback.java"
$CoefficientJavaSource = Join-Path $PSScriptRoot "comsol\CommonP1Epstein.java"
$ActualReceiptReader = Join-Path $PSScriptRoot "actual_run_receipt.py"
$BoundaryResponseMapping = Join-Path $PSScriptRoot "boundary_response_mapping.py"
$Preparer = Join-Path $PSScriptRoot "prepare_m3c3_caseP_reference_tables.py"
$Normalizer = Join-Path $PSScriptRoot "normalize_m3c3_caseP_three_current.py"
$Compiler = Join-Path $ComsolRoot "bin\win64\comsolcompile.exe"
$Batch = Join-Path $ComsolRoot "bin\win64\comsolbatch.exe"

$RequiredInputs = @(
        $ReferenceConfigPath,
        $CampaignConfigPath,
        $CandidateInput,
        $ReleaseState,
        $ReleaseReceiptPath,
        $PrimitiveReceiptPath,
        $JavaSource,
        $ReadbackJavaSource,
        $CoefficientJavaSource,
        $ActualReceiptReader,
        $BoundaryResponseMapping,
        $Preparer,
        $Normalizer,
        $Compiler,
        $Batch
    )
if ($DiagnosticForceExport) {
    $RequiredInputs += $DiagnosticJavaSource
}
foreach ($Required in $RequiredInputs) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required M3-C3 Case-P reference input does not exist: $Required"
    }
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "M3-C3 Case-P reference output already exists: $OutputDirectory"
}

$Reference = Get-Content -LiteralPath $ReferenceConfigPath -Raw | ConvertFrom-Json
$Campaign = Get-Content -LiteralPath $CampaignConfigPath -Raw | ConvertFrom-Json
$ReleaseReceipt = Get-Content -LiteralPath $ReleaseReceiptPath -Raw | ConvertFrom-Json
$ExpectedSourceHash = "3bbf08e3469758313eac5de473a7a0dd4cc9a6f72c9722393229f0b856e9b524"
$ExpectedPrimitiveHash = [string]$Campaign.primitive_input.sha256
if ($ExpectedPrimitiveHash -notmatch '^[0-9a-f]{64}$') {
    throw "Campaign must register an explicit canonical primitive input SHA256"
}
$ExpectedChargeRevision = "aggregate_relative_drift_regularized_three_current_v1"
$ExpectedIonDragRevision = "relative_flow_screened_collection_orbital_aggregate_ion_v1"

if ([int]$Reference.schema_version -ne 1 -or
    [string]$Reference.case_id -cne "caseP_100nm_three_current" -or
    [string]$Reference.charge_revision -cne $ExpectedChargeRevision -or
    [string]$Reference.ion_drag_revision -cne $ExpectedIonDragRevision -or
    [double]$Reference.maximum_relative_ion_speed_m_s -ne 1.0e6 -or
    [double]$Reference.comsol.fixed_rk4_step_s -ne 1.0e-5 -or
    [double]$Reference.comsol.time_end_s -ne 0.03 -or
    [int]$Reference.comsol.output_count -ne 121 -or
    [int]$Reference.comsol.particle_count -ne 287) {
    throw "Prepared M3-C3 reference configuration differs from the fixed campaign"
}
if ([int]$Campaign.schema_version -ne 1 -or
    [string]$Campaign.campaign_id -cne "M3-C3-caseP-100nm-three-current" -or
    [string]$Campaign.source_mph.sha256 -cne $ExpectedSourceHash -or
    [string]$Campaign.primitive_input.sha256 -cne $ExpectedPrimitiveHash -or
    [string]$Campaign.physics.charge_revision -cne $ExpectedChargeRevision -or
    [string]$Campaign.physics.ion_drag_revision -cne $ExpectedIonDragRevision) {
    throw "Prepared M3-C3 campaign configuration differs from the fixed campaign"
}

$SourceModel = Join-Path $RepositoryRoot ([string]$Reference.source_mph.path)
$SourceModel = [IO.Path]::GetFullPath($SourceModel)
$ExpectedPreparedPaths = @{
    candidate = [string]$Reference.canonical_input.path
    primitive_receipt = [string]$Reference.primitive_receipt.path
    release_state = [string]$Reference.release_state.path
    release_receipt = [string]$Reference.release_receipt.path
}
if ($ExpectedPreparedPaths.candidate -cne "candidate_input_three_current_z0.h5" -or
    $ExpectedPreparedPaths.primitive_receipt -cne "primitive_receipt.json" -or
    $ExpectedPreparedPaths.release_state -cne "three_current_release_state.csv" -or
    $ExpectedPreparedPaths.release_receipt -cne "three_current_release_receipt.json") {
    throw "Prepared M3-C3 reference paths are not the fixed local artifact names"
}
foreach ($Item in @(
        @($SourceModel, [string]$Reference.source_mph.sha256),
        @($CandidateInput, [string]$Reference.canonical_input.file_sha256),
        @($PrimitiveReceiptPath, [string]$Reference.primitive_receipt.sha256),
        @($ReleaseState, [string]$Reference.release_state.sha256),
        @($ReleaseReceiptPath, [string]$Reference.release_receipt.sha256)
    )) {
    if (-not (Test-Path -LiteralPath $Item[0] -PathType Leaf)) {
        throw "Locked M3-C3 artifact does not exist: $($Item[0])"
    }
    $ActualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Item[0]).Hash.ToLowerInvariant()
    if ($ActualHash -cne $Item[1]) {
        throw "Locked M3-C3 artifact hash differs: $($Item[0])"
    }
}
if ([string]$Reference.source_mph.sha256 -cne $ExpectedSourceHash -or
    [string]$Reference.canonical_input.parent_file_sha256 -cne $ExpectedPrimitiveHash -or
    [string]$Campaign.source_mph.repository_relative_path -cne
        [string]$Reference.source_mph.path) {
    throw "M3-C3 parent/source identity differs from the locked campaign"
}
if ([int]$ReleaseReceipt.schema_version -ne 1 -or
    [string]$ReleaseReceipt.status -cne "PASS" -or
    [string]$ReleaseReceipt.case_id -cne "caseP_100nm_three_current" -or
    [string]$ReleaseReceipt.formula_revision -cne $ExpectedChargeRevision -or
    -not [bool]$ReleaseReceipt.single_charge_authority -or
    [bool]$ReleaseReceipt.comsol_recomputes_equilibrium -or
    [string]$ReleaseReceipt.primitive_input.file_sha256 -cne $ExpectedPrimitiveHash -or
    [string]$ReleaseReceipt.derived_input.file_sha256 -cne
        [string]$Reference.canonical_input.file_sha256 -or
    [string]$ReleaseReceipt.derived_input.content_hash -cne
        [string]$Reference.canonical_input.content_hash -or
    [string]$ReleaseReceipt.release_state.sha256 -cne
        [string]$Reference.release_state.sha256 -or
    [int]$ReleaseReceipt.release_state.rows -ne 287) {
    throw "Three-current release receipt does not bind the prepared reference"
}

$VersionText = (& $Batch -version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $VersionText -notmatch "6\.4\.0\.429") {
    throw "COMSOL version does not match the locked 6.4.0.429 environment: $VersionText"
}
$SourceHashBefore = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel
).Hash.ToLowerInvariant()
if ($SourceHashBefore -cne $ExpectedSourceHash) {
    throw "Source MPH differs from the locked M3-C3 campaign"
}

Push-Location $SolverRoot
try {
    & uv run --locked python -m tools.vv.comsol.prepare_m3c3_caseP_reference_tables $CandidateInput $ReleaseState $OutputDirectory `
        --expected-content-hash ([string]$Reference.canonical_input.content_hash)
    if ($LASTEXITCODE -ne 0) {
        throw "M3-C3 reference table preparation failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

$SourceCopy = Join-Path $OutputDirectory "source_copy.mph"
$StagedJava = Join-Path $OutputDirectory "RunM3C3CasePThreeCurrent.java"
$StagedReadbackJava = Join-Path $OutputDirectory "ParticleRunReadback.java"
$StagedCoefficientJava = Join-Path $OutputDirectory "CommonP1Epstein.java"
$StagedDiagnosticJava = Join-Path `
    $OutputDirectory "RunM3C3CasePThreeCurrentDiagnostic.java"
$FormalClassFile = Join-Path $OutputDirectory "RunM3C3CasePThreeCurrent.class"
$DiagnosticClassFile = Join-Path $OutputDirectory "RunM3C3CasePThreeCurrentDiagnostic.class"
$ClassFile = $(
    if ($DiagnosticForceExport) { $DiagnosticClassFile } else { $FormalClassFile }
)
$CompiledClassFiles = @($FormalClassFile,
    (Join-Path $OutputDirectory "ParticleRunReadback.class"),
    (Join-Path $OutputDirectory "CommonP1Epstein.class"))
if ($DiagnosticForceExport) {
    $CompiledClassFiles += $DiagnosticClassFile
}
$ClassStatusFiles = @($CompiledClassFiles | ForEach-Object { "${_}.status" })
$StagedReferenceConfig = Join-Path $OutputDirectory "reference_run_config.json"
$StagedCampaignConfig = Join-Path $OutputDirectory "campaign_config.json"
$StagedPrimitiveReceipt = Join-Path $OutputDirectory "primitive_receipt.json"
$StagedReleaseReceipt = Join-Path $OutputDirectory "three_current_release_receipt.json"
$ProcessLog = Join-Path $OutputDirectory "comsol_process.log"
$ProcessErrorLog = Join-Path $OutputDirectory "comsol_process_error.log"
$BatchLog = Join-Path $OutputDirectory "comsol_batch.log"
$ProcessMetricsPath = Join-Path $OutputDirectory "comsol_process_metrics.json"
$ExecutionInputsPath = Join-Path $OutputDirectory "execution_inputs.json"
$PreferencesDirectory = Join-Path $OutputDirectory ".comsol_preferences"
$Succeeded = $false
$ReadbackProducerLocks = @(
    @($ReadbackJavaSource, (Get-FileHash -Algorithm SHA256 -LiteralPath $ReadbackJavaSource).Hash.ToLowerInvariant()),
    @($CoefficientJavaSource, (Get-FileHash -Algorithm SHA256 -LiteralPath $CoefficientJavaSource).Hash.ToLowerInvariant()),
    @($ActualReceiptReader, (Get-FileHash -Algorithm SHA256 -LiteralPath $ActualReceiptReader).Hash.ToLowerInvariant()),
    @($BoundaryResponseMapping, (Get-FileHash -Algorithm SHA256 -LiteralPath $BoundaryResponseMapping).Hash.ToLowerInvariant())
)
try {
    Copy-Item -LiteralPath $SourceModel -Destination $SourceCopy
    Copy-Item -LiteralPath $JavaSource -Destination $StagedJava
    Copy-Item -LiteralPath $ReadbackJavaSource -Destination $StagedReadbackJava
    Copy-Item -LiteralPath $CoefficientJavaSource -Destination $StagedCoefficientJava
    if ($DiagnosticForceExport) {
        Copy-Item -LiteralPath $DiagnosticJavaSource -Destination $StagedDiagnosticJava
    }
    $StepToken = "__M3C3_FIXED_STEP_SECONDS__"
    $StagedJavaText = [IO.File]::ReadAllText($StagedJava)
    $StepTokenCount = [Text.RegularExpressions.Regex]::Matches(
        $StagedJavaText,
        [Text.RegularExpressions.Regex]::Escape($StepToken)
    ).Count
    if ($StepTokenCount -ne 1) {
        throw "M3-C3 staged Java must contain exactly one fixed-step token"
    }
    $StagedJavaText = $StagedJavaText.Replace($StepToken, $FixedStepText)
    [IO.File]::WriteAllText(
        $StagedJava,
        $StagedJavaText,
        [Text.UTF8Encoding]::new($false)
    )
    Copy-Item -LiteralPath $ReferenceConfigPath -Destination $StagedReferenceConfig
    Copy-Item -LiteralPath $CampaignConfigPath -Destination $StagedCampaignConfig
    Copy-Item -LiteralPath $PrimitiveReceiptPath -Destination $StagedPrimitiveReceipt
    Copy-Item -LiteralPath $ReleaseReceiptPath -Destination $StagedReleaseReceipt
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $SourceCopy).Hash.ToLowerInvariant() `
        -cne $SourceHashBefore) {
        throw "Isolated source copy differs from the locked source MPH"
    }
    New-Item -ItemType Directory -Path $PreferencesDirectory | Out-Null

    Push-Location $OutputDirectory
    try {
        & $Compiler $StagedCoefficientJava
        if ($LASTEXITCODE -ne 0) { throw "Common-P1 coefficient Java compilation failed" }
        & $Compiler -classpathadd $OutputDirectory $StagedReadbackJava
        if ($LASTEXITCODE -ne 0) { throw "Actual readback Java compilation failed" }
        & $Compiler -classpathadd $OutputDirectory $StagedJava
        if ($LASTEXITCODE -ne 0 -or
            -not (Test-Path -LiteralPath $FormalClassFile -PathType Leaf)) {
            throw "COMSOL M3-C3 Java compilation failed with exit code $LASTEXITCODE"
        }
        if ($DiagnosticForceExport) {
            & $Compiler -classpathadd $OutputDirectory $StagedDiagnosticJava
            if ($LASTEXITCODE -ne 0 -or
                -not (Test-Path -LiteralPath $DiagnosticClassFile -PathType Leaf)) {
                throw "COMSOL M3-C3 diagnostic Java compilation failed with exit code $LASTEXITCODE"
            }
        }
        $BatchArguments = @(
            "-inputfile", "`"$ClassFile`"",
            "-nosave",
            "-error", "on",
            "-prefsdir", "`"$PreferencesDirectory`"",
            "-np", "1",
            "-batchlog", "`"$BatchLog`""
        )
        $Stopwatch = [Diagnostics.Stopwatch]::StartNew()
        $Process = Start-Process -FilePath $Batch -ArgumentList $BatchArguments `
            -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $ProcessLog -RedirectStandardError $ProcessErrorLog
        [long]$PeakRssBytes = 0
        while (-not $Process.HasExited) {
            try {
                $Process.Refresh()
                $PeakRssBytes = [Math]::Max($PeakRssBytes, [long]$Process.PeakWorkingSet64)
            } catch {
                # The process can exit between HasExited and Refresh.
            }
            Start-Sleep -Milliseconds 200
        }
        $Process.WaitForExit()
        $Stopwatch.Stop()
        try {
            $Process.Refresh()
            $PeakRssBytes = [Math]::Max($PeakRssBytes, [long]$Process.PeakWorkingSet64)
        } catch {
            # A completed process may no longer expose counters.
        }
        $ProcessMetrics = [ordered]@{
            schema_version = 1
            exit_code = $Process.ExitCode
            wall_time_s = $Stopwatch.Elapsed.TotalSeconds
            peak_rss_bytes = $PeakRssBytes
            process_count = 1
            fixed_rk4_step_s = $FixedStepS
            numerical_run_role = $NumericalRunRole
        }
        if ($DiagnosticForceExport) {
            $ProcessMetrics["diagnostic_force_export"] = $true
        }
        [IO.File]::WriteAllText(
            $ProcessMetricsPath,
            (($ProcessMetrics | ConvertTo-Json -Depth 8) + "`n"),
            [Text.UTF8Encoding]::new($false)
        )
        if ($Process.ExitCode -ne 0) {
            throw "COMSOL M3-C3 batch failed with exit code $($Process.ExitCode)"
        }
    } finally {
        Pop-Location
    }
    $CompletionMarker = $(
        if ($DiagnosticForceExport) {
            "M3C3_CASEP|diagnostic_run_pass|case=caseP_100nm_three_current|step_s=$FixedStepText|state=diagnostic_state_raw_wide.csv|force=diagnostic_force_raw_wide.csv|model_saved=false"
        } else {
            "M3C3_CASEP|run_pass|case=caseP_100nm_three_current|step_s=$FixedStepText|time_end_s=0.03|output_times=121|particles=287|model_saved=false"
        }
    )
    if (@(Select-String -LiteralPath $ProcessLog -SimpleMatch $CompletionMarker).Count -ne 1) {
        throw "COMSOL M3-C3 runner did not emit its completion receipt"
    }
    if (Select-String -LiteralPath $ProcessLog, $BatchLog, $ProcessErrorLog -Pattern `
            'M3C3_CASEP\|fatal\||Error running java class\.|/\*+Error\*+/') {
        throw "COMSOL M3-C3 native log contains an execution error"
    }
    $NativeClassStatusPath = "$ClassFile.status"
    $NativeClassStatus = $(
        if (Test-Path -LiteralPath $NativeClassStatusPath) {
            (Get-Content -LiteralPath $NativeClassStatusPath -Raw).Trim()
        } else { "not emitted" }
    )
    $ProcessMetrics["batch_completion"] = [ordered]@{
        status = "COMPLETE"
        expected_completion_record = $CompletionMarker
        native_error_record_absent = $true
        class_status_raw = $NativeClassStatus
        class_status_authority = "non_authoritative_for_6_4_class_input_verified_by_controls"
        completion_authority = "native_process_log_and_registered_native_artifacts"
    }
    [IO.File]::WriteAllText(
        $ProcessMetricsPath,
        (($ProcessMetrics | ConvertTo-Json -Depth 8) + "`n"),
        [Text.UTF8Encoding]::new($false)
    )
    if (-not (Test-Path -LiteralPath (Join-Path $OutputDirectory "trajectory_raw_wide.csv"))) {
        throw "COMSOL M3-C3 runner did not export trajectory_raw_wide.csv"
    }
    if ($DiagnosticForceExport) {
        foreach ($DiagnosticOutput in @(
                "diagnostic_state_raw_wide.csv",
                "diagnostic_force_raw_wide.csv"
            )) {
            $DiagnosticOutputPath = Join-Path $OutputDirectory $DiagnosticOutput
            if (-not (Test-Path -LiteralPath $DiagnosticOutputPath -PathType Leaf) -or
                (Get-Item -LiteralPath $DiagnosticOutputPath).Length -eq 0) {
                throw "COMSOL M3-C3 diagnostic output is missing or empty: $DiagnosticOutput"
            }
        }
    }
    $SourceHashAfter = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel
    ).Hash.ToLowerInvariant()
    if ($SourceHashAfter -cne $SourceHashBefore) {
        throw "Locked source MPH changed during M3-C3 execution"
    }
    foreach ($Lock in $ReadbackProducerLocks) {
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Lock[0]).Hash.ToLowerInvariant() -cne $Lock[1]) {
            throw "A readback producer changed during M3-C3 execution"
        }
    }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $StagedReadbackJava).Hash.ToLowerInvariant() -cne $ReadbackProducerLocks[0][1] -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $StagedCoefficientJava).Hash.ToLowerInvariant() -cne $ReadbackProducerLocks[1][1]) {
        throw "A staged readback producer differs from its source"
    }

    Push-Location $SolverRoot
    try {
        & uv run --locked python -m tools.vv.comsol.actual_run_receipt $OutputDirectory
        if ($LASTEXITCODE -ne 0) { throw "COMSOL actual readback materialization failed" }
    } finally {
        Pop-Location
    }
    $Artifacts = @(
        [ordered]@{ role = "source_mph"; path = $SourceModel; sha256 = $SourceHashBefore },
        [ordered]@{
            role = "canonical_three_current_input"
            path = $CandidateInput
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $CandidateInput).Hash.ToLowerInvariant()
        },
        [ordered]@{
            role = "reference_run_config"
            path = "reference_run_config.json"
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $StagedReferenceConfig).Hash.ToLowerInvariant()
        },
        [ordered]@{
            role = "primitive_receipt"
            path = "primitive_receipt.json"
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $StagedPrimitiveReceipt).Hash.ToLowerInvariant()
        },
        [ordered]@{
            role = "release_receipt"
            path = "three_current_release_receipt.json"
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $StagedReleaseReceipt).Hash.ToLowerInvariant()
        },
        [ordered]@{
            role = "release_state"
            path = "three_current_release_state.csv"
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (
                Join-Path $OutputDirectory "three_current_release_state.csv"
            )).Hash.ToLowerInvariant()
        },
        [ordered]@{
            role = "table_receipt"
            path = "m3c3_table_receipt.json"
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (
                Join-Path $OutputDirectory "m3c3_table_receipt.json"
            )).Hash.ToLowerInvariant()
        },
        [ordered]@{
            role = "java_runner"
            path = "RunM3C3CasePThreeCurrent.java"
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $StagedJava).Hash.ToLowerInvariant()
        },
        [ordered]@{
            role = "actual_run_readback"
            path = "actual_binding_receipt.json"
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $OutputDirectory "actual_binding_receipt.json")).Hash.ToLowerInvariant()
        },
        [ordered]@{
            role = "readback_java"
            path = "ParticleRunReadback.java"
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $StagedReadbackJava).Hash.ToLowerInvariant()
        },
        [ordered]@{
            role = "coefficient_java"
            path = "CommonP1Epstein.java"
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $StagedCoefficientJava).Hash.ToLowerInvariant()
        },
        [ordered]@{
            role = "actual_receipt_reader"
            path = $ActualReceiptReader
            sha256 = $ReadbackProducerLocks[2][1]
        },
        [ordered]@{
            role = "boundary_response_mapping"
            path = $BoundaryResponseMapping
            sha256 = $ReadbackProducerLocks[3][1]
        },
        [ordered]@{
            role = "powershell_runner"
            path = $PSCommandPath
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath).Hash.ToLowerInvariant()
        },
        [ordered]@{
            role = "table_preparer"
            path = $Preparer
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Preparer).Hash.ToLowerInvariant()
        }
    )
    $Artifacts += [ordered]@{
            role = "normalizer"
            path = $Normalizer
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Normalizer).Hash.ToLowerInvariant()
    }
    if ($DiagnosticForceExport) {
        $Artifacts += @(
            [ordered]@{
                role = "diagnostic_java_runner"
                path = "RunM3C3CasePThreeCurrentDiagnostic.java"
                sha256 = (
                    Get-FileHash -Algorithm SHA256 -LiteralPath $StagedDiagnosticJava
                ).Hash.ToLowerInvariant()
            },
            [ordered]@{
                role = "diagnostic_state_raw"
                path = "diagnostic_state_raw_wide.csv"
                sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (
                    Join-Path $OutputDirectory "diagnostic_state_raw_wide.csv"
                )).Hash.ToLowerInvariant()
            },
            [ordered]@{
                role = "diagnostic_force_raw"
                path = "diagnostic_force_raw_wide.csv"
                sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (
                    Join-Path $OutputDirectory "diagnostic_force_raw_wide.csv"
                )).Hash.ToLowerInvariant()
            },
            [ordered]@{
                role = "diagnostic_configuration_log"
                path = "comsol_process.log"
                sha256 = (
                    Get-FileHash -Algorithm SHA256 -LiteralPath $ProcessLog
                ).Hash.ToLowerInvariant()
            }
        )
    }
    $NumericalRun = [ordered]@{
        role = $NumericalRunRole
        fixed_rk4_step_s = $FixedStepS
        base_config_fixed_rk4_step_s = [double]$Reference.comsol.fixed_rk4_step_s
    }
    if ($DiagnosticForceExport) {
        $NumericalRun["diagnostic_force_export"] = $true
        $NumericalRun["comparison_scope"] = "active_rows_only"
    }
    $ExecutionInputs = [ordered]@{
        schema_version = 1
        status = "LOCKED_FOR_EXECUTION"
        case_id = "caseP_100nm_three_current"
        expected_source_mph_sha256 = $ExpectedSourceHash
        source_mph_sha256_before = $SourceHashBefore
        source_mph_sha256_after = $SourceHashAfter
        reference_run_config_sha256 = (
            Get-FileHash -Algorithm SHA256 -LiteralPath $StagedReferenceConfig
        ).Hash.ToLowerInvariant()
        numerical_run = $NumericalRun
        artifacts = $Artifacts
    }
    [IO.File]::WriteAllText(
        $ExecutionInputsPath,
        (($ExecutionInputs | ConvertTo-Json -Depth 12) + "`n"),
        [Text.UTF8Encoding]::new($false)
    )

    # Every invocation exports and normalizes the same complete trajectory.
    # DiagnosticForceExport only adds observations after the identical solve.
    Push-Location $SolverRoot
        try {
            & uv run --locked python -m tools.vv.comsol.normalize_m3c3_caseP_three_current $OutputDirectory
            if ($LASTEXITCODE -ne 0) {
                throw "M3-C3 reference normalization failed with exit code $LASTEXITCODE"
            }
        } finally {
            Pop-Location
        }
        $Summary = Get-Content -LiteralPath (
            Join-Path $OutputDirectory "normalization_summary.json"
        ) -Raw | ConvertFrom-Json
        if ([string]$Summary.status -cne "COMPLETE_NORMALIZED_NOT_EVALUATED" -or
            [int]$Summary.trajectory_rows -ne (287 * 121) -or
            [double]$Summary.fixed_rk4_step_s -ne $FixedStepS) {
            throw "M3-C3 normalized reference did not pass its structural checks"
        }
    $Succeeded = $true
} finally {
    $CleanupTargets = @($SourceCopy) + $CompiledClassFiles + $ClassStatusFiles
    Remove-Item -LiteralPath $CleanupTargets -ErrorAction SilentlyContinue
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel).Hash.ToLowerInvariant() `
        -cne $SourceHashBefore) {
        throw "Locked source MPH changed during M3-C3 reference execution"
    }
}

if (-not $Succeeded) {
    throw "M3-C3 Case-P three-current COMSOL reference did not complete"
}
$RunLabel = $(
    if ($DiagnosticForceExport) { "external V&V diagnostic" } else { "reference" }
)
Write-Output "M3-C3 Case-P three-current COMSOL $RunLabel passed: $OutputDirectory"
