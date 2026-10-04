param(
    [string]$ComsolRoot = "C:\Program Files\COMSOL\COMSOL64\Multiphysics_copy1",
    [string]$CandidateRoot = "",
    [string]$OutputDirectory = "",
    [ValidateSet("caseA", "caseP", "all")]
    [string]$Case = "all",
    [switch]$Execute
)

$ErrorActionPreference = "Stop"

$InvocationDirectory = (Get-Location).ProviderPath
$SolverRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$RepositoryRoot = (Resolve-Path (Join-Path $SolverRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($CandidateRoot)) {
    $CandidateRoot = Join-Path $SolverRoot "_out_m3c1\theory_100nm_30ms_candidate_v2"
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $SolverRoot "_out_m3c1\theory_100nm_30ms_comsol_v2"
}
$ComsolRoot = [IO.Path]::GetFullPath($ComsolRoot, $InvocationDirectory)
$CandidateRoot = [IO.Path]::GetFullPath($CandidateRoot, $InvocationDirectory)
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory, $InvocationDirectory)

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Format-InvariantDouble {
    param([Parameter(Mandatory = $true)][double]$Value)

    if ([double]::IsNaN($Value) -or [double]::IsInfinity($Value)) {
        throw "Cannot format a non-finite numeric contract value"
    }
    return $Value.ToString("R", [Globalization.CultureInfo]::InvariantCulture)
}

function Confirm-PreparedTables {
    param([Parameter(Mandatory = $true)][string]$CaseRoot)

    $ReceiptPath = Join-Path $CaseRoot "common_p1_table_receipt.json"
    $Receipt = Get-Content -LiteralPath $ReceiptPath -Raw | ConvertFrom-Json
    if ([int]$Receipt.schema_version -ne 1 -or
        [string]$Receipt.tool_revision -ne "m3c1_full_physics_common_p1_tables_v1") {
        throw "Unsupported exact-P1 prepared-table receipt"
    }
    $Entries = @($Receipt.artifacts.PSObject.Properties)
    if ($Entries.Count -ne 26 -or [int]$Receipt.component_count -ne 22) {
        throw "Exact-P1 receipt must attest 22 primitive components and 26 artifacts"
    }
    $Verified = @()
    [long]$TotalSize = 0
    foreach ($Entry in $Entries) {
        $Name = [string]$Entry.Name
        if ([string]::IsNullOrWhiteSpace($Name) -or $Name -in @(".", "..") -or
            [System.IO.Path]::IsPathRooted($Name) -or
            [System.IO.Path]::GetFileName($Name) -ne $Name) {
            throw "Unsafe exact-P1 artifact name: $Name"
        }
        $Path = Join-Path $CaseRoot $Name
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "Missing exact-P1 artifact: $Name"
        }
        $Size = (Get-Item -LiteralPath $Path).Length
        $Hash = Get-Sha256 $Path
        if ($Size -ne [long]$Entry.Value.size_bytes -or
            $Hash -ne [string]$Entry.Value.sha256) {
            throw "Exact-P1 artifact differs from its receipt: $Name"
        }
        $TotalSize += $Size
        $Verified += [ordered]@{ path = $Name; sha256 = $Hash; size_bytes = $Size }
    }
    return [ordered]@{
        status = "PASS"
        receipt_sha256 = Get-Sha256 $ReceiptPath
        artifact_count = $Entries.Count
        total_size_bytes = $TotalSize
        artifacts = $Verified
        receipt = $Receipt
    }
}

function Confirm-SharedContract {
    param([Parameter(Mandatory = $true)][object]$Config)

    if ([int]$Config.schema_version -ne 1 -or [int]$Config.evaluation_revision -ne 2) {
        throw "Unsupported shared M3-C1 30 ms contract"
    }
    if ([int]$Config.matrix.particle_count -ne 287 -or
        [double]$Config.matrix.particle_diameter_m -ne 1.0e-7 -or
        [double]$Config.matrix.time_end_s -ne 0.03 -or
        [int]$Config.matrix.output_count -ne 121) {
        throw "Shared M3-C1 matrix differs from the COMSOL reference contract"
    }
    $ExpectedRunKeys = @("coarse", "medium", "fine")
    $ActualRunKeys = @($Config.matrix.run_keys)
    if ($ActualRunKeys.Count -ne $ExpectedRunKeys.Count) {
        throw "Shared M3-C1 run-key series has the wrong length"
    }
    for ($Index = 0; $Index -lt $ExpectedRunKeys.Count; $Index++) {
        if ([string]$ActualRunKeys[$Index] -cne $ExpectedRunKeys[$Index]) {
            throw "Shared M3-C1 run-key series differs at index $Index"
        }
    }
    $ExpectedSteps = [ordered]@{
        caseA = @(6.25e-7, 3.125e-7, 1.5625e-7)
        caseP = @(4.6875e-8, 2.34375e-8, 1.171875e-8)
    }
    $ExpectedLipschitz = [ordered]@{
        caseA = 287013.1640482939
        caseP = 10403456.669581516
    }
    $Maximum = [double]$Config.acceptance.maximum_dt_charge_lipschitz
    if ([double]::IsNaN($Maximum) -or [double]::IsInfinity($Maximum) -or $Maximum -ne 0.5) {
        throw "Shared M3-C1 charge Lipschitz limit differs"
    }
    $WorkflowNames = @($Config.workflows.PSObject.Properties.Name)
    if ($WorkflowNames.Count -ne 2 -or
        -not ($WorkflowNames -contains "caseA") -or
        -not ($WorkflowNames -contains "caseP")) {
        throw "Shared M3-C1 workflows must be exactly caseA and caseP"
    }
    foreach ($CaseName in @("caseA", "caseP")) {
        $Workflow = $Config.workflows.$CaseName
        $ActualSteps = @($Workflow.fixed_rk4_steps_s)
        if ($ActualSteps.Count -ne $ExpectedRunKeys.Count) {
            throw "$CaseName fixed-step series has the wrong length"
        }
        $Lipschitz = [double]$Workflow.charge_lipschitz_s_inv
        if ([double]::IsNaN($Lipschitz) -or [double]::IsInfinity($Lipschitz) -or
            $Lipschitz -ne [double]$ExpectedLipschitz[$CaseName]) {
            throw "$CaseName charge Lipschitz certificate differs"
        }
        for ($Index = 0; $Index -lt $ActualSteps.Count; $Index++) {
            $Step = [double]$ActualSteps[$Index]
            if ([double]::IsNaN($Step) -or [double]::IsInfinity($Step) -or $Step -le 0.0 -or
                $Step -ne [double]$ExpectedSteps[$CaseName][$Index]) {
                throw "$CaseName fixed-step series differs at index $Index"
            }
            $StepCount = 0.03 / $Step
            if ([math]::Abs($StepCount - [math]::Round($StepCount)) -gt 1.0e-9) {
                throw "$CaseName $($ExpectedRunKeys[$Index]) step does not divide 30 ms"
            }
            if ($Step * $Lipschitz -gt $Maximum) {
                throw "$CaseName $($ExpectedRunKeys[$Index]) violates the charge Lipschitz limit"
            }
        }
        if ([double]$ActualSteps[0] -ne 2.0 * [double]$ActualSteps[1] -or
            [double]$ActualSteps[1] -ne 2.0 * [double]$ActualSteps[2]) {
            throw "$CaseName fixed RK4 steps are not exact h, h/2, h/4"
        }
    }
    if ([bool]$Config.physics.brownian_active -or [bool]$Config.physics.saffman_active) {
        throw "The formal deterministic contract must disable Brownian and Saffman forces"
    }
}

function Write-RunSpec {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$CaseName,
        [Parameter(Mandatory = $true)][object]$Workflow,
        [Parameter(Mandatory = $true)][object]$SharedWorkflow,
        [Parameter(Mandatory = $true)][string[]]$RunKeys,
        [Parameter(Mandatory = $true)][double]$MaximumDtChargeLipschitz,
        [Parameter(Mandatory = $true)][string]$CommonConfigHash,
        [Parameter(Mandatory = $true)][string]$ComsolConfigHash
    )

    $StepValues = @($SharedWorkflow.fixed_rk4_steps_s) |
        ForEach-Object { Format-InvariantDouble ([double]$_) }

    @(
        "case_name=$CaseName"
        "physics_tag=$($Workflow.physics_tag)"
        "background_study=$($Workflow.background_study)"
        "background_solution=$($Workflow.background_solution)"
        "source_dataset=$($Workflow.source_dataset)"
        "particle_geometry=$($Workflow.particle_geometry)"
        "position_dof_r=$($Workflow.position_dofs[0])"
        "position_dof_z=$($Workflow.position_dofs[1])"
        "charge_state=$($Workflow.charge_state)"
        "run_keys=$($RunKeys -join ',')"
        "fixed_rk4_steps_s=$($StepValues -join ',')"
        "charge_lipschitz_s_inv=$(Format-InvariantDouble ([double]$SharedWorkflow.charge_lipschitz_s_inv))"
        "maximum_dt_charge_lipschitz=$(Format-InvariantDouble $MaximumDtChargeLipschitz)"
        "common_config_sha256=$CommonConfigHash"
        "comsol_config_sha256=$ComsolConfigHash"
    ) | Set-Content -LiteralPath $Path -Encoding ascii
}

function Get-StepArtifacts {
    param(
        [Parameter(Mandatory = $true)][string]$CaseRoot,
        [Parameter(Mandatory = $true)][object]$Config,
        [Parameter(Mandatory = $true)][string]$CaseName
    )

    $Runs = [ordered]@{}
    $RunKeys = @($Config.matrix.run_keys)
    $Workflow = $Config.workflows.$CaseName
    $Steps = @($Workflow.fixed_rk4_steps_s)
    $Lipschitz = [double]$Workflow.charge_lipschitz_s_inv
    $Maximum = [double]$Config.acceptance.maximum_dt_charge_lipschitz
    for ($Index = 0; $Index -lt $RunKeys.Count; $Index++) {
        $RunKey = [string]$RunKeys[$Index]
        $Step = [double]$Steps[$Index]
        $DtLipschitz = $Step * $Lipschitz
        if ($DtLipschitz -gt $Maximum) {
            throw "$CaseName $RunKey violates the charge Lipschitz limit"
        }
        $StepRoot = Join-Path $CaseRoot $RunKey
        $Tables = [ordered]@{}
        foreach ($Name in @("state_raw_wide.csv", "force_raw_wide.csv", "primitive_raw_wide.csv")) {
            $Path = Join-Path $StepRoot $Name
            if (-not (Test-Path -LiteralPath $Path -PathType Leaf) -or
                (Get-Item -LiteralPath $Path).Length -le 0) {
                throw "Missing or empty COMSOL raw export: $Path"
            }
            $Tables[$Name] = [ordered]@{
                path = "$RunKey/$Name"
                sha256 = Get-Sha256 $Path
                size_bytes = (Get-Item -LiteralPath $Path).Length
            }
        }
        $Runs[$RunKey] = [ordered]@{
            step_s = $Step
            charge_lipschitz_s_inv = $Lipschitz
            dt_charge_lipschitz = $DtLipschitz
            maximum_dt_charge_lipschitz = $Maximum
            step_count_30ms = [long][math]::Round(0.03 / $Step)
            raw_tables = $Tables
        }
    }
    return $Runs
}

function Write-ArtifactLedger {
    param([Parameter(Mandatory = $true)][string]$CaseRoot)

    $Rows = Get-ChildItem -LiteralPath $CaseRoot -Recurse -File |
        Where-Object { $_.Name -ne "artifact_hashes.csv" } |
        Sort-Object FullName |
        ForEach-Object {
            [pscustomobject]@{
                path = [IO.Path]::GetRelativePath($CaseRoot, $_.FullName).Replace("\", "/")
                sha256 = Get-Sha256 $_.FullName
                bytes = $_.Length
            }
        }
    $Rows | Export-Csv -LiteralPath (Join-Path $CaseRoot "artifact_hashes.csv") `
        -NoTypeInformation -Encoding utf8
}

$Compiler = Join-Path $ComsolRoot "bin\win64\comsolcompile.exe"
$Batch = Join-Path $ComsolRoot "bin\win64\comsolbatch.exe"
$JavaSource = Join-Path $PSScriptRoot "comsol\RunM3C1Theory100nm30ms.java"
$JavaClass = Join-Path (Split-Path -Parent $JavaSource) "RunM3C1Theory100nm30ms.class"
$TablePreparer = Join-Path $PSScriptRoot "prepare_m3c1_common_p1_tables.py"
$SharedConfigPath = Join-Path $PSScriptRoot "cases\m3c1_theory_100nm_30ms_v2.json"
$ComsolConfigPath = Join-Path $PSScriptRoot "cases\m3c1_theory_100nm_30ms_comsol_v1.json"
$SourceModel = Join-Path $RepositoryRoot `
    "model_dataset\cf4_o2_etch_caseA_nonlinear_sass\model\icp_rf_bias_cf4_o2_si_etching_caseP_caseA_SASS_formal_iondrag_theory_consistent_10_30_100nm.mph"

foreach ($RequiredPath in @(
    $Compiler, $Batch, $JavaSource, $TablePreparer, $SharedConfigPath,
    $ComsolConfigPath, $SourceModel
)) {
    if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
        throw "Required M3-C1 30 ms COMSOL input does not exist: $RequiredPath"
    }
}

$SharedConfig = Get-Content -LiteralPath $SharedConfigPath -Raw | ConvertFrom-Json
$ComsolConfig = Get-Content -LiteralPath $ComsolConfigPath -Raw | ConvertFrom-Json
Confirm-SharedContract $SharedConfig
if ([int]$ComsolConfig.evaluation_revision -ne 2 -or
    [int]$ComsolConfig.shared_contract.required_evaluation_revision -ne 2 -or
    [string]$ComsolConfig.shared_contract.relative_path -cne
    "tools/vv/comsol/cases/m3c1_theory_100nm_30ms_v2.json") {
    throw "COMSOL adapter contract does not select the shared M3-C1 revision 2 authority"
}
$SharedConfigHash = Get-Sha256 $SharedConfigPath
$ComsolConfigHash = Get-Sha256 $ComsolConfigPath
$SourceHashBefore = Get-Sha256 $SourceModel
if ($SourceHashBefore -ne [string]$ComsolConfig.source_model.sha256) {
    throw "The source MPH hash differs from the registered COMSOL contract"
}

$Cases = if ($Case -eq "all") { @("caseP", "caseA") } else { @($Case) }
foreach ($CaseName in $Cases) {
    $CandidateInput = Join-Path $CandidateRoot "$CaseName\candidate_input.h5"
    if (-not (Test-Path -LiteralPath $CandidateInput -PathType Leaf)) {
        throw "Candidate exact-P1 input is missing: $CandidateInput"
    }
}

Push-Location (Split-Path -Parent $JavaSource)
try {
    & $Compiler $JavaSource
    if ($LASTEXITCODE -ne 0) {
        throw "COMSOL Java compilation failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

try {
    if (-not $Execute) {
        Write-Output "M3-C1 COMSOL readiness PASS: shared/config/source/candidate inputs and Java compile"
        Write-Output "Resolved candidate root: $CandidateRoot"
        Write-Output "Resolved output root: $OutputDirectory"
        Write-Output "No COMSOL trajectory study was started; pass -Execute to run the 30 ms matrix."
        return
    }
    if (Test-Path -LiteralPath $OutputDirectory) {
        throw "No-clobber output already exists: $OutputDirectory"
    }
    New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
    $Version = (& $Batch -version 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to read COMSOL version"
    }

    $CampaignCases = [ordered]@{}
    foreach ($CaseName in $Cases) {
        $CandidateInput = Join-Path $CandidateRoot "$CaseName\candidate_input.h5"
        $CaseRoot = Join-Path $OutputDirectory $CaseName
        Push-Location $SolverRoot
        try {
            & uv run --locked python $TablePreparer $CandidateInput $CaseRoot
            if ($LASTEXITCODE -ne 0) {
                throw "Exact-P1 table preparation failed for $CaseName"
            }
        } finally {
            Pop-Location
        }

        $PreparedBefore = Confirm-PreparedTables $CaseRoot
        $RunKeys = @($SharedConfig.matrix.run_keys)
        foreach ($RunKey in $RunKeys) {
            New-Item -ItemType Directory -Path (Join-Path $CaseRoot $RunKey) | Out-Null
        }
        Copy-Item -LiteralPath $SourceModel -Destination (Join-Path $CaseRoot "source_copy.mph")
        $Workflow = $ComsolConfig.workflows.$CaseName
        $SharedWorkflow = $SharedConfig.workflows.$CaseName
        $MaximumDtChargeLipschitz = [double]$SharedConfig.acceptance.maximum_dt_charge_lipschitz
        $RunSpecPath = Join-Path $CaseRoot "run_spec.properties"
        Write-RunSpec -Path $RunSpecPath `
            -CaseName $CaseName -Workflow $Workflow -SharedWorkflow $SharedWorkflow `
            -RunKeys $RunKeys -MaximumDtChargeLipschitz $MaximumDtChargeLipschitz `
            -CommonConfigHash $SharedConfigHash -ComsolConfigHash $ComsolConfigHash

        # COMSOL's Java security policy denies direct Java file reads. Compile an
        # otherwise identical staged source with values taken from the already
        # validated shared and COMSOL contracts. The properties file remains an
        # independently hashable execution receipt for the same values.
        $StagedJava = Join-Path $CaseRoot "RunM3C1Theory100nm30ms.java"
        $JavaText = [IO.File]::ReadAllText($JavaSource)
        $EmbeddedSpec = [ordered]@{
            "__M3C1_CASE_NAME__" = $CaseName
            "__M3C1_PHYSICS_TAG__" = [string]$Workflow.physics_tag
            "__M3C1_BACKGROUND_STUDY__" = [string]$Workflow.background_study
            "__M3C1_BACKGROUND_SOLUTION__" = [string]$Workflow.background_solution
            "__M3C1_SOURCE_DATASET__" = [string]$Workflow.source_dataset
            "__M3C1_PARTICLE_GEOMETRY__" = [string]$Workflow.particle_geometry
            "__M3C1_POSITION_DOF_R__" = [string]$Workflow.position_dofs[0]
            "__M3C1_POSITION_DOF_Z__" = [string]$Workflow.position_dofs[1]
            "__M3C1_CHARGE_STATE__" = [string]$Workflow.charge_state
            "__M3C1_COMMON_CONFIG_SHA256__" = $SharedConfigHash
            "__M3C1_COMSOL_CONFIG_SHA256__" = $ComsolConfigHash
        }
        foreach ($Entry in $EmbeddedSpec.GetEnumerator()) {
            if ([string]$Entry.Value -notmatch '^[A-Za-z0-9_]+$') {
                throw "Unsafe staged Java contract value for $($Entry.Key)"
            }
            $JavaText = $JavaText.Replace([string]$Entry.Key, [string]$Entry.Value)
        }
        $EmbeddedNumericSpec = [ordered]@{
            "__M3C1_STEP_COARSE_S__" = [double]$SharedWorkflow.fixed_rk4_steps_s[0]
            "__M3C1_STEP_MEDIUM_S__" = [double]$SharedWorkflow.fixed_rk4_steps_s[1]
            "__M3C1_STEP_FINE_S__" = [double]$SharedWorkflow.fixed_rk4_steps_s[2]
            "__M3C1_CHARGE_LIPSCHITZ_S_INV__" = [double]$SharedWorkflow.charge_lipschitz_s_inv
            "__M3C1_MAXIMUM_DT_CHARGE_LIPSCHITZ__" = $MaximumDtChargeLipschitz
        }
        foreach ($Entry in $EmbeddedNumericSpec.GetEnumerator()) {
            $Value = Format-InvariantDouble ([double]$Entry.Value)
            if ($Value -notmatch '^[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$') {
                throw "Unsafe staged Java numeric value for $($Entry.Key)"
            }
            $JavaText = $JavaText.Replace([string]$Entry.Key, $Value)
        }
        if ($JavaText -match '__M3C1_[A-Z0-9_]+__') {
            throw "Failed to stage every COMSOL Java contract value"
        }
        [IO.File]::WriteAllText($StagedJava, $JavaText, [Text.UTF8Encoding]::new($false))
        Push-Location $CaseRoot
        try {
            & $Compiler $StagedJava
            if ($LASTEXITCODE -ne 0) {
                throw "COMSOL staged Java compilation failed for $CaseName with exit code $LASTEXITCODE"
            }
        } finally {
            Pop-Location
        }

        $BatchLog = Join-Path $CaseRoot "comsol_process.log"
        $LauncherLog = Join-Path $CaseRoot "comsol_launcher.log"
        Push-Location $CaseRoot
        try {
            & $Batch -inputfile (Join-Path $CaseRoot "RunM3C1Theory100nm30ms.class") `
                -nosave -np 1 -batchlog $BatchLog *> $LauncherLog
            if ($LASTEXITCODE -ne 0) {
                throw "COMSOL 30 ms study failed for $CaseName with exit code $LASTEXITCODE"
            }
        } finally {
            Pop-Location
        }
        if (-not (Select-String -LiteralPath $BatchLog -SimpleMatch `
            "M3C1_30MS|run_pass|case=$CaseName" -Quiet)) {
            throw "COMSOL did not emit the run-pass receipt for $CaseName"
        }

        $PreparedAfter = Confirm-PreparedTables $CaseRoot
        if ($PreparedAfter.receipt_sha256 -ne $PreparedBefore.receipt_sha256 -or
            $PreparedAfter.total_size_bytes -ne $PreparedBefore.total_size_bytes) {
            throw "A prepared exact-P1 table changed during the COMSOL run"
        }
        $PreparedValidation = [ordered]@{
            schema_version = 1
            status = "PASS"
            criterion = "all 26 exact-P1 artifacts match the no-clobber table receipt before and after COMSOL"
            receipt_sha256 = $PreparedBefore.receipt_sha256
            artifact_count = $PreparedBefore.artifact_count
            total_size_bytes = $PreparedBefore.total_size_bytes
            pre_comsol = [ordered]@{ status = "PASS" }
            post_comsol = [ordered]@{ status = "PASS" }
            artifacts = $PreparedAfter.artifacts
        }
        $PreparedValidation | ConvertTo-Json -Depth 8 | Set-Content `
            -LiteralPath (Join-Path $CaseRoot "prepared_table_validation.json") -Encoding utf8

        Remove-Item -LiteralPath (Join-Path $CaseRoot "source_copy.mph")
        Remove-Item -LiteralPath (Join-Path $CaseRoot "RunM3C1Theory100nm30ms.class")
        Remove-Item -LiteralPath (Join-Path $CaseRoot "RunM3C1Theory100nm30ms.class.status") `
            -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $StagedJava
        $SourceHashAfter = Get-Sha256 $SourceModel
        if ($SourceHashAfter -ne $SourceHashBefore) {
            throw "The source MPH changed during the isolated COMSOL run"
        }

        $Runs = Get-StepArtifacts -CaseRoot $CaseRoot -Config $SharedConfig -CaseName $CaseName
        $Receipt = $PreparedAfter.receipt
        $Report = [ordered]@{
            schema_version = 1
            tool_revision = "m3c1_theory_100nm_30ms_comsol_exact_p1_v2"
            status = "COMPLETE"
            workflow = $CaseName
            config_sha256 = $SharedConfigHash
            comsol_config_sha256 = $ComsolConfigHash
            source_sha256_before = $SourceHashBefore
            source_sha256_after = $SourceHashAfter
            comsol_version = $Version
            java_source_sha256 = Get-Sha256 $JavaSource
            process = [ordered]@{
                load = "ModelUtil.loadCopy"
                flags = @("-nosave", "-np", "1")
                model_saved = $false
            }
            field_representation = "canonical_exact_connectivity_p1"
            primitive_source = "prepared_common_p1_tables"
            candidate_input_sha256 = [string]$Receipt.candidate.file_sha256
            candidate_input_content_hash = [string]$Receipt.candidate.content_hash
            prepared_table = [ordered]@{
                status = "PASS"
                receipt_sha256 = $PreparedAfter.receipt_sha256
                validation_path = "prepared_table_validation.json"
                artifact_count = $PreparedAfter.artifact_count
            }
            physics = [ordered]@{
                brownian_active = $false
                saffman_active = $false
                dynamic_charge_active = $true
                deterministic_contributions = @(
                    "electric", "relative_flow_ion_drag", "epstein_drag",
                    "thermophoresis", "lift", "dielectrophoresis", "gravity_buoyancy"
                )
            }
            scope = [ordered]@{
                diameter_m = 1.0e-7
                particle_count = 287
                time_start_s = 0.0
                time_end_s = 0.03
                output_count = 121
            }
            charge_step_safety = [ordered]@{
                status = "PASS"
                run_keys = $RunKeys
                charge_lipschitz_s_inv = [double]$SharedWorkflow.charge_lipschitz_s_inv
                maximum_dt_charge_lipschitz = $MaximumDtChargeLipschitz
                all_steps_admissible = $true
            }
            boundary_semantics = [ordered]@{
                material = "stick"
                gas_inlet_37 = "freeze_hold"
                pump_35 = "disappear_escape"
                axis_5 = "coordinate_axis_not_material_event"
                escape_hit_position = "NOT_DIRECTLY_OBSERVED_WHEN_NAN"
            }
            runs = $Runs
            claim_policy = $ComsolConfig.claim_policy
        }
        $Report | ConvertTo-Json -Depth 10 | Set-Content `
            -LiteralPath (Join-Path $CaseRoot "reference_run_report.json") -Encoding utf8
        Write-ArtifactLedger $CaseRoot
        $CampaignCases[$CaseName] = [ordered]@{
            status = "COMPLETE"
            report_sha256 = Get-Sha256 (Join-Path $CaseRoot "reference_run_report.json")
            artifact_ledger_sha256 = Get-Sha256 (Join-Path $CaseRoot "artifact_hashes.csv")
        }
    }

    $Campaign = [ordered]@{
        schema_version = 1
        tool_revision = "m3c1_theory_100nm_30ms_comsol_exact_p1_v2"
        status = "COMPLETE"
        config_sha256 = $SharedConfigHash
        comsol_config_sha256 = $ComsolConfigHash
        source_sha256_before = $SourceHashBefore
        source_sha256_after = Get-Sha256 $SourceModel
        cases = $CampaignCases
    }
    $Campaign | ConvertTo-Json -Depth 6 | Set-Content `
        -LiteralPath (Join-Path $OutputDirectory "campaign_run_report.json") -Encoding utf8
    Write-Output "M3-C1 COMSOL 30 ms reference completed: $OutputDirectory"
} finally {
    Remove-Item -LiteralPath $JavaClass -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath "$JavaClass.status" -ErrorAction SilentlyContinue
}
