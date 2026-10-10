param(
    [string]$ComsolRoot = "C:\Program Files\COMSOL\COMSOL64\Multiphysics_copy1",
    [string]$ContractPath = "",
    [string]$CandidateInput = "",
    [string]$OutputDirectory = "",
    [ValidateSet("RunnerValidation", "FullPilot", "FinalCampaign")]
    [string]$Mode = "RunnerValidation",
    [string]$CampaignRegistration = "",
    [string]$PilotAuthorization = ""
)

$ErrorActionPreference = "Stop"

function Confirm-PreparedTableArtifacts {
    param([string]$ReceiptPath, [string]$ArtifactRoot)

    $Receipt = Get-Content -LiteralPath $ReceiptPath -Raw | ConvertFrom-Json
    if ([int]$Receipt.schema_version -ne 1 -or
        [string]$Receipt.tool_revision -ne "m3c1_full_physics_common_p1_tables_v2") {
        throw "Prepared-table receipt schema or tool revision is not supported"
    }
    $Entries = @($Receipt.artifacts.PSObject.Properties)
    if ($Entries.Count -ne 27) {
        throw "Prepared-table receipt must bind exactly 27 common-P1 artifacts"
    }
    [long]$TotalSize = 0
    $Verified = @()
    foreach ($Entry in $Entries) {
        $Name = [string]$Entry.Name
        if ([string]::IsNullOrWhiteSpace($Name) -or $Name -in @(".", "..") -or
            [IO.Path]::IsPathRooted($Name) -or [IO.Path]::GetFileName($Name) -ne $Name) {
            throw "Prepared-table receipt contains an unsafe artifact name: $Name"
        }
        $Path = Join-Path $ArtifactRoot $Name
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "Prepared-table artifact is missing: $Name"
        }
        $Size = (Get-Item -LiteralPath $Path).Length
        $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
        if ($Size -ne [long]$Entry.Value.size_bytes -or
            $Hash -ne [string]$Entry.Value.sha256) {
            throw "Prepared-table artifact differs from its receipt: $Name"
        }
        $TotalSize += $Size
        $Verified += [ordered]@{ path = $Name; sha256 = $Hash; size_bytes = $Size }
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

$InvocationDirectory = (Get-Location).ProviderPath
$ComsolRoot = [IO.Path]::GetFullPath($ComsolRoot, $InvocationDirectory)
$SolverRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$RepositoryRoot = (Resolve-Path (Join-Path $SolverRoot "..\..")).Path
$ContractPath = $(
    if ([string]::IsNullOrWhiteSpace($ContractPath)) {
        Join-Path $PSScriptRoot "cases\m3c2_caseA_100nm_stochastic_pilot_v1.json"
    } elseif ([IO.Path]::IsPathRooted($ContractPath)) {
        [IO.Path]::GetFullPath($ContractPath)
    } else {
        [IO.Path]::GetFullPath((Join-Path $RepositoryRoot $ContractPath))
    }
)
$JavaSource = Join-Path $PSScriptRoot "comsol\RunM3C2StochasticCampaign.java"
$ValidationJavaSource = Join-Path $PSScriptRoot `
    "comsol\RunM3C2StochasticRunnerValidation.java"
$ReadbackJavaSource = Join-Path $PSScriptRoot "comsol\ParticleRunReadback.java"
$CoefficientJavaSource = Join-Path $PSScriptRoot "comsol\CommonP1Epstein.java"
$Preparer = Join-Path $PSScriptRoot "prepare_m3c1_common_p1_tables.py"
$Normalizer = Join-Path $PSScriptRoot "normalize_m3c2_comsol_pilot.py"
$Compiler = Join-Path $ComsolRoot "bin\win64\comsolcompile.exe"
$Batch = Join-Path $ComsolRoot "bin\win64\comsolbatch.exe"

foreach ($Required in @(
        $ContractPath,
        $JavaSource,
        $ValidationJavaSource,
        $ReadbackJavaSource,
        $CoefficientJavaSource,
        $Preparer,
        $Normalizer,
        $Compiler,
        $Batch
    )) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required M3-C2A COMSOL pilot input does not exist: $Required"
    }
}

$Contract = Get-Content -LiteralPath $ContractPath -Raw | ConvertFrom-Json
$Workflow = [string]$Contract.scope.workflow
if ($Workflow -notin @("caseA", "caseP")) {
    throw "Unsupported M3-C2 workflow: $Workflow"
}
$HasExplicitCampaign = (
    $null -ne $Contract.PSObject.Properties["campaign"] -and $null -ne $Contract.campaign
)
$Campaign = $Contract.campaign
$Comsol = $Contract.stochastic_physics.comsol
if ($Workflow -eq "caseA" -and $null -eq $Campaign) {
    $Campaign = [pscustomobject]@{
        case_id = "formal_iondrag_theory_consistent/caseA_100nm"
        evaluation_case_id = "M3-C2A_caseA_100nm_common-P1"
        output_slug = "caseA_100nm"
        final_registration_kind = "m3c2_caseA_100nm_final_campaign"
        candidate_case_name_prefix = "m3c2_caseA_100nm"
    }
}
if ($null -eq $Campaign) { throw "M3-C2 contract is missing campaign identity" }
foreach ($Name in @(
        "case_id", "evaluation_case_id", "output_slug", "final_registration_kind",
        "candidate_case_name_prefix"
    )) {
    $Value = [string]$Campaign.$Name
    if ([string]::IsNullOrWhiteSpace($Value) -or $Value -match '[\r\n"]') {
        throw "M3-C2 contract contains an invalid campaign.$Name"
    }
}
if ([string]$Campaign.output_slug -notmatch '^[A-Za-z][A-Za-z0-9_-]*$') {
    throw "M3-C2 campaign output_slug is unsafe"
}

function Get-ComsolValue {
    param([string]$Name, [string]$CaseADefault)
    $Property = $Comsol.PSObject.Properties[$Name]
    if ($null -ne $Property -and -not [string]::IsNullOrWhiteSpace([string]$Property.Value)) {
        return [string]$Property.Value
    }
    if ($Workflow -eq "caseA") { return $CaseADefault }
    throw "Case-P M3-C2 contract is missing stochastic_physics.comsol.$Name"
}

$PhysicsTag = Get-ComsolValue "physics_tag" "fptas"
$BackgroundStudy = Get-ComsolValue "background_study" "stdASf"
$BackgroundStudyStep = Get-ComsolValue "background_study_step" "stat"
$BackgroundSolution = Get-ComsolValue "background_solution" "sol26"
$SharedVariableTag = Get-ComsolValue "shared_variable_tag" "varAS"
$ViscosityExpression = Get-ComsolValue "viscosity_expression" "root.comp1.AS_muB"
$TemperatureExpression = Get-ComsolValue "temperature_expression" "root.comp1.AS_Tg"
$PressureExpression = Get-ComsolValue "pressure_expression" `
    "m3c1_rhog(r,z)*k_B_const*m3c1_Tg(r,z)/1.2753471408396638e-25[kg]"
$SeedParameter = Get-ComsolValue "seed_parameter" "AS_brownian_seed"
if ($null -ne $Comsol -and $null -ne $Comsol.PSObject.Properties["position_expressions"]) {
    $PositionExpressions = @($Comsol.position_expressions)
} else {
    $PositionExpressions = @()
}
if ($PositionExpressions.Count -eq 0 -and $Workflow -eq "caseA") {
    $PositionExpressions = @("q3r", "q3z")
}
if ($null -ne $Comsol -and $null -ne $Comsol.PSObject.Properties["velocity_expressions"]) {
    $VelocityExpressions = @($Comsol.velocity_expressions)
} else {
    $VelocityExpressions = @()
}
if ($VelocityExpressions.Count -eq 0 -and $Workflow -eq "caseA") {
    $VelocityExpressions = @("fptas.vr", "fptas.vz")
}
$ChargeStateExpression = Get-ComsolValue "charge_state_expression" "ZAS"
$ParticleGeometry = Get-ComsolValue "particle_geometry" "pgeom_fptas"
if ($PositionExpressions.Count -ne 2 -or $VelocityExpressions.Count -ne 2) {
    throw "M3-C2 COMSOL position and velocity expressions must contain exactly two entries"
}
$ExpectedSeedAuthority = "$PhysicsTag.bf1.i"
foreach ($Value in @(
        $PhysicsTag, $BackgroundStudy, $BackgroundStudyStep, $BackgroundSolution, $SharedVariableTag,
        $SeedParameter, $ChargeStateExpression, $ParticleGeometry,
        $PositionExpressions[0], $PositionExpressions[1]
    )) {
    if ([string]$Value -notmatch '^[A-Za-z][A-Za-z0-9_]*$') {
        throw "Unsafe COMSOL identifier in M3-C2 contract: $Value"
    }
}
foreach ($Value in $VelocityExpressions) {
    if ([string]$Value -notmatch '^[A-Za-z][A-Za-z0-9_]*\.[A-Za-z][A-Za-z0-9_]*$') {
        throw "Unsafe COMSOL velocity expression in M3-C2 contract: $Value"
    }
}
foreach ($Value in @($ViscosityExpression, $TemperatureExpression, $PressureExpression)) {
    if ([string]$Value -match '[\r\n"]') {
        throw "Unsafe COMSOL expression in M3-C2 contract"
    }
}
$SourceModel = Join-Path $RepositoryRoot $Contract.source_model.path
if ([string]::IsNullOrWhiteSpace($CandidateInput)) {
    $CandidateInput = Join-Path $RepositoryRoot $Contract.common_p1_input.path
} elseif (-not [IO.Path]::IsPathRooted($CandidateInput)) {
    $CandidateInput = Join-Path $RepositoryRoot $CandidateInput
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $Leaf = $(
        if ($Mode -eq "FullPilot") { "$($Campaign.output_slug)_comsol_pilot_v1" }
        elseif ($Mode -eq "FinalCampaign") { "$($Campaign.output_slug)_comsol_final_campaign_v1" }
        else { "$($Campaign.output_slug)_comsol_runner_validation_v1" }
    )
    $OutputDirectory = Join-Path $SolverRoot "_out_m3c2\$Leaf"
} elseif (-not [IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $SolverRoot $OutputDirectory
}
$CandidateInput = [IO.Path]::GetFullPath($CandidateInput)
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if ($Mode -eq "FinalCampaign") {
    if ([string]::IsNullOrWhiteSpace($CampaignRegistration)) {
        throw "FinalCampaign requires -CampaignRegistration"
    }
    if (-not [IO.Path]::IsPathRooted($CampaignRegistration)) {
        $CampaignRegistration = Join-Path $RepositoryRoot $CampaignRegistration
    }
    $CampaignRegistration = [IO.Path]::GetFullPath($CampaignRegistration)
    if (-not (Test-Path -LiteralPath $CampaignRegistration -PathType Leaf)) {
        throw "M3-C2A final campaign registration does not exist: $CampaignRegistration"
    }
} elseif (-not [string]::IsNullOrWhiteSpace($CampaignRegistration)) {
    throw "-CampaignRegistration is only valid with -Mode FinalCampaign"
}
$PilotAuthorizationRequired = $Mode -eq "FullPilot" -and $HasExplicitCampaign
$PilotAuthorizationRelative = ""
if ($PilotAuthorizationRequired) {
    if ([string]::IsNullOrWhiteSpace($PilotAuthorization)) {
        throw "FullPilot with an explicit campaign requires -PilotAuthorization"
    }
    if (-not [IO.Path]::IsPathRooted($PilotAuthorization)) {
        $PilotAuthorization = Join-Path $RepositoryRoot $PilotAuthorization
    }
    $PilotAuthorization = [IO.Path]::GetFullPath($PilotAuthorization)
    $RepositoryPrefix = $RepositoryRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $PilotAuthorization.StartsWith(
            $RepositoryPrefix,
            [StringComparison]::OrdinalIgnoreCase
        )) {
        throw "M3-C2 pilot authorization must be inside the repository"
    }
    if (-not (Test-Path -LiteralPath $PilotAuthorization -PathType Leaf)) {
        throw "M3-C2 pilot authorization does not exist: $PilotAuthorization"
    }
    $PilotAuthorizationRelative = [IO.Path]::GetRelativePath(
        $RepositoryRoot,
        $PilotAuthorization
    ).Replace('\', '/')
} elseif (-not [string]::IsNullOrWhiteSpace($PilotAuthorization)) {
    throw "-PilotAuthorization is only valid with -Mode FullPilot and an explicit campaign"
}

foreach ($Required in @($SourceModel, $CandidateInput)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Locked M3-C2A input does not exist: $Required"
    }
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "M3-C2A COMSOL output already exists: $OutputDirectory"
}

$SourceHashBefore = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel
).Hash.ToLowerInvariant()
$CandidateHash = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $CandidateInput
).Hash.ToLowerInvariant()
if ($SourceHashBefore -ne [string]$Contract.source_model.sha256) {
    throw "Source MPH differs from the locked M3-C2A execution contract"
}
if ($CandidateHash -ne [string]$Contract.common_p1_input.file_sha256) {
    throw "Common-P1 HDF5 differs from the locked M3-C2A execution contract"
}
if ([string]$Contract.stochastic_physics.comsol.random_number_args -ne "UserDefined" -or
    [string]$Contract.stochastic_physics.comsol.sole_seed_authority -ne $ExpectedSeedAuthority) {
    throw "M3-C2A execution contract does not bind the required COMSOL seed semantics"
}
$VersionText = (& $Batch -version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $VersionText -notmatch "6\.4\.0\.429") {
    throw "COMSOL version does not match the locked 6.4.0.429 environment: $VersionText"
}

$ContractHash = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $ContractPath
).Hash.ToLowerInvariant()
$JavaHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $JavaSource).Hash.ToLowerInvariant()
$ValidationJavaHash = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $ValidationJavaSource
).Hash.ToLowerInvariant()
$PreparerHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Preparer).Hash.ToLowerInvariant()
$NormalizerHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Normalizer).Hash.ToLowerInvariant()
$ReadbackJavaHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ReadbackJavaSource).Hash.ToLowerInvariant()
$CoefficientJavaHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $CoefficientJavaSource).Hash.ToLowerInvariant()
$ActualReceiptReader = Join-Path $PSScriptRoot "actual_run_receipt.py"
$BoundaryResponseMapping = Join-Path $PSScriptRoot "boundary_response_mapping.py"
$ActualReceiptReaderHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ActualReceiptReader).Hash.ToLowerInvariant()
$BoundaryResponseMappingHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $BoundaryResponseMapping).Hash.ToLowerInvariant()
$RunnerHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath).Hash.ToLowerInvariant()
$RegistrationHash = $(
    if ($Mode -eq "FinalCampaign") {
        (Get-FileHash -Algorithm SHA256 -LiteralPath $CampaignRegistration).Hash.ToLowerInvariant()
    } else { "" }
)
$PilotAuthorizationHash = $(
    if ($PilotAuthorizationRequired) {
        (Get-FileHash -Algorithm SHA256 -LiteralPath $PilotAuthorization).Hash.ToLowerInvariant()
    } else { "" }
)

Push-Location $SolverRoot
try {
    & uv run --locked python -m tools.vv.comsol.prepare_m3c1_common_p1_tables $CandidateInput $OutputDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "M3-C2A common-P1 table preparation failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

$TableReceipt = Join-Path $OutputDirectory "common_p1_table_receipt.json"
$TableValidationBefore = Confirm-PreparedTableArtifacts $TableReceipt $OutputDirectory
$RequestPath = Join-Path $OutputDirectory "m3c2_pilot_request.csv"
$ExecutionRequestPath = Join-Path $OutputDirectory "m3c2_execution_request.json"
Push-Location $SolverRoot
try {
    $PrepareArguments = @(
        "-m", "tools.vv.comsol.normalize_m3c2_comsol_pilot",
        "prepare-request",
        $OutputDirectory,
        "--mode",
        $Mode,
        "--contract",
        $ContractPath
    )
    if ($Mode -eq "FinalCampaign") {
        $PrepareArguments += @("--registration", $CampaignRegistration)
    }
    if ($PilotAuthorizationRequired) {
        $PrepareArguments += @("--pilot-authorization", $PilotAuthorization)
    }
    & uv run --locked python @PrepareArguments
    if ($LASTEXITCODE -ne 0) {
        throw "M3-C2A execution-request preparation failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
$ExecutionRequest = Get-Content -LiteralPath $ExecutionRequestPath -Raw | ConvertFrom-Json
if ([string]$ExecutionRequest.mode -ne $Mode -or
    [string]$ExecutionRequest.participant -ne "comsol") {
    throw "Prepared M3-C2A execution request has the wrong identity"
}
if ([string]$ExecutionRequest.campaign_identity.case_id -ne [string]$Campaign.case_id -or
    [string]$ExecutionRequest.comsol_semantics.physics_tag -ne $PhysicsTag -or
    [string]$ExecutionRequest.comsol_semantics.sole_seed_authority -ne $ExpectedSeedAuthority) {
    throw "Prepared M3-C2 execution request differs from the contract semantics"
}
$RequestAuthorizationProperty = $ExecutionRequest.PSObject.Properties["pilot_authorization"]
if ($PilotAuthorizationRequired) {
    if ($null -eq $RequestAuthorizationProperty -or
        $null -eq $RequestAuthorizationProperty.Value -or
        @($RequestAuthorizationProperty.Value.PSObject.Properties).Count -ne 2 -or
        [string]$RequestAuthorizationProperty.Value.path -cne $PilotAuthorizationRelative -or
        [string]$RequestAuthorizationProperty.Value.sha256 -cne $PilotAuthorizationHash) {
        throw "Prepared M3-C2 execution request differs from the pilot authorization"
    }
} elseif ($null -ne $RequestAuthorizationProperty -and
    $null -ne $RequestAuthorizationProperty.Value) {
    throw "Prepared M3-C2 execution request contains an unexpected pilot authorization"
}
$Seeds = @($ExecutionRequest.seeds | ForEach-Object { [int]$_ })
$StepsNs = @($ExecutionRequest.steps_ns | ForEach-Object { [long]$_ })
$NormalizedManifest = [string]$ExecutionRequest.normalized_manifest
$RequestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $RequestPath).Hash.ToLowerInvariant()
$ExecutionRequestHash = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $ExecutionRequestPath
).Hash.ToLowerInvariant()
$RequestRows = @(Import-Csv -LiteralPath $RequestPath)
$OutputPrefix = $OutputDirectory.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
$JavaRequestRows = @()
foreach ($Request in $RequestRows) {
    $RequestSeed = [int]$Request.seed
    $RequestStepNs = [long]$Request.step_ns
    $StepLabel = $(
        if ($RequestStepNs % 1000 -eq 0) { "dt_$($RequestStepNs / 1000)us" }
        else { "dt_${RequestStepNs}ns" }
    )
    $ExpectedDirectory = "levels/$StepLabel/replicas/seed_$RequestSeed"
    if ([string]$Request.relative_directory -cne $ExpectedDirectory) {
        throw "M3-C2A request directory does not match its seed and step"
    }
    $RequestedDirectory = [IO.Path]::GetFullPath(
        (Join-Path $OutputDirectory $Request.relative_directory)
    )
    if (-not $RequestedDirectory.StartsWith(
            $OutputPrefix,
            [StringComparison]::OrdinalIgnoreCase
        )) {
        throw "M3-C2A request directory escaped the output root"
    }
    New-Item -ItemType Directory `
        -Path $RequestedDirectory -Force | Out-Null
    $JavaRequestRows += '    "{0},{1},{2}"' -f `
        $RequestSeed, $RequestStepNs, $ExpectedDirectory
}
if ($JavaRequestRows.Count -eq 0 -or $JavaRequestRows.Count -gt 32) {
    throw "M3-C2A request authority must contain between one and 32 rows"
}

$SourceCopy = Join-Path $OutputDirectory "source_copy.mph"
$StagedJava = Join-Path $OutputDirectory "RunM3C2StochasticCampaign.java"
$StagedValidationJava = Join-Path $OutputDirectory `
    "RunM3C2StochasticRunnerValidation.java"
$StagedRequestJava = Join-Path $OutputDirectory `
    "RunM3C2StochasticRequest.java"
$StagedReadbackJava = Join-Path $OutputDirectory "ParticleRunReadback.java"
$StagedCoefficientJava = Join-Path $OutputDirectory "CommonP1Epstein.java"
$StagedContract = Join-Path $OutputDirectory ([IO.Path]::GetFileName($ContractPath))
$StagedPreparer = Join-Path $OutputDirectory ([IO.Path]::GetFileName($Preparer))
$StagedNormalizer = Join-Path $OutputDirectory ([IO.Path]::GetFileName($Normalizer))
$StagedRegistration = $(
    if ($Mode -eq "FinalCampaign") {
        Join-Path $OutputDirectory ([IO.Path]::GetFileName($CampaignRegistration))
    } else { "" }
)
$StagedPilotAuthorization = $(
    if ($PilotAuthorizationRequired) {
        Join-Path $OutputDirectory ([IO.Path]::GetFileName($PilotAuthorization))
    } else { "" }
)
if ($PilotAuthorizationRequired -and
    $StagedPilotAuthorization -in @($StagedContract, $StagedPreparer, $StagedNormalizer)) {
    throw "Pilot authorization filename collides with another staged artifact"
}
$FullClassFile = Join-Path $OutputDirectory "RunM3C2StochasticCampaign.class"
$ValidationClassFile = Join-Path $OutputDirectory `
    "RunM3C2StochasticRunnerValidation.class"
$RequestClassFile = Join-Path $OutputDirectory "RunM3C2StochasticRequest.class"
$ClassFile = $(if ($Mode -eq "RunnerValidation") { $ValidationClassFile } else { $FullClassFile })
$ClassStatus = "$ClassFile.status"
$CompiledClassFiles = @($FullClassFile, $ValidationClassFile, $RequestClassFile,
    (Join-Path $OutputDirectory "ParticleRunReadback.class"),
    (Join-Path $OutputDirectory "CommonP1Epstein.class"))
$ProcessLog = Join-Path $OutputDirectory "comsol_process.log"
$ProcessErrorLog = Join-Path $OutputDirectory "comsol_process_error.log"
$ProcessMetricsPath = Join-Path $OutputDirectory "comsol_process_metrics.json"
$BatchLog = Join-Path $OutputDirectory "comsol_batch.log"
$InternalLog = Join-Path $OutputDirectory "comsol_internal.log"
$ComsolPreferencesDirectory = Join-Path $OutputDirectory ".comsol_preferences"
$StatusPath = Join-Path $OutputDirectory "run_status.json"
$Succeeded = $false
$FailureText = ""
$RequestJavaHash = ""

try {
    Copy-Item -LiteralPath $SourceModel -Destination $SourceCopy
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $SourceCopy).Hash.ToLowerInvariant() `
        -ne $SourceHashBefore) {
        throw "Isolated source copy differs from the audited source MPH"
    }
    Copy-Item -LiteralPath $JavaSource -Destination $StagedJava
    Copy-Item -LiteralPath $ValidationJavaSource -Destination $StagedValidationJava
    Copy-Item -LiteralPath $ReadbackJavaSource -Destination $StagedReadbackJava
    Copy-Item -LiteralPath $CoefficientJavaSource -Destination $StagedCoefficientJava
    Copy-Item -LiteralPath $ContractPath -Destination $StagedContract
    Copy-Item -LiteralPath $Preparer -Destination $StagedPreparer
    Copy-Item -LiteralPath $Normalizer -Destination $StagedNormalizer
    if ($Mode -eq "FinalCampaign") {
        Copy-Item -LiteralPath $CampaignRegistration -Destination $StagedRegistration
    }
    if ($PilotAuthorizationRequired) {
        Copy-Item -LiteralPath $PilotAuthorization -Destination $StagedPilotAuthorization
    }
    $StagedLocks = @(
            @($StagedJava, $JavaHash),
            @($StagedValidationJava, $ValidationJavaHash),
            @($StagedContract, $ContractHash),
            @($StagedPreparer, $PreparerHash),
            @($StagedNormalizer, $NormalizerHash),
            @($StagedReadbackJava, $ReadbackJavaHash),
            @($StagedCoefficientJava, $CoefficientJavaHash)
        )
    if ($Mode -eq "FinalCampaign") {
        $StagedLocks += , @($StagedRegistration, $RegistrationHash)
    }
    if ($PilotAuthorizationRequired) {
        $StagedLocks += , @($StagedPilotAuthorization, $PilotAuthorizationHash)
    }
    foreach ($Lock in $StagedLocks) {
        $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Lock[0]).Hash.ToLowerInvariant()
        if ($Actual -ne $Lock[1]) { throw "A staged M3-C2A tool differs from its source" }
    }

    $RequestRowsJava = $JavaRequestRows -join ",`r`n"
    $RequestJavaText = @"
/** Generated from the validated, hash-locked M3-C2 execution request. */
public final class RunM3C2StochasticRequest {
  private RunM3C2StochasticRequest() {}

  static String[] rows() {
    return new String[] {
$RequestRowsJava
    };
  }

  static String mode() {
    return "$Mode";
  }

  static boolean isCaseP() { return "$Workflow".equals("caseP"); }
  static String caseSlug() { return "$($Campaign.output_slug)"; }
  static String physicsTag() { return "$PhysicsTag"; }
  static String backgroundStudy() { return "$BackgroundStudy"; }
  static String backgroundStudyStep() { return "$BackgroundStudyStep"; }
  static String backgroundSolution() { return "$BackgroundSolution"; }
  static String sharedVariableTag() { return "$SharedVariableTag"; }
  static String viscosityExpression() { return "$ViscosityExpression"; }
  static String temperatureExpression() { return "$TemperatureExpression"; }
  static String pressureExpression() { return "$PressureExpression"; }
  static String seedParameter() { return "$SeedParameter"; }
  static String positionRExpression() { return "$($PositionExpressions[0])"; }
  static String positionZExpression() { return "$($PositionExpressions[1])"; }
  static String velocityRExpression() { return "$($VelocityExpressions[0])"; }
  static String velocityZExpression() { return "$($VelocityExpressions[1])"; }
  static String chargeStateExpression() { return "$ChargeStateExpression"; }
  static String particleGeometry() { return "$ParticleGeometry"; }
}
"@
    [IO.File]::WriteAllText(
        $StagedRequestJava,
        $RequestJavaText,
        [Text.UTF8Encoding]::new($false)
    )
    $RequestJavaHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $StagedRequestJava
    ).Hash.ToLowerInvariant()

    New-Item -ItemType Directory -Path $ComsolPreferencesDirectory | Out-Null

    Push-Location $OutputDirectory
    try {
        & $Compiler $StagedCoefficientJava
        if ($LASTEXITCODE -ne 0) { throw "Common-P1 coefficient Java compilation failed" }
        & $Compiler -classpathadd $OutputDirectory $StagedReadbackJava
        if ($LASTEXITCODE -ne 0) { throw "Actual readback Java compilation failed" }
        & $Compiler $StagedRequestJava
        if ($LASTEXITCODE -ne 0 -or
            -not (Test-Path -LiteralPath $RequestClassFile -PathType Leaf)) {
            throw "COMSOL M3-C2A request compilation failed with exit code $LASTEXITCODE"
        }
        & $Compiler -classpathadd $OutputDirectory $StagedJava
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $FullClassFile -PathType Leaf)) {
            throw "COMSOL M3-C2A Java compilation failed with exit code $LASTEXITCODE"
        }
        if ($Mode -eq "RunnerValidation") {
            & $Compiler -classpathadd $OutputDirectory $StagedValidationJava
            if ($LASTEXITCODE -ne 0 -or
                -not (Test-Path -LiteralPath $ValidationClassFile -PathType Leaf)) {
                throw "COMSOL M3-C2A validation entry compilation failed with exit code $LASTEXITCODE"
            }
        }
        $BatchArguments = @(
            "-inputfile", "`"$ClassFile`"",
            "-nosave",
            "-error", "on",
            "-prefsdir", "`"$ComsolPreferencesDirectory`"",
            "-np", "1",
            "-batchlog", "`"$BatchLog`""
        )
        $BatchStopwatch = [Diagnostics.Stopwatch]::StartNew()
        $BatchProcess = Start-Process -FilePath $Batch -ArgumentList $BatchArguments `
            -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $ProcessLog -RedirectStandardError $ProcessErrorLog
        [long]$PeakRssBytes = 0
        while (-not $BatchProcess.HasExited) {
            try {
                $BatchProcess.Refresh()
                $PeakRssBytes = [Math]::Max(
                    $PeakRssBytes,
                    [Math]::Max(
                        [long]$BatchProcess.WorkingSet64,
                        [long]$BatchProcess.PeakWorkingSet64
                    )
                )
            } catch {
                # The process may exit between HasExited and Refresh.
            }
            Start-Sleep -Milliseconds 250
        }
        $BatchProcess.WaitForExit()
        $BatchStopwatch.Stop()
        try {
            $BatchProcess.Refresh()
            $PeakRssBytes = [Math]::Max(
                $PeakRssBytes,
                [long]$BatchProcess.PeakWorkingSet64
            )
        } catch {
            # The sampled scalar peak remains authoritative after process exit.
        }
        $InternalLogSource = Get-ChildItem `
            -LiteralPath (Join-Path $ComsolPreferencesDirectory "logs") `
            -Filter "batch*.log" -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -notlike "*_render.log" } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($null -ne $InternalLogSource) {
            Copy-Item -LiteralPath $InternalLogSource.FullName -Destination $InternalLog
        }
        [ordered]@{
            schema_version = 1
            measurement = "tracked_comsolbatch_process_bounded_250ms_poll"
            wall_time_s = $BatchStopwatch.Elapsed.TotalSeconds
            peak_rss_bytes = $PeakRssBytes
            sample_history_retained = $false
            process_id = $BatchProcess.Id
            exit_code = $BatchProcess.ExitCode
        } | ConvertTo-Json -Depth 3 | Set-Content `
            -LiteralPath $ProcessMetricsPath -Encoding utf8
        if ($BatchProcess.ExitCode -ne 0) {
            throw "COMSOL M3-C2A run failed with exit code $($BatchProcess.ExitCode)"
        }
    } finally {
        Pop-Location
    }

    $ClassStatusText = $(
        if (Test-Path -LiteralPath $ClassStatus -PathType Leaf) {
            (Get-Content -LiteralPath $ClassStatus -Raw).Trim()
        } else {
            "not emitted"
        }
    )
    if (-not (Test-Path -LiteralPath $ProcessLog -PathType Leaf) -or
        (Get-Item -LiteralPath $ProcessLog).Length -eq 0) {
        throw "COMSOL returned without a nonempty M3-C2A process log; class status: $ClassStatusText"
    }
    if (Select-String -LiteralPath $ProcessLog, $BatchLog, $ProcessErrorLog -Pattern `
            'M3C2_COMSOL\|fatal\||Error running java class\.|/\*+Error\*+/') {
        throw "COMSOL Java program reported a fatal M3-C2A exception"
    }
    $ExpectedCompletion = "M3C2_COMSOL|run_pass|case=$($Campaign.output_slug)|requests=$($RequestRows.Count)|mode=$Mode|time_end_s=0.03|output_times=121|model_saved=false"
    if (@(Select-String -LiteralPath $ProcessLog -SimpleMatch $ExpectedCompletion).Count -ne 1) {
        throw "COMSOL did not emit the unique registered execution completion record"
    }
    foreach ($Request in $RequestRows) {
        $Raw = Join-Path (Join-Path $OutputDirectory $Request.relative_directory) `
            "trajectory_raw_wide.csv"
        if (-not (Test-Path -LiteralPath $Raw -PathType Leaf) -or
            (Get-Item -LiteralPath $Raw).Length -eq 0) {
            throw "COMSOL did not emit expected M3-C2A trajectory: $Raw"
        }
    }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $RequestPath).Hash.ToLowerInvariant() `
            -ne $RequestHash -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $ExecutionRequestPath).Hash.ToLowerInvariant() `
            -ne $ExecutionRequestHash) {
        throw "M3-C2A registered execution request changed during COMSOL execution"
    }

    $SourceHashAfter = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel
    ).Hash.ToLowerInvariant()
    if ($SourceHashAfter -ne $SourceHashBefore) {
        throw "The audited source MPH changed during the loadCopy/no-save run"
    }
    foreach ($Lock in @($StagedLocks) + @(
        @($ActualReceiptReader, $ActualReceiptReaderHash),
        @($BoundaryResponseMapping, $BoundaryResponseMappingHash)
    )) {
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Lock[0]).Hash.ToLowerInvariant() -ne $Lock[1]) {
            throw "A readback or execution producer changed during the run"
        }
    }
    $TableValidationAfter = Confirm-PreparedTableArtifacts $TableReceipt $OutputDirectory
    if ($TableValidationBefore.receipt_sha256 -ne $TableValidationAfter.receipt_sha256 -or
        $TableValidationBefore.total_size_bytes -ne $TableValidationAfter.total_size_bytes) {
        throw "Prepared common-P1 artifacts changed during the COMSOL run"
    }
    [ordered]@{
        schema_version = 1
        status = "PASS"
        pre_comsol = $TableValidationBefore
        post_comsol = $TableValidationAfter
    } | ConvertTo-Json -Depth 7 | Set-Content `
        -LiteralPath (Join-Path $OutputDirectory "prepared_table_validation.json") -Encoding utf8

    Push-Location $SolverRoot
    try {
        & uv run --locked python -m tools.vv.comsol.normalize_m3c2_comsol_pilot normalize $OutputDirectory
        if ($LASTEXITCODE -ne 0) {
            throw "M3-C2A COMSOL normalization failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
    if (-not (Test-Path -LiteralPath (Join-Path $OutputDirectory $NormalizedManifest) `
            -PathType Leaf)) {
        throw "M3-C2A normalizer did not emit the registered participant manifest"
    }

    [ordered]@{
        schema_version = 1
        tool_revision = "m3c2_comsol_campaign_runner_v5"
        status = $(
            if ($Mode -eq "FullPilot") { "COMPLETE_FULL_PILOT_NOT_EVALUATED" }
            elseif ($Mode -eq "FinalCampaign") { "COMPLETE_FINAL_CAMPAIGN_NOT_EVALUATED" }
            else { "COMPLETE_RUNNER_VALIDATION_NOT_SCIENTIFIC_PILOT" }
        )
        mode = $Mode
        campaign_identity = [ordered]@{
            case_id = [string]$Campaign.case_id
            evaluation_case_id = [string]$Campaign.evaluation_case_id
            output_slug = [string]$Campaign.output_slug
            final_registration_kind = [string]$Campaign.final_registration_kind
            candidate_case_name_prefix = [string]$Campaign.candidate_case_name_prefix
        }
        source_mph = [IO.Path]::GetFullPath($SourceModel)
        source_sha256_before = $SourceHashBefore
        source_sha256_after = $SourceHashAfter
        source_unchanged = $true
        source_load_policy = "ModelUtil.loadCopy"
        process_policy = "comsolbatch_-nosave_-error_on_-isolated_prefsdir_-np_1"
        model_saved = $false
        common_p1_input = $CandidateInput
        common_p1_input_sha256 = $CandidateHash
        common_p1_content_hash = [string]$Contract.common_p1_input.content_hash
        contract_sha256 = $ContractHash
        runner_sha256 = $RunnerHash
        java_sha256 = $JavaHash
        validation_entry_java_sha256 = $ValidationJavaHash
        generated_request_java_sha256 = $RequestJavaHash
        preparer_sha256 = $PreparerHash
        normalizer_sha256 = $NormalizerHash
        readback_java_sha256 = $ReadbackJavaHash
        coefficient_java_sha256 = $CoefficientJavaHash
        actual_receipt_reader_sha256 = $ActualReceiptReaderHash
        boundary_response_mapping_sha256 = $BoundaryResponseMappingHash
        actual_binding_artifacts = @(
            foreach ($Request in $RequestRows) {
                foreach ($Name in @("actual_binding_receipt.json", "fdt_probe_raw.csv")) {
                    $ArtifactRelativePath = Join-Path $Request.relative_directory $Name
                    [ordered]@{ path = $ArtifactRelativePath; sha256 = (
                        Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $OutputDirectory $ArtifactRelativePath)
                    ).Hash.ToLowerInvariant() }
                }
            }
        )
        execution_request_sha256 = $ExecutionRequestHash
        request_csv_sha256 = $RequestHash
        campaign_registration = $(
            if ($Mode -eq "FinalCampaign") {
                [ordered]@{
                    filename = [IO.Path]::GetFileName($CampaignRegistration)
                    sha256 = $RegistrationHash
                }
            } else { $null }
        )
        pilot_authorization = $(
            if ($PilotAuthorizationRequired) {
                [ordered]@{
                    path = $PilotAuthorizationRelative
                    sha256 = $PilotAuthorizationHash
                }
            } else { $null }
        )
        comsol_version = $VersionText
        batch_completion = [ordered]@{
            status = "COMPLETE"
            process_exit_code = $BatchProcess.ExitCode
            expected_completion_record = $ExpectedCompletion
            native_error_record_absent = $true
            class_status_raw = $ClassStatusText
            class_status_authority = "non_authoritative_for_6_4_class_input_verified_by_controls"
            completion_authority = "process_log_and_registered_artifacts_and_normalization"
        }
        seeds = $Seeds
        fixed_rk4_steps_s = @($StepsNs | ForEach-Object { $_ * 1.0e-9 })
        output_schedule = $Contract.scope.output_schedule_segments
        normalized_manifest = $NormalizedManifest
        scientific_claim = "NOT_EVALUATED"
        generated_utc = [DateTime]::UtcNow.ToString("o")
    } | ConvertTo-Json -Depth 8 | Set-Content `
        -LiteralPath (Join-Path $OutputDirectory "run_receipt.json") -Encoding utf8
    $Succeeded = $true
} catch {
    $FailureText = $_.Exception.Message
    throw
} finally {
    if (Test-Path -LiteralPath $ComsolPreferencesDirectory -PathType Container) {
        $PreferencesFullPath = [IO.Path]::GetFullPath($ComsolPreferencesDirectory)
        $OutputPrefix = $OutputDirectory.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
        if (-not $PreferencesFullPath.StartsWith(
                $OutputPrefix,
                [StringComparison]::OrdinalIgnoreCase
            ) -or [IO.Path]::GetFileName($PreferencesFullPath) -ne ".comsol_preferences") {
            throw "Refusing to remove an unexpected COMSOL preferences directory"
        }
        Remove-Item -LiteralPath $PreferencesFullPath -Recurse -Force -ErrorAction SilentlyContinue
    }
    foreach ($Temporary in @($SourceCopy, $ClassStatus) + $CompiledClassFiles) {
        Remove-Item -LiteralPath $Temporary -Force -ErrorAction SilentlyContinue
    }
    [ordered]@{
        status = $(if ($Succeeded) { "COMPLETE" } else { "INCOMPLETE" })
        mode = $Mode
        failure = $FailureText
        source_copy_retained = Test-Path -LiteralPath $SourceCopy
        compiled_class_retained = @(
            $CompiledClassFiles | Where-Object { Test-Path -LiteralPath $_ }
        ).Count -gt 0
        generated_utc = [DateTime]::UtcNow.ToString("o")
    } | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $StatusPath -Encoding utf8
}

if (-not $Succeeded) { throw "M3-C2A COMSOL $Mode run did not complete" }

$OutputRoot = [IO.Path]::GetFullPath($OutputDirectory)
Get-ChildItem -LiteralPath $OutputRoot -Recurse -File |
    Where-Object { $_.Name -ne "artifact_hashes.csv" } |
    Sort-Object FullName |
    ForEach-Object {
        $Prefix = $OutputRoot + [IO.Path]::DirectorySeparatorChar
        if (-not $_.FullName.StartsWith($Prefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Artifact escaped M3-C2A output root: $($_.FullName)"
        }
        [pscustomobject][ordered]@{
            path = $_.FullName.Substring($Prefix.Length).Replace("\", "/")
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            bytes = $_.Length
        }
    } | Export-Csv -LiteralPath (Join-Path $OutputRoot "artifact_hashes.csv") `
        -NoTypeInformation -Encoding utf8

Write-Output "M3-C2A COMSOL $Mode completed: $OutputDirectory"
