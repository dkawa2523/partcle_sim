param(
    [string]$ComsolRoot = "C:\Program Files\COMSOL\COMSOL64\Multiphysics_copy1",
    [string]$DatasetRoot = "",
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$InvocationDirectory = (Get-Location).ProviderPath
$ComsolRoot = [IO.Path]::GetFullPath($ComsolRoot, $InvocationDirectory)
$SolverRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$RepositoryRoot = (Resolve-Path (Join-Path $SolverRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
    $DatasetRoot = Join-Path $RepositoryRoot "model_dataset\cf4_o2_etch_caseA_nonlinear_sass"
} else {
    $DatasetRoot = [IO.Path]::GetFullPath($DatasetRoot, $InvocationDirectory)
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $SolverRoot "evidence\m3c2\model_semantics_probe_v1"
} else {
    $OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory, $InvocationDirectory)
}

$Compiler = Join-Path $ComsolRoot "bin\win64\comsolcompile.exe"
$Batch = Join-Path $ComsolRoot "bin\win64\comsolbatch.exe"
$JavaSource = Join-Path $PSScriptRoot "comsol\InspectM3C2ModelSemantics.java"
$RunnerSource = $PSCommandPath
$SourceModel = Join-Path $DatasetRoot (
    "model\icp_rf_bias_cf4_o2_si_etching_caseP_caseA_SASS_" +
    "formal_iondrag_theory_consistent_10_30_100nm.mph"
)
$ExpectedSourceSha256 = "3BBF08E3469758313EAC5DE473A7A0DD4CC9A6F72C9722393229F0B856E9B524"
$ExpectedComsolVersion = "COMSOL Multiphysics 6.4.0.429"
$Prefix = "M3C2_MODEL_SEMANTICS_JSON|"

foreach ($RequiredPath in @($Compiler, $Batch, $JavaSource, $RunnerSource, $SourceModel)) {
    if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
        throw "Required M3-C2 model-semantics input does not exist: $RequiredPath"
    }
}

$JavaText = Get-Content -LiteralPath $JavaSource -Raw
if (([regex]::Matches($JavaText, "ModelUtil\.loadCopy\s*\(")).Count -ne 1) {
    throw "The M3-C2 inspector must contain exactly one ModelUtil.loadCopy call"
}
foreach ($ForbiddenCall in @(".run(", ".runNoGen(", ".save(")) {
    if ($JavaText.Contains($ForbiddenCall)) {
        throw "The M3-C2 inspector contains a forbidden model mutation call"
    }
}

if (Test-Path -LiteralPath $OutputDirectory) {
    if (@(Get-ChildItem -LiteralPath $OutputDirectory -Force).Count -ne 0) {
        throw "Output directory already exists and is not empty: $OutputDirectory"
    }
} else {
    New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
}

$SourceHashBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel).Hash
$JavaHashBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $JavaSource).Hash
$RunnerHashBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $RunnerSource).Hash
if ($SourceHashBefore -ne $ExpectedSourceSha256) {
    throw "Locked source MPH hash mismatch: expected $ExpectedSourceSha256, found $SourceHashBefore"
}

$VersionText = & $Batch -version 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read COMSOL version; exit code $LASTEXITCODE"
}
if ((@($VersionText)[0]).Trim() -ne $ExpectedComsolVersion) {
    throw "COMSOL version mismatch: expected '$ExpectedComsolVersion', found '$(@($VersionText)[0])'"
}
$VersionText | Set-Content -LiteralPath (Join-Path $OutputDirectory "comsol_version.txt") -Encoding utf8

$ClassFile = Join-Path (Split-Path -Parent $JavaSource) "InspectM3C2ModelSemantics.class"
$ClassStatus = "$ClassFile.status"
$LauncherLog = Join-Path $OutputDirectory "comsol_launcher.log"
$ProcessOutputLog = Join-Path $OutputDirectory "comsol_process_output.log"
$Jsonl = Join-Path $OutputDirectory "model_semantics.jsonl"

Push-Location (Split-Path -Parent $JavaSource)
try {
    Remove-Item -LiteralPath $ClassFile, $ClassStatus -ErrorAction SilentlyContinue
    & $Compiler $JavaSource
    if ($LASTEXITCODE -ne 0) {
        throw "COMSOL Java compilation failed with exit code $LASTEXITCODE"
    }
    $CompiledClassSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ClassFile).Hash

    Push-Location (Split-Path -Parent $SourceModel)
    try {
        & $Batch -inputfile $ClassFile -nosave -np 1 -batchlog $LauncherLog *> $ProcessOutputLog
        if ($LASTEXITCODE -ne 0) {
            throw "COMSOL read-only M3-C2 inspection failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }

    $ExtractedLines = @(
        Get-Content -LiteralPath $ProcessOutputLog | ForEach-Object {
            if ($_ -match "M3C2_MODEL_SEMANTICS_JSON\|(\{.*\})(?: \[com\.comsol\.util\])?$") {
                "$Prefix$($Matches[1])"
            }
        } | Select-Object -Unique
    )
    if ($ExtractedLines.Count -eq 0) {
        throw "No machine-readable M3-C2 records were extracted from the COMSOL server log"
    }
    $ExtractedLines | Set-Content -LiteralPath $Jsonl -Encoding utf8

    $Records = @(
        $ExtractedLines | ForEach-Object {
            $_.Substring($Prefix.Length) | ConvertFrom-Json
        }
    )
    $AuditPass = @($Records | Where-Object { $_.type -eq "audit_pass" })
    if ($AuditPass.Count -ne 1 -or $AuditPass[0].study_run -ne "false" -or
        $AuditPass[0].model_save -ne "false" -or $AuditPass[0].load_copy -ne "true") {
        throw "The M3-C2 completion record is absent or violates the read-only contract"
    }
    foreach ($PhysicsTag in @("fpt", "fptas")) {
        if (@($Records | Where-Object {
                    $_.type -eq "physics_root" -and $_.physics_tag -eq $PhysicsTag
                }).Count -ne 1) {
            throw "Expected one physics_root record for $PhysicsTag"
        }
        if (@($Records | Where-Object {
                    $_.type -eq "brownian" -and $_.physics_tag -eq $PhysicsTag
                }).Count -ne 1) {
            throw "Expected one Brownian record for $PhysicsTag"
        }
        if (@($Records | Where-Object {
                    $_.type -eq "particle_dataset" -and $_.physics_tag -eq $PhysicsTag
                }).Count -ne 1) {
            throw "Expected one particle-dataset record for $PhysicsTag"
        }
    }

    $SourceHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceModel).Hash
    $JavaHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $JavaSource).Hash
    $RunnerHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $RunnerSource).Hash
    if ($SourceHashAfter -ne $SourceHashBefore) {
        throw "The locked source MPH changed during the read-only probe"
    }
    if ($JavaHashAfter -ne $JavaHashBefore -or $RunnerHashAfter -ne $RunnerHashBefore) {
        throw "A durable probe tool changed while it was running"
    }

    @(
        [pscustomobject][ordered]@{
            role = "locked_source_mph"
            path = $SourceModel
            sha256_before = $SourceHashBefore
            sha256_after = $SourceHashAfter
            unchanged = $true
        },
        [pscustomobject][ordered]@{
            role = "java_inspector"
            path = $JavaSource
            sha256_before = $JavaHashBefore
            sha256_after = $JavaHashAfter
            unchanged = $true
        },
        [pscustomobject][ordered]@{
            role = "powershell_runner"
            path = $RunnerSource
            sha256_before = $RunnerHashBefore
            sha256_after = $RunnerHashAfter
            unchanged = $true
        }
    ) | Export-Csv -LiteralPath (Join-Path $OutputDirectory "source_tool_hashes.csv") `
        -NoTypeInformation -Encoding utf8

    $Interfaces = @(
        foreach ($PhysicsTag in @("fpt", "fptas")) {
            $Root = @($Records | Where-Object {
                    $_.type -eq "physics_root" -and $_.physics_tag -eq $PhysicsTag
                })[0]
            $Brownian = @($Records | Where-Object {
                    $_.type -eq "brownian" -and $_.physics_tag -eq $PhysicsTag
                })[0]
            $Dataset = @($Records | Where-Object {
                    $_.type -eq "particle_dataset" -and $_.physics_tag -eq $PhysicsTag
                })[0]
            [pscustomobject][ordered]@{
                case = $Root.case
                physics_tag = $PhysicsTag
                formulation = $Root.formulation
                include_out_of_plane = $Root.include_out_of_plane
                random_number_args = $Root.random_number_args
                brownian_feature = $Brownian.feature_tag
                brownian_active = $Brownian.active
                brownian_i = $Brownian.i
                brownian_mu = $Brownian.mu
                brownian_temperature = $Brownian.temperature
                brownian_temperature_source = $Brownian.temperature_source
                particle_dataset = $Dataset.dataset_tag
                particle_position_dofs = $Dataset.posdof
            }
        }
    )
    $Manifest = [pscustomobject][ordered]@{
        schema_version = 1
        evidence_id = "M3-C2-model-semantics-probe-v1"
        evaluation_revision = 1
        tool_revision = "m3c2_model_semantics_probe_v1"
        overall_status = "COMPLETE_READ_ONLY_CHARACTERIZATION"
        scientific_status = "CHARACTERIZED_NOT_COMPARISON"
        accuracy_claim = "NOT_EVALUATED"
        source_mph = $SourceModel
        source_mph_sha256 = $SourceHashAfter
        source_hash_unchanged = $true
        java_inspector_sha256 = $JavaHashAfter
        powershell_runner_sha256 = $RunnerHashAfter
        compiled_class_sha256 = $CompiledClassSha256
        comsol_version = $ExpectedComsolVersion
        process_policy = "comsolbatch_-nosave_-np_1"
        load_copy = $true
        study_run = $false
        model_save = $false
        records = $Records.Count
        interfaces = $Interfaces
    }
    $Manifest | ConvertTo-Json -Depth 6 |
        Set-Content -LiteralPath (Join-Path $OutputDirectory "probe_manifest.json") -Encoding utf8

    @(
        "comsol_version.txt",
        "comsol_launcher.log",
        "comsol_process_output.log",
        "model_semantics.jsonl",
        "source_tool_hashes.csv",
        "probe_manifest.json"
    ) | ForEach-Object {
        $ArtifactPath = Join-Path $OutputDirectory $_
        [pscustomobject][ordered]@{
            artifact = $_
            size_bytes = (Get-Item -LiteralPath $ArtifactPath).Length
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ArtifactPath).Hash
        }
    } | Export-Csv -LiteralPath (Join-Path $OutputDirectory "artifact_hashes.csv") `
        -NoTypeInformation -Encoding utf8

} finally {
    Remove-Item -LiteralPath $ClassFile, $ClassStatus -ErrorAction SilentlyContinue
    Pop-Location
}

Write-Output "M3-C2 model-semantics probe completed: $OutputDirectory"
