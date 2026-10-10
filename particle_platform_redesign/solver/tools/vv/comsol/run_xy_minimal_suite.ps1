param(
    [string]$ComsolRoot = "C:\Program Files\COMSOL\COMSOL64\Multiphysics_copy1",
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
$ToolRevision = "comsol_xy_minimal_suite_v1"
$SolverRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$Compiler = Join-Path $ComsolRoot "bin\win64\comsolcompile.exe"
$Batch = Join-Path $ComsolRoot "bin\win64\comsolbatch.exe"
$JavaSource = Join-Path $PSScriptRoot "comsol\RunXYMinimalSuite.java"
$Evaluator = Join-Path $PSScriptRoot "evaluate_xy_minimal_suite.py"
$RunnerSource = $PSCommandPath
foreach ($Required in @($Compiler, $Batch, $JavaSource, $Evaluator, $RunnerSource)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required XY comparison input does not exist: $Required"
    }
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "XY comparison output already exists: $OutputDirectory"
}

New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$RawRoot = Join-Path $OutputDirectory "raw"
$RegisteredRoot = Join-Path $OutputDirectory "registered_sources"
New-Item -ItemType Directory -Path $RawRoot, $RegisteredRoot | Out-Null
$Scenarios = @(
    "ballistic",
    "electric",
    "linear_drag",
    "surface_departure",
    "specular",
    "stick",
    "hold"
)
$StepDirectories = @("dt_2.000e-02", "dt_1.000e-02", "dt_5.000e-03")
foreach ($Scenario in $Scenarios) {
    foreach ($StepDirectory in $StepDirectories) {
        New-Item -ItemType Directory -Path (Join-Path $RawRoot "$Scenario\$StepDirectory") |
            Out-Null
    }
}

$SourcePaths = [ordered]@{
    java = $JavaSource
    evaluator = $Evaluator
    runner = $RunnerSource
}
$RegisteredSources = [ordered]@{}
foreach ($Name in $SourcePaths.Keys) {
    $Source = $SourcePaths[$Name]
    $Destination = Join-Path $RegisteredRoot ([System.IO.Path]::GetFileName($Source))
    Copy-Item -LiteralPath $Source -Destination $Destination
    $SourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Source).Hash.ToLowerInvariant()
    $DestinationHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $Destination
    ).Hash.ToLowerInvariant()
    if ($DestinationHash -ne $SourceHash) {
        throw "Registered-source copy differs: $Name"
    }
    $RegisteredSources[$Name] = [ordered]@{
        path = [System.IO.Path]::GetRelativePath($OutputDirectory, $Destination).Replace("\", "/")
        sha256 = $SourceHash
    }
}

$VersionText = (& $Batch -version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read COMSOL version; exit code $LASTEXITCODE"
}
$CompileRoot = Join-Path (
    [System.IO.Path]::GetTempPath()
) ("xy-minimal-compile-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $CompileRoot | Out-Null
$StagedJava = Join-Path $CompileRoot "RunXYMinimalSuite.java"
$ClassFile = Join-Path $CompileRoot "RunXYMinimalSuite.class"
$CompileLog = Join-Path $OutputDirectory "comsol_compile.log"
$ProcessLog = Join-Path $OutputDirectory "comsol_process.log"
$BatchLog = Join-Path $OutputDirectory "comsol_batch.log"
$ConfigurationLog = Join-Path $OutputDirectory "configuration_receipt.log"
$CompiledClassHash = ""
try {
    Copy-Item -LiteralPath $JavaSource -Destination $StagedJava
    & $Compiler $StagedJava *> $CompileLog
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $ClassFile -PathType Leaf)) {
        throw "COMSOL Java compilation failed with exit code $LASTEXITCODE"
    }
    $CompiledClassHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $ClassFile
    ).Hash.ToLowerInvariant()

    Push-Location $RawRoot
    try {
        & $Batch -inputfile $ClassFile -nosave -np 1 -batchlog $BatchLog *> $ProcessLog
        if ($LASTEXITCODE -ne 0) {
            throw "COMSOL XY suite failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }

    $ConfigurationLines = @(
        Get-Content -LiteralPath $ProcessLog | Where-Object { $_ -match '^XYMIN\|' }
    )
    if ($ConfigurationLines.Count -eq 0) {
        $ConfigurationLines = @(
            Get-Content -LiteralPath $BatchLog | Where-Object { $_ -match '^XYMIN\|' }
        )
    }
    if ($ConfigurationLines.Count -eq 0) {
        throw "COMSOL emitted no XY configuration receipt"
    }
    $ConfigurationLines | Set-Content -LiteralPath $ConfigurationLog -Encoding utf8

    $RawArtifacts = @(
        foreach ($Scenario in $Scenarios) {
            foreach ($StepDirectory in $StepDirectories) {
                $Path = Join-Path $RawRoot "$Scenario\$StepDirectory\state_raw_wide.csv"
                if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
                    throw "COMSOL did not emit expected XY history: $Path"
                }
                [ordered]@{
                    path = [System.IO.Path]::GetRelativePath($RawRoot, $Path).Replace("\", "/")
                    sha256 = (
                        Get-FileHash -Algorithm SHA256 -LiteralPath $Path
                    ).Hash.ToLowerInvariant()
                    bytes = (Get-Item -LiteralPath $Path).Length
                }
            }
        }
    )
    $VersionLine = Get-Content -LiteralPath (
        Join-Path $RawRoot "ballistic\dt_5.000e-03\state_raw_wide.csv"
    ) | Where-Object { $_ -match '^% Version,' } | Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($VersionLine)) {
        throw "COMSOL raw output has no version metadata"
    }
    $RawComsolVersion = ($VersionLine -split ',', 2)[1].Trim().Trim('"')
    $Receipt = [ordered]@{
        schema_version = 1
        tool_revision = $ToolRevision
        run_status = "completed"
        generated_utc = [DateTime]::UtcNow.ToString("o")
        comsol_root = [System.IO.Path]::GetFullPath($ComsolRoot)
        comsol_launcher_version = $VersionText
        raw_comsol_version = $RawComsolVersion
        process_policy = "fresh_in_memory_model_comsolbatch_-nosave_-np_1"
        model_saved = $false
        raw_root = "raw"
        configuration_log = "configuration_receipt.log"
        compiled_class_sha256 = $CompiledClassHash
        registered_sources = $RegisteredSources
        raw_artifacts = $RawArtifacts
    }
    $ReceiptPath = Join-Path $OutputDirectory "run_receipt.json"
    $Receipt | ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath $ReceiptPath -Encoding utf8

    Push-Location $SolverRoot
    try {
        & uv run --locked python $Evaluator $RawRoot (
            Join-Path $OutputDirectory "evaluation"
        ) --run-receipt $ReceiptPath
        if ($LASTEXITCODE -ne 0) {
            throw "XY candidate evaluation failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }

    foreach ($Name in $SourcePaths.Keys) {
        $After = (
            Get-FileHash -Algorithm SHA256 -LiteralPath $SourcePaths[$Name]
        ).Hash.ToLowerInvariant()
        if ($After -ne $RegisteredSources[$Name].sha256) {
            throw "Registered source changed during run: $Name"
        }
    }

    $Artifacts = Get-ChildItem -LiteralPath $OutputDirectory -Recurse -File |
        Where-Object { $_.Name -ne "artifact_hashes.csv" } |
        Sort-Object FullName |
        ForEach-Object {
            [ordered]@{
                path = [System.IO.Path]::GetRelativePath(
                    $OutputDirectory,
                    $_.FullName
                ).Replace("\", "/")
                sha256 = (
                    Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName
                ).Hash.ToLowerInvariant()
                bytes = $_.Length
            }
        }
    $Artifacts | Export-Csv -LiteralPath (
        Join-Path $OutputDirectory "artifact_hashes.csv"
    ) -NoTypeInformation -Encoding utf8
} finally {
    $NormalizedTemp = [System.IO.Path]::GetFullPath($CompileRoot)
    $SystemTemp = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
    if ($NormalizedTemp.StartsWith($SystemTemp, [System.StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $NormalizedTemp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Output "COMSOL Cartesian XY minimal suite completed: $OutputDirectory"
