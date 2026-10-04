param(
    [Parameter(Mandatory = $true)]
    [string]$Manifest,
    [Parameter(Mandatory = $true)]
    [string]$ArtifactRoot,
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
$SolverRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$ManifestPath = (Resolve-Path -LiteralPath $Manifest).Path
$ArtifactRootPath = (Resolve-Path -LiteralPath $ArtifactRoot).Path
$OutputPath = [System.IO.Path]::GetFullPath($OutputDirectory)

if (Test-Path -LiteralPath $OutputPath) {
    throw "Matched-case output directory already exists: $OutputPath"
}
New-Item -ItemType Directory -Path $OutputPath | Out-Null

Push-Location $SolverRoot
try {
    $PreflightPath = Join-Path $OutputPath "preflight.json"
    & uv run --locked python tools/vv/comsol/compare_matched_case.py preflight `
        $ManifestPath `
        --artifact-root $ArtifactRootPath `
        --output $PreflightPath
    if ($LASTEXITCODE -ne 0) {
        throw "Matched-case preflight is blocked; inspect $PreflightPath"
    }

    $ComparisonPath = Join-Path $OutputPath "comparison.json"
    & uv run --locked python tools/vv/comsol/compare_matched_case.py compare `
        $ManifestPath `
        --artifact-root $ArtifactRootPath `
        --output $ComparisonPath
    if ($LASTEXITCODE -ne 0) {
        throw "Matched-case comparison did not pass; inspect $ComparisonPath"
    }
} finally {
    Pop-Location
}

Write-Output "Matched-case comparison completed: $OutputPath"
