param(
    [Parameter(Mandatory = $true)]
    [string]$CaseDirectory,
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,
    [string]$ComsolRoot = "C:\Program Files\COMSOL\COMSOL64\Multiphysics_copy1"
)

$ErrorActionPreference = "Stop"

$CaseRoot = (Resolve-Path -LiteralPath $CaseDirectory).Path
$Runner = Join-Path $PSScriptRoot "run_m3c1_common_p1_reference.ps1"
& $Runner `
    -ComsolRoot $ComsolRoot `
    -OutputDirectory $OutputDirectory `
    -RunProfile campaign `
    -ConfigPath (Join-Path $CaseRoot "reference_run_config.json") `
    -RunSpecPath (Join-Path $CaseRoot "run_spec.properties")
if ($LASTEXITCODE -ne 0) {
    throw "Case-A size/ion-drag companion run failed with exit code $LASTEXITCODE"
}
