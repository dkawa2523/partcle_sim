param(
    [Parameter(Mandatory=$true)]
    [string]$ComsolExe,

    [Parameter(Mandatory=$false)]
    [string]$Mph = "data\icp_rf_bias_cf4_o2_si_etching (2).mph",

    [Parameter(Mandatory=$false)]
    [string]$Config = "external\comsol_icp_export\config\icp_cf4_o2_v20.json",

    [Parameter(Mandatory=$false)]
    [string]$OutDir = "_external_exports\icp_cf4_o2_v20"
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
function Resolve-FromRoot([string]$PathText) {
    if ([System.IO.Path]::IsPathRooted($PathText)) {
        return Resolve-Path $PathText
    }
    return Resolve-Path (Join-Path $root $PathText)
}

$javaFile = Join-Path $root "external\comsol_icp_export\java\IcpCf4O2SiEtchExporter.java"
$classFile = Join-Path $root "external\comsol_icp_export\java\IcpCf4O2SiEtchExporter.class"
$mphPath = Resolve-FromRoot $Mph
$configPath = Resolve-FromRoot $Config
$outPath = if ([System.IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $root $OutDir }
New-Item -ItemType Directory -Force $outPath | Out-Null
$prefsDir = Join-Path $outPath "comsol_export_prefs"
New-Item -ItemType Directory -Force $prefsDir | Out-Null
$basePrefs = Join-Path $env:USERPROFILE ".comsol\v64\comsol.prefs"
$prefsFile = Join-Path $prefsDir "comsol.prefs"
if (Test-Path $basePrefs) {
    Copy-Item -Force $basePrefs $prefsFile
} else {
    New-Item -ItemType File -Force $prefsFile | Out-Null
}
function Set-PrefLine([string]$Key, [string]$Value) {
    $lines = @()
    if (Test-Path $prefsFile) {
        $lines = Get-Content $prefsFile
    }
    $pattern = "^" + [regex]::Escape($Key) + "="
    if ($lines | Where-Object { $_ -match $pattern }) {
        $lines = $lines | ForEach-Object { if ($_ -match $pattern) { "$Key=$Value" } else { $_ } }
    } else {
        $lines += "$Key=$Value"
    }
    Set-Content -Path $prefsFile -Value $lines -Encoding UTF8
}
Set-PrefLine "security.external.filepermission" "all"
Set-PrefLine "security.external.propertypermission" "all"
Set-PrefLine "security.external.reflectpermission" "on"
Set-PrefLine "security.external.enable" "off"

$compileExe = Join-Path (Split-Path $ComsolExe -Parent) "comsolcompile.exe"
if (!(Test-Path $compileExe)) {
    throw "Could not find comsolcompile.exe next to $ComsolExe"
}

if (Test-Path $classFile) {
    Remove-Item -Force $classFile
}
& $compileExe $javaFile
if ($LASTEXITCODE -ne 0 -or !(Test-Path $classFile)) {
    throw "COMSOL Java compilation failed with exit code $LASTEXITCODE"
}

$oldMph = $env:COMSOL_ICP_MPH
$oldConfig = $env:COMSOL_ICP_CONFIG
$oldOutDir = $env:COMSOL_ICP_OUTDIR
try {
    $env:COMSOL_ICP_MPH = [string]$mphPath
    $env:COMSOL_ICP_CONFIG = [string]$configPath
    $env:COMSOL_ICP_OUTDIR = [string]$outPath
    & $ComsolExe -prefsdir $prefsDir -inputfile $classFile -batchlog (Join-Path $outPath "comsol_export.log") -nosave

    if ($LASTEXITCODE -ne 0) {
        throw "COMSOL export failed with exit code $LASTEXITCODE"
    }
}
finally {
    $env:COMSOL_ICP_MPH = $oldMph
    $env:COMSOL_ICP_CONFIG = $oldConfig
    $env:COMSOL_ICP_OUTDIR = $oldOutDir
}

if (!(Test-Path (Join-Path $outPath "field_samples.csv"))) {
    throw "COMSOL export did not produce field_samples.csv. Check comsol_export.log in $outPath"
}

Write-Host "Raw COMSOL export written to $outPath"
