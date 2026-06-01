# 10. How to Place This Docset

## Recommended location

```text
partcle_sim/
  docs/
    productization/
      sim_rev3/
        README.md
        MANIFEST.md
        docs/
        implementation_layers/
        checklists/
        adrs/
        templates/
        codex_notes/
```

## PowerShell example

```powershell
git switch sim_rev3
git switch -c productization/docset-v2

$zip = "$env:USERPROFILE\Downloads\sim_rev3_productization_docset_v2.zip"
$dest = "docs\productization\sim_rev3"

New-Item -ItemType Directory -Force $dest | Out-Null
Expand-Archive -Path $zip -DestinationPath $dest -Force

$nested = Join-Path $dest "sim_rev3_productization_docset_v2"
if (Test-Path $nested) {
  Copy-Item "$nested\*" $dest -Recurse -Force
  Remove-Item $nested -Recurse -Force
}

git status --short
```

## Commit separately

資料配置だけを先にcommitする。

```powershell
git add docs/productization/sim_rev3
git commit -m "Add sim_rev3 productization docset v2"
```

その後、Phase 0 auditを別commitにする。
