# Phase 0 Checklist: Repo Audit / Cleanup Inventory

## 目的

現状の正本、探索枝、temporary diagnostics、drifted testsを棚卸しする。コードは変更しない。

## 作るもの

```text
docs/productization/sim_rev3/codex_notes/phase_0_repo_audit.md
```

## 調べる対象

- run entrypoints
- COMSOL/VIGUS tools
- build_comsol_case
- source_preprocess
- high_fidelity_collision
- compare tools
- sharded runner
- tests related to builder / release / boundary / compare
- examples/vigus* configs
- outputs/_tmp* references
- obsolete docs/comments

## 分類

```text
keep
delete_candidate
quarantine_to_research
needs_review
```

## 禁止

- production code変更
- test変更
- output追加
- new framework追加
