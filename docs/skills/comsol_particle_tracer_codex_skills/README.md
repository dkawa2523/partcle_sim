# COMSOL Comparison Skills for Codex

このドキュメントセットは、VS Codeで対象リポジトリを開いているCodexが、COMSOLモデルまたはCOMSOL export資産から必要情報を抽出し、本粒子追跡コードを実行し、結果比較と評価を行うためのスキル群です。

目的は「VIGUSに合わせた局所パッチ」ではありません。COMSOLモデルは、2D/3D、axisymmetric、stationary/time-dependent、field-only、particle tracing込み、壁条件やrelease featureの構成が異なります。このため、本スキルは **モデル差を最初に検出し、比較可能な断面だけを評価する** ことを優先します。

## 使い方

Codexには、まず以下を読ませます。

```text
comsol_particle_tracer_codex_skills/README.md
comsol_particle_tracer_codex_skills/skills/00_global_rules/SKILL.md
comsol_particle_tracer_codex_skills/skills/01_model_intake/SKILL.md
```

その後、作業内容に応じて以下の順に進めます。

```text
01_model_intake
02_comsol_access_and_export
03_export_manifest
04_geometry_boundary
05_field_bundle
06_particle_release_reference
07_wall_force_semantics
08_case_build_preflight
09_run_solver
10_compare_layers
11_evaluation_decision_tree
12_residual_triage
13_safe_extension
14_validation_record
```

## 基本方針

- COMSOL `.mph` をsolver coreに読ませない。
- COMSOL API / LiveLink / GUI export は `external/` または `tools/` 側で行う。
- solverは明示された YAML / CSV / NPZ / manifest だけを読む。
- COMSOL faithful comparison と production surface-release analysis を混ぜない。
- 評価順序は `import -> preprocess -> first-step -> wall events -> ensemble -> runtime` を守る。
- endpoint数だけで修正の正否を判断しない。
- モデル差が不明な場合は、推測で補正せず、manifestに不足として記録して止める。

## 生成される主要成果物

```text
case_manifest.yaml
extraction_manifest.yaml
field_mapping.csv
boundary_part_map.csv
wall_law_map.csv
release_mapping.csv
reference_particle_schema.csv
preflight_report.json
first_step_compare_summary.json
boundary_hit_comparison.json
comparison_summary.json
residual_gap_report.md
validation_record.md
```

## 推奨配置

リポジトリに入れる場合は以下を推奨します。

```text
partcle_sim/
  docs/
    codex_skills/
      comsol_particle_tracer/
        README.md
        skills/
        templates/
        checklists/
        examples/
```

ただし、実装コードとは混ぜないでください。これはCodex作業用の参照資料です。
