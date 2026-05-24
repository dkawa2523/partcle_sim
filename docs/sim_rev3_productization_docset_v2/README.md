# sim_rev3 Productization Docset v2

この資料群は、`dkawa2523/partcle_sim` の `sim_rev3` を、COMSOL比較に耐える高速・柔軟な粒子追跡エンジンへ段階的に育てるための実装前提資料です。

v1 からの最も重要な変更は、**比較・評価基盤を実装後半ではなく最初に置いたこと**です。元の課題は「Codex が局所数値だけ見て修正し、全体傾向を悪化させる」ことだったため、新しい物理機能やwall処理を触る前に、post-import、post-preprocess、post-first-step、event、ensemble を比較できるレールを先に固定します。

## この資料群の使い方

リポジトリでは以下に配置してください。

```text
docs/productization/sim_rev3/
```

推奨配置後の構成:

```text
docs/productization/sim_rev3/
  README.md
  MANIFEST.md
  docs/
  implementation_layers/
  checklists/
  adrs/
  templates/
  codex_notes/
```

Codex に実装させるときは、まずこの順で読ませます。

1. `README.md`
2. `docs/00_original_goal_and_non_goals.md`
3. `docs/01_vigus_lessons_generalized.md`
4. `docs/03_failure_modes_and_diagnostic_strategy.md`
5. `docs/04_revised_phase_roadmap.md`
6. `docs/06_minimalism_policy.md`
7. `codex_notes/read_before_implementation.md`

## 実装の基本順序

```text
Phase 0   repo audit / cleanup inventory
Phase 0.5 focused tests and baseline smoke stabilization
Phase 1   minimal comparison rails
Phase 2   mode separation + minimal manifest gate
Phase 3   import / coordinate / axisymmetric minimum semantics
Phase 4   release canonicalization
Phase 5   force breakdown / first-step parity
Phase 6   release grace / wall event simplification
Phase 7   axisymmetric_rz completion
Phase 8   V&V productization
Phase 9   cleanup / simplification / performance
```

## 実装原則

- VIGUS専用のwall idやcase固有補正をsolver coreへ入れない。
- COMSOL faithful comparison と production surface-release run を混ぜない。
- 比較レールを先に作る。物理変更はその後。
- diagnostic は常時詳細ログではなく、通常は小さいsummaryとcounterだけにする。
- helper、contract、test は「読みにくさを減らす」場合だけ追加する。
- product code は第三者が読んで追える小さな責務単位に保つ。
- 古い探索枝、`_tmp_*`、one-off出力、失敗分岐は mainline の根拠にしない。
