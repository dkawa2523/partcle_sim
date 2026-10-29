# Particle Platform Redesign

外部の場とgeometryから、半導体製造チャンバー内の粒子軌道・電荷・境界イベントを計算する独立基盤です。
現行solverは`solver/`のuv project、package versionは**0.3.0**です。
利用方法は[solver README](solver/README.md)、変更点は[release notes](solver/RELEASE_NOTES.md)を参照してください。

## 現行0.3.0

Cartesian XYとRZ meridional、static/固定topology linear-timeのregular/P1/Q1場を扱います。
対応する決定論・Brownianの数値処理、wall law、独立接触半径、XY translation periodic、
checkpoint/resumeを一つのengineへ統合しています。対応組合せと適用域はprepareで検査します。

`simulate(case, output, *, workers=1)`とCLI `--workers`で同一runのprocess数を指定できます。
親がmacro時計・release・single writer・checkpointを所有し、永続workerが排他的な粒子群を更新します。
全workerの完了待ちは活動粒子の1 macroステップごとです。複数macroの同期間隔設定は将来項目に留めます。

大粒子数では並列効果を確認しましたが、少粒子・軽い各macro・短いevent用途では遅くなる場合があります。
粒子数やstep数だけで速度を決めず、[用途別性能とメモリ](solver/docs/parallel_execution_plan.md)の実測条件を確認してください。
最新READMEで再構築したWindows installed-wheelと、検証済みsourceとの照合は
[0.3.0 branch検証記録](reviews/v0_3_branch_2026_10_13/source_qualification.json)へ保存しています。
Git branch、remote CI、正式tag/Release、package registryの公開状況は、それぞれの実操作で確認します。

## 利用者向けの入口

| 確認したい内容 | 所有文書 |
|---|---|
| Quick Start、三公開API、process実行 | [solver README](solver/README.md) |
| 対応組合せ、適用域、例外、failure | [support and errors](solver/docs/support_and_errors.md) |
| canonical入力 | [case format v3](solver/docs/case_format_v3.md) |
| 結果・checkpoint・現行revision | [result format v3](solver/docs/result_format_v3.md) |
| 物理modelと数値method | [physics models](solver/docs/physics_models.md)、[numerics](solver/docs/numerics.md) |
| CPU実行・同期・メモリ・用途別性能 | [parallel execution](solver/docs/parallel_execution_plan.md) |
| COMSOL比較の方法、合格範囲、未評価層 | [V&V methodology](vv_methodology.md) |

COMSOL Case A/P Brownianの現行版回帰は、保存参照に対する登録した位置・終端人口分布の評価です。
全native右辺、詳細な境界応答、速度・電荷分布、一般的な製造予測精度の認証へ拡張しません。

## 開発者向けの所有文書

| 責務 | 文書 |
|---|---|
| clean-room境界、数値不変条件、変更規約 | [AGENTS.md](AGENTS.md) |
| 製品目的、範囲、非目標 | [product specification](product_specification.md) |
| module責務、依存方向、状態と実行境界 | [architecture](architecture_proposal.md) |
| file責務、現行backlog、将来同期案 | [implementation plan](implementation_plan.md) |
| locked環境、静的品質、数値・配布gate | [quality tooling](quality_tooling_plan.md) |
| 理論調査と根拠 | [technical research](technical_research.md) |
| 参照datasetの許可用途と制約 | [data quality](data_quality_assessment.md)、[dataset rebuild plan](model_dataset_rebuild_plan.md) |

通常の開発・検証は`solver/`からuvのlocked環境で行います。
具体的なcommandは[quality tooling](quality_tooling_plan.md)へ一元化し、旧環境を起動しません。
設計時の`analysis/`・`evidence/`・`report_app/`は外部監査資料で、solver runtimeの依存先ではありません。

## 参照資料と履歴

3D化と背景場の時間変化について、2D定常計算の完成を前提にした追加実装の検討表と4区分の図版は、
[3D粒子軌道・非定常背景場の拡張検討](reviews/3d_transient_trajectory_extensions/README.md)を参照してください。

`../model_dataset/`は入力provenanceと外部V&V用の参照資料です。物理modelの正解やsolver dependencyにはしません。
`../old_code/`はread-only archiveで、旧package・環境・実行経路を新solverへ復活させません。
新production codeの置き場は`solver/`だけです。

完了済みstageや旧revisionの逐次進捗は現行仕様へ重ねません。履歴はGitとimmutable evidenceへ保持し、
[Stage 1a](solver/docs/stage1a_validation.md)・[Stage 1b](solver/docs/stage1b_validation.md)は当時のcloseoutとして扱います。
0.2.0の公開記録は[GitHub Release](https://github.com/dkawa2523/partcle_sim/releases/tag/v0.2.0)、
[検証記録](reviews/v0_2_release_qualification_2026-10-10.json)、
[公開後確認](reviews/v0_2_release_publication_2026-10-10.json)に固定し、0.3.0の資格へ付け替えません。
