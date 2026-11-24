# Chamber Particles 文書案内

現行は `solver/` の独立uv project、**0.5.0 alpha**です。
利用者が自分の形状・背景場・実現粒子源を準備し、検査・計算・保存結果の解釈まで進める基盤です。
このページは読む場所を案内します。手順、式、schema、実行結果をここへ複製しません。

## 使う

[solver README](solver/README.md)を入口にしてください。`models → init/create → check → run → inspect → analyze/export/view`
と、同じ操作を使うPython例があります。通常の入力作成・結果利用は同じwheelに含まれます。

| 確認したいこと | 読む文書 |
|---|---|
| 自分の入力、CLI/Python、結果の再利用 | [利用ガイド](solver/README.md) |
| 対応する組合せ、適用域、例外とfailure | [support and errors](solver/docs/support_and_errors.md) |
| canonical geometry/field/sourceとcase設定 | [case format](solver/docs/case_format_v3.md) |
| context、結果、checkpointと整合性 | [result format](solver/docs/result_format_v5.md) |
| 物理式・仮定と数値規則 | [physics models](solver/docs/physics_models.md)、[numerics](solver/docs/numerics.md) |
| 実行資源・性能測定の読み方 | [parallel execution](solver/docs/parallel_execution_plan.md) |

## 変更する

最初に[AGENTS.md](AGENTS.md)を読み、対象の事実を所有する文書へ進みます。

| 判断したいこと | 読む文書 |
|---|---|
| 製品の目的、現在の範囲、非目標 | [product specification](product_specification.md) |
| module責務、依存、受渡し、追加・削減の影響 | [現行architecture](architecture_proposal.md) |
| 採用した未実装拡張と条件付きの残件 | [implementation plan](implementation_plan.md) |
| 変更手順、品質tool、確認範囲、配布 | [quality tooling](quality_tooling_plan.md) |
| 採用した式の導出・研究根拠 | [technical research](technical_research.md) |

文書の形式・能力を変更する場合も、実装のない項目を対応済みにせず、既存の能力を未実装へ戻しません。
詳細な式はphysics、methodはnumerics、責務はarchitecture、残件はimplementation planに一度だけ記載します。

## 比較と証拠を読む

[外部V&V](solver/tools/vv/README.md)は通常実行から独立しています。
COMSOL比較の判定は[V&V methodology](vv_methodology.md)、参照資料の許可用途は[data quality](data_quality_assessment.md)を読みます。
[可視report](report_app/README.md)は、そのsnapshotへ明示して取り込んだ設計と証拠を説明するconsumerです。

現行0.5の再構成・品質・性能・保存A/P比較・配布確認は[foundation実行記録](solver/evidence/foundation_2026_11_23/README.md)へ束縛されています。
実行source、母集団、時計、分母、未確認範囲を伴う記録として読みます。実用的な輸送対応、数値的なnative parity、
実験による装置予測の妥当性は別の主張です。

## 履歴・設計参考

以下は現行仕様や新しい実装指示ではありません。「現行」「次工程」、採点、未修正判定は各記録のsource時点を指します。

- [初期architecture review](reviews/history/architecture_review.md)、[旧dataset再構築計画](reviews/history/model_dataset_rebuild_plan.md)：初期設計と当時の監査・工程。
- [2026-10-09専門レビュー](reviews/expert_review_2026-10-09.md)、[理論・数値レビュー](reviews/theory_numerics_validity_review_2026-10-09.md)、[同日の改良設計](reviews/improvement_design_2026-10-09.md)：当時のsourceでの指摘と実施範囲。
- [2026-11-23旧sourceの採点・レビュー](reviews/final_project_review_2026-11-23.md)：78c8c4基点のhash固定記録。公開入力・出力の修復前の判定であり、現行の未修正一覧ではありません。
- [foundation根因改良の履歴提案](reviews/foundation_root_cause_review_2026-11-23.md)：採用後の実装状況は上記foundation記録を参照。
- [3D・時間変化の設計参考](reviews/3d_transient_trajectory_extensions/README.md)：2D定常を出発点とした追加範囲の説明。採用済み未実装項目はimplementation planで区別。

元のnative観測、独立科学oracle、実験資料、実行記録のbytesを説明の更新で書き換えません。
旧softwareの互換入口と、必要な参照資料の保持を混同しないでください。
