# Particle Platform Redesign

自分の形状、背景場、粒子源を与え、検査・計算・結果の解釈と再利用までCLIとPythonで完結する基盤です。
現行は`solver/`の独立uv project、**0.5.0 alpha**です。
[solver README](solver/README.md)に、wheelだけで使える例題と自分の入力を作る手順があります。

計算core `chamber_particles`は`load_case`、`check_case`、`simulate`、`open_result`を公開します。
`check_case`は実行と同じprepareを使い、計算開始前の対応条件・field support・源・境界・資源を検査します。
通常の入力作成、場のimport/生成・cache、解析・export・表示は同じwheelの`chamber_particles_tools`が所有し、
CLIとPythonが同じ処理を呼びます。COMSOLがなくても入力から結果まで使えます。

Cartesian XYとRZ meridional、静的または固定topologyの線形時間snapshot場、対応する決定論・Brownian輸送、
独立した粒子物性・接触半径、壁応答、XY translation periodic、checkpoint/resumeを一つのengineで扱います。
`simulate(..., workers=...)`のprocess数は実行資源です。対応組合せと適用域は検査し、worker数で物理を変更しません。
入力case/dataはschema 3、結果はschema 5、checkpointはlayout 2です。

保存結果には形状・境界・sourceの解釈に必要なimmutable `context.h5`が含まれ、
`analyze`、`export`、`view`は元caseや背景場なしで使えます。contextは背景場や完全な実現sourceを複製しません。
旧結果の互換readerは提供しません。

| 確認したい内容 | 所有文書 |
|---|---|
| 例題、自分の入力、CLI/Python、結果再利用 | [solver README](solver/README.md) |
| 対応組合せ、適用域、例外・failure | [support and errors](solver/docs/support_and_errors.md) |
| canonical geometry/field/sourceと設定 | [case format v3](solver/docs/case_format_v3.md) |
| context、final、event、frame、checkpoint | [result format v5](solver/docs/result_format_v5.md) |
| 物理式と数値method | [physics models](solver/docs/physics_models.md)、[numerics](solver/docs/numerics.md) |
| process実行・メモリ・性能の証拠と制約 | [parallel execution](solver/docs/parallel_execution_plan.md) |
| COMSOL比較の意味・認定範囲 | [V&V methodology](vv_methodology.md) |

live COMSOL抽出と開発V&Vは`solver/tools/vv/`に残る外部機能です。通常のCSV importと入力作成へ
native依存を持ち込みません。Case A/Pの保存Brownian比較や30 nm Vigusの過去レポートは、それぞれに
同定された入力と観測量の証拠です。別modelの設定・許容値や一般的な製造予測精度へ外挿しません。

| 開発上の責務 | 所有文書 |
|---|---|
| clean-room境界、数値不変条件、変更規則 | [AGENTS.md](AGENTS.md) |
| 製品目的と非目標 | [product specification](product_specification.md) |
| moduleの所有と依存方向 | [architecture](architecture_proposal.md) |
| file責務と実装工程 | [implementation plan](implementation_plan.md) |
| locked環境・品質・配布gate | [quality tooling](quality_tooling_plan.md) |
| 理論と入力資料の扱い | [technical research](technical_research.md)、[data quality](data_quality_assessment.md) |

開発は`solver/`からuvのlocked環境で行います。`report_app/`は保存結果を説明するconsumerで、
solver runtimeとは分離します。`../model_dataset/`は参照入力、`../old_code/`はread-only archiveです。
過去版の資格・実行結果はimmutable evidenceとGitに保持し、現行の変更後に自動継承しません。
