# 粒子輸送基盤の実装と保守

## 現行の実装単位

0.4.0はcoreと通常clientを一つのwheelへ収録し、CLI/Pythonで入力から結果再利用までを提供する。
契約は[製品仕様](product_specification.md)、[architecture](architecture_proposal.md)、solver/docsの各ownerが所有する。
全体変更の調査・実行・未確認は[全体対応表](solver/evidence/retirement_2026_11_23/README.md)へ残す。

| 単位 | 完了条件 |
|---|---|
| 入力作成 | CSV/背景bundle→最初のinternal/surface source→単一publisher→hash付きcase。単位/ID/物性/基底保持 |
| 能力/検査 | catalog由来metadata、静的loadと実行同一prepare、readonly report、workers依存memory |
| core実行 | typed state/workspace、共通transport、bounded execution、親時計/single writer/毎macro commit |
| 結果 | result4 context、strict completed/partial、checkpoint2、旧readerなし、元caseなしの解釈 |
| 解析/表示 | counts/weights/分母/coverage、interaction/first-hit/terminal区別、bounded/finite/gap/equal aspect |
| 外部経路 | native/V&V分離、現行descriptorと明示primitive、native/解析参照。旧固定campaign経路は提供しない |
| 配布/説明 | checkout外のwheel一巡、README/AGENTS/skills、reviewed report、CIとsource同一性 |

## 実装と受入の順序

1. 変更前source/input/resultを保全し、全領域と独立経路を対応付ける。
2. 通常clientと単一publisherを配布へ接続する。
3. 実行同一check、context、core責務・容量を既存数値規則へ接続する。
4. 呼出側/例題/tests/説明を更新し、旧入口と不要な重複を撤去する。
5. 独立小型reference、全通常scenario、portable V&V、installed wheelを実行する。
6. 影響する科学比較/性能/reportを再生成し、三観点の独立レビュー必須指摘を修正する。

実装済み範囲へ目的を縮めない。過去sourceのPASSを新sourceの完了証拠にしない。

## 条件付きbacklog

| 残件 | 引き継ぐ条件と証拠 |
|---|---|
| G1/G2 入力/外部場 | realized相関/物性/frame/identityとselected nodal域/side/回復を保持。時間/高次/不連続/多領域は独立転送検証 |
| G3 gas interaction | 既存collision quadrature/jump/frozen oracleを維持。非線形noise/粒子熱状態/high-drift輸送は抵抗/拡散/熱泳動の仮定、保存則、漸近を閉じてから同stageへ追加 |
| G4 精度qualification | deterministic、field/time、event、確率path、参照精度を分離。controllerは必要性/費用を実測後。charge-only subcycle/別engineを作らない |
| G5/P17 物理3D | RZ場+XYZと完全3D meshを区別。source/state/basis/axis/normal/hit/wall/noise/output/checkpoint/memoryの縦切り、回転共変/解析面hit/独立分布 |
| 非定常の追加 | 固定topology linear-timeは実装済み。周期/位相/不連続/逐次読込は別契約。moving mesh/feedbackを暗黙追加しない |
| 独立全材料電場 | gas-only縮約を全材料と呼ばない。材料epsilon、導体V/浮遊Q、界面/個別Q/global balance、gas回復とcharge/ion/DEP再生成まで |
| surface charging/再飛散 | wall state/保存則/lifecycle/release authorityを定義。terminalを暗黙再活性化しない |
| G6/多macro同期 | IPC支配をprofileで確認。dt/knots/release/RNG/毎macro集計/観測/bounded ACK/checkpoint、途中死/resume、科学/時間/memoryを検証 |
| NUMA/GPU/out-of-core | 実host/用途/accuracy/SLAが必要性を示す場合だけ。単一経路、転送/serialization込みの容量/費用 |
| 装置予測 | source rate/weight、capture/reemission物理、実験を要する。数値stick/COMSOL対応だけで認定しない |

既存G1/G2変換、linear-time、連成charge、midpoint OU、durable出力/processは完了能力として維持する。
過去P00〜P22/G1〜G4/0.1〜0.3工程は元source/evidenceへ固定し、未指定拡張/追加JIT/全CPU比例を今回の前提に戻さない。
未実装flag/空interfaceを先行しない。

## 変更時の受入

追加/category/state/壁則と削減はarchitectureの縦切りを追い、式/適用域/連成/結果意味/容量/呼出側を同時確認する。
code移動は同入力/seed/clockの科学payloadを保持。数値変更は別変更として独立reference/budget/revisionを要する。
最終判断は[品質gate](quality_tooling_plan.md)、全体対応表、利用者/実装者/科学レビューによる。
