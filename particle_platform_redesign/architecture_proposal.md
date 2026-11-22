# 粒子輸送基盤の責務と受渡し

## 依存と公開境界

数値coreと通常clientは同じ独立uv project/distributionに収録する。数値core/APIはclient、COMSOL、V&V、reportをimportしない。
consoleとmoduleの入口だけが唯一のclient dispatcherへ接続する。

```text
外部CSV/解析/計測 → 入力publisher・場builder・cache → canonical SI bundle/case
公開API → prepare → engineの親時計・release・single writer
                         ↓ executionのworker・共有state・bounded ACK
                         ↓ transportの共通slab更新
                         ↓ fields / physics / integration / event / boundary
                         ↓ result + immutable context → analysis/export/view/V&V/report
```

| owner | 所有する事実 | 所有しないもの |
|---|---|---|
| case/case_format/yaml_input | 設定、SIデータ、hash/schema、構文/静的参照 | producer式、実行能力の別validator |
| catalog/runtime/compiled | model契約・必要field・適用域・stage合成 | mesh/I/O/native tag |
| sources/coordinates/topology | realized schedule、基底/axis、periodic seam | sampling、材料wall law |
| fields/geometry | 補間/support、containment/first-hit候補 | cache/CAD修復、物理式、付着 |
| integrators/stochastic | 連成proposal/dense state、joint OU/条件付きtree | BVH/wall/output時計 |
| events/boundaries | earliest prefix・残時間、hit後の応答 | endpoint押戻し、親時計/writer |
| cpu | typed resident state、workspace容量、phase峰値/slab fit | physics policy、I/O |
| transport | serial/process共通slab数値更新 | process起動、親時計、writer |
| execution | spawn/shared transport/bounded IPC/ACK | engine import、第二数値loop、worker writer |
| engine | prepare、親macro/release/accepted commit/checkpoint | importer/plot/COMSOL比較 |
| output | durable result/context/publication/recovery、readonly ResultView | 科学集計、再sampling |
| 通常client | 入力作成・変換・生成、解析/export/表示、唯一CLI | 数値core再実装 |
| 外部V&V/report | 対応問題の比較とreviewed結果の説明 | 物理authority、精度の一括認定 |

## 入力と検査

DataBundleがgeometry/layout/field/実現source、SimulationSpecが選択model・壁・時計・出力を所有する。
単一publisherがcanonical書込みとYAML hash結線を行い、新sourceに仮table/既存caseを要求しない。
load_caseは構文/hash/静的参照、check_caseはsimulateと同じprepareを使う。
prepareはgeometry/topology/required field/source/physics/event/数値path/workers依存memoryを解決する。
CheckReportは小さいimmutable結果で、PreparedRun/worker/writer/workspaceを公開しない。
model metadataはcatalogを一度定義し、key検査・能力表示・manifestへ使う。stageでmetadataを生成しない。

## 状態・連成・容量

resident rowは固定IDと対応しrelease順/active indexと区別する。物理配列は一つの型で親/workerがviewを共有し、
親はclock/cursor/統計だけを加える。field/force/G/Jは実stageの位置/時刻/Zで評価する。
accepted prefixだけがstate/cell hintをcommitし、hitと観測は同じproposalから求める。反射後はpost-impact stateから残時間を進める。
Brownianの条件付きsplitとwall/axis後fresh root、ID/物理interval/stream/ordinal RNGを維持し、worker/slab/保存時計を乱数authorityにしない。

workspace ownerがallocationと同じshape/dtype/capacityからbytes/private/shared区分を得る。
plannerはload/prepare/run同時live峰値とslab fitを所有し、context読込・writer transientも含める。OS RSS capとは異なる。
汎用allocator/DI/registryを作らない。親が毎macroで全task/outputを消費してcommitし、workerは一batch ACK後にbufferを再利用する。

## 結果と変更単位

必須contextはcanonical geometry、SI/基底、resolved source/boundaryの既存IDと意味、axis/contact/periodic対応を一度保存する。
同じserializer/validatorを使いfile/logical hashをmanifest/checkpointへbindする。resumeでは検証して解放し再生成しない。
初期全配列やfield/traceを保存contextへ重ねず、contextだけで再計算可能とはしない。

既存categoryの追加はcatalog→式/bounds→stage binding→compiled合成→独立検証で閉じ、engineを触らない。
新categoryでは合成順/全bounds/constant証明/容量/結果説明も確認する。新stateは初期値からstage/dense/event/checkpoint/resultまで縦切りを要する。
wall lawは応答/RNG/manifest/保存則/残時間検証を変え、first-hit処理を複製しない。削減は旧selector/経路/呼出側/設定/tests/docsを同時撤去する。
exact/curved/OU、global/local bound、support/誤差認証は固有の役割があり、似たcodeだけを理由に統一しない。
production数値規則は一owner、独立式oracleはverification/V&Vに保持する。

## 検証と履歴

解析/製造解、独立確率reference、時間/場/event収束、公開scenario、installed-wheel一巡を組み合わせる。
旧出力一致は構造変更のidentity検査で、唯一の正しさ判定ではない。
式/schema/性能資格は各ownerへリンクし、architectureへrevision一覧や進捗diaryを重ねない。以前の詳細・採否はGit履歴とimmutable evidenceに保持する。
