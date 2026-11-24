# 現行アーキテクチャ：責務と受渡し

本書は採用済みのmodule構成、依存、事実と状態の受渡しを定義します。
ファイル名は継続していますが、未採用proposalや過去レビューではありません。
提供範囲は[製品仕様](product_specification.md)、未実装拡張は[implementation plan](implementation_plan.md)、
式・method・永続形式はsolver/docs、変更手順と確認は[quality tooling](quality_tooling_plan.md)を参照します。

## 依存と公開境界

数値coreと通常clientは同じ独立uv project/distributionに収録します。
core/APIはclient、COMSOL、V&V、reportをimportしません。consoleとmodule入口だけが唯一のclient dispatcherへ接続します。

```text
外部CSV/解析/計測 → 入力publisher・場builder・cache → canonical SI bundle/case
公開API → prepare → engineの親時計・release・single writer
                         ↓ executionのworker・共有state・bounded ACK
                         ↓ transportの共通slab更新
                         ↓ fields / physics / integration / event / boundary
                         ↓ result + immutable context → analysis/export/view/V&V/report
```

計算APIの四操作は`load_case`、`check_case`、`simulate`、`open_result`です。
通常clientの公開操作は`chamber_particles_tools`が所有します。数値coreを再実装するCLI経路は持ちません。

## module所有表

この表がmodule責務の所有者です。別の文書へ同じ表を持たず、必要な行・受渡しへリンクします。

| owner | 所有する事実・処理 | 所有しないもの |
|---|---|---|
| `api` / `errors` | 四つの公開計算操作、公開例外三分類と遅延readerの同じ境界 | 数値式、client別error体系 |
| `case` / `yaml_input` | 設定の構文・正規化、SimulationCase、静的参照 | producer固有解釈、別実行validator |
| `case_format` | canonical read/write、SI schema、logical identity、layout/sourceのデータ表現 | 場生成・軌道計算 |
| `physics/catalog` | 選択model identity、scalar/field role、値制約、対応・proof要件 | mesh探索・式の複製 |
| `physics/forces` / `charge` | 局所の力・rate/Jacobian、固有の適用域・独立bounds | field locate、I/O、lifecycle |
| `physics/runtime` / `compiled` | role結線、連成準備、global/local証明、固定numeric stage recordと評価 | 親の説明・出力選択、第二model registry |
| `coordinates` | XY/RZの基底とaxis意味 | field検索、wall law |
| `sources` | canonical realized internal/surface schedule、release位置・時刻・weight | 分布sampling、source RNG、再飛散 |
| `rng` | 物理identityを持つcounter key、stream、uniform/normal | 壁や源のpolicy |
| `topology` | 静的XY translation periodicの対応とtransfer | 材料wall law、一般periodic framework |
| `fields` / `field_numerics` | locate、補間、support、同じ数値field owner | cache生成、mesh修復 |
| `geometry` | containment、BVH、first-hit候補 | 付着・反射、field値 |
| `integrators` / `stochastic` | 連成StepProposalとdense state、joint OU、条件付きtree | BVH、wall law、観測時計、I/O |
| `events` / `boundaries` | earliest hitと残時間work / hit後の物理応答とRNG | 親時計、writer、endpoint押戻し |
| `time_grid` / `lifecycle` | macro/knot併合・丸め・window照会 / wire code・名称・基本分類 | path精度認定、表示、物理付着の認定 |
| `cpu` | typed resident/persistent列、bounded workspace、容量とslab fit | 物理policy、永続file形式 |
| `transport` | serial/process共通のslab数値更新、accepted状態の反映 | process起動、親時計、writer |
| `execution` | worker起動・監視、共有transport、bounded IPC/ACK、cleanup | engine import、第二数値loop、worker writer |
| `engine` | prepare、親macro/release、accepted commit/checkpoint、single writer調停 | importer、plot、native比較 |
| `output` | durable result/context/checkpoint、publication/recovery、readonly ResultViewとidentity | 科学集計、表示、再sampling |
| 通常入力client / `inputs` | CSV/source/cache/縮約電場の変換・生成、完成case出版 | 物理validator、数値engineの複製 |
| `field_generation` | 明示した派生field群の依存・全output freshness | 依存の推測、物理係数の自動再生成 |
| `selection` / `analysis` / `visualization` | 束縛cohort、分母・coverage、解析/export/形状付き表示 | 元caseへのprivate依存、暗黙の再計算 |
| 唯一のclient CLI | console/moduleから同じ公開操作へのdispatchとI/O | 数値式、独立した検査経路 |
| 外部V&V / report | native観測・対応問題の評価 / reviewed結果の説明 | coreの物理定義、資格の一括認定 |

## 入力と検査

DataBundleがgeometry/layout/field/実現source、SimulationSpecが選択model・壁・時計・出力を所有する。
単一publisherがcanonical書込みとYAML hash結線を行い、新sourceに仮table/既存caseを要求しない。
load_caseは構文/hash/静的参照、check_caseはsimulateと同じprepareを使う。
prepareはgeometry/topology/required field/source/physics/event/数値path/workers依存memoryを解決する。
geometryとfield layoutは別のauthorityであり、同一meshを一般前提にしない。特定のpathを受理するためのmesh/support条件は[support and errors](solver/docs/support_and_errors.md)へ明示する。
CheckReportは小さいimmutable結果で、PreparedRun/worker/writer/workspaceを公開しない。
model metadataはcatalogを一度定義し、key検査・能力表示・manifestへ使う。stageでmetadataを生成しない。
fieldの値制約はfinite/positive/nonnegativeとして同じ宣言からfield準備・runtime・能力表示へ投影する。
prepare全域範囲、stage行別失敗、local interval証明は固有の検査であり、同じvalidatorへ統合しない。
選択descriptor identityとfield bindingをplanが保持し、compiledはfamily別固定numeric recordとsample済み配列を受け取る。
各modelはstage以外に必要なglobal/local proof roleを宣言し、compositionが連成と証明coverageを確定する。
未対応の証明をzero contributionと扱わず、constant specializationには全作用・state・noise・座標前提の肯定的証拠を要する。

recipe 2は完成bundleの全source table集合を明記し、既存table参照とCSV生成を区別する。未記載tableは保持しない。
settingsによる実release選択は別の役割である。caseと編集元、producerのbundleとreceiptは完成directoryとして一度に公開する。
source CSVは一度読んだbytesから解釈とhashを得る。宣言済み派生fieldのfreshnessは通常clientの
field_generationが一つの生成recordで照合し、canonical field/layout identityを使う。
未宣言の独立prescribed fieldへ依存を推測せず、publisherは新しい物理係数を自動生成しない。

## 状態・連成・容量

resident rowは固定IDと対応しrelease順/active indexと区別する。物理配列は一つの型で親/workerがviewを共有し、
親はclock/cursor/統計だけを加える。field/force/G/Jは実stageの位置/時刻/Zで評価する。
prepare後は親の入力/provenanceと解決済みactive facts、worker向けtransportを分ける。
resume identityと説明manifestは同じactive factsの別projectionであり、再開条件・資源・最終統計を無理に統一しない。
particle永続列はcpuの宣言に一度所有し、checkpointはそのsnapshotと親cursor/index/hintを組み合わせる。
accepted prefixだけがstate/cell hintをcommitし、hitと観測は同じproposalから求める。反射後はpost-impact stateから残時間を進める。
Brownianの条件付きsplitとwall/axis後fresh root、ID/物理interval/stream/ordinal RNGを維持し、worker/slab/保存時計を乱数authorityにしない。

workspace ownerがallocationと同じshape/dtype/capacityからbytes/private/shared区分を得る。
plannerはload/prepare/run同時live峰値とslab fitを所有し、context読込・writer transientも含める。OS RSS capとは異なる。
親coordinatorは全workerのconnection/sentinelを監視し、空いたworkerへ排他的slabを渡す。
workerは高々一batchのevent/failureを送り、親single writerの同期消費・ACK後にbufferを再利用する。
storageは到着順を許し、公開canonical順はoutput readerが解決する。macro全量の並べ替えbufferは置かない。
private scratch templateは全workerのcopy完了ready barrier後に解放し、transportをsealする。
非空replayはrun所有のshared rootとworkerのreaderを使い、全task/output消費後にviewを解放する。
曲線/exact eventの残時間workは数値配列に持ち、各rowのtarget、residual、refinement、物理ordinalを保持する。
汎用allocator/DI/registryや別schedulerを作らない。具体的な資源計画と測定は[parallel execution](solver/docs/parallel_execution_plan.md)が所有する。

## 結果と変更単位

必須contextはcanonical geometry、SI/基底、resolved source/boundaryの既存IDと意味、axis/contact/periodic対応を一度保存する。
同じserializer/validatorを使いfile/logical hashをmanifest/checkpointへbindする。resumeでは検証して解放し再生成しない。
初期全配列やfield/traceを保存contextへ重ねず、contextだけで再計算可能とはしない。
ResultViewのmetadataは再帰的に不変とし、outputが正規化identityとJSON用独立documentを提供する。
通常clientは同じResultViewへ束縛したcohortを再利用し、整列済みIDを索引として一度解決する。
analysis/export/viewは観測意味とcoverageを維持し、export/viewは完成物だけを公開する。
result5は全commit済segmentのdigestをcheckpointへ保持し、再開・sealで全prefixを照合する。
読取りはopen済の同じhandleでhashとHDF5解釈を行い、再openでは再検証する。
固定identityと消費したfileの検証範囲を分け、全payload検査は公開verify_contentsで明示実行する。
同時in-place書換えのsnapshotや科学的精度は、このbyte integrityの保証に含めない。

既存categoryの追加はcatalog→式/bounds→stage binding→compiled合成→独立検証で閉じ、engineを触らない。
新categoryでは合成順/全bounds/constant証明/容量/結果説明も確認する。新stateは初期値からstage/dense/event/checkpoint/resultまで縦切りを要する。
wall lawは応答/RNG/manifest/保存則/残時間検証を変え、first-hit処理を複製しない。削減は旧selector/経路/呼出側/設定/tests/docsを同時撤去する。
exact/curved/OU、global/local bound、support/誤差認証は固有の役割があり、似たcodeだけを理由に統一しない。
production数値規則は一owner、独立式oracleはverification/V&Vに保持する。
macro grid設定・実partition・accepted coefficient root/path・certificate細分・観測は別の事実である。
engineと外部評価は同じ時計規則を消費し、評価windowの実際のrefinement前提を確認してから差を解釈する。
macro前提だけではaccepted path収束やBrownian強収束を証明しない。用途に対する実用比較と数値資格は別に判断する。

## 検証と履歴

解析/製造解、独立確率reference、時間/場/event収束、公開scenario、installed-wheel一巡を組み合わせる。
旧出力一致は構造変更のidentity検査で、唯一の正しさ判定ではない。
式/schema/性能資格は各ownerへリンクし、architectureへrevision一覧や進捗diaryを重ねない。以前の詳細・採否はGit履歴とimmutable evidenceに保持する。
