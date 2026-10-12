# 半導体チャンバー粒子軌道計算基盤：アーキテクチャ

## 0. この文書の位置づけ

本書はmodule責務、依存方向、runtime sequenceの詳細設計である。製品目的、実装範囲、
coreと外部toolの境界は [product_specification.md](product_specification.md)、具体的な実装順とwork packageは
[implementation_plan.md](implementation_plan.md) を正とする。本製品はCOMSOL比較基盤ではなく、
外部場を利用する独立した高速粒子輸送solverである。`model_dataset` は外部V&V資産として用い、
不備があればCOMSOL再実行・再抽出・version更新を行う。

Python環境、static quality tool、CI gateは [quality_tooling_plan.md](quality_tooling_plan.md) を正とする。
architectureを機械検査へ写す時も、少数のstable import ruleだけをimport-linterへ置き、設計書全体を
contract化しない。

既存`particle_tracer_unified`は実装母体にせず、継承するのは再検証できる数式、入力の由来情報、
失敗事例、少数の設計原則だけとする。既存のディレクトリ構成、実行経路、診断フレームワーク、
テスト構造は新基盤へ持ち込まない。

principal architecture／数値計算／performance reviewの指摘と採否は
[architecture_review.md](architecture_review.md) に記録した。以後、本書で決めるものは次に絞る。

- 製品として何を「COMSOL代替」と呼ぶか
- 粒子、場、メッシュ、境界、電荷、乱数、結果の標準表現
- 一つのengine内で複数のintegrator strategyを扱う数値方式
- 2D、軸対称、3D、定常場、時間依存場への拡張経路
- 10,000～1,000,000粒子を想定したCPU/GPU・メモリ戦略
- 機能追加によって再び複雑化しないコード配置と開発規律

データ品質の詳細は [data_quality_assessment.md](data_quality_assessment.md)、比較手順は
[vv_methodology.md](vv_methodology.md)、機械可読な監査結果は `evidence/` を参照する。

---

## 1. 経営・開発上の結論

### 1.1 実装方針

独立uv projectとして再構築したsolverを維持する。旧solverはread-only archiveであり、
新productionの実装母体・環境・dependencyにしない。

解析解・manufactured・収束で確立したXY/RZの数値pathへ、既存charge/force/Brownian、
静的boundary/periodic、時間場、bounded output/processを同じengineで統合する。
現在の能力は第3.3節、実行境界は第3.4節、適用域はmodel/数値仕様、用途資格はsource固定evidenceが所有する。
COMSOL比較、adapter、field builder、analysis/可視化はcore外に保ち、実装順や物理modelをCOMSOLの個別caseに従属させない。

完全3D、GPU、別物理は独立利用caseとreferenceを用意して追加する。
完了済み初期stageや過去の試作順を、次のreleaseの必須工程として残さない。

### 1.2 COMSOLからの置換可能性

本製品がCOMSOL利用者の実務を置換できるかは、coreの定義ではなく外部評価として次を公開する。

| 適合レベル | 意味 | 合否の根拠 |
|---|---|---|
| 設定互換 | 同じ入力物理・境界・刻み・乱数方針を表現できる | capability matrix とcase manifest |
| 数値互換 | 決定論microcaseと対象ケースで軌道・力・電荷・イベントが許容範囲内 | versioned V&V report |
| 統計互換 | Brownianや確率付着を同一分布と判断できる | replica、信頼区間、分布距離 |

adapterが変換できないCOMSOL featureは外部変換時に拒否する。coreはCOMSOL feature名を知らず、
一般的な物理modelとsolver設定だけを扱う。全COMSOL機能との一般的な互換性は主張せず、外部V&Vが
検証済みのモデル、座標系、要素次数、物理、境界、versionを表にする。

### 1.3 性能目標の扱い

2026-10-12の利用者指示で追加JIT化を停止し、現行数値ownerを変更せず同一runのCPU process並列を優先する。
親はmacro時計・release・出力・checkpointを所有し、永続workerは排他的なglobal particle rowを既存slab関数で進める。
入力とresident/replayを共有し、mutable scratchはworkerごとに独占する。全taskの成功と同期出力handoffを消費した
macro barrierでだけcheckpointをcommitする。旧thread poolやNumba parallel maskを復活させない。
現行提供範囲はimplementation plan第21.4節、条件付き将来項目は第17節・第21.3節が所有する。

「COMSOLより高速」は設計目標であり、実測前の仕様保証にはしない。同一の物理、時間
刻み、出力頻度、粒子数で、cold/warm、計算、イベント、I/O、メモリを分解して測る。
10^4、10^5、10^6粒子の実ケース性能が揃ってから製品主張に変える。

---

## 2. 監査から確定した出発点

### 2.1 `model_dataset` の構造

新しい評価軸は12ケースで構成される。

- ion drag式：`formal_iondrag_theory_consistent` と
  `formal_iondrag_image_minimal_corrected`
- 場の作り方：保存プラズマ場を用いるCase P、熱流体結果からnonlinear SASS静電場を
  構成したCase A
- 粒径：10、30、100 nm
- 各ケース：287粒子、121保存時刻、0～0.03 s
- 座標：2D axisymmetric r-z、粒子domain 3
- 積分：固定刻み10 µsの陽的RK4
- 形状：7,732頂点、726境界edge、10,481三角形、2,426四辺形、47境界ID

入力と結果の内部整合性は高い。12ケースすべてで粒子・時刻キーの重複はなく、初期表は
軌道t=0と一致し、最終表は最後の保存行と一致する。速度、電荷、半径、質量、力/質量の
関係も浮動小数点誤差内で一致した。詳細な測定値は
`evidence/dataset_audit.json` と `evidence/case_audit_summary.csv` に保存した。

### 2.2 評価軸としての限界

この12ケースだけでは製品要件を検証できない。

- 発生点はパーツ表面ではなく内部の規則格子である。
- 反射は無効で、freezeの正例もなく、確率付着もない。
- 定常2D軸対称場のみで、時間依存場と3Dを含まない。
- Brownianの内部乱数増分とRK4各stageは保存されていない。
- COMSOL有限要素の自由度、形状関数、要素次数を完全には保存していない。
- mesh-point場は33,449行だが、一意座標は7,732点で25,717行が重複する。
- 規則格子座標には微小な浮動小数揺らぎがあり、丸め前のunique数は301ではない。
- `inside_model_domain` は粒子domain 3の権威あるsupport maskではない。
- 背景ファイル中の粒子派生列は各粒径向けに更新されていない。

従って、このデータは「COMSOLが保存した軌道・局所物理・状態のsystem regression」として
利用し、境界や確率過程の正しさは別の小型ケースで検証する。

### 2.3 時間軌道を見なければ見逃す差

同じCase・粒径で2種類のion drag式を比較すると、Case Pの10/30 nmは最終状態が全287粒子
で一致する。しかし1 ms時点の位置差中央値はそれぞれ約14.8 mm、4.40 mmで、全粒子が
一度は1 mmを超えて分離する。最終状態だけなら「一致」と誤判定する例である。

Case Aでは最終状態そのものも大きく異なる。最終status不一致数は10/30/100 nmで
181/203/158粒子である。この測定は `evidence/variant_divergence_time_series.csv` と
`evidence/variant_particle_divergence_summary.csv` に再現可能な形で保存した。

### 2.4 既存コードから残すもの、捨てるもの

残す候補は、すべて独立検証後に再実装する。

- SI標準化、`mass_kg`を慣性の権威とする考え方
- 入力の由来・hash・座標系をmanifestへ残す考え方
- 場の値を返すことと、その点が厳密にsupport内かを分離する考え方
- geometry scaleと浮動小数ULPから境界数値を決める考え方
- 線形dragの安定な指数更新式、法線・接線への壁速度分解
- 粒子ID・step・componentで決まるcounter-based RNG
- load_case/simulate/open_resultの公開境界、simulate内部のatomicな成果物書込み
- COMSOL adapterと比較器をsolver coreの外へ置く原則

捨てるものは次である。

- 現行のruntime orchestration、collision/refinement/contactの経路
- 物理名による巨大な分岐とbackendごとの重複実装
- 任意物理を有効にするとPython粒子loopへ落ちる実行方式
- accuracy不足を境界停止へ変換する状態機械
- exact-keyの位置だけを比べる比較器、最終snapshot中心の診断
- private helperの配置や文書ファイル数を固定するテスト
- 2～3粒子だけの性能基準、過去負債を合格扱いするbaseline運用

---

## 3. 製品境界と非目標

### 3.1 製品が所有する責務

新基盤は、外部で求めた場を受け取る一方向連成Lagrangian point-particle solverである。
製品が所有する処理は次である。

1. 標準caseの読込みと一度だけの意味検証
2. メッシュ内点・support・最初の境界交差の照会
3. 場の空間・時間補間
4. 粒子の力、電荷、確率過程の評価
5. 運動・内部状態の時間積分
6. 壁イベントと粒子lifecycle
7. 大規模粒子群のバッチ実行とstreaming出力
8. solver固有の最小限の実行統計

### 3.2 coreに入れない責務

- COMSOLのstudy名、式名、CSV quirks、version別回避策
- MPHの操作やCOMSOL Java exporter
- COMSOLとの差の診断・可視化・HTML report
- CAD修復、一般purpose mesh generator
- 流体・熱・プラズマの本格的な自己無撞着solver
- 旧設定の自動migration
- GUI固有の状態

これらはadapter、field builder、validation tool、applicationという外側のパッケージに置く。
ただし「熱流体場＋任意のプラズマパラメータから静電場を作る」機能は、solverとは別の
`electrostatic_builder` として公式に提供する。

### 3.3 現行0.3.0の能力と適用境界

| 軸 | 現行能力 | 未実装・別scope |
|---|---|---|
| 座標 | Cartesian XY、RZ meridional | 軸対称場中Cartesian 3D粒子、完全3D |
| 場 | static/固定topology linear-time、regular/P1/Q1 | 高次要素、moving topology、部分supportの一般化 |
| 積分 | ballistic/一定加速度exact、一般固定RK4、exponential midpoint、joint inertial OU | 一般適応ODE/DAE、GPU方式 |
| 物理 | 対応drag、Coulomb、gravity/buoyancy、fixed/continuous charge、ion drag、熱泳動、DEP、lift | 未指定のmodel、適用域外の暗黙近似 |
| 確率 | XY/RZ Brownian、active wall restart、一様base depthと条件付き追加refinement | 連続OU exact first-passage、3D Brownian |
| 壁/topology | stick/escape/hold/specular/restitution/Maxwell/probabilistic stick、独立接触半径、静的XY translation periodic | moving geometry、resuspension、surface charging、RZ/3D periodic |
| 実行・出力 | 同じengineのserial/process、bounded slab、共有transport、single writer、checkpoint/resume | thread/backend選択、複数macro同期間隔設定、remote store |

これは全組合せの受理を意味しない。対応matrixとmodel適用域をprepareで一度解決し、未対応を拒否する。
case/result schemaと完全なalgorithm rosterは[case形式](solver/docs/case_format_v3.md)と
[result形式](solver/docs/result_format_v3.md)、数値pathの詳細は第6～10節と
[数値仕様](solver/docs/numerics.md)が所有する。旧engine revisionの逐次実装履歴を現行仕様として重ねない。

scalar evaluatorはverification oracleでありproduction fallbackではない。一般RK4ではglobal support/model
boundを先に試し、必要なrowだけ同じproposalのdense pathとlocal cell rangeでcertificateを得る。
certificate-only restrictionをhidden数値substepと混同しない。accepted pathのfirst hitとvalid prefixを調停し、
同じintegratorでhit状態を再評価してpost-wall残時間を進める。

位置押戻し、field-support bypass、wall/axisの混同は行わない。sourceはrealized internal/surface schedule、
接触は独立半径とgroup別surface/center意味を共有する。terminal rowは通常更新を止めsnapshotを保持する。
output sampleはaccepted pieceから評価し、production partitionを変えない。

### 3.4 現行の実行境界：既存数値ownerによるprocess並列

0.3.0は既存数値ownerを使うv62のprocess実装を提供する。追加JIT化は利用者指示で停止した。
現在の資格・提供範囲は[implementation_plan.md](implementation_plan.md)第21.4節、
source別の科学・性能・memory・品質・配布資格は[性能文書](solver/docs/parallel_execution_plan.md)が所有する。

engineは一つで、workers=1と複数workerが同じslab数値処理・physics/integrator/event/RNG ownerを使う。
親はprepare、macro時計、release、active compact、replay確定、single writer、durable checkpointを所有する。
run内で永続するspawn workerは排他的global rowとprivate scratchを所有し、fileへ直接書かない。
prepared field/geometry/model/lawはread-only、state/hint/replayは担当rowだけを更新する。

親は全connection/sentinelを監視し、完了workerへ次slabを渡す。各workerのevent/failureは一batch ACKで
容量を制限し、親writerの同期消費後にACKする。到着順storageを許し、公開canonical event順は既存readerが所有する。
全taskとoutput消費の完了を**1 macroステップごと**に待つ。RK stage/OU leafはworker内で処理する。
起動ready、event/failure ACK、終了joinは別の待機で、現在は全worker同期間隔の設定を持たない。
checkpointは全task/outputを消費したaccepted macro/epoch barrierだけでcommitする。

全子のscratch copy後に共有templateを解放してprepared transportをsealする。非空replayはrun所有の
一shared rootと永続readerを使い、consumer完了後にmacro/slab viewを解放する。
serialization/handoffの同時liveとpage paddingも総memory planへ含める。
owned array削減、総OS memory、JIT/private allocator、NUMA配置の資格を区別する。
child突然死は未完runとして扱い、worker/SHM cleanupと最後のdurable checkpointからのresumeを保証範囲で検証する。

worker数はAPI keyword/CLIで指定し、数値設定やRNG keyへ混ぜない。
旧ThreadPool、Numba内部parallel mask、model別engine、backend framework、第二schedulerを追加しない。
複数macro連続実行は将来項目とし、dt/RNG/output/checkpointの不変条件は実装計画第21.3節に置く。

用途別性能とWindows/Linux local gate、runtime-only installed wheelは同じproduction sourceへ結合して完了した。
Case A/P少粒子full-stepでは並列遅化、P14-U 1万粒子320stepではW1/W4=1.964倍を観測した。
総時間や粒子数だけで高速化を保証せず、主matrix・補足・個別caseのscopeを分ける。
Git push、remote CI、tag、Release/registry公開の実施状況はrelease記録が所有する。
local Linux/WSLのfunctional資格をnative20worker scalingへ拡張しない。

## 4. 論理アーキテクチャ

### 4.1 一方向の依存関係

```text
COMSOL / CFD / plasma / CSV / 計測 / user mesh
                       │
                       ▼
             tools/importers + builders
                       │ case_format.write
                       ▼
          Canonical DataBundle ── SimulationSpec
                       │
                       ▼
                  load_case
                       │ SimulationCase
                       ▼
                 engine.prepare
                       │ PreparedRun（非公開）
                       ▼
              single production engine
      source → integrator/physics → event → wall law
                       │
                       ▼
                 canonical ResultStore
                 ┌─────┴──────────┐
                 ▼                ▼
              analysis       external V&V
                 │                └─ COMSOL
                 ▼
            visualization
```

source adapterはDataBundleまで、coreはSimulationCaseからResultStoreまでを扱う。COMSOL比較器は
通常のResultViewと参照結果を扱う。core内にCOMSOL列名、study、state aliasを入れない。

### 4.2 推奨ディレクトリ

```text
particle-platform/
├─ pyproject.toml
├─ src/chamber_particles/
│  ├─ __init__.py
│  ├─ api.py
│  ├─ case.py
│  ├─ yaml_input.py
│  ├─ case_format.py
│  ├─ coordinates.py
│  ├─ geometry.py
│  ├─ fields.py
│  ├─ sources.py
│  ├─ physics/
│  │  ├─ catalog.py
│  │  ├─ forces.py
│  │  ├─ charge.py
│  ├─ integrators.py
│  ├─ events.py
│  ├─ boundaries.py
│  ├─ engine.py
│  ├─ cpu.py
│  └─ output.py
├─ tools/
│  ├─ importers/comsol/
│  ├─ importers/tables/
│  ├─ case_builder/
│  ├─ electrostatic_builder/
│  ├─ field_preprocessor/
│  ├─ analysis/
│  ├─ visualization/
│  └─ vv/comsol/
├─ tests/
│  ├─ verification/
│  ├─ scenarios/
│  └─ performance/
└─ docs/
```

これは1機能1ファイルを強制する図ではない。小さなhelperを別ファイルにせず、意味のある所有者へ
まとめる。一般型置き場の`models.py`やre-export専用moduleは作らない。moduleが二つの独立した変更
理由を持つほど大きくなってから分割する。module ownershipと依存方向は本書を権威とし、
[architecture_review.md](architecture_review.md) は判断理由と指摘のreview記録として読む。

### 4.3 公開API

```python
from chamber_particles import load_case, simulate, open_result

case = load_case("case.yaml")
summary = simulate(case, output="runs/case_001")
result = open_result("runs/case_001")
```

- `load_case`：先にYAML全体とresource設定をparseし、HDF5 metadataからcanonical numeric footprintを
  求めて上限外をpayload展開前に拒否する。その後、既にSIへ正規化されたcanonical dataの
  schema、unit metadata、局所connectivity、owner参照、静的参照整合を検査する。source unit変換はadapterの責務である。
- `simulate`：prepareでmodel requirements、coordinate/integrator/backend capability、compiled
  evaluator、大域geometry topology、memory planを解決し、数値計算をstreaming出力する。
- `open_result`：final、event、保存trajectoryを遅延読込みする。

`PreparedRun`とatomic writerは内部実装であり、公開APIにしない。独立した巨大なpreflight subsystemも
作らない。`load_case`と`simulate`内部準備で入力エラーを一度だけ検出し、詳細監査は外部tool側で行う。

### 4.4 主要データ型

```text
DataBundle
  GeometryDomain
  FieldLayout / FieldSet
  boundary groups/materials
  provenance/content hashes

SimulationSpec
  source/particle/physics/boundary/time/resource/output

PreparedRun (immutable, non-public)
  coordinate_kernel
  field_sampler_plan
  PhysicsPlan + StateLayout
  integrator_profile
  event_locator_plan
  memory_plan
  output_plan

ParticleState (SoA)
  identity: id, source_id, rng_key
  physical: x, v, mass, diameter, charge/internal state
  lifecycle: pending, active, stuck, escaped, failed, held
  geometry: cell_id, last_boundary_id
  failure_reason: uint16
```

accuracy、support、geometry判定はstep-local resultとrun counterで持ち、全粒子へ常設配列を増やさない。
失敗時だけ`failure_reason`へ確定する。精度budget枯渇を壁停止や粒子消失へ変換しない。

---

## 5. Canonical DataBundleとSimulationSpec

### 5.1 物理量は原始量だけを保存する

背景場には、粒子を変えても不変な原始量だけを保存する。

- gas velocity、temperature、pressure、density、viscosity、mean free path
- electric potential/field、density、species state、temperature gradient、vorticity
- 場を作ったsolution/time/parameter index

次は粒子状態と原始場からruntimeで求める。

- Knudsen数
- Epstein response time
- Brownian diffusivity
- Reynolds数、drag coefficient
- charge依存force、screeningに粒子情報が入る派生量

`model_dataset`では背景CSVの10/30/100 nm版が数値的に同じで、粒子派生列が実質10 nm相当
のまま複製されている。これを「入力値」として信じる設計は、粒径を変えても一部物理が
変わらないsilent errorを生む。

### 5.2 YAMLとHDF5の所有を分ける

`case.yaml`は人が変更する`SimulationSpec`と、対応するHDF5への参照だけを持つ。
文法解析は`yaml_input.parse_document(bytes)`に一元化し、全階層の重複keyとmerge keyを拒否する。
このleafはI/O、hash、domain検査を持たず、`case.py`と外部producerが各domainの検証を所有する。
通常のaliasはdomain検査へ渡し、provenanceは元のUTF-8 bytesから計算する。

```yaml
format_version: 2
case:
  name: chamber_screening_001
  data_path: case.h5
  expected_content_hash: "sha256:..."
time: {...}
solver: {...}
resources: {...}
physics: {...}
sources: [...]
boundaries: [...]
output: {...}
```

`case.h5`だけが座標系、SI単位、geometry topology、field layout/value/support、boundary
group/material、realized table source、producer provenanceを所有する。YAMLにこれらを再記述しない。
逆に、HDF5は物理model選択、時間刻み、壁面law、resource、出力scheduleを持たない。

canonical必須の由来情報はHDF5の`/meta/provenance_json`に保存する。v1のtop-level必須keyは
`producer`、`producer_version`、`source_sha256`、`field_semantics_revision`、
`producer_metadata`とする。COMSOL version、study/dataset、smoothing/recovery等は
`tools/importers/comsol`が`producer_metadata.comsol`へ条件付きで保存し、他solverや計測dataへ
必須にしない。quantityの単位・成分・座標基底は各field record、node/element orderingと
evaluation policyはversioned schemaが所有し、provenanceと二重化しない。

### 5.3 ファイル形式

- `case.yaml`：人が編集する設定、model選択、HDF5 path、少量metadata。
- `case.h5`：mesh、field、particle/source table、boundary metadata。chunk、compression、dtypeを明示。
- result：JSON manifestと閉じたHDF5 epoch segment。詳細は§11.5と主仕様§14。
- CSV：adapterの入口・人間確認用。coreの高性能native形式にはしない。

NaNを材料領域や境界の唯一の表現にしない。geometry topologyとdomain IDをsupportの権威に
し、各fieldは「このdomain/timeをsupportする」という独立maskを持つ。NaNは欠損値として
扱い、位置判定には使わない。

### 5.4 adapterでの正規化

全adapterはSI、quantity semantics、topology ID、support、provenanceを確定してから
`case_format.write()`を呼び、coreへproducer固有の曖昧さを渡さない。COMSOL adapterでは特に次を
完了する。

1. SIへ変換する。
2. COMSOL node/element/boundary IDを保持してdense local indexを別に作る。
3. 重複mesh-point行をnode IDで統合し、同一点の値衝突を検査する。
4. 新しいexportは座標の浮動小数丸めjoinではなく、topology IDで値とmeshを結ぶ。
5. 物理量名をcanonical quantityへ一度だけmapする。
6. domain/boundary selectionをmanifestへ固定する。
7. 派生物理列を除去し、必要なら監査用参照値として別namespaceへ隔離する。

既存producerがfield側のnode IDを出力できない場合に限り、adapter境界でcoordinate-only legacy入力を受けてもよい。
ただし、明示した物理長tolerance内で各canonical nodeにprovider点が厳密に1個、各provider点も厳密に1回だけ対応する
全単射を証明し、最大対応距離と入力hashをprovenanceへ残す。補間、丸めbucket、最近傍fallback、欠損埋めは許さない。
これはF02の既存CSVを移行する限定例外であり、新しいexporterの契約を弱めない。

### 5.5 設定モデル

物理ファイルとsweep設定を次へ分ける。

- `DataBundle`：geometry、field layout/value、境界タグ、由来など再利用する事実。
- `SimulationSpec.physics`：有効な力と電荷model。
- `SimulationSpec.sources`：canonical HDF5内のrealized internal/surface table参照。
- `SimulationSpec.boundaries`：境界groupごとのwall law、priority、law parameter。
- `SimulationSpec.run`：時間、積分profile、backend、resource、出力、seed。

物理moduleの選択は明示的にし、依存するfieldがなければ`load_case`または`simulate`内部準備が
拒否する。パラメータの
defaultと外部solver由来の値を混ぜない。COMSOL adapterは一般的なphysics/run設定へ明示展開し、
元のCOMSOL設定とversionは外部V&V manifestにだけ記録する。

---

## 6. 座標系、geometry、mesh

### 6.1 共通抽象と専用kernel

「すべてを3Dベクトルとして同じコードで解く」方式は、軸対称の意味論と高速化を曖昧に
する。一方で座標系ごとに製品全体を複製しても保守できない。共通にするのは状態遷移、
field/force/eventのinterface、出力schemaであり、hot loopは次の専用kernelとする。

- `cartesian_xy`
- `axisymmetric_rz_meridional`
- `axisymmetric_field_cartesian3d`
- `cartesian_xyz`

この列挙はparticle motion modeであり、入力dataの座標表現ではない。canonical HDF5はgeometry/fieldの
`cartesian_xy | axisymmetric_rz | cartesian_xyz`を所有し、SimulationSpecはparticle motion modeを所有する。
`engine.prepare`が両者の対応を一度だけ検査し、result manifestへ両方を記録する。現在の未release v1では
XY/XYとRZ/RZ-meridionalだけを有効にし、Stage 2AでRZ data/Cartesian3D motionをschema revisionとして追加する。

`axisymmetric_rz_meridional`では状態は `(r,z,v_r,v_z)` で、方位運動を持たない。`r=0`は壁ではなく
座標seamである。trial pathを軸で分け、`r <- -r`, `v_r <- -v_r`に相当する基底変換を
`coordinates.py`だけが行う。wall/reflection eventには数えない。
force-coupled RK4ではtrialだけをsigned radial chartで進め、stageごとにcanonical RZへ写してfield/physicsを
評価する。accepted stateは常に`r >= 0`である。axis accessibilityは、geometryのaxis接触、またはboundaryless
fully-supported regular boxの`r_min=0`から`fields.py`が一度だけ導き、axis nodeのvector regularityに使う。
`gravity_buoyancy_standard_v1`はaxis accessibilityと無関係に、すべての`axisymmetric_rz` domainで
`gravity_m_s2[0] = g_r = 0`を要求する。`cartesian_xy`では同じ配列の第1成分`g_x`を非零にできる。
P05当時はaxis path分割をproduction loopに接続せず、軸到達が
最初の材料hitより先になり得るpathを拒否したが、exact pathはP07、一般RK4はP06-RZで解禁済みである。
両端`r=0`のvolume外周edgeはaxis seamとして材料BVHから除外し、
材料boundaryとして登録した入力も拒否する。
`axisymmetric_field_cartesian3d`では背景場だけを`r=sqrt(x²+y²)`で評価し、粒子は
Cartesian 3D状態を持つ。等方Brownian、方位速度、磁気力が必要な場合はこちらを使う。
円柱座標swirl専用kernelは初期scopeへ入れず、実caseと独立検証が揃った後の候補とする。

### 6.2 2D軸対称から3Dを再構成する責務

軸対称場からの3D表示・source展開は、場を3D voxelへ複製する処理ではない。

- 場のsampling：`r=sqrt(x^2+y^2)`で2D場を評価し、vector成分をCartesian基底へ回転。
- sourceの3D展開：外部入力作成側が面積要素`2πr ds`でsurfaceを重み付けし、`theta~Uniform(0,2π)`を
  realizeした粒子rowをcanonical入力へ書く。
- 軌道の3D表示：meridionalなら初期thetaを保持してtrajectoryを回転表示。
- 統計：2D点数をそのまま3D面積密度とみなさず、2πr weightを使う。
- 境界：3D pathを`(r(s),z(s))`へ写した曲線とRZ境界のfirst hitをbracket/localizeし、
  RZ法線を衝突点thetaでCartesianへ回転する。

座標変換は`coordinate_kernel`、確率分布のrealizeは外部producerが所有する。geometry importer、field補間、
visualizer、solver coreの各所で同じsampling policyを別々に実装しない。

### 6.3 GeometryDomainとFieldLayout

衝突geometryとfield離散化は別の型にする。同じFE meshの時だけstorageとlocatorを共有する。

`GeometryDomain`は次を保持する。

- vertex座標とstable external ID
- 粒子domainまたはvolume connectivity
- oriented boundary facet、boundary/material ID、owner、向き
- 全体scale、AABB、局所element scale
- boundary/containment用BVHとadjacency

`FieldLayout`はregular axesまたはunstructured vertex/element、basis、support domain、time axis、
point-location indexを持ち、`FieldQuantity`がlayout IDを参照する。一つのrunに複数layoutを許せるが、
architecture上はgeometryとfieldを常に独立の事実として扱う。Stage 1の初期実装能力は同じP1/Q1 mesh、
または明示選択された検証済みregular layoutへ限定できるが、型やIDを同一視しない。同じlayout groupの
quantityだけ一回のelement lookupを共有する。この制限はtemporary capability checkであり、producerや
mesh identityの定義ではない。

三角形P1はbarycentric、四辺形Q1はisoparametric bilinear形状関数を使う。四辺形を暗黙に
三角形分割すると、COMSOL Q1補間とは別の場になるため禁止する。P2/高次要素は、実modelの
node orderingと自由度を固定したcaseが揃うまで拒否し、頂点だけをP1と偽らない。

現`model_dataset`のquadはCSV列順`[node1,node2,node3,node4]`が周回順ではなく、正しい周回順は
`[node1,node2,node4,node3]`である。列順でpolygon化すると2,426要素中2,019がbow-tieまたは
面積ほぼ0になる。adapterはこの順序をversioned import ruleとして明示し、全要素の向き、
自己交差、Jacobianを検査する。座標からの自動並べ替えでDOF対応を失わない。

### 6.4 point locationと粒子追跡

P10/P14のregular locatorはsupported containing-cell common pathだけ軸indexからO(1)個の候補を評価する。
outside/masked provisionalは物理最近傍意味論を守るためcompiled全cell走査を使う。P1/Q1はaccepted endpointのprevious-cell
hintがstrict interiorを含む場合だけfast pathで採用する。共有面は最小supported cell IDというowner規則を
守る。P14ではhint missと初回sampleのsupported containmentだけをfield-owned stackless BVHで候補化し、同じexact
predicateをID順に評価する。containing supported cellがなければoutside/masked最近傍full scanへ戻る。trial hintは
resident stateへcommitしない。

P03のscalar locatorは正しさを固定するverification oracleとして残すがproduction fallbackには使わない。
unstructured indexはP10へ入れず、初回stageのfallbackを稀と仮定しなかった。P14のrealistic cell count、initial
localization、cross-cell motionで支配的と確認したため上記一つのindexへ置換した。別samplerやruntime fallbackを作らない。
inside判定は参照座標への固定ULPだけにせず、物理空間の後退誤差、局所element scale、座標ULP、
Jacobian conditioning、再構成残差を一つのalgorithm revisionで扱う。悪条件cellのinside領域を無制限に
広げず、revisionが根拠とともに固定するmesh品質上限超過と非有限補間は明示errorにする。この上限を
利用者向けtuning knobにしない。

table sourceのstrict-interior検査はboundary-first分類を維持し、その後のvolume union containmentだけを
`geometry.py`所有のmixed tri/quad BVHで候補化する。AABBはbroad phaseで、既存half-space predicateがauthorityである。
局所diameterに対してedge/非incident vertex clearanceがfloat64で解像不能なcellはprepareでfail-closedとし、
絶対座標が大きいだけでは拒否しない。edge長はscalar/compiled parityのためCPython `math.hypot`でprepareする。
この内部index、field index、build toleranceをcase schemaやproducer設定へ露出しない。

境界交差も全edge/triangle走査を行わない。

1. 予測segmentのAABBをBVHへ照会。
2. 候補facetとrobustなsegment intersectionを行う。
3. 最小の正のhit parameterを選ぶ。
4. corner/edgeで複数facetが同時なら、versioned tie-break policyを適用する。

数値許容差は固定m値や散在する`1e-12`ではなく、mesh scale、facet scale、速度、時間、
float64 ULPから`GeometryNumerics`として一度解決する。位置を内側へ微小移動して判定を通す
repairは行わない。

### 6.5 surface release

製品の主用途なので、後付けの初期位置処理ではなく第一級のsource型とする。

```text
RealizedSurfaceSource
  particle_id[N]
  facet_id[N] / facet_parameter[N]
  release_time_s[N] / velocity_m_s[N,2]
  particle_properties[N]
```

分布、重み場、時刻schedule、粒径・質量・電荷分布、角度分布は外部入力作成側の責務とする。coreは
内部sourceの`position_m`またはsurface sourceの`facet_id + facet_parameter`と、共通の粒子ごとの物理量を
一度だけruntime scheduleへ変換する。source生成frameworkをcoreへ追加しない。

RZの3-D回転面surface samplingは`2πr ds`、2-D断面用の`meridional_length`は`ds`、3D triangleは面積で
重み付けする。canonical surface rowはfacet IDとstrict interiorのfacet内座標を保持し、runtime scheduleへ
変換する時にprepared geometryからpositionを一度導出する。runtimeはそのpositionとcanonical source facet IDを
保持し、ownerとnormalはprepared geometryが所有する。residentなbarycentric/local-coordinate authorityを
重複して持たない。壁からの出発は「接触中だが
流体側へ離れるsegment」としてevent locatorが扱い、任意のinward epsilonで位置をずらさない。
発生面の法線向きと粒子domain側をimport時に検証する。

engine v10では`v·n`のscale-aware符号を最初のdeparture authorityとする。明確な内向き/外向き速度を
加速度で上書きしない。budget内の非零法線速度は曖昧として拒否し、厳密tangentだけを証明済み一定加速度の
`a·n`で分類する。内向き加速度のdeparture tokenは別wall hitで速度が変わるまで保持する。`events.py`は各intervalで
source supporting lineの内側/境界band、tangentまたは明確な内向き速度、明確な内向き加速度を再証明し、成立時だけ
source facetを除外する。他facetは常に通常のfirst-hit authorityを保ち、別wall応答後はsource facetも通常検索へ戻す。
外向き加速度はzero-time impactであり、terminal応答は許すが反射後に内向きdepartureを作れなければ失敗する。
releaseがmacro終端ならtokenを次macroへ渡し、run終端ならright-continuousなrelease frameだけを保存する。
各intervalで再証明するため、微小`dt`を含むmacro partitionで物理eventを変えない。

surface粒子のIDはcanonical rowに明示し、tableおよび他surfaceとの重複をcase load時に拒否する。
確率分布が必要なら外部入力作成側でrealizeし、coreのseedや実行partitionへ依存させない。

### 6.6 geometryの異常判定

次は警告ではなくcase build errorにする。

- non-manifold boundary、zero-area element、自己交差
- 粒子domainにownerを持たないwall facet
- boundary selectionの重複でpriority未定義
- source facetの法線向きが不明
- support islandやholeの意味がmanifestにない

検査ownerを分ける。`case_format.read`はindex範囲、局所node順、boundary rowと宣言owner edgeの整合だけを
検査する。`geometry.prepare`はvolume incidenceから大域外周を導出し、non-manifold、重複boundary、
内部edgeのwall登録、RZ axis seam以外の外周欠落を拒否する。現行は内部thin baffleを未定義のまま受理しない。
修復やfacet生成は外部builderに残し、coreは判定と拒否だけを行う。
P05で一つ以上の材料facetを与える場合、このtopology-completeな`line2`を必須とする。
boundary rowが0件の明示的なcollision-free解析caseは、同じengineのno-hit profileとして残す。

geometryとfield topologyが異なること自体はerrorではない。各々のsupport、単位、座標系、
transformが未定義、または宣言したshared mesh IDの内容が一致しない場合をerrorにする。

CAD修復そのものはcoreで行わず、adapterの修復reportをcase provenanceへ残す。

---

## 7. 場の表現と補間

### 7.1 field samplerの最小戻り値

field samplerは値だけでなく、計算の根拠を返す。

```text
SampleResult
  values[N, Q]
  spatial_support[N]
  temporal_support[N]
  element_id[N]
  time_interval[N]
```

これは新しいcontract frameworkやclass hierarchyを意味しない。`fields.py`のscalar referenceと
compiled batch関数が同じ小さな戻り意味論を持てばよく、provider registryやDI containerは作らない。

trial stepを有限にするため境界外の近傍値を一時評価する場合でも、`values`と
`spatial_support`を混ぜない。値が返ったことを「粒子がdomain内」と解釈しない。support外の値は
provisional trial専用であり、そのstageを含むstepをそのまま受理しない。先行wall eventを局在して
supported区間だけ再積分するか、eventがなければ`FieldError`とする。

spatial supportはsupported cellの閉包の和とする。共有面を含むcandidateにsupported cellが一つ以上あれば
insideとし、最小supported cell IDを決定論的ownerにする。全candidateがmaskedならoutsideであり、
provisional値は物理距離が最小のsupported cellへlocal coordinateを射影した有限値に限る。regular/P1/Q1、
cell hint、探索順でsupport規則を変えない。補間が有限にならない場合は明示errorとし、値をclampしない。

現行で運動を駆動するrequired fieldはparticle domain全体を覆うcontinuous node-associated
regular/P1/Q1 fieldへ限定する。cell-associated quantityは保持・外部解析には使えるが、不連続面の側選択modelを
定義するまでforce/chargeへ使わない。部分supportを将来許可する時はstage点のbooleanだけでなく、path上の
最初のsupport exitをeventとして扱う。

producerのNaNはcanonical supportではない。adapterは明示supportを確定した後、masked-only DOFを決定論的な
有限placeholderへ正規化し、方法と件数をprovenanceへ残す。supported cellが参照するDOFはfinite必須であり、
placeholderを物理補間へ使用しない。

### 7.2 reference場と高速cache

一つのcaseに二種類の場を持てる。

- `reference`：元meshと形状関数で評価。V&Vの権威。
- `cache`：規則格子、texture、cell-local係数など高速化用。

現行B1/B2の外部`tools/field_preprocessor`設定format v2、producer v4は、node-associated P1・regular・exact affine Q1
sourceからfully-supported regular targetへのstatic/固定topology linear-time cacheを公開する。source/target cellの共通partitionを
流し、入力float64座標をexact binary rationalとして扱ってsource overlapとtarget coverageを別検査する。正面積sliverを
epsilonで捨てず、float64積分patchのcollapseやproduction locatorのowner不一致は`unresolved_validation`で停止する。
warped Q1、unstructured/partial targetは公開拒否。warped Q1はinverse map/Jacobian/導関数/積分の証明がないため、
coarse/fine sampled差で認証を代用しない。元runtimeのwarped Q1評価を制限するものではない。

共通patch上のvalue・stored-component gradientのweighted relative L2とtarget support boundary上のvalue relative L2を
公開gateにする。XYは`dx dy`/`ds`、RZは`2 pi r dr dz`/`2 pi r ds`を使い、RZ axisの境界measureは0である。
positive 4×4 Duffy Gauss ruleはこのpolynomial spatial scopeのdegree 5までを積分し、nodal/basis/coordinate/accumulationの
roundoff allowanceを含む`relative_l2_upper`を閾値と比較する。reference lower normが0のときはerrorとallowanceも0の場合にだけ
relative errorを0とし、それ以外は公開しない。積分が非表現可能、またはbudgetを超える場合も停止する。
linear-time場は全snapshotを保持し、隣接snapshotの共通partitionからerror/referenceのGram momentsを計算する。
時間内の二次norm比をbounded Bernstein区間とroundoff allowanceで認証し、snapshot endpointだけの比較は用いない。
stationary rootの推定はreport-onlyで、公開判定の証明には使わない。時間相殺・下界不足・52段分割で未解決なら公開拒否。
`sample_absolute_max`はsample値であり、certified L∞上界ではない。
support boundaryの誤差は材料wall-near誤差を認証せず、このgateだけでtrajectory/charge/arrival/fate精度も認定しない。

`validation.memory_limit_mb`、`workspace_rows`、`max_patch_work`を必須にする。source/candidate resident arraysとwriter/index
transientを含むowned-array planをallocation前に検査し、bounded rowsとpatchごとの小さいrational polygonでpartitionをstreamする。
work counterはcoverageと全選択fieldを共有し、AABB候補行とintersection作業を数える。これはprocess RSSのhard limitでも
out-of-core方式でもない。original fields/layouts/geometry/sourcesを保持し、生成条件・誤差・資源をoutput provenanceへ保存する。
NaNを跨ぐbilinear補間やsolid越しDelaunayは許可しない。

現行はphysics設定が参照する一つのlayoutをrunのauthorityにする。cacheは外部preprocessorが別fieldまたは
別DataBundleとして作り、利用者が明示選択する。場所ごとにcacheとmesh-nativeを切り替えるhybrid samplerは、
実ケースで必要性・精度・speedupが確認されるまで延期する。runtime failure時のsilent fallbackは作らない。

### 7.3 unstructured補間

- P1 triangle：barycentric形状関数。
- Q1 quad：逆写像Newton法＋bilinear形状関数。逆写像失敗を近傍値で隠さない。
- vector/tensor：格納basisをmanifestに記載し、座標kernelで変換。
- gradient：可能ならsolverから出力された同一意味の場を使用。節点値からの数値gradientを
  COMSOLの回復gradientと同一視しない。

`fields.py`のoptional `PreparedFieldSet.spatial_gradient`はnodal P1/regular/Q1の形状関数を微分し、
直前の`sample`と同じsupported cell ID・basis weightを使う。出力はcaller-ownedの`[N,C,2]`であり、第二locatorや
逆写像、least-squares fit、通常production stageへのgradient scratch追加をしない。canonical二座標に関するstored componentの
偏微分であり、RZ vectorの3D covariant gradient、COMSOL recovery、物理primitiveの回復を所有しない。
gradient操作はstatic nodal fieldだけを受ける。preprocessorは各snapshotのstatic viewで同じ値/gradient samplerを使い、隣接snapshotのGram momentsを外部で積分する。Q1のJacobianは同じcell/basisから構成し、第二inverse mapを作らない。

COMSOL内部補間を再現するには、element order、DOF、形状関数、smoothing/recoveryを保存する。
現在のnode sampleだけで一致しない場合、COMSOL評価APIをreference oracleとしてprobe点を
追加出力し、外部補間の近似誤差と物理式の差を分離する。

### 7.4 時間依存場

現行の時間依存fieldは、固定された空間topologyに複数snapshotを持たせる最小revisionである。

```text
TemporalField v2
  time_s[T]                     # finite、狭義単調、T >= 2
  values[T, N, C]
  interpolation: linear
  outside_policy: error
  mesh_motion: fixed
```

値はsnapshot-major layoutで保存し、現revisionは全snapshotをprepare時にresident保持する。既存のregular/P1/Q1/cell
空間locationとweightを一度だけ作り、各RK/指数法stageの実時刻で隣接snapshotを線形補間する。runが参照する全fieldの
time knot和集合でmacro intervalを分割し、範囲外はclampせずprepareまたはsampling時に失敗する。`hold`、periodic、
明示discontinuity、streaming/double buffer、moving topologyは未実装であり、固定mesh samplerへ条件分岐を足さず
それぞれ独立revisionとする。

外部`tools/field_preprocessor`は設定format v2、producer v4でP1/regular/exact affine Q1→full regularの
linear-time cacheを公開できる。元snapshotとtime knotを保持し、全隣接区間のcache error/reference norm比を認証する。
元物理場のsnapshot adequacy、未解像の時間変化、discontinuityを認証するものではなく、undersampled inputを
細かいparticle stepで修復できるとはみなさない。

### 7.5 field builder

熱流体結果とプラズマパラメータから静電場を作る処理は、次の独立pipelineとする。

```text
thermal-flow DataBundle
      + plasma closure parameters
      + electrostatic boundary conditions
                    │
                    ▼
        electrostatic_builder
          Poisson equation
          charge-density closure
          nonlinear solve / convergence report
                    │
                    ▼
        augmented FieldSet + provenance
```

Poisson問題にはcharge closureと境界電位/fluxが必要であり、任意のプラズマparameterだけから
一意の電場は決まらない。builderは仮定、式、境界、収束、残差を成果物にし、solverは完成した
fieldを読むだけとする。Case Pの外部プラズマ場とCase Aの内部生成場は、同じcanonical
quantity interfaceを介して使い分ける。

`caseP` / `caseA`はdataset/COMSOL workflow名であり、trajectory engineの設定値にはしない。製品設定は
`imported_external_plasma_fields`または`reduced_electrostatic`というfield-production modeを選び、後者だけが
versioned builder model、plasma parameter、semantic electrostatic boundary group、nonlinear solve設定を持つ。
両modeは同じcanonical `FieldSet`とprovenanceを出力し、その後のcharge/force model選択と直交する。
`model_dataset`のion-drag二式は外部V&Vのmodel-form感度に留め、coreへ複製しない。将来、独立した利用caseと
referenceに基づき別式を採用する場合だけ、既存modelを変更せずphysics catalogの別revisionとして追加する。

F01ではこの境界を`tools/electrostatic_builder`として実装した。最初のcapabilityはstatic
`axisymmetric_rz`、geometry完全一致のtriangle-only P1、単一準中性bulk、C2-smoothed
Boltzmann--Bohm closureに限定する。`2 pi r` P1 FEM、analytic Jacobian、明示continuation、damped Newton、
matrix-free restarted GMRESで完成fieldをcanonical writerへ渡す。粒子engine、公開三API、case schemaは変更しない。
F02ではmixed triangle/quad参照meshをadapterが一度だけP1へcanonicalizeし、builder/runtimeに第二mesh経路を
作らず完了した。代表meshは1,987 node / 3,779 P1 cell、1,826 free nodeで、10 continuation ramp、総linear
iteration 2,821、最終relative residual `1.8441e-13`、charge-balance error `5.8498e-21 C`を得た。
GMRES basis＋Hessenbergは1,235,088 B、solve-only観測は0.689/0.737 sであり、fallback、反復上限増加、第二solverは
追加しなかった。完成fieldは既存RZ solverのfixed-charge＋Coulomb smokeへ渡した。外部比較は同一export nodeの
記述比較に限定し、独立mesh convergence、COMSOL trajectory、wall/Freeze parityを検証済みとはしない。

---

## 8. 粒子物理

### 8.1 運動方程式

基本状態は位置`x`、速度`v`、内部状態`y`（電荷など）である。

```text
dx/dt = v
dv/dt = linear_relaxation(x,v,t,y) + Σ a_k(x,v,t,y) + stochastic process
dy/dt = G(x,v,t,y)
```

runtime寄与は加速度へ統一する。診断力だけ`mass_kg * acceleration`から作る。粒子属性は
`mass_kg / drag_diameter_m / electrostatic_radius_m / displaced_volume_m3 / model_weight`を独立して
持ち、runtimeで相互に逆算しない。

### 8.2 物理moduleの形

第三者が力を追加する入口は、hot loop内の任意Python callbackではなく、最小宣言とcompiled
evaluatorの組とする。

```text
PhysicsModel
  model_id/revision
  category/contribution_kind/exclusive_group
  required_fields
  parameters
  applicability summary
  supported_coordinates/integrators
  compiled evaluator
```

`load_case`と`simulate`内部準備はrequired fieldとparameterを一度確認し、寄与を
`linear_relaxation / explicit_acceleration / internal_rate / noise_coefficients`の四種へ並べる。
primary drag、charge、noiseと各additive-force categoryは0または1 modelとし、blend/compositeは
versioned modelを明示する。同じinternal-state sliceにownerを二つ置かない。一般dependency
graphやkernel-family型階層は作らない。scalar referenceは通常関数としてverificationに使い、
production規模で未compiled modelがPython loopへsilent fallbackすることは禁止する。

`engine.py`は時間work、field sampling、event、commitの調停だけを所有する。modelが二つ目のdragへ
増える前に、展開済み`PhysicsPlan`から作る小さなphysics runtimeへ、sample済みprimitiveのreference評価、
primitive extremaからの力bound、model applicability、一定加速度certificateを集約する。このruntimeは
field locatorやgeometryをimportせず、manager/registry/DI containerにも拡張しない。`fields.py`は位置から
primitiveとそのsupport/extremaを返し、`physics/`はその数値を物理量へ変換し、engineは両者を順序付ける。

`PhysicsPlan`は`has_force`、`evolves_continuous_state`、`requires_stage_evaluation`を別の事実として保持する。
`has_force`は加速度寄与、`evolves_continuous_state`は`Z`等の連続状態rate、`requires_stage_evaluation`は
integrator stageごとの再評価を表し、Stage 1の`force_coupled`一個へ畳み込まない。
`oml_stationary_maxwellian_debye_huckel_v1`単体は`false/true/true`であり、電気力等との合成planでは
`has_force`だけが別寄与によりtrueになり得る。continuous chargeは電気力が無いcharge-only caseでもstage evaluatorを
通り、動的`Z`を持つproposalはlinear/quadratic exact specializationを使わない。proposal/event/memoryの選択は
この三特性を一度だけ参照し、capability frameworkや二つ目のengineを作らない。

### 8.3 drag

希薄気体の主経路はEpstein dragとし、relative velocity `u_g-v_p`、gas temperature、density、
mean thermal speed、accommodation parameterから計算する。Knudsen数、response time、Reynolds数は
粒子径と局所原始場から毎回求める。

連続域のStokes/Cunninghamや中間域modelを追加する場合、Kn範囲を宣言し、自動切替は
versioned composite modelとして実装する。適用範囲外で別式へ黙って切替えない。near-wall
drag補正は独立moduleであり、壁距離・法線の品質要件を持たせる。

### 8.4 決定論的な追加力

物理catalog全体では、重力/浮力、電気力、熱泳動、DEP、free-molecular lift、ion dragを扱う。
Stage 1は重力/浮力と電気力までで閉じ、P15-Fでcollisionless Barnes ion drag、P16で
Waldmann--Gallis free-molecular thermophoresisを追加した。現在の比較caseについて必要な式と入力が確認できたため、
Stage 3ではP18-Dの球形準静的DEP、P18-Iの二つのion-drag sensitivity revision、P18-LのRZ
rarefied-vorticity lift sensitivity、P22のTalbot thermophoresisとSaffman liftを追加済みである。各moduleは
acceleration vectorを返し、field samplingや粒子状態更新を
所有しない。

- 電気力：`F_E=qE`。電荷更新とstage時刻を一致させる。
- DEP：P18-DはproducerがDC/cycle-mean、solution、回復法を固定した`grad(mean_E_squared)`をcanonical vectorとして出し、
  `electrostatic_radius_m`と`mass_kg`から球形準静的dipole力を計算する。coreはEを微分せず、producer認証済みの
  `maximum_point_dipole_radius_m`をprepareで検査する。
- 熱泳動：`physics/forces.py`が局所並進熱流束形、Kn/低drift適用域、global boundを所有する。catalogは
  field authority、runtime/compiledはsample済みstage評価だけを所有し、gradient回復とproducer/COMSOL比較は外部に置く。
  P18-Rのeffective-gas revisionではproducerが一つの有効Maxwellian/pseudogasと有効並進伝導熱流束`q_eff`を認証し、
  coreはspecies配列やtemperature gradientを回復しない。P22 Talbotはproducer提供の`grad(T)`とcase明示の
  `k_p,Cs,Cm,Ct`を同じstage passで使い、Waldmannとのblendや自動切替を行わない。
- lift：P18-Lはaxisymmetric no-swirlのr/z運動だけを対象に、
  `F=K (omega_phi e_phi) x (u_g-v)`、`K=C_L*pi*rho_g*lambda_g*a^2`、`a=drag_diameter_m/2`を使う。
  `C_L`は有限正値を明示し、producer所有のsigned方位vorticityを消費してcore内で速度場を微分しない。
  `lambda_g/a>=10`をfail-closedに要求し、一般Saffman liftや3-D revisionへ暗黙拡張しない。速度依存項はruntimeの
  全非drag加速度bound callbackへ統合し、exponential enclosure v3が開始速度とhalf predictorで再評価する。
  integrator v2、engine v30、proposal v7は維持し、追加決定論力を許さないB02 Brownianとの併用は拒否する。
  P22 Saffmanは同じcategoryの別revisionとしてXY/RZのproducer提供signed面外vorticityを使い、Stokes--Cunningham
  またはdragなしだけを受理する。連続体・低Re・`Re_s<<sqrt(Re_G)`の固定policyを全pathでfail-closedに検査し、
  near-wall liftやP18-L式へfallbackしない。
- ion drag：P15-Fはrelative-velocity方向、collection＋orbital、linear two-species Debye screening、非正電位の
  一つの明示revisionである。electric-field方向、collisional/nonlinear screeningは別versionとして扱う。

`model_dataset`の2種類のion dragは、どちらかを恣意的に正解へ寄せる材料ではない。P18-Iではfloor、image補正、
方向、screeningを含む各式を独立に再定義し、optionalなversioned sensitivity modelとして実装する。既存Barnesを
変更せず、自動選択・blend・Case P/A分岐を作らない。比較式と一致することと物理妥当性は別statusで評価する。

P18-Rのauditでは、native linear Epstein式parityは約`1.1e-15`でPASSしたが、既存P15-E/P16の
physical applicabilityはmixture/model authority不一致により12/12 caseで`NOT_APPLICABLE`、保存PPRに`q_eff`が無いため
thermophoresis pointwise replayは`NOT_TESTED`だった。このため既存式ownerを共有する
`epstein_linear_effective_gas_sensitivity_v1`と
`waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1`を追加した。両者はproducer認証済みの
one-effective-Maxwellian/pseudogas reference/sensitivityで、`lambda/a>=10`と明示`maximum_speed_ratio<=1`を要求する。
既存linear/P16の`0.1` gateは維持し、species-resolved mixture truth、COMSOL branch、gradient recoveryを追加しない。

後続M3-C1では、熱泳動に必要なPPR heat-flux primitiveだけを最小COMSOL補足として再exportした。既存v6と座標が完全一致する
13,202 active saved rowについて、producer-form replayのcomponent-scale normalized residual最大は
`4.4046499933294035e-16`、global relative L2は`1.0992667494449471e-16`であり、frozen saved-state区分を8/8閉じた。
これはP18-Rのspecies/physical-applicability判断を変更せず、continuous pathまたは統合軌道を認定しない。

### 8.5 電荷

電荷moduleを一種類の「charge」へまとめない。

- `fixed_charge`
- `oml_stationary_maxwellian_debye_huckel_v1`：stationary Maxwellian OMLとDebye--Hückel capacitanceで
  `dZ/dt=Γ_i-Γ_e`を運動と連成
- `oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1`：単一・単価正イオンのshifted-Maxwellian
  OMLを非正表面電位だけで運動と連成
- `discrete_charge_jump`：整数電荷の確率jump
- `aerosol_charge_relaxation`：COMSOLの空気aerosol系モデルを必要時に別実装

プラズマ粒子へaerosol charging式を流用しない。OMLも衝突性sheathや非Maxwell分布で常に正しい
わけではないのでvalidity domainをmanifestに残す。

P15の初回modelは`a/lambda_D <= 0.1`を必須とし、ion driftは収集率へ入れず、resolved
`M_i <= 0.1`をstationary近似のapplicability gateにだけ使う。`model_dataset`のheuristicやCOMSOL側の
model名は外部V&Vの比較条件であり、core catalog、rate式、capacitance、applicabilityのauthorityにしない。
adapterは必要なcanonical primitiveへ変換するだけで、COMSOL専用evaluatorを作らない。

P18-Cの`aggregate_relative_drift_regularized_two_current_v1`も同じ`internal_rate` ownerへ置くが、P15/P15-Dを
置換しないoptional reference revisionとする。局所有効正イオン質量、電子・正イオンthermal voltage、背景screening長は
canonical scalar field一つずつに統一し、uniformなproducerも定数fieldを書く。revisionは粒子ごとに
`lambda_eff=max(electrostatic_radius, screening_length)`を適用する。parameter-or-fieldの二重authority、COMSOL式parser、
charge専用subcycleを追加しない。
正負電位branch、正則化、floor、指数範囲、有限invariant/rate/derivative boundを同じmodel revisionが所有する。
有限boundのためcaseは正則化前の`|u_i-v|`の有限正値上限を必須宣言し、actual stageとcontinuous pathで検査する。
これは速度を切り詰めるparameterではない。

P15-Dのshifted revisionは
`M_i=|u_i-v|/sqrt(8 k_B T_i/(pi m_i))`の有限正値上限をcaseへ必須とし、actual stageとcontinuous
pathの双方で検査する。初期`Z<=0`と全primitive/drift rangeでの非正平衡をprepareで認証し、invariantを
`[Z_min,0]`へ固定する。zero-driftはstationary負電位branchへ連続に一致する。負イオン、複数イオン種、正電位、
emission、collisional/magnetized chargingは別revisionであり、外部比較式のfloor/clipをcoreへ持ち込まない。

連続電荷ODEは固定RK4では各RK stageで場、速度、Zに対して評価し、explicit `hL_Z<=0.5`を維持する。
現行native exponential midpointとB03は同じ予測midpointで`G`と`J=dG/dZ<=0`を凍結し、root基準の
affine lawを`expm1`で解析更新する。運動も同じ`Z_mid`を使うので一つのcoupled proposalである。
exponential pathは`hL_Z<=0.5`をstability gateにせず、別runの`h,h/2,h/4`がaccuracyを選ぶ。
clip、charge-only subcycle、第二engine、一般IMEX、暗黙equilibrium置換は作らない。
charge数`Z`とCoulomb`q`の権威を一つにし、両方を独立入力にしない。

動的電荷modelは、局所rateだけでなく有限な`Z` invariant/bracket、rate上界`B_Z >= |R_Z|`、
derivative上界`L_Z >= |dR_Z/dZ|`、適用域を`physics/charge.py`から提供する。RK4区間は
`h L_Z <= 0.5`と全stage/accepted endpointのinvariant包含を必須とする。charge-aware path enclosureはその
charge区間から`max |Z|`を求め、electric
acceleration boundへ渡す。prepare時の初期`Z`だけで全stepの電気加速度を囲わない。stage値、区間、または
加速度上界を有限に認証できないrowは、0や平衡値へ置換せずfail-closedとする。

### 8.6 Brownian

Brownianは「各stepにランダムな力を足す」だけでは、step refinement時に別過程になる。

- 外部V&V：必要なら一般solverへ同じstepと乱数方針を指定し、比較情報を外部manifestへ保存。
- native method：速度緩和を含むOrnstein–Uhlenbeck/Langevinのjoint `(x,v)` 更新を使う。
- adaptive refinement：joint Gaussianの親incrementをbinary tree pathで条件付き分割する。
- wall近傍：一般曲面でexact first passageとは主張しない。最初のrevisionは固定depthのdyadic nodeと
  cubic Hermite leaf pathをgeometry/outputから独立に定め、depthに対するfirst-passage統計収束で規定する。

乱数identityは`seed, particle_id, macro_interval, root_stochastic_interval, tree_level, tree_index,
component, stream`から構成し、accepted-step番号やevent ordinalへ依存させない。処理順、thread数、
tileサイズ、inactive compactionで乱数列が変わらない
counter-based Philox/Threefry系を使う。

B01はjoint OU平均・full covariance、親endpointを保存するconditional half-split、上記identityの
Philox4x32-10 normal drawまでを数値基盤として実装した。B02は`noise` categoryと`ou_langevin`をengineへ接続し、
各macro-rootで係数を凍結して固定depth treeを時間順に処理し、OU leaf endpointから作るcubic Hermite pathを
材料eventとframe/probeの共通authorityにした。Cartesian XY、Epstein linear drag-only、fixed-charge state、terminal
stick/escapeだけを受理し、continuous charge、drag以外の決定論力、反射、RZ、3-Dは暗黙に近似せず拒否する。
engine v30はroot covariance、conditional split、deterministic meanの実表現可能性を検査し、例外時だけrow別に
局在して不良粒子を`nonfinite_physics`で停止し、正常粒子を継続する。Gaussian bridgeは有限tubeへ決定論的に
囲えないため、単なるRMS距離でintervalをclearにしない。B05でも`interval_tree_depth`は全rootに一様な
numerical-path精度のauthorityである。`adaptive_max_depth`は、その有限depth pathがwall、RZ axis、または
証明不能区間の候補になった時だけ保存済み親endpointから条件付き二分する上限であり、確率的clearやbase depthの
自動置換ではない。base=maxは従来の固定depth pathへ退化する。どちらもsampleしていない連続OU軌道のexact
first-passageやzero miss probabilityを主張しない。
memory plan v16は観測候補率でなく`adaptive_max_depth`からtree workを保守的に解決する。

同一COMSOL軌道を再現するには、COMSOL乱数増分と内部stepの情報が必要である。現在の
`model_dataset`だけでは粒子単位のexact replayはできないため、Brownian offの決定論比較と、
複数seed/replicaの統計比較を分ける。

現データでは`v_phi=0`かつ運動DOFがr/zだけである一方、診断用`Brownian_force_phi_N`は非zeroで
3D合力magnitudeに含まれる。RZ no-swirl kernelはr/zへ投影された成分だけを積分し、phiを含む
magnitudeから加速度を再構成しない。将来のexportもtrajectory-effective forceと3D diagnostic
sampleを別schemaにする。

B03は第二engineを作らず、既存`ou_langevin` strategyへ一つのversioned stochastic proposalを追加して完了した。比較用の
RZ revisionは2自由度meridional projectionであり、manifestのcoefficient policy
`macro_root_frozen_midpoint_v1`、composition `stochastic_exponential_midpoint_v1`を使う。
各rootでnoise-free決定論exponential-midpoint predictorからmidpointを得て、線形Epsteinの`gamma,u,T`、
全additive acceleration `a`、
continuous charge rate `G`と`J=dG/dZ<=0`を一度だけ評価して凍結する。`u_eff=u+a/gamma`に対するjoint exact OUで
`(x,v)`をroot全幅進め、Zは`macro_root_affine_exponential_v2`のdense stateとする。
このpredictorはstateをhalf-step commitするStrang/K-O-K分割ではない。

既存conditional treeは同じ凍結係数でroot Gaussianを分割し、root endpointを保存する。RZ axisに到達したら
accepted prefixをcommitしてfoldし、macro残時間を次の`root_stochastic_interval`ordinalとして係数再評価・
独立root drawで再開する。元rootのcubic remainderはfold/restrictしない。B04は同じprefix-commit/fresh-root規則を
active wallにも適用し、terminal lawと反射・確率wallを受理する。native/effective-gas線形Epstein、fixed/continuous charge、
既存additive forceを明示的に解決し、B02のXY/fixed-charge/drag-only revisionはbitwise不変である。dense chargeは
root基準`G_mid+J(Z_root-Z_mid)`のaffine exponentialとして全leaf intervalをprepared charge invariantへ照合し、
外れまたは証明不能をfail-closedにする。

検証は定数係数mean/full covarianceのexactness、noise-offの2次収束、manufactured charge-electricの
weak mean観測次数`>=0.9`、axis restart、tree-depth first-passage、slab/output/resume identityとする。
一般state-dependent SDEのstrong orderやweak 2次を主張しない。初回受入revisionはengine v34、proposal v9、catalog v16、
event v15、runtime v17、compiled tile v16、memory plan v13であり、公開済み0.2.0のsupersessionはengine v46、proposal v10、
event v22、catalog v23、runtime v22、compiled tile v21、memory plan v16である。その保存sourceでのB03 path arrayの静的な保守上限は一slab rowあたり
`648 B`で、受入上限`2048 B`内に収まる。正式characterizationは2,000/20,000粒子×4構成×3反復を全粒子active・
failure 0で完了し、計時/RSSは[`solver/evidence/b03/`](solver/evidence/b03/README.md)のmachine-local・non-gating証跡とする。
characterizationで検出した反復加算由来の終端tailはengine v34で除去した。macro timeは補償積和による
`start + n*dt`のindexed gridから構築し、endへのsnapはfloat64構築roundoff内だけに限定する。

event v15は物理position budgetとroundoff budgetを加算し、facet-local offset dotを補償演算で、cubic Hermiteの
評価・enclosureをroot-relative TwoDiffで扱った。event v16はvalidなRK4 dense rowに限り、integratorが所有する
root-relative position Bernstein control enclosureをfacetの外向きhalf-spaceへinterval射影する。4制御点の外向き上限が
すべて既存position budgetの負側に厳密に入る候補だけを、Bernstein convex-hull性により非接触と認証する。入力が欠損・
非有限・不整合、または不等式を証明できない場合は従来どおりsplit/fail-closedとする。monotone-approach clearはcubic
Hermite derivative-Bernstein enclosureの明示opt-inだけに限定し、exponential、scalar pathへv16 certificateを適用しない。
position-control payloadは`144 B/row`、構築時の追加peakは`128 B/row`で、保守的な同時live見積りは約`1.76 KB/row`と
既存の一般stage scratch上限`2048 B/row`内に収まるため、memory plan v13を変更しない。broad AABB pre-countは
このfacet-local clear前の保守的なwork量であり、event数や物理結果ではない。
現行event v23は、曲線pathでも個々のfacetを独立に認証して最初のhit時刻を決め、その局在budget内で同時かつ
共有nodeへincidentなfacet集合を保持する。exact、一般RK4、exponential midpoint、Brownianは同じ候補集合、
priority、combined-normal意味論を使う。有限半径material contactとperiodic center crossingは同一のfirst-event順序で比較し、
証明不能な候補やmaterial/periodic混在cornerは一面へ丸めずsplit/fail-closedする。

P17後のCartesian 3-D isotropic Brownianは別revisionである。RZ projected revisionの合格から3-D isotropyや3-D
COMSOL parityを主張しない。どちらも同じRNG identity、event/output/checkpoint経路を使い、Python callbackや別runtimeを作らない。

---

## 9. 境界と粒子lifecycle

### 9.1 event-firstの処理

macro stepの終点がdomain内かだけを判定すると、薄い部品を飛び越す。integratorは内部
`StepProposal(endpoint_state, state_at, path enclosure, provisional validity)`を返し、各accepted numerical
pieceに対して最初の境界eventを解く。`StepProposal`が表すのはversioned integratorで定義した離散軌道であり、
path enclosureはその離散軌道を保守的に包む。真のODE解まで包むとは主張しない。ODE離散化誤差は別runの
`h / h/2 / h/4`収束、解析解、manufactured solutionで評価し、幾何判定幅へ混ぜない。

`integrators.py`はreadonlyなRK4 dense rootに対応するfull区間enclosureをproposalの寿命内で保持する。
同じroot pathを参照し、要求の始終時刻が選択行のroot時刻とuint64表現でbitwise一致し、現在のstatusも保持値と一致する場合だけ、
保持結果から選択行の独立copyを返す。厳密な部分区間、full区間と部分区間の混在、statusまたはpathの不一致は既存のstrict restriction ownerで再計算する。
callerによる返却配列の変更はrootや保持enclosureへ書き戻さない。

```text
integratorがtrial pathを提示
    → EventLocatorが最初のhitを返す
        → hitまで状態を積分・局在
        → BoundaryLawがoutcomeを決定
        → 残り時間を新状態で継続
```

P05は解析的なballistic直線segmentと`line2`のexact intersectionにこの流れを限定した。
scale-awareなposition/time budgetで最初hitと同時candidateを求め、candidateはfacet ID順に保持するが、
当時のproductionは単一candidateのhitだけを応答へ渡した。table初期点は材料boundaryのstrict interiorとする。
engine v9のP07 exact-path sliceはforce-free surface zero-time departureと複数candidate responseを解禁した。
engine v10 / event v6はCartesian XYの証明済み一定加速度surfaceへ拡張した。engine v11 / event v7は
Cartesian XY一般RK4について、厳密内向きsurface departureとsingle-facet activeな壁応答後の残時間継続を解禁した。
tangent、facet端点/corner、またはその他の曖昧なstart-contactはfail-closedである。
engine v12 / event v8は同じpiece列へRZ axisをmaterial wallとは別のfirst-event候補として加え、axis prefixの
再積分、basis fold、残時間継続を行う。双方が局在済みで認証時刻の不確かさがwallと重なる場合だけwallを優先し、
axis端点とmaterial cornerの完全tieは一般RK4 cornerとしてfail-closedにする。

curved pathはchordだけでno-hitを判定せず、physics係数・速度・加速度から保守的なpath enclosureを
構成できる場合だけcheap no-hitを許す。boundを構成できない区間は、区間始点から順番にdyadicな
RK4 pieceとして再積分し、各pieceで力評価とevent探索を行う。このgeometry-drivenなpiece列そのものを
accepted numerical pathとし、出力時刻は分割を変えない。hit状態は同じintegratorでhit時刻まで再評価する。
refinement budgetが尽きた場合は
`AccuracyStatus=budget_exhausted`またはrun failureであり、架空の壁停止にしない。幾何的に
交差の有無を証明できない場合だけ`IntegrityStatus=indeterminate_geometry`とする。

各pieceの`swept path tube`は、力から導いた離散path変位boundとgeometry/roundoff budgetを包含しなければ
ならない。full-step/two-half-step差、Hermite–chord偏差、midpoint差はaccuracy indicatorには使えるが、
それだけをtrajectory enclosureとはみなさない。BVHのAABB分離、または候補facet支持線に対するtubeの
法線方向intervalがfacet別event budgetを含めても厳密に分離する時だけno-hitを確定し、
重なる時は次の逐次pieceへ二分する。hit rootはtime bracket、spatial tube、candidate-facet setの三条件が
収束するまで局在する。stage・endpointが非有限なら即時失敗とする一方、support/applicability逸脱は
provisional flagとして保持し、先行eventで切り落とされなかったaccepted prefixに対してだけ確定する。

engine v7の各粒子はleft-firstなdyadic workの次の1 pieceだけをwaveに出す。wave内でfloat64の
target timeが完全に同じ行だけをstableにまとめ、候補粒子を256件ずつの有界chunkでproposalする。
`inspect_rk4_piece`とBVH candidate queryは行ごとのままであり、accepted path、event順、集計の意味を変えない。
event v5は、`integrators.py`所有のcomponentwise chord-deviation/roundoff boundを用いて、
`events.py`が単一facetの外向き横断、normal time bracket、tangent/time-shiftを含むposition radius、
endpoint clearanceを証明する。証明失敗時はsplitし、最後はfull-tube budgetへfallbackする。engine v7、
proposal v3、enclosure v1、schema/APIはこの数値変更では不変である。engine v8は数値pathを変えず、frameと
重なるaccepted rowだけをreplay用に保持した。engine v9はexact linear/quadratic pathへsurface/wall/RZ axis
意味論を追加し、C08～C10を公開経路で実行する。engine v10 / event v6は一定加速度surfaceの
start-contact certificateを追加した。engine v11 / event v7は一般RK4へinterior-facet start-contact certificateと
single-facet active-boundary residualを拡張した。interaction capはparticle residual stateで共有し、次hitが実際にある時だけ
残区間を分割する。event時刻のactive jumpとterminal transitionは右連続である。
engine v12 / event v8でこの意味論を保ったままRZ signed-stage/axis eventを追加し、現行sourceも維持する。
P11の指数pathも同じleft-first piece列とfirst-event arbitrationを使う。現行event locatorはintegratorが渡すmethod固有の
component-wise deviationを受け、指数法では全短縮secantを含むvelocity enclosureから
`h * (v_upper - v_lower) + roundoff`を外向きに作る。position enclosure全幅を流用した過剰分割や、
指数法専用event loopは残さない。

### 9.2 event record

すべのmaterial wall interactionとperiodic topology transferをlogical canonical logへ残す。

```text
particle_id, event_ordinal, interaction_kind, time_s
primary_facet_id, destination_facet_id, candidate_offset/count, boundary_id, material_id
contact_radius_m, position_m, position_post_m, normal
velocity_pre, velocity_post
charge_number_pre, charge_number_post
law_id, outcome, model_weight
localization_residual_m, position_budget_m, time_budget_s
```

`normal`はlaw適用に使ったeffective response normalである。単一facetではその外向き法線、
combined-normal反射では選択subsetの正規化合成法線を保存し、raw candidate facet集合は別tableへ保持する。

release、boundary interaction、failureは別groupとし、boundaryの`interaction_kind`は`wall`または
`periodic_translation`に固定する。同時hit候補facetは別columnar candidate tableへoffset/countで保存し、
primary facetだけへ情報を潰さない。

現在の物理配置v1はnullableな統合tableではなく`/events/release`と`/events/boundary`を別groupにし、
group名をevent typeとする。candidateはboundary groupのoffset tableで保持する。P05で常に確定済みの
`geometry_status`と未実装failure groupを空列として先行追加しない。実装列は
`solver/docs/result_format_v3.md`を権威とする。P13のsegment拡張はlogical identityを変えず、P18-Hは
`held` lifecycle series列の追加だけでresult/checkpoint schemaを一度だけv2へ更新した。

保存時刻のstatus変化からhitを推定しない。このlogが境界比較と付着統計の権威になる。
P05はreleaseを粒子ごとのevent ordinal 0、最初のboundary eventをordinal 1とする。

### 9.3 BoundaryLaw

幾何交差と物理応答を分離する。

- `stick`：lifecycleをstuckへ。位置をhit点に固定。
- `escape`：escapedへ。以後のkinematicsはlogical nullとする。
- `hold`：heldへ。hit位置、hit時速度、hit時電荷を保持するinactiveな非deposition終端。
- `specular`：parameterを持たない完全鏡面反射`v'=v-2(v·n)n`。
- `restitution`：法線・接線反発係数`e_n`、`e_t`を必須とし、`v'=-e_n v_n+e_t v_t`。
- `maxwell_thermal`：`wall_temperature_K`、`diffuse_reflection_fraction`、`wall_velocity_m_s`を必須とし、
  完全熱適応half-range Maxwell fluxと壁frame鏡面を混合する。wall velocityは静的groupの全facetへ接線方向だけ。
- `probabilistic_stick`：定数確率でstickし、非stick時の`specular | restitution | maxwell_thermal`を`otherwise`へ明示。

以上が現行BoundaryLawである。point/finite-radiusの差はgeometry/eventが所有し、lawはevent候補に対応する
接触法線だけを受ける。有限半径は静的2-D XY disk / RZ sphere中心とmaterial `line2`のcapsule first contact、
surface sourceのinward offset、table sourceのclearanceまでを扱う。rolling/sliding、速度依存rebound、
resuspension、surface charging、moving geometryは後続catalogとする。
groupの`contact_geometry`は未指定時`particle_surface`で、明示`particle_center`だけ候補の有効半径を0とする。
`case.py`が選択を正規化し、prepareした一つのfacet surface maskをgeometry/event/source/memory planが共有する。
粒子の物理半径、mass、drag/電気半径は変更せず、一mesh/BVH上で候補固有の法線・残差・clearanceを解決する。
混在modeの時刻順が証明できない同時候補はfail-closedとし、lawのaliasや第二geometry engineを作らない。
COMSOL raw stateのfreeze/disappearは外部adapterがrawのまま保持し、正例microcaseで実際に出力された
hit後位置、速度、statusだけを比較manifestでcanonical意味へ対応付ける。P18-H初回referenceは電荷を
出力していなかったためCOMSOL電荷保持は認定せず、非零電荷保持はprovider非依存の公開scenarioが所有する。P18-Hは半導体装置内の非deposition終端という
producer非依存の利用caseに対して`hold/held`をstick/escapedと分けて追加した。COMSOL名やraw status codeはcoreへ
入れず、外部adapterが明示的に対応付ける。
P05はparameterを持たない`stick`と`escape`だけをprepareした。engine v9は静止壁の`specular`と、
定数確率でstickし非付着側を明示specularへ渡す`probabilistic_stick`を追加する。stickはpost速度0、escapeは
表面impulseなしでpost速度をpre速度と同じにする。
現行`contact_wall_laws_v7`では`specular`を係数なしの完全鏡面だけに限定し、係数を持つ反射を
`restitution`へ分離した。`probabilistic_stick.otherwise`はparameterなしの`specular`、または二つの係数を
明示した`restitution`、または完全なparameterを持つ`maxwell_thermal`だけを受理する。`maxwell_thermal`は
`v_w-c_n n+c_t(-n_y,n_x)`のpolicy-free half-range変換とcounter RNGを使い、静的geometryへ法線wall motionを
持ち込まない。`hold`はparameterを持たず、hit時payloadを保持したままinactiveにする。
hit後のchargeやkinematicsを進めるpaused状態、再開、再飛散はこのlawの責務外である。

確率を0～1へruntime clampしない。設定やmodel出力が範囲外ならerrorとする。一つのboundary groupへは
wall lawを一つだけ設定する。cornerで異なるgroupのfacetが同時候補になった場合は、manifestのpriorityで
選択し、実行順に依存させない。

確率wall lawは`seed/particle_id/physical_boundary_event_ordinal/law_stream`をkeyにする。event locatorの
二分やrejectはphysical ordinalを進めない。source、wall、Brownian、charge-jumpのstreamを分離する。

### 9.4 corner、同時hit、連続接触

corner/edge同時hitは、最小時刻だけでは法線が一意でない。候補facet集合をeventに残し、
吸収系lawなら一致する結果を適用し、反射系は明示した合成法線/逐次反射policyを使う。
同時刻の反復hitを無限loopにしないよう、hit ordinalと相対法線速度で「壁から離れる」segmentを
departureとして扱う。位置nudgeは使わない。static geometryの壁速度は法線成分0だけを許し、moving
normal wallはmoving geometry対応まで拒否する。contact slidingは点粒子追跡の基本境界ではなく、
必要なら別物理moduleとして追加する。

---

## 10. 数値積分method

### 10.1 一つのengineでstep strategyを分ける

一般性、高速、stiff drag、Brownianを一つの数値式へ押し込まない。一方で別runtimeも作らない。
production engineは一つで、決定論の二つとBrownianの一つ、計三つのstep strategyを持つ。

| method | 目的 | 方法 |
|---|---|---|
| `rk4_fixed` | 一般決定論、verification・外部solver照合 | 古典RK4、stage連成、指定固定step |
| `exponential_midpoint` | 高速・安定なnative標準 | drag解析更新＋midpoint |
| `ou_langevin` | 有限深さの慣性Langevin SDE | B02はstart-frozen drag-only、B03はmacro-root stochastic exponential-midpoint。どちらもjoint OU＋固定depth conditional tree＋cubic Hermite leaf |

COMSOL adapterは必要なcaseで`rk4_fixed`とstepを指定するだけであり、coreにCOMSOL version別profileや
`comsol_default`を作らない。

### 10.2 固定RK4 reference

運動と電荷を同じ状態vectorとして、各stageで位置、速度、電荷に対応する場と力を評価する。
決定論計算を標準とする。外部V&Vで確率式を同条件にする必要がある場合、乱数規則は外部manifestで
固定する。field time knot、global source discontinuity、wall eventで必要な区間を切る。個別releaseは
その粒子の残時間workとして扱い、保存時刻は積分刻みとは別にaccepted `StepProposal.state_at()`から
評価する。保存時刻を増やすために積分stepやRNG pathを変えない。

RK4のpathは端点位置・速度から作るcubic Hermiteを候補表現にできるが、Hermite–chord偏差だけを
保守的path enclosureとはみなさない。最初に、physics係数とfield extremaから各proposalの連続軌道全体が
required field support内にあり、選択modelのapplicability内にあることを証明する。revision 3aでは、globalな
canonical field extremaとmodel係数からRK4内部stageの速度・加速度を保守的に包絡し、float64で外向きに丸めた
位置boundが、macro終端だけでなく`state_at()`が生成する任意の短縮区間の内部stageとaccepted endpointを
すべて含むことを要求する。Epsteinでは同じ速度boundから`lambda/a`下限と`|u-v|/c_bar`上限を作る。
fully-supported regular boxへの包含または連続applicabilityを証明できないcaseはhidden subdivisionせず
fail-closedにする。これによりtrajectory scheduleは新しいvalidity判定を導入しない。

revision 3bで初めて、各piece始点から作ったpath tubeがBVH node、facet AABB、または候補facet支持線の
認証済み法線intervalから分離する時だけno-hitを確定する。
候補が残る区間は、元のmacro proposalをparameter方向に切るのではなく、時間順のdyadicな逐次RK4 pieceへ
分けて再積分する。急峻な位置依存場では、同一始点からの短縮RK4 endpoint曲線の部分区間が局所時間幅とともに
縮むとは限らないためである。分割はgeometry、support、applicabilityだけで決まり、frame scheduleには依存しない。
budget内に関係を証明できなければ成功扱いにせず、対応する明示failureとする。先行wall hitが後続trialの逸脱を
救済し得るため、材料boundaryではevent探索前にproposal全体のvalidityを拒否せず、pieceごとに
`event/valid prefix → no-hitまたは残存prefixのvalidity → commit`を維持する。frameは確定済みaccepted
pieceから評価し、frame評価のために数値pathを作り直さない。

P19-Lでは、上のgeometry-drivenなaccepted piece意味論とは分離して、強い局在場に対するapplicability certificateだけを
局所化した。P19-L完了時点の`integrators.py`は一つの固定step RK4 proposalに対応する
`rk4_position_hermite_state_extension_v2`とparameter
部分区間のstate enclosureを所有し、`fields.py`がそのtubeと重なるcellのprimitive range、`physics/runtime.py`がrange上の
model boundを所有する。certificate workは不変なroot pathをdyadicに制限してもendpointを再積分・commitせず、元proposalが
全区間で証明された時だけその元endpointを受理する。したがって固定stepの時間離散化、frame schedule、event順序を変えない。

局所rangeは一row最大64 cellで、overflowなら同じ`solver.event.max_refinements` budget内で区間を分割する。memory plan
v13はdense path 176 B/row、split budget 2のinterval stack 72 B/row、candidate arena最大544 B/rowを名前付きcomponentへ
計上する。global certificateを先に試すため一様caseへ常時local探索を課さない。4,096粒子×20 stepの公開API観測はglobal/local
`0.9276381/0.9345506 s`、比`1.0074517`、科学payload bitwise一致であり、machine-localな非gating証拠である。
代表Case-A profileでは局所certificate全rowが分割なしで閉じ、global support enclosureに連動するevent refinementが支配した。
同じgeneric chord算術のbatch化だけを残してbitwise identityを維持した。support証明とevent refinementの結合を変えるなら、
P19-Lへ条件分岐を重ねず、同じfirst-hit/support意味論を解析・回帰できる別の数値work packageとする。

actual stage/sampleまたは局所rangeが適用域外なら`model_applicability`、有限なcertificate budgetを使い切っても真偽を
証明できない場合は`indeterminate_applicability_certificate`とする。証明不能を物理違反へ畳み込まず、反対にgateを
無効化して成功扱いにもしない。一般interval framework、第二engine、比較case固有の閾値は導入しない。

P06 revision 2では一般enclosureに先行して、dragなし・厳密一様fieldから証明した一定加速度だけを解析放物線として
扱う。topology-completeな材料boundaryではfirst hitが流体domainからの退出を捕捉する。boundaryなしでは、全cell
supportedな`RegularLayout`の矩形support boxに対し、各座標の端点と内部極値をproposalごとに解析評価する。
材料boundaryの候補探索はendpoint chordを`|a|h^2/8`で膨張して漏れを防ぎ、facetとの二次根がfloat64 budget内で
決定できない場合は明示的に失敗する。一般非一様場・dragはrevision 3aで上記boundaryless regular
support/applicability enclosureを完成した。revision 3bで材料boundary用tube・sequential accepted pieces・
fail-closed判定を接続した。P06-Uでは、exact meshかつ全cell supportedなP1/Q1とtopology-completeな
材料boundaryの組合せを解禁した。boundaryless unstructured supportとRZは、それぞれの包含・basis規則が
完成するまでprepareで拒否する。
一定係数の
exponential strategyは局所解析pathを使い、どちらもhit時刻の状態は同じstrategyで再評価する。

V&V用にdebug modeでは、指定した少数粒子・stepだけ次を出力できるようにする。

- 各stageの`t,x,v,Z`
- sampled fields
- force/charge derivative
- 合成increment

全粒子全stageの常時出力はしない。

### 10.3 native deterministic profile

小粒子ではdrag relaxation timeが非常に短く、陽的methodは安定性のため極小stepを要求する。
局所的に

```text
dv/dt = (u-v)/tau + a
```

とみなせる区間は指数関数で更新する。`E=exp(-h/tau)`、`A=1-E`として

```text
v1 = u + E (v0-u) + tau A a
x1 = x0 + u h + tau A (v0-u) + tau {h-tau A} a
```

を使い、`A`は`-expm1(-h/tau)`、小さい`h/tau`の残差係数は級数で評価する。start係数の半step
predictorでmidpointを作り、そこでfield、charge、drag、加算加速度を評価する。非線形dragをこの式へ
暗黙に入れない。

Stage 1は固定macro stepと別runの`h,h/2,h/4`収束を使い、精度目的のhidden adaptivityを持たない。
field knot/global source discontinuity、wall eventだけ必要な区間を分け、個別releaseは粒子別残時間workで
処理する。Stage 2以降で必要になった場合だけ、step-doublingを用いた2の冪`bounded_dyadic`
controllerへ拡張する。

将来のcontrollerでも次を別に扱う。

- accuracy reject：dtを減らし再試行。
- event refinement：境界時刻の局在。
- support uncertainty：field/geometryの安全性。
- output sampling：accepted pathから評価し、保存要求のためにstepを分けない。

これらを一つの「substep split回数」に畳み込まない。位置[m]、速度[m/s]、電荷[個またはC]の
scale別許容差を設定し、固定の内部定数へ隠さない。

### 10.4 収束と誤差budget

methodごとに、解析解または十分細かいreferenceに対して観測収束次数を確認する。製品caseの
許容値は「COMSOLとの差を小さく見せる値」ではなく、次の和として設定する。

```text
総比較budget
  = COMSOL referenceの時間離散誤差
  + 場export/補間誤差
  + 本solverの時間離散誤差
  + event局在誤差
  + stochastic sampling uncertainty
```

dt半減で差がどう変わるかを必ず記録し、誤差plateauが場補間由来か積分由来かを分ける。

---

## 11. 実行engineと高速化

### 11.1 CPUを第一backendにする理由

初期製品はNumPy＋Numba CPU backendを推奨する。理由は、複雑なwall eventとactive/inactiveの
分岐、unstructured mesh locationとBVH query、float64、1回限りのcase準備に適し、既存Python利用者へ
配布しやすいためである。JAX/GPUは規則格子・分岐の少ない経路では強いが、最初から全機能の
共通backendにするとcompile shapeとcontrol flowが設計を支配する。

### 11.2 データ配置

particle state/propertyは固定particle IDに対応するflat SoAで保持し、active/pending indexだけを整理する。
accepted endpointだけがstateとfield hintをcommitし、trial/output sampleは変更しない。
全stateはresident、stage/proposal/event/OU workはmemory-fitしたworker専用bounded slabへ置く。
各rowのtarget time、residual、depth、interaction/event ordinalをflat数値配列で保持する。
terminal stuck/heldは通常更新を止め、final/frameのsnapshotを保持する。

prepared field/geometry/model factsはread-only、resident state/hints/replayの更新は排他的rowとする。
共有transport、worker scratch、writer/handoff/serialization reserve、page padding、safety marginを
一つのmemory planへ含める。全Nのproposal/event bufferや全Kstep outputを先行確保しない。
Brownian work容量は到達し得るadaptive max depthから保守的に決める。
最小slabが収まらなければrun開始前に拒否し、要求worker数を黙って減らさない。

predictしたowned arrayとOS private commit/WSを区別する。Python/JIT/native allocatorを含むhard capではなく、
WS合計は共有pageを重複計上し得る。正確なbyte式とcapacityはcpu memory planと
[性能文書](solver/docs/parallel_execution_plan.md)が所有する。
out-of-core、別slot scheduler、native teamは現在の工程に追加しない。

fastmathを無効にし、model加算順、prefix/compactionとcanonical event順を固定する。
worker/slab/output schedule、checkpoint-resumeで科学payloadとRNGを変えない。
Numbaの内部parallel maskやthread IDに依存せず、同じ既存数値bodyを親またはworkerから使う。

### 11.3 lookup共有と拡張性

prepareは有効modelからrequired fieldの最小集合を解決する。同じlayoutのelement lookupを共有し、
同じslabのstageでcharge/forceを固定順に評価する。別layout間にlookup共有を強制しない。
追加JIT化は停止した。profile、cache適合、model追加は具体的利用caseの独立変更として扱い、
物理組合せごとのkernelや常設backend selectorを先回りして増殖させない。

### 11.4 将来のGPU等

GPU/別言語は現行profileと用途が必要性を示した時に検討する。
同じcase/model/RNG/event/result意味論を保ち、float64 accuracy、event outcome、統計、compile/転送/I/O込み
end-to-end時間とmemoryを評価する。bitwise CPU/GPU一致を一般要求にしない。
未指定backendの詳細設計や追加dependencyを0.3.0へ持ち込まない。

### 11.5 memoryと出力

全粒子×全時刻×多数列をmemoryへ保持しない。現在状態はresident、選択frame/probeとevent/seriesはboundedに
streamする。exact byte式と実memory資格はcpu memory planと性能文書、永続列はresult形式が所有する。

標準resultはfinal、release/boundary/failure event、選択trajectory frame、選択probe、lifecycle seriesである。
JSON manifestとsegmented HDF5、A/B checkpoint、`LATEST`を使い、形式を並存させない。
frameはeventに対して右連続、holdはhit位置/速度/電荷のinactive snapshot、stickはpost-event速度0、
escapeは以後frameに行を持たない。finalのlogical nullと非権威な最後hit payloadをfailureのNaNと区別する。

epoch順はsegment close/rename → inactive checkpoint replace → `LATEST` replace。
`LATEST`が唯一のcommit pointで、全task/output消費後のaccepted macro barrierに限り既存work cadenceで判定する。
resumeはLATESTより新しいorphanを無視し、最初のcommit前は初期状態から再実行する。
final publicationはfinal → complete manifest → `_SUCCESS` → directory renameで、途中状態も次runが検査する。

参照checkpoint/最新segmentのhashと全segment構造/countを検査する。過去segmentの同shape値改変、power loss、
remote filesystem、同一OUTへ複数runは保証外である。worker writerやbackground queueを追加しない。
公開boundary orderは既存readerが所有し、到着順storageを物理順として使わない。

### 11.6 性能計測

同一hardware/input/model/dt/seed/outputでcold/warm、公開API時間、総process CPU、科学identity、
owned array、sampled private/WS、cleanupを測る。worker起動/import/JIT/IPC/writer/shutdownを含める。
粒子数、macro仕事量、活動粒子の時間変化、field/event/outputを区別し、
短時間・小Nの不利も保存する。kernel単体速度やprofile self timeをend-to-end改善へ読み替えない。

10^4/10^5/10^6の重いmatrixはrelease/manualで実施し、毎PRに重複させない。
現行v62の主用途、時間場補足、COMSOL少粒子full-step、P14-U 1万粒子320stepを別scopeで扱う。
数値・性能・memory判定の詳細は[性能文書](solver/docs/parallel_execution_plan.md)が所有する。
旧thread試作や以前のsourceの値をこの文書へ複製して現行資格にしない。

COMSOLに対する速度比にはhardware、実方程式/field、刻み、粒子数、保存量を併記する。
物理や出力を減らした計測を「COMSOLより高速」の根拠にしない。

## 12. 結果、可観測性、障害

### 12.1 最小限のrun manifest

各実行は次を必ず残す。

- case/config/content hash、code version、backend、hardware概要
- 展開済みphysics/integrator/output plan
- seedとRNG algorithm version
- 展開済みevent設定、boundary law/priority、algorithm revision
- 粒子数、step/event/reject数、状態別最終数
- refinement深さ・interaction数と、hit/no-hit/indeterminate/failure reasonの集計
- wall-clock内訳、peak memory、出力bytes
- accuracy/integrity warning count

これはデバッグdumpではなく、結果の再現性に必要なprovenanceである。

### 12.2 failure分類

```text
InputError       schema/unit/hash/local connectivity/reference/parameter
CapabilityError  未対応の座標×物理×backend組合せ
PhysicsError     モデル適用域外、非有限derivative
GeometryError    point location/eventが決定不能
FieldError       spatial/temporal support外
NumericalError   reject継続、非収束、accuracy budget
ResourceError    memory/disk/backend resource
```

粒子単位で継続できる障害とrun全体を無効にする障害をmanifestで明示する。`NaN`を出力して成功扱い
することも、任意epsilonへclampして続行することも避ける。escaped後の座標・速度は
`kinematics_valid=0`のlogical nullとし、有限だが非権威なpayloadと計算失敗のNaNを明確に区別する。
P08では、粒子に局在できるevent budget、不定event/boundary/departure、field support、model applicability、
nonfinite physicsを小さいfailure reasonへ変換し、他粒子を継続する。入力/schema、静的・共有model/field bound、
topology、writer/publicationなどrun全体の意味を失う問題はrun-fatalな`SimulationError`とする。

### 12.3 診断の範囲

coreはcounterと指定probeだけを出す。原因分析、COMSOL列との照合、グラフ、decision treeは
`tools/vv/comsol`が担当する。常設の巨大な診断frameworkをsolverへ組み込まない。

### 12.4 analysisとvisualization

`open_result`はsegmentを隠したlazy `ResultView`を返し、particle ID、時刻、event typeで選択読込みする。
`tools/analysis`はfate、source-to-target、deposition、arrival time、impact energy/angle、residence、
ensemble confidence intervalを計算する。`tools/visualization`はResultViewまたはanalysis tableだけを
描画し、solver private stateやCOMSOL adapterをimportしない。

---

## 13. V&V戦略の要約

詳細手順は [vv_methodology.md](vv_methodology.md) に分離する。判定順序は固定する。

1. 入力・単位・topology・provenance
2. COMSOL参照軌道上のfield sampling
3. 同じ`x,v,Z,t`で各forceとcharge derivative
4. Brownian offの1 stepとRK stage
5. wallなしの全時間軌道
6. 最初および複数のwall event
7. 決定論ensemble
8. stochastic ensemble
9. dt/mesh/cache収束
10. 製品規模performance

ある層が不合格なら、その先の総合軌道を調整して直そうとしない。たとえばfieldが違うのに
drag coefficientを変更して最終位置を合わせる行為を禁止する。

### 13.1 軌道指標

- 共通時刻へ補間した位置・速度・電荷・各forceの誤差時系列
- L∞、time-weighted L2、p50/p90/p99、最大値
- chamber scale、局所mesh size、粒子の移動距離で正規化した誤差
- 最初に閾値を超えた時刻と、その直前のoperator差
- active区間、event前後、終了後を分けた評価

最終位置・最終statusは補助指標であり、主判定ではない。

### 13.2 event指標

- hit/no-hit、boundary ID、outcomeの混同行列
- event ordinalごとの時刻差・位置差
- pre/post velocity、normal、charge
- event順序の一致
- stick/escape/reflection率と到達時間分布

COMSOL保存時刻間にeventがある場合、保存行の初回status変化ではなくevent時刻を使う。

### 13.3 stochastic指標

同一seedのparticle-wise一致は、乱数増分とstepが一致するときだけ要求する。それ以外は同一点
多数replicaと複数seedで、平均、共分散、MSD、分位、occupancy、survival、wall別到達確率、
Wasserstein/energy distanceを信頼区間付きで比較する。

### 13.4 新たに必要なbenchmark

これは対応stageで順次追加するcatalogであり、初期必須suiteではない。初期は
`model_dataset_rebuild_plan.md`のV01～V06だけを作る。その後、現`model_dataset`へ必要に応じて次を追加する。

- Brownian offの12 deterministic case
- 10 µsごとのmicro-traceとRK4全stage（少数粒子）
- 同一点からのBrownian replicaと内部random increment replay
- uniform acceleration、linear drag、一定E、charge relaxationの解析case
- 平面stick/specular/restitution/probabilistic、corner、axis、thin obstacle、複数hit
- 真のsurface release、面積重み、発生直後departure
- escape、stick、reflectionの正例、および外部adapterのfreeze/disappear対応表
- time-dependent field、2D RZ→3D、small 3D tetra/triangle case
- P1 triangle、Q1 quad、将来P2の補間oracle

---

## 14. テスト方針と複雑化防止

### 14.1 三層だけにする

```text
tests/verification
  解析解、manufactured solution、収束次数、幾何event、保存則

tests/scenarios
  surface release、wall response、field support、charge couplingの代表case

tests/performance
  10^4/10^5/10^6、memory、throughput、scaling
```

公開API、数値結果、物理不変量、schema compatibilityを検査する。COMSOL回帰は
`tools/vv/comsol`のsuiteとしcore testへ入れない。private関数の所属、helper名、
ファイル数、re-exportの形をテストしない。coverage率を物理妥当性の代替にしない。

### 14.2 一つの不変条件に一人の所有者

- unit/quantity：case adapter
- topology：geometry builder
- capability：plan compiler
- parameter range：physics spec
- runtime finite check：kernel boundary
- result schema：result writer

同じ条件をparser、preflight、runtime、writer、compareで少しずつ再実装しない。外側は内側の
typed resultを利用する。

### 14.3 complexity budget

絶対的な行数gateではなく、review時の停止条件として使う。関数complexityの観測と上限は
[quality_tooling_plan.md](quality_tooling_plan.md) のRadon規約を正とする。

- 新規・変更functionは原則CC 10以下、数値上の理由がある場合でも15以下とする。
- file分割は行数やscoreではなく、独立した変更理由が二つあるかで判断する。
- 一つのfeature追加が5層以上のswitch文変更を要求したらplan/spec境界を直す。
- production粒子loopへPython callbackを入れない。
- 同じ物理式のCPU、GPU、reference版は、共通のgolden vectorで検証する。
- 新経路を足す際は旧経路の統合・削除方針を同じPRに含める。
- 1 PRは一つの数値判断。小さなADRへ「なぜ」を残す。
- 古いbenchmarkや失敗caseを削除してgreenにしない。

### 14.4 設計判断の記録

ADRを機能ごとに増やさず、主仕様のdecision tableに理由を集約する。外部形式の破壊的変更、
物理結果を変える式変更、backend間の再現性規則など、後から戻しにくい判断だけ短い記録を残す。

---

## 15. 現行milestoneと拡張条件

0.3.0はv62の単一engine、既存CPU数値owner、同一runのprocess並列、共有transport、
bounded slab、single writer、durable checkpoint/resumeを提供する。
標準品質、用途別科学/性能/memory、Windows/Linux local gate、実installed-wheel入口は完了した。
source別の資格・不合格・制約は性能文書が所有し、個別caseの観測を全用途へ外挿しない。

完了した初期bootstrap/Stage 1の手順、旧native chain/team工程、90日ロードマップは現行計画から除いた。
実装責務の索引、条件付きbacklog、変更順、0.3.0提供範囲は
[implementation_plan.md](implementation_plan.md)第16～21節を参照する。

完全3D、別物理、GPU/別言語、NUMA、out-of-coreは具体的利用case・独立reference・profileを先に用意する。
既存のmodel/field/event/RNG/result意味論を同じengineへ統合し、置換した旧経路を同じ変更で削除する。
追加JIT/native teamの再開や全CPU数比例効果を今回の必須工程へ戻さない。

## 16. 将来項目：複数macro連続実行

現在は全workerの完了を1 macroごとに待ち、同期間隔の公開設定はない。
将来は数値dtと独立した最大連続macro数を検討する。積分刻み、field knot、release、stage、
Brownian counter/条件付き分割、毎macro人口、要求frame/probe、bounded output、single writerを維持する。
checkpointは全workerが同じaccepted時刻まで完了しoutputを消費した状態だけを公開する。
child途中死・resume・将来の相互作用/field feedbackの連成周期も含め、
[実装計画第21.3節](implementation_plan.md#213-将来項目複数macroの連続実行)の条件で設計・評価する。
0.3.0へAPI/config/stubを先行追加しない。

## 17. 採否を決める主要設計判断

| 論点 | 採用 | 採らない案 | 理由 |
|---|---|---|---|
| 実装開始 | clean-room package | 現runtimeを整理 | 正しさと性能経路が絡み、benchmark未対応 |
| 場の権威 | mesh-native primitive | 粒径別derived CSV | 派生値の陳腐化を防ぐ |
| 補間 | shape function | solid越しDelaunay | topologyと境界を保持 |
| 四辺形 | Q1 | 暗黙tri分割 | COMSOL補間意味を変えない |
| 境界 | continuous first event | step終点inside判定 | thin-wall飛越しを防ぐ |
| 状態 | physical/safety/accuracy分離 | 単一stop code | 原因と物理結果を混ぜない |
| COMSOL比較 | 外部V&Vが通常の`rk4_fixed`設定を指定 | COMSOL名のcore profile | 方法差を物理差に見せない |
| 高速化 | fused CPU batch first | 全機能JAX first | event/unstructured/float64に適合 |
| 乱数 | counter-based | global sequential RNG | 並列順序に依存しない |
| 出力 | streaming + events | 全N×T memory | 10^6粒子へ拡張可能 |
| 比較 | operator→trajectory→event | 最終座標中心 | 最初の原因を特定できる |
| テスト | 数値・物理・公開契約 | private tree shape | 再設計可能性を保つ |

---

## 18. 技術根拠と一次資料

本設計では、COMSOL固有の挙動は公式documentation、数値実装は一次資料または公式library文書を
根拠にする。主要な参照先は以下である。

- COMSOLの対象CF4/O2 etching model：
  [ICP RF Bias CF4/O2 Silicon Etching](https://doc.comsol.com/6.4/doc/com.comsol.help.models.plasma.icp_rf_bias_cf4_o2_si_etching/icp_rf_bias_cf4_o2_si_etching.html)
- COMSOL Particle Tracing全体：
  [Particle Tracing Module User's Guide](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/ParticleTracingModuleUsersGuide.pdf)
- 粒子運動方程式と数値解法：
  [Theory for the Mathematical Particle Tracing Interface](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_introduction.02.05.html)
- 壁条件：
  [Wall](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.43.html)
- Brownian力：
  [Brownian Force](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.08.html)、
  [Brownian Motion model](https://doc.comsol.com/6.4/doc/com.comsol.help.models.particle.brownian_motion/brownian_motion.html)
- 希薄気体drag：
  [Drag Force](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.37.html)
- charge accumulation：
  [Charge Accumulation](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.33.html)
- COMSOL補間・結果設定：
  [Interpolation](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_ref_results.37.224.html)、
  [Mesh API](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_api_mesh.49.047.html)
- eventを持つreference ODE solver：
  [SciPy solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
- CPU JIT/parallel：
  [Numba performance tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html)、
  [Numba automatic parallelization](https://numba.readthedocs.io/en/stable/user/parallel.html)
- JAXのPRNG、dtype、制御、cache：
  [Random numbers](https://docs.jax.dev/en/latest/random-numbers.html)、
  [Default dtypes and X64](https://docs.jax.dev/en/latest/101/default_dtypes.html)、
  [Control flow](https://docs.jax.dev/en/latest/201/control-flow.html)、
  [Persistent compilation cache](https://docs.jax.dev/en/latest/501/compilation-cache.html)
- counter-based RNG：
  [Random123 official documentation](https://random123.com/releases/docs/)
- chunked/streaming storage：
  [h5py dataset documentation](https://docs.h5py.org/en/stable/high/dataset.html)

公式文書だけでは固定されないCOMSOLの乱数順序、内部stage、finite element recoveryは、推測で
coreへ実装せず、専用exportとmicrocaseで観測して外部V&V manifestへ閉じ込める。

---

## 19. 実装判断の原則

入力、field、physics、integrator、event、law、outputの責務を一意に保ち、差が出た最初の層を確認する。
解析解と収束で数値意味を説明し、同じ入力のend-to-end時間とmemoryで性能を評価する。
機能追加は同じengineへ統合し、置換した旧code・設定・test・文書を同じ変更で削除する。
