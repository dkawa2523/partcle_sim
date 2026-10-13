# 半導体チャンバー粒子輸送基盤 実装・保守計画

## 0. 文書の目的と権威

本書は、[product_specification.md](product_specification.md) と、architecture authorityである
[architecture_proposal.md](architecture_proposal.md) を、実装担当者がそのまま作業へ分解できる粒度へ
落とした計画である。[architecture_review.md](architecture_review.md) は設計判断時点のreview recordであり、
module ownershipや依存方向のauthorityではない。物理式と数値的根拠は
[technical_research.md](technical_research.md)、開発環境と品質gateは
[quality_tooling_plan.md](quality_tooling_plan.md) を参照する。実装中のagentは
[AGENTS.md](AGENTS.md) を必ず適用する。

文書間で重なる場合の権威は次とする。

1. 製品目的、対象範囲、非目標：`product_specification.md`
2. module ownership、依存方向、runtime境界：`architecture_proposal.md`
3. 方程式、数値method、適用域：`technical_research.md`
4. 実装順、作業単位、stage出口条件：本書
5. Python環境、静的品質tool、quality gate：`quality_tooling_plan.md`
6. COMSOL比較、参照dataset診断：`vv_methodology.md` と外部tool

`architecture_review.md`は上記authorityを評価した記録として読む。review後の修正で両文書が食い違う場合は、
`architecture_proposal.md`を正とし、review recordをproduction仕様として補完しない。

本書は新しい契約frameworkを追加するものではない。`DataBundle / SimulationSpec / PreparedRun`、
`PhysicsPlan`、`StepProposal`、`BoundaryEvent`、`ResultStore`という五つの変更境界を、実装順へ
対応させるためのものである。

---

## 1. 現行製品と実装範囲

packageは`chamber-particles 0.3.0`である。production engineを含む現行revision一覧は
[result形式](solver/docs/result_format_v3.md)と実行manifestが所有する。
公開済み0.1.0/0.2.0のtag、資格、証跡は固定した履歴として維持する。

| 領域 | 現行で実装した内容 | 今回のscopeに含めないもの |
|---|---|---|
| 座標・場 | Cartesian XY、RZ meridional、static/固定topology linear-time、regular/P1/Q1 | 完全3D、moving mesh、高次要素 |
| 積分 | ballistic/一定加速度exact、固定RK4、exponential midpoint、inertial OU | 一般ODE/DAE framework、適応高次controller |
| 物理 | fixed/continuous charge、対応drag、電気、重力/浮力、ion drag、熱泳動、DEP、lift | 適用域外modelの暗黙代替、実験的製造予測保証 |
| 確率 | counter RNG、joint OU、RZ投影、active wall後restart、条件付き境界refinement | 連続OUのexact first-passage、3D Brownian |
| 境界 | stick/escape/hold/specular/restitution/Maxwell/probabilistic stick、独立接触半径、XY translation periodic | resuspension、rolling/sliding、moving geometry、surface charging |
| 実行・出力 | 同じengineの1/複数process、bounded slab、共有入力/state/replay、single writer、durable resume | thread/backend切替、MPI/Dask、複数stepを跨ぐworker同期設定 |
| 外部tool | canonical adapter、reduced electrostatic builder、field cache、V&V、最小analysis/可視化 | COMSOL依存core、汎用plugin/DI、常時full trace |

表は全組合せの受理を意味しない。対応する座標・model・integrator・boundaryの組合せは
`load_case`/prepareで一度解決し、未対応は具体的理由とともに拒否する。
科学的適用域は物理model owner、schemaはcase形式、用途保証は各登録条件が所有する。
現在の資格と残件は第21.4節、条件付き将来項目は第17節へ集約する。

## 2. clean-room実装の配置

new productionは`particle_platform_redesign/solver/`の独立uv projectだけへ置く。
旧package/import graph/環境を共有しない。`old_code/`はread-only、`model_dataset/`は外部V&V入力である。

```text
solver/
  pyproject.toml / uv.lock / .python-version
  src/chamber_particles/        唯一のproduction package
  tests/verification/           解析解・manufactured・収束
  tests/scenarios/              三APIによる小型end-to-end
  tests/performance/            用途付き時間・memory・identity
  tools/                        adapter・builder・analysis・外部V&V
  scripts/check_complexity.py   Radonのthreshold exit statusだけ
  docs/                         現行契約・性能の所有文書
  evidence/                     hashとscopeを固定した証跡
```

project/lockを環境・dependencyのauthorityとし、通常実行はsolver directoryから`uv run --locked`で行う。
rootへ別project/lockを復元しない。runtime/build/tool versionをこの文書へ複製しない。
将来stageのempty module、NotImplemented stub、一般plugin/DI、第二runtimeを先行作成しない。

## 3. 公開操作と内部データ

### 3.1 公開Python API

```python
from chamber_particles import load_case, simulate, open_result

case = load_case("case.yaml")
summary = simulate(case, output="runs/case_001")
result = open_result("runs/case_001")
```

公開する操作は三つだけにする。

| 操作 | 戻り値 | 責務 |
|---|---|---|
| `load_case(path)` | `SimulationCase` | YAML/HDF5読込み、canonical schemaと静的意味の検査 |
| `simulate(case, output, *, workers=1)` | 小さい`RunSummary` | prepare、計算、streaming出力、finalize |
| `open_result(path, recovery=False)` | lazy `ResultView` | 完了結果または明示されたrecovery結果の遅延参照 |

worker数はAPI/CLIの実行資源であり、case数値設定やRNG keyへ混ぜない。
`PreparedRun`、preflight report、backend object、writer objectは公開しない。巨大なin-memory trajectoryを
返さない。CLIの`check/run/inspect`も上の処理を呼び、別の検査・実行経路を作らない。

### 3.2 公開CLI

```console
chamber-particles check case.yaml
chamber-particles run case.yaml -o runs/case_001
chamber-particles inspect runs/case_001
```

`check`は公開`load_case`だけを実行し、case/data hash、path、座標系、motion modeという読込み時に確定する
canonical identityを表示する。大域geometry、physics runtime、予測memoryを含むprepareは公開`simulate`が一度だけ
所有し、CLI専用のprivate prepareや独立preflightを作らない。

### 3.3 最小例外境界

- `CaseError`：入力schema、単位、局所connectivity・owner参照、静的参照整合の問題
- `SimulationError`：大域geometry prepare不能、event局在不能、run全体を継続できないI/O・内部不変条件の問題
- `IncompleteResultError`：`_SUCCESS`のない結果を通常modeで開いた場合

粒子単位に局在できる数値不能は`failed`と小さいreasonで記録し、他粒子を継続する。
入力不正、共有model/field、writer等run全体の意味を失う問題は即時停止する。
particle failureをstick/escapeへ読み替えず、accuracy不足とsafety failureを区別する。

---

## 4. fileごとの具体的責務

| file | 現行の責務 | 入れてはいけない内容 |
|---|---|---|
| `api.py` | 三APIの薄い呼出し、公開例外の整形 | 物理式、HDF5 dataset操作 |
| `case.py` | frozen設定dataclass、case YAMLのdomain検査、`SimulationCase` | producer固有列、backend kernel |
| `yaml_input.py` | UTF-8 bytesからobjectへの一意なYAML文法解析。重複key・merge keyを拒否 | I/O、hash、domain validator |
| `case_format.py` | `case.h5` schema、read/write、version、hash | COMSOL名、軌道計算 |
| `coordinates.py` | XY/RZ/axis規則、座標・vector・normal変換 | field locate、wall law |
| `geometry.py` | domain、facet、BVH、containment、first-hit query | 反射・付着、field値 |
| `topology.py` | XY translation periodicの対応・transfer | material wall応答、一般periodic framework |
| `fields.py` | layout、strict support、値・時間補間、同じsample locationによるoptional static nodal P1/regular/Q1 gradient | cache生成、mesh修復、COMSOL recovery |
| `sources.py` | canonical internal/surface scheduleのrelease、surface位置解決、発生時刻、weight | 分布sampling、source RNG、engine lifecycle、wall再飛散 |
| `rng.py` | counter key、uniform/normal、stream ID | sourceやwallのpolicy |
| `physics/catalog.py` | category、model code/revision、required field、対応mode | mesh探索、I/O |
| `physics/forces.py` | drag、electric、gravity/buoyancy、後続加算力 | lifecycle、boundary response |
| `physics/charge.py` | fixed/continuous charge、rate・平衡残差、有限invariantとrate/derivative bound | 全状態のstep orchestration、stiff method選択 |
| `physics/runtime.py` / `compiled.py` | prepared model pass、sample済みprimitiveからのcompiled加算 | field locate、event、fallback dispatch |
| `integrators.py` | RK4、exponential midpoint、`StepProposal.state_at()` | BVH、law、file output |
| `stochastic.py` | joint OU、conditional split、stochastic tree | wall law、I/O、第二engine |
| `events.py` | path tube、earliest hit局在、残時間work | 反射式、永続形式 |
| `boundaries.py` | hit後のstick/escape/specular/probability | 交差探索、位置押戻し |
| `engine.py` | prepare、唯一のmacro loop、lifecycle、barrier調停 | importer、plot、COMSOL比較 |
| `cpu.py` | SoA、compiled kernel、bounded slab/workspace、共有transport、総memory plan | modelの物理的意味、永続result |
| `output.py` | ResultWriter、epoch commit、checkpoint、ResultView | deposition等の科学集計 |

final tableの行数はwriterのparticle capacityおよびmanifestの粒子数と一致させる。
非負・厳密昇順のparticle IDはbounded blockで検査し、block境界も比較する。
publish前、`open_result`、`read_final`は同じfile validatorを利用する。
exact source-ID membershipや全datasetの参照整合・checksumはこの検査の保証に含めない。

file分割は「行数が増えた」だけでは行わない。一つのfileが二つの独立した変更理由を持ち、かつその双方を
実際に変更するcaseが現れた時だけ分割する。`models.py`、`helpers.py`、`contracts.py`、`providers/`のような
一般置き場は作らない。

import方向も固定する。

- coreは`tools/`をimportしない。
- `physics/forces.py`と`physics/charge.py`はsample済みprimitive値だけを受け、geometry、fields、I/Oを
  importしない。
- `physics/catalog.py`だけがformulaをmodel codeへ対応させる。formula側からcatalogを参照しない。
- integratorはstage evaluatorだけを呼び、BVH、wall law、HDF5を知らない。
- `cpu.py`はoutputへ書かず、engineがcompute結果とwriterを調停する。
- analysis/visualization/V&Vは`ResultView`だけを読む。

この依存方向はimport-linterの最大三つの高価値contractで管理する。physics、integrator、低位numericsから
outer layerへの逆依存だけを検査し、file単位のcontractや同内容のpytest architecture testは作らない。
moduleが実装されたwork packageでcontractを追加し、contractのためのempty moduleは作らない。詳細は
[quality_tooling_plan.md](quality_tooling_plan.md) に従う。

---

## 5. canonical入力の実装

設定YAML、数値array HDF5の二ファイルを標準とする。schema、列、shape、関連参照、content hashは
[case_format_v3.md](solver/docs/case_format_v3.md)が唯一の所有者である。
本書で旧schemaのlayoutを再定義しない。quantityはSI、量名に単位を含める。

- YAMLは`yaml_input.py`の一意なparserで読み、重複key/merge keyを拒否する。
- canonical read/write、versionとhashは`case_format.py`へ一元化する。
- producer固有名・unit換算・species/quantity意味は外部adapterで正規化する。
- field supportは`fields.py`、geometry/topologyは各owner、model parameterはphysicsが検査する。
- missing/非有限/適用外をdefault、epsilon clamp、別modelで隠さない。
- runtime capabilityはprepareが一度解決し、scalar evaluatorやCOMSOL専用pathへ切り替えない。
- schema破壊変更に互換alias/暗黙migrationを重ねない。

caseの数値設定とworkersなどの実行資源を分ける。input schemaへ同期step数を先行追加しない。
同じvalidatorをparser、preflight、runtime、writerへ複製しない。

## 6. runtime状態とprepare

### 6.1 particle SoA

粒子IDと物理配列の対応はrun中に変えない。ここでSoAはproperty列を別配列にする意味であり、
2成分vectorだけはbatch演算と入出力を単純にするため連続な`[N, 2]`で保持する。P09時点のresident列は次とする。

```text
particle_id[N]                 int64
source_id[N]                   int32
release_time_s[N]              float64
position_m[N, 2]               float64
velocity_m_s[N, 2]             float64
charge_number[N]               float64
mass_kg[N]                     float64
drag_diameter_m[N]             float64
electrostatic_radius_m[N]      float64
displaced_volume_m3[N]         float64
model_weight[N]                float64
material_id[N]                 int32
source_facet_id[N]             int64
lifecycle[N]                   uint8   # pending/active/stuck/escaped/failed/held
active[N]                      bool
terminal_time_s[N]             float64
failure_reason_code[N]         uint16
event_ordinal[N]               uint32
physical_boundary_ordinal[N]   uint32
exact_origin_time/position/velocity       # exact pathの再現用
start_contact_state[N]         uint8       # surface departure状態
active_particle_index[N]       int64       # sorted resident row、容量固定
active_keep_mask[N]            bool        # in-place stable compact用
```

並べ替えるのはactive resident-row indexだけにする。species tableによる圧縮はprofileで属性配列がmemory支配と確認した後に
必要な利用caseで検討する。現行は`cpu.py`が固定容量のactive indexを所有し、
resident rowとparticle identityを分離しないままmicrotile単位でstable compactする。source接触は
`source_facet_id`と`start_contact_state`が所有し、汎用的なlast-contact cacheは置かない。field/geometry hint配列は、現行のregular
locatorが消費せず未使用のO(N)常駐配列になるためP09では作らない。P10はregular layoutにhintを追加せず、
実際に消費する共通P1/Q1 layout一つにだけ`last_field_cell[N]`を追加した。accepted endpointだけをcommitし、
trial、`state_at()`、output sampleはresident hintを変更しない。geometry containment hintとは共有しない。

### 6.2 `PreparedRun`

`engine.prepare()`は次を一度だけ解決した非公開immutable dataを返す。

- coordinate modeとndim
- geometry arrays、boundary BVH、local feature scale
- field layout、locator、sampler code、必要field buffer layout
- physics model code/revision、固定加算順、parameter arrays
- state sliceとparticle property authority
- boundary groupからdense law codeへのmap
- integrator codeとtime policy
- source scheduleとpending index
- counter RNG seed/stream layout
- resident/shared memory、worker専用slab、handoff/同期writer reserveを含むresource plan
- output scheduleとResultWriter設定

hot loopではYAML文字列、dict lookup、capability探索、Python callbackを使わない。未対応組合せはprepareで
拒否し、scalar evaluatorへ黙って切り替えない。
YAMLの`boundaries[].law`は入力selectorであり、prepare時に非公開dense `law_code`へ解決できるが永続化しない。
eventの`law_id`は選択されたtop-level lawの安定semantic IDとし、compound lawのleaf分岐は`outcome`と
draw referenceで表す。`law_name`という別名を作らない。

### 6.3 memory planner

resident input/state/property、active/pending index、geometry/BVH、field snapshots、prepared facts、
共有transport/replay、worker専用scratch、writer/handoff/serialization reserve、page padding、safety marginの
同時liveを`cpu.py`の総予算へ含める。workerごとのscratchはmemory-fitしたbounded slabで再利用する。
全N proposal/event bufferを作らず、Brownian workは到達し得るadaptive max depthから保守的に計画する。

通常利用者へslab幅を要求しない。完全なplanはsimulateのprepareが一度解決し、最小slabでも予算を
超える場合はrun開始前に拒否する。要求worker数を暗黙削減してmemory不足を隠さない。
`check`はload_caseの薄い呼出しに留め、実行prepareや別validatorを作らない。

owned arrayのphase peak/componentとOS private commit/WSは別の指標である。
Python/JIT/native allocatorを含むOS hard capを意味せず、WS合計は共有pageを重複計上し得る。
共有memoryだけで総OS memory削減やNUMA配置を保証しない。
粒子stateはresident、scratchだけboundedとし、out-of-coreは実測上必要な用途が定まるまで追加しない。
正確なbyte式とcapacityは[性能文書](solver/docs/parallel_execution_plan.md)とrun memory planが所有する。

## 7. field、geometry、座標の実装

### 7.1 field sampling

P03のreferenceは一物理点を一つのproduction経路でsampleし、小さいimmutable resultを返す。P10で
同じ意味論をtile向けpreallocated output bufferへ移す。P03をbatch interfaceへ見せるwrapperや、P10で
別のsampling policyを作らない。戻り意味論は次で固定する。

```text
values
cell_id
support_inside
distance_or_reason when outside
```

実装順序は次とする。

1. P03：regular bilinear、P1 barycentric、Q1 isoparametric Newtonのsingle-point reference
2. P03補正：large-offset/high-aspectでも平行移動不変なconditioning-aware location
3. P10：regular supported containing-cell common pathのO(1)候補、P1/Q1 previous-cell strict-interior
   fast pathとNumba内full-search fallback、同じ意味論のcompiled tile sampling（完了）

P03補正では物理空間の後退誤差、element scale、座標ULP、局所Jacobianのconditioning、再構成残差から
inside判定を一度だけ解決する。悪条件cellのinside領域をtoleranceで無制限に広げず、algorithm revisionが
根拠とともに固定するmesh品質上限を
超えるcellと非有限補間を明示errorにする。large-offset/high-aspect P1/Q1と片側support共有面を必須回帰にする。

trial stageがparticle domain外へ出ても先行wall eventを局在できるよう、物理距離が最小のsupported cellへ
local coordinateを射影した有限値をprovisionalとして返してよい。ただし`support_inside=False`を保持し、
そのproposalをeventなしで受理しない。値の有限化とsupport判定を一つのbooleanへ潰さない。

現行で運動を駆動するrequired fieldは、particle domain全体を覆うcontinuous node-associated
regular/P1/Q1 fieldに限定する。cell-associated quantityはcanonical dataとして保持できるが、共有面の値を
cell IDで選ぶ物理modelが定義されるまではforce/charge入力に使わない。各required fieldのunit、component、
stored basis、data座標との整合をprepareで一度だけ検査する。

producerのNaNをcanonical coreへ持ち込まない。外部adapterは明示supportを先に確定し、supported cellが参照する
DOFだけを物理値としてfinite必須にする。masked-only DOFは決定論的な有限placeholderへ正規化し、その方法と
件数をprovenanceへ残す。placeholderを物理補間へ使用せず、NaN自体からsupportを推測しない。

cell hintは物理状態ではない。trial中に得たhintはparallel tile scratchへ置き、accepted endpointだけresident
`last_field_cell`へcommitする。棄却したtrialが次の探索経路へ影響してはならない。

### 7.2 geometryとfirst-hit query

Stage 1A geometryは2D/RZのline2 boundaryとする。

- domain cell connectivityからcontainmentを判定する。
- prepareでvolume edge incidenceを導出し、non-manifold、重複boundary、内部edgeのwall登録、外周欠落を拒否する。
- RZ axis seam以外の全外周edgeを一度ずつboundaryへ対応させる。現行は内部thin baffleを受理しない。
- boundary edgeのAABB BVHをprepareで構築する。
- queryはcandidate facetと交差parameterを返し、lawを決めない。
- cornerでは時間差が局在budget内のfacet集合を全て返す。
- epsilon位置押戻しをしない。

surface releaseは初期位置を境界上に保持する。`source_facet_id`と`start_contact_state`から、同じfacetを
離れるzero-time contactだけを認証して無視し、正の時間後の再衝突は通常eventとする。一定加速度の内向き
departure stateは別wall hitまで保持し、一般RK4では各intervalのpath enclosureから離脱を再認証する。

### 7.3 座標mode

dataの座標表現とparticle motion modeを同じflagで所有しない。P04で未release v1 schemaを整理し、
HDF5は`cartesian_xy | axisymmetric_rz`のdata表現、YAMLは
`cartesian_xy | axisymmetric_rz_meridional`のmotion modeを明示する。prepareが有効な組合せを一度だけ
検査し、`run.json`へ両方を保存する。互換aliasは作らない。

- `cartesian_xy`：2D Cartesian verificationと単純case
- `axisymmetric_rz_meridional`：RZ二自由度。軸横断は`coordinates.py`の基底変換
- `axisymmetric_field_cartesian3d`：Stage 2A。XYZ stateからRZ fieldをsample
- `cartesian_xyz`：Stage 4B。tet4/tri3だけから開始

RZ meridionalのsurface sourceを外部でrealizeする際はsampling measureを必須選択にする。

- `meridional_length`：断面長に比例
- `revolved_area`：物理的な回転面積 `2πr ds` に比例

暗黙にどちらかを選ばない。半導体チャンバーの物理surface releaseは通常`revolved_area`で外部生成し、
edge選択だけでなくedge内の位置も`r(s)`に比例する密度からsampleしてcanonical rowへ固定する。

### 7.4 remesh/cacheの境界

runtimeはremeshもregular cache生成もしない。B1/B2の`tools/field_preprocessor`は設定format v2、producer v4で、nodal
P1/regular/exact affine Q1 sourceからfull-support regular targetへのstatic/linear-time cacheを生成する。source/targetの
共通cell partitionをstreamし、binary float64座標のexact rational clippingでsource overlapとtarget coverageを別検査する。
正面積sliverを捨てず、積分patchのfloat64 collapseやproduction locatorのowner不一致は`unresolved_validation`にする。
warped Q1、unstructured/partial targetは公開拒否。旧固定stencil validatorや第二runtime samplerは残さない。

公開条件はvalue、stored-component gradient、target support boundary valueのweighted relative L2である。
XYは面積/長さ、RZは`2 pi r`付き面積/長さを使う。positive 4×4 Duffy Gauss ruleによるpolynomial空間積分と
nodal/basis/coordinate/accumulation roundoff allowanceから`relative_l2_upper`を比較する。reference lower normが0のときはerrorと
allowanceも0の場合にだけrelative errorを0とし、それ以外は停止する。sampled maxはL∞保証ではなく、support boundary normは
材料wall近傍・trajectory/event同等性を認定しない。
source/candidateの同step公開API比較と実測speedup・memoryは、cacheの用途別採用判断として別に実施する。

`validation.memory_limit_mb`、`workspace_rows >= 16`、`max_patch_work`は必須であり、owned numeric arrayとwriter/index
transientのmemory planをallocation前に検査する。canonical readにも同じnumeric-array limitを渡す。coverageと全fieldで共有する
work counterをAABB行とintersection作業へ適用し、bounded sampled/gradient rowsとpatchごとの小さいrational polygonを使う。
元のfields/layouts/geometry/sourcesとcache値はresident保持し、process RSS hard capやout-of-coreは主張しない。
資源・pair・誤差・設定hashはoutput provenanceへ残し、公開前に既存required-field preflightでparticle domainとRZ axis規則も検査する。

gradientは`fields.py`のoptional `PreparedFieldSet.spatial_gradient`が所有する。直前sampleのsupported cell ID/basisを共有し、
caller-owned `[N,C,2]`へP1/regular/Q1のstored component偏微分を書き、通常stageのscratchを増やさない。
第二locator、least-squares gradient、RZの3D covariant gradient、COMSOL回復値との同一視を導入しない。
隣接snapshotの空間Gram momentsから時間norm比をbounded Bernstein上界とroundoff allowanceで検査し、
interval全体の最大値をgateへ使う。推定stationary rootを証明に使わず、時間相殺・reference下界不足・資源超過は無公開とする。
現revisionはpreprocessor `canonical_field_preprocessor_v4`、validation `polynomial_common_partition_time_interval_weighted_norm_v3`、
support `exact_binary_rational_unique_coverage_v3`、field semantics `weighted_polynomial_space_linear_time_cache_v3`、
gradient `static_nodal_p1_regular_q1_gradient_v2`である。canonical schemaとproduction value-location/interpolation revisionはこの変更で増やさない。
現行はphysics設定が参照する一つの明示layoutをrunのauthorityとする。cacheを使う場合、外部preprocessorが
別fieldまたは別DataBundleとして作成し、利用者が明示選択する。runtimeが場所ごとにreference/cacheを
暗黙切替するhybrid modeは、実ケースで必要性とspeedupが確認されるまで実装しない。

---

## 8. physicsと電荷の実装

### 8.1 `PhysicsPlan`

category mapはrunごとに0または1 modelとする。

```text
drag                  0/1 linear_relaxation owner
charge                exactly 1; fixed means dZ/dt=0, Z0 belongs to each source
noise                 0/1
electric              0/1 explicit_acceleration
gravity_buoyancy      0/1 explicit_acceleration
thermophoresis        0/1 explicit_acceleration
dielectrophoresis     0/1 explicit_acceleration
lift                  0/1 explicit_acceleration
ion_drag              0/1 explicit_acceleration
```

各catalog entryは`id/revision/category/contribution kind/required fields/parameters/applicability/
supported coordinates/supported integrators/model code`だけを持つ。継承階層やregistry discoveryは不要で、
静的dictと明示的なresolverでよい。

### 8.2 stage evaluator

integratorが呼ぶ内部処理は一つにする。

```text
evaluate_stage(ids, t, x, v, z, scratch):
  1. required fieldsを一度sample
  2. charge rateまたはfixed chargeを評価
  3. linear relaxation rate/targetを評価
  4. enabled additive accelerationを固定順で加算
  5. support/applicability/nonfinite flagを返す
```

forceごとのN粒子temporary arrayを作らず、同じacceleration bufferへ固定順で加算する。最初からmodel組合せ
ごとのkernel code generationは行わない。profileで複数passが支配的と分かった組合せだけ、同じ式revisionの
fused kernelを後で追加する。

### 8.3 Stage 1のmodel

- `stokes_cunningham`
- `epstein_linear`
- `electric_coulomb`
- `gravity_buoyancy_standard`
- `fixed_charge`

model式revision、係数、適用域を`docs/physics_models.md`とcatalog entryの両方へ重複記述せず、catalogの
machine-readable ID/revisionと文書の説明を対応させる。現行productionの適用域policyは`error`だけとし、
範囲外はfail-closedで拒否する。別dragへ自動切替したり、未検証のまま件数だけ数えて続行しない。

### 8.4 continuous charge

P15では位置・速度・電荷を同じstep strategyで連成し、production実装まで完了した。

- 最初のmodelは`oml_stationary_maxwellian_debye_huckel_v1`とする。stationary Maxwellian OMLの
  電子・イオン収集rateとDebye–Hückel表面電位を一つのversioned式として扱い、ion driftと
  `a / lambda_D`の適用域を明示gateにする
- `physics/charge.py`はrate、平衡残差、有限なcharge invariant、`|R_Z|`と
  `L = max |dR_Z/dZ|`のboundだけを所有し、field探索やstep orchestrationを持たない
- 最初に受け入れたproduction sliceは既存`rk4_fixed`で`(x,v,Z)`を同じ4 stageにより評価する。
  全accepted intervalで`hL <= 0.5`を証明できる非stiff caseだけを受理し、証明不能または超過は
  implicit fallbackやcharge-only subcycleへ切り替えずfail-closedにする
- RK4は現行でもexplicit `hL <= 0.5`を要求する。native指数運動とB03は同じ予測midpointで
  `G=dZ/dt`と`J=dG/dZ<=0`を凍結し、root基準のaffine lawを`expm1`で解析更新する
- exponential pathは`hL<=0.5`をstability gateにせず、精度は別runの`h,h/2,h/4`で選ぶ

chargeだけを小刻みに進めて運動へ後から渡す一次operator splitを使わない。現行pathはclip、charge-only
subcycle、第二engine、専用multirate法、一般IMEX frameworkを持たない。

---

## 9. integrator、StepProposal、event

### 9.1 `rk4_fixed`

`(x,v,Z)`を一つの状態として、4 stageの各`t,x,v,Z`でfieldとphysicsを再評価する。既知の線形dragで
`h/tau_min >= 2.5`となるcaseはprepareで拒否する。固定stepとは精度目的のhidden adaptivityを行わない
意味であり、wall eventの局在とfield discontinuityでは必要な区間だけ再積分する。

### 9.2 `exponential_midpoint`

`dv/dt=-(v-u)/tau+a`の局所解析更新を使う。`A=-expm1(-h/tau)`と小引数級数を実装し、start係数の
half-step predictorでmidpointのfield、charge、drag、加算加速度を評価する。非線形dragを線形緩和へ
黙って近似しない。`1/tau → 0`では一定加速度式へ連続的に移行し、極小・極大`h/tau`の双方で有限性を
verificationする。

continuous chargeは同じmidpointの`G_mid,J_mid<=0,Z_mid`から
`A_Z=G_mid+J_mid(Z_0-Z_mid)`を作り、`Z_1=Z_0+expm1(J_mid*h)/J_mid*A_Z`で更新する。
`J_mid=0`は`Z_0+h*A_Z`の連続極限である。運動とchargeは同じ`StepProposal`/rootを所有する。

### 9.3 `StepProposal`

初期実装の非公開dataは次を持つ。

```text
start_state / end_state
t0 / h
accepted numerical piece representation
state_at(theta)
path enclosure or explicit "not available"
finite stage/endpoint values and provisional validity flags
```

`StepProposal`へ未使用placeholderを先行追加しない。local truncation estimateとgeometry/roundoff budgetは、
それを実際に使うmethodまたはevent planが所有する。現行`rk4_fixed`の精度は別runの`h / h/2 / h/4`で評価し、
`dt/tau` gateは安定性条件であってaccuracy estimateではない。

P04/P05のballistic proposalは`x(theta)=x0+theta*h*v0`という厳密な直線path、厳密な一定speed、解析的な
`state_at(theta)`を返す。この能力だけでline boundary first-hitを完成させる。

P06でforce-coupledな曲線pathを追加する時は、full stepとtwo half stepsの差をaccuracy indicatorとして使えるが、
それだけを軌道包含の証明済みupper boundとは呼ばない。enclosureはversioned integratorが定義する離散pathを
包むもので、真のODE解のlocal truncation errorまでは含めない。physics側の係数・加速度boundから保守的
enclosureを構成できるcaseだけcheap no-hitを許可し、構成できないcaseは区間分割または明示failureにする。放物線、
near-grazing hit/no-hit、turning trajectoryのmanufactured caseとstep半減でevent収束を検証する。Stage 1Bでは、
壁までのclearanceが保守的最大移動量より十分大きいparticleだけcheap no-hit pathを許可し、近壁では同じ
reference規則へ戻す。これは宣言済みのevent algorithmであり、別solverへのfallbackではない。

`state_at(theta)`は任意多項式補間で次状態を作らず、同じintegratorをstart stateから`theta*h`だけ再実行する。
revision 3aのboundaryless macro proposalでは、出力用に評価したstateを次stepへ引き継がない。revision 3bの
材料eventでは、geometry-drivenに確定した逐次pieceのendpointだけを次pieceへ引き継ぎ、frame時刻は分割を
発生させず、その時刻を含むaccepted pieceから評価する。

P06以降の曲線pathは次の単純な順序で実装する。

1. revision 3aでは、boundaryless Cartesian XY、fixed charge、全cell supportedな`RegularLayout`に範囲を
   限定した。canonical node extremaと既存model係数から、Epstein/electric/gravityを合成した各RK4 stageの
   速度・加速度boundを作る。
2. 全macro区間だけでなく、`state_at()`が作る任意の短縮区間について、内部stage位置・速度とaccepted endpointを
   含む連続path enclosureを外向き丸めで構成した。同じboundでEpsteinの`lambda/a`下限と
   `|u-v|/c_bar`上限も連続的に証明する。
3. enclosure全体がregular support box内にあり、applicabilityも全区間で成立するproposalだけを実行する。
   証明不能区間はhidden subdivisionで別の数値pathへ変えず、出力scheduleに依存しない形でfail-closedにする。
4. revision 3bでは、各local piece始点から同じboundとendpoint/stage情報でswept AABBを作る。
5. path tubeと拡張BVH nodeが分離していればno-hitとする。leaf候補ではfacet支持線に対するtubeの法線方向intervalを
   外向き丸めで評価し、facet別event budgetを含めても厳密に分離するfacetだけを候補から除外する。
6. candidateがある場合だけmidpoint/path deviationを評価する。
7. tubeとfacetの関係が曖昧なら、元のmacro proposalのparameter部分区間を流用せず、時間順のdyadicな
   sequential RK4 pieceとしてphysics評価ごと二分する。各no-hit leaf endpointが次pieceの始点になる。
8. earliest bracketを縮め、同じintegratorでhit時刻まで再評価する。
9. 時間幅、空間tube、candidate facet集合の安定を満たせなければ`indeterminate_geometry`にする。

同じ始点から短縮時間だけRK4を再実行したendpoint曲線は、急峻な位置依存fieldの下で任意のparameter部分区間を
局所時間幅に比例して包絡できない。このためrevision 3bのaccepted pathは上記sequential piece列と定義する。
piece分割はgeometry、support、applicabilityだけで決まり、出力scheduleに依存させない。global extrema boundを
最初の正しさ基準に使った。M3-C1の強い局在場では、全初期sampleが適用域内でも遠方cellのextremaにより時刻0で
証明不能となる反例が実測されたため、P19-Lでlocal cell primitive boundとdense path上のcertificate-only restrictionを
追加する。これは固定step endpointを作り直すhidden substepではなく、同じproposalを認証するためだけの局所化である。

revision 3aは材料boundary、P1/Q1の非凸support、RZ、continuous chargeを解禁しない。特に一般曲線の
材料boundaryでは先行hitが後続のsupport/applicability逸脱を救済し得るため、proposal全体をevent探索前に
拒否してはならない。stage・endpointの非有限値は即時失敗とするが、support/applicabilityはprovisional flagとし、
`first event / valid prefix → no-hitまたは残存prefixのvalidity → commit`の順で確定する。この
event-before-validity意味論はrevision 3bで、accepted piece列と一緒に実装した。

### 9.4 macro timeと保存時刻

production time boundaryは次だけで決まる。

- base macro `dt`
- field time knot/discontinuity（Stage 4A）
- pulse等のglobal source discontinuity
- 物理boundary event

**出力時刻はstepを分割しない。** accepted path中で時刻を含むpieceの`state_at()`を評価してframeへ保存するだけで、
次の状態更新には使わない。trajectoryの保存間隔や対象粒子を変えてもfinal state、event、piece partition、RNG pathが変わらない
ことを受入試験にする。

個別粒子のrelease時刻も全粒子のglobal step boundaryにしない。macro interval内にreleaseされる粒子を
`(release_time, remaining_time)` work itemとして追加し、その粒子だけ正確な残時間を進める。これにより
10^6個の異なるrelease時刻でglobal loopを分断しない。release前の粒子はframeへ行を持たず、保存時刻と
release時刻が一致する場合はrealized sourceの初期状態を保存する。release timeは任意のfinite SI時刻を許し、
`[time.start_s,time.end_s]`内であることだけを要求する。

---

## 10. source、boundary、RNG

### 10.1 source

production sourceは二つだけ実装する。

- `table`：粒子ごとの位置、速度、release time、粒子属性をcanonical HDF5へ明示
- `surface`：粒子ごとのfacet ID、strict interiorのfacet内座標、速度、release time、粒子属性を明示

単一点は一行のtableとして表し、専用point-source classを作らない。単一
`ParticleSchedule/realize_sources`が両表を統合し、surface位置だけをprepared geometryから導出する。
分布sampling、粒子ID採番、熱速度生成は外部入力作成側で完了させ、coreにsource RNGやdistribution catalogを置かない。

### 10.2 boundary law

- `stick`：terminal `stuck`
- `escape`：terminal `escaped`
- `hold`：hit payloadを保持する非deposition terminal `held`
- `specular`：parameterなしの完全鏡面反射
- `restitution`：法線・接線反発係数をともに明示
- `maxwell_thermal`：正の壁温度、`[0,1]`のMaxwell拡散混合率、静的facetへ接線方向のwall velocityを
  必須とし、壁frame鏡面／完全熱適応half-range Maxwell fluxを混合
- `probabilistic_stick`：一回だけ乱数を引き、非付着側の
  `specular | restitution | maxwell_thermal`を完全なparameterとともに明示

lawは既に局在した`BoundaryEvent`を受け、位置を動かさず速度・lifecycle・event recordだけを決める。
corner候補集合に吸収lawが含まれる場合と複数反射面の場合のpolicyをcase formatで一つに固定し、iteration順へ
依存させない。

### 10.3 RNG

`rng.py`はNumbaへ移しても同じcounter意味論を維持できるPhilox4x32-10を一つ実装し、既知vectorで検証する。

- wall：`seed, particle_id, physical_boundary_event_ordinal, law_stream`
- Brownian：`seed, particle_id, macro_interval, root_stochastic_interval, tree_level, tree_index, component, stream`

thread、tile、保存schedule、event局在の数値反復回数をkeyに含めない。数値refinementはphysical event
ordinalを進めない。

現行wall RNG revisionは`philox4x32_10_v2`である。probabilistic stick、Maxwell mixture、Maxwell法線
open-uniform、Maxwell接線standard-normalを四つの明示streamへ分ける。`wall_uniform_batch`はstream引数を
必須とし、callerが物理domainを選ぶ。`boundaries.half_range_maxwell_flux_velocity`とそのNumba batch wrapperは
RNG streamを選ばず、drawから速度への変換だけを所有する。

### 10.4 boundary後続revisionの分離

P07-Tは瞬時`maxwell_thermal`反射、boundary v6、wall RNG v2までを完了した。
P07-T初回のsurface sourceはcanonical realized schedule v4へ置換し、熱速度が必要なら外部入力作成側が
`half_range_maxwell_flux_velocity_batch`等で明示velocity rowを作る。Brownianのzero/tangent速度を
position nudgeで通さず、prepareで明示拒否する。

finite-radius contactは独立`contact_radius_m`、静的2-D XY/RZのsegment capsule first contact、
surface center offset、table clearance、candidate固有法線を`case_format.py`、`sources.py`、`geometry.py`、
`events.py`、`engine.py`の既存責務へ統合した。linear/quadratic exact、RK4、exponential、Brownianは第二engineを
持たず同じevent経路を使う。0はpoint semanticsをbitwiseに維持する。
現行H2では`boundaries[].contact_geometry`の未指定を`particle_surface`とし、明示`particle_center`だけ
候補の有効半径とsource offsetを0にする。物理半径は保持し、一つのfacet surface maskを既存owner間で共有する。
source v5、geometry v7、event v22、engine v46、memory plan v16で、端capの証明済み離脱と反射originのmacro保持を含む。
時刻順が未解決の混在mode同時候補はfail-closedとし、第二mesh/BVH/lawを追加しない。

次の機能は責務が異なるため同じpatchへstubやplaceholderを置かない。

1. surface chargingはwall material/electrical state、粒子charge transfer law、保存event payload、checkpoint/resume、
   電荷保存または外部回路authorityを`boundaries.py`/physics/outputで定義する。static law parameterだけで済ませない。
2. resuspension/desorptionはwall-resident lifecycle、dwell clock、adhesion、接線stress、release ordinal、再開schedulerを
   source/lifecycle ownerで定義する。terminal `stuck`/`held` rowを暗黙に再活性化しない。
3. rolling/sliding、粒子間衝突、moving geometry、3-D surface offsetは現行の瞬時2-D接触へ押し込まない。

---

## 11. engineの唯一の実行loop

```text
prepare once
  ├─ resolve geometry / fields / physics / laws / resources
  ├─ allocate fixed SoA and bounded buffers
  └─ open ResultWriter in OUT.partial

for each macro interval [t0, t1]:
  1. interval内のreleaseをflat SoA work queueへ追加
  2. stable active IDsをbounded tile slabへ分ける
  3. integrator.proposeでStepProposalを作る
  4. events.first_hitでearliest eventを局在
  5. eventまで同じintegratorで再評価
  6. boundary lawを適用
  7. activeならrow target timeを保ったまま次のevent roundへ戻す
  8. accepted pathからrequested frameをemit
  9. event/countを固定columnar rowへemitしstable prefixする
 10. accepted macro-step epoch barrierでwriter batchと必要なepoch commit

finalize
  ├─ final particle table
  ├─ run summary
  └─ _SUCCESS and directory rename
```

`macro_epoch`は、区間内の全particleについてevent残時間処理まで完了し、proposalが受理・stateへcommitされた
後のbarrierでだけ一つ進む。trajectory frameの要求時刻、segment flush、writer呼出しの都合ではepochを進めず、
solver stepも分割しない。

`max_interactions_per_step`へ達した時は、残区間に次のhitが存在する場合だけ、そのresidual-work intervalの
残時間を二分して再試行する。event-freeな残区間は追加depthを消費せず受理する。子intervalのinteraction
countは0へ戻し、refinement depthは引き継ぐ。明示depth budgetを超えた場合は
`failed:numerical_event_budget`であり、stickに変えない。residual-work bufferとevent output bufferを
共有しない。

---

## 12. CPU高速化と並列化

### 12.1 現行の数値実装

同じSoA、model code、proposal/event意味論を一つのengineで使う。
既存NumPy/Numba数値ownerを維持し、scalar evaluatorはverification oracleへ限定する。
追加JIT化は停止した。Numba内部parallel、ThreadPool、第二backendを作らない。

### 12.2 現行execution ownership

- 親macro coordinatorと明示workers数のspawn processが同じslab数値処理を使う。
- prepared field/geometry/model factsはread-only、state/hint/replayの更新rowは排他的とする。
- stage/proposal/event/Brownian workはworker専用slabで再利用し、共有transport/handoffを総予算へ含める。
- target time、depth、interaction/event ordinal、statusを持つflat数値work列で残時間を処理する。
- 一roundでrowごと高々一event/failureをbounded slotへ書き、prefix/compactionで出力と次roundを作る。
- 親は全connection/sentinelを監視して完了workerへ次taskを渡す。
- event/failureはworkerごと一batch in-flight、single writer消費後にACKする。
- 公開event identity/orderは既存output readerが所有し、到着順を科学順のauthorityにしない。
- 全taskとoutput消費を1 macroごとに待つ。RK stage/OU leafはworker内処理である。
- workerはfileへ書かず、readyとshutdown/cleanupは親が所有する。

fastmathを既定無効にし、force加算、prefix/compaction、canonical orderを固定する。
同じbinary/hardwareでworker/slab/output schedule、checkpoint-resumeを変えて全公開科学payloadとRNGを
変えない。worker数はAPI/CLIの実行資源で、数値caseやdraw keyへ混ぜない。

### 12.3 次の性能変更の条件

具体的利用caseを先に閉じ、同一input/model/dt/seed/outputでprofile、科学identity、時間、memoryを比較する。
kernel単体速度、allocation数、profile self timeをend-to-end改善へ読み替えない。
cold/warmを分け、startup/import/JIT/IPC/writer/shutdownを含める。
粒子数だけで損益分岐を決めず、各macroの仕事量・活動粒子変化・event/出力を確認する。
同期Kの変更は第21.3節の将来項目であり、追加JIT/native teamを先行条件にしない。

## 13. ResultStore、checkpoint、後処理

result schema 3、checkpoint schema 2の永続契約は[result形式](solver/docs/result_format_v3.md)が所有する。
同期single writerがclosed segment、交互A/B checkpoint、`LATEST`、final publicationを扱う。
strict input/schema/model/algorithm identityが一致するpartialだけをresumeする。

commit順はsegment close/rename → inactive checkpoint replace → `LATEST` replace。
`LATEST`より新しいorphanをresumeで無視し、最初のcommit前は初期状態から再実行する。
最終化はfinal → complete manifest → `_SUCCESS` → directory rename。
最終公開途中も次runが検査して完了させる。migration、第二writer、複数永続形式を作らない。

engineの既存work-scaled cadenceで、全task/outputを消費したaccepted macro barrierに限りcommitする。
output scheduleやslab幅でcadenceを変えない。worker event/failureは一batch ACKでbackpressureし、
frame/probeはboundedに読み書きする。全粒子×全時刻のsnapshot listを作らない。

参照checkpoint/最新segmentのhashと全segment構造/countを検査する。過去segmentの同shape値改変、
power loss、remote filesystem、同じOUTへの複数runは保証外である。

analysis/可視化は`ResultView`またはcanonical artifactから派生し、source runへ追記しない。
最小T03はfate count/weight、boundary/material別deposition、outcome別arrival、保存軌道/event描画を扱う。
arrivalはsimulation絶対時刻で、飛行時間・residence・impact指標・汎用exportは必要な利用caseごとの追加とする。
常時全粒子のstage/force traceを診断のために保存しない。

## 14. microcaseと受入試験

test directoryは`verification / scenarios / performance`の三層だけにする。private helper、関数呼出し順、
file行数、directory内部をtestしない。

### 14.1 基本microcase

| ID | 内容 | 主に固定するもの |
|---|---|---|
| C01 | force無しballistic | `x=x0+v0t`、frame補間 |
| C02 | uniform linear drag | exponential velocity/position、RK4収束 |
| C03 | linear drag＋constant acceleration | exponential midpoint式 |
| C04 | uniform electric field＋fixed charge | charge sign、mass authority |
| C05 | gravity＋buoyancy | displaced volumeとmassの分離 |
| C06 | two-cell affine P1 / mapped Q1 field | locate、basis、shared-edge連続性、vector component |
| C07 | plane wallへのfirst hit | event time/point/normal |
| C08 | wall上release | zero-time departureと再衝突 |
| C09 | thin gapと複数反射 | residual time、boundary ordinal、interaction budget failure |
| C10 | corner同時hit | candidate facet set、combined normal、priority、曖昧policy failure |

charge relaxation、OU統計と独立first-passage参照も現行verificationが所有する。
未実装3Dは独立利用caseとreferenceを用意して追加する。microcase IDをCOMSOL case名にしない。

### 14.2 method別最低条件

| 対象 | 受入条件 |
|---|---|
| RK4 | smooth coupled problemでglobal order 3.5以上 |
| exponential midpoint | 一定係数は丸め誤差、可変係数でorder 1.8以上 |
| P1/Q1 | analytic fieldの再現、cell境界で連続性とsupportが一致 |
| first hit | 固定meshのstep半減でtime/point/facet ID、mesh系列ではtime/point/normal/boundary groupが収束 |
| output independence | output selection/schedule変更でfinal/event/RNG不変 |
| runtime identity | slab幅、output selection/schedule、checkpoint-resume変更で粒子別final/event/RNG/state一致 |
| checkpoint | 任意commit境界で停止・resumeして重複/欠落無し |
| continuous charge | rate・平衡残差、有限invariant/bound、`hL <= 0.5`受理と超過時の明示拒否、coupled trajectory収束 |
| axisymmetric 3D | 方位回転共変、円筒/円板/円錐hit一致 |
| OU | mean/covariance/MSD/平面first-passageが統計区間内 |

収束次数の閾値は丸め・eventの影響を避けたmanufactured problemにだけ用いる。製品caseへ一つの絶対座標差を
機械適用しない。

### 14.3 scenario test

scenarioは公開APIを通る少数のend-to-end caseだけにする。

1. table release → ballistic → stick → result read
2. surface release → drag/electric/gravity → mixed boundary laws
3. event-heavy specular gap → terminalまたはend time
4. incomplete output → recovery read → resume → final
5. RZ axis crossing → wall eventと混同しない

COMSOL benchmarkはcore scenario suiteへ入れず、`tools/vv/comsol`の外部workflowとする。

---

## 15. performance計画

測定policy、source/input/lock/revision、cold/warm各raw値、科学readerの範囲、memory/cleanupを保存する。
10^4/10^5/10^6の重いmatrixはrelease/manual用途で実施し、毎PRへ重複させない。
warm中央値と観測範囲、cold、総process CPU、sampled private/WSを区別する。
新sourceの数値/性能資格を古いsourceから継承しない。

0.3.0の主用途・品質は性能文書第9.27節、Case A/P実条件full-stepは第9.28節、
P14-U 1万粒子320stepは第9.29節と各raw evidenceが所有する。
旧thread benchmark表、試作の時刻やRSS、不採用の経緯を本書へ重複記載しない。
COMSOL人口gateからnative FE、全RHS、全boundary、後続velocity/chargeの同等性を推定しない。

## 16. 完了済みwork package

| group | 完成した役割 |
|---|---|
| P00～P08 | 独立project、canonical、座標/場、geometry/event、source/RNG、coupled RK4、public result/CLI |
| P09～P14 / P14-P/U/R | resident SoA/slab、既存compiled CPU、指数法、durable I/O、代表用途・配布資格、旧thread経路撤去 |
| P15～P16 / P18 / P19-L / P21～P22 | 対応charge/drag/ion drag/熱泳動/DEP/lift、局所certificate、hold |
| B01～B05 | joint OU、XY/RZ production、wall restart、conditional boundary refinement |
| finite contact / periodic / time field | 独立接触半径・group接触意味、XY translation topology、固定topology linear time |
| F01/F02 / T01～T04 | canonical builder、provider adapter、reduced electrostatics、外部V&V、最小analysis/可視化、cache producer |
| 0.3 process/runtime | 共有transport、完了順slab配分、一batch ACK、single writer、異常終了/worker変更resume |

これは実装責務の索引であり、全model・組合せの一般的な保証表ではない。
旧native/JIT試作の過程、旧versionの不合格、source固定receiptは
[性能文書](solver/docs/parallel_execution_plan.md)と`evidence/`へ残す。
完了済み手順を現行の着手条件・残件へ重複転記しない。

## 17. 現在のbacklogと着手条件

| 項目 | 着手条件 | 完了条件 |
|---|---|---|
| 汎用source・外部場・物理/数値qualification | 第22節のG1〜G4の入力・独立referenceがある | realized source/場転送/物理/積分を別判定し、既存ownerへ統合 |
| 多macro連続実行 | 小N・多step用途で通信/同期費が支配するとprofileで確認 | 第21.3節の不変条件、科学一致、同一条件の時間/memory改善 |
| 別出力量・時間場・CPU構成 | 対象case、hardware、accuracy/SLAを指定 | 観測前policyとsource binding、科学/速度/memoryを別判定 |
| NUMA/SMT配置 | 対応hostのtopologyと利用caseがある | actual affinity/page配置と費用を測り、共有memoryだけで保証しない |
| P17/RZ場・Cartesian 3D粒子 | 第22.7節と既存P17 decisionの独立referenceを用意 | RZ場・回転面を使うXYZ state、normal/corner、収束・memory・性能。一般3D meshは別利用case |
| surface charging/再飛散 | wall state・保存則・lifecycle・release authorityを定義 | event/output/checkpointを同じownerへ統合、terminal rowを暗黙再活性化しない |
| GPU/別言語/out-of-core | 現行profileと具体的用途が必要性を示す | 一つのproduction経路、意味論parity、転送込み費用、旧経路削除 |

追加JIT/native teamの再開、全CPU数比例の速度、未指定の物理拡張を0.3.0の必須工程へ戻さない。
未実装のAPI、flag、空interfaceを先行追加しない。

## 18. 変更workflowと品質

1. 利用case、数値的期待値、所有module、置換・削除対象を閉じる。
2. 解析解、manufactured、または独立した小referenceを定める。
3. 同じproduction engineへ必要な最小変更を行う。
4. 原因を表すverificationと小型公開scenarioを通す。
5. hot pathはprofileと同一条件before/after時間・memoryを測る。
6. 旧code、設定、fixture、test、文書を同じ変更で削除する。
7. 所有文書と必要な一つのdecision recordを更新する。

uv、Ruff、Pyrefly、import-linter、Radonの標準CLIを直接使う。変更functionは原則CC≤10、
数値的に一体が明確でも≤15。blanket ignore、baseline、薄いhelperでscoreだけを改善しない。
標準quality gateは[品質tool文書](quality_tooling_plan.md)を唯一のcommand ownerとする。

## 19. revisionと証跡

数式、補間、event順、tolerance、RNG key、離散pathを変更する場合は科学revisionを更新する。
dispatchや容量所有だけの変更もruntime/memory revisionの必要性をreviewし、refactor名で科学変更を隠さない。
case/result/checkpointの破壊的変更を暗黙migrationで吸収しない。

保存archiveのbytes、inventory、historical registrationは変更しない。現行文書のcleanupは現行仕様を整理する作業で、
旧FAILをPASSへ読み替えたり、過去観測へ新sourceを付け替えたりしない。

## 20. 完了条件とGit提供

変更の完了には、一意なowner、数値reference、三API/single engine、lock整合、全静的gate、該当数値/scenario、
必要なidentity/性能/memory検証、旧経路削除、科学revision整合が必要である。
lint/type/coverageだけで数値実装を完成扱いにしない。

0.3.0は検証済み現行sourceと文書をrelease branchへcommitし、pushする工程として進める。
source commit、version、検証scope、wheel/installed identityを結合する。
Git branch push、remote CI、tag、Release asset公開、package registry公開は個別の実操作であり、
実行した操作のreceiptだけを公開記録とする。local Linux/WSL資格をremote CIの成功へ読み替えない。

公開済み0.1.0/0.2.0のtag・資格・receiptは固定した履歴として維持する。
[0.2.0公開後receipt](reviews/v0_2_release_publication_2026-10-10.json)を新sourceの資格へ付け替えない。

## 21. 0.3.0と将来の同期改善

### 21.1 採用した方針

既存数値ownerを用いたprocess並列を、一つのengineへ統合した。
追加JIT化、native chain→yield/slots→native teamという旧計画は現在の工程から外す。
既存Numba bodyを使い、二実装・backend切替・自動tunerを増やさない。

### 21.2 現行の同期境界

1 macroごとの全task完了待ちが固定仕様である。event/failure ACK、ready、shutdownは別の待機で、
RK stage/OU leafごとの全worker同期ではない。同期間隔を変える設定は0.3.0へ追加しない。

### 21.3 将来項目：複数macroの連続実行

将来は数値`dt`とは独立した「最大連続実行macro数」を検討する。既定1を維持し、
具体的caseで1/4/8/16などを比較して採否を決める。以下を先に設計する。

- dt、field knot、個別release時刻、各stageのfield/physics評価、same-method hit prefixを維持する。
- 粒子ID・物理ordinal・macro/root時計・conditional Brownian drawを維持し、KをRNG keyへ入れない。
- 毎macroの人口系列と要求frame/probeを同じaccepted pathから得る。全Kstep×全Nの出力bufferを作らない。
- 終端後の更新停止、時刻順のboundary/failure payload、bounded ACKとsingle writerを維持する。
- checkpointは全workerが同じaccepted時刻まで完了しoutputを消費した状態だけを公開する。
  途中child死で未完blockを公開せず、resumeが1-step基準と科学一致することを確認する。
- 独立粒子・外部場という適用条件を明示し、粒子間相互作用や場feedbackでは必要な連成周期を越えない。
- 科学一致、時間、総process CPU、owned array、sampled private memoryを同じ入力で測る。

これは将来の設計候補で、公開API、case設定、実装済み能力の約束ではない。

### 21.4 v62のprocess実装・資格と0.3.0提供範囲

現行production27-file mapは`49906998c297d23c9be8578fb8c5a757168fa8b7c2dddecc0695290c5df263a2`。
主八用途41role/164結果、別scopeの時間場5role/20結果、Windows/Linux local gateと
runtime-only installed wheelの実workers=2 API/console/moduleは完了した。
C3測定とC4最終品質のproduction sourceは同一で、外部fixture literalの差・negative・rawログを区別して保存した。
詳細な品質、用途gate、memory、source別bindingは性能文書第9.27節が所有する。

一つのengineで、親がmacro時計・release・active compact・replay確定・single writer・checkpointを所有する。
run内で永続するspawn workerは排他的global rowを同じslab数値関数で進め、private scratchを独占する。
全connection/sentinel監視による完了順再配分、一batch ACK、ready後template解放、sealed transport、
単一shared replay root/永続reader、serialization reserve/page paddingを既存engine/cpuへ統合した。
全task/outputを消費したmacro barrierだけでcommitし、writerをworkerへ複製しない。
実child突然死のcause回帰は未完macro非公開・durable resume・worker/SHM cleanupとW変更後の全科学一致を確認した。

追加の実ケース確認では、Case A/P各32 seedの287粒子・30 ms・1500stepを完了した。
保存COMSOLとの登録人口分布gateは両方PASS、1/2/4processの計測各3 seedでは全科学出力が一致した。
この条件では並列が遅くworkers=1を推奨する。新native solve、全force/charge RHS、詳細boundary、後続v/Z分布を
同じ合格範囲へ追加していない。[固定した比較証跡](solver/evidence/comsol_brownian_fullstep_parallel_2026_10_12/README.md)を参照する。

P14-U 1万粒子・0～4 s・dt=0.0125 s・320step・Brownianなし・軌道/probeなしでは、
warm中央値W1/W4は1.964倍、全8runの科学出力が一致した。
この測定をBrownian/COMSOL/全CPU構成へ外挿しない。
[測定証跡](solver/evidence/p14_10000_320step_process_2026_10_13/README.md)がraw値とmemory条件を所有する。

0.3.0は検証済み用途に限定したprocess実行と既存数値機能を提供する。
小N・event短case・時間場補足の遅化、cold費用、parallel memory増加、20worker非比例を制約として残す。
今回のGit提供操作と公開状況はrelease記録が所有し、未実行tag/registry公開を製品機能の完成と混同しない。

## 22. 汎用的な改良計画：初期条件・場・物理・精度（2026-10-14）

### 22.1 目的と現時点の判断

本節は改良計画と実装範囲を所有する。目的は物理・数値的に正しい粒子追跡である。
COMSOLの設定された数学問題の再現は、入力と方程式を照合する検証の一工程とする。
COMSOL保存結果への一致を製品の物理的正しさ、実験検証、あるいはmodel選択のauthorityにしない。
2026-10-14の実装では外部G1/G2/G4を置換し、G3の独立衝突referenceを追加した。
production physics、integrator、schema、engine v64は変更していない。
既存の連成RK4を選択した新caseで再計算し、数値設定の効果をmodel差から分離する。
実装・実行結果は
[再評価campaign](solver/evidence/comsol_generic_redesign_2026_10_14/scientific_run_plan.json)と
[衝突qualification](solver/evidence/generic_improvement_2026_10_14/gas_collision/scientific_conclusion.json)に束ねる。
G3のnonlinear noise/熱状態、G5の物理3D、G6の新性能施策は未実装である。

| 調査結果 | 汎用的な判断・改良owner |
|---|---|
| A/P各287粒子の初期位置・速度ベクトルはcanonical入力と全行一致 | 今回の主要差を初期条件変換ミスとは判断しない。将来のsource意味解釈と生成はG1で外部producerを整える |
| 元のlinear dragとcandidate finite-speed dragの初期力の比は約1.018〜4.922。candidateは元のBrownian/Waldmannを含まない | 設定値だけで同じ方程式にならない。物理選択・適用域・揺らぎとの整合性をG3で扱う。元式再現を優先してlinear適用gateを緩めない |
| t0全粒子の同一状態でnativeの7決定論的外力とcharge式を再構成できたが、native FE→P1で力/chargeの差が残る | native式の読取りと場転送を分けられる。G2はsolution・演算子・unit・sideを束ね、off-nodeで検証する。全歴史stageの再構成を主張しない |
| 初めの10 µsで原結果/candidateの位置RMS差はA/Pとも約127 µm、candidate細分化差より大きい | 壁後の最終値だけでなく接触前の状態/RHSを使う。時間stepの模倣や際限ない細分化は原因識別にならない |
| candidateの3刻みでfinal fate・event列のidentity/件数は一致するが、event時刻/状態には差があり、PのPID11/30に最大約174/151 µmの接触前細分化感度が残る | fateの安定と軌道精度は別。G4で位置・速度・charge・eventごとの誤差予算を判断する。これは追加core defectの証明ではない |
| 不適切な局所field enclosureは既に修正・独立回帰検証済み | 修正済み事実を未解決課題へ戻さない。G4の試験は原因となる非一様場とsupport証明を維持する |

数値値のauthorityは保存済み
[軌道評価](solver/evidence/comsol_speed10000_comparison_2026_10_13/original_comparison/particle_trajectory_review/trajectory_review_report.html)、
[同一状態の方程式監査](solver/evidence/comsol_speed10000_comparison_2026_10_13/original_comparison/equation_audit/equation_audit_summary.json)、
[今回の初期条件監査](analysis/generic_improvement_2026_10_14/initial_condition_audit.json)である。
自己収束、原COMSOLとの方程式同等性、物理妥当性、製造予測資格の判定を混ぜない。
現行modelの0.1/1等のdrift gateはそのrevisionの受理policyであり、普遍的な物理境界ではない。

### 22.2 実装単位・依存関係

| package | 具体的変更とowner | 着手/完了条件 | 置換・削除対象 |
|---|---|---|---|
| G1 joint sourceの抽出・生成 | 既存`tools/comsol_adapter`がnative active設定と実現値を取り込む。外部`tools/source_builder`が物理的なsourceを実現し、既存canonical writerへ渡す。`sources.py`はscheduleだけ維持 | native/canonical全行照合、moment/measure検証、相関・ID・release・weight固定。未対応DOF/式は明示 | 有効性を見ないproperty読み、caseごとの速度/単位変換、runtime sampling。source builderは必要な実用modeだけ実装 |
| G2 native場とgeometryの転送 | 既存adapterのthermal固定inventoryを明示field mappingへ置換。`fields.py`はSI場sample/support、geometryはcontainment/first hitのownerを維持 | 同一状態のprimitive→force/rate、mesh/時間/geometry精度を独立に確認。高次/不連続拡張は条件付き | 同じ変換責務を持つcase-bound preparation helper、固定tag/287行/5場仮定、根拠のないaxis修復 |
| G3 gas interactionの物理整合 | `physics/forces.py`等のpure物理ownerと既存catalog/runtime/compiledを一経路で更新。平均抵抗、速度拡散、熱泳動の仮定を閉じる | collision oracle・漸近・保存則・近似誤差のqualificationを先行。新modelは第22.5節のstop条件を満たす | finite抵抗へのlinear FDT代入、適用外Waldmannの流用、COMSOL専用force分岐を作らない |
| G4 精度qualificationと必要時の積分改良 | `integrators.py`/`stochastic.py`/`events.py`の既存owner。比較器はcore外で既存ResultViewを使う | deterministic、spatial、event、SDEの誤差を分離。accuracy controllerは必要性と費用を実測してから | failureをaccuracyへ読み替える評価、chargeだけの別積分、独立乱数によるsplit、常時stage trace |
| G5 P17/RZ場・XYZ粒子 | 既存P17を一つのengine/schema/output/event/memory変更として実装。G3物理noiseの3D資格に連動 | 3D回転共変、円筒/円錐/平面hit、方位運動、source面積、RNG/resumeを検証 | 幅3配列だけの部分追加、2D結果へのphi sidecar、非零第三成分drop。一般3D meshは必須化しない |
| G6 精度を固定した性能 | 既存process所有境界と第21.3節。外部producer/exportはfield・sourceをhash再利用 | 同じ科学条件でend-to-end時間・memory・identityを測る。多macro化はIPC支配を確認した時だけ | 追加JIT、第二engine、model別scheduler、未計測の自動tunerを作らない |

G1/G2とG3の理論qualificationは並行できる。G4の既存手法の独立検証は先行できるが、
新しいmultiplicative-noise積分はG3の式・適用域が閉じてから実装する。
物理3D thermal/noiseの軌道認定はG5のstate/event完成が必要である。
G6はaccuracy/model/inputを固定した後に進める。日数を先に約束せず、各packageの完了条件で区切る。

#### 22.2.1 実装済み範囲

| package | 2026-10-14の成果と限界 |
|---|---|
| G1 | `tools/source_builder`にrealized CSV、vector、speed/direction、非相対論momentum/energy、明示static基底変換を実装。canonical writerとpublic load_caseを使う一経路で、入力bytesのparse/hashは同じsnapshot。任意式・分布sampler・source RNG・非零第三成分・dynamic frameは未対応 |
| G2 | `tools/comsol_adapter`のthermal固定v2 dialectをselected static nodal XY/RZ v3へ置換しF02設定/testを移行。domain witnessとprimitive有限性を別検査。時間snapshot、cell/不連続場、multi-domain union、高次native FE再現は未対応 |
| G3 | `tools/vv/gas_collision`に独立collision quadrature/jump/frozen参考過程を実装。平均法則を明示仮定内で確認。nonlinear SDE、粒子熱状態、high-drift熱泳動、physical3D軌道は未認定・未追加 |
| G4 | 既存`evaluate_matched_trajectory`の固定ID/時計/電荷仮定を置換。実在3刻みの自己感度と2保存参加者のraw比較を分離。既存連成RK4でA/P再計算。図はcanonical geometry、joint precontact、欠測/terminal意味を保持。精度controllerや第二engineは追加しない |
| G5/G6 | 本節と既存P17/第21.3節の将来項目。今回の成果に含めない |

外部toolの変換・評価検証とproduction engineの方程式qualificationを区別する。
初期条件・場payloadを従来入力と全行照合した後に既存integratorだけを変更したため、
今回の軌道数値感度にsource/fieldの再生成差を混ぜない。

### 22.3 G1：今回の初期条件と取り込みの実態

今回のactive featureはA=`comp1/fptas/relg1`、P=`comp1/fpt/relg1`のReleaseGridである。
AllCombinationsにより41個のr座標×7個のz座標、全粒子t=0に放出される。
raw位置の14〜18、2.3〜2.6はgeometryのcmであり、canonicalではr=0.14〜0.18 m、
z=0.023〜0.026 mである。ID順をgrid列順から推測せず、native実現IDを使って確認した。

有効な初速度はSpecifyVelocity/Expressionで、成分式はnativeの`[r,phi,z]`として保持される。
phi式は0でありout-of-plane DOFは使わない。実際のretained velocityは次である。

```text
theta(r,z) = (15 deg)*sin(2*pi*(0.61803398875*(r-0.14 m)/(0.001 m)
                              +0.41421356237*(z-0.023 m)/(0.0005 m)))
s(r,z)     = 10000 * release_speed_multiplier * |u_g(r,z)|
v_r        = s*sin(theta)
v_z        = release_axial_sign*s*cos(theta)
```

ここではmultiplier=1、axial_sign=1であり、10000は10 nmに対する100×100の条件式から生じる。
AはAS_release_speed→AS_ugr/z→withsol(sol2, RF=100 V)、Pはrelease_speed_P→u/wを使う。
背景primitiveの実際のsolution/memberとsource評価の依存先を別々にbindする。
ガス速度の大きさを使用しても、速度方向をガス流方向に合わせたりガス速度を加算したりしていない。
角度は位置に相関した決定論的値であり、uniform coneやMaxwellianではない。
保存初速は両caseとも約90.16〜2478.19 m/s、thetaは約±14.9998°であった。

stored `T0=293.15 K`、`Ep0=1 keV`、`alphac=pi/3`、`M=50`、`mp0=1e-9 kg`、`dp0=1 µm`は
このactive選択の初速度/粒子質量/径ではない。慣性質量はdensity/diameter選択の
1.1519173063162575e-21 kg、drag径10 nm、electrostatic半径5 nm、point contact、Z0=-1、weight=1である。
SubtractMovingFrameVelocity=0だけから慣性系を断定しない。今回のinventoryにはrotating-frame featureがない。
保存vector値の一致とlaboratory-frame意味の認定は別であり、datasetのframetypeも運動frameの証明ではない。
point contact・weight/material等のcandidate解釈と直接保存された状態を区別する。

gas抵抗で用いる分子と粒子表面のdiffuse/accommodation fraction（今回candidateは0.9）と、
粒子がチャンバー壁で付着する確率（今回material wallは0.5）は別の量である。
後者の残り0.5は鏡面反射であり、gas collision modelのdiffuse fractionへ転記しない。

監査では保存t0とcanonicalの位置・速度の最大成分差はいずれも0、別のprimitive probeを使う
式の再計算の最大成分差は5.573e-11 m/s以下であった。export座標/演算roundoffを含む。
既存exportを再使用した算術検証であり、独立した二回目のnative抽出や統計的source妥当性の検証ではない。

### 22.4 G1：汎用source producerの具体的な仕様

coreへ任意expression、分布名、COMSOL propertyや新source RNGを足さない。
現在のcanonical `position_m[N,2]`、`velocity_m_s[N,2]`、release、mass、各径/半径、volume、Z、weightで
2D realized sourceを表現できる。G1だけではcore schema変更は不要である。
surface sourceの位置はfacetと厳密な内部parameterを使い、現在のcontact geometry ownerが中心を解決する。
速度は常にcanonical基底へ変換済みとする。固定nudgeを導入しない。

外部producerは、まず実用modeとして(1)実現tableの取り込み、(2)明示vectorまたはspeed+direction、
(3)非相対論momentum/energyの変換を実装する。native complex expressionはproducer側で評価・exportする。
任意式DSLは作らない。thermal/角度/物理flux modeは利用caseと以下のreferenceがあるものだけ順次追加する。

| 与え方 | 正しい意味・変換 | 必要な検証 |
|---|---|---|
| vector | unit・成分順・基底・frameを指定して変換。速度の大きさだけでは不足 | 非軸方向vector、回転基底、符号、全行比較 |
| speed+direction | `v=s*l/|l|`、zero directionは拒否。場所・時刻・粒子径等との相関を保持 | norm/directionとconditional correlation |
| momentum / kinetic energy | 非相対論慣性modelで`v=p/m`、`s=sqrt(2E/m)`。EはJ、mはrowのmass_kg | energy/momentum、異なるmassのrow、eV→J |
| 流れを基準にする | `v=u_g`、`v=u_g+w`、`v=s*n`を別々に指定。release時刻/位置の場を使う | 一様/非一様/時間場の既知値、絶対/相対の区別 |
| volume thermal | 保持Cartesian各成分の分散`kBT/m`。speed PDFは次元に依存 | 平均・covariance・`E[|v-u|²]=d*kBT/m`、size/temperatureとの相関 |
| 球面/半球/cone | 3D等方はcos(theta)を一様、2D arcはsigned angleを一様。決定論quadratureと乱数sampleを別記 | solid-angle moment、cone支持、粒子数/multiplicity |
| Lambertian / surface thermal flux | angular cosine lawとspeed lawを別指定。物理的平衡fluxはnormal正向きRayleigh、tangent Gaussian。native thermal lawとは式を照合してから対応 | inward normal・right-handed基底、normal/energy moment、flux-weighting |
| RZ体積/表面 | 物理率なら`2*pi*r dr dz`/`2*pi*r ds`、meridional診断なら`dr dz`/`ds`。annulus一様体積ではr²を一様 | total area/volume、weighted数/率、半径moment |
| 放出時刻・weight | list、time quadrature、Poisson rateは別。計算粒子数と物理粒子数/質量/電流を区別 | realized release list、rate積分、曝露時間、weight合計 |
| 第三成分・moving frame | 非零phi/面外成分、旋回frameの運動は現行2DOFでは完全再現不能 | reduction明示またはunsupported。初値変換だけで動的frameを対応済みとしない |

sourceの成果物は一つのrealized tableと、既存`producer_metadata`に収める小さい生成記録とする。
入力hash、sampler/変換revision、対象measure、基底/frame、active expression依存、
release/position/velocity/propertyのjoint意味、seedとdraw identity、native実現との対応を記録する。
属性ごとに別validator/manifestを増やさない。生成後のtableを固定し、workers/tile/output/resumeで再sampleしない。
source RNGとBrownian/wall RNGは別streamである。

nativeとの再現はactual realized rowを全IDで比較する。独立にsampleしたsourceでは、
joint/conditional分布とweighted observableを比較し、同じIDの軌道差を再現誤差にしない。
初期thermal速度と運動中Brownian forcingは異なる仕組みであり、片方が他方を代替しない。
詳しい運用は準備skillの
[initial-conditions.md](../.agents/skills/prepare-comsol-field-case/references/initial-conditions.md)が所有する。

### 22.5 G2/G3：場転送と物理modelを別々に閉じる

G2では既存adapterのFieldSpecificationを、必要なfieldだけの有限listへ置換する。
各量にcanonical名/quantity/unit/components/basis、native expressionとqualified scope、
solution/member/time、domain/side、association、recovery/PPR/smoothing、sampling位置と変換根拠を持たせる。
hash済み`producer_metadata`と既存extraction recordを使い、coreへnative演算子を渡さない。
node IDがある時はIDを使い、座標joinは一意性と誤差を検査する。regularというfilenameから格子reshapeしない。
quad→triによる近似、axis parity変更、heat-flux再構成は変換記録と対象観測量への誤差を必要とする。

検証はnative→canonicalのnode確認に加え、cell内部・interface近傍・source・代表trajectory位置を含める。
primitiveのSI差と同一状態のforce/charge差を測り、mesh h/h2/h4、field snapshot、wall facetingを別々に調べる。
場とgeometryは別責務だが、現行productionのrequired場は一つのfully-supported nodal layoutを要求する。
unstructured P1/Q1ではparticle volume meshとの完全一致が必要である。regular場は独立gridを使える。
一般のcell-associated/discontinuous required場は現行対応済みではない。
P1/Q1凸補間のnodal extrema証明を高次overshootへ流用しない。
mesh refinementで物理budgetに届かない場合だけ、side-aware表現、高次/曲面、internal wallを個別提案する。
いずれもsupport、保守bound、stage/eventとgeometryを同時に閉じることが着手条件である。

G3は次の順で進める。式の根拠と未確定事項は
[technical_research.md 第12節](technical_research.md)が所有する。

1. finite-speed mean dragを、gas組成、温度、sphere/accommodation、mass/size conventionまで独立確認する。
   mixtureはspecies sumが根拠になる場合だけ採用し、平均分子質量一つで必ず等価とはしない。
2. 同じcollision kernelから速度drift Aとcovariance-rate Qを導出し、低速OUへの極限を検証する。
   Qの正定性だけでは高速度SDE近似を認定しない。jump/collision oracleで質量比・高次moment・tail/eventを調べる。
3. Qが速度依存なら既存frozen-linear OUとは別の明示SDE revisionが必要である。
   Itô/Stratonovich、drift補正、stage評価、conditional increments、event residualをG4と一緒に閉じる。
4. 高drift熱泳動はWaldmannの式へ上限変更を施して流用しない。gas分布/heat-flux closureを導出するか、
   当該範囲の熱泳動を未対応として、物理的不確実性と対象observable感度を残す。

不均一・非Maxwellian gas、混合/熱非平衡、collision memory、near-wall、非球形、rotation、
charge/ion collisionの影響は必要性が示されたmodelだけを追加する。係数fitや万能regime blendを先行しない。
原COMSOLのlinear数学問題をそのまま解く検証が必要なら、外部のmanufactured/reference testとして扱う。
productionへ追加する場合もCOMSOL固有名やgate bypassではなく独立した根拠と利用caseを必要とする。

### 22.6 G4：独立した精度判定と最小の原因識別

現行のdtは固定であり、support/event/applicabilityのsplitはsafety制御である。
それをtruncation-error adaptivityやaccuracy保証と呼ばない。exponential chargeは同じmidpointで
凍結する連成式であり、chargeの安定性と長時間の連成accuracyも別である。

最小検証は次とする。既存の解析/OU/charge/dense/corner/grazing/axis試験を再使用し、
不足する原因例だけを追加する。case/observationはCOMSOLなしで作る。

| 検証 | referenceと判定 |
|---|---|
| 非一様deterministic force/charge | 解析またはmanufactured trajectory、独立高精度reference、h/h2/h4。x/v/Zを別budgetで評価。P1の非滑らかさで形式次数を強制しない |
| 空間/時間field | polynomial既知場とrefined representation、off-node/各stage/field knotを検査。dtで転送biasを隠さない |
| 境界 | ballistic/加速度/曲線のplane/corner/grazing first hit、同じ積分器でhitまで再評価、反射energy/残時間。連続noiseの未sample first passageはexactとしない |
| frozen linear noise | OUの平均・covariance・速度相関・MSD、joint位置速度とconditional splitのreference。source thermalと独立に検査 |
| 新nonlinear noise | G3が閉じたcollision oracleのmoment/relaxation/energy/tailと弱収束。強収束比較では同じBrownian incrementをcoupleする。seed整数だけでは不足 |
| 再現性/費用 | 出力時刻/worker/tile/checkpoint変更で同じ科学payload。state/scratch/IPCを同じmemory budgetに計上 |

外部比較は(1)joint initial source、(2)同一状態のnative式、(3)primitive転送、(4)candidate pure/compiled、
(5)candidate数値細分化、(6)reference数値/確率、(7)event前後の順で最初の差を切り分ける。
完全な順番を毎回やり直さず、hashで有効な証拠を再使用する。
第一差のpacketは選択ID・時刻・状態・cell/support・primitive・force/rate・model/revision・event bracketだけに限定する。
input/code/algorithm hash、直前accepted checkpoint、例外原因、観測した最初の保存intervalもbindする。
現行probesはx/v/Z/lifecycleであり、歴史的なprimitive/force stage traceではない。
必要なprimitive/rateは別に識別したfrozen-state probeとし、取得不能な値を埋めない。
現行probes、checkpoint、例外context、ResultViewを使い、第二engine/常時stage保存/診断frameworkを作らない。
native Brownianの保存時再評価を歴史的RK stage drawと呼ばない。

PASS前にreference自身の細分化と物理目的からの許容誤差を登録する。最細二段階差は経験的感度であり、
漸近性を示さず誤差上界にしない。Monte Carloの不確実性、source分布、場/model biasは数値感度と別に記載する。

現行外部評価器`deterministic_pre_event_trajectory_v3`はcross budgetと独立した
candidate/referenceの`self_sensitivity_budgets`を消費する。未登録・超過・実partition証拠不足では
qualified parityを`NOT_TESTED`に保ち、raw差分は出す。public manifestの共通runspan、
増加するmacro counts、選択窓にcoarsest base stepが1つより多いことを保守的な必要条件とする。
この条件は全accepted particle stageの再構成でも絶対誤差保証でもない。
CSV-only native provenanceをpublic resultとして偽装せず、nativeの実行・解像度資格は別に根拠を得る。

軌道は全粒子のjoint pre-contact x/v/Z、first arrival、semantic surface、impact energy/angle、
event列、survival/weighted fateを評価する。最初のevent/乱数分岐後のID軌道差はその原因を併記し、
coverage/censoringを保持する。今回PのPID11/30は細分化pilot対象であり、新caseのdefault IDにはしない。
rare wafer hitは今回の少数例だけで精密な率を認定しない。

既存methodで予算を満たせないことと費用支配が実測された場合だけ、local accuracy controlを提案する。
比較する状態はcoupled x/v/Z、scaleは目的から決めたabsolute/relative budgetであり、
既存safety/event認証を維持する。導入時は一つのproposal/accepted-state/engineへ統合し、
RNG再draw・charge-only subcycle・output時刻による積分step変更を導入しない。

### 22.7 G5：物理3D追跡に必要なstateとsource

場が軸対称であることは、粒子の方位速度が常に0であることを意味しない。
非零phi初速や3D thermal/Brownianは、radial dynamicsと衝突方向を変える。
meridionalのnoise投影は制約付き2D modelであり、full3D Brownian輸送の認定に使わない。
sourceだけに第三速度を足す改良では解けない。

既存P17のRZ field/geometry→Cartesian XYZ粒子を進める。一般tet mesh、voxel複製や次元pluginを前提にしない。
最初はdeterministic・fixed charge・検証済みforceと回転面eventで閉じ、続いてfinite-speed、continuous charge、
3D collision/noiseを**各組合せの根拠が閉じた順に**同じengineへ追加する。
元の2DOF COMSOLの再現と物理3D拡張は異なるcomparison intentである。
releaseでのphi/azimuth、場basis、軸極限、3D normal、schema/result/checkpoint、memoryを同じchangeで閉じる。
軸上からの平均0・等方transverse Gaussian displacementのradial momentは、3DのRayleighと
2D投影のfolded-normalを区別する独立microcaseになる。軸から離れた初期位置や非零平均は
同じRayleigh則にはならない。単なる回転可視化では3D dynamicsを検証できない。

### 22.8 次の担当agentへ渡すものと終了条件

実装者はこの第22節から一つのpackageを選び、そのowner文書、必要な小さいreference、
既存extraction/case/result hashを読む。A/P・tag・boundary ID・速度・seed・dtを次のchamberのdefaultにしない。
未実装modelをskillsに対応済みとして記述しない。skillsは既存の四単位を維持する。

- inspect：native active設定、依存式、実現source、solution、unit、selectionを一回抽出。
- prepare：物理model対応、source生成、geometry/field変換、独立numerical plan。
- validate：同一実現/分布比較、独立精度、差のownerと限界。
- visualize：保存済みの状態/eventとgeometryを使い、比較のmask・未知を表示。

新しい巨大統合skillやcase別skillは作らない。詳細調査資料は
[field/adapter設計レビュー](analysis/generic_improvement_2026_10_14/field_adapter_design.json)、
[physics設計レビュー](analysis/generic_improvement_2026_10_14/physics_design.json)、
[数値/V&V設計レビュー](analysis/generic_improvement_2026_10_14/numerical_vv_design.json)にある。
これらは調査記録であり、本節の実装順、technical researchの式、architectureのownerを上書きしない。
各実装は第18〜20節のquality/verification/性能条件と置換削除を満たした時だけ完了とする。
資料・skillsだけの更新をsolver改良やCOMSOL軌道一致の完了とは報告しない。
