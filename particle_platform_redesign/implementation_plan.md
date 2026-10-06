# 半導体チャンバー粒子輸送基盤 詳細実装計画

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

## 1. 実装する製品と、実装しないもの

### 1.1 最初に完成させる製品

最初の製品版 `v0.1` は Stage 0、Stage 1A、Stage 1Bに加え、P14-Pのparallel runtime収束、P14-Uの代表用途gate、P14-Rの配布gate、
T03の最小analysis/visualizationの完了で定義する。

- 外部で計算済みの静的2D場を読む。
- `cartesian_xy` と `axisymmetric_rz_meridional` を扱う。
- tableまたはpart surfaceから粒子を発生させる。
- fixed charge、drag、electric、gravity/buoyancyを連成して積分する。
- point particleのfirst hitを連続軌道上で検出する。
- stick、escape、restitutionを含む現行`specular`、probabilistic stickを扱う。完全鏡面は反発係数1/1である。
- 10^4～10^6粒子のstateをresidentに保ち、thread数非依存のbounded tile slabによるCPU batchで処理する。
- final state、boundary event、選択trajectoryをstreaming保存し、中断runをresumeできる。

`v0.1` は「全物理を薄く実装した試作品」ではなく、静的・決定論問題を正しく速く最後まで解ける
完成した縦切りとする。

### 1.2 後続stageで追加するもの

- Stage 2A：P15 continuous charge、外部M3-V applicability/relevance評価、P15-Dのspecies制約付き
  shifted-Maxwellian relative-drift charge、P15-E有限速度drag、P15-F collisionless Barnes ion drag、
  P16 Waldmann--Gallis thermophoresis、Brownian数値基盤B01とproduction縦切りB02まで完了。solver本体を
  凍結した外部M3-V deterministic trajectory certificationも、Case-A 100 nmのhash固定pre-event sliceで完了。
  M3-C0aでreference式・derived field・admissible刻み候補・Brownian seed cohortを外部lockし、P18-C/I/D/L/Rまで完了した。
  M3-C0bのv5旧判定は原点依存な位置relative L2のため無効化した。逐次確認v6ではCase-A 100 nmの0--450 usについて
  0.625/0.3125/0.15625 us系列をpre-eventの運用上の刻み選択としてだけ受理した。M3-C1のpre-event
  frozen saved-state replayは、最小のPPR熱泳動補足exportを加えて8/8 producer-formを閉じた。solver candidateは
  COMSOL native fieldそのものではなくexport済みexact-connectivity P1である。globalな連続適用域包絡による時刻0の
  過保守拒否はP19-Lの局所certificateで解消し、fixed-step endpointとrev3b event意味論は維持した。provenance固定した
  M3-C1 exported-P1/native-field比較は3 runを完了したが、空間表現をまたぐ6 gateはすべてFAILした。条件付きで事前登録していた
  全決定論物理のexact-connectivity common-field COMSOL診断は、固定済み9 gateをすべてPASSした。これはCase-A 100 nm、
  Brownian-off、0--450 us、event前のsame-field solver agreementだけを閉じる。続くM3-C0 boundary semantics probeは
  Freeze/Disappearを力なしの解析解で分離した。common-P1の最初のmaterial-stick eventはevent v14で20/20、再計算prefix
  9/9をPASSし、field表現差の最初の層も局在化済みである。P18-Hも解析・公開回帰と外部15/15 gateを閉じた。
  B03もcoreの解析・manufactured・identity gateを閉じた。後続の100 nm・30 ms candidate v3はCase A/Pともcandidate固有の
  `h,h/2,h/4`自己収束をPASSした。保存COMSOLはfixed RK4 10 us・Brownian-on・native-fieldの単一runなので外部記述に限定する。
  charge-stable coupling、長時間run向けdurable I/O cadence、P20 performance closeoutも完了した。外部V&V/M3-C2は
  common-P1 Case-A/Case-P 100 nmの32+32 seed finalまで完了し、受理済みCase-P workloadの287粒子owner discoveryも完了した。
  このanchor benchmarkは`CLOSED_ACCEPTED_WITH_LIMITATIONS`である。科学identityは完全一致し、支配ownerは`integrators`だったが
  事前登録済みbounded ownerではないため最適化は未承認である。10,000粒子以上の性能適格性は、利用SLAを先に定義した別work packageとする。
  後続の明示指示によるbounded chord follow-upは、同じaccepted seed 3件で科学payload/work/revisionを完全一致させ、
  end-to-end中央値を12.29%短縮して完了した。これは製品scale認定ではなく、追加ownerへ進まず閉じた一回限りの実装効率改善である。
  M3-C1 compact evidence/evaluatorのevent v14は履歴として固定する。
  P17はstate-dimension workstreamとして分離
- Stage 2B：B01のinertial joint OU、Brownian用counter stream、conditional splitに続き、B02で固定depth
  dyadic/Hermite pathのstochastic crossing/replayをCartesian XY・Epstein linear drag-only・fixed-charge state・
  terminal stick/escapeへproduction接続済みで、P18-Hでは同じterminal subsetへholdを追加済みである。B03では、現在のCOMSOL RZ比較に必要な明示的なmeridional投影revisionと、
  deterministic force・continuous chargeを合成するmacro-root stochastic exponential-midpoint経路を追加済みである
- field-production track：F01のcanonical RZ/P1 reduced electrostatic builderとF02のmixed-mesh
  provider adapter、代表規模linear-solve gate、fixed-charge/electric統合、外部field V&Vは完了
- Stage 3：現在のCOMSOL 12 packageを設定可能にする比較capabilityを、P18-C/I/D/L/R/H/P19-L/B03（完了）、M3-C0/C1/C2として
  独立sliceで追加する。実装済みDEP、documented sensitivity lift、aggregate two-current charge、2種類のion drag、
  必要ならdrag/mixture thermophoresisをoptionalなversioned modelとする。terminal holdは確認済みの非deposition保持を
  最小generic `hold/held`として実装し、COMSOL名やCase P/A分岐をcoreへ入れない
- Stage 4A：fixed-topology time-dependent field
- Stage 4B：tet4 field、tri3 boundaryによる完全3D
- Stage 5：適合caseだけのGPU backend、配布・viewer・製品運用

### 1.3 初期に実装しないもの

- 旧solverのclass、helper、fallback、比較用分岐の移植
- COMSOL version別profileやCOMSOL専用solver path
- DI container、entry-point plugin、抽象base class階層
- 一般ODE/DAE/IMEX framework
- MPI、Dask、distributed scheduler
- moving mesh、CAD repair、高次要素、hex8、adaptive octree
- 有限半径接触、rolling/sliding、film、resuspension
- checkpoint migration、複数永続形式、常時full trace
- GPUとCPUで異なる物理・境界意味論

実際の利用caseが二つ以上現れる前に一般化しない。後続機能は同じengineへ一つずつ追加し、置換した
旧経路を残さない。

---

## 2. clean-room実装の配置

既存packageとimport graphを共有しないため、新しい独立projectを次へ作る。

```text
particle_platform_redesign/
├─ AGENTS.md
├─ implementation_plan.md
├─ quality_tooling_plan.md
└─ solver/                         # Stage 0で新規作成する独立project
   ├─ .python-version
   ├─ pyproject.toml
   ├─ uv.lock
   ├─ README.md
   ├─ src/chamber_particles/
   │  ├─ __init__.py
   │  ├─ api.py
   │  ├─ case.py
   │  ├─ case_format.py
   │  ├─ coordinates.py
   │  ├─ geometry.py
   │  ├─ fields.py
   │  ├─ sources.py
   │  ├─ rng.py
   │  ├─ physics/
   │  │  ├─ __init__.py
   │  │  ├─ catalog.py
   │  │  ├─ forces.py
   │  │  └─ charge.py
   │  ├─ integrators.py
   │  ├─ events.py
   │  ├─ boundaries.py
   │  ├─ engine.py
   │  ├─ cpu.py
   │  ├─ output.py
   │  └─ __main__.py
   ├─ tools/
   │  ├─ importers/comsol/
   │  ├─ importers/tables/
   │  ├─ case_builder/
   │  ├─ field_preprocessor/
   │  ├─ electrostatic_builder/
   │  ├─ analysis/
   │  ├─ visualization/
   │  └─ vv/comsol/
   ├─ examples/
   │  ├─ microcases/
   │  └─ chamber_screening/
   ├─ tests/
   │  ├─ verification/
   │  ├─ scenarios/
   │  └─ performance/
   ├─ scripts/
   │  └─ check_complexity.py
    └─ docs/
       ├─ case_format_v2.md
       ├─ result_format_v2.md
       ├─ physics_models.md
      ├─ numerics.md
      └─ decisions.md
```

`solver/`は独自の`pyproject.toml`と`uv.lock`を持ち、旧repository rootのpackageをdependencyにしない。
Stage 1B完了後、そのまま別repositoryへ切り出せる構造とする。旧コードからのcopyは行わず、式や
dataset名を参照した場合は、採用理由を`docs/decisions.md`へ短く記録する。

P00では`pyproject.toml`を作成した後に`uv lock`を一度実行して初期`uv.lock`を生成する。その後の通常操作と
clean checkoutでは`uv sync --locked`を使い、意図したdependency変更時だけ`uv lock`を再実行する。本書の
commandは`particle_platform_redesign/solver/`をcurrent directoryとする。repository rootから実行する場合は
同じsubcommandを`uv --project particle_platform_redesign/solver ...`の形で呼び、root側に別環境やlockを
作らない。

初期runtime dependencyは `numpy`、`h5py`、`PyYAML`、Stage 1Bで`numba`とする。build backendは
`uv_build`をuvと同じminor rangeでlockし、別のbuild systemを並存させない。CLIは標準`argparse`で
十分であり、web frameworkやplugin managerを加えない。可視化・COMSOL adapterのdependencyはcoreへ
入れずtool側のoptional extraにする。

P00では三公開APIと三公開例外のimport境界だけを固定し、APIを呼ぶと
`NotImplementedError`でfail closedする。これは第二のruntimeではない。`load_case`はP01、
`simulate` / `open_result`はP04で同じ関数を利用可能な縦切りへ置き換え、暫定失敗分岐は
同じchangeで削除する。それ以外の将来moduleをempty skeletonとして先行作成しない。

---

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
| `simulate(case, output)` | 小さい`RunSummary` | prepare、計算、streaming出力、finalize |
| `open_result(path, recovery=False)` | lazy `ResultView` | 完了結果または明示されたrecovery結果の遅延参照 |

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

v0.1では粒子単位で局在できる数値不能は粒子statusを`failed`にし、小さなreason codeをeventへ残す。P05時点は
failure event未実装のため局在不能をrun-fatalとしていたが、P08のStage 1A closureで継続policyを追加済みである。入力不正、
非有限model係数、writer破損などrun全体の意味が失われる問題は即時停止する。粒子failureをstickやescapeへ
読み替えない。

---

## 4. fileごとの具体的責務

| file | 最初に実装する内容 | 入れてはいけない内容 |
|---|---|---|
| `api.py` | 三APIの薄い呼出し、公開例外の整形 | 物理式、HDF5 dataset操作 |
| `case.py` | frozen設定dataclass、YAML parse、`SimulationCase` | producer固有列、backend kernel |
| `case_format.py` | `case.h5` schema、read/write、version、hash | COMSOL名、軌道計算 |
| `coordinates.py` | XY/RZ/axis規則、座標・vector・normal変換 | field locate、wall law |
| `geometry.py` | domain cell、line/triangle facet、BVH、containment、first-hit query | 反射・付着、field値 |
| `fields.py` | layout、strict support、P1/Q1/tet4 sampling、time interpolation | cache生成、mesh修復 |
| `sources.py` | table/surface release、発生時刻、weight | engine lifecycle、wall再飛散 |
| `rng.py` | counter key、uniform/normal、stream ID | sourceやwallのpolicy |
| `physics/catalog.py` | category、model code/revision、required field、対応mode | mesh探索、I/O |
| `physics/forces.py` | drag、electric、gravity/buoyancy、後続加算力 | lifecycle、boundary response |
| `physics/charge.py` | fixed/continuous charge、rate・平衡残差、有限invariantとrate/derivative bound | 全状態のstep orchestration、stiff method選択 |
| `physics/compiled.py` | sample済みprimitiveからのNumba model加算 | field locate、event、fallback dispatch |
| `integrators.py` | RK4、exponential midpoint、`StepProposal.state_at()` | BVH、law、file output |
| `events.py` | path tube、earliest hit局在、残時間work | 反射式、永続形式 |
| `boundaries.py` | hit後のstick/escape/specular/probability | 交差探索、位置押戻し |
| `engine.py` | prepare、唯一のmacro loop、lifecycle、barrier調停 | importer、plot、COMSOL比較 |
| `cpu.py` | SoA、直列compiled kernel、bounded slab/workspace容量、memory plan | modelの物理的意味 |
| `output.py` | ResultWriter、epoch commit、checkpoint、ResultView | deposition等の科学集計 |

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

### 5.1 二ファイルだけを標準にする

```text
case.yaml   # 人が編集するrun設定
case.h5     # geometry、field、boundary metadata、optional source table
```

`case_format.write()`は出力先と同じdirectoryの一時fileへ書いてclose/fsync後、hard linkで
no-clobber publishし、schema versionとcontent hashを返す。既存出力は上書きせず、hard linkを
提供しないfilesystemでは非atomic fallbackを使わず明示的に失敗する。adapterは必ずこのwriterを
介し、core loaderはCSVやCOMSOL exportを直接読まない。

### 5.2 `case.h5` v1の最小layout

```text
/meta/schema_version
/meta/coordinate_system                  # geometry/field data表現だけ。motion modeではない
/meta/coordinate_units                 # "m"
/meta/provenance_json

/geometry/nodes_m                       # [Nnode, ndim], float64
/geometry/node_external_id              # [Nnode], int64, optional traceability
/geometry/cells/tri3                    # [Ntri, 3], int64, optional
/geometry/cells/quad4                   # [Nquad, 4], int64, optional
/geometry/cells/tri3_domain_id           # [Ntri], int32; tri3がある時だけ
/geometry/cells/quad4_domain_id          # [Nquad], int32; quad4がある時だけ
/geometry/boundary/line2                # [Nedge, 2], int64 for 2D/RZ
/geometry/boundary/external_id           # [Nedge], int64, optional traceability
/geometry/boundary/boundary_id           # [Nedge], int32
/geometry/boundary/group_id              # [Nedge], int32
/geometry/boundary/material_id           # [Nedge], int32
/geometry/boundary/owner_cell_type       # [Nedge], uint8; 1=tri3, 2=quad4
/geometry/boundary/owner_cell_local_index # [Nedge], int64; type内の0-based index
/geometry/boundary/orientation           # [Nedge], int8; +1=ownerのCCW edge順、-1=逆順
/geometry/groups/names                   # UTF-8 [Ngroup]; dense group ID 0..Ngroup-1

/layouts/<layout>/kind                   # regular | p1_tri | q1_quad
/layouts/<layout>/regular/axes/axis0_m   # kind=regularだけ。strictly increasing
/layouts/<layout>/regular/axes/axis1_m
/layouts/<layout>/regular/cell_support   # [n0-1,n1-1], uint8; 1だけsample可能
/layouts/<layout>/unstructured/nodes_m   # kind=p1_tri|q1_quadだけ
/layouts/<layout>/unstructured/connectivity # [Ncell,3|4], kindでarityを固定
/layouts/<layout>/unstructured/cell_support # [Ncell], uint8; 1だけsample可能

/fields/<field>/layout
/fields/<field>/association             # node | cell
/fields/<field>/components
/fields/<field>/stored_basis
/fields/<field>/values                  # static: [N,C], later time: [T,N,C]
/fields/<field>/time_s                   # Stage 4Aまで無し
/fields/<field>/unit

/sources/<name>/particle_id              # [N], int64
/sources/<name>/release_time_s           # [N], float64
/sources/<name>/position_m               # [N,ndim], float64
/sources/<name>/velocity_m_s             # [N,ndim], float64
/sources/<name>/charge_number            # [N], float64
/sources/<name>/mass_kg                   # [N], float64
/sources/<name>/drag_diameter_m           # [N], float64
/sources/<name>/electrostatic_radius_m    # [N], float64
/sources/<name>/displaced_volume_m3       # [N], float64
/sources/<name>/model_weight              # [N], float64
/sources/<name>/material_id               # [N], int32
```

mixed elementを一つのragged tableへ押し込まず、`tri3`と`quad4`を分ける。boundaryのownerは
`(owner_cell_type, owner_cell_local_index)`の組で一意にし、typeを無視したlocal indexだけを渡さない。
tri3は反時計回り`(v0,v1,v2)`、Q1は参照座標
`(-1,-1),(+1,-1),(+1,+1),(-1,+1)`に対応する反時計回りの周回順に固定し、正のJacobianを
要求する。COMSOL adapterはproducer側の順序をこの順序へ一度変換する。boundary `line2`の
`orientation=+1`は線分がowner cellのCCW local edge順に一致すること、`-1`は逆順を表す。
線分方向`t=(dx,dy)`に対し、外向き法線は`orientation * (dy,-dx)/|t|`である。

v1のgeometry cellはすべて粒子が移動可能なdomainを表す。solid volume cellを同じtableへ混ぜず、
固体表面はboundary facetで表す。`domain_id`は移動可否のflagではなくcanonical labelである。
2D/RZのparticle domainとして`tri3`または`quad4`を少なくとも一要素要求する。一方、壁なしの
解析microcaseを表せるようboundary facetは0件を許可し、その場合group namesとYAML boundary lawも
0件とする。
data座標表現`axisymmetric_rz`ではgeometry nodeの`r >= 0`を要求する。volume外周の`r=0` edgeは常に
axis seamであり、material `line2`として登録した入力は拒否する。axis seamを明示wallへ切り替えるv1設定は持たない。

layout groupは`kind=regular`なら`regular/`だけ、`kind=p1_tri|q1_quad`なら`unstructured/`だけを持ち、他方を
作らない。regular connectivityはaxisの直積から暗黙に決まり、field値のnode/cell順はC-orderでflattenする。
unstructured connectivityはlayout内のnode indexを参照する。supportは曖昧なmetadataではなく上記の
cell単位maskをauthorityとし、supported cellの閉包の和をsample可能領域とする。共有面のcandidateに
supported cellが一つ以上あれば`support_inside=True`とし、最小supported cell IDをownerにする。
inside候補が全てmaskedならoutsideとする。provisional trial値は全supported cellの物理閉包への
Euclidean最近傍射影から取り、同距離だけを最小cell IDで決める。masked cellのplaceholderや参照空間の
近さを値ownerにせず、regular/P1/Q1とcell hintでこの意味を変えない。

fieldを使わないballistic caseのため、layoutとfieldはともに0件を許可する。fieldがある場合は既存layoutを
必ず参照する。Stage 1Aの実行profileは検証済みのP1 triangle、Q1 quad、regular gridだけを
許可する。schemaの将来欄を先回りして空datasetとして作らない。realized table sourceは初期状態と
粒子ごとの権威値だけを持ち、分布modelや壁面物理をHDF5へ入れない。

schema versionはint32 scalar、全の物理実数はlittle-endian float64、index/external IDはint64、
domain/boundary/group/material IDはint32、supportとowner typeはuint8、orientationはint8、文字列は
UTF-8に固定する。HDF5 attribute、soft/external link、VDS、未知objectは受理しない。
realized tableの`particle_id`は全sourceを通して非負かつ一意とする。`mass_kg`、
`drag_diameter_m`、`model_weight`は正、`electrostatic_radius_m`と`displaced_volume_m3`は非負、
`material_id`は非負とする。

content hashはHDF5のobject address、chunk、compressionに依存しない論理SHA-256とする。
`chamber-particles-case\0v1\0`をdomain headerとし、全datasetをfull pathでsortし、path、論理dtype、
rank/shape、C-order little-endian値を長さ付きrecordとしてhashする。文字列は値ごとのUTF-8長と
byte、provenance JSONはsorted key・compact separator・NaN禁止のcanonical表現を用いる。`-0.0`は
正規化しない。戻り値は`sha256:<lowercase hex>`とし、hash自身をHDF5へ格納しない。

### 5.3 `case.yaml`の実装単位

geometry/fieldのdata座標表現、layout、boundary ID、単位、producer provenanceはHDF5側だけが所有する。
粒子状態の自由度を決めるmotion modeはYAMLのSimulationSpecだけが所有する。YAMLはこのmotion mode、
実行設定、HDF5 path、期待content hashを持ち、data座標表現を二重記載しない。
YAMLの`format_version` は1とし、top-levelと共通構造の未知keyを拒否する。必須値にdefaultを
補わない。physics categoryの欠落だけを無効の表現とし、`null`と欠落を並存させない。
無帯電でもcharge categoryは`model: fixed`を明示し、各sourceの`charge_number`を0にする。`fixed`は
`dZ/dt=0`だけを所有し、初期値を重複して持たない。model固有parameterの中身はmappingとして保持し、
適用域とrequired fieldは`engine.prepare`だけが解釈する。

`case.py`には次だけをfrozen dataclassとして置く。

- `TimeSpec`
- `EventSpec`
- `SolverSpec`
- `MotionSpec`
- `ResourceSpec`
- `ParticleProperties`
- `SourceSpec`
- `PhysicsSpec`
- `BoundarySpec`
- `TrajectoryOutputSpec`
- `OutputSpec`
- `SimulationSpec`
- `SimulationCase`

model固有parameterはそのmodel節のmappingとして保持し、prepare時に該当catalog entryが一度だけ解釈する。
全modelのparameterを巨大なunion classへ集約しない。

P04時点の`TrajectoryOutputSpec`は`selection: all`と`schedule.explicit_times_s`だけを型付けして受理する。
保存時刻はfinite、狭義単調増加、重複なし、`[time.start_s,time.end_s]`内とする。final particle表とevent logは
resultの必須部分なのでenable flagを持たない。将来の`none/sample/ids/interval_s`は、その利用caseを実装する
packageで同じ型を拡張し、任意mappingをruntimeへ流さない。

### 5.4 検査の所有者

| 境界 | 一度だけ検査する内容 |
|---|---|
| importer | 元形式の列、単位、local node order、source固有ID |
| `case_format.read` | schema version、dataset、shape、dtype、有限性、index範囲、局所node順、owner edge整合 |
| `load_case` | YAML参照、期待content hash照合、全boundary groupのlaw、simulation/field時間範囲 |
| `geometry.prepare` | cell incidence、外周facetの完全性と一意性、non-manifold/重複/internal wall拒否 |
| `engine.prepare` | model重複、required field、適用域、data座標×motion×integrator×backend、memory |
| runtime | prepare後の内部不変条件だけをassert |

同じ検査をpreflight、provider、runtimeへコピーしない。使えない値をdefault、epsilon、別modelで修復しない。
geometry修復や外周facet生成はimporter/case builderの責務で、coreは判定と拒否だけを行う。boundaryを使う
v0.1 caseでは、RZ axis seamを除く全てのincidence-1 edgeをちょうど一度boundaryとして与える。内部baffleは
別revisionで意味を定義するまで拒否する。boundaryが空のcaseはP04のcollision-free profileに限る。

---

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
検討し、v0.1では直接SoAを優先する。P09は`cpu.py`が固定容量のactive indexを所有し、
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
- resident memory、thread非依存slab、同期writer reserveを含むresource plan
- output scheduleとResultWriter設定

hot loopではYAML文字列、dict lookup、capability探索、Python callbackを使わない。未対応組合せはprepareで
拒否し、scalar evaluatorへ黙って切り替えない。
YAMLの`boundaries[].law`は入力selectorであり、prepare時に非公開dense `law_code`へ解決できるが永続化しない。
eventの`law_id`は選択されたtop-level lawの安定semantic IDとし、compound lawのleaf分岐は`outcome`と
draw referenceで表す。`law_name`という別名を作らない。

### 6.3 memory planner

```text
resident state
+ active/pending index
+ geometry/BVH
+ resident field snapshot(s)
+ parsed DataBundle / prepared indexの保持分
+ thread-independent parallel tile slab
+ residual-work buffer
+ event/frame writer buffers
+ safety margin
<= resources.memory_limit_mb
```

scratchは全N分でもthreadごとでもなく一つのslab分だけ確保する。plannerは通常利用者へslab幅を要求せず、実行時の予測内訳を
`run.json.memory_plan`へ出す。`check`は公開`load_case`の薄い呼び出しに留め、HDF5 metadataから得る
canonical numeric bytesと設定上限だけを報告する。geometry、source realization、physics runtime、writerを
含む完全なplanは`simulate`内のprepareが一度だけ解決する。最小slabでも予算を超える場合は
run開始前に停止する。

v0.1は粒子状態全体をresidentに置き、scratchだけをslab化する。100万粒子で必要性が実測される前に
outer/out-of-core particle batchを作らない。outer batchはcheckpoint、時系列集計、event順序を複雑にする。
P09の`solver_owned_memory_plan_v1`はload/prepare/runのphase peak、resident/component内訳、自動選択した
microtile幅を示すが、Python、native library、allocatorを含むOSのhard RSS上限ではない。`p09_memory.py`が
fresh/warm processのload/prepare/run peak RSSを別系統で測定し、plannerと実測の差をP10/P14の最適化判断に渡す。
P10の`solver_owned_memory_plan_v2` / `resident_soa_microtile_v2`は、実際に配置したP1/Q1 resident hintと
compiled runtime配列を同じphase/component計算へ含める。memory limitの意味は変えない。
P12の`solver_owned_memory_plan_v3` / `resident_soa_worker_microtile_v3`は、要求thread数から解決したworker数、
workers×proposal scratch、workers×geometry query scratchを同じrun peakへ含める。memory不足をthread数の
暗黙削減で隠さず、最小microtileが収まらなければ開始前に拒否する。
これは履歴revisionである。現行P14-P closeout後の`resident_soa_serial_slab_v5` /
`solver_owned_memory_plan_v11`はthread数非依存のstage/proposal/output slabへ置換し、worker別scratch componentと
runtime layoutを削除した。deferred event depthは `event_work_bytes_per_particle = 24 *
(max_refinements + 1)` として解決し、`slab_event_work`へ独立計上する。depth依存workをproposal scratchへ
隠さない。候補、event/failure staging、surface release、direct replayはnamed componentへ分離し、pack時だけの
gatherは12.5% safety marginが所有する。正確なbyte式とcapacity規則は
[`solver/docs/parallel_execution_plan.md`](solver/docs/parallel_execution_plan.md)を権威とする。thread scaling gateは
P14-Pで不採用まで判定済みである。P14-Uは現行直列runtimeのmemory plan、RSS、end-to-end時間を代表用途で測る。

---

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

v0.1で運動を駆動するrequired fieldは、particle domain全体を覆うcontinuous node-associated
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
- RZ axis seam以外の全外周edgeを一度ずつboundaryへ対応させる。v0.1で内部thin baffleは受理しない。
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

RZ meridionalのsurface sourceはsampling measureを必須選択にする。

- `meridional_length`：断面長に比例
- `revolved_area`：物理的な回転面積 `2πr ds` に比例

暗黙にどちらかを選ばない。半導体チャンバーの物理surface releaseは通常`revolved_area`をsample caseへ
明記する。`revolved_area`ではedge選択だけでなくedge内の位置も`r(s)`に比例する密度からsampleする。

### 7.4 remesh/cacheの境界

runtimeはremeshもregular cache生成もしない。`tools/field_preprocessor`が作成し、manifestへ原meshとの
support、field/gradient誤差、wall-near誤差、trajectory/event同等性、実測speedupを記録する。
v0.1はphysics設定が参照する一つの明示layoutをrunのauthorityとする。cacheを使う場合、外部preprocessorが
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
machine-readable ID/revisionと文書の説明を対応させる。適用域外は設定された`error | count`のどちらかとし、
別dragへ自動切替しない。

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

Stage 1Aは二つだけ実装する。

- `table`：位置、速度、release time、粒子属性を明示
- `surface`：boundary group、非重複`particle_id_start`、位置分布、角度分布、速度分布、時間分布、
  model weightを明示

単一点は一行のtableとして表し、専用point-source classを作らない。surface samplingはfacet measureのCDFを
prepareし、counter RNGで`source_id/source_particle_ordinal/draw_kind`から再現する。

P07の最初のproduction subsetは、単一`ParticleSchedule/realize_sources`でtableとsurfaceを統合する。
surfaceは固定時刻、`edge_fraction`またはuniform位置、固定vectorまたは固定speedのdomain内向き法線速度に限る。
uniform measureはXYの`line_length`、RZの`meridional_length | revolved_area`を明示し、暗黙defaultを持たない。
realized scheduleは位置とfacet IDを保持し、ownerと法線はprepared geometryから参照する。source drawは
scalarと同じcounter/keyを保つvectorized Philox batchで生成する。

### 10.2 boundary law

- `stick`：terminal `stuck`
- `escape`：terminal `escaped`
- `specular`：P07では静止壁に対する速度を法線・接線へ分解
- `probabilistic_stick`：一回だけ乱数を引き、非付着側lawを明示

lawは既に局在した`BoundaryEvent`を受け、位置を動かさず速度・lifecycle・event recordだけを決める。
corner候補集合に吸収lawが含まれる場合と複数反射面の場合のpolicyをcase formatで一つに固定し、iteration順へ
依存させない。

### 10.3 RNG

`rng.py`はNumbaへ移しても同じcounter意味論を維持できるPhilox4x32-10を一つ実装し、既知vectorで検証する。

- source：`seed, source_id, source_particle_ordinal, draw_kind`
- wall：`seed, particle_id, physical_boundary_event_ordinal, law_stream`
- Brownian：`seed, particle_id, macro_interval, root_stochastic_interval, tree_level, tree_index, component, stream`

thread、tile、保存schedule、event局在の数値反復回数をkeyに含めない。数値refinementはphysical event
ordinalを進めない。

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

### 12.1 Stage 1Aと1Bの分離

Stage 1Aは同じSoA、同じengine、同じmodel codeをPython/NumPy reference evaluatorで動かし、数値意味を
確定する。Stage 1Bはorchestratorを変更せず、tile kernelだけをNumbaへ置換する。reference evaluatorは
test oracleとして残すが、productionのsilent fallbackには使わない。

P10はfield location/interpolation、sample済みprimitiveのphysics、classical RK4算術を
`compiled_cpu_tile_v1`へ移した。`state_at()`、wall hit、residual、outputは同じproposal/event loopを使う。
P10時点ではNumba 0.67、NumPy `<2.6`、`fastmath=False, parallel=False`を固定し、parallel schedulerは
P12へ残した。P11のnative integratorをcompiled tile v2、P12のevent-heavy worker経路をcompiled tile v3へ
更新したが、同じproduction engineとproposal/event意味論を使う。

### 12.2 serial execution ownership

- computeはsingle-thread compiled engine一つとし、outer ThreadPool、Numba内部parallel、multiprocessingをsolver内に持たない。
- field/geometryはread-onlyで再利用する。
- stage/proposal/event workはthread数非依存の一つのpreallocated tile slabへ置く。
- event/residualは行ごとのtarget time、depth、interaction/event ordinal、statusを持つflat SoA queueでround処理する。
- 各roundは一行あたり高々一event/failureを固定slotへ書き、stable prefix/compactionで出力と次roundを作る。
- shared list、Python粒子callback、可変thread-local container、private thread IDを使わない。
- boundary eventのidentityは`(particle_id,event_ordinal)`、公開時のcanonical順は
  `(time_s,particle_id,event_ordinal)`とし、thread完了順とtile幅へ依存させない。
- compute threadはfileへ書かず、engine barrierだけがcolumnar batchをwriterへ渡す。

deterministic CPUではfastmathを既定無効にし、force加算順とstable prefix/reduction順を固定する。同一binary/
hardwareではslab幅、trajectory保存scheduleを変えても粒子別final/event/RNG、failure/refinement counter、
frame/probe/seriesが一致することを要求する。

P12の最大`W` worker wave、futureのtile順merge、worker-local scratchは履歴であり、現行engine v37では削除済みである。
geometry broad phaseのexact predicate、保守的RK4 clear/split証明、同時刻wall意味論は維持する。boundary BVHの
stackless traversal、linear/quadratic exactと一般曲線eventのflat SoA wavefront、boundary/RNG batch、row numerical
status、batch surface release、direct columnar replay、bounded event/failure stagingはengine接続済みである。
詳細な実装・計測・削除条件は
[`solver/docs/parallel_execution_plan.md`](solver/docs/parallel_execution_plan.md)を権威とする。

### 12.3 最適化順

1. profile可能なreference vertical slice
2. SoAとtemporary reuse
3. regular supported containing-cell O(1) common pathとP1/Q1 previous-cell strict-interior hint
4. field/physics/RK4 array passのNumba化
5. thread数非依存のreusable tile slabと`into` pass
6. stackless boundary BVHとflat SoA event/residual wavefront
7. compiled boundary/RNG、stable prefix/columnar output
8. output backpressureとI/O分離
9. 実測で支配的なfield/force passだけfusion
10. profileで必要ならP1/Q1 adjacency/BVH、cacheが精度とspeedupを満たすcaseだけregular化
11. CPU完成後、適合caseだけGPU

JAX/GPUの都合でcase format、boundary意味論、RNG keyを変えない。

---

## 13. ResultStore、checkpoint、後処理

### 13.1 P04の最小resultとP13のdurable拡張

```text
OUT.partial/
  run.json
  segments/epoch-000000.h5
  final.h5
  _SUCCESS
```

P04は上記の一つのclosed segmentとfinalizeだけを実装する。segmentにはrelease eventと明示時刻frameを入れ、
`final.h5`は常に作る。`run.json`はcase/input hash、data座標表現、motion mode、algorithm revision、status、
粒子/event/frame件数を持つ。`open_result`はこのlogical datasetをlazyに読む最小操作だけを提供する。
P04時点では`recovery=True`を明示的に拒否した。

P13は同じlogical resultへmulti-segment、`checkpoints/A.h5`、`checkpoints/B.h5`、`LATEST`を追加し、
checkpoint/resumeとrecoveryを有効化した。P04に空checkpoint、background writer、queue、将来用groupを
先行作成しなかった経緯は維持する。physical layoutを増やしても、P04で固定したevent/frame/finalの意味は
変更していない。

P04 segmentのlogical datasetは次だけにする。

```text
/events/*                     # releaseだけ。型と列はresult schema v1で固定
/frames/time_s
/frames/offset
/frames/particle_id
/frames/position_m
/frames/velocity_m_s
/frames/charge_number
/frames/lifecycle
```

P05/P07がboundary eventとcandidate facet、P08以降が小さいseries、明示probeを同じsegmentへ追加する。
未実装datasetを空で先行作成しない。常時全粒子の全force、全RK stage、全局在iterationを保存しない。

### 13.2 P13 durable commit（初回履歴と現行supersession）

P13はengine v19、result algorithm `durable_segmented_result_v3`、checkpoint schema 1として完了した。
epoch cadenceはoutput scheduleやworker分割から独立した固定64 macro-stepであり、最終macro後は64未満でも
commitする。

1. segment tempをflush/closeし、確定名へrename
2. inactive checkpoint tempをcloseし、A/Bへreplace
3. `LATEST.tmp`を`LATEST`へreplace。ここだけをaccepted macro-step epochのcommit pointとする
4. resumeは`LATEST`より新しいorphanを無視
5. finalは`final.h5 → complete run.json → _SUCCESS → OUT directory rename`

failure injectionを各境界へ入れ、event/frameの重複・欠落がないことをStage 1Bで確認済みである。checkpoint
migrationは作らず、input hash、case/result/checkpoint schema、physics/algorithm/backend revision、座標・method、
粒子ID集合が一致する場合だけresumeする。HDF5は単一background writer threadだけが所有し、容量1 queueへの
各commandは完了ackを待つためdisk遅延はcomputeへbackpressureする。checkpointもepoch barrierでcomputeを止め、
writerの完了後にだけA/Bと`LATEST`を進める。巨大なstate snapshotの複製と、曖昧なresume境界を作らない。
最初の`LATEST`前は初期stateからepoch 0を正確に再実行し、final/run.json/`_SUCCESS`/directory publicationの
各途中状態は次回runが検査して最終化を再完了する。確率wall RNGのphysical ordinalもcheckpointから復元する。
checkpoint cadenceはoutput selection、frame schedule、segment flush cadenceから独立させ、accepted
macro-step epochのbarrierでだけ適用可否を判定する。output schedule変更でcheckpoint stateやresume境界を
変えてはならない。

### 13.3 bounded output

P04～P12はmain threadだけが同期writerを呼び、workerはfileへ直接書かなかった。P13はこの入口を維持したまま、
worker-wave単位のevent/failure stream、容量1のbounded writer queue、command ackを追加した。diskが遅い場合は
computeへbackpressureし、eventを捨てない。memory plan v4は`worker_output_staging`をproposal scratchや
`output_buffer`と分離して計上する。frameはtime/frame単位でchunkし、全粒子×全保存時刻をmemoryへ持たない。
同期ackはbounded memoryの意味論でありI/O overlap性能を主張しない。参照checkpointと最新segmentはhash、過去segmentは
構造と累積countを検証するため、同shapeの過去値改変は検出契約外である。power loss、remote filesystem、同じOUTへの
複数process同時実行もv0.1の保証外とする。
このP13範囲は標準品質gateとverification/scenario 322件でcloseした。製品規模I/O/RSSは後続P14で測定した。
P14-Pでは、各command直後にackを待つP13 queueがcompute/I/Oを重ねないことからbackground threadとqueueを削除し、
main threadの同期single-owner writerへ戻した。segment/checkpoint/finalのdurable commitとschemaは維持する。

現行`cumulative_solver_work_v1`は固定64 cadenceを置換する。engineが
`W=macro_step_count+accepted_particle_pieces+candidate_queries+refinements`と
`T=max(2^20,128N)`を所有し、accepted macro barrierでepoch開始時との差が`T`以上、または最終macroならcommitを決める。
cadence revision、resolved threshold、components、barrierはmanifestとresume identityに入り、output scheduleとslab幅から
独立である。`output.py`の同期single-owner writerはsegment→inactive A/B checkpoint→`LATEST`のatomic persistenceを
所有し、cadenceを再決定しない。

### 13.4 analysisとvisualization

`tools/analysis`は`ResultView`だけを読み、次を派生する。

- fate count/weight
- source-to-target matrix
- boundary/material別deposition
- arrival/residence time
- impact energy/angle
- ensemble confidence interval

`tools/visualization`はtrajectory、deposition、population time seriesを描くが、物理式やwall判定を再実装しない。
COMSOL比較plotは`tools/vv/comsol`だけに置く。CSV/Parquet exportもanalysis側で必要時だけ行う。
analysis成果物はsource runへ追記せず、source run hash、analysis revision、parameterを持つ別directoryへ書く。
v0.1のT03はこの全候補を一括実装せず、fate count/weight、boundary/material別deposition、outcome別arrival
time集計、保存済み代表trajectory、boundary event SVGだけを完成範囲とする。arrival timeはrelease後の飛行時間ではなく、
resultに保存されたsimulation絶対時刻である。source-to-target、impact
energy/angle、residence、confidence interval、汎用exportは具体的な利用caseが決まった後の独立追加とする。

---

## 14. microcaseと受入試験

test directoryは`verification / scenarios / performance`の三層だけにする。private helper、関数呼出し順、
file行数、directory内部をtestしない。

### 14.1 Stage 0で固定するmicrocase

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

Stage 2Aでcharge relaxation、axisymmetric-field/3D回転面caseを追加し、Stage 2BでOU統計と平面
first-passageを追加する。microcase IDをCOMSOL case名にしない。

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

### 15.1 P14履歴benchmark matrix

次表はP14当時のsynthetic評価軸であり、現行runtimeのthread幅選択表ではない。P14-Pで内部parallelと
`resources.threads`を削除済みで、現行productionはsingle-thread compiled engineだけを使う。

| 軸 | 条件 |
|---|---|
| 粒子数 | 10^4 / 10^5 / 10^6 |
| event密度 | 0 / 1 / 5 / 20 hit per particleのsynthetic case |
| field | regular / mesh-native P1 / mesh-native Q1 / accepted cache |
| output | none / sample / all（短いcaseだけ） |
| P14履歴worker | 1 / 2 / 4 / 8 / physical core数（重複除去） |
| 起動 | cold compile / warm |

記録する値はwall time、particle-step/s、event/s、field sampling time、physics time、event time、I/O time、
peak RSS、bytes/particle、writer throughput、compile timeである。kernel時間だけを性能値にしない。

### 15.2 合格方法

絶対秒数はhardware baselineを測る前に固定しない。Stage 1A reference、Stage 1B compiled、必要時のCOMSOLを
同一physics・mesh・dt・output量で測る。次を満たした時だけ高速化を採用する。

- final/event意味論が受入budget内で同じ
- memory予測と実測RSSの差を説明できる
- N増加に対する時間・memoryが概ね線形
- event-heavyでshared atomicやunbounded queueへ退化しない
- end-to-endで有意なspeedupがある

P10のmanual harnessはfield-heavy/event-lightと小さいevent-heavyの二caseに限定し、fresh processの空Numba
cacheによるcold JITとsame-process warmを分離する。semantic digest、RSS、全revision、speedupを記録するが、
絶対thresholdは置かない。上記10k/100k/1M・field・event・output・threadの全matrixはP14が所有し、
23行×3観測の直交型matrixとして完了した。unstructured行列にはrealistic cell count、initial localization、cross-cell motionを含めた。
read-only synthetic local profileではP1 stripの1000 warm sampleでhintなしfull searchが
100/500/1000/5000 cellに0.026/0.131/0.249/1.271 s、正しいstrict-interior hintが約0.0005 s
（51x～2576x）だった。初回stageは全粒子がhintなしであり、large-mesh P1/Q1性能はP10では未証明とする。
P14でfull searchが支配的と判定し、supported containmentだけをfield v3のBVHへ置換した。

P12の同一machine・512粒子×4 macro stepのwarm非gating実測（3回median）では、直前serial baseline
3.066786 sに対し、P12当時の1/2/4 threadは0.440460/0.528229/0.632361 sだった。1 threadはbaseline比6.96倍、
T1/T2=0.83384、T1/T4=0.69653である。workは8,704 accepted piece、19,968 candidate query、
11,264 refinement、最大深さ21で、全9観測のpayload digestは一致した。compiled BVH query、
保守的RK4 clear/split事前認証、同時刻wall prefix batchによるevent-heavy改善は採用する一方、正のthread scalingは
まだ確認できなかった。P12はparallel ownershipとstable mergeの正しさまでを完了とした。P14ではregular大規模で
正のscale、unstructured 10kで小さいscale、event-heavyで負のscaleを確認し、1 workerを既定推奨とした。

P14-Pはこの結果を理由にouter ThreadPoolを温存せず、Numba内部parallel runtimeを一度だけ事前閾値で評価した。
科学payload同一性をhard gateとし、single-thread回帰、regular/representative/P1/Q1/event-heavy/RZのspeedupと
memory増加を測ったが、限定修正後もregular 1Mの4-thread speedupは0.923xだった。このため削除条件を実行し、
`resources.threads`、thread mask、並列専用code/testを除去してcompiled single-thread engineへ一本化した。
5独立processの測定条件、具体的閾値、削除根拠は
[`solver/docs/parallel_execution_plan.md`](solver/docs/parallel_execution_plan.md)を権威とする。

「COMSOLより高速」という主張は、測定したcase、hardware、出力条件と一緒に外部benchmark reportへ記載し、
製品coreの合格条件にはしない。

---

## 16. stage別の詳細実装順

### Stage 0：schemaと小型正解

実装順：

1. `solver/`をuv projectとしてbootstrapし、Python pinと`pyproject.toml`を作成後、初回だけ`uv lock`で
   `uv.lock`を生成する。以後は`uv sync --locked`を使い、Ruff、import-linter、Pyrefly、Radon、pytestの
   直接commandを固定
2. `case_format.py`のv1 writer/reader/hash
3. `case.py`の最小YAML parseと静的検査
4. C01～C10のcase generatorとanalytic expected data
5. `coordinates.py`のXY/RZ規則
6. `docs/case_format_v2.md`、`physics_models.md`、`numerics.md`

出口条件：初回`uv lock`で生成したlockに対して`uv sync --locked`と五つの品質gateがclean checkoutで動き、
旧solverもCOMSOLも使わず、
case writer→loader→expected data読込みが再現できる。未決定のmodel係数はsample caseへ明記し、global
defaultにしない。

### Stage 1A：正しい決定論vertical slice

実装を水平module単位ではなく、次の縦切り順に進める。

1. P03補正：conditioning-aware P1/Q1 locationと有限provisional
2. table source＋SoA ballistic＋final/release event＋明示時刻frame＋最小lazy ResultView
3. 大域topology audit＋line boundary BVH＋ballistic first hit＋stick/escape
4. required field全domain certificate＋XY node field＋fixed charge＋RK4＋Epstein/electric/gravityの
   kernel/physics verification（完了。一般`rk4_reintegrated`の連続support証明は項目6で追加）
5. 厳密一様場から証明した一定加速度pathを、topology-completeな材料boundary、またはboundaryなし・全cell
   supportedな`RegularLayout`の解析的座標極値検査と組み合わせて解禁（完了）
6. revision 3a：boundaryless Cartesian XY、fixed charge、全cell supportedな`RegularLayout`について、
   非一様場・dragのglobal field/model boundから、全短縮RK4評価を含む連続support enclosureとEpsteinの
   連続applicabilityを証明する。証明不能caseはhidden subdivisionせずfail-closedにする（完了）
7. revision 3b：provisional finite値とvalidity flag、geometry-drivenなsequential dyadic RK4 piece、各pieceの
   離散path tube、accepted-piece replay、event-before-validity、証明不能時のfail-closedを実装する。
   同一macro proposalのparameter区間分割は使わない。当該revisionではunstructured supportを別の包含証明まで、
   RZ force couplingとStokes–Cunninghamをbasis/axis規則、field schema、適用域、独立oracleの確定まで拒否した
   （Cartesian XY・fully-supported regular field・terminal stick/escapeの範囲で完了）。engine v7では数値意味を
   変えず、候補粒子のrefinement proposalを256件chunkのdeterministic wavefront batchへ置換した。
8. P06-U：exact-mesh fully-supported P1/Q1とtopology-complete material domainの一般RK4（完了）
9. P07：counter RNG、specular/probabilistic stick、surface release、RZ measure、RZ axis path split
   （engine v11 / event v7でexact linear/quadraticに加え、Cartesian XY一般RK4の厳密内向き
   surface departureとsingle-facet active-boundary residualを実装。richer distribution、moving wallは未解禁）
10. P06-RZ：P07のaxis意味論を使うforce-coupled RZ basis/axis gate
    （engine v12 / event v8 / physics catalog v2 / required field v3として完了。schema、proposal、enclosureは不変）
11. P06-S：小さいphysics runtimeへ既存drag責務を集約後、Stokes–Cunninghamの
    schema/applicability/oracleをXY/RZへ追加（engine v13 / physics catalog v3 / runtime v1として完了）
12. P08：公開三API、薄いCLI、particle-local failure、series/probeによるStage 1A closure
    （engine v14 / result algorithm v2として完了）
13. P09：HDF5 payload展開前のmetadata memory gate、固定ID対応のresident stateとresident-row active index、上限付き
    microtile scratch、phase別memory plan、fresh/warm RSS/semantic harnessと代表規模characterization
    （engine v15 / CPU runtime layout v1 / memory plan v1として完了）

各増分は前のend-to-end caseを残したまま一つの能力だけ追加する。別engine、別output、別case formatを作らない。

出口条件：C01～C10をStage 1Aのmethodで実行し、RK4収束、公開API scenarioが合格して、主要な
failure reasonがresultへ残る。C03の一定係数exponential midpoint exactnessはStage 1B/P11で追加済みである。

### Stage 1B：高速CPU製品経路

1. memory planner、stable active list、bounded microtile scratchを既存SoA engineへ適用（P09で完了）
2. field/physics/RK4 tile kernelをNumba化し、endpointだけでなく`state_at`、wall hit、residual piece、
   output schedule不変性までreferenceと比較。regular supported-containing common pathとP1/Q1
   strict-interior hint/full-searchを実装し、実際に消費するhintだけをresident化（P10、engine v16として完了。
   outside/masked regular provisionalとP1/Q1 miss/initialはcompiled full scan。BVHはP14のprofileへ延期）
3. exponential midpointのreference、analytic verification、compiled実装とreference parity
   （P11、engine v17 / proposal v4 / event v9 / physics runtime v3として完了）
4. geometry BVH queryのevent-heavy batch化とworker-local residual/event/statistics buffer
   （P12、engine v18 / compiled tile v3 / geometry v3として完了）
5. stable multithread mergeとthreads 1/2/4のbitwise determinism test（P12として完了）
6. bounded writer queueとepoch segment（P13、engine v19 / result v3 / memory plan v4として完了）
7. A/B checkpoint、LATEST、resume、failure injection（P13として完了）
8. 10^4/10^5/10^6 benchmarkとprofile（P14として完了）
9. profileで確認したbottleneckだけを最適化（P14として完了）

出口条件：主要modelがcompiled pathを通り、referenceとの意味論、thread/output independence、memory plan、
checkpoint、synthetic performance matrixが合格する。P14はこのbaselineを完了したが、主用途を結合した性能・精度は
P14-Pでparallel runtimeを閉じた後、P14-Uで判定する。

### P14-P：serial runtime convergence（完了）

1. P14の文書化済みmatrix・profile・RSSと当時の336件を履歴上の移行根拠にする
2. `cpu.py`でreusable tile slabを所有し、field/physics/integratorを
   preallocated `into` passへ変更する
3. boundary BVHをstackless skip traversalへ変更し、per-rowのBVH-sized stack allocationをなくす
4. exact/curved event、residual、boundary response、Philox drawをrow target time付きflat SoA wavefrontへ移し、
   deterministic count/prefix/fillとstable compactionでsimultaneous facet・event・failureをcolumnar化する
5. accepted state/hintだけをcommitし、frame/probe/series、checkpoint/resume、writer backpressureを同じengineで維持する
6. 削除済みの`ThreadPoolExecutor`、future/wave merge、worker別scratch/stagingを復活さず、完了changeまでに
   Python粒子別event state machineとその旧memory/config/test記述を削除する
7. 詳細gateを実行し、一度の限定修正後も未達だったため`resources.threads`、thread mask、parallel-only
   test/harnessを削除し、case schema v2とsingle-thread compiled engineへ収束する

この作業は第二backend、experimental flag、一般scheduler frameworkを作らない。field→physics→stage融合はprofileで
配列往復が支配的と確認した箇所だけに限定し、model×layoutのkernelを増殖させない。詳細な順序、benchmark matrix、
速度・memory・科学同一性閾値は
[`solver/docs/parallel_execution_plan.md`](solver/docs/parallel_execution_plan.md)を権威とする。

手順2～6をengine v26へ統合後、focused correctionを測定した。regular 1Mは1/2/4 threadで
9.32/10.09/10.10 s、4-thread speedup 0.923xで、1-threadもv20履歴比23.7%退行した。microkernelは約3.75xでも
proposal/enclosureの直列調停が支配したため、巨大融合kernelを追加せず手順7の削除条件を実行した。
現行engine v37 / compiled tile v18 / proposal v10 / event v16 / runtime layout v6 / memory plan v14 / geometry v5は
single-thread compiled runtimeである。field/physics/integratorはpreallocated workspaceを使い、exact/curved/
axis/residualはflat SoA wavefront、boundary/Philoxとsurface releaseはcompiled batch、failureはrow numerical status、
frame/probeはdirect columnar replay、event/failureはbounded SoA/CSR stagingを使う。outer pool、worker別scratch、
即ack writer、thread mask、`_AcceptedPiece` / `_StateJump`、pending event/failure objectは残していない。
v27直列化後の同じregular 1M・4 macro-stepを独立process、各1回warm-upで3回測定したsimulate時間は
10.249/10.273/10.100 s、median 10.249 sだった。これはv26の1-thread観測より約10.0%、v20履歴値より約36.0%
遅かった。P14-Uはこのsynthetic値を解消する工程ではなく、結合した代表用途を別に測る工程として完了した。
どちらの値も並列runtimeを戻す根拠にはしない。

### P14-U：代表用途のutility/performance gate

1. 現行modelだけで、surface release、強い非一様場、材料wall、多数macro stepを同時に使う代表caseを一つ作る
2. 固定した64 x 64空間mesh上で`h,h/2,h/4`の時間同期位置・速度・hit時刻/位置/facetを自己収束と
   独立fine referenceで確認し、時間誤差へmesh変更を混ぜない
3. `nx=ny`の固定aspectでregular/P1/Q1を同時細分化し、各layoutの独立fine reference、物理target、
   normalized facet clearanceを確認する
4. sourceと局在済みhitを含むdense pathとglobal field extremaの比、refinement深度、failure率、candidate数を
   外部harnessで記録する
5. 10k/100k/1Mのnone/sampleを各3 fresh processで測り、raw値とmedian、RSS、solver plan、mode間のcore
   payload identityと各mode内のprobe identityを保存する。1M noneのprofileは別の非計時runとする
6. 可変なaxis-regular fieldを横切るRZ caseで自己収束を確認し、実際に読んだ全required canonical fieldと
   standard gravityからscalar/axial parity、radial zero不足を検出する
7. surface-source結論は実測したCartesian XY `line_length`に限定する。RZ `revolved_area`や任意の表面実現値を
   検証済みとせず、このgateだけを理由に`realized_surface_table`を追加しない
8. `point_wall_laws_v4`の完全鏡面`specular`／係数付き`restitution`分離と、全RZ domainのstandard
   gravity `g_r=0`はP14-Uの性能値から推論せず、先行するmodel revisionとして参照する

P14-Uは新しい診断frameworkではなく、既存public API/resultを読む一つのperformance/V&V sliceである。parallel
runtimeの再設計はP14-Pで完結させ、P14-Uへ持ち越さない。profileが単一の責務内bottleneckを示した場合だけ
layout/BVH所有の局所boundまたは限定的step controlを独立変更として検討する。opaqueなruntime/dependencyが
最大ownerの場合はowner固有の最適化結論を出さない。multiprocessing、第二scheduler、GPUで未解決の数値費用を
隠さない。

P14-Uの正式releaseは`solver/evidence/v0.1/p14u_release_v1.json`へ保存した。18 raw観測、6 median、別1M `none`
profileを保存し、XY時間/mesh収束、RZ収束/parity、失敗0、出力utility、科学payload・revision・event workの
identityを確認した。1M median `simulate`は`none` 550.43 s、sample 533.95 s、raw peak RSS最大767.3 MiB、
solver-owned plan 614.5 MiBだった。profile self-timeはevents 28.7%、fields 26.8%ほかへ分散しており、
単一owner支配を示さないためproduction変更を行わない。秒数は当該machineだけの非gating値で、COMSOL比や
portable性能ではない。RSSは公開`load_case`/`simulate`/`open_result`までのprocess high-waterであり、
後続の外部検証読込みを含まない。`none`/sampleは順次実行なので差をwriter単体costとしない。実測したsourceは
Cartesian XY `line_length`に限り、RZ `revolved_area`の分布品質は未主張である。再実行可能なrelease evidenceの
保存はP14-Rが所有する。

### P14-RとT03：配布可能なv0.1 closure

1. P14-R保存対象production v27のP14 18条件×3反復＝54 raw observationと、P14-Uの18 raw／6 median／別1M profileについて、
   machine fingerprint、実行条件、semantic digestを再実行可能なrelease evidenceとして保存する。旧v20の69 raw
   artifactは現存しないため、文書化済みsummaryだけを履歴として保持し、現行結果から再構成しない
2. Windows/Linux・Python 3.12で`uv sync --locked`、標準品質gate、wheel build、clean install、三公開API smoke
3. `ResultView`へboundary eventのbounded batch iteratorを追加し、既存一括readerは同じprimitiveから構成
4. `tools/analysis`と`tools/visualization`へ、fate/deposition/arrival集計、代表trajectory、event可視化だけを実装
5. v0.1 surface source範囲を現行のuniform/edge-fraction、fixed/normal velocity、fixed releaseへ固定

P14-R/T03はsolverの物理・event意味を変えない。query DSL、汎用dataframe layer、第二reader、巨大report frameworkは
追加しない。T01最小case builderとT02のproducer adapter/V&Vはcore外の独立trackであり、実producer workflowを
製品として主張する前に閉じる。Git baselineの作成はrepository運用者が行い、solver codeから自動commitしない。

2026-09-29時点で、`solver/evidence/v0.1/`へP14-R baseline v27の54 raw observation、P14-U、Windows/WSL2の
platform smokeを保存した。v27 matrixは18条件×3反復、全repeat/revision/science-key identity、memory-plan fitを満たし、
regular 10k→1Mの`simulate` log slopeは0.9864だった。絶対秒数は引き続きmachine-localかつnon-gatingである。
2026-10-05に新workflowの初回remote実行がWindows/Linuxとも成功し、receipt固定のtested head/lockで620件、
性能smoke 7行（cold 1＋warm 6）、wheel、runtime-only clean install、三公開API smokeを完了した。
receiptは`solver/evidence/v0.1/release_remote_ci_v1.json`であり、T03、P14-R、`0.1.0.dev0`開発baselineの
配布可能性closureは完了した。これは正式版packageの公開を意味しない。

### Stage 2A：電荷・熱・軸対称場中の3D粒子

Stage 2Aを一括実装せず、次の三つの縦切りにする。

#### P15：continuous charge

物理・数値decisionとproduction実装は完了した。選択modelは
`oml_stationary_maxwellian_debye_huckel_v1`であり、stationary Maxwellian OML収集rateと
Debye–Hückel表面電位、ion drift / `a / lambda_D`適用gate、有限charge invariant、
rate/derivative bound、`hL <= 0.5`のexplicit stiffness gateを一組として扱う。P15着手を
P14-Rのexact Git baselineと初回remote CI成功まで禁止していた旧順序は、ユーザーの明示指示で解除した。
P14-Rのremote CIは後に完了した。旧順序の解除とP15の実装判断を遡って変更しない。

最初のproduction changeは次のRK4縦切りとして受け入れた。

1. `physics/charge.py`へscalar rate、平衡残差、有限invariant、`|R_Z|` / `L` boundを一経路で実装した。
   `physics/catalog.py`はmodel ID、required field、parameter、座標・適用域を解決し、
   `physics/compiled.py`は同じ式のcompiled evaluatorだけを所有する
2. `has_force`、`evolves_continuous_state`、`requires_stage_evaluation`を分離し、charge-only caseも既存stage evaluatorへ通した。
   `integrators.py`が既存RK4の4 stageで`(x,v,Z)`を連成し、`engine.py`はdispatch、fail-closed gate、
   revision記録だけを調停する
3. charge-aware state/path enclosureとelectric acceleration boundを追加し、動的電荷では
   linear/quadratic exact specializationを無効化する。各accepted intervalで`hL <= 0.5`を証明できないcaseは
   implicit solve、equilibrium置換、charge-only subcycleへfallbackせず拒否する
4. verificationは、rate符号と平衡残差、平衡両側の単調性、有限invariant/bound、driftと
   `a / lambda_D`の適用域拒否、`hL`境界、smooth coupled `(x,v,Z)`のRK4次数3.5以上、
   reference/compiled parity、wall event、slab幅・output schedule・checkpoint-resume identityとする
5. genericなmodel parameter mappingと既存charge state/result/checkpoint列を再利用し、case/result/checkpoint/event
   schemaは変更しなかった。XY/RZ、wall、output、checkpointも既存経路を再利用する

RK4縦切りの受入後に、同じ予測midpoint時刻で場・運動・電荷を評価する非stiff explicit midpointを
第二のproduction sliceとして追加し、可変係数次数1.8以上とRK4 sliceと同じparity/identityを確認して受け入れた。
動的電荷はexact specializationを使わず、精度目的のhidden subdivisionも追加していない。
bounded dyadicまたはL安定法は、両explicit sliceの剛性・性能測定で必要性が確認された場合だけ別decisionで追加する。

#### M3-V：外部target applicability/relevance＋matched companion（完了）

M3-Vは`solver/tools/vv/comsol/`だけが所有し、core testまたはruntime dependencyにしない。COMSOL 6.4で二つの
MPHを`loadCopy`・`-nosave`で直接監査し、前後hash不変、12 packageの履歴/provenance、保存時刻全体のvariant感度を
確認した。保存primitiveからCOMSOLのEpstein係数とrelative-drift regularized two-current charge rateを再構成でき、
式のparityはそれぞれ約`1.1e-15`、`1.6e-12`以下である。一方、P15 stationary OMLは全12 caseでion drift gateを
満たさず、Case Pは負イオンを含み、局所有効正イオン質量も単一scalar mass契約へ一致しない。linear Epsteinも
粒径/状態により部分的に適用外である。したがって元の12 packageに対するproduction全軌道
比較は`NOT_APPLICABLE`とし、閾値緩和やCOMSOL専用分岐は追加しない。boundary、stochastic distribution、独立builder
parity、Cartesian 3-Dは`NOT_TESTED`のまま明示する。
reference charge式の保存点上の局所`h|dR/dZ|`は最大約`0.721`だが、これは記録された10 us刻みに対する
frozen-state characterizationであり、continuous-pathまたはintegrator安定性の証明には使わない。

後続は単一の全体順位へ潰さず、次の独立workstreamで進める。

1. field production：F01で独立reduced electrostatic builderをcanonical writerへ接続し、F02で
   provider adapterのmixed triangle/quad→P1変換、代表規模linear-solve gate、代表入力、外部field V&Vまで完了した。
2. trajectory physics：reference式を普遍化せず、species制約を明示したrelative-drift charge、P15-E有限速度Epstein、
   versioned relative-flow ion drag、P16 Waldmann--Gallis、Brownian B02までを完了した。続く外部M3-Vでは、
   Brownianを無効化した共通力・共通canonical P1場のpre-event sliceについて時間刻み収束と全時系列parityを完了した。
   stochastic評価は単一path比較へ混ぜず、multi-seed ensemble gateとして残す。
3. state dimension：P17 Cartesian 3-Dを上記model追加から独立して進める。

#### P15-D：species-constrained relative-drift charge（完了）

M3-Vの外部式を互換実装せず、電子＋単一・単価正イオン、非正表面電位に限定した
`oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1`を独立revisionとして追加した。

1. `physics/charge.py`へshifted-Maxwellian ion moment、負電位rate/derivative、`[Z_min,0]` invariantと
   run-wide boundを追加した。zero-driftはstationary OMLへ一致し、小driftは解析級数で評価する
2. catalogは既存continuous mappingに、このrevisionだけ必須の有限正値`maximum_ion_drift_ratio`を追加した。
   比は`|u_i-v|/sqrt(8 k_B T_i/(pi m_i))`で、clampやfitting parameterとして使わない
3. runtime/compiled passの一つのmodel branchへ接続し、actual stageとcontinuous pathのdrift、Debye比、
   非正平衡をfail-closedで検査した。integrator、event、geometry、writer、checkpointは変更しなかった
4. 独立3-D Maxwell速度quadrature、zero-drift parity、rate微分、primitive corner bound、compiled/reference parity、
   RK4/explicit-midpoint材料wall、frame identity、XY/RZ parity、64-step checkpoint/resumeを受け入れた
5. catalog v6、physics runtime v5、compiled tile v8へ更新し、engine v28、proposal/enclosure、memory plan、
   case/result/checkpoint schemaを維持した

未対応は負イオン、複数正イオン種、正電位、emission、collisional/magnetized charging、sheath内surface releaseである。
現行continuous-path gateは成分絶対上界によりco-flowを偽拒否し得るが、安全性を局所的に緩めない。必要ならRZを含む
符号付きvelocity interval enclosureを別revisionで扱う。

#### P15-E：finite-speed Epstein（完了）

`epstein_finite_speed_maxwell_mixed_equal_temperature_v1`を独立revisionとして追加した。鏡面反射と完全熱適応・
等温拡散再放出の混合率、有限正値の速度比上限、`lambda/a>=10`を明示し、低速linear modelやStokesへ
自動切替しない。小速度級数、全速度閉形式、rateと速度Jacobianの別global boundを同じforce ownerに置いた。
独立分子速度積分、低速・高速極限、compiled parity、両積分器の公開収束caseを受け入れ、catalog v7、
runtime v6、compiled tile v9へ更新した。engine、integrator、event、memory、schemaは変更していない。

この時点の後続だったP15-F ion dragとP16 Waldmann--Gallisは続くsliceで完了し、その時点で次だったBrownianも
B02で完了した。
P17はstate-dimension独立workstreamのままとする。

#### P15-F：versioned collisionless ion drag（完了）

最初のrevisionを`barnes_collisionless_effective_speed_single_positive_ion_negative_debye_huckel_v1`へ固定した。単一・単価正イオン、
非正粒子電位、Debye--Huckel screening、collisionless・非磁化backgroundだけを扱い、collection＋orbital項を
イオン相対流方向へ加える。`ion_drag`は`explicit_acceleration` categoryのまま、neutral dragのrelaxationへ
混ぜない。必要primitive、連続path適用域、impact-parameter積分oracle、global加速度boundは同じmodel ownerへ置く。
外部datasetのfloor、clamp、image補正、電場方向化、scaleは実装しない。catalog/runtime/compiledの既存単一passへ
最小統合し、両積分器の収束、XY/RZ basis、compiled parity、無効時性能を受入条件とする。
impact-parameter quadrature、zero-flow/neutral limit、global bound、fixed/continuous chargeの同一stage結合、
RK4 3.5次以上、explicit midpoint 1.8次以上、XY/RZ parityを受け入れた。catalog v8、runtime v7、
compiled tile v10へ更新し、engine v28、proposal/enclosure、runtime layout、memory plan、schemaは維持した。

#### F02：provider adapter / representative integration（完了）

F02は`solver/tools/comsol_adapter/`だけにCOMSOL CSVのsyntax、外部entity ID、Q1 node順を閉じ込め、指定domainの
triangle/quadを一度だけtriangle-only canonical P1へ変換する。外周とownerはvolume-cell incidenceから再構築し、
axis seamを明示除外して残る全facetをsemantic groupへ過不足なく割り当てる。新規exportはtopology ID結合を必須とし、
参照CSVのようにfield node IDがないlegacy入力だけは、明示tolerance内でexactly-oneかつ全体が全単射となる座標対応を
証明した場合に限って転記する。NaN補間・最近傍埋め・solver側mixed-mesh分岐は追加しない。軸上radial gas velocityの0投影は
設定上限を超えれば拒否し、補正量をprovenanceへ保存する。

代表Case-A 100 nm packageでは、1,987 node、2,127 source triangle、826 source quadから3,779 P1 triangleを生成し、
193外周edge中33 axis edgeを除いた160物理facetを6 groupへ再構築した。quad splitは品質最大化とexternal-ID tie-breakで
決定し、対角別件数561/265、最小triangle品質`0.0423878260`だった。同一入力の再生成content hashは一致した。

F01 builderは1,826 free nodeを10 continuation rampで解き、総linear iteration 2,821、最終relative residual
`1.8441e-13`、global charge-balance error `5.8498e-21 C`となった。GMRES basis＋Hessenbergの設定上storageは
1,235,088 B、同一machineのsolve-only観測は0.689/0.737 sである。現解法は代表規模gateを満たすため、fallback、
反復上限増加、第二linear solverは追加しない。

完成fieldを既存三APIへ渡すfixed-charge＋Coulomb smokeは、wafer surfaceから32粒子を放出し、10 macro step、
3 frame/96 row、failure 0で完了した。20 usの最大変位は`2.0715e-5 m`、最大速度変化は`7.1474e-2 m/s`で、
全粒子のchargeは-1のまま保たれた。このrunはfield-production接続の検査であり、COMSOL軌道、wall、Freeze parityを
主張しない。`gas_inlet: escape`は未到達の通常境界設定で、COMSOL Freezeの写像ではない。

外部field比較は同じexport nodeでの記述比較に限定する。axisymmetric lumped-volume normでpotential/Eのrelative L2は
約2.00%/17.05%だが、合否閾値には使わない。単一reference meshのため独立mesh convergenceは`NOT_TESTED`であり、
Q1→P1化、COMSOL側recovery/smoothing、closure/離散化差を分離できない。比較差を減らすためにbuilderやcoreを
分岐させない。小さい再実行証跡は`solver/evidence/f02/`、比較実装は`solver/tools/vv/comsol/`が所有する。

#### P16：Waldmann--Gallis thermophoresis（完了）

`thermophoresis / waldmann_gallis / waldmann_gallis_free_molecular_single_species_heat_flux_v1`を、既存の
`explicit_acceleration`合成へ追加した。入力は中性気体速度、並進温度、局所質量平均座標の並進伝導熱流束、
平均自由行程、単一分子質量である。半径は`drag_diameter_m/2`、慣性は`mass_kg`をauthorityとし、coreは
温度gradientやFourier熱流束を再構成しない。

`physics/forces.py`が式、`lambda/a>=10`とrelative-drift比0.1以下の適用域、componentwise global boundを所有する。
catalogはrequired fieldとneutral-gas authorityを一意化し、Epstein併用時はvelocity/temperature/mean-free-path/
molecular-massの完全一致を要求する。適用域が重ならないStokes--Cunninghamとの併用は拒否する。runtime/compiled passは
実stageで同じfield sampleから評価し、RK4とexponential midpoint、XY/RZ、event/output/checkpointの既存経路を変えない。

一次Chapman--Enskog分布の3-D Gauss--Hermite momentを独立oracleとし、係数、方向、`a^2/m`・`T^-1/2` scaling、
非有限入力と局所/path適用域、global bound、reference/compiled parityを検証した。affine熱流束の公開caseではRK4が
3.5次以上、exponential midpointが1.8次以上で収束し、XY/RZ parityも合格した。変更revisionはcatalog v9、
runtime v8、compiled tile v11だけで、engine v28、integrator/proposal/enclosure、event、memory plan、
case/result/checkpoint schemaは不変である。Talbot/continuum、mixture、near-wall、negative thermophoresis、
accommodation fittingへの自動切替は追加しない。この時点の次工程だったBrownianはB02で完了し、P17は独立
workstreamのままとする。

#### P17：軸対称場中のCartesian 3-D粒子

XYZ particle＋RZ field mapping、方位回転共変性、revolved boundary first hit、3-D normal、schema/output/memoryを
一つの縦切りで追加する。今から一般可変次元frameworkへ書き換えない。

### Stage 2B：Brownian

1. joint `(x,v)` OU analytic update（B01完了、B02でproduction接続）
2. physical interval-tree normal drawとbinary conditional split（B01完了）
3. geometry/outputと独立な固定depth dyadic nodeを生成し、leaf endpointの位置・速度からcubic Hermite
   numerical pathを一意に定める
4. stochastic pathを既存event interfaceへ接続し、planar first-passage、mean/covariance/MSD、frame/probe、
   checkpoint-resumeを同じ縦切りで閉じる。最初はCartesian XY、Epstein linear drag、fixed charge、terminal
   stick/escapeだけを受理し、他の決定論力と反射はsilent omissionせず拒否する
5. 一般wallで固定depthを増やした時のfirst-passage分布収束を確認する。RMS距離だけをclear certificateにしない
6. ensemble outputとconfidence analysis

一般曲面で連続OU pathのexact first passageを主張しない。初期revisionのauthorityは有限depthのHermite numerical
pathである。adaptive clearは全runのmiss-probability budgetを明示できる場合だけ別revisionで追加する。
B02は上記1～5を完了したproduction capabilityである。engine v30では`gamma*h` gateだけに頼らず、root covariance、
conditional split、deterministic meanのfloat64表現可能性を実際に検査する。通常はvector batchを維持し、例外時だけ
row別に局在して不良粒子を`nonfinite_physics`で停止し、正常粒子を継続する。ensemble confidence analysisと
COMSOL比較はcore外のV&Vに留め、連続OU first-passageやCOMSOL同精度を根拠なしに主張しない。

### Stage 3：比較可能なoptional plasma physicsと外部V&V

Stage 3の目的を、現在のCOMSOL 12 packageを唯一の正解にすることではなく、そこで選択されている物理を
producer非依存のoptional modelとして設定し、各寄与と全軌道を外部V&Vで比較可能にすることへ具体化する。
Case P/A、COMSOL feature tag、dataset directory名はcore設定に入れない。field producerは
`imported_external_plasma_fields | reduced_electrostatic`を選び、その後のtrajectory physicsは同じcategoryと
revisionで選ぶ。

#### Stage 3 capability matrix

| 比較対象 | 現在 | Stage 3出口 |
|---|---|---|
| RZ meridional motion、287 source、121 frame | 実装済み。保存COMSOLはfixed RK4 10 us、candidateは独立step authority | candidate自身の`h,h/2,h/4`自己収束を主要gateとし、10 usの模倣を要求しない |
| external plasma field / reduced electrostatic field | F01/F02完了。現M3-C1 candidateはexported exact-connectivity P1、参照側だけがCOMSOL native field | exported-P1対native-fieldのworkflow差と、必要時だけのcommon-field parityを別成果物・別statusにする |
| electric、Epstein、gravity/buoyancy | 実装済み | frozen RHSと全軌道で再確認 |
| Waldmann thermophoresis | P16とP18-R effective-gas sensitivityを実装済み。M3-C1の最小PPR補足exportで13,202 saved rowのproducer-form replayを閉じた | producer-owned heat-flux authorityを維持する。saved-row式一致をcontinuous-path applicabilityや統合軌道一致へ昇格しない |
| neutral drag / mixture thermophoresis applicability | P18-R監査で既存P15-E/P16は12/12 case `NOT_APPLICABLE` | producer認証済みpseudogas用のlinear Epstein / heat-flux thermophoresis sensitivityを設定可能。truth認定ではない |
| aggregate dynamic charge | P18-Cの明示comparison revisionまで実装済み | 保存式再生と厳密provider一致を外部V&Vの別gateとして維持 |
| ion drag二式 | P18-Iの二つの独立revisionまで実装済み | 保存式再生とproduction式差を外部V&Vの別gateとして維持する |
| DEP | P18-Dの準静的球modelを実装済み | producer由来gradientのformula/trajectory parityを外部V&Vの別gateとして維持 |
| documented free-molecular lift sensitivity | P18-L実装済み | RZ/no-swirl限定の感度revisionを維持し、COMSOL軌道一致はM3-C1で判定 |
| Brownian＋全決定論力＋continuous charge | B03の明示RZ投影を単一stochastic proposalへ実装済み | 段階的決定論matrixとM3-C2A Case-A/Case-P 100 nm multi-seed外部統計比較まで完了。追加coverageは独立work package |
| inlet Freeze | 外部M3-C0正例でstatus 2、hit点R-Z保持、衝突前velocity保持を確認。stick/escapeとは異なる | producer非依存の最小`hold/held`をP18-Hで追加し、解析回帰と同じcandidate microcase 15/15を完了 |

#### M3-C0：比較入力と境界意味の確定（core変更なし）

1. COMSOLは監査済みMPHのcopyを`loadCopy`し、原本へsaveしない。前後SHA-256、COMSOL version、feature inventoryを
   receiptへ残す。
2. boundary 37 Freezeとboundary 35 Disappearへ確実に到達する力なしのnormal-impact正例を別model copyで作り、
   hit前後の位置・速度・status・event時刻を細かい時刻で出力する。2 scenario×3刻みのexact 6 configuration receiptを
   process logからfail-closedで照合し、全active frameを`x=x0+v0*t` / `v=v0`で検証する。証拠なしにFreezeをstickへ
   読み替えない。
3. Case-A 100 nmのcommon-P1軌道を最初の自然なmaterial-stick eventまで延長し、event前same-field agreementと
   material-stick意味を一つのanchorで閉じる。そのanchorを固定した後、native-field/exported-P1差を
   field/RHS/trajectoryの最初の層へ局在化する。
4. candidateは各対象caseで自身のaccuracy・stability・applicability条件から`h,h/2,h/4`を事前登録し、自己収束を主要な
   数値gateとする。保存COMSOLのfixed stepをcandidateへ強制しない。primitive field、各力、`dZ/dt`、受理stepまたは
   RK stage、時間付き状態、疎なboundary eventを外部比較するのは、両側の物理・field・boundary・stochastic意味を
   同じにできたcaseだけとする。broader COMSOL campaignはcharge-stable couplingとdurable I/O cadenceの後段へ置く。
5. DEP用`grad(mean_E_squared)`、lift用方位vorticity、aggregate charge/ion-drag用の正イオン密度・速度・有効質量・温度、
   screening長のproducer authorityと単位をmanifestへ固定する。節点Eや速度をtrajectory coreで数値微分しない。
6. 二つのion-drag variant間で、ion drag以外の式、field、初期条件、boundary、solver設定が同一であることをhashと
   normalized expressionでgateする。既知のlift構文差を残したrunからion-drag感度を帰属しない。
7. Brownian-on比較用には12 packageごとにseedを変えた32 replica、または事前登録した同等のconfidence budgetを
   用意する。既存の各条件1 saved realizationと設定seed parameter値は記述用だけとし、実効seedまたは粒子ID軌道を
   正解にしない。

M3-C0が欠けても各modelの解析的実装は進められるが、COMSOL一致または差異原因の主張は進めない。既存CSVを
不足列の推測、最近傍補完、保存時刻からのevent逆算で補わない。

2026-10-01にM3-C0a offline reference lockを完了した。`tools/vv/comsol/lock_m3c0_reference.py`は本体をimportせず、
2個のMPH identity、12 packageの287粒子×121時刻、式・parameter・必要field名/単位、全入力hash、候補刻み、
package別32 seedを新しい外部成果物へ固定する。既存履歴がBrownian-onの各条件1 saved realizationで実効seed未証明であること、RK stageとFreeze正例が
ないこと、derived-fieldのsolution/平均化/回復/topology provenanceが不足することを`NOT_TESTED`のまま保持した。
またCase Aはion dragだけが相違する一方、Case Pにはlift式差と独立再export由来の非同一field payloadが残るため、
ion-drag-only gateを`FAIL`とした。よってM3-C0全体は進行中であり、この時点の次作業は共通の監査済みtheory MPH copyをbaseに
代替ion-drag式だけを注入する専用no-clobber runner/exporterと、Brownian-off pilot、正例boundary microcaseである。

2026-10-01にM3-C0bのCase-A 100 nm deterministic pilotを外部実行した。最初の30 ms v3はBrownian/Saffmanを
無効にし、dynamic chargeと7個の決定論寄与を有効にした287粒子×121時刻の10/5/2.5 us系列である。原本MPHは
`loadCopy`/`-nosave`の前後で同一SHA-256だったが、観測次数は位置`0.659`、速度`0.951`、電荷`0.241`であり、
2.5 usをreferenceへ昇格しなかった。全粒子がactiveな0--450 usだけでも旧系列は概ね1次だったため、境界eventだけを
原因とはしない。

このM3-C0b v3はBrownianを明示的に無効化して作った履歴上のdeterministic companionであり、現在の`model_dataset`に
保存されたBrownian-on・fixed RK4 10 usの100 nm Case A/P原軌道、またはBrownian-off candidate v3とは別成果物である。

続くpre-event v4は事前登録した位置・速度・電荷のfine-pair上限を満たしたが、電荷次数`0.716 < 0.75`でFAILとし、
結果後に閾値を緩めなかった。計画済みの0.3125 usだけを加えたv5は当時全7 gateをPASSと記録したが、後続監査で
位置relative L2が絶対RZ座標を分母として原点依存だったことが判明した。このためv5の旧PASS/admissionを
`INVALIDATED`とし、v5そのものは履歴raw artifactを保持した`CHARACTERIZED`とする。

未見の0.15625 us結果を使う逐次確認v6は0.625/0.3125/0.15625 usの各runで13,202 recordすべてがactiveだった。
0.3125→0.15625 usの原点不変な変位relative L2、速度relative L2、電荷relative L2は
`3.102727085428027e-5`、`3.92483251084038e-5`、`1.3511393490811483e-6`、位置・速度・電荷の観測次数は
`0.9041136/0.944312/1.123838`である。v6のPASSはこの一caseのpre-eventにおける運用上の刻み選択だけを確認し、
solver agreement、普遍的な物理精度、COMSOLの形式RK4次数、30 ms event収束、12 case全体、Freeze、Brownianを
認定しない。小型証跡は`solver/evidence/m3c0/deterministic_pilot_v6/`に置き、v5は履歴証拠として変更しない。

#### P18-C：aggregate relative-drift continuous charge

`charge / plasma_continuous / aggregate_relative_drift_regularized_two_current_v1`を追加する。これはP15/P15-Dの
置換ではなく、正負表面電位branch、相対drift、正則化速度、ion-energy floor、有限指数範囲を式の一部として明示する
optional revisionである。正イオン有効質量はproducerが出す正値canonical scalar field一つをauthorityとし、uniformな
producerも定数fieldを書く。parameter-or-fieldの分岐を増やさず、species-resolved modelとは呼ばない。

電子・正イオンthermal voltageと背景screening長も正値canonical scalar fieldを一つずつauthorityとし、revision内で
`lambda_eff=max(electrostatic_radius, screening_length)`を適用する。有限invariant/rate boundを構成するため、caseは
正則化前の`|u_i-v|`の有限正値`maximum_relative_ion_speed_m_s`を必須指定する。actual stageとcontinuous path enclosureで
超過を拒否し、速度clipや比較結果へのfitには使わない。

- owner：`physics/charge.py`がrate、derivative、finite invariant/bound、適用域を所有し、catalog/runtime/compiledは
  既存の一つのcontinuous-state passへ接続する
- 数値：RK4とexplicit midpointで`(x,v,Z)`を同じstage時刻に評価し、charge-only subcycle、平衡置換、clipによる
  repairを追加しない。revisionに宣言したfloor/指数範囲だけを物理式として適用する
- verification：両電位branchの連続性、rate/derivativeの独立高精度reference、保存primitive上のfrozen-rate parity、
  charge invariant、XY/RZ、compiled parity、時間刻み収束、event/checkpoint identity
- 非目標：負イオンを独立運動speciesとして解くこと、emission、collisional sheath、任意のCOMSOL式parser

2026-10-01にP18-Cを完了した。`physics/charge.py`へ純粋rate・導関数・有限invariant/boundを置き、catalog v11、
physics runtime v10、compiled tile v12から既存の一つのcontinuous-state passへ接続した。engine v30、proposal、event、
resident state、memory plan、case/result/checkpoint schemaは変更していない。独立Decimal oracle、全branch/floor/clamp、
finite bound、compiled parity、XY/RZ、RK4 4次・explicit midpoint 2次の時間収束、material event、checkpoint/resumeを
検証した。保存済み12 packageを使う外部frozen-rate評価では、export済み`phi1`による同式再生はPASS、coreのscreeningと
標準`epsilon0`から再構成する厳密provider一致は定数規約差によりFAILのまま保持した。数値と解釈のauthorityは
[`solver/evidence/p18c/README.md`](solver/evidence/p18c/README.md)とし、閾値緩和やdataset fitは行わない。この結果から
integrated chargeまたはtrajectory一致は主張しない。

#### P18-I：二つのselectable ion-drag revision

既存の`barnes_collisionless_effective_speed_single_positive_ion_negative_debye_huckel_v1`を維持し、次を別revisionとして
追加する。

1. `relative_flow_screened_collection_orbital_aggregate_ion_v1`：相対流方向、collection＋orbital、
   `min(lambda_D, lambda_in)` screening、documented regularizationを式として明示する。
2. `electric_field_directed_image_orbital_sensitivity_v1`：電場方向、documented image-form collection/orbital、
   zero-field方向正則化を持つ感度model。基本推奨modelまたはBarnes fallbackとはしない。

両者は`ion_drag` categoryで排他的に選択し、Case P/Aやvariant directory名をmodel IDにしない。`physics/forces.py`が
式・方向・適用域・global boundを所有し、catalogは必要primitiveを宣言、runtime/compiledは既存stage passへ一寄与として
接続する。独立scalar/vector reference、zero-force limit、方向、charge符号、保存primitive上の全成分parity、RK4/midpoint
収束、XY/RZ basisを受入条件とする。二式のblend、自動選択、差を小さくするparameter fittingは行わない。

実装着手前の最初の変更で、`solver/docs/decisions.md`へAGENTS所定の一表だけを追加する。そこで二式の正確な式と固定定数、
共有plasma primitiveのauthority、aggregate chargeとの整合条件、有限applicability/global bound、対応座標・積分器、独立reference、
memory/performance影響、置換しない既存経路を確定する。これが確定する前にcatalog keyやcompiled branchを先行追加しない。

2026-10-01にP18-Iを完了した。二式をcatalog v12、physics runtime v11、compiled tile v13から既存の一つの
stage passへ接続し、engine v30、proposal、event、resident state、memory plan、case/result/checkpoint schemaは変更していない。
独立Decimal oracle、min/max/floor/zero limit、global bound乱択包含、relative-flow continuous path gate、P18-Cとの
同一stage charge/background共有、compiled parity、XY/RZ、relative-flowのRK4 4次・explicit midpoint 2次、imageの
両積分器での定加速度関係、relative-flow材料eventを検証した。checkpointは新state/schemaを追加していないため、既存の
全payload checkpoint/resume identity gateを再利用した。
保存済み12 package・397,820 active rowの外部frozen-force評価はproducer保存式再生がPASS、production theory式は
`epsilon0`定数規約差と整合する差を分離してFAIL、image式はCase P/Aのion-speed authority差を
`DOCUMENTED_MODEL_DEFINITION_DIFFERENCE`として記録した。これは統合軌道精度や物理適用性の認定ではない。

#### P18-D：quasistatic spherical DEP

`dielectrophoresis / quasistatic_spherical_gradient_e2_v1`を追加し、
`F=2*pi*epsilon_m*a^3*K_CM*grad(mean_E_squared)`を`explicit_acceleration`として合成する。

- `electrostatic_radius_m`と`mass_kg`をauthorityとし、有限正値の媒質比誘電率と`[-0.5,1]`内の実数`K_CM`を一度だけ入力する。粒子誘電率との二重authorityを作らない
- producer認証済みの有限正値`maximum_point_dipole_radius_m`を必須にし、全particle radiusがこれ以下でなければprepare時に拒否する。認証方法と誤差基準はprovenanceへ残す
- `grad(mean_E_squared)`はproducerが単位`V^2/m^3`のcanonical vector fieldとして与え、そのDC/RF/RMS convention、solution、時間平均、gradient
  recoveryをprovenanceへ保存する。coreは節点Eを微分しない
- 一様Eで0、解析的quadratic potential、符号・`a^3/m` scaling、XY/RZ basis、compiled parity、両積分器収束を検証する
- 複素周波数依存Clausius--Mossotti、travelling-wave DEP、粒子相互作用は別revisionとする

2026-10-01にP18-Dを完了した。`dielectrophoresis` categoryへ
`quasistatic_spherical / quasistatic_spherical_gradient_e2_v1`を一つだけ追加し、catalog v13、physics runtime v12、
compiled tile v14から既存のadditive-acceleration stage passへ接続した。engine v30、proposal、event、resident state、
runtime layout、memory plan、case/result/checkpoint schemaは変更していない。

coreはunit/basisを検査済みの`gradient_mean_e_squared_field`だけを各stage位置でsampleし、Eの微分、DC/RF/RMS変換、
gradient recovery、point-dipole誤差認証を所有しない。prepareはproducer認証済み
`maximum_point_dipole_radius_m`に対して全particle radiusをfail-closedに検査し、field extremaと`abs(K_CM)*a^3/m`から
既存enclosureへ加えるcomponent boundを作る。B02 Brownianとの併用は既存の追加力禁止条件で明示拒否する。

独立scalar式、zero/sign、`a^3/m` scaling、2-D回転共変性、乱択global-bound包含、catalog/runtime/compiled parity、
XY/RZ field basisを
検証した。公開APIでは解析的調和振動子を使い、Cartesian XY/RK4で3.5次以上、axisymmetric RZ/explicit midpointで
1.8次以上を確認した。このP18-D完了時点では、保存COMSOL primitiveの式再生と統合軌道比較をcore受入へ混ぜず、
外部M3-C1まで`NOT_TESTED`としていた。後続common-P1複合sliceはPASSしたが、DEP単独・native-field parityと物理妥当性は
未認定である。100,000-rowの同一workspace warm評価はdisabled/enabled median
`22.94/23.83 ms`（1.039x、約3.9% overhead）、prepared boundは双方`1,600,000 B`だった。これはmachine-localな
非gating値で、詳細は[`solver/evidence/p18d/`](solver/evidence/p18d/)が所有する。

#### P18-L：documented rarefied-vorticity lift sensitivity

`lift / rarefied_vorticity_sensitivity_rz_v1`を追加した。これは一般Saffman liftではなく、現在のreferenceにある
`rho_g*lambda_g*a^2`、meridional relative velocity、方位vorticity、明示係数`C_L`から成る感度modelである。

- producerが`azimuthal_gas_vorticity_s_inv`を出力し、coreはgas velocityを微分しない
- 初回revisionは`axisymmetric_rz_meridional`、no-swirl、球形、one-way dilute、高Kn範囲だけを受理する
- zero vorticity/zero relative velocity、向き、`rho*lambda*a^2/m` scaling、解析的一様vorticity、compiled parity、
  RK4/midpoint収束を検証する
- 物理的推奨modelではなくsensitivity revisionであることをmanifestへ残し、連続体SaffmanやCartesian 3-Dへ流用しない

実装前reviewで、liftを従来のvelocity-independent external boundへ加えるだけでは指数中点法の材料経路包絡が
成立しないことを確認した。P18-Lでは既存static additive配列を第二経路として残さず、runtimeが
`particle_index`とcomponent-wise velocity上界を受けて**全非drag加速度**を返す一つのbound callbackへ置換する。
RK4は既存の全加速度callbackを使い、指数中点包絡v3はstartとhalf predictorで非drag callbackを再評価する。
積分器はlift式やfield名を知らず、積分器本体`exponential_midpoint_v2`は維持する。

初回revisionの設定は`gas_velocity_field`、`gas_density_field`、`gas_mean_free_path_field`、
`azimuthal_gas_vorticity_field`、有限正値`lift_coefficient`、`applicability: error`だけとする。
`lambda/(drag_diameter/2)>=10`を固定policyとし、任意speed cap、隠れた`C_L=1`、core内速度微分、
Case P/A分岐を追加しない。drag/thermophoresis/gravityと共用する中性気体field authorityはcatalogで一度だけ照合する。

2026-10-01にP18-Lを完了した。力は
`F=K (omega_phi e_phi) x (u_g-v)`、`K=C_L*pi*rho_g*lambda_g*a^2`、
`a=drag_diameter_m/2`として、catalog v14、physics runtime v13、compiled tile v15から既存の一つの
additive-acceleration stage passへ接続した。必須fieldはRZ vector gas velocity `[m/s]`、正scalar gas density
`[kg/m^3]`、正scalar mean free path `[m]`、producer所有のsigned scalar azimuthal vorticity `[1/s]`である。
coreは速度場を微分せず、有限正値`C_L`と`lambda/a>=10`をfail-closedに検査する。Stokes--Cunninghamと
B02 Brownianとの併用はplan解決時に拒否する。

全非drag加速度boundを`particle_index`とcomponent-wise velocity上界を受ける単一callbackへ統合し、
exponential midpoint enclosure v3はstartとhalf predictorでこれを再評価する。exponential midpoint v2、engine v30、
proposal v7、event/runtime layout/memory plan、case/result/checkpoint schemaは変更していない。3-D cross-product射影oracle、
zero/comoving、符号、`rho*lambda*a^2/m` scaling、直交性、乱択bound、Kn拒否、pure/compiled parity、RZ公開caseの
RK4 3.5次以上・explicit midpoint 1.8次以上と短縮state包絡を受け入れた。

100,000-row warm direct stageの9回medianはdisabled `0.023354 s`、enabled `0.0255142 s`、比`1.0925`で、
prepared boundは`+900016 B`だった。machine-localな非gating観測であり、end-to-end release性能を置換しない。
exported-P1/native-field比較は異なる空間表現をまたいでFAILした。後続full-physics common-field診断は限定sliceの
same-field軌道一致をPASSしたが、native-field pointwise parityとlift modelの物理妥当性は`NOT_TESTED`のままとする。

#### P18-R：既存drag / thermophoresisの比較適用域closure

保存状態auditではnative linear Epstein式parityが最大相対残差約`1.1e-15`でPASSした。一方、既存P15-E/P16の
physical applicabilityはmixture/model authorityの不一致により12/12 caseで`NOT_APPLICABLE`、PPRの
thermophoresis heat-flux primitiveが無いためpointwise replayは`NOT_TESTED`だった。保存frameは連続pathを認証しない。

この結果から、既存`physics/forces.py`の式ownerと単一compiled stage passを再利用する
`epstein_linear_effective_gas_sensitivity_v1`と
`waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1`を追加してP18-Rを完了した。両revisionは
producer認証済みのone-effective-Maxwellian/pseudogas reference/sensitivityだけを対象に、`lambda/a>=10`とcase必須の
`0 < maximum_speed_ratio <= 1`をactual stageと既存連続path enclosureでfail-closedに検査する。既存linear/P16の
上限`0.1`は変更しない。thermophoresisの`q_eff`はproducer所有の有効並進伝導熱流束で、coreはgradient recovery、
species配列、mixture rule、COMSOL分岐を持たない。catalog v15 / runtime v14 / compiled tile v16へ更新し、engine v30、
proposal v7、integrator v2、case/result/checkpoint schemaは不変である。このturnではCOMSOL studyを再実行しておらず、
設定可能なreference/sensitivity runはspecies-resolved mixture truthやCOMSOL軌道一致を確立しない。
同一入力・同一`maximum_speed_ratio=0.1`の100,000-row warm stageでは、旧/new revision pairのmedianが
`0.0267117/0.0267643 s`（`1.00197x`）で、payload、applicability、prepared bound、共有workspaceは同一だった。
これはmachine-localな非gating観測であり、end-to-end性能を置換しない。

#### P18-H：generic terminal hold boundary

M3-C0の力なしnormal-impact正例は、boundary 37 Freezeがstatus 2になり、hit点のR-Zを以後固定する一方、保存velocityは
衝突前の有限値を保持することを確認した。これはdepositionとして速度0にする`stick/deposited`でも、kinematicsをnullにする
`escape/escaped`でもない。COMSOL名やstatus codeをcoreへ入れず、非depositionの終端保持を表す最小`hold` lawと`held`
lifecycleを既存first-hit経路へ統合した。velocityは再積分されるactive状態ではなく、hit時payloadとして保持する。charge等の内部状態をevent後も
時間発展させる一般的なpaused-particle modelはこのpackageに含めない。

`boundaries.py`が応答、`events.py`がfirst hit、`engine.py`がterminal遷移、`output.py`がsparse eventとlifecycleを所有する。
stick/deposited、escape/null、heldを同一statusへ畳み込まない。lifecycle seriesへ`held`を追加したためresult/checkpoint schemaを
2、result algorithmをv4へ一度だけ更新し、互換shimやCOMSOL status codeをcoreへ追加していない。case schema v2、event、
integrator、memory planは不変である。解析的直線hit、曲線hit、frame右連続性、resume、材料stickとの区別に加え、
slab identity、OU/Brownian、zero-time surface、analysis非deposition分類を公開経路で検証した。外部正例と同じcandidate
microcaseも時刻・位置・lifecycle・payloadの15/15 gateをPASSした。COMSOLは再実行していない。この外部比較は
P18-H実装の正しさを単独で定義せず、解析・公開回帰を先に通した。

#### B03：RZ projected Brownianと全寄与の合成

現在の2-D axisymmetric referenceと比較するため、`noise` categoryへ
`inertial_langevin_fdt_epstein_linear_rz_meridional_projected_v1`を追加して完了した。これはr/zへ投影した2自由度近似であり、
物理的なisotropic 3-D Brownianと同一とは主張しない。P17の`axisymmetric_field_cartesian3d`と3-D Brownianは独立revisionで、
現在の比較をblockしない。

B03は第二engineを作らず、既存joint OU/conditional tree/Hermite eventを一つのstochastic proposalへ拡張した。
manifestはcoefficient policy `macro_root_frozen_midpoint_v1`、composition
`stochastic_exponential_midpoint_v1`を記録する。root始点からnoise-free決定論
exponential-midpoint predictorで
midpoint stateを作り、そこで`gamma,u,T`、全additive acceleration `a`、`G=dZ/dt`、`J=dG/dZ<=0`を1回評価してroot内で
固定する。`u_eff=u+a/gamma`を平衡速度とするjoint exact OUで`(x,v)`をroot終点へ進め、
`macro_root_affine_exponential_v2`のdense chargeを同じproposalが所有する。dense chargeの
全leaf intervalはprepared invariantへ照合し、逸脱または数値的に証明不能ならfail-closedにする。predictorは係数評価点のみを作り、
stateを先にhalf-step commitするStrang/K-O-K分割ではない。

固定depth conditional treeはrootと同じ凍結係数を使い、親root endpointを保存する。RZ axis hitでは
accepted prefixをcommitしてfoldし、macro残時間を次の`root_stochastic_interval`ordinalとして係数再評価・
独立root drawで再開する。元のcubic remainderのfold/restrictはしない。terminal boundaryは`stick`/`escape`/`hold`だけとする。
native linear Epsteinとeffective-gas linear Epstein sensitivity、fixed/continuous charge、既存additive forceを受理し、
B02のXY/fixed-charge/drag-only revisionの数値pathとpayloadはbitwise不変とする。

受入条件は凍結定数係数のexact mean/full covariance、noise-off deterministic limitの2次収束、
一定外力Langevinの独立解、manufactured charge-electric couplingのweak mean観測次数`>=0.9`、RZ axis restart、
全optional forceとのcompiled parity、first-passageのtree-depth収束、slab/output/checkpoint identityである。
一般state-dependent SDEのstrong orderまたはweak 2次は主張しない。COMSOLとの判定はpathwise誤差ではなく32 seedを独立clusterとした
時刻別平均・共分散、分位点、分布距離、fate確率、first-arrival CDFの事前登録区間で行う。

初回closeout revisionはcatalog v16、engine v34、proposal v9、event v15、runtime v17、compiled tile v16、memory plan v13である。
現行supersessionはengine v37、proposal v10、event v16、catalog v17、runtime v20、compiled tile v18である。
CPU layoutは新schedulerを作らずv6を維持し、case/result/checkpoint schemaにB03専用配列やmutable RNG cursorを追加していない。
B03 path arrayの静的な保守上限は一slab rowあたり`648 B`で、受入上限`2048 B`内にある。最終characterizationは
2,000/20,000粒子×4構成×3反復の24/24実行を全粒子active・failure 0で完了した。20,000粒子のmachine-local medianは
B02 fixed `5.7722 s`、B03 fixed `11.7142 s`、continuous charge＋gravity `11.8427 s`、axis restart `16.3002 s`で、
最大process peak RSSは`226,316,288 B`だった。これはportable gateでなく、詳細は
[`solver/evidence/b03/`](solver/evidence/b03/README.md)が所有する。
performance characterizationで反復加算由来の終端tailを検出したため、engine v34はmacro timeを補償積和による
`start + n*dt`のindexed gridから構築し、endへのsnapをfloat64構築roundoff内だけに限定する。科学的に意味のある
残時間はsnapで消さず、従来どおりproposal/eventへ渡す。

event v15は物理position budgetとroundoff budgetを加算し、facet-local offset dotを補償演算で、Hermiteの
評価・enclosureをroot-relative TwoDiffで扱う。monotone-approach clearはcubic Hermite derivative-Bernstein
enclosureだけが明示opt-inする。一般RK4、exponential、scalar pathは証明不能時のsplit/fail-closedを維持する。
現行event v16はvalidなRK4 dense rowについて、integrator所有のroot-relative position Bernstein control enclosureを
facetの外向きhalf-spaceへinterval射影する。全4制御点の外向き上限が既存position budgetの負側に厳密に入る時だけ、
convex-hull性から候補全区間の非接触を認証する。証明不能、不正・非有限なcontrol、非RK4 pathは従来どおりsplit/fail-closedとし、
tolerance、first-hit localization、boundary response、endpoint、schemaは変更しない。
position-control payloadは`144 B/row`、構築時追加peakは`128 B/row`で、保守的な同時live見積り約`1.76 KB/row`は
既存一般stage scratch上限`2048 B/row`内に収まる。memory plan v13は変更せず、broad AABB pre-countはfacet-local
clear前の保守的なperformance work量として扱う。

#### P19-L：局所continuous-path applicability certificate（完了）

M3-C1で、全287初期sampleは選択modelの局所適用域内にあるにもかかわらず、遠方cellを含むglobal primitive extremaから
作った速度・加速度包絡が時刻0の全rowを`model_applicability`として拒否した。これはmodel式の実測違反ではなく、現在の
certificateでは局所pathを証明できないことを示す。gate値を緩める、非restrictive設定で迂回する、比較case専用branchを作る、
または証明不能をactual violationとして記録する対応は採らない。

P19-Lは一つの数値work packageとして、次だけを追加した。

1. P19-L完了時点の`integrators.py`が元の固定step RK4 proposalとendpointを変えず、
   `rk4_position_hermite_state_extension_v2`と任意parameter部分区間の
   state enclosureを所有する。certificateのためにendpointを逐次再積分・commitせず、出力時刻もstep境界にしない。
2. `fields.py`はdense path tubeと重なる一row最大64 cellのsupport・primitive range、`physics/runtime.py`はそのrangeに対する力・電荷・
   model applicability boundを返す。`engine.py`はcertificate-onlyなdyadic部分区間workをbounded配列で調停し、候補overflow時は
   物理違反にせず共有するevent `maximum_refinements` budget内で分割する。
3. 全部分区間が証明できた時だけ元proposalを一回commitする。実stage/sampleが適用域外なら`model_applicability`、有限budget内で
   証明だけが閉じなければ`indeterminate_applicability_certificate`として分ける。後者を物理違反数へ数えない。
4. 遠方の極値cellを含むが局所pathは安全なmicrocase、実pathが適用域を横切る反例、budget exhaustion、既存のglobal certificate
   成功caseに対するendpoint/final/event/output schedule/tile identityを受け入れた。Case-A 100 nmのstrict candidateが
   時刻0を越えることはblocker解除条件であって、COMSOL一致のPASSではない。

一般interval framework、第二integrator、COMSOL専用path、adaptive accuracy controllerは作らない。既存
`rk4_global_abs_enclosure_v2`はsupportとrev3b材料/RZ event geometryのauthorityとして維持し、local restrictionはapplicability
だけを所有する。locator-before-validity、hit prefixの同一RK4再積分、fresh residual proposalも変更しない。

P19-L完了revisionはengine v31、proposal v8、event v12、field location v4、memory plan v13、dense path v2である。
後続M3-C1はevent v14、現行はevent v16である。memory planは
dense path 176 B/row、split budget 2のinterval stack 72 B/row、最大64 cell候補arena 544 B/rowを計上する。4,096粒子×20 stepの
公開API観測はglobal/local `0.9276381/0.9345506 s`、比`1.0074517`、科学payload bitwise一致だった。これはmachine-localな
非gating観測である。P19-L性能snapshotはphysics runtime v16、v17はDEP上限の1 ULP外向き境界だけを変更した。
現行runtime v20はv19のoptional aggregate three-currentを維持し、relative-flow ion dragの局所interval force boundと
prepared charge invariantをexponential path certificateへ提供する。
代表Case-A 287粒子1 stepでは、意味論を変えないgeneric chord算術のbatch化でwarmed medianを
`1.647475→1.304153 s`（`1.263x`）とし、final＋frameをbitwise同一に保った。このP19-L時点ではwork形状を変えず、global
enclosureとevent refinementの結合を後続課題として残した。後続event v14がそのevent-query側だけを閉じ、P19-L履歴値は
置換せずに残す。
測定条件とidentityは[`solver/evidence/p19l/candidate_hot_path_v1.json`](solver/evidence/p19l/candidate_hot_path_v1.json)へ固定した。
最終gateはwall/event/integrator 178件、全verification/scenario 575件、lock、Ruff、Pyrefly、import-linter、complexityに合格した。

後続event v14は、global absolute safety enclosureをevent BVHのbroad-phase boundにも流用していた過保守を分離した。
global supportを独立に証明済みのvalid `rk4_dense` rowだけは現在区間の外向きBernstein position/velocity boundをevent queryへ使い、global enclosureはsupport、
global-first applicability、短縮RK4再積分のauthorityとして維持する。dense boundが無効なrowはglobal boundを保持して
必ずsplitし、global enclosureでsupportを証明できないrowはevent queryもglobal boundへfallbackする。first-hit順序、
accepted endpoint、wall law、case/result schemaは変えず、COMSOL専用分岐や第二enclosure設定を追加しない。

現行dense path `rk4_position_hermite_state_extension_v3`は、dense位置評価、位置Bernstein enclosure、物理的な微分・曲率を
root始点相対で構成し、TwoDiffの厳密残差を保存する。dense評価は相対値と残差を足してworld座標へ戻し、
始終点では保存済みendpointを厳密に上書きする。enclosureの下限は`-inf`、上限は`+inf`方向へ外向きに丸める。
公開chord-deviation boundだけはworld座標評価とendpoint chordを覆う狭い`8*eps`絶対座標termを加えるため、
戻り値自体は完全な平行移動不変ではない。v3は、v2の広いroundoff幅がdyadic制限で縮まず、幾何的に
分離した経路でevent certificateが閉じない原点依存を解消する。保存するendpoint、数学的なstate path、
first-hit/event algorithmは変更せず、v3導入時のengine v32、proposal v8、event v14、`rk4_global_abs_enclosure_v2`も維持した。
P19-L完了revisionとその証拠にdense path v2を記録するのは履歴として正しい。

#### M3-C1：12 packageの段階的外部比較

1. 最初に受理したCase-A 100 nm pre-event範囲でfrozen saved-state RHSを評価し、既存のproducer-form区分を
   最小のPPR熱泳動COMSOL補足export後に8/8閉じた。PPR補足は13,202 active saved rowで参照座標が既存v6と完全一致し、Waldmann replayの
   component-scale normalized residual最大`4.4046499933294035e-16`、global relative L2
   `1.0992667494449471e-16`だった。これはsaved-row式parityであり、連続pathまたは軌道一致を認定しない。
2. integrated candidateはexport済みnode値とexact connectivityを使うP1 fieldであり、COMSOL native finite-element fieldとは
   同一表現ではない。したがって比較名は`exported-P1 candidate vs COMSOL native-field reference`とし、candidateを
   `native-field`と呼ばない。P19-L完了後、prepare設定、canonical input、run manifest、result、source exportのhashを一つの
   provenance chainに固定し、0.625/0.3125/0.15625 usの3 runを完了した。各runは287粒子×46 frame、event 0、failure 0である。
3. 旧三つのmacro `dt`はmanifestどおり使われたが、global safety enclosureをevent BVH boundにも流用した
   event/path certificateの人工的なdyadic再積分により、全runが同じ
   `8,450,307` accepted particle pieceへ収束した。最大depth 8/7/6で最深nominal leaf幅も同じ`2.44140625 ns`である。
   candidate fine-pair relative L2は位置`2.6773e-14`、速度`1.4486e-14`、電荷`5.4562e-15`で、設定済みfloat64
   representation-scale floor未満のmacro-step-halving production-output stabilityである。これはRK4の形式4次、測定次数、
   floor未満の誤差、または三つの独立した実効積分gridでの収束を認定しない。この履歴判定を独立な時間収束としては
   採用せず、後続event v14のsolver-only 3刻み判定で置換する。
4. COMSOL native-field referenceのpre-event運用上の自己収束はPASSした。一方、fine同士のcross-representation差は
   位置・速度・電荷のRMS/max 6 gateをすべてFAILした。異なるfield表現をまたぐ結果なので、same-field agreement、
   全物理精度、物理modelの適用可能性を認定しない。exact値はV&V methodologyとcompact evidenceが所有する。
5. このFAILで事前登録した条件が成立したため、full-physics exact-connectivity common P1をCOMSOL側にも与える初回診断を
   独立成果物として実行した。動的電荷、electric、relative-flow ion drag、effective-gas Epstein、Waldmann thermophoresis、
   DEP、RZ lift、gravity/buoyancyを同じcanonical P1場で評価し、287粒子×46 frame、0--450 us、event 0で完了した。
   t=0の1,435値は軌道差を読む前に固定した4096-ULP基準をすべてPASSした。fine軌道差のRMS/max/relative L2は、位置
   `2.4360183916994177e-11/2.3107811844634912e-10 m/1.2165503469347329e-8`、速度
   `1.8055229266990532e-7/1.1037594178013694e-6 m/s/1.6810197404693313e-8`、電荷
   `7.930847904707173e-7/7.105382977101726e-6 e/3.133867418408752e-9`で、事前登録9 gateをすべてPASSした。
   これはこのcommon-field・Brownian-off・event前sliceのsame-field solver agreementだけを認定する。native-field等価性、
   modelの物理妥当性、境界、Brownian、30 ms、他case・粒径・variant、普遍的COMSOL同精度は認定しない。旧M3-Vの
   固定電荷・共通3力runnerとは別成果物であり、coreの式やgateは比較へ合わせて変更していない。cross-representation判定は
   [`solver/evidence/m3c1/case_a_100nm_pre_event_v6/`](solver/evidence/m3c1/case_a_100nm_pre_event_v6/)、common-field判定は
   [`solver/evidence/m3c1/case_a_100nm_common_p1_v1/`](solver/evidence/m3c1/case_a_100nm_common_p1_v1/)を当時のauthorityとする。
6. M3-C0 boundary semantics probeを力なしの解析的normal impactで実行した。現行v2は2 scenario×3刻みのexact 6
   configuration receiptをprocess logから照合し、欠落・重複・形式不正・設定差をfail-closedで拒否する。boundary 37
   Freezeと35 Disappearを10/5/2.5 us、0--150 us、2.5 us保存で分離し、両者のeventは73 us、最初のterminal frameは
   75 usだった。全active frameは`x=x0+v0*t` / `v=v0`に一致し、全run最大の位置/速度誤差は
   `4.726604209672303e-16 m` / `1.7763568394002505e-15 m/s`（各上限`1e-12`）だった。Freezeはstatus 2で
   hit点R-Zを固定し衝突前velocityを保持、Disappearはstatus 4で位置・速度をNaNにした。step間のevent-time spreadは
   `3.07371315899641e-17/2.71050543121376e-20 s`、56 gate PASS、FAIL 0、velocity記述6件である。原本MPH hashは
   不変でcore変更はない。これは隔離したCOMSOL意味だけで、production parity、grazing/corner、full physicsを認定しない。
   v1は科学的に無効ではなく、configuration receiptとactive-flight oracleの監査強度が不足した履歴成果物としてv2に
   supersedeされた。current compact authorityは
   [`solver/evidence/m3c0/boundary_semantics_v2/`](solver/evidence/m3c0/boundary_semantics_v2/)である。
7. Case-A 100 nm common-P1を0--458.75 usへ延長し、particle 57の最初のwafer `stick`を閉じた。event v14 candidate v3は
   failure 0で、material gate 20/20と再計算した0--450 us prefix 9/9をPASSした。prefixのRMS/max/relative L2は位置
   `4.0680739379876864e-13/1.2035672231612963e-12 m/2.0316007372844697e-10`、速度
   `2.1084754731725404e-9/3.844306466969233e-9 m/s/1.9630816315240495e-10`、電荷
   `1.1818881686693635e-7/2.37588949403289e-7 e/4.670220471379047e-10`だった。event時刻・hit位置・terminal電荷の
   candidate/reference絶対差は`2.157542807607049e-13 s`、`2.683964162031316e-14 m`、
   `4.7283812421028415e-9 e`である。このscopeはBrownian-off、共通canonical exact-connectivity P1、最初のwafer stick
   だけで、terminal velocityはCOMSOL Freezeとproduction stickの保存契約が異なるためcross-solver gateにしない。
8. event v13 candidateのquery/refinement/accepted/depthは
   `16,427,517/7,792,306/8,635,211/16`で、refinementの`7,623,460/7,792,306`
   （`97.8331703092769%`）が450 us checkpoint以前に発生していた。event v14はこれを
   `842,927/11/842,916/11`へ減らし、failure 0を維持した。operator観測wall time約`853 s→36.5 s`
   （約`23.4x`）はmachine-localでmanifest外の非gating値である。COMSOL入力・reference・原本MPHは不変なので再実行せず、
   hash固定済みreferenceをevaluation v5で再評価した。compact authorityは
   [`solver/evidence/m3c1/case_a_100nm_material_event_v1/`](solver/evidence/m3c1/case_a_100nm_material_event_v1/)である。
9. event v14で0.625/0.3125/0.15625 usのsolver-only candidateを再実行した。query=acceptedは
   `206,927/413,567/826,847`、refinement 0である。位置・速度・電荷のRMS観測次数は
   `2.029875353701904/2.0816971911764033/2.044084026475049`、fine relative L2は
   `6.099791486063973e-8/8.321356016032579e-8/1.3796067752988052e-8`で、三量とも`ORDER_EVALUATED`の自己収束PASSとなった。
   piecewise P1場とmesh crossingを含むこの実caseの観測次数であり、RK4の形式4次を証明も否定もしない。このfine candidateを
   使う現行common-P1 comparisonも9/9をPASSした。RMS/max/relative L2は位置
   `4.06807283316903e-13/1.2035778717837921e-12 m/2.0316001855367802e-10`、速度
   `2.10847522277453e-9/3.844306466969233e-9 m/s/1.9630813983926987e-10`、電荷
   `1.1818881019499895e-7/2.3758877887303242e-7 e/4.670220207738041e-10`である。初期v2結果は履歴として保持する。
10. P18-Hの最小generic `hold/held`は解析的直線・曲線hit、右連続frame、resume、slab identity、OU/Brownian、
    zero-time surface、analysis非deposition分類を閉じた。既存hash固定Freeze referenceへのcandidate比較も
    COMSOL再実行なしで15/15 PASSした。
11. B03 coreは解析・manufactured・identity gateで完了し、COMSOLを再実行していない。
12. 100 nm・30 ms candidate v3はCase A/PのBrownian-off `h,h/2,h/4`系列についてstate自己収束をPASSし、Case Aは
    event identityとevent量の自己収束もPASSした。これはcandidateの主要な数値判定であり、COMSOL stepへの一致を要求しない。
13. 同じ時刻の保存COMSOL Case A/Pはmanual explicit fixed RK4 10 us、Brownian-on、native finite-element fieldの単一runである。
    Brownian-off・exported-P1 candidate v3とのtime-history差は`CHARACTERIZED`な外部記述であり、particlewise parity、
    candidate coreのFAIL、または10 us模倣の根拠にしない。compact authorityは
    [`solver/evidence/m3c1/existing_30ms_reference_v1/comparison.json`](solver/evidence/m3c1/existing_30ms_reference_v1/comparison.json)である。

現行の主要数値gateはcandidate自身の解析・manufactured caseと自己収束である。charge-stable continuous couplingと
deterministic work-scaled durable cadenceとP20 performance closeoutは完了し、直近の全品質gateはPASSした。
同じ物理・field・boundary・stochastic意味を持つ外部V&V/M3-C2A common-P1 Case-A/Case-P 100 nm anchorも完了した。
Case-P finalは20 us、32+32独立seed、各287粒子、30 ms、121 frameで、登録済みR-Z/fate分布gateを`PASS`した。
受理済みCase-P設定を固定した287粒子owner discoveryも、3 seedすべてで受理済み科学payload・work・case identity・revisionを
完全一致させて完了した。支配ownerは`integrators`（自己時間比42.58--42.86%）だったが、事前登録済みbounded ownerではないため
`optimization_authorized=false`でproduction変更はない。このanchor benchmarkは下記closeout authorityにより
`CLOSED_ACCEPTED_WITH_LIMITATIONS`とする。10,000粒子以上の性能測定は自動的な次工程ではない。
旧約`3e8` accepted piece/run見積りは人工的event subdivisionを含むpre-v14観測なので現行性能計画値にせず、保存済み
M3-C1 evidence/evaluatorはv14の履歴として固定する。

#### M3-C2A：Case-A / Case-P 100 nm anchor benchmark closeout authority

この節をM3-C2A anchor benchmarkの目的、完了状態、再開条件の単一authorityとする。

- **有限な目的**：選定ion-drag、common-P1場、100 nm、R-Z、30 ms、動的帯電・決定論力・Brownianを含む固定same-formモデルについて、
  candidate自身の数値健全性と、独立したCOMSOL実装との事前登録observableの整合を一度だけ評価する。
- **完了条件**：入力・物理・seed・出力意味の固定、COMSOL非依存の解析・manufactured・自己収束、Case-A/Pの登録ensemble gate、
  failure 0、科学payload・work・revision identity、hash付き外部evidence、machine-local実行費と非認定範囲の記録を満たすこと。
- **状態**：`CLOSED_ACCEPTED_WITH_LIMITATIONS`。以下1--6で全完了条件を満たした。
- **このbenchmarkで認定しないもの**：pathwise乱数一致、native-field parity、普遍的COMSOL同等性、元Case-Pとは別の
  aggregate three-current拡張、全12 package、他粒径・第2 ion-drag・物理3-D、COMSOLより高速という主張、10^4--10^6粒子の実用時間。
- **停止規則**：追加COMSOL実行、追加seed、全12 package総当たり、10,000粒子以上のscale、外部1/2/4 process並列、hotspotの反復探索を
  このbenchmarkへ追加しない。「さらに速くできそう」は再開理由にしない。
- **再開条件**：このaccepted caseの式、integrator、event意味、入力identityを変えるrevision、evidence無効化、または認定claim自体の変更だけとする。
  新しい物理、case、scale、性能SLAは既存benchmarkを再開せず、独立work packageとして開始する。

M3-C2全体の製品coverageは将来12 packageへ展開できるが、M3-C2A anchorの完了条件ではない。現存の12履歴は各packageに
単一実現しかなく、事前登録済み384行のseed表もすべて`PLANNED_NOT_RUN`である。保存COMSOL modelの
particle interfaceはout-of-plane無効で、解かれる運動状態は`q3r/q3z/v4r/v4z`のR-Z 2自由度である。
Brownian表の`r/phi/z`列と非零`Fbphi`はfeatureが報告する成分であり、第3運動自由度や遠心力連成の証拠ではない。
一方、保存modelの乱数modeは`GenerateUnique`で、`bf1.i`へ置いたparameter値1/21が実効乱数系列を支配した証拠はない。
COMSOLのstepwise BrownianとB03のexact joint OUは、それぞれ独立にobservable収束を確認する必要がある。従って保存履歴を
seed cohortやpathwise referenceへ昇格させない。read-only model authorityは
[`solver/evidence/m3c2/model_semantics_probe_v1/`](solver/evidence/m3c2/model_semantics_probe_v1/README.md)である。

1. **M3-C2A inventoryからexecution lockへ（完了）**：theory-consistent Case-A 100 nmを最初のanchorとする。
   保存package、source MPH、Case A/P各32 seedの事前監査は
   [`solver/evidence/m3c2/anchor_preflight_v1/`](solver/evidence/m3c2/anchor_preflight_v1/)で完了したが、statusは
   `BLOCKED_MISSING_MEANING_MATCHED_COHORT`、64行の`run_matrix.csv`はseed予約だけでcampaign lockではない。
   次にexact-connectivity common-P1入力hash、287粒子、121 frame / 30 ms、release、geometry/boundary、charge・全決定論力の
   revision、candidate case、COMSOL `r/z` FDT companionの生成式・温度・線形Epstein摩擦・更新間隔、pilot/final stepを
   一つの実行契約へ固定する。COMSOL側はout-of-plane無効を維持し、既存built-in `bf1`とEpstein-equivalent effective
   viscosityを使い、interface乱数modeを`UserDefined`へ変更して`bf1.i`をreplica seedのauthorityにする。custom random-force
   featureは増やさない。native-field保存履歴や実効seed不明のrunをgateに使わない。
   この入力・意味lockは
   [`solver/evidence/m3c2/caseA_100nm_pilot_contract_v1/`](solver/evidence/m3c2/caseA_100nm_pilot_contract_v1/README.md)で
   `PASS_INPUT_IDENTITY_AND_SEMANTICS_LOCKED`を完了した。source/common-P1 hash、287粒子、30 ms/121出力、release、geometry、
   boundary、全決定論revision、R-Z Brownian target、pilot/final seed分離を固定した。この時点の履歴statusは
   `NOT_AUTHORIZED / NOT_EVALUATED`だったが、後続runner validation、pilot、final registrationで解除した。
2. **独立pilot（完了）**：本比較に使わないseed集合で、COMSOLとcandidateをそれぞれのstep/tolerance系列で評価する。
   両solverへ同じ固定刻みを強制せず、共通の出力時刻におけるobservableの自己収束で受理刻みを決める。
   一般SDEのpathwise収束次数は主張せず、時刻別moment、occupancy、fate、生存、first-arrivalの統計量に対する
   fine-pairの数値不確かさを固定する。本比較の同値幅は、結果を見た後ではなくこのpilotと明示的な
   Monte Carlo confidence budgetから事前登録する。
   V3 policyでは4 seedを推論に使わずconfiguration screeningだけに限定し、COMSOL 20 us、candidate 20 us / Brownian tree
   depth 3を受理した。旧V2 pilot推論は無効化し、final seedと分離した。
3. **32-seed anchor（完了）**：完了済みB03の解析・統計microcaseとpilot合格後に、既存lockの32独立seedで
   Case-A anchorを比較する。seedが独立cluster、粒子がcluster内sampleであり、同じ整数seedでもRNGの異なる
   solver間でpaired pathとはみなさない。単一seedの粒子ID軌道差をfailureにしない。
   COMSOL 32 seedとcandidate 32 seedを各287粒子、30 ms、121時刻で完走した。全candidate failureは0、COMSOL study isolationは
   32/32 PASSである。
4. **anchor評価（完了）**：時刻別平均・共分散・固定分位点、固定binのRZ occupancy、fate確率、生存曲線、first-arrival CDFを
   whole-seed resamplingの区間で評価する。粒子行を独立標本とするbootstrap、有意差がないことだけを同値とする
   判定、結果後の閾値変更を禁止する。
   確認的authorityはwhole seed×固定source populationに対する4終端人口曲線とした。最大差0.5989%、95%同時上限3.877%で
   事前登録5%幅を`PASS`した。連続軌道moment/quantile/occupancyとR-Z overlayは非判定の説明証拠である。authorityは
   [`solver/evidence/m3c2/caseA_100nm_final_campaign_v1/`](solver/evidence/m3c2/caseA_100nm_final_campaign_v1/README.md)とする。
5. **Case-P一軸展開（完了）**：meaning-matched common-P1 Case-P 100 nmを20 us、32+32独立seed、各287粒子、
   30 ms、121 frameで完走した。83区分R-Z/fate gateは最大empirical TV `0.010670731707317093`、同時上限
   `0.13119968456545308 < 0.15`で`PASS`した。終端gateも`PASS`だがevent 0のため境界parityには情報を持たない。
   元Case-P COMSOL `auxq`は電子＋正イオン二電流を意図して使い、負イオン密度は診断量に留めるため、candidateも同じ二電流式を
   選んだsame-form比較である。この一致をspecies-resolvedな物理真値や後続three-current拡張の認定へ広げない。
   pathwise RNG、普遍的COMSOL同等性、eventful boundary parityは認定しない。第2 ion-dragと10/30 nmは別の未認定軸として残す。
   authorityは[`solver/evidence/m3c2/caseP_100nm_final_campaign_v1/`](solver/evidence/m3c2/caseP_100nm_final_campaign_v1/README.md)とする。
6. **性能owner discovery（完了、適格性とは分離）**：受理済みCase-P設定を変えず、287粒子のcandidate seed
   `319032`、`319047`、`319063`を
   外部profileしてowner候補を発見する段階は完了した。全rerunは受理済み科学payload・work・case identity・revisionと完全一致し、
   支配ownerは3 seedとも`integrators`（自己時間比42.58--42.86%）だった。ただし事前登録済みbounded ownerではないため、
   automatic decisionは`optimization_authorized=false`でproduction変更はない。authorityは
   [`solver/evidence/m3c2/caseP_100nm_owner_profile_v1/`](solver/evidence/m3c2/caseP_100nm_owner_profile_v1/README.md)である。
   final計時は`NON_AUTHORITATIVE_EXTERNAL_WORKLOAD_OVERLAP`であり性能baselineに使わない。この結果はhotspot仮説の記録であり、
   10^4--10^6粒子のwall timeまたはCOMSOL速度比を認定しない。
7. **bounded chord follow-up（完了）**：利用者の明示指示で、owner discoveryが限定した
   `curved_chord_deviation_bounds`だけをserial compiled batchへ置換し、Python scalar helperとevent側の重複式を削除した。
   accepted 3 seedのpublic end-to-end wall中央値は45.4668 sから39.8801 sへ12.29%短縮し、科学payload、revision、
   3,444,000 accepted piece/candidate queryを完全一致させた。新しい診断、設定、runtime、比較分岐、threadingは追加していない。
   5%/3 MAD gateを通過したため保持し、このfollow-upは終了する。authorityは
   [`solver/evidence/m3c2/caseP_100nm_chord_optimization_v1/`](solver/evidence/m3c2/caseP_100nm_chord_optimization_v1/README.md)である。
   これは287粒子accepted workloadの実装効率改善であり、10,000粒子以上の製品scaleやCOMSOL速度比を認定しない。

製品性能は、対象hardware、粒子数、30 msの物理条件、出力mode、許容wall time、peak RSSを先に定義した
`accepted-workload performance` work packageだけで判定する。開始する場合も、10,000粒子一workloadで
curved-event enclosure/chord preparationを低オーバーヘッド計測し、25%以上を説明する場合だけ一箇所を一度評価する。
科学payload・work identityを維持し、改善が`max(5%, 3 MAD)`を超えなければrevertして終了する。通過しても同じrunから
次ownerを探索しない。100,000/1,000,000粒子と外部process並列は、明示的SLAが要求した場合だけ別work packageとする。
詳細は[`solver/docs/parallel_execution_plan.md`](solver/docs/parallel_execution_plan.md)が所有する。solver内部threading、第二engine、
物理組合せ別kernel生成、scheduler frameworkは再導入しない。受理済みstep/tree depthの変更も性能測定へ混ぜず、別の
accuracy/convergence work packageで再認定する。

#### P21 / M3-C3：aggregate three-currentとcritical 2-D closure

M3-C2Aを再開せず、元Case-Pとは別の選択可能な物理拡張と、2Dで最も重要な境界正例を一つの有限work packageで閉じる。
順序と停止条件は次の4件だけとする。

1. **aggregate three-current production revision（完了）**：
   `aggregate_relative_drift_regularized_two_current_v1`を変更せず、aggregateな単一価負イオン電流を加える
   `aggregate_relative_drift_regularized_three_current_v1`を同じcharge ownerとsingle compiled engineへ追加する。
   `R_Z=Gamma_+-Gamma_e-Gamma_-`、明示`n_-`,`V_-`,`u_-`,`m_-` field、正負両イオン共通の相対speed envelopeを使う。
   `n_-=0`ではrate/Jacobian/global boundを二電流へ厳密退化させる。三電流revisionを選んだcaseの負イオンfieldと相対speed
   applicabilityは密度0でも検査する。screening lengthは既存明示fieldが唯一のauthorityで、負イオンprimitiveからcoreで
   再計算しない。species-resolved current、Case-P分岐、第二engine、診断frameworkは追加しない。独立oracle、finite bound、
   compiled parity、公開API scenario、標準quality gateまでをproduction受入条件とし、catalog v17、runtime
   `signed_ion_compiled_physics_runtime_v19`、compiled tile v18で完了した。engine/state/schemaは変更せず、標準
   P21受入時のverification/scenario 621件とRuff、Pyrefly、import-linter、complexity、lock gateを通過した。
2. **critical boundary microcase（完了、PASS）**：一つのfrom-scratch 2D axisymmetric rectangleで、3粒子による
   surface contact departure、specular reflectionと同step残時間、R-Z axis passageをCOMSOL、production public API、解析解で
   比較した。`dt=1/0.5/0.25 ms`の132 gateは全PASS、最大solver間位置差は`1.61339e-17 m`である。authorityは
   [`solver/evidence/m3c0/critical_boundaries_v1/`](solver/evidence/m3c0/critical_boundaries_v1/README.md)とする。
   grazing/corner、multiple material hit、probabilistic law、有限半径、力・native fieldへ一般化しない。
3. **conditional Case-P派生companion入力authority（`PASS`）**：priority 1の三電流を明示選択する
   代表100 nm full-physics caseの入力authorityをproducer companionで生成した。common-P1全1987節点について、有限な`n_-`、
   total `u_{-,r}`、total `u_{-,z}`、`m_-`、`V_{T,-}`を、domain cacheとdomain-3側boundary cacheからcanonical H5へ格納した。
   COMSOL geometryのcm→SI m変換とRZ axis正則性射影を一度だけ明示し、source MPHを変更していない。
4. **conditional Case-P派生trajectory evaluation（`PASS`）**：共有three-current `Z0`を用いた
   common-P1、Brownian-off、100 nm、287粒子、30 msの比較で、candidateと明示drag COMSOL referenceの各3刻み収束、
   全frameのlifecycle/finite mask exact、共通の有限lifecycle stateの`r,z,Z`、共通active stateの`v`、141件のevent/fate identityと
   event時刻gateがすべて`PASS`した。元Case-P M3-C2Aの二電流anchorは変更しない。compact authorityは
   [`solver/evidence/m3c3/caseP_three_current_companion_v1/`](solver/evidence/m3c3/caseP_three_current_companion_v1/README.md)である。

P21/M3-C3は`CLOSED_ACCEPTED_WITH_LIMITATIONS`とする。priority 1の任意三電流revision、priority 2のcritical boundary、
priority 3の入力authority、priority 4のtrajectory evaluationは完了した。priority 3/4は元Case-P二電流anchorの同等性に必要な
作業ではなく、異なる任意物理の外部coverageであるためP21の出口から分離する。
これにより、100 nm common-P1二電流anchorと上記critical boundaryを明示scopeとする`2D_CRITICAL_VV_COMPLETE`を宣言する。
これはnative-field、追加粒径、第2 ion-drag、3D、普遍的COMSOL同等性、物理modelの妥当性を認定しない。

三電流coverageの再開条件だったproducer側one-sided domain-3 canonical 5 primitive exportを満たし、P21や既存benchmarkを
再開せず、独立した外部V&V work packageと100 nm・287粒子・30 msの代表比較を一回だけ実行した。
このscoped numerical agreementを全12 package、追加seed、native-field、性能比較、物理model validationへ拡張しない。

外部成果物は`tools/vv/comsol`だけが所有し、core testのgolden fileにしない。各caseの結果は、model revision、field producer、
入力hash、step、seed集合、対応可能/不可能な量を一つのmanifestへ保存する。

#### 実装順と変更単位

M3-C0a、P18-C、P18-I、P18-D、P18-L、P18-R、P19-L、M3-C1 frozen saved-state producer-form 8/8、Case-A 100 nm
pre-event exported-P1/native-field比較とfull-physics common-P1診断、M3-C0 boundary semantics probeは完了した。M3-C0b v6は
Case-A 100 nmのpre-event刻みを運用上選択しただけで、全30 ms・全case/variantは未完了である。
cross-representationの6 gateはすべてFAILしたが、原因分離用common-field診断は事前登録9 gateをすべてPASSした。境界正例probe
単独ではCOMSOL側のFreeze/Disappear意味だけを閉じていたが、後続P18-Hが解析回帰と同じcandidateでisolated production parityを
15/15 PASSした。field-representation局在化、event v14の
common-P1 first material-stick 20/20＋prefix 9/9、v14 solver-only 3刻み自己収束も完了した。P18-Hのproduction境界は
解析・公開回帰と外部Freeze candidate 15/15で完了した。B03 coreも完了し、100 nm・30 ms candidate v3のCase A/P
自己収束もPASSした。保存COMSOLとの差はBrownian-on/offとfield表現差を含むためcharacterizationに限定する。
charge-stable coupling、durable I/O cadence、P20 performance closeoutも完了した。後続の意味一致M3-C2A common-P1 Case-A/Case-P
100 nm finalは各32+32 seedで`PASS`した。Case-Pの非自明なauthorityは登録済みR-Z/fate分布gateであり、終端event 0のgateを
境界parityへ拡張しない。native-field providerは引き続き`NOT_TESTED`である。別のdeterministic common-P1 work packageでは、
size-specific入力/provenanceを修正し、10/30 nm relative-flow ion dragと100 nm image ion dragの各287粒子・0--450 us・
3刻み自己収束とcross-solver 9 gateをすべて`PASS`した。このevent-free、Brownian-off結果をM3-C2A stochastic anchorへ混ぜない。
authorityは[`solver/evidence/m3c1/case_a_size_ion_drag_companion_v1/`](solver/evidence/m3c1/case_a_size_ion_drag_companion_v1/README.md)である。
aggregate three-currentのCase-P派生外部評価はcanonical負イオン5 primitive入力blockerを解消し、common-P1、Brownian-off、
100 nm・287粒子・30 msの代表trajectory比較を`PASS`したconditional coverageへ分離した。P21/M3-C3 productionと
critical boundary microcaseも完了し、後者は132/132 PASSした。
P21/M3-C3と明示scopeの2D benchmarkは`CLOSED_ACCEPTED_WITH_LIMITATIONS`である。
accepted seed 3件の287粒子owner discoveryは科学identityを完全一致させて完了したが、
`integrators`は事前登録済みbounded ownerでないため最適化は未承認である。M3-C2A anchorは
`CLOSED_ACCEPTED_WITH_LIMITATIONS`であり、10,000粒子以上の性能は利用SLAを定義した別work packageとする。
gate bypassやCOMSOL fittingは使わない。
各矢印は同一PRを意味せず、一つのmodelごとに式・compiled evaluator・最小verification・一つの
公開scenarioを閉じてから次へ進む。P17は別state-dimension trackとして並行可能だが、P18各modelへXYZ対応を先行実装しない。

追加categoryは既存`physics.models`の0/1選択へだけ加え、boolean alias、COMSOL profile、一般registry、Python callbackを
作らない。無効時はresident particle stateを増やさず、追加fieldとbounded scratchをmemory planへ計上する。各hot-path
変更は同一caseでon/offの時間・peak memoryを測り、既存single-thread compiled engineを維持する。

reduced electrostatic builderは粒子engine外のfirst-party field-production componentとしてpotential/E fieldを生成し、
external plasma field adapterと同じcanonical writerへ出力する。F01のRZ/P1、C2 Boltzmann--Bohm closure、Newton/GMRES、
canonical出力とF02 adapterは完了済みである。P18で必要なderived primitiveはproducer側へ追加し、builder/runtimeへ
COMSOL専用分岐や第二mesh経路を作らない。
- F02では`model_dataset`固有処理をbuilderへ入れず、adapterがdomain抽出、quadの品質基準付き決定論的分割、
  boundary owner/semantic group再構築、thermal field転記を行った。代表P1 meshの収束・反復数・solve-only時間・
  Krylov storage、fixed charge＋electric統合を確認し、現GMRESを維持した。COMSOL field比較は外部V&Vに置き、
  同一node上の記述比較と単一mesh制約を明示した。
- model uncertaintyと数値離散誤差を別のsensitivity結果として出す。

### Stage 4A：時間依存field

- topology固定のtime snapshots
- hold/linear time interpolation
- knot/discontinuityでmacro interval分割
- 前後二snapshotのdouble bufferとprefetch
- snapshot時間解像度のadequacyは外部preprocessorが評価

### Stage 4B：完全3D

- tet4 barycentric field
- tri3 boundary BVH
- 3D surface release
- 3D wall normalとcorner candidate
- full 3D memory/scaling benchmark

Stage 4Aと4Bを同じreleaseで実装しない。

### Stage 5：GPUと製品運用

regular/static/event-lightなど適合caseから追加する。CPUと同じcase hash、RNG key、event、ResultStoreを使い、
CPU parityと転送を含むend-to-end speedupがないcaseではGPUを選ばない。

---

## 17. 実装backlogの推奨分割

| 順番 | work package | 完成物 | 状態 |
|---|---|---|---|
| P00 | clean-room bootstrap | uv/Python pin、初回lock、五品質tool、独立install、旧package非依存 | 完了 |
| P01 | canonical v1 | YAML/HDF5 writer/reader/hash、schema関連decisionの確定 | 完了 |
| P02 | core microcase pack | C01～C10、analytic expected、physics/numerics decisionの確定 | 完了 |
| P03 | coordinates/single-point field | XY/RZ、regular/P1/Q1、conditioning-aware location | 完了 |
| P04 | ballistic engine/output | table source、SoA ballistic、release event、final/frame、最小ResultView | 完了 |
| P05 | geometry/event | 大域topology audit、line BVH、ballistic exact first hit、stick/escape | 完了 |
| P06 core | coupled RK4/physics | revision 1：既存force verification。revision 2：証明済み一定加速度。revision 3a：boundaryless XY regular enclosure。revision 3b：sequential accepted RK4 material tube、wavefront、transverse certificate、accepted-path memory checkpoint | 完了（P06 close時engine v8 / event v5。現行engine v37 / proposal v10 / RK4 enclosure v2が意味論を維持） |
| P06-U | unstructured material | exact-mesh fully-supported P1/Q1、Cartesian XY一般RK4、topology-complete material boundary | 完了。boundaryless unstructuredは未解禁 |
| P06-RZ | RZ force | signed stage basis、RZ metadata/axis regularity、canonical support像、RK4 axis event/residual | 完了（engine v12 / event v8 / physics catalog v2 / required field v3） |
| P06-S | Stokes–Cunningham | 小さいphysics runtimeへのdrag責務集約、Stokes schema/applicability/oracleのXY/RZ追加 | 完了（engine v13 / physics catalog v3 / physics runtime v1） |
| P07 | wall/source RNG | Philox counter RNG、fixed-time surface、静止壁law、corner、exact-path residual、RZ axis split | engine v11 / event v7までのsliceを完了し、現行engine v37が意味論を維持。分布拡張、moving wallは未完了 |
| P08 | Stage 1A closure | particle-local failure reason/継続規則、小型series/probe、薄いCLI、公開API scenario、収束report | 完了（engine v14 / result algorithm v2） |
| P09 | memory/runtime layout | YAML/resources先行parse、HDF5 metadata gate、固定ID対応のresident stateとstable resident-row active list、bounded microtile、load/prepare/run memory plan、RSS/semantic harness | 完了（engine v15 / runtime layout v1 / memory plan v1。代表実測を完了し、synthetic全規模判定はP14、target-use判定はP14-U。hintは消費者とともにP10へ移動） |
| P10 | compiled CPU | Numba field/physics/RK4、regular supported-containing O(1) common path、P1/Q1 hint/full-search、`state_at`/wall/residual/output schedule parity | 完了（engine v16 / compiled tile v1 / runtime layout v2 / memory plan v2 / physics runtime v2。large-mesh P1/Q1性能はP14） |
| P11 | native integrator | exponential midpoint reference、解析検証、compiled実装/parity、既存material/RZ/output経路との統合 | 完了（`deterministic_particle_engine_v17` / `compiled_cpu_tile_v2` / proposal v4 / event v9 / physics runtime v3 / exponential midpoint/enclosure v1） |
| P12 | event-heavy parallel | 非重複particle tile ownership、compiled BVH/RK4事前認証/wall prefix batch、worker-local residual/event/statistics、最大W tile wave、stable merge、thread identity | 完了（engine v18 / compiled tile v3 / geometry v3 / runtime layout v3 / memory plan v3。製品規模thread scaling判断はP14） |
| P13 | durable result | 固定64 macro epoch、worker-wave stream、容量1 writer queue、A/B checkpoint、LATEST、auto-resume、recovery、failure injection | 完了（engine v19 / result algorithm v3 / checkpoint schema 1 / memory plan v4。case/result schema v1は不変） |
| P14 | synthetic performance | 10k/100k/1M、regular/P1/Q1、realistic unstructured cell count、initial localization/cross-cell motion、event/output/thread matrixとbottleneck判断 | 完了（engine v20 / compiled tile v4 / field v3 / geometry v4 / memory plan v6。23行×3観測、336 verification/scenario。case/result schema v1は不変） |
| P14-P | serial runtime convergence | bounded slab、stackless boundary BVH、row-target flat SoA event wavefront、compiled boundary/RNG、stable output。並列gate未達時のthread API削除 | 完了。engine v27 / compiled tile v6 / proposal v5 / event v11 / runtime layout v5 / memory plan v10 / geometry v5。case schema v2、single-thread compiled engineへ収束 |
| P14-U | representative use | surface＋非一様場＋材料wall＋多数step、時間/mesh収束、global-bound/event cost、直列end-to-end性能 | 完了。18 raw＋6 median、別1M profileを受入れ、単一owner支配なしのためproduction変更なし |
| P14-R | release closure | performance evidence保存、Windows/Linux wheel・clean install・三API smoke、文書authority整理 | 完了。baseline v27/P14-U/local auditを履歴として保持し、receipt固定のtested head/lockについてremote Windows/Linuxで620件、性能smoke 7行（cold 1＋warm 6）、wheel、runtime-only clean install、三API smokeが合格。`release_remote_ci_v1.json`をreceiptとする |
| T03 | analysis/visualization | bounded boundary-event iterator、ResultViewだけを読む最小集計・軌道・event可視化 | 完了。ResultViewだけを読み、source resultへ書き戻さず、revision・source hash・parameterを派生成果物へ保存する |
| P15 | continuous charge | `oml_stationary_maxwellian_debye_huckel_v1`、RK4-first連成、charge-aware enclosure、explicit stiffness拒否、その受入後のnative explicit midpoint | 完了。engine v28 / proposal v6 / RK4・指数enclosure v2 / memory plan v11。後続P14-R remote CIも完了 |
| M3-V | external target applicability/relevance＋matched companion | 直接MPH inventory、12 package構造、formula parity、sampled applicability、force/variant感度、別成果物の決定論exact-P1 pre-event軌道 | 完了。元の12 package全軌道は`NOT_APPLICABLE`。共通P1場・共通3力のCase-A 100 nm companionだけは事前登録幅内でPASS。native-field、boundary、stochastic、builder/3-Dは未認定 |
| P15-D | relative-drift continuous charge | 単一正イオン種、shifted-Maxwellian OML、非正電位invariant、明示drift envelope、compiled/runtime parity、両積分器・壁・XY/RZ・resume統合 | 完了。compiled tile v8 / catalog v6 / runtime v5。engine/proposal/schemaは不変 |
| P15-E | finite-speed Epstein drag | 有限相対速度の一つのversioned free-molecular drag、低速極限、明示Kn/速度比適用域、独立速度積分oracle、両積分器の収束・compiled parity | 完了。Maxwell鏡面/等温拡散混合、rate/Jacobian別bound、catalog v7 / runtime v6 / compiled tile v9。linear modelへの自動切替なし |
| P15-F | collisionless Barnes ion drag | 単一正イオン、linear two-species Debye、Debye--Hückel表面電位、collection＋orbital、弱結合/collisionless適用域、impact-parameter oracle | 完了。explicit acceleration、catalog v8 / runtime v7 / compiled tile v10。外部datasetのfloor/clamp/image/E方向modelを移植せず、engine/schema不変 |
| F01 | reduced electrostatic builder | canonical thermal-flow RZ/P1、C2 Boltzmann--Bohm Poisson、semantic BC、Newton/GMRES、potential/E/plasma primitive、provenance | 完了。particle engine/schema/dependencyは不変 |
| F02 | provider adapter / representative integration | mixed triangle/quad→canonical P1、boundary owner/group再構築、thermal primitive、代表linear solve、fixed-electric smoke、外部field比較 | 完了。core/schema/revisionは不変。同一node field比較のみTESTED、独立mesh収束と軌道・Freeze parityは未検証 |
| P16 | Waldmann--Gallis | 単一気体の局所並進熱流束によるfree-molecular thermophoresis、連続適用域、独立運動論oracle、compiled parity、両積分器収束、XY/RZ parity | 完了。explicit acceleration、catalog v9 / runtime v8 / compiled tile v11。gradient回復・Talbot blendなし、engine/schema不変 |
| B01 | inertial Brownian numerics | exact joint OU covariance/update、物理interval-tree Philox normal、親endpoint保存conditional half-split、ensemble verification | 完了。`stochastic.py`と`rng.py`だけ。production integrator/case/event/output/checkpoint/memory planは未接続 |
| B02 | inertial Brownian production | Cartesian XY、Epstein linear、fixed charge、固定depth OU/Hermite、terminal stick/escape、frame/probe、resume、row-local数値失敗 | 完了。engine v30 / proposal v7 / catalog v10 / runtime v9 / runtime layout v6 / memory plan v12。連続OU first-passageは非主張 |
| P17 | axisymmetric field / Cartesian 3-D | XYZ state、RZ mapping、回転面event、3-D normal、schema/output/memory | state-dimension独立workstream。一般可変次元frameworkは作らない |
| M3-C0 | comparison evidence freeze | 12 packageの式・parameter・field意味、admissibleなBrownian-off `h/h2/h4`、frozen RHS/RK stage、正例Freeze/Disappear、package別seed cohortを外部成果物として固定 | `DEFERRED_NOT_RELEASE_BLOCKING`。M3-C0a offline lock、Case-A 100 nm pre-event v6の運用刻み選択、力なしboundary正例v2の56 PASS / 0 FAILは完了。v5の旧PASSは原点依存な位置relative L2のため`INVALIDATED`、履歴statusは`CHARACTERIZED`。30 ms全12 package総当たり、完全derived-field provenance、RK probeは現行v0.1と`2D_CRITICAL_VV_COMPLETE`の完了条件へ含めず、科学revisionまたは明示的なcoverage拡張時だけ独立work packageとして再開する。core変更なし |
| P18-C | benchmark-reference continuous charge | regularized relative-drift aggregate two-currentをoptional versioned revisionとして独立実装 | 完了。catalog v11 / runtime v10 / compiled tile v12、engine/schema不変。独立oracle・bound・両積分器・XY/RZ・event/resume合格。外部保存式再生PASSと定数規約を含む厳密provider一致FAILを分離 |
| P18-I | selectable ion-drag sensitivity models | relative-flow screened式とelectric-field-directed image式を二つの排他的revisionとして実装 | 完了。catalog v12 / runtime v11 / compiled tile v13、engine/schema不変。Barnesを変更せず、blend・fallback・Case P/A分岐なし。外部frozen-force判定は`evidence/p18i/`へ分離 |
| P18-D | quasistatic spherical DEP | producer提供`grad(mean_E_squared)`を使うoptional explicit-acceleration model | 完了。catalog v13 / runtime v12 / compiled tile v14、engine/schema不変。半径認証、bound、pure/compiled parity、XY/RZ、両積分器の解析収束を確認。Case-A 100 nm common-P1複合sliceはM3-C1でPASSしたが、DEP単独・native-field parityと物理妥当性は未認定 |
| P18-L | rarefied-vorticity lift sensitivity | RZ no-swirl専用のdocumented free-molecular lift感度revision | 完了。catalog v14 / runtime v13 / compiled tile v15 / exponential enclosure v3。producer提供signed方位vorticity、正の明示係数、`lambda/a>=10`、速度依存bound、Brownian併用拒否を検証。engine v30 / integrator v2 / proposal v7とschemaは不変。cross-representation比較はFAIL、Case-A 100 nm common-P1 event前same-field診断はPASS |
| P18-R | drag / thermophoresis applicability closure | 保存状態でP15-E/P16の式と適用域を監査し、不成立時だけreference revisionを追加 | 完了。P18-R closeout時点ではnative linear Epstein replay PASS（約`1.1e-15`）、既存P15-E/P16 applicability 12/12 `NOT_APPLICABLE`、PPR `q_eff`欠損でthermophoresis replay `NOT_TESTED`。effective-gas二revisionをcatalog v15 / runtime v14 / tile v16へ追加し、engine/schema不変。後続M3-C1の最小PPR補足でsaved-row producer-form replayを閉じたが、P18-Rの物理判断は変更しない |
| P18-H | terminal hold boundary | nondeposition terminal保持をproducer非依存のgeneric `hold/held`として追加 | 完了。engine v32 / boundary v5 / result v4 / result・checkpoint schema 2。case schema、event、integrator、memory planは不変。解析・resume・Brownianを含む公開回帰と外部Freeze candidate 15/15 PASS。COMSOL再実行なし |
| P19-L | localized continuous-path certificate | integrator-owned dense RK4 path、local cell primitive bound、certificate-only subinterval restriction、actual violationと証明不能の分離 | 完了時はengine v31 / proposal v8 / event v12 / field v4 / memory plan v13 / dense path v2。global enclosureをsupport/event authorityとして保持し、applicabilityだけglobal-first/local-fallbackで認証した。後続event v14はevent BVH queryだけdense Bernstein boundへ分離し、global support/applicability/reintegration authorityを維持 |
| M3-C1 | candidate-first deterministic validation and qualified external comparison | candidate自身の解析・manufactured referenceと`h/h2/h4`自己収束を主要gateとし、COMSOLは同じ物理・field・boundary・stochastic意味のcaseだけ別statusで比較 | 既存Case-A anchorのfrozen replay 8/8、cross-representation 6/6 FAIL、common-P1 pre-event 9/9、material 20/20＋prefix 9/9、v14 solver-only自己収束、P18-H 15/15を各scopeで維持。100 nm・30 ms candidate v3はCase A/P自己収束PASS。保存COMSOLはfixed RK4 10 us・Brownian-on・native-fieldなのでcharacterizationだけで、30 ms COMSOL parityや全caseは未認定 |
| B03 | charged/forced RZ Brownian | RZ meridional投影、fixed/continuous charge、native/effective-gas線形Epstein、全決定論力をmacro-root stochastic exponential-midpoint/OU proposalで合成 | 完了。engine v34 / proposal v9 / catalog v16 / event v15 / runtime v17 / compiled tile v16 / memory plan v13。P17の物理3-D Brownianとは別revisionで、一般SDEのstrong/weak 2次を主張しない。COMSOL再実行なし。正式characterizationは24/24 active・failure 0 |
| charge-stable coupling | deterministic/B03 continuous charge | midpoint-frozen affine exponential `J<=0` root、RK4 explicit gate維持、`h/h2/h4` accuracy | 完了。engine v36 / tile v17 / proposal v10 / exponential v3 / runtime v18。clip・charge-only subcycle・第二engineなし |
| durable cadence | work-scaled atomic result commit | `W=macro+accepted+queries+refinements`、`T=max(2^20,128N)`、accepted macro barrier、resume identity | 完了。engine v36 / result v5 / cadence `cumulative_solver_work_v1`。同期single-owner writer、output/slab非依存 |
| P20 | operational efficiency closeout | work-scaled cadenceとcharge-stable larger-step utilityのpublic-API manual evidence | 完了。`solver/evidence/p20_efficiency/`のmachine-local・non-gating証拠。portable timing、異なるstep間のequal-accuracy、COMSOL比較は非主張 |
| M3-C2A | stochastic anchor benchmark | 独立seedの時刻別moment・分布・fate・first-arrivalとCI | `CLOSED_ACCEPTED_WITH_LIMITATIONS`。common-P1 Case-A/Case-P 100 nm final完了。Case-Pは20 us、32+32独立seed、各287粒子×121 frame / 30 ms。83区分R-Z/fate TVは`0.010670731707317093`、同時上限`0.13119968456545308 < 0.15`でPASS。終端gateはevent 0で非情報的。元Case-P `auxq`どおりの二電流same-form結果であり、後続three-current・species-resolved物理・pathwise RNG・普遍的COMSOL・boundary parityは非主張。287粒子owner discoveryも科学identity完全一致で完了。scale性能とcoverage拡張は別work package |
| P21 / M3-C3 | aggregate three-currentとcritical 2-D closure | optionalな単一価負イオン集約電流、critical boundary microcase、Case-P派生companionの入力監査を一軸ずつ閉じる | `CLOSED_ACCEPTED_WITH_LIMITATIONS`。priority 1はcatalog v17 / runtime v19 / tile v18と標準品質gateで完了。priority 2も`solver/evidence/m3c0/critical_boundaries_v1/`で132/132 PASS、最大solver間位置差`1.61339e-17 m`。priority 3はproducer-owned one-sided cacheによりfull canonical 5 primitive authorityを全1987節点へ生成して入力blockerを解消。priority 4は共有Z0とcommon-P1、Brownian-off、100 nm・287粒子・30 msの代表比較でcandidate/明示drag COMSOLの各3刻み収束、全frameのlifecycle/finite mask exact、共通有限stateの`r,z,Z`、共通active stateの`v`、141 event/fate identityをすべてPASS。元M3-C2A二電流anchorは不変で、普遍的COMSOL同等性や物理model validationは非主張 |

P14は全組合せの直積を作らず、粒子数、layout/motion、hit数、output、JIT、threadを直交させた23行×3観測を
三公開APIのfresh childで測った。全行でrevision、result shape、memory fit、要求JIT環境、科学payload identityを
hard gateとし、秒数、RSS、artifact byte rateはmachine-localな記述値とした。当時のouter-worker方式ではregular
1 worker 10k/100k/1Mが0.095647/0.783874/7.534041 s、20 worker 100k/1Mが0.417049/1.570727 sだった一方、
event 10k×20 hitは1/20 workerで33.509603/37.728221 sだった。この履歴結果を現行host幅選択へ使わない。
後続P14-Pでparallel production経路と`resources.threads`を削除し、現行はsingle-thread compiled engine一つである。
独立seed/caseのprocess並列はsolver外で1/2/4 workerを実測し、採用gateを満たす運用だけを残す。

実測profileで判明したP1/Q1初期/missの全cell走査はfield v3のsupported-containment BVH、table startのvolume全走査は
geometry v4のmixed-cell BVHへ置換した。両者ともindexはbroad phaseで、既存exact predicateが最終authorityである。
field outside/masked provisionalは物理最近傍/tieを保つfull scanなのでO(cell数)制約を残す。geometryは局所形状が
float64で解像不能ならprepareでfail-closedとし、大きな絶対座標だけでは拒否しない。memory plan v6はindex resident、
field 256 B/cell、geometry 1,024 B/cellのprepare transientを含む。solver-coreへtimer、第二engine、cache frameworkを
追加しない。現在はP14-Pの直列収束、P14-Uの代表用途gate、T03とP14-R、およびP15の二つのproduction sliceを完了した。
配布可能なv0.1はremote Windows/Linux workflow成功で閉じた。P15出口の外部M3-V評価と
field-production F01/F02、trajectory physicsのP15-D relative-drift charge、P15-E finite-speed Epstein、
P15-F collisionless Barnes ion drag、P16 Waldmann--Gallis thermophoresis、Brownian B01/B02、P18-C/I/D/L/R、P19-Lは完了した。
M3-C0b v6はpre-eventの運用上の刻み選択だけを受理済みである。M3-C1 frozen saved-state producer-form replayは
最小PPR補足後8/8を閉じた。P19-Lで過保守なglobal applicability certificateによる時刻0 blockerを解消し、exported-P1
candidate 3 runとnative-field referenceのcross-representation評価を完了した。6 gateはすべてFAILしたが、続くfull-physics
exact-connectivity common-field COMSOL診断は事前登録9 gateをすべてPASSした。続く外部boundary正例v2はexact 6
configuration receiptと全active-frame解析解を含む56 PASS / 0 FAILでFreeze/Disappearの隔離意味を閉じた。field表現差局在化、
event v14のcommon-P1 first material-stick 20/20＋prefix 9/9、v14 solver-only 3刻み自己収束も完了した。P18-Hも
解析・公開回帰と外部Freeze candidate 15/15で完了し、B03 coreも完了した。100 nm・30 ms candidate v3のCase A/P
自己収束もPASSしたが、保存COMSOLはfixed RK4 10 us・Brownian-on・native-fieldなので外部characterizationに限定する。
charge-stable coupling、durable I/O cadence、P20 performance closeoutと、意味を揃えたcommon-P1 Case-A/Case-P 100 nmの
独立seed ensemble外部V&V/M3-C2Aは`CLOSED_ACCEPTED_WITH_LIMITATIONS`として完了した。受理済みCase-P seed 3件の
287粒子owner discoveryも科学identity完全一致で完了し、支配ownerは`integrators`だったが事前登録済みbounded ownerではないため
最適化は未承認である。10,000粒子以上のscale性能、aggregate three-currentの外部Case-P coverage、残るcase coverageは
それぞれ独立work packageであり、
このbenchmarkを開いたままにしない。不要なCOMSOL総当たりやfittingは行わない。
後続のbounded chord follow-upは12.29%のend-to-end短縮と完全なscience/work identityで完了し、次ownerへ進まず閉じた。
本体を変更しない外部M3-V matched trajectory比較の最初の決定論sliceも完了した。同一P1 connectivity・共通力の
287粒子×41時刻で、position RMS `8.643e-16 m`、velocity RMS `1.999e-14 m/s`となり、事前登録幅内でPASSした。
native finite-element fieldとexported-P1の空間表現差、boundary、stochasticは同じ合格表示へ混ぜず、独立した外部gateとして残す。
state dimensionのP17は独立workstreamとして上記判断に従う。Case-P相当の直接plasma-field producer対応は、
必要primitiveとspecies契約を固定してからT02の別adapter sliceとして行い、F02のthermal adapterへ条件分岐を重ねない。
T04は後続profile条件付きである。

外部toolはcore milestoneと混ぜず、次の独立trackで進める。これらはP06 core実装のblockerにしない。

| ID | 外部tool | 最小成果物 |
|---|---|---|
| T01 | canonical case builder | producer非依存のSI/quantity/topology/provenance変換 |
| T02 | COMSOL adapter/V&V | COMSOL export変換とoperator→trajectory→event比較 |
| T03 | analysis/visualization | bounded event iteratorと`ResultView`だけを使う最小集計・軌道・event可視化 |
| T04 | field cache preprocessor | profileで必要性が確認された場合だけ再mesh/cacheを生成 |

P04は次の順で一つの縦切りとして実装した。

1. `case.py`：data座標とmotion modeを分離し、`TrajectoryOutputSpec`とrelease time規則を厳格化する。
2. `sources.py`：複数tableを`(release_time, particle_id)`でstableに統合する。
3. `integrators.py`：ballistic `StepProposal`と解析的`state_at(theta)`だけを実装する。
4. `engine.py`：固定ID SoAとparticle-local release残時間を使う唯一のloopを実装する。
5. `output.py`：result schema v1、同期一segment writer、必須final、最小lazy `ResultView`を所有する。
6. `api.py`：`simulate/open_result`のstubを削除し、この一経路へ接続する。
7. C01と、release時刻が異なる2粒子の公開API scenarioでframe/final/output独立性を検査する。

P04ではgeometry、field sampling、physics evaluator、RNG、background writer、checkpoint、汎用model registryを
追加しない。当時のresult外部契約v1は同じchangeで短いowner文書へ固定した。P18-Hで旧文書を
現行[`docs/result_format_v2.md`](solver/docs/result_format_v2.md)へ置換したが、別writer契約や巨大schema frameworkは作らない。

P04は上記1～7を完了した。複数tableのrelease順、負時刻を含むparticle-local release、release時刻frame、
`dt_s`・output schedule不変性、no-clobber publication、未完result、XY/RZの対応組合せを公開API scenarioで
確認した。

P05は次の順で同じproduction経路を拡張した。

1. `geometry.py`：volume incidenceから外周を監査し、outward normal、point classification、flat line BVHを準備する。
2. `events.py`：geometry scale・facet長・速度・時刻からbudgetを一度だけ解決し、ballistic直線の最初のhitと
   同時facet集合を決定する。
3. `boundaries.py`：交差探索を持たないparameterless `stick` / `escape`だけを適用する。
4. `engine.py`：boundaryなしもno-hitとして扱う単一loopへfirst hitとterminal lifecycleを接続する。
5. `output.py`：releaseとは別のboundary event、candidate offsets、`kinematics_valid`を同じ一segment resultへ保存する。
6. C07、topology異常、escape frame、output schedule独立性、RZ seamとaxis順序をverification/scenarioで検査する。

P05は上記1～6を完了した。区切り評価では、BVH leafのfacet AABB再確認、boundary vertex manifold性と
self-intersection、facet固有budget、区間外交点のposition/time二重gateを追加した。boundaryなしcaseも同じ
geometry prepareを通り、event始点・accepted endpoint・途中frameは`StepProposal`から取得する。canonical
P05時点でDataBundleを含むresident-memory下限gateも回帰化した。後にP09のphase memory planが
置き換えた。frameはeventに対して右連続、escaped finalは
`kinematics_valid=0`、release/boundary ordinalは0/1で固定した。RZ軸が材料hitより先または同時のpath、複数facet
応答、反射、surface release、曲線path、compiled kernelは先回りしていない。

P06 revision 1では、同じproduction engineのproposalを`rk4_step`へ統一し、無力場を`linear_exact`退化形、
force-coupled XYをclassical RK4として構成した。required fieldは同一layoutのcontinuous node fieldに限定して
全particle-domain coverageをprepareで証明し、fixed charge、Epstein linear drag、Coulomb電気力、重力・浮力を
実stage位置・時刻で連成する。4段に加えてaccepted endpoint（releaseがstep終端と一致するzero-durationを含む）の
field support/applicabilityも検査した。C02/C03の解析時系列とRK4 4次収束、C04/C05の一定加速度、位置依存加速度の
direct RK4 stage再評価、物理式の適用域、不安定な`dt/tau`の拒否はkernel/physics verificationとして有効である。

ただしpost-reviewで、有限個のstage/endpoint sampleだけでは連続pathのsupport包含を証明できず、追加frame sampleが
run成否を変え得ることを確認した。前版`coupled_rk4_engine_v3`は一般`rk4_reintegrated`を一律拒否したが、
revision 3aの`coupled_rk4_engine_v4`は証明済みのboundaryless regular subsetだけを公開productionへ移した。

P06 revision 2は一般曲線eventを一度に解禁せず、fixed charge、dragなし、必要fieldの全node値が厳密に
一定であることをprepareで証明できるcaseを先に扱う。この場合は合成加速度が粒子ごとに一定で、RK4 pathが
厳密放物線になる。`events.py`はchordを`|a|h^2/8`で膨張したBVH broad phaseとparabola-line二次根から
first hitを求める。trial endpointがsupport外でも先行hitがあれば救済し、hitがない行だけsupport/
applicabilityを判定する。このrevisionではcontinuous charge、force-coupled RZとの壁連成を対応gateまで明示拒否した。
RZ壁連成は後続P06-RZで解禁済みである。
Epstein dragまたは非一様fieldを含む一般`rk4_reintegrated`は、revision 3aでboundaryless regular supportと
連続applicabilityを証明したsubsetだけを実行し、材料boundaryとの連成にはrevision 3bの各local始点から
再構築するbound、sequential accepted RK4 pieces、離散path tube、event-before-validityを要求する。

revision 2は完了した。canonical node値の厳密一致を一定場certificateとし、粒子別合成加速度をprepareで一度だけ
構築する。`StepProposal`は解析的な放物線位置・速度を保持し、`events.py`はchord偏差を含むBVH候補抽出と
facetごとの二次根から最初のhitを求める。hit前terminalをfull trialのsupport判定より先に確定し、hit時刻の
速度・電荷をboundary eventへ保存する。turning/chord-miss、false-positive no-hit、near-grazing hit/miss、
接線・共線failure、macro step・output schedule不変性を検証済みである。topology-completeな材料boundaryがある
caseではfirst hitが連続pathのparticle domain退出を捕捉する。boundaryなしcaseは全cell supportedな
`RegularLayout`に限定し、各proposalで座標ごとの端点と内部極値を解析的に求め、矩形support box内にあるか検査する。
頂点だけがsupport外へ出る反例はframe有無の双方で同じfailureになる。これによりC04/C05は公開API scenarioとして
有効である。revision 3aも完了した。対象はboundaryless Cartesian XY、fixed charge、全cell supportedな
`RegularLayout`だけである。global field extremaとmodel係数から各粒子・各macro区間のRK4速度・加速度を包絡し、
外向き丸めした位置・速度enclosureが、`state_at()`による全短縮区間の内部stageとaccepted endpointを含むことを
要求する。Epsteinを使う場合は`lambda/a`下限と`|u-v|/c_bar`上限も同じ速度boundから全区間で証明する。
supportまたはapplicabilityを証明できないcaseは別の数値pathへ暗黙分割せずfail-closedにする。C02/C03の公開API
解析時系列、frameなし・疎・密scheduleでのfinal identity、step途中releaseの粒子別残時間、安全な非一様regular
field、短縮pathだけがsupportを逸脱する反例、Epstein適用域反例のschedule非依存failureを検証した。

revision 3bは`coupled_rk4_engine_v6`として完了した。各local piece始点から同じboundを再構築するsequential
dyadic RK4、材料boundary用の離散path tube、event-before-validity、証明不能時のfail-closed判定を接続した。
accepted piece列が数値pathのauthorityであり、frameはそれを再生する。同じmacro proposalのparameter区間を
局所tubeとして流用しない。当該revisionの範囲はCartesian XY、fully-supported `RegularLayout`、fixed charge、既存の
Epstein/electric/gravity、terminal stick/escapeだった。exact-mesh unstructuredはP06-U、RZ forceはP06-RZで後に
解禁した。boundaryless unstructuredとcontinuous chargeは後続gateを保ち、Stokes--Cunninghamは後のP06-Sで
小さいphysics runtime、明示air schema、Kn/Re適用域、独立oracleを揃えて解禁した。

revision 3b着手前の小さいhardeningでは、(1) `E_x ∝ -x`の調和振動子を公開APIで
`h / h/2 / h/4`評価し、位置・速度とも観測次数3.5以上、各刻みでframe scheduleによらないbitwise同一の
final状態を確認した。(2) 放物線の座標極値はdense stateと同じ演算順で評価し、式の絶対項scaleとfloat64の
演算回数から導くroundoff幅で全区間を外向きに拡張するintegrator所有intervalへ一本化した。このsupport受理
意味論の変更を当時の`coupled_rk4_engine_v5`として記録した。
(3) frameなし/macro終端/interior frame、boundaryless/material-hitを三公開APIで測る非gating baselineを
`tests/performance/p06_baseline.py`へ置いた。particles/s、stack比較、現行manifestのplanned memory、artifact bytes/particleは
記録できる。revision 3bではaccepted particle-piece数、BVH candidate query数、refinement数、最大深さを
production manifestで集約し、同じbaselineへ追加した。8粒子・warmupなしのlocal smokeでは一般RK4材料hitが
boundaryless一般RK4の約14～15倍、最大深さ40であり、正しさの参照実装としては完了したが高速化余地は明確である。
engine v7は各粒子からleft-firstなoutstanding pieceを1件だけwaveに出し、float64で完全に同じ
target timeのproposal行を256粒子chunk内でbatch化した。event v5はintegrator所有のcomponentwise
chord deviationとroundoff幅を使い、events所有の単一facet外向き横断、normal time bracket、
tangent/time-shiftを含むposition radius、endpoint clearanceを証明する。証明失敗はsplit/full-tube fallbackとし、
engine v7、proposal v3、enclosure v1、schema/APIは不変である。64粒子の同一machine baselineで、material medianは
event v4の0.9666176 sから0.6450654 s（約1.50倍）、boundaryless medianは0.0471402 s、比は
21.77から13.6840となった。初期scalarの2.3706326 sからは約3.68倍で、accepted piece / candidate query /
refinement / 最大深さは1088 / 2496 / 1408 / 21である。accepted-pathは要求frameと時間的に重なるproposal rowだけを
保持する。`tracemalloc`による64/256/1024粒子の同一scenarioではframeなしpeakが
669,511/2,142,293/7,118,215 Bから356,077/913,336/2,108,319 Bへ減少した。この測定はprocess RSSでも
P09のload-to-run memory planでもない。これをもってrevision 3b regular-XY sliceをcloseした。
local regular-grid bound、dense output、Numba化は独立した必要性を測定せず先行導入しない。

P06-Uはengine v8として完了した。required fieldとparticle volume meshが完全一致し、全cell supportedなP1/Q1を、
topology-completeなCartesian XY材料境界と組み合わせる一般`rk4_reintegrated`だけを解禁した。continuous supportは
exact-mesh coverage、strict-inside start、RK4 tubeと全出口boundaryのfirst-event判定で証明する。boundaryless
unstructuredは別の連続包含証明がないため引き続きprepareで拒否する。

P07のforce-free exact-path sliceはengine v9として完了し、engine v10 / event v6でCartesian XYの証明済み
一定加速度surfaceまで拡張した。旧`TableSchedule/build_table_schedule`を単一の
`ParticleSchedule/realize_sources`へ置換し、terminal専用boundary APIも一般law経路へ置換した。壁面sourceは
初期位置を動かさず、まず法線速度のscale-aware符号でdomain内向きdeparture、壁向き即時impact、曖昧failureを
分ける。厳密tangentだけは証明済み一定加速度の法線成分を同じ規則で分類する。内向き加速度ではeventsがsource
facetのsupporting line、内向き速度、内向き加速度を各intervalで再証明し、別wall hitまでsource facetだけを
除外する。これによりmacro partitionへ依存しない。外向き加速度ではzero-time impactを適用する。terminal
stick/escapeは可能だが、反射で内向きdepartureを作れない応答は失敗する。specular/probabilistic stick後の
linear/quadratic残時間、priorityとcombined normal、C09のcap-driven split、ballistic RZ axis foldを同じengineへ
接続した。engine v11 / event v7はCartesian XY一般RK4について、厳密内向きsurface departureとsingle-facet activeな壁応答後の
残時間継続を同じwork loopへ追加した。start-contact facetを除外するのは区間の速度包絡が厳密内向きを証明した時だけで、
tangent、facet端点/corner、または曖昧な接触はfail-closedとする。interaction countはparticleのresidual stateで共有し、上限到達後に
実際の次hitを確認した時だけ分割する。反射を含むevent時刻のframeはpost-stateを選ぶ右連続である。seed、Philox
revision、draw kind、resolved law/priority、wall/residual/axis集計はmanifestに残る。角度・速度・時刻の分布拡張と
moving wallはこのsliceに含めず明示拒否する。後続P06-RZはengine v12 / event v8でsigned radial stage、
canonical RZ field basis、axis regularity、support区間像、wall/axis event arbitrationを同じwork loopへ追加した。
axis foldはboundary event、wall RNG ordinal、interaction countを消費せず、case/result schema、proposal v3、
enclosure v1を変更しない。

P06-RZの実装境界は次で固定する。`physics/catalog.py`はcase座標に応じてvector required fieldを
`(x,y)/cartesian_xy`または`(r,z)/axisymmetric_rz`へ解決する。`coordinates.py`はsigned stageとcanonical
samplingの変換、およびsigned enclosureの`abs`像を所有する。`fields.py`はaxis nodeのradial required-field値、
geometryのaxis接触またはboundaryless fully-supported regular boxの`r_min=0`から導く単一`axis_accessible`を所有し、
regularity証明済みRZ vectorをcanonical `r==0`でsampleした時だけ補間丸めによるradial値をexact `+0.0`へ復元する。
近軸値のtolerance clampや一般的なfield修復にはしない。
`engine.py`は同じ判定でradial gravityをprepare時に0と確認する。`events.py`は一般RK4 axis pieceの
clear/split/hitと残差、および斜めfacet AABB候補に対する支持線法線intervalの認証済み除外を所有し、engineは双方が局在済みの場合だけmaterial wallとaxisの認証時間を比較して、
不確かさが重なるtieではwallを優先する。axis端点とmaterial cornerの完全tieは一般RK4 cornerとしてfail-closedにする。
解析Epstein axis crossingの`h/h/2/h/4`収束、frame schedule identity、away-axis Cartesian退化一致、
axis→wall順序、regular/P1/Q1のaxis regularityを完了条件とし、230件のverification/scenarioと標準品質gateで確認した。
新しいschema selector、第二integrator、RZ専用physics式、診断treeは追加しない。

同一machineの非gating characterizationでは、Philoxを含むuniform surface realizationは100万粒子を
約0.35 s（約285万粒子/s）で実現した。一方、C08の2 wall eventを含むexact pathは1000粒子で約0.33 s
（約3000粒子/s）であり、処理時間の大半は粒子単位の残時間loopとfirst-hit探索だった。P10はまず共通の
field/physics/RK4 array passをcompiled化し、その後のprofileでもevent-heavy側の改善幅が小さいことを確認した。
P12はphysical event順を保つresidual work、compiled first-hit前処理、同時刻wall prefix batch、非重複tile ownership、
stable mergeを実装した。512粒子×4 macro stepではserial baselineから約6.7～7.1倍改善したが、2/4 threadは
1 threadより遅く、製品規模のthread scalingは未証明である。P09はsource
realizationの一時copyをdirect scatterで減らし、load/prepare/runの
solver-owned phase peakと外部RSS characterizationを分離した。この測定値は
hardware依存の観測であり、受入閾値や100万粒子wall runの実測値ではない。

P06-Sは、展開済み`PhysicsPlan`からsample済みprimitiveを評価する小さなphysics runtimeへ、reference force、
primitive-extrema bound、applicability、一定加速度certificateを集約した。field samplingとevent順序はengineに残し、
registry/manager/base-class階層は作っていない。P09はYAML全体とresource上限を先にparseし、HDF5
metadataからcanonical numeric footprintを計算してpayload展開前に拒否する。完全なmemory planはprepareで、
DataBundle、geometry、schedule、physics runtime、resident state/active index、output/writer、microtile scratchを
一度だけ解決する。

一つのwork packageは一つの利用可能能力とそのverificationを同時に含める。先に全interfaceだけを作る、または
全moduleのempty skeletonを埋める進め方はしない。

---

## 18. 継続運用のルール

### 18.1 changeの入口

新しいmodel・座標・integrator・backendを追加する前に、次の短い表を`docs/decisions.md`へ追記する。

```text
解く利用case
既存modelで解けない理由
所有module
必要field/state
対応座標・integrator
解析解またはreference
性能・memory影響
置換・削除する旧経路
```

長いADR directoryや承認workflowは作らない。判断が一ページを超える場合だけ別文書へ分ける。

### 18.2 complexity budget

- public APIは三つを維持する。
- production engineは一つを維持する。
- physics categoryごとにactive modelは一つ。
- state slice ownerは一つ。
- runtime永続形式はJSON/HDF5だけ。
- constant diagnosticsはcount、timing、memory、failure reasonまで。
- full traceは明示probe IDだけ。
- model追加のために新しいmanager/provider/facadeを作らない。
- 同じ能力の旧実装と新実装を長期間並存させない。

文書も一つの事実に一つのownerを保つ。現在のmilestone状態は本書、algorithm decision/historyは
`solver/docs/decisions.md`、方程式は`technical_research.md`と`solver/docs/physics_models.md`、永続revisionは
result format文書が所有する。product/architecture文書へengine versionや進捗を重複追記せず、READMEは短い入口と
linkだけにする。新しい文書を増やす前に既存ownerへ追記できない理由を確認し、置換した状態記述は同じchangeで削除する。

### 18.3 test/CI

環境と品質toolの役割・設定は [quality_tooling_plan.md](quality_tooling_plan.md) を権威とする。通常changeでは
独自runnerを介さず、次を安い順に直接実行する。

```console
uv lock --check
uv run --locked ruff format --check src tests tools
uv run --locked ruff check src tests tools
uv run --locked pyrefly check --summarize-errors
uv run --locked lint-imports
uv run --locked python scripts/check_complexity.py
uv run --locked pytest tests/verification tests/scenarios -q
```

Ruffはformat/lint、Pyreflyは型、import-linterは少数の依存方向、Radonは関数complexityだけを所有する。
Ruffの`C90`、Pyrefly baseline、Nox/Toxによる単純なcommand中継、同内容の独自architecture testは導入しない。
Radonは原則CC 10以下、数値上の理由がある場合でも15以下とし、16以上だけを小さいcheck scriptで拒否する。
Maintainability Indexは参考値でありgateにしない。

全scenarioとwarm performance smokeはmain/nightly、10^6粒子・cold compile・failure injection matrixは
release/manual gateとする。coverage率を目的にprivate branchのtestを増やさない。

少なくともWindowsとLinuxでwheel build、clean install、三公開APIのsmokeを行う。性能gateは共有runnerの
単発秒数ではなく、同一hardware上の複数回medianとmemory/identity結果を使う。
2026-09-29のlocal release auditではWindows 11 / Python 3.12.13とWSL2 Ubuntu / Python 3.12.3のwheel、
runtime-only clean install、三公開API smokeが合格し、Linuxのlocked標準gateも合格した。旧root workflowは復元せず、
新solverだけを対象にしたWindows/Linux workflowを新設した。P14-R baseline v27の18条件×3反復＝54観測、P14-U、
両platformのlocal auditは`solver/evidence/v0.1/`へ保存済みである。2026-10-05に新workflowをrepositoryへ反映し、
初回remote Windows/Linux実行を成功させた。run `37217687830`とtested head、tested lock、両job結果は
`solver/evidence/v0.1/release_remote_ci_v1.json`に固定し、P14-Rを完了とする。

### 18.4 versioning

- case schema version
- physics model revision
- algorithm revision
- result schema version

を別に持つ。code package versionだけで科学的意味を表さない。初期版のresumeは全revision一致を要求し、
migrationを作らない。

---

## 19. stageごとのblocking decision

未使用機能の値をP03前に一括決定してhidden default化しない。責務を実装する直前に、対応するsampleと
独立referenceを伴って確定する。短い状態表の権威は`solver/docs/decisions.md`とする。

- P01/P02で、particle property authority、P1/Q1順序、boundary orientation、C01～C10の具体値、
  fixed chargeの初期値authority、初期corner policy、event要求値のschemaを確定した。
- P03 revision 1のlarge-offset/high-aspect defectは`field_location_v2`で置換した。物理polygon包含、
  座標ULPと局所Jacobianによる認証、最近傍supported射影、非有限failureを一経路へ統一し、回帰と品質gateを
  完了した。旧locatorとruntime selectorは残していない。
- P04でdata座標表現とmotion mode、typed trajectory schedule、必須final/release event、frameのrelease時刻
  意味論を未release schemaへ反映した。table source、ballistic proposal、一segment、final/frame、最小lazy
  `ResultView`だけを実装し、geometry/field/physics/checkpointを先回りしていない。
- P05で大域topology audit、`solver.event`からposition/time budgetを一度だけ解決する処理、ballisticの
  厳密直線path、facet固有budget、対称な同時hit結合、曖昧交差のfailureを実装した。境界なしもprepareを通し、
  `StepProposal`をpath/frameの唯一の評価元にした。曲線path boundはP06へ置く。
- P06 revision 1ではXY、同一layoutのcontinuous node field、fixed charge、Epstein/electric/gravity、
  `applicability=error`とkernel/physics verificationを確定した。revision 2では厳密一様場・
  dragなしの一定加速度を、topology-completeな材料boundary、または全cell supportedな`RegularLayout`の解析的
  座標極値検査と連成した。revision 3aはboundaryless XY regular supportに限定し、
  global field/model bound、全短縮RK4評価を含む外向きenclosure、Epstein連続applicability、output schedule不変性を
  実装した。
  revision 3bはengine v6/proposal v3で、同一proposalのparameter refinementを使わず、geometry-drivenな
  sequential accepted RK4 pieces、provisional validity、event-before-validity、accepted-piece frame replayを
  接続した。engine v7はこの意味を保ったままdeterministic refinement wavefrontを導入し、event v5で
  transverse certificateを完了した。engine v8はaccepted replayの寿命を要求frameへ限定し、exact-mesh P1/Q1
  material domainをP06-Uとして追加した。engine v9はP07のforce-free exact-path wall/source sliceを追加し、
  engine v10 / event v6は証明済み一定加速度surfaceへ拡張した。engine v11 / event v7はCartesian XY一般RK4の
  厳密内向きsurface departureとsingle-facet active-boundary residualを追加した。engine v12 / event v8は
  P06-RZのforce-coupled RZ basis/axis gateを完了した。engine v13はP06-S、engine v14/result v2はP08、
  engine v15/runtime layout v1/memory plan v1でP09を完了した。engine v16/compiled tile v1/runtime
  layout v2/memory plan v2/physics runtime v2はP10を完了した。engine v17/compiled tile v2/proposal v4/
  event v9/physics runtime v3はP11を完了した。engine v18/compiled tile v3/geometry v3/runtime layout v3/
  memory plan v3はP12、engine v19/result algorithm v3/checkpoint schema 1/memory plan v4はP13を完了した。
  engine v20/compiled tile v4/field v3/geometry v4/memory plan v6はP14のsynthetic baselineを完了した。
  現行engine v37/compiled tile v18/proposal v10/event v16/runtime layout v6/memory plan v14/geometry v5はP14-Pの
  single-thread compiled engineへ収束し、outer pool、thread mask、worker別scratch、object event/replayを削除済みで、
  exact/curved wavefront、row status、batch release、direct replay、bounded stagingも接続済みである。P14-Uで
  代表用途を閉じ、T03も完了した。P15の物理・数値decisionとproduction実装は
  `oml_stationary_maxwellian_debye_huckel_v1`、RK4-first、`hL <= 0.5`のexplicit stiffness gateとして完了し、
  式revisionは`solver/docs/physics_models.md`が所有する。RK4-firstを受け入れた後、explicit midpoint chargeも
  第二sliceとして完了した。P15-Dはその同じ経路へ単一正イオン・非正電位のshifted-Maxwellian revisionを
  catalog v6 / runtime v5 / compiled tile v8として追加し、engineとschemaを維持した。P15-Eはcatalog v7 / runtime v6 /
  compiled tile v9、P15-Fはcatalog v8 / runtime v7 / compiled tile v10、P16はcatalog v9 / runtime v8 /
  compiled tile v11として同じengineへ追加した。P18-Cはcatalog v11 / runtime v10 / compiled tile v12として
  aggregate two-current continuous chargeを同じengineへ追加した。P18-Iはcatalog v12 / runtime v11 / compiled tile v13、
  P18-Dはcatalog v13 / runtime v12 / compiled tile v14、P18-Lはcatalog v14 / runtime v13 / compiled tile v15として
  同じstage passへ追加した。P18-Lだけが速度依存boundに必要なexponential enclosure v3へ更新し、integrator v2、
  engine v30、proposal v7は維持した。P18-Rは既存式ownerを共有する二つのeffective-gas sensitivityをcatalog v15 /
  runtime v14 / compiled tile v16として同じpassへ追加し、engine、proposal、integrator、schemaを維持した。P19-Lは
  P19-L完了時点のengine v31 / proposal v8 / event v12 / field location v4 / memory plan v13 / dense path v2として
  局所applicability certificateを
  同じengineへ追加し、global support/event enclosureとrev3b逐次event意味論を維持した。physics runtime v16はその性能snapshot、
  v17はDEP上限の1 ULP外向き境界だけを変更した。runtime v18はcharge Jacobian、exponential midpoint v3は
  charge-stable affine exponential root、result v5はwork-scaled cadenceを追加した。旧P14-R着手blockerは
  ユーザーの明示指示で解除し、別release trackのremote CIも2026-10-05に完了した。
- P21/M3-C3は既存二電流revisionを保持したままaggregate three-currentをcatalog v17 / runtime v19 / tile v18で同じ
  charge plan/runtimeへ統合した。critical 2-D boundary microcaseも132/132 PASSで完了した。Case-P派生companionは
  canonical負イオン5 primitive authorityを全1987節点へ生成して入力blockerを解消し、common-P1、Brownian-off、
  100 nm・287粒子・30 msの代表trajectory比較でcandidate/明示drag COMSOLの3刻み収束、全frameのlifecycle/finite mask exact、
  共通有限stateの`r,z,Z`、共通active stateの`v`、141 event/fate identityをPASSした。
  P21と明示scopeの2D critical benchmarkは`CLOSED_ACCEPTED_WITH_LIMITATIONS`である。
- P07の現行sliceでsurface measure、fixed position/velocity/time、model weight、velocity-sign departure、
  law組合せと「interaction cap後に次hitがある時だけresidual intervalをsplitする」C09規則を固定した。
  seed、RNG revision、event/law解決値もmanifestへ含めた。一定加速度のone-sided certificateはevent v6、
  一般RK4のstrict-inward interior-facet certificateとsingle-facet active-boundary residualはevent v7 / engine v11で完了した。
  force-coupled RZはevent v8 / engine v12で完了し、richer distribution、moving wallは後続判断とする。
- P08でparticle-local failureとrun-fatal failureの境界、failure reason集計、小型series/probe、薄いCLIを確定した。
- P06-S/P08の再実行可能な受入条件は`solver/docs/stage1a_validation.md`へ集約し、同じ説明を各owner文書へ
  重複させない。
- P09でmemory limitをsolver-owned predicted peakとして定義し、OS hard RSS capとは分離した。
  10k/100k/1Mのfresh/warm load/prepare/run RSSは手動performance scriptで測る。P10/P14は同じ
  hardware、mesh、粒子数、出力量、warm-up条件を使ってplanner差とbottleneckを評価する。
- P10でfield/physics/RK4 array passをNumbaへ置換し、scalar evaluatorをverification oracleへ限定した。
  accepted endpoint hint、cold/warm semantic/RSS harness、非gating speedupを固定し、P1/Q1 BVHはprofileで
  必要になるまで延期した。初回stageのhintなしfull searchを稀とは仮定せず、P14のrealistic mesh profileで
  支配的と確認してfield v3へ一つのsupported-containment BVHを追加した。outside/masked provisionalはO(cell数)の
  exact fallbackを維持する。P12/P13/P14をP10完了へ遡及させない。
- P11でphysics runtimeの同じcompiled passから線形drag rate、target velocity、加算加速度を返し、
  start half-step predictorとmidpoint係数による指数更新を同じ`StepProposal`へ追加した。C03一定係数、
  極小/極大`h/tau`、可変係数次数1.8以上を独立referenceで固定し、材料wall/residual、RZ axis、
  `state_at()`、output scheduleを既存event loopへ統合した。曲線/chord偏差は全短縮secantを含むvelocity
  enclosureからmethod-neutralに作り、surface departure後の同面再衝突、Stokes一定primitive閉形式、C03全frameを
  回帰に加えた。RK4の`dt/tau`gateだけをRK4へ残し、P11 closeout時点では
  非零charge rateを未対応として明示拒否した。第二engine、COMSOL専用path、別result形式は追加していない。
  event-heavy parallelの初版はP12、durable resultはP13、synthetic全規模性能判断はP14で完了した。P14-Pでparallel runtimeを収束させ、P14-Uでtarget-use判断を完了した。

これらを製品全体の質問票や汎用設定systemへせず、対応packageに必要な最小判断だけを追加する。

---

## 20. 完了の定義

### v0.1完了

- 三つの公開APIだけでstatic RZ caseを最後まで実行できる。
- C01～C10と公開scenarioが合格する。
- RK4、exponential midpoint、P1/Q1、first hitの収束が確認される。
- exact-mesh P1/Q1材料domain、RZ axis splitを含むforce coupling、Stokes–Cunninghamの適用域/oracleが
  それぞれ独立scenarioで合格する。
- surface release、stick/escape/specular/probabilityがevent logで追跡できる。
- output scheduleとslab幅変更でdeterministic final/event/RNGが変わらない。
- 10^6粒子caseがmemory plan内で動作し、途中停止から重複・欠落なくresumeできる。
- analysis/visualizationがResultViewだけから代表結果を作れる。
- COMSOLなしで上記をverifyできる。

P14完了によりsynthetic solver-core、P14-U完了により代表用途の数値・utility・直列性能baselineを満たした。
T03も`ResultView`だけから代表集計・軌道・event可視化を生成して完了した。receipt固定のremote Windows/Linux release
workflowも成功したため、P14-Rと`0.1.0.dev0`開発baselineの配布可能性closureは完了である。正式版packageの公開ではない。
synthetic baseline、target utility、配布gate、
外部tool完了を混同しない。

正式版`0.1.0`は、このbaselineと同じsingle-engineへ受理済みの現行static 2-D機能を固定する。
`v0.1.0` tagのWindows/Linux release gateが双方成功した場合だけ、検証済みwheelとSHA-256を
GitHub Releaseへ公開する。PyPI、sdist、3-D、時間依存場、COMSOL比の速度保証はこのreleaseへ含めない。

### 後続stage完了

各stageは、該当する解析解・scenario・performanceの三層が揃い、前stageの経路を複製せず同じengineへ
追加された時だけ完了とする。COMSOL一致は外部V&Vの有力な証拠だが、core completionの代替にはしない。
