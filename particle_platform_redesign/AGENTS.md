# AGENTS.md

この規約は`particle_platform_redesign/`以下に適用する。`report_app/`には専用の`AGENTS.md`があり、
可視レポートを変更する時はそちらも読む。新solverの実装者は本規約を最初に読み、`old_code/`へ退避した
旧`AGENTS.md`を新設計へ適用しない。

## 1. 目的

利用者が機能を選び、自分の外部場・geometry・粒子源を準備し、検査・計算・結果の解釈と再利用まで
CLIとPythonで完結できる粒子輸送基盤を作る。通常の利用は配布wheelだけで完結させる。

優先順位は次のとおり。

1. 物理・数値的に正しい。
2. 各事実と状態の所有者が一つである。
3. 10^4～10^6粒子へ拡張できる。
4. 第三者が読み、修正できる。
5. 機能追加で旧経路、契約、診断frameworkを増殖させない。

COMSOLは入力adapterと外部V&Vの一つであり、solver coreの目的でも依存先でもない。COMSOLがなくても、
解析解、manufactured case、収束試験によりcoreの正しさを説明できなければならない。

## 2. 実装しないもの

次を安易に導入しない。

- 旧solverのclass、helper、fallback、設定構造の移植
- COMSOL専用solver経路
- 一般plugin framework、DI container、抽象base class階層
- 汎用ODE/DAE framework
- 複数の同等runtime、永続形式、設定parser
- 常時full trajectory、巨大診断report、診断code体系
- private関数、helper名、directory形状を固定するtest
- 実利用例が一つしかない段階での一般化

## 3. 文書の権威と読む順序

| 事実 | 権威文書 |
|---|---|
| 製品目的、範囲、非目標、stage | `product_specification.md` |
| module責務、dependency、runtime境界 | `architecture_proposal.md`（権威）、`architecture_review.md`（review記録） |
| 現行の方程式、物理model、数値method、適用域 | `solver/docs/physics_models.md`、`solver/docs/numerics.md` |
| 導出・文献調査・研究根拠 | `technical_research.md` |
| file責務、work package、実装順、完了条件 | `implementation_plan.md` |
| 同一runのprocess並列、資源境界、現行の性能測定 | `solver/docs/parallel_execution_plan.md` |
| 品質toolとgate | `quality_tooling_plan.md` |
| COMSOL比較 | `vv_methodology.md` |
| datasetの許可用途 | `data_quality_assessment.md` |

矛盾を局所fallbackや条件分岐で吸収しない。事実の所有文書を直してから実装する。実装詳細で
`implementation_plan.md`とarchitecture文書が衝突する場合は、目的を変えずに両方を同じchangeで整合させる。

## 4. clean-room境界

新solverは`particle_platform_redesign/solver/`の独立uv projectとして実装する。

- 旧`particle_tracer_unified`をdependencyにしない。
- 旧コードをcopyしない。
- 旧コードは失敗例、式候補、既知caseの調査対象としてのみ読む。
- 採用する式や考え方は独立した根拠とmicrocaseで再検証する。
- 数値core/APIから`chamber_particles_tools`、`tools/`、COMSOL、比較器、analysis、visualizationをimportしない。
- 同じuv projectの`src/chamber_particles_tools/`が通常の入力作成・場変換/生成・解析・表示と唯一のCLIを所有する。
  consoleと`python -m chamber_particles`はそのdispatcherへ接続する。live COMSOLと開発V&Vは外部に置く。
- adapterはcanonical形式へ変換し、producer専用のsolver分岐を作らない。
- `model_dataset`をcore testのgolden truthにしない。

## 5. 公開境界

計算coreの公開Python APIは次の四つとする。通常clientの公開操作は`chamber_particles_tools`が所有する。

```python
load_case(path)
check_case(case, *, workers=1)
simulate(case, output, *, workers=1)
open_result(path)
```

`simulate`の実行資源keyword `workers=1`でprocess数をAPI/CLIから指定し、
caseの数値設定へ混ぜない。

`check_case`は実行と同じprepareを使い、小さいimmutableな解決結果・制限・memory planを返す。
worker、writer、粒子更新を開始せず、mutableなPreparedRunは公開しない。
CLIの`check`、`run`、`inspect`も同じ操作を呼び、別の検査・実行経路を作らない。公開例外は原則
`CaseError`、`SimulationError`、`IncompleteResultError`の三分類に留める。

importer向けcanonical writerは`case_format`が所有するが、package rootの利用者向けAPIには増やさない。
`PreparedRun`、backend、writer、mutableなpreflight workspaceを公開しない。

## 6. file責務とdependency

詳細な所有表は`implementation_plan.md`を権威とする。特に次を守る。

| owner | 所有する | 所有しない |
|---|---|---|
| `case.py` | 設定、正規化、`SimulationCase` | 物理式、kernel |
| `case_format.py` | canonical read/write/version/hash | producer固有名、軌道計算 |
| `time_grid.py` | 固定macro grid・field knot併合・丸め規則と窓の照会 | 粒子別root、観測clock、精度認定 |
| `lifecycle.py` | 保存code・名称・基本状態分類 | 表示色、付着の物理認定、failure原因 |
| `coordinates.py` | XY/RZ/3D基底とaxis規則 | field locate、wall law |
| `geometry.py` | containment、BVH、first-hit query | 付着、反射、field値 |
| `topology.py` | XY periodic translation対応とtransfer | material wall law、一般periodic framework |
| `fields.py` | locate、補間、support | mesh修復、cache生成 |
| `sources.py` | canonical internal/surface scheduleのrelease、位置解決、時刻、weight | 分布sampling、source RNG、lifecycle、再飛散 |
| `rng.py` | counter key、uniform/normal、stream ID | source/wall policy |
| `physics/` | sample済みprimitiveからrate/加速度 | mesh探索、I/O、lifecycle |
| `integrators.py` | stage積分、`StepProposal` | BVH、boundary law、出力 |
| `stochastic.py` | joint OU、conditional split、stochastic tree | wall law、出力、第二engine |
| `events.py` | earliest hit、残時間work | hit後の物理応答 |
| `boundaries.py` | stick/escape/hold/reflection/probability | 交差探索、位置押戻し |
| `engine.py` | prepare、親時計・release・checkpoint・single writer | importer、plot、COMSOL比較 |
| `transport.py` | serial/process共通の粒子slab更新 | 親時計、writer、process管理 |
| `execution.py` | worker、共有transport、bounded IPC・ACK | 物理式、第二engine、worker側writer |
| `cpu.py` | SoA、bounded slab、compiled kernel、memory plan | model意味、永続形式 |
| `output.py` | epoch、checkpoint、immutable解釈context、`ResultView` | 科学集計、描画 |
| `chamber_particles_tools` | 通常入力publisher、解析・export・表示、唯一のCLI | 数値coreの再実装、native専用engine |

`models.py`、`helpers.py`、`utils.py`、`contracts.py`のような一般置き場を作らない。型と関数は、その事実を
所有するmoduleへ置く。fileは行数だけで分割せず、独立した変更理由が二つ現れた時に分ける。

catalogがfieldのfinite/positive/nonnegative条件、選択model identity、証明に必要な入力roleを宣言する。
runtimeは全域bound・局所適用域・連成を準備し、compiledは固定形の数値recordだけを消費する。
stage値が計算可能であることをpathの証明やconstant specializationの成立へ読み替えない。
親の入力/provenanceと解決済み実行事実をworkerへ渡さず、transportは必要なtyped settings/配列だけを持つ。

依存方向：

- `physics`はsample済みprimitiveだけを受け、geometry、fields、I/Oをimportしない。
- integratorはstage evaluatorだけを呼び、geometry、event、boundary、I/Oを知らない。
- geometryとfield layoutは同一meshを前提にしない。
- `cpu.py`はoutputへ書かず、engineがcomputeとwriterを調停する。
- analysis、visualization、V&Vは`ResultView`またはcanonical artifactだけを読む。
- import違反をfacade、re-export、TYPE_CHECKING importで隠さない。

## 7. 数値計算の不変条件

以下を変更時に必ず維持する。

- canonical値とruntime値はSIを使い、量名へ単位を含める。
- `mass_kg`を慣性のauthorityとし、実行中に粒径から再構成しない。
- drag径、electrostatic半径、排除体積、model weightを混同しない。
- fieldの値とsupport判定を分ける。
- trial値を有限化しても、support外proposalをeventなしで受理しない。
- safety failureとaccuracy不足を同じstatusにしない。
- 欠損、非有限、適用外modelをdefault、epsilon clamp、別modelで隠さない。
- 固定長の位置押戻しや散在するmagic toleranceを追加しない。
- toleranceはgeometry scale、時間、速度、float64 roundoffから一箇所で解決する。
- field、force、chargeは実際のintegrator stage位置と時刻で評価する。
- continuous chargeと運動を同じ状態として連成する。
- chargeだけをsubcycleして後から運動へ渡さない。
- accepted stepだけがparticle stateとcell hintをcommitする。
- first hitをendpointだけでなく`StepProposal`の経路上で求める。
- hit状態は同じintegratorでhit時刻まで再評価し、残時間を再実行する。
- 要求出力時刻のためにproduction stepを分割しない。
- 個別release時刻のために全粒子stepを分割しない。
- RZ meridionalのaxis crossingをwall reflectionとして扱わない。
- point particleと有限半径接触をtoleranceで混同しない。
- 数値式、補間、event順、tolerance、RNG keyの変更をrefactorと呼ばない。

## 8. state、event、RNG

- physical state、lifecycle、support、accuracy、run integrityを一つのstatusへ畳み込まない。
- 永続particle列はresident stateと同じ宣言からprojectionし、checkpointのcursor/index/hintは各ownerで検査する。
- 粒子IDは固定し、compact配列のindexをidentityにしない。
- output frameはaccepted `StepProposal.state_at()`から評価し、resident stateを出力時刻へcommitしたり
  production stepを分割したりしない。曲線pathの`state_at()`はproposal始点から同じintegrator規則で再評価する。
- wall eventは最初のhit時刻だけを応答へ渡し、その局在budget内で同時なincident facet集合を保持する。
  exact、一般RK4、exponential、Brownianで同じ候補集合、corner policy、combined normalを使う。
- zero-time surface departureを位置nudgeで解決しない。
- RNGはparticle ID、model stream、物理的draw ordinalから決める。
- tile幅、出力頻度、checkpoint/resumeでdrawとeventを変えない。
- stochastic stepを分割する場合、独立乱数を引き直さず条件付き分割を使う。
- Brownianの`interval_tree_depth`は全rootへ一様に適用する数値path精度のauthorityとする。
  `adaptive_max_depth`はそのpathがwall、RZ axis、または証明不能区間の候補になった時だけ条件付き分割する上限であり、
  base depthを置き換えない。両者が同じなら従来の固定depth pathへ退化する。
- 有限depthのcubic Hermite pathはsampleしていない連続OU軌道のexact first-passage、zero miss probability、
  またはbase depth不足の自動補償を主張しない。

## 9. 性能実装

- 最初にscalar/NumPyの正しい縦切りを完成させ、その意味論を同じengine内でcompiled CPUへ移す。
- production hot loopへparticleごとのPython callbackを入れない。
- 現行は全particle state resident、scratchだけbounded slabとする。
- outer particle batchやout-of-core stateは実測上必要になるまで作らない。
- 既存数値ownerを使い、同一runをCPU processで並列実行する。追加JIT化は今回の前提にしない。
  計算engineは一つとし、親がmacro/release/output/checkpoint、永続workerが排他的particle slabを所有する。
  `ThreadPoolExecutor`、Numba内部parallel mask、model別engine、backend選択frameworkは導入しない。
  共有入力・state/replay、worker専用scratch、IPCと移送時の同時live容量を一つの予算へ入れる。
- 親coordinatorは全workerのconnection/sentinelを監視し、完了したworkerへ次の排他的slabを渡す。
  event/failureはworkerごとに高々一batchを送信し、親single writerの同期消費後にACKする。
  storage batchは到着順を許し、公開canonical event順は既存output reader一箇所が所有する。
  macro-wide並べ替えbuffer、別scheduler、worker側HDF5 writerを追加しない。
- private scratch templateは全workerがcopyを終えたready barrier後に解放し、prepared transportをsealする。
  非空replayはrun所有の一shared rootとworkerの永続readerを使い、macro/slabのviewだけを再利用する。
  全taskと同期output消費後にだけviewを解放し、容量・serialization・page paddingの会計を省かない。
- 粒子更新はbounded slab上のflat SoA wavefrontで行い、field/geometryをread-onlyとして再利用する。
- 曲線/exact eventの継続状態はPythonのlist、dict、dataclass queueではなく数値配列で保持する。各rowは独立した
  target time、residual、refinement depth、interaction/event ordinalを持ち、一roundに高々一件のevent/failureを
  bounded bufferへ書く。
- scratch容量はtile幅から決めて一度確保し再利用する。workerごとの全量proposal/event buffer、thread IDに依存する
  private APIを導入しない。
- Brownian stochastic-tree scratchは実行時に到達し得る`adaptive_max_depth`から保守的に計画し、
  実際にrefineしたrow数が少ないことを根拠にmemory planを過小評価しない。
- event、failureの公開canonical reader、frame、probe、finalはparticle ID・時刻・物理ordinalによるstable orderを保ち、tile幅、
  実行順でbitwise科学payloadとRNG drawを変えない。境界eventのbounded storage batchは既存result契約どおり大域sortを保証せず、
  その保存順を物理順のauthorityにしない。completion順の受信にmacro-wide並べ替えbufferを追加しない。
- 最適化前後でtrajectory/event identity、時間、peak memoryを同じcaseで測る。
- GPUはCPUと同じcase、model revision、RNG、event、result schemaを使い、end-to-end speedupを測る。

実行資源と性能測定は`solver/docs/parallel_execution_plan.md`を参照する。

独立caseのprocess並列はsolver外の運用とする。同一runは明示workers数で実行し、追加JIT化を前提にしない。
自動tuner、experimental flag、第二scheduler、互換aliasで失敗経路を復活させない。

過去の用途matrixと配布資格は、その実行sourceに限定する。現行変更の品質・性能・配布は変更後の実行receiptで確認する。
品質・用途性能・source別bindingと制約は性能文書が所有する。
Git branch push、remote CI、tag、Release/registry公開の実施状況はrelease記録が所有し、local gateから推定しない。
追加JIT、NUMA、全CPU数比例効果を今回の必須工程へ追加しない。

全worker同期は1 macroごとの固定仕様であり、同期間隔設定はない。
複数macro連続実行は`implementation_plan.md`の将来項目とし、dt、field/release時計、
RNG、毎macro集計、bounded output、single writer、accepted checkpoint/resumeを維持して評価する。
必要な利用と検証が定まる前に設定やstubを先行追加しない。

## 10. 開発tool

詳細は`quality_tooling_plan.md`を読む。

### uv

uvをPython、dependency、lock、環境、command実行の唯一の基盤にする。

- `pyproject.toml`と`uv.lock`をauthorityとする。
- `pip install`、Poetry、Conda、手書きrequirementsを並存させない。
- 通常commandは`uv run --locked ...`、CI同期は`uv sync --locked`を使う。
- `--frozen`を標準にしない。lockの鮮度を検査する。
- dependency upgradeとfeature変更を混ぜない。
- project toolを`uvx`で都度取得しない。

### Ruff

Ruffをformat、lint、import整形の唯一のtoolにする。Black、isort、Flake8を追加しない。

```console
uv run --locked ruff format --check src tests tools scripts examples
uv run --locked ruff check src tests tools scripts examples
```

Ruffの`C90`を使わず、complexityはRadonへ一元化する。blanket `noqa`や全directory ignoreを作らない。
auto-fixは編集対象へ限定し、数値式を読みにくいhelperへ分けてlintを通さない。

### import-linter

import-linterをdependency方向の唯一の自動検査にする。同じ内容のarchitecture testをpytestで作らない。

```console
uv run --locked lint-imports
```

import契約はphysics純粋性、integrator純粋性、低位numericsの境界、計算APIから通常clientへの
逆依存禁止を固定する。module入口だけがCLIを呼び、計算APIは外層をimportしない。
契約を満たすためのempty moduleやfacadeを作らない。

### Pyrefly

Pyreflyを型検査の唯一のtoolにする。mypyやPyrightを重ねない。

```console
uv run --locked pyrefly check --summarize-errors
```

公開API、dataclass、model境界、array ownerを型付けする。shape、unit、物理適用域を巨大な型体系へしない。
現行の性能clientも同じPyrefly対象へ含め、数値ownerの変更でbenchmark入口が静かに陳腐化しないようにする。
clean-room projectではbaselineを作らない。ignoreは最小行へerror codeと理由を書く。

### Radon

Radonを関数complexityの検出に使う。

```console
uv run --locked radon cc src/chamber_particles tools -s -a
uv run --locked radon mi src/chamber_particles tools -s
```

新規・変更functionは原則CC 10以下、数値上分割しない方が明確な場合でも15以下とする。CC 16以上を
`scripts/check_complexity.py`で拒否する。MIは参考値でありgateにしない。score低下だけを目的に薄いhelper、
manager、facadeを作らない。

## 11. 標準quality gate

通常changeでは安い順に次を実行する。

```console
uv lock --check
uv run --locked ruff format --check src tests tools scripts examples
uv run --locked ruff check src tests tools scripts examples
uv run --locked pyrefly check --summarize-errors
uv run --locked lint-imports
uv run --locked python scripts/check_complexity.py
uv run --locked pytest tests/verification tests/scenarios -q
```

これを包む独自runnerを作らない。mainでは全scenarioとwarm performance smoke、release/manualではclean
install、wheel、Windows/Linux、cold JIT、failure injection、10^4/10^5/10^6粒子を追加する。COMSOL比較は
外部V&Vでありcore PRの常時gateにしない。

## 12. change workflow

1. 対象work packageと所有文書を読む。
2. 解く利用case、数値的期待値、所有module、削除対象を決める。
3. 解析解、manufactured case、または小さいreferenceを定める。
4. 一つのproduction経路へ最小実装する。
5. 該当verificationと小さい公開API scenarioを通す。
6. hot pathならprofileし、計測結果に基づいて最適化する。
7. 置換した実装、設定、test、文書を同じchangeで削除する。
8. 所有文書と必要な一つのdecision recordだけを更新する。

新model、座標、integrator、backendを追加する前に`solver/docs/decisions.md`へ次だけを書く。

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

長いADR tree、承認workflow、機能ごとのdesign documentを増やさない。

## 13. test規約

testは三層だけにする。

- `tests/verification/`：解析解、manufactured solution、収束、event geometry
- `tests/scenarios/`：core公開APIと通常clientを通る小型end-to-end
- `tests/performance/`：規模、memory、同一runのprocess数別throughput・科学identity・cleanup。独立caseのprocess scalingは必要時だけcore外で測る

次をtestしない。

- private helper名や呼出し順
- file配置や行数
- mockだけで再現した内部構造
- 診断文言の大量snapshot
- coverage率を上げるためだけのbranch
- COMSOL固有挙動をcoreの正解とするtest

bug修正では症状ごとではなく、原因を表す最小回帰caseを一つ追加する。全performance suiteや10^6粒子を
通常changeごとに実行しない。

## 14. 古い処理を残さない

- `legacy`、`new`、`v2`、`temporary`という並列production経路を作らない。
- call siteを新経路へ切り替えるchange内で旧経路を削除する。
- commented-out code、到達不能fallback、未使用feature flagを残さない。
- 内部APIへcompatibility shimを置かない。Git履歴をarchiveとする。
- stale test、fixture、config、文書も同じchangeで削除する。
- 削除後は`rg`とimport-linterで旧symbolへの参照がないことを確認する。
- 二実装の比較が必要なら片方をverificationまたは外部benchmarkに置き、production dispatchにしない。
- 現行仕様だけを受理する。過去schemaを読む互換reader、移行用dispatch、旧入口aliasは提供しない。

必要な役割は現行の所有者へ移し、呼出側・設定・検証・説明まで同時に更新する。撤去の経緯は作業記録へ置く。

## 15. 診断と出力

常設diagnosticsは次に限定する。

- lifecycle、event、failureの件数
- step、event refinement、I/Oの集約timing
- 予測またはpeak memory
- 小さいfailure reason
- resolved model/schema/algorithm revision

full trace、stage値、局所probeは明示particle IDだけで有効化する。診断のために別engine、別state配列、
別result形式を作らない。analysis、deposition、可視化、COMSOL差分はcore外でResultViewを読む。
ResultViewのmetadataは入れ子まで不変とし、JSON用の独立documentと結果identityはoutputが所有する。
通常clientは同じResultView/束縛cohortを再利用でき、ID選択・分母・coverageを勝手に再定義しない。
caseと編集元、producer bundleとreceipt、export/viewは操作単位で完成してから公開する。
macro設定・実partition・accepted path・安全証明細分・観測を区別し、全区間件数から選択窓の収束を推定しない。

## 16. 完了条件

変更は次を満たした時だけ完了とする。

- 目的と所有moduleが一意である。
- 物理・数値結果を検証するcaseがある。
- 四つのcore公開API、通常clientの公開経路、single engineが整合する。
- `uv.lock`が意図したdependencyと一致する。
- Ruff、import-linter、Pyrefly、Radon gateが合格する。
- 該当verificationと公開API scenarioが合格する。
- tile、出力scheduleに関わる変更ではidentity不変性を確認した。
- hot path変更では同一条件のbefore/after時間とmemoryを測定した。
- 置換されたcode、test、設定、文書が削除されている。
- schemaまたはmodelの科学的意味が変わる場合はrevisionを更新した。
- COMSOLがなくても正しさと制約を説明できる。

lint、型、coverageの合格だけで数値実装を完了扱いにしない。
