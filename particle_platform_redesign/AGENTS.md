# AGENTS.md

この規約は`particle_platform_redesign/`以下に適用する。`report_app/`には専用の`AGENTS.md`があり、
可視レポートを変更する時はそちらも読む。新solverの実装者は本規約を最初に読み、`old_code/`へ退避した
旧`AGENTS.md`を新設計へ適用しない。

## 1. 目的

外部で計算された場とgeometryを入力し、半導体製造チャンバー内の粒子軌道、粒子電荷、境界eventを
高速かつ再現可能に計算する基盤を作る。

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
| 方程式、物理model、数値method、適用域 | `technical_research.md` |
| file責務、work package、実装順、完了条件 | `implementation_plan.md` |
| P14-Pの並列評価、否定的closeout、直列runtime方針 | `solver/docs/parallel_execution_plan.md` |
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
- coreから`tools/`、COMSOL、比較器、analysis、visualizationをimportしない。
- adapterはcanonical形式へ変換し、producer専用のsolver分岐を作らない。
- `model_dataset`をcore testのgolden truthにしない。

## 5. 公開境界

利用者向けPython APIは次の三つだけとする。

```python
load_case(path)
simulate(case, output)
open_result(path)
```

CLIの`check`、`run`、`inspect`もこの三操作を呼び、別の検査・実行経路を作らない。公開例外は原則
`CaseError`、`SimulationError`、`IncompleteResultError`の三分類に留める。

importer向けcanonical writerは`case_format`が所有するが、package rootの利用者向けAPIには増やさない。
`PreparedRun`、backend、writer、preflight reportを公開しない。

## 6. file責務とdependency

詳細な所有表は`implementation_plan.md`を権威とする。特に次を守る。

| owner | 所有する | 所有しない |
|---|---|---|
| `case.py` | 設定、正規化、`SimulationCase` | 物理式、kernel |
| `case_format.py` | canonical read/write/version/hash | producer固有名、軌道計算 |
| `coordinates.py` | XY/RZ/3D基底とaxis規則 | field locate、wall law |
| `geometry.py` | containment、BVH、first-hit query | 付着、反射、field値 |
| `fields.py` | locate、補間、support | mesh修復、cache生成 |
| `sources.py` | table/surface release、時刻、weight | lifecycle、再飛散 |
| `rng.py` | counter key、uniform/normal、stream ID | source/wall policy |
| `physics/` | sample済みprimitiveからrate/加速度 | mesh探索、I/O、lifecycle |
| `integrators.py` | stage積分、`StepProposal` | BVH、boundary law、出力 |
| `events.py` | earliest hit、残時間work | hit後の物理応答 |
| `boundaries.py` | stick/escape/hold/reflection/probability | 交差探索、位置押戻し |
| `engine.py` | prepare、唯一のproduction loop | importer、plot、COMSOL比較 |
| `cpu.py` | SoA、bounded slab、compiled kernel、memory plan | model意味、永続形式 |
| `output.py` | epoch、checkpoint、`ResultView` | 科学集計、描画 |

`models.py`、`helpers.py`、`utils.py`、`contracts.py`のような一般置き場を作らない。型と関数は、その事実を
所有するmoduleへ置く。fileは行数だけで分割せず、独立した変更理由が二つ現れた時に分ける。

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
- 粒子IDは固定し、compact配列のindexをidentityにしない。
- output frameはaccepted `StepProposal.state_at()`から評価し、resident stateを出力時刻へcommitしたり
  production stepを分割したりしない。曲線pathの`state_at()`はproposal始点から同じintegrator規則で再評価する。
- wall eventは最初のhitだけを応答へ渡す。corner policyは明示する。
- zero-time surface departureを位置nudgeで解決しない。
- RNGはparticle ID、model stream、物理的draw ordinalから決める。
- tile幅、出力頻度、checkpoint/resumeでdrawとeventを変えない。
- stochastic stepを分割する場合、独立乱数を引き直さず条件付き分割を使う。

## 9. 性能実装

- 最初にscalar/NumPyの正しい縦切りを完成させ、その意味論を同じengine内でcompiled CPUへ移す。
- production hot loopへparticleごとのPython callbackを入れない。
- v0.1は全particle state resident、scratchだけbounded slabとする。
- outer particle batchやout-of-core stateは実測上必要になるまで作らない。
- production computeはP14-Pで確定したsingle-thread compiled engine一つにする。outer
  `ThreadPoolExecutor`、Numba内部parallel mask、multiprocessing backendをsolver内部へ再導入しない。
- 粒子更新はbounded slab上のflat SoA wavefrontで行い、field/geometryをread-onlyとして再利用する。
- 曲線/exact eventの継続状態はPythonのlist、dict、dataclass queueではなく数値配列で保持する。各rowは独立した
  target time、residual、refinement depth、interaction/event ordinalを持ち、一roundに高々一件のevent/failureを
  bounded bufferへ書く。
- scratch容量はtile幅から決めて一度確保し再利用する。workerごとの全量proposal/event buffer、thread IDに依存する
  private APIを導入しない。
- event、failure、frame、probe、finalはparticle ID・時刻・物理ordinalによるstable orderで出力し、tile幅、
  実行順でbitwise payloadとRNG drawを変えない。
- 最適化前後でtrajectory/event identity、時間、peak memoryを同じcaseで測る。
- GPUはCPUと同じcase、model revision、RNG、event、result schemaを使い、end-to-end speedupを測る。

P12/P14のouter worker-waveとP14-PのNumba内部parallel試行は履歴baselineであり、製品機能ではない。P14-Pの
focused correction後もregular 1Mは1/2/4 threadで9.32/10.09/10.10 s、4-thread speedup 0.923xに留まり、
v20の1-thread 7.534 sに対しても23.7%遅かった。end-to-end profileではPython/NumPyのproposal/enclosure調停が
支配し、kernel単体の3.75xを製品価値へ変換できなかったため、case schema v2は`resources.threads`を削除し、
engine v27をsingle-thread compiled runtimeへ一本化した。詳細は`solver/docs/parallel_execution_plan.md`を権威とする。

独立caseのprocess並列はsolver外の運用とする。再検討は代表用途profileが別work packageの必要性を示した時だけ行い、
自動tuner、experimental flag、第二scheduler、互換aliasで失敗経路を復活させない。

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
uv run --locked ruff format --check src tests tools
uv run --locked ruff check src tests tools
```

Ruffの`C90`を使わず、complexityはRadonへ一元化する。blanket `noqa`や全directory ignoreを作らない。
auto-fixは編集対象へ限定し、数値式を読みにくいhelperへ分けてlintを通さない。

### import-linter

import-linterをdependency方向の唯一の自動検査にする。同じ内容のarchitecture testをpytestで作らない。

```console
uv run --locked lint-imports
```

v0.1の契約は最大三つを目安とし、physics純粋性、integrator純粋性、低位numericsからouter layerへの
逆依存禁止だけを固定する。契約を満たすためのempty moduleやfacadeを作らない。

### Pyrefly

Pyreflyを型検査の唯一のtoolにする。mypyやPyrightを重ねない。

```console
uv run --locked pyrefly check --summarize-errors
```

公開API、dataclass、model境界、array ownerを型付けする。shape、unit、物理適用域を巨大な型体系へしない。
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
uv run --locked ruff format --check src tests tools
uv run --locked ruff check src tests tools
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
- `tests/scenarios/`：三公開APIを通る小型end-to-end
- `tests/performance/`：規模、memory、直列throughput。独立caseのprocess scalingは必要時だけcore外で測る

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
- 公開schema互換が必要な場合だけ、終了versionと削除条件を明示する。

新旧経路を一時併存させる必要がある場合、そのchangeを未完了と扱う。次の通常featureを積む前に統合または
削除する。

## 15. 診断と出力

常設diagnosticsは次に限定する。

- lifecycle、event、failureの件数
- step、event refinement、I/Oの集約timing
- 予測またはpeak memory
- 小さいfailure reason
- resolved model/schema/algorithm revision

full trace、stage値、局所probeは明示particle IDだけで有効化する。診断のために別engine、別state配列、
別result形式を作らない。analysis、deposition、可視化、COMSOL差分はcore外でResultViewを読む。

## 16. 完了条件

変更は次を満たした時だけ完了とする。

- 目的と所有moduleが一意である。
- 物理・数値結果を検証するcaseがある。
- 三公開APIとsingle engineを壊していない。
- `uv.lock`が意図したdependencyと一致する。
- Ruff、import-linter、Pyrefly、Radon gateが合格する。
- 該当verificationと公開API scenarioが合格する。
- tile、出力scheduleに関わる変更ではidentity不変性を確認した。
- hot path変更では同一条件のbefore/after時間とmemoryを測定した。
- 置換されたcode、test、設定、文書が削除されている。
- schemaまたはmodelの科学的意味が変わる場合はrevisionを更新した。
- COMSOLがなくても正しさと制約を説明できる。

lint、型、coverageの合格だけで数値実装を完了扱いにしない。
