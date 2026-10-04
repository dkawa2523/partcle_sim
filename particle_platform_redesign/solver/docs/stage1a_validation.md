# Stage 1A closeout and P09 runtime acceptance

この文書はP09完了時点のengine v15/runtime layout v1/memory plan v1を凍結した履歴closeoutである。
現行P10/P11 Stage 1B closeoutは[`stage1b_validation.md`](stage1b_validation.md)を参照する。

この文書はP06-S/P08のStage 1A実装区切りとP09 runtime layoutを再実行可能な受入条件へまとめる。
COMSOL比較reportではなく、production solver自身の数値意味と公開境界のcloseoutである。

## 確定したrevision

- engine: `coupled_rk4_engine_v15`
- event: `line_quadratic_rk4_axis_first_hit_v8`
- proposal / enclosure: `coupled_rk4_proposal_v3` / `rk4_global_abs_enclosure_v1`
- physics catalog / runtime: `deterministic_xy_rz_catalog_v3` / `deterministic_physics_runtime_v1`
- result algorithm: `coupled_state_result_v2`。case/result schema versionは1のまま
- CPU runtime layout / memory plan: `resident_soa_microtile_v1` / `solver_owned_memory_plan_v1`

## P06-S acceptance

`stokes_cunningham_allen_raabe_air_v1`は

```text
Kn_a = 2 lambda / d
Cc = 1 + Kn_a * (1.142 + 0.558 exp(-0.999 / Kn_a))
dv/dt = 3 pi mu d / (Cc m) * (u_g - v)
```

を固定し、係数と平均自由行程規約をcase producerの明示責務にした。根拠は
[Allen--RaabeのNIST評価](https://nvlpubs.nist.gov/nistpubs/jres/111/4/V111.N04.A01.pdf)である。
Epsteinとの自動切替はない。

受入scenarioは、一定primitiveの解析的線形緩和に対する`h/h/2/h/4`の位置・速度次数3.5以上、
frame schedule間のfinal bitwise同一、away-axis XY/RZ一致を要求する。Kn範囲外は公開APIで
`model_applicability`、連続Re boundだけがoverflowする混在batchは該当粒子だけ`nonfinite_physics`となり、
survivorは継続する。Re/Kn境界式は純粋unit verificationでも固定する。

## P08 acceptance

粒子へ局在できる動的失敗はfailure eventへ保存し、入力、共有field/model/bound、topology、I/Oの失敗は
run-fatalのままにした。failure reasonはevent budget、indeterminate event/boundary/departure、field support、
model applicability、nonfinite physicsの7種類である。all-failed runも正常にpublishし、finalはrun終了時刻の
snapshot、最後の物理時刻はboundary/failure eventをauthorityとする。

出力はfailure event、macro-step lifecycle count、明示particle/time probeだけを既存streamへ追加した。
全stage/force/refinement traceは追加していない。CLIは`check`=`load_case`、`run`=`load_case + simulate`、
`inspect`=`open_result`の薄い三commandであり、第二の検査・実行経路を持たない。

## P09 acceptance

`load_case`はYAML/resourcesをHDF5 payloadより先に確定し、canonical numeric array footprintが上限を
超えるcaseをmetadata scanだけで拒否する。source realizationは最終scheduleへ直接scatterし、
resident stateはparticle ID対応を固定、別のresident-row active indexだけをin-place stable compactする。

prepareはsolver-owned load/prepare/run phase peakとcomponent内訳からbounded microtileを自動選択する。
最小1粒子のscratchも入らないcaseは運動開始前に拒否する。memory limitの違いによってtile幅が変わっても、
final/event/RNG/outputが一致するpublic-API scenarioを要求する。`memory_limit_mb`はOS hard RSS capではない。
10k/100k/1Mのfresh/warm load/prepare/run peak RSSは手動`tests.performance.p09_memory`で測り、
solver planとprocess RSSを分けて報告する。

## Quality gate

closeoutは次をすべて満たした時だけ完了とする。

```console
uv lock --check
uv run --locked ruff format --check src tests
uv run --locked ruff check src tests
uv run --locked pyrefly check --summarize-errors
uv run --locked lint-imports
uv run --locked python scripts/check_complexity.py
uv run --locked pytest -q
```

2026-09-27のP09 closeoutでは全gateが合格し、test suiteは265件合格した。
metadata preflight、memory rejection、stable active compaction、tile identity、manifest planの小型回帰を追加済みである。
import-linterは3契約を保持し、
Radon gateは全function/methodで`CC < 16`を満たした。

scenarioは三公開APIを通り、COMSOL、`model_dataset/`、`old_code/`へ依存しない。

## 残す制約と次gate

boundaryless P1/Q1、continuous charge、時間依存field、3D、richer source distribution、moving wall、
dragの自動blendは未解禁である。P09のper-layout hintは現行locatorが消費しないため先行配列化せず、
当時の次gateだったP10でcompiled CPU kernelと同時に、実際に消費するP1/Q1 strict-interior hintだけを追加した。
P10の実装はwalk/BVHを追加せず、Numba内full-search fallbackを使う。性能値が出る前にGPUや第二engineを追加しない。
