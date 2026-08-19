# Architecture

## 目的

このpackageは、入力解釈、数値計算、成果物保存を分離し、solverがYAML、CSV、
COMSOL、可視化の都合を知らない構成にします。設計上の基準は次の3点です。

1. 入力は境界で一度だけtypedな表現へ変換する。
2. mutableな粒子状態と、run中に変わらない資源の所有者を分ける。
3. 数値式、衝突判定、永続化を同じ処理へ混在させない。

## 全体の動線

```text
canonical YAML/CSV ──> configuration + io adapters ──> SolverContext
COMSOL manifest/export ──────────────────────────────┘
                                      |
                      +---------------+---------------+
                      |                               |
                validate_case()                  simulate()
                  preflight                    solver runtime
                                                      |
                                               SolverOutcome
                                                      |
                                              SimulationResult
                                                      |
                                               write_result()
                                                      |
                                       canonical v0.2 artifacts
```

`load_case()`は設定と入力artifactを読み、solverが直接使える`SolverContext`まで
解決します。`validate_case()`と`simulate()`はファイルを書きません。
永続化は`write_result()`だけが担当します。

`migrate`はlegacy入力からcanonical入力を作る別経路です。`compare`と`tools/`は
caseや成果物を読む外側の利用者であり、通常solverから参照されません。

## 層とファイルの責務

| 場所 | 所有する責務 |
|---|---|
| `configuration.py`, `_configuration_*.py` | 公開設定APIと、入力・物理・run文書ごとのstrict parser |
| `force_models.py`, `_force_model_*.py` | force設定の公開APIと、型・値検証・parse・serializeの各owner |
| `domain.py`, `core/` | backend非依存の値型、protocol、geometry・boundary・samplingの基本処理 |
| `io/` | canonical table、NPZ、COMSOL manifestをdomain/solver型へ変換 |
| `providers/` | nativeまたはprecomputedなgeometry/field providerの実装 |
| `preflight.py`, `_preflight_*.py` | 実行前checkの順序制御と、対象別の検証・report組立て |
| `preflight_physics.py` | force・gas・particle物性の入力源要件を検証 |
| `preflight_types.py` | 公開する`ValidationIssue`と`ValidationReport`のschema |
| `drag_validation.py` | 初期drag regimeのsampling・評価・finding生成 |
| `application.py` | `load_case`、`validate_case`、`simulate`のuse case調停 |
| `_application_types.py` | 公開case・result・artifactのimmutableな値型 |
| `_application_runtime.py` | transient time support検証とsolver outcomeから公開resultへの変換 |
| `solvers/` | 解決済みcontextから粒子状態を時間発展 |
| `writer.py`, `artifacts.py` | immutableな成果物の発行とschema検証 |
| `migration.py`, `_migration/` | 公開migration入口と、legacy解析・table・physics・書込みの各変換 |
| `comsol_case/` | 明示的なCOMSOL exportからcase artifactを構築 |
| `compare/`, `tools/`, `validation/` | V&V、可視化、性能測定 |
| `cli.py` | 単一console scriptから各use caseへ引数をrouting |

同じ物理設定を複数層で辞書として再解釈しません。native設定とCOMSOL manifestの
force指定は、入力境界で同じ`ForceModel`へ変換します。solver hot pathでは、そこから
作ったscalar/arrayのruntime parameterだけを使います。

`configuration.py`は公開型と読書き関数だけを再公開します。共通scalar検証、入力provider、
charge、その他のphysics、top-level YAML文書は対応する`_configuration_*.py`が所有し、
adapter間の排他条件とphysics間の整合性を入力境界で一度だけ検証します。
`force_models.py`も公開型・定数・変換関数だけを再公開します。`_force_model_types.py`が値型、
`_force_model_values.py`がscalarとsemantic検証、`_force_model_parsing.py`がnative/manifest入力、
`_force_model_serialization.py`が出力形式を所有します。
`comsol build-case`は書込み前に`GeometryOnlyBuild`または`RunnableBuild`へ確定します。
geometry-onlyとfield入力を同時指定した矛盾は、この境界で拒否するため、途中artifactを
残してから失敗する経路はありません。

COMSOL manifestの公開型は`io/comsol_manifest.py`に置き、YAML解析、artifact file検証、
semantic検証、値型は`io/_comsol_manifest_*.py`がそれぞれ所有します。precomputed providerも
`providers/precomputed.py`を安定した入口とし、geometry、regular grid、triangle meshの
builderは別moduleで実装します。

大きな処理は、互換性を保つ公開入口と、変更理由が一つだけのownerへ分けます。

| 公開入口 | 実装owner |
|---|---|
| `preflight.py` | 検査順とreportだけを集約。`_preflight_runtime.py`が粒子・実験機能、`_preflight_initial_state.py`がrelease時刻別field supportと初期位置の内外判定、`_preflight_boundary.py`がboundary coverage・RZ・境界上のfield support、`_preflight_comsol.py`がmanifest・時刻・part coverageを検証 |
| `io/comsol.py` | `_comsol_release_projection.py`が2D release投影、`_comsol_provider_validation.py`がquantity・ghost cell・時刻範囲を検証 |
| `io/runtime_builder.py` | `_runtime_adapter.py`がnative/COMSOL入力を解決し、`_runtime_context.py`がimmutableな`SolverContext`を組み立てる |
| `comsol_case/fields.py` | `_field_normalization.py`が軸・時刻・quantity・maskを正規化し、`_field_support.py`がfinite supportとartifactを確定、`_field_profile.py`がprofile bundleとmanifestを構築 |
| `comsol_case/mesh.py` | `_mesh_parsing.py`がMPHTXT読込み・scale・domain選択、`_mesh_topology.py`が境界topology・part割当・precomputed配列、`_mesh_artifacts.py`がentity mapとgeometry NPZを保存 |
| `comsol_case/contracts.py` | `_contract_inputs.py`がbuild入力、`_raw_export_contract.py`が外部export、`_case_contract.py`がforce・gas・manifest/config契約を検証 |
| `core/boundary_hits.py` | `_boundary_hits_2d.py`が2D edge交差・batch query・最近傍特徴、`_boundary_contact_2d.py`がcontact frameとedge投影、`_boundary_hits_3d.py`が3D triangle・polyline queryを所有 |
| `tools/compare_against_reference.py` | `_reference_compare_inputs.py`が設定・入力、`_reference_compare_metrics.py`が比較計算、`_reference_compare_runs.py`が実行・artifact発行を所有 |

入口moduleは処理順の制御または直接再公開に限定し、数値式、artifact組立て、検証を複製しません。

## 主要なデータ所有者

- `RunConfig`: 検証済みcanonical設定。
- `SimulationCase`: `RunConfig`、入力path、解決済み`SolverContext`を束ねる公開case。
- `SolverContext`: particle、field、geometry、wall、force、plan、optionsを持つ
  immutableなsolver入力。coreではplan/optionsの具体型を知らないgeneric containerとし、
  solver側の`RuntimeSolverContext`だけが具体的なplan/options型を結び付けます。
- `RunExecutionContext`: compile済みfield、boundary query、runで固定した物性などを
  step間で共有する参照束。
- `SolverState`: 位置、速度、電荷、terminal/contact mask、乱数器、scratchを持つ
  唯一のmutable状態。
- `StageFields`: ある時刻・位置集合でsampleした場とsupport状態。
- `StageFieldPlan`: drag・force・charge・stochastic motionから一度だけ解決した、
  stepで必要なfieldの集合。`resolve_stage_field_requirements()`が唯一の判定箇所です。
- `SegmentMotionRequest` / `SegmentMotionTrace`: 自由飛行segmentの入力と受理済み軌道。
- `SolverOutcome`: solverからapplicationへ渡す内部結果。
- `SimulationResult`: public APIが返すread-onlyな最終状態と集計。
- `ArtifactManifest`: 保存したfileのpath、size、SHA-256。

状態を別名のmappingや第二の配列集合へ複製せず、層を跨ぐときは上記の型で渡します。

## Solver内部

`simulate_context()`は次の順で実行します。

```text
prepare_runtime_execution
  -> field backendとboundary queryを一度だけ構築
  -> SolverStateとrelease scheduleを初期化
  -> 名目global step内でrelease時刻別のparticle cohortを構築
  -> optional charge half-step、またはcharge-motion連成trace
  -> contact中の粒子を更新
  -> mobile粒子のETD2 segmentをbatch計算
  -> motion LTEとoptional Brownian係数scheduleを解決
  -> Brownian pathを構成し、壁近傍だけdyadic first-passage探索
  -> field supportとwall collisionを分類
  -> safe state、retry、collision responseを確定
  -> optional charge half-step
  -> standard集計、debug時だけ詳細bufferを更新
finalize_runtime_execution
  -> SolverOutcome
```

主な責務境界は次のとおりです。

- `runtime_execution.py`: runtime構築・終了処理の安定した公開入口だけを再公開。
- `_runtime_execution_context.py`: run中に共有するimmutable context。
- `_runtime_preparation.py`: physics入力検証、backend compile、初期stateと実行資源の構築。
- `_runtime_outcome.py`: snapshot、memory集計、診断と`SolverOutcome`の確定。
- `runtime_state.py`: mutable粒子配列とscratchの初期化・所有。
- `high_fidelity_runtime.py`: release/step eventと各処理の実行順。
  trace refinement、charge lifecycle、collision batch反映、valid-mask停止、terminal state更新は
  `_runtime_*.py`が所有します。
- `_runtime_release_schedule.py`: 名目step境界を保ったrelease cohortと各積分区間。
- `_runtime_timing.py`: debug modeだけで使うsection timerと累積処理。
- `_runtime_trace_refinement.py`: 受理済みmotion traceのgeometry・field-support安全性判定と再計算。
- `_runtime_valid_mask.py`: valid-mask違反のprefix retryと停止結果の確定。
- `_runtime_collisions.py`: step単位のcollision分類、solver呼び出し、結果commit。
- `drag_models.py`: continuum drag law、rarefaction correction、局所物性からのrelaxation time。
- `_coupled_charge_leaf.py` / `_coupled_charge_motion.py`: electric forceとdynamic chargeを同じ
  accepted leaf・event時刻上で進める2次連成trace。
- `_stochastic_coefficients.py`: accepted motionとは独立にBrownian OU係数のleaf解像度を確定。
- `_stochastic_path.py` / `_stochastic_first_passage.py`: 再現可能なOU pathと壁近傍のadaptive
  dyadic crossing探索。`_stochastic_composition.py`は決定論traceとの合成だけを担当します。
- `_runtime_terminal_state.py`: escaped/invalid/numerical-boundary終端stateの一貫した更新。
- `_runtime_charge.py`: charge half-step、終端時刻へのreplay、diagnostic集計。
- `drag_models.py`: 公開drag名の互換写像、連続流則・希薄化補正、有効緩和時間。
- `integrator_common.py`: substep、ETD2 stage、状態更新式。従来のdrag importだけ再公開。
- `segment_motion.py`: scalar/batch motionの安定した公開入口だけを再公開。
- `_segment_motion_contracts.py`: scalar/batch requestとbatch destinationのimmutable契約。
- `_segment_stage_dynamics.py`: field sampling、force合成、ETD2 stage状態更新。
- `_segment_motion_scalar.py`: scalar motionのfull-step対two-half-step LTE、受理trace、prefix state。
- `motion_kernel_numba.py`: compiled motionの同じLTE判定と受理trace生成。
- `segment_motion_batch.py`: batch実行順と、必要な場合のscalar precise fallback。
- `_segment_motion_batch_state.py` / `_segment_motion_batch_backend.py`:
  batch入力・destination buffer検証と、regular/triangle compiled kernel adapter。
- `runtime_plan.py`: solver設定と、全実行経路で共有するfield requirementを確定。
- `field_runtime.py`: typedな`StageFieldPlan`を一つの`FieldRequest`へ変換し、明示指定が
  ある場合だけplanを上書き。
- `field_compilation*.py`: regular/triangle fieldをimmutable backendへcompile。
- `core/field_backend.py`: configured providerを共通stage sampling contractへ適合させます。
  support・provenance reportは`core/field_backend_reporting.py`が所有します。
- `base_field_sampling.py`: compiled backendからforce非依存のbase fieldだけをsampleします。
  gas sampling診断の組立ては`_field_sampling_report.py`が所有します。
- `force_field_assembly.py`: 公開sampling入口、particle入力検証、backend dispatchを所有します。
- `_force_field_sources.py`: force間で共有するgas/electric/flow入力を解決します。
- `_force_field_regular.py`: regular grid固有のderived fieldを組み立てます。
- `_force_field_triangle.py`: triangle mesh固有のderived fieldを組み立てます。
- `force_runtime.py`: force pipelineの安定した公開入口。
  `_force_pipeline.py`がfield requirement・入力検証・評価計画を所有し、
  `_force_evaluators.py`が各forceの数値式と評価loopを所有します。
- `collision_detection.py`: 2D/3D trial分類の公開入口とdimension dispatch。
  2D containment・edge hit、3D exact分類、conservative候補、stage trace昇格、
  diagnostic更新、結果型は`_collision_detection_*.py`がそれぞれ所有します。
- `collision_hit_localization.py`: 最初の物理hitの時刻・位置・速度と有限primitiveの再照合。
- `wall_response.py`: wall lawからoutcomeと出射速度を計算。
- `high_fidelity_collision.py`: wall responseとevent commitを所有。
  `_collision_trial.py`が軌道再積分・valid-mask retry・trialを、`_collision_particle.py`が
  粒子state遷移とsegment loopを所有します。共有値型、hit解決、wall event形式は
  `_collision_types.py`、`_collision_resolution.py`、`_collision_wall_events.py`が所有します。
- `contact_sliding.py`: 継続接触運動の安定した公開入口。
  `_contact_sliding_2d.py`と`_contact_sliding_3d.py`がdimension別のrelease・hold・tangent移動、
  `_contact_geometry.py`、`_contact_dynamics.py`、`_contact_state.py`がframe・field/drag・state
  commitを所有します。
- `stochastic_motion.py`: stochastic motionの安定した公開入口。
  設定・累積diagnostic、OU path、temperature sampling、path生成、deterministic
  segmentとのcompositionは`_stochastic_*.py`がそれぞれ所有します。乱数生成は
  path生成ownerだけが行い、velocity、position、bridge seedの順序を固定します。
- `charge_model.py`: chargingの安定した入口。設定型、OML計算、background sampling、
  runtime更新は`_charge_*.py`へ分離。
- `valid_mask_retry.py`: field support外へ進んだsegmentのprefix確定と停止。

backend固有kernelはsamplingとbatch計算を担当します。入力parser、artifact生成、
wall eventの表形式化は担当しません。

## Field・geometry・boundary

fieldはregular gridまたは2D triangle meshのcompile済みbackendへ変換し、同じbatch
sampling APIから使います。field support外と物理壁への到達は別の終了理由として保持します。

solverが使うgeometry入口は`BoundaryQuery`です。inside判定、polyline hit、最近傍投影の
2D/3D差は`core/`側に閉じ込めます。境界toleranceとcontact offsetは
`BoundaryNumerics`としてcase構築時に一度だけ決まり、各solver処理で独自の固定値を
追加しません。

3D triangle geometryは`core/geometry3d.py`を安定した入口とし、
`_triangle_topology.py`がULP解像・barycentric・閉曲面検証、`_triangle_surface.py`が
immutable surface・uniform grid・候補query、`_triangle_queries.py`がintersection・投影・
inside判定を所有します。`boundary_triangles`は衝突面、任意の
`containment_boundary_triangles`は閉外殻という一方向の役割分担で、legacy artifactは
同一surfaceを両用途に再利用します。

2Dのbatch inside判定では、`core/boundary_core.py`が入力shape、boundsによる候補抽出、
edge・loop・SDFの優先順と結果反映を担当します。`core/geometry2d.py`はloop topologyと、
候補点に対するboundary距離・ray crossingだけを担当します。solverはこの表現差を判定しません。

geometryと座標metadataは、用途別に一度だけ解決してruntimeへ渡します。

```text
geometry ──> boundary_numerics ──> runtime context ──> boundary_service
geometry ────────────────────────────────────────────────┘
                                                           |
                                                 boundary_hits facade
                                                           |
                                         dimension別 hit/contact owner
coordinate metadata ──> coordinate_systems ──> RZ report / preflight
regular grid ──────────> grid_sampling ───────> sampled scalars
```

- `core/coordinate_systems.py`: 座標系名、軸名、RZの軸・ring weight・geometry report。
- `core/boundary_numerics.py`: geometry scaleから分類toleranceとcontact offsetを一度だけ導出。
- `core/boundary_service.py`: geometryから2D broad phaseまたは3D queryを組み立て、
  solverへ`BoundaryService`として渡す。
- `core/grid_sampling.py`: regular gridの区間探索とscalar/batch補間だけを担当。

## 診断と成果物

standard modeは最終状態と運用上必要な集計だけを保持し、次の3fileを保存します。

- `final_particles.csv`
- `run_summary.json`
- `wall_summary.csv`

debug modeだけがtrajectory、wall event、step summary、force contribution、詳細診断を
収集します。内部の`SolverOutcome.debug`はstandard modeでは`None`です。debug情報を
空listへ捨てるためにstandard hot pathからrow builderを呼びません。

`write_result()`はstaging directoryへ全fileを書いた後にrenameするため、途中状態を
公開しません。既存の成果物directoryは上書きしません。schema検証は`artifacts.py`が
担当します。

比較系のJSON変換は`compare/_common.py`を唯一の所有者とし、通常のNumPy scalar変換と、
NaN/Infを`null`へ変えるstrict診断用変換を明示的に分けます。
first-step比較は`first_step_compare.py`がCLIとrun順を制御し、force sampling、誤差metrics、
report変換は`_first_step_*.py`へ分離します。

可視化は`export_visualizations.py`がroutingだけを行い、用途別のexporterへ渡します。
`export_result_graphs.py`は安定した公開入口であり、`_result_graph_*.py`がcompact表示、
field map、event、trajectory、summaryを所有します。`export_mechanics_visuals.py`はgeometry上の
力学量、`export_trajectory_animation.py`は時間発展を所有します。各exporterは入力を
一度だけ読んで検証済みcontextへまとめ、plot helperとreport生成へ渡します。
CSV/NPY/NPZの読込み・正規化は`visualization_data.py`、directory・index・Markdown reportは
`visualization_reports.py`、geometry描画primitiveは`visualization_common.py`が所有します。
可視化用NPY/NPZもobject arrayを契約外とし、pickleを許可せず読み込みます。

## 依存方向

import-linterは`pyproject.toml`の4 contractを検査します。

1. `core`からsolver実装へ依存しない。型注釈を含め例外は置かない。
2. `preflight`から`application`へ依存しない。
3. `solvers`からapplication、I/O、provider、writer、compare等の外層へ依存しない。
4. `configuration`とそのowner module、`domain`、`force_models`からadapter/runtime層へ依存しない。

```text
CLI / writer / compare / tools / migration / COMSOL case
                          |
             application / io / providers / preflight
                          |
                       solvers
                          |
        configuration / force_models / domain / core
```

図は上から下への依存方向を示します。実際の禁止方向は`pyproject.toml`を正本とし、不要に
なったignoreが残った場合もgateを失敗させます。

## 公開互換性

root packageの公開操作は次の4つです。

```python
load_case(config_path)
validate_case(case, detail="summary")
simulate(case)
write_result(result, output_dir)
```

公開CLIは`particle-tracer`だけです。専門workflowはsubcommandとしてroutingし、
process-globalな`sys.argv`を書き換えません。comparison toolingは必要な場合に
`SimulationCase.solver_context`をread-onlyで参照し、private fieldへ越境しません。
`load_case()`はadapter配列をowned copyにし、solver入力配列とnested metadataを再帰的に
read-only化してからcaseへ格納します。mutableなのは実行時の`SolverState`とscratchだけです。

runtime入力と成果物はschema version 2を正本とします。旧YAML/CSVや旧wall/material
tableの解釈は`migrate`に限定し、変換後は通常のstrict parserで再検証します。
一意に変換できないlegacy機能は推測で近似しません。

## 品質gate

品質依存はuvの`quality`/`nightly` groupに分離し、Noxを共通入口にします。

```console
uv run --frozen nox -s quality-fast -- path/to/changed.py
uv run --frozen nox -s quality-pr
uv run --frozen nox -s quality-nightly
uv run --frozen nox -s quality-baseline
```

- `quality-fast`は変更Python fileだけをformat・安全なfixの対象にし、その後に
  baseline-aware Ruff/Pyreflyとpytestを実行します。
- `quality-pr`はcheck-onlyで、import contract、Radon、branch coverage、変更行coverage、
  Bandit、pip-audit、detect-secrets、Vultureを追加します。
- `quality-nightly`はJIT有効/無効、nightly Hypothesis、複数seed、2D/3D性能・memory、
  mutation testを追加します。mutmutはLinux/WSLで実行します。
- `quality-baseline`だけが`.quality/baseline.json`、Pyrefly baseline、secret baselineを
  更新できます。通常commandとCIはbaselineを更新しません。

`quality_tools/runner.py`は4 commandの順序制御とCLIだけを所有します。
`_runner_tools.py`がtool実行・収集・security、`_runner_baseline.py`が悪化比較と明示更新、
`_runner_diff.py`がGit/snapshot差分と変更path解決を所有します。

baselineは既存診断を免除する一覧ではなく、悪化を検出する比較点です。新規lint/type/
security違反、test node ID削減、新規suppression、既存関数の複雑度増加、branch coverage
低下を失敗させます。CIではGit履歴から変更行coverage 90%以上も検査します。

PR/pushでは[quality workflow](../.github/workflows/quality.yml)がPython 3.12の
`quality-pr`とPython 3.10のpytest互換性を実行します。毎日18:00 UTCとmanual実行では
[nightly workflow](../.github/workflows/quality-nightly.yml)が`quality-nightly`を実行します。
