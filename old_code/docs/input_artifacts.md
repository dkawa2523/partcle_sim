# 入力と成果物の契約

この文書は schema version 2 の公開入出力だけを扱います。設定の正本は
[`configuration.py`](../particle_tracer_unified/configuration.py)、CSV は
[`canonical_tables.py`](../particle_tracer_unified/io/canonical_tables.py)、成果物は
[`writer.py`](../particle_tracer_unified/writer.py) です。COMSOL 固有の manifest は
[`comsol_vv.md`](comsol_vv.md) を参照してください。

## 実行の流れ

```text
run_config.yaml
  -> load_case()      設定、CSV、provider を検証して immutable な case に変換
  -> validate_case()  副作用のない preflight
  -> simulate()       メモリ上の SimulationResult を生成
  -> write_result()   結果を新規または空の directory へ一度だけ発行
```

`load_case()` と `simulate()` は成果物を書きません。`validate_case()` も入力や solver
state を変更しません。永続化の責務は `write_result()` だけにあり、既存成果物を上書き
しません。

## run_config.yaml

root は次の6 keyで固定です。未知key、legacy alias、前後空白のある文字列、文字列で
書いたbooleanは受理しません。

```yaml
schema_version: 2
case:
  spatial_dim: 2
  coordinate_system: cartesian_xy
  adapter: native
inputs:
  particles: particles.csv
  boundaries: boundaries.csv
  geometry:
    kind: box
    parameters:
      bounds: [-1.0, 1.0, -1.0, 1.0]
      grid_shape: [41, 41]
      boundary_part_ids: [10, 20, 20, 10]
  field:
    kind: linear_shear
    parameters:
      shear_rate: 1.0
      dynamic_viscosity_Pas: 1.8e-5
physics:
  drag: {model: stokes}
  gas: {dynamic_viscosity_Pas: 1.8e-5}
  forces: {}
  seed: 12345
time: {dt: 0.01, t_end: 0.05}
output: {mode: standard}
```

- `spatial_dim` は2または3です。座標系との有効な組合せは後述の3通りだけです。
- native adapter は `particles`、`boundaries`、`geometry`、`field` を必須とします。
  geometry は `box` または `precomputed_npz`、field は `linear_shear`、
  `precomputed_npz`、`precomputed_triangle_mesh_npz` のいずれかです。
- COMSOL adapter の入力は `comsol_manifest` だけです。artifact、field mapping、force
  inventory を run config に重複させると失敗します。
- native の `physics.drag.model` は `none`、`stokes`、`stokes_cunningham`、
  `schiller_naumann`、`epstein` のいずれかです。任意forceは明示的に有効化したもの
  だけを使います。
- `time.dt` は有限かつ正、`time.t_end` は有限かつ0以上です。積分器は ETD2 に固定
  され、設定keyではありません。
- `time.max_substep_splits` は省略可能な整数で、既定は4、範囲は0以上12以下です。
  1 nominal stepを二分できる回数、すなわちsubstep予算 `2 ** max_substep_splits` を
  決めます。滑らかな自由飛行と、シース通過や壁接近とでは必要な分割数が違うため、
  既定値を暗黙に引き上げることはしません。
- `physics.wall_interaction` は省略可能で、`contact_sliding`（既定 `true`）と
  `max_hits_per_step`（既定5、1以上64以下）を取ります。`contact_sliding` は
  同一壁への繰り返し接触を接線運動へ正則化する数値的装置の有無で、COMSOLには
  対応する接触modelがありません。`comsol build-case` の生成caseは `false` です。
- `output.mode` は `standard` または `debug` です。debug では正の整数
  `trajectory_interval_steps` が必須で、standard では指定できません。

## 粒子CSV

すべての物理量はSI単位です。共通の必須列は次のとおりです。

```text
particle_id, release_time_s, mass_kg, drag_diameter_m, charge_C,
source_part_id
```

座標列と速度列は座標系で決まります。

| `coordinate_system` | `spatial_dim` | 位置 [m] | 速度 [m/s] |
|---|---:|---|---|
| `cartesian_xy` | 2 | `x_m`, `y_m` | `vx_mps`, `vy_mps` |
| `axisymmetric_rz` | 2 | `r_m`, `z_m` | `vr_mps`, `vz_mps` |
| `cartesian_xyz` | 3 | `x_m`, `y_m`, `z_m` | `vx_mps`, `vy_mps`, `vz_mps` |

任意列は `density_kgm3`、`material_id`、
`dep_particle_rel_permittivity`、`thermophoretic_coeff`、`metadata_json` です。
未知列は拒否します。

主な値契約は次のとおりです。

- `particle_id` は重複しない0以上の整数、`source_part_id` は正の整数です。
- `mass_kg` と `drag_diameter_m` は有限かつ正です。慣性の正本は `mass_kg` であり、
  densityとdiameterから再構成しません。`density_kgm3` が明示されたnative粒子では、
  massとdensityから材料等価球径を導出し、drag径とは独立に幾何依存の力と帯電へ使います。
- 位置、速度、release time、charge、およびCSVに存在する任意数値列は有限値必須です。
  `release_time_s` は0以上、RZの `r_m` は0以上です。
- 任意のparticle propertyを指定しない場合は列を省略します。loaderは省略された
  `density_kgm3`、`dep_particle_rel_permittivity`、`thermophoretic_coeff` を内部
  `float64 NaN` sentinelへ変換します。CSVに `NaN` や `Inf` を直接書くことはできません。
- `metadata_json` は空欄またはJSON objectです。

loader後の `ParticleTable` は粒子数を `N`、次元を `D` とすると、位置・速度が
`float64 (N, D)`、物理scalarが `float64 (N,)`、IDが `int64 (N,)` です。

## 境界CSV

`boundaries.csv` はpart、material、wall lawを1行で結びます。必須列は次のとおりです。

```text
part_id, part_name, role, material_id, material_name, wall_law,
wall_stick_probability, wall_restitution, wall_diffuse_fraction,
wall_critical_sticking_velocity_mps
```

任意列は `metadata_json` だけです。`part_id` は重複しない正の整数、`material_id` は
0以上の整数です。`role` は `wall`、`inlet`、`outlet`、`internal`、
`field_support`、wall lawは次のいずれかです。

```text
stick, freeze, absorb, escape, pass_through, specular, cosine_diffuse,
mixed_specular_diffuse, critical_sticking_velocity
```

確率とdiffuse fractionは `[0, 1]`、restitutionとcritical velocityは0以上です。
数値はすべて有限値を要求します。未登録partへ暗黙のwall lawを補いません。
`freeze` は衝突位置と衝突時速度を保持し、速度をゼロにする `stick` とは区別します。
`pass_through` が透明面になるのは `role=internal` の場合だけで、外部境界では離脱です。

## precomputed NPZ

NPZはloaderでNumPy配列へ正規化されます。座標scaleを持つCOMSOL入力ではSIへ変換した
後に以下の契約を評価します。field quantityは実数配列に限ります。複素phasorを暗黙に
実部へ変換せず、exporter側でRMS・peak・位相などの意味を決めた実数semantic quantity
として出力してください。

NPY/NPZは数値、bool、Unicode配列だけを受け付けます。object dtypeとpickle依存の配列は
実行環境依存かつ安全に検証できないため、provider・可視化のどちらでも読み込み時に拒否します。

### regular geometry / field

- `axis_0` ... `axis_(D-1)` は有限で狭義単調増加する `float64 (G_i,)` です。
- geometryの `sdf` は `float64 (G_0, ..., G_(D-1))`、`valid_mask` は同じshapeの
  `bool` です。明示境界は2Dの `boundary_edges` または3Dの
  `boundary_triangles` とpart IDで保持します。3Dで内部界面を含む場合だけ、包含判定用の
  閉外殻を `containment_boundary_triangles` に分けます。省略時は従来どおり
  `boundary_triangles` が衝突と包含の両方を担います。
- fieldの `times` は有限で狭義単調増加する `float64 (T,)` です。省略時は
  `[0.0]` のsteady fieldです。
- scalar componentはsteadyならgrid shape、transientなら
  `(T, G_0, ..., G_(D-1))` です。vectorはsemantic componentごとの配列として
  読みます。
- field axisとgeometry axisはshapeを含めて一致し、各値の差が
  `64 * spacing(max(abs(a), abs(b)))` 以下でなければなりません。
- `valid_mask` 内のfield値は有限値必須です。mask外のfill値を物理値として補間しません。

### 2D triangle field

`precomputed_triangle_mesh_npz` は2D専用です。`mesh_vertices` は有限な
`float64 (V, 2)`、`mesh_triangles` は有効なvertex IDを持つ整数 `(K, 3)` です。
quantityはsteadyなら `(V,)`、transientなら `(T, V)` で、全値が有限でなければ
なりません。退化triangleと範囲外vertex IDは拒否します。

COMSOLではNPZの配列名を直接solver APIにせず、manifestのsemantic quantity、component、
unit、正の `scale_to_si` で一度だけ対応付けます。

## 標準成果物

standard modeは次の3ファイルだけを発行します。

### `final_particles.csv`

全行に `schema_version=2` を持ちます。共通列は次のとおりです。

```text
particle_id, release_time_s, released, final_state, invalid_stop_reason,
source_part_id, material_id, mass_kg, drag_diameter_m, charge_C,
contact_part_id, final_step_name, final_segment_name
```

さらに各axis `a` について `a_m`、`va_mps`、`contact_normal_a` を持ちます。例えば
Cartesian 2Dでは `x_m`, `y_m`, `vx_mps`, `vy_mps`,
`contact_normal_x`, `contact_normal_y` です。writerは入力と同じaxis命名を使い、
`x` や `v_x` のような別名は出力しません。

### `wall_summary.csv`

列は `schema_version, part_id, outcome, wall_mode, count` です。part、結果、適用した
wall modeの組合せごとに件数を集計します。

### `run_summary.json`

`artifact_type` は `particle_tracer.run_summary`、`schema_version` は2です。粒子・終端
状態・wall結果の件数、座標系とaxis名、dragとexperimental feature、実行時間、memory
見積り、safety counterを記録します。`execution` には解決済みの `dt_s`、`t_end_s`、
seed、force/gas/charge/stochastic、numerical policy、入力hash、software versionを保存
します。JSONには `NaN` / `Infinity` を書かず、非有限値は `null` に変換します。

## debug成果物

debug modeは標準3ファイルに次を追加します。

- `trajectory.npy`: `float64 (S, N, D)` の保存時刻・粒子・axis順の位置 [m]
- `trajectory_frames.csv`: save index、時刻、step/segment名
- `wall_events.csv`: hit時刻、particle/part/primitive、位置 [m]、normal、速度 [m/s]、結果
- `step_summary.csv`: 時刻ごとのrelease・active・terminal・support停止件数
- `force_contributions.csv`: 解決済みforce、必要field、parameter
- `debug_diagnostics.json`: collision診断とmax-hit event

CSVには `schema_version` 列、debug JSONには
`artifact_type: particle_tracer.debug_diagnostics` と `schema_version: 2` を持ちます。
debug収集を有効にしても数値結果は変えないことを回帰testで固定しています。

## 検証、互換性、migration

```console
particle-tracer artifacts result_dir
particle-tracer artifacts debug_result_dir --require-debug
particle-tracer migrate legacy/run_config.yaml -o migrated
```

artifact validatorは必須ファイル、宣言外ファイル、schema version、JSON artifact typeを
検査します。CSVの全物理値を再検証する機能ではないため、入力検証の代用にはなりません。

公開互換性の単位は `schema_version: 2`、列名、axis順、SI単位、shape、artifact集合です。
consumerは座標列を固定推測せず、`run_summary.json` の `coordinate_system` と
`axis_names` を使ってください。legacy入力は実行時に暗黙変換せず、`migrate` で別directory
へ明示変換します。意味が一意に移せない旧機能は近似せずエラーになります。
