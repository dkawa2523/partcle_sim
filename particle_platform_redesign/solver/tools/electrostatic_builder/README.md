# Reduced electrostatic field builder

このtoolは、canonicalな熱流体`case.h5`と利用者指定のbulk plasma条件から、粒子計算に渡す静的な
電位・電場・plasma primitiveを生成するfirst-party field-production componentです。粒子trajectory
engineとは独立しており、COMSOLや`model_dataset/`へ依存しません。旧workflowの`Case A`はこの利用形態を
調べる参照名であり、設定やsolver分岐の名前ではありません。

## v2の範囲

`reduced_electrostatic_builder_v2`は意図的に次へ限定します。

設定は`format_version: 2`を要求します。共通YAML parserは全階層の重複keyとmerge key（`<<`）を拒否するため、
mergeを使った設定はkeyを明記して移行します。通常のaliasは利用でき、元のUTF-8 bytesのhashを保存します。
この変更は設定解釈を一意にするもので、closureとfield semanticsの物理revisionは維持します。

- `axisymmetric_rz`
- 静的、単一gas domain
- geometryと完全一致する、全cell supportedのP1三角形layout
- 単一の準中性bulk密度 `n_e0 = n_i0 = n0`
- Boltzmann electron / collisionless Bohm-ionの縮約space-charge closure
- canonical node fieldとして与えたgas速度とgas温度
- semantic boundary groupに対する固定電位または半径方向指数遷移電位

四角形とのmixed mesh、時間依存、3-D、負イオン、species別ion輸送、ion continuity、RF周期、誘電体内部、
表面充電、磁場、二方向plasma couplingはこのrevisionでは扱いません。未対応入力を別解法へ黙ってfallback
せず、生成前に拒否します。

## 物理model

解く式は軸対称Poisson方程式です。

```text
-(1/r) d/dr(r epsilon0 dV/dr) - d/dz(epsilon0 dV/dz) = e (ni - ne)
E = -grad(V)
uB = sqrt(e Te_V / mi)
psi = (V - Vp) H2(-(V - Vp), dV)
ne = max(n_floor, n0 exp(psi / Te_V))
ui = sqrt(max(u_floor^2, uB^2 - 2 e psi / mi))
ni = max(n_floor, n0 uB / ui)
```

`Te_V`は電子温度をpotentialとして表した量[V]です。`H2`は幅`dV`のC2 quintic smooth
Heavisideで、遷移区間`y=x/dV`では次です。

```text
H2(y) = 1/2 + 15 y/16 - 5 y^3/8 + 3 y^5/16
```

出力するion速度は`ui`と、thermal-flow fieldを使う次のdrift-diffusion fluxの正規化方向から作ります。
fluxがほぼゼロの点では方向が未定義になるため、正規化を滑らかに0へ近づけます。このためベクトルの大きさは
`ui`以下であり、別fieldの`positive_ion_speed`がclosure上の`ui`を保持します。

```text
Di = mu_i kB Tg / e
Gamma_i = ni ug + mu_i ni E - Di grad(ni)
u_i_vector = ui Gamma_i / sqrt(|Gamma_i|^2 + (n0 u_regularization)^2)
grad(ni) = (dni/dV) grad(V)
```

これはversioned reduced modelであり、一般のplasma解ではありません。仮定とparameterはすべて出力
provenanceへ保存されます。

## 数値解法

- `2 pi r`重みを含むP1 Galerkin FEM
- source/Jacobianの三点triangle quadrature
- analytic `d rho/dV`
- 設定に明記した境界電位continuation
- line search付きNewton
- 非正定値Jacobianを許す、対角scale付き行列フリーrestarted GMRES
- cell gradientの軸対称mass-lumped nodal projection
- axis nodeでの`E_r = 0`の厳密化

収束しない場合はartifactを成功扱いにせず、失敗したcontinuation levelを例外へ含めます。別closure、
別linear solver、緩いtoleranceへの自動切替はありません。絶対残差と電荷収支は省略なしの`2 pi r`
積分に基づくC単位です。

## 入力設定

設定は小さいstrict YAMLです。未知key、暗黙単位、既定modelは受理しません。次は全keyを示す例です。

```yaml
format_version: 2
input:
  data_path: thermal_flow_case.h5
  layout: plasma
  gas_velocity_field: gas_velocity
  gas_temperature_field: gas_temperature
model:
  revision: boltzmann_bohm_sheath_c2_v1
  bulk_number_density_m3: 1.0e+15
  electron_temperature_V: 4.0
  positive_ion_mass_kg: 8.302695335869234e-26
  bulk_potential_V: 15.0
  ion_mobility_m2_V_s: 1.0
  ion_flux_regularization_speed_m_s: 1.0
  sheath_smoothing_V: 0.05
  density_floor_m3: 1.0e+6
  ion_speed_floor_m_s: 1.0e-3
boundaries:
  - group: wafer
    priority: 30
    potential: {kind: constant, value_V: -68.2710575}
  - group: grounded_wall
    priority: 20
    potential: {kind: constant, value_V: 0.0}
  - group: focus_transition
    priority: 25
    potential:
      kind: radial_exponential
      inner_value_V: -68.2710575
      outer_value_V: -5.0
      start_radius_m: 0.152
      transition_length_m: 0.005
  - group: outer_dielectric
    priority: 20
    potential: {kind: constant, value_V: -5.0}
  - group: bulk_opening
    priority: 40
    potential: {kind: constant, value_V: 15.0}
solver:
  continuation_ramps: [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
  max_newton_iterations: 30
  relative_residual_tolerance: 1.0e-9
  absolute_residual_tolerance_C: 1.0e-22
  max_linear_iterations: 1200
  linear_krylov_dimension: 80
  linear_relative_tolerance: 1.0e-11
  minimum_line_search_factor: 0.0009765625
```

`r=0`は自然な軸対称条件であり、Dirichlet groupや材料壁へ含めません。異なる電位groupが共有するcorner
nodeは大きい`priority`を採用します。同じpriorityで値が異なる場合は拒否します。この解決結果と入力条件は
provenanceへ保存され、mapping順には依存しません。

## 実行と出力

solver project directoryから実行します。

```powershell
uv run --locked python -m tools.electrostatic_builder builder.yaml augmented_case.h5 `
  --report builder_report.json
```

入力geometry、layout、既存field、realized sourceは保持し、次のnode fieldを追加します。

- `electric_potential` [V]
- `electric_field` [V/m], `(r,z)`
- `electron_number_density` [1/m^3]
- `positive_ion_number_density` [1/m^3]
- `space_charge_density` [C/m^3]
- `electron_temperature` [K]
- `positive_ion_temperature` [K]
- `positive_ion_velocity` [m/s], `(r,z)`
- `positive_ion_speed` [m/s]
- `positive_ion_mass` [kg]

出力先とfield名は上書きしません。reportとHDF5 provenanceは、入力content hash、設定hash、model/
discretization revision、境界条件、continuation別反復・残差、floor active数、電荷収支、field rangeを持ちます。
`electric_field`等をどのtrajectory physicsへ接続するかは通常の`case.yaml`が所有し、builderは粒子modelを
選びません。`positive_ion_mass`はbuilder closureのprovenance fieldであり、現行continuous-charge OML revisionsが
要求するscalar `positive_ion_mass_kg`を暗黙設定しません。

## 検証と参照モデルへの接続

通常のtool gateはCOMSOLなしで実行できます。

```powershell
uv run --locked python -m pytest tools/electrostatic_builder/tests -q
```

現行testはclosure/Jacobian、準中性bulk、軸方向affine Laplace解、非線形sheath、nested 2-D mesh
self-convergence、annular `A log(r)+B` Laplace解の約2次収束、global charge balance、axis regularity、
canonical write/read、非有限な派生closure量のfail-closedを検査します。

`model_dataset`の参照Case-A domainはtriangle 2,127個とquad 826個のmixed meshです。F02ではCOMSOL adapter側で
domain抽出、node再採番、品質基準付きの決定論的quad分割、boundary owner再構築、semantic group付与、thermal
field転記を一度だけ行い、1,987 node / 3,779 cellのtriangle-only canonical inputを作りました。builderや
trajectory runtimeにmixed-elementの第二経路は追加していません。

代表solveは1,826 free node、10 continuation ramp、総linear iteration 2,821、最終relative residual
`1.8441e-13`、charge-balance error `5.8498e-21 C`で完了しました。GMRES basis＋Hessenbergは1,235,088 B、
同一machineのsolve-only観測は0.689/0.737 sでした。この規模で現v1がgateを満たしたため、preconditioner、
第二linear solver、自動fallback、反復上限の水増しは追加していません。

COMSOL照合は外部V&Vが所有します。F02では`2 pi r`軸対称lumped-volume norm、semantic boundary trace、builder
charge balanceを同一export node上で記述しました。単一reference meshなので独立mesh convergenceは未検証であり、
corner共有nodeはexclusive traceと分けています。差を消すためのcore/builder分岐やtolerance緩和は行いません。
