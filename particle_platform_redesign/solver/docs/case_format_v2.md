# Canonical case format v2

この文書は、外部producerとsolverの間で受け渡す`case.yaml`と`case.h5`の実装契約です。
COMSOL固有の列名、study、selectionはここへ持ち込まず、adapterがSIとcanonical semanticsへ変換します。

現記載は現行engine `particle_engine_v37`、compiled CPU tile v18、proposal v10、event v16、physics catalog
`inertial_langevin_rz_catalog_v17`、physics runtime `signed_ion_compiled_physics_runtime_v20`、
RK4 enclosure v2、dense path `rk4_position_hermite_state_extension_v3`、charge-stable exponential midpoint v3 / enclosure v4、
field location v4、runtime layout v6、memory plan v14、boundary algorithm `point_wall_laws_v5`までの実装契約である。case schema v2は実測で製品価値を示せなかった粒子内multithreading設定を削除し、
`resources`をmemory budgetだけへ戻した。cumulative solver workで決めるdurable epoch、checkpoint、resume identity、
同期single-owner writerはsolver内部とresult formatの責務であり、YAML/HDF5へcheckpoint cadenceやresume modeの
設定keyを追加しない。format v1は未releaseの試作schemaとして拒否し、過去の`resources.threads`を受ける
compatibility aliasや二重parserは作らない。

engine v7以降の一般曲線材料壁refinement batchとengine v8のaccepted replay保持選択は内部work partitionであり、
YAMLに選択子を追加しない。現行event v16のtransverse/start-contact/axis certificateとrow-local statusも内部数値判定であり、`case.yaml`、`case.h5`、
case schema v2を変更しない。
P19-Lのglobal-first/local-fallback applicability certificate、64-cell候補上限、dense path、failure code 9、
`solver.event.maximum_refinements`を共有する分割budgetも内部数値判定である。producerが局所認証方式、cell上限、第二の
refinement設定を選ぶYAML/HDF5 keyは追加しない。
event v16のquery authorityも内部実装である。global enclosureはshortened-stage、field support、applicability、acceptance
safetyを所有し、global supportを独立に証明済みのvalid `rk4_dense` rowだけcurrent dense Bernstein boundでevent broad-phase
queryを行う。Hermite pathでは物理position budgetとroundoff budgetを加算し、facet-local offset dotは補償演算、評価・包絡は
root-relative TwoDiffを使う。validなRK4 dense rowのposition Bernstein control enclosureがfacetの既存budget込みinside
half-spaceへ全4点で厳密に入る時だけ候補をclearし、証明不能なら保持してsplitする。このcertificateをproducerが選ぶ
YAML/HDF5 keyは追加しない。monotone clearはcubic Hermite derivative-Bernstein enclosureの明示opt-inだけとし、一般
RK4/exponential/scalarは証明不能時にfail-closedとする。この選択をcase入力へ公開しない。
dense path v3の原点相対Bernstein enclosure、TwoDiff残差、方向付き外向き座標変換も内部丸め規則であり、
endpoint・path・event意味やcase keyを増やさない。
P10/P14のregular compiled locator、P1/Q1 previous-cell strict-interior hint、supported-containment BVHと
outside/masked full-search fallbackも内部runtime詳細である。hintはaccepted endpointだけが更新し、YAML/HDF5 keyや
producer責務を追加しない。geometry volume-cell BVH、局所float64解像性gate、準備済みedge長も内部prepare責務であり、
producerがindexやtoleranceを指定するkeyは追加しない。
P11の線形緩和分解、指数更新、path enclosureも内部numericsであり、producerはCOMSOL固有設定や事前計算した
drag係数をHDF5へ追加しない。

## 所有境界

- `case.h5`は再利用可能な`DataBundle`、すなわちgeometry、field layout/value/support、boundary、
  realized particle table、producer provenanceを所有する。
- `case.yaml`は`SimulationSpec`、HDF5への相対または絶対path、期待logical content hashを所有する。
- 同じ座標系、単位、boundary group、field定義を両方へ重複記載しない。
- `case_format.py`だけがHDF5 schemaのread/write/hashを所有し、`case.py`だけがYAMLと二ファイル間の
  参照整合を所有する。
- `read()`は検証済みDataBundle、`read_with_info()`は同じ単一読込経路からDataBundleと
  `CaseFileInfo(schema_version, content_hash)`を返す。`load_case`は後者を使い、大規模配列のschema検査を
  hash照合のために重複実行しない。
- P09の`load_case`はYAML全体と`resources.memory_limit_mb`を先にparseし、HDF5のdataset metadataから
  canonical numeric array bytesを求める。上限超過はnumeric payloadをmaterializeする前に`CaseError`とする。
  このmetadata scanは完全なprepareや別のpreflight subsystemではない。
- `case_format.read`はindex範囲、局所node順、boundary rowと宣言owner edgeの整合までを所有する。
  外周完全性、重複boundary、non-manifold、内部edgeのwall登録は`geometry.prepare`が一度だけ検査する。

## 現行の設定境界

- `/meta/coordinate_system`はgeometry/fieldのdata表現、必須`motion.mode`は粒子状態の運動座標を所有する。
  P05の有効組合せは`cartesian_xy + cartesian_xy`と
  `axisymmetric_rz + axisymmetric_rz_meridional`だけである。
- `output.trajectories`は`null`、または`selection: all`と
  `schedule.explicit_times_s`からなる小さいtyped設定である。`output.probes`は省略または`null`、あるいは
  strictly increasingな明示`particle_ids`と`schedule.explicit_times_s`からなる。probeは粒子stateだけを保存し、
  全force/RK stage/debug traceを常時記録しない。
- final particle表とrelease eventは必須resultであり、enable flagを持たない。
- table release timeはfiniteかつ`[start_s,end_s]`内とし、時刻原点を0へ固定しない。
- engineはtable/surface source、fixed charge、P15の`plasma_continuous` charge、CPU固定stepを受理する。
  無力場は厳密なballistic退化形、
  dragなし・空間的に一定な加速度は証明済み`quadratic_exact` pathを使う。P06 revision 3a以降は、
  全cell supportedな`RegularLayout`に限り、`epstein_linear_v1`、
  `epstein_finite_speed_maxwell_mixed_equal_temperature_v1`、`epstein_linear_effective_gas_sensitivity_v1`、
  `electric_coulomb_v1`、`gravity_buoyancy_standard_v1`、明示されたP15-F/P18-I ion drag、P16/P18-R thermophoresis、
  P18-D quasistatic spherical DEP、P18-L RZ rarefied-vorticity lift sensitivityを合成した
  一般`rk4_reintegrated` runも、全短縮RK4評価の
  continuous support/applicability enclosureを証明して受理する。revision 3bはtopology-completeな材料boundary、
  parameterなしの`stick`/`escape`とこの経路を連成する。P06-Uは材料domain meshと完全一致するfully-supported
  P1/Q1にもCartesian XYの同じ一般RK4経路を許可する。P06-RZはregular fieldおよびtopology-completeな材料domainで
  RZ一般RK4を同じ経路へ追加する。boundaryless一般RK4は引き続きregular support boxを必須とする。model固有mappingは
  [`physics_models.md`](physics_models.md)が所有する。
   B01の数値primitiveをproductionへ接続したB02は、`physics.noise`と`solver.integrator: ou_langevin`を
   同時に選ぶ場合だけ受理する。組合せはCartesian XY、Epstein linear drag-only、fixed-charge state、terminal
  `stick`/`escape`に限定し、continuous charge、他の力、反射、RZはfail-closedとする。固定depth interval treeの
   各leafはcubic Hermite numerical pathとしてevent/replayへ渡すが、連続OU first-passageの厳密解とは扱わない。
  B03の完了revisionは同じ`physics.noise` categoryと`ou_langevin`を使い、別integratorまたは
  COMSOL専用case keyを追加しない。`inertial_langevin_fdt_epstein_linear_rz_meridional_projected_v1`だけが
  `axisymmetric_rz_meridional`、native/effective-gas線形Epstein、fixed/continuous charge、既存additive forceを
  受理する。terminal boundaryは`stick`/`escape`/`hold`に限り、反射・確率wallはfail-closedのままとする。
  係数評価policy、root内線形charge、axis hit後の新root ordinalはengine/proposal/RNGの内部数値契約であり、
  YAML/HDF5へ選択keyを追加しない。B02 revisionの受理範囲とpayloadは変更しない。
   再利用可能なDataBundleに未参照のlayout/fieldが含まれることは許すが、有効なphysicsや未対応modelを
  黙って無視しない。P07の静止壁lawはforce-freeな`linear_exact`、証明済み一定加速度の`quadratic_exact`、
  およびXY/RZの一般`rk4_reintegrated`で利用できる。一般RK4のsingle-facet active応答はhit後stateから
  残時間を継続する。surface departureはexact pathに加え、一般RK4ではfacet interiorにあり初速度が明確に
  domain内向きの場合だけ受理する。
- required fieldはすべて同じlayoutのnode fieldとし、unit、component、basisをmodel要求と
  完全一致させる。regularは全cell supportとgeometry全nodeのaxis box内包、P1/Q1はgeometryとの
  nodes/connectivity完全一致と全cell supportをprepareで要求する。vector fieldはXYなら
  `components=(x,y), stored_basis=cartesian_xy`、RZなら`components=(r,z), stored_basis=axisymmetric_rz`である。
  `fields.py`は、RZ geometryが`r=0`へ接する場合、またはboundaryless fully-supported regular support boxの
  `r_min=0`である場合を`axis_accessible`と一度だけ判定する。この時はrequired vectorの全layout-axis node radial成分を
  厳密に0とする。これはvector fieldのaxis regularityだけの判定である。`gravity_buoyancy_standard_v1`は
  `axis_accessible`と無関係に、annular domainを含む全RZ caseで`gravity_m_s2[0]`を厳密に0とする。
- fixed charge、dragなし、`cartesian_xy`で、参照するfieldの全canonical node値が厳密に一致する時だけ、
  粒子別一定加速度の`quadratic_exact` pathを使う。一様性に入力toleranceを設けない。材料boundaryを
  使うならtopology-completeを必須とし、boundarylessならrequired fieldが共有するfully-supported regular
  layoutのaxis boxをsupport domainとして必須とする。後者は各proposalの各座標極値を解析評価する。
- revision 3b/P06-U/P06-RZ/P06-Sのenclosure/material-boundary subsetに入らないstage-evaluated runと
  boundaryless unstructuredは、それぞれの包含証明またはmodel固有gateが完了するまでprepareで拒否する。
  P15 continuous chargeも新しいfield/layout経路を作らず、この既存subsetとsupport certificateを再利用する。
  catalog planは`has_force`、`evolves_continuous_state`、`requires_stage_evaluation`を別々に解決する。
  `plasma_continuous`だけを選んだplanは`false/true/true`であり、動的`Z`をlinear/quadratic exact pathへ送らない。
- boundaryを有効にするcaseは、volume cellから導いた全材料外周edgeを`line2`にちょうど1回含める。
  RZの両端`r=0`のaxis seamは材料壁ではなく`line2`へ登録しない。boundary rowを0件とした
  collision-free解析caseも同じengineのno-hit profileとして許可するが、stage評価が必要なら上記regular-box
  certificateを必須とする。
- 材料boundaryを持つtable sourceの初期位置はstrict interiorに限る。surface sourceだけが境界上releaseを
  明示し、位置とcanonical facet IDをrealized scheduleへ保持する。ownerと法線はprepared geometryが所有する。
  固定距離nudgeやresident local-coordinate authorityは作らない。
- RZ axis seamはwall eventにしない。linear/quadratic pathと一般RK4のいずれも、最初の材料hitより前の軸到達で
  meridional座標をfoldして残時間を継続する。一般RK4はsigned radial trial chartからcanonical field basisへ
  stageごとに写し、axis event prefixを再積分してからfoldする。axis crossingはboundary event/RNG ordinalを作らない。
  wallとaxisの双方を局在できた時だけ認証時間を比較し、不確かさが重なる場合はwallを優先する。一方が未決定なら
  refineし、axis端点とmaterial cornerの完全tieは一般RK4 cornerとしてfail-closedにする。
- 同時hitは`priority_then_combined_normal_v1`で解決する。最小priority subsetのresolved response signatureが
  一致しなければfailureとする。決定論的な`specular`と`restitution`は解決済み反発係数が同じ場合に限り互換で、
  入射中法線の正規化合成法線へ応答を一回だけ適用する。

## `case.yaml`

top-levelは次の10 keyを必須とし、未知keyとdefault補完を許可しません。

```yaml
format_version: 2
case:
  name: example
  data_path: case.h5
  expected_content_hash: "sha256:<64 lowercase hex>"
motion:
  mode: cartesian_xy
time:
  start_s: 0.0
  end_s: 1.0
  dt_s: 0.001
solver:
  integrator: rk4_fixed
  backend: cpu
  seed: 1234
  event:
    geometry_rtol: 1.0e-12
    roundoff_ulps: 64
    max_refinements: 48
    max_interactions_per_step: 8
    corner_policy: priority_then_combined_normal_v1
resources:
  memory_limit_mb: 4096
physics:
  charge:
    model: fixed
sources:
  - name: releases
    type: table
    table: particles
boundaries: []
output:
  trajectories: null
  probes: null
```

B02 inertial Brownianを選ぶ最小mappingは次です。`noise`は独自の温度fieldを持たず、FDT温度は
`drag.gas_temperature_field`が唯一のauthorityです。`interval_tree_depth`は数値pathを定める必須整数で、
既定値を持ちません。

```yaml
motion:
  mode: cartesian_xy
solver:
  integrator: ou_langevin
  backend: cpu
  seed: 1234
  event: {geometry_rtol: 1.0e-12, roundoff_ulps: 64, max_refinements: 48, max_interactions_per_step: 8, corner_policy: priority_then_combined_normal_v1}
physics:
  charge:
    model: fixed
  drag:
    model: epstein_linear
    revision: epstein_linear_v1
    gas_velocity_field: gas_velocity
    gas_density_field: gas_density
    gas_temperature_field: gas_temperature
    gas_mean_free_path_field: gas_mean_free_path
    gas_molecular_mass_kg: 6.6335209e-26
    delta: 1.0
    applicability: error
  noise:
    model: inertial_langevin_fdt
    revision: inertial_langevin_fdt_epstein_linear_frozen_start_v1
    interval_tree_depth: 6
```

この組合せでは`interval_tree_depth`を`0..10`とし、prepareした最大Epstein rateについて`gamma*dt<=1e6`、
実行中の各macro-rootについて有限な`0<gamma*h<=1e6`を要求します。係数はmacro-root開始stateで凍結します。
depth、noise revision、Brownian RNG/OU/split revision、coefficient policyはresult/checkpoint resume identityへ
必要なprovenanceとして使われます。

P18-R effective-gas linear EpsteinはB02 noiseと組み合わせず、次の正確なmappingを使います。

```yaml
physics:
  charge:
    model: fixed
  drag:
    model: epstein_linear
    revision: epstein_linear_effective_gas_sensitivity_v1
    gas_velocity_field: effective_gas_velocity
    gas_density_field: effective_gas_density
    gas_temperature_field: effective_gas_temperature
    gas_mean_free_path_field: effective_gas_mean_free_path
    gas_molecular_mass_kg: 6.6335209e-26
    delta: 1.3534291735288517
    maximum_speed_ratio: 1.0
    applicability: error
```

producerはこれらを一つの有効Maxwellian/pseudogasとして認証する。`maximum_speed_ratio`は省略不可の有限正値で
`<=1`、全stage・連続pathの適用域であって速度clampではない。`epstein_linear_v1`はこのkeyを受けず、従来どおり
上限`0.1`を使う。species配列、mixture rule、COMSOL名をYAMLへ追加しない。

P15 continuous chargeを選ぶ場合、`charge` mappingは次のkeyを正確に持ちます。field名は`case.h5`内の
canonical fieldを参照し、alias、既定値、追加optionを許可しません。

```yaml
physics:
  charge:
    model: plasma_continuous
    revision: oml_stationary_maxwellian_debye_huckel_v1
    electron_number_density_field: electron_number_density
    positive_ion_number_density_field: positive_ion_number_density
    electron_temperature_field: electron_temperature
    positive_ion_temperature_field: positive_ion_temperature
    positive_ion_velocity_field: positive_ion_velocity
    positive_ion_mass_kg: 6.6335209e-26
    applicability: error
```

P18-Cの集約二電流revisionはP15のmappingへkeyを足さず、次の独立した正確なmappingを使います。

```yaml
physics:
  charge:
    model: plasma_continuous
    revision: aggregate_relative_drift_regularized_two_current_v1
    electron_number_density_field: electron_number_density
    positive_ion_number_density_field: positive_ion_number_density
    electron_thermal_voltage_field: electron_thermal_voltage
    positive_ion_thermal_voltage_field: positive_ion_thermal_voltage
    positive_ion_velocity_field: positive_ion_velocity
    effective_positive_ion_mass_field: effective_positive_ion_mass
    screening_length_field: screening_length
    maximum_relative_ion_speed_m_s: 20000.0
    applicability: error
```

`maximum_relative_ion_speed_m_s`は正則化前の`|positive_ion_velocity-particle_velocity|`を囲う有限正値で、
速度clipではありません。電子・正イオンthermal voltageは`unit=V`、有効正イオン質量は`unit=kg`、screening長は
`unit=m`の正値scalar fieldです。集約revisionはこれらを単一の背景authorityとして使い、uniform値もparameterでなく
定数fieldとして保存します。single-species mass/temperature authorityを持つ既存Barnes ion dragとの併用は拒否し、
P18-Iのaggregate ion-drag revisionだけが完全一致する背景authorityを共有できます。

P21の集約三電流revisionは二電流mappingを暗黙に変更せず、次の独立した正確なmappingを使います。

```yaml
physics:
  charge:
    model: plasma_continuous
    revision: aggregate_relative_drift_regularized_three_current_v1
    electron_number_density_field: electron_number_density
    positive_ion_number_density_field: positive_ion_number_density
    negative_ion_number_density_field: negative_ion_number_density
    electron_thermal_voltage_field: electron_thermal_voltage
    positive_ion_thermal_voltage_field: positive_ion_thermal_voltage
    negative_ion_thermal_voltage_field: negative_ion_thermal_voltage
    positive_ion_velocity_field: positive_ion_velocity
    negative_ion_velocity_field: negative_ion_velocity
    effective_positive_ion_mass_field: effective_positive_ion_mass
    effective_negative_ion_mass_field: effective_negative_ion_mass
    screening_length_field: screening_length
    maximum_relative_ion_speed_m_s: 20000.0
    applicability: error
```

`negative_ion_number_density_field`は`unit=1/m^3`の有限非負scalarで、0を許します。
`negative_ion_thermal_voltage_field`と`effective_negative_ion_mass_field`はそれぞれ`unit=V`、`unit=kg`の正値scalar、
`negative_ion_velocity_field`は`unit=m/s`でcase座標と同じcomponents/basisを持つvectorです。一つの
`maximum_relative_ion_speed_m_s`が正負両イオンの相対speedを囲い、片方でも超過すればfail-closedにします。
`screening_length_field`は二revisionで共通する唯一のscreening authorityであり、coreは負イオン密度からscreeningを
再計算しません。負イオンspecies配列や集約rule、Case-P/COMSOL profileをYAMLへ追加せず、集約の根拠はcanonical field
producer provenanceに保存します。負イオン密度が0ならrate、Jacobian、global boundは二電流revisionと厳密に一致します。
三電流revisionの負イオンfield検証と
`|negative_ion_velocity-particle_velocity|` applicabilityは密度0でも省略しないため、public run全体の同一性には両revisionの
適用域条件が成立する必要があります。

P18-Iの二つのion-drag revisionは排他的に一つだけ選びます。relative-flow revisionは次の正確なmappingを使います。

```yaml
physics:
  charge:
    model: fixed
  ion_drag:
    model: screened_collection_orbital
    revision: relative_flow_screened_collection_orbital_aggregate_ion_v1
    positive_ion_number_density_field: positive_ion_number_density
    positive_ion_thermal_voltage_field: positive_ion_thermal_voltage
    positive_ion_velocity_field: positive_ion_velocity
    effective_positive_ion_mass_field: effective_positive_ion_mass
    screening_length_field: screening_length
    ion_neutral_mean_free_path_field: ion_neutral_mean_free_path
    maximum_relative_ion_speed_m_s: 20000.0
    applicability: error
```

image sensitivityは`ion_neutral_mean_free_path_field`と最大相対速度を持たず、次を使います。

```yaml
physics:
  charge:
    model: fixed
  ion_drag:
    model: image_orbital_sensitivity
    revision: electric_field_directed_image_orbital_sensitivity_v1
    positive_ion_number_density_field: positive_ion_number_density
    electron_thermal_voltage_field: electron_thermal_voltage
    positive_ion_thermal_voltage_field: positive_ion_thermal_voltage
    positive_ion_velocity_field: positive_ion_velocity
    effective_positive_ion_mass_field: effective_positive_ion_mass
    screening_length_field: screening_length
    electric_field: electric_field
    applicability: error
```

image revisionは`electron_thermal_voltage`と`positive_ion_number_density`からimage screeningを内部で一意に構成し、
producer由来のscalar ion speedやderived screeningを受けません。Coulomb electric categoryも選ぶ場合、その
`electric_field`名はimage revisionと一致しなければなりません。P18-C continuous chargeと併用する時は、各mappingの
共通field名を完全一致させ、relative-flowでは`maximum_relative_ion_speed_m_s`、imageでは
`electron_thermal_voltage_field`も共有します。

P18-D DEPは独立categoryとして次の正確なmappingを使います。

```yaml
physics:
  charge:
    model: fixed
  dielectrophoresis:
    model: quasistatic_spherical
    revision: quasistatic_spherical_gradient_e2_v1
    gradient_mean_e_squared_field: gradient_mean_e_squared
    medium_relative_permittivity: 1.0
    real_clausius_mossotti_factor: 0.5
    maximum_point_dipole_radius_m: 1.0e-6
```

`gradient_mean_e_squared_field`はunit `V^2/m^3`のnode vectorで、XYでは
`components: [x,y]` / `stored_basis: cartesian_xy`、RZでは`[r,z]` / `axisymmetric_rz`を要求します。
`medium_relative_permittivity`と`maximum_point_dipole_radius_m`は有限正値、
`real_clausius_mossotti_factor`は`[-0.5,1]`内の有限値です。全sourceの`electrostatic_radius_m`が宣言上限以下でなければ
prepareで拒否します。この上限はsolverが推測する安全率ではなく、同じfield solution/recoveryと明示した誤差基準を使って
producerが認証する値です。

DC/RF区分、solution/frequency、peak/RMSと時間平均の規約、gradient recovery、元field hash、半径上限の認証法と誤差基準は
`case.h5`のproducer provenanceへ保存します。これらはcanonical fieldの意味を説明するproducer情報であり、solver coreは
Eからgradientを再構成せず、YAMLへCOMSOL tagや回復optionを重複記載しません。B02の`noise` modelとの併用は拒否します。

P18-L liftはRZ/no-swirl専用の独立categoryとして、次の正確なmappingを使います。

```yaml
physics:
  charge:
    model: fixed
  lift:
    model: rarefied_vorticity_sensitivity
    revision: rarefied_vorticity_sensitivity_rz_v1
    gas_velocity_field: gas_velocity
    gas_density_field: gas_density
    gas_mean_free_path_field: gas_mean_free_path
    azimuthal_gas_vorticity_field: azimuthal_gas_vorticity
    lift_coefficient: 1.0
    applicability: error
```

`gas_velocity_field`はunit `m/s`、`components: [r,z]`、`stored_basis: axisymmetric_rz`のnode vector、
gas densityは正scalar `kg/m^3`、mean free pathは正scalar `m`、azimuthal vorticityは符号付きfinite scalar `1/s`である。
`lift_coefficient`は有限正値を明示し、既定値を持たない。producerはvorticityの円筒座標符号、元velocity solution、
回復規則をprovenanceへ保存し、coreはgas velocityを微分しない。粒子半径は`drag_diameter_m/2`だけから作り、
全stage/pathで`gas_mean_free_path/radius>=10`をfail-closedに要求する。

このrevisionは`motion.mode=axisymmetric_rz_meridional`だけを受理する。drag併用時はgas velocity/density/mean-free-path、
thermophoresis併用時はvelocity/mean-free-path、gravity併用時はdensityのfield名を完全一致させる。
Stokes--Cunningham、B02 `physics.noise`、Cartesian/3-Dとの併用は拒否する。resolved result manifestのlift entryには
model/revision、各field bindingと明示`lift_coefficient`を保存する。

P15-F ion dragはcharge categoryと独立に明示する。`charge: fixed`と併用する場合も各粒子の実際の
`charge_number`を使う。P15/P15-Dのcontinuous chargeと併用する場合は、両mappingのdensity、temperature、positive-ion velocity、
positive-ion massが完全一致しなければprepareのplan解決時、粒子配列を確保する前に拒否する。mean free pathは
ion-neutral collisionless gate専用である。

```yaml
physics:
  charge:
    model: fixed
  ion_drag:
    model: barnes_collisionless
    revision: barnes_collisionless_effective_speed_single_positive_ion_negative_debye_huckel_v1
    electron_number_density_field: electron_number_density
    positive_ion_number_density_field: positive_ion_number_density
    electron_temperature_field: electron_temperature
    positive_ion_temperature_field: positive_ion_temperature
    positive_ion_velocity_field: positive_ion_velocity
    ion_neutral_mean_free_path_field: ion_neutral_mean_free_path
    positive_ion_mass_kg: 6.6335209e-26
    maximum_ion_drift_ratio: 2.0
    applicability: error
```

field metadataはelectron/positive-ion densityが`unit=1/m^3`・scalar、両temperatureが`unit=K`・scalar、
positive-ion velocityが`unit=m/s`・XYでは`cartesian_xy`、RZでは`axisymmetric_rz`の2成分、ion-neutral mean
free pathが`unit=m`・scalarでなければならない。このrevisionのscalar値は正の有限値、ion velocityはcase座標basisの有限2成分、
`electrostatic_radius_m`は正の有限値でなければならない。`maximum_ion_drift_ratio`は
`|positive_ion_velocity-particle_velocity| / sqrt(8*k_B*T_i/(pi*m_i))`の連続path上限で、速度clampではない。
`applicability`は`error`だけを許し、適用外で別ion-drag modelへ切り替えない。

P16 thermophoresisも独立categoryとして明示する。`qtr`は中性気体の局所mass-average frameにおける
translational conductive heat fluxだけを表す。XYではvector fieldが`components: [x,y]` / `cartesian_xy`、
RZでは`[r,z]` / `axisymmetric_rz`で、unitは`W/m^2`である。他のgas fieldも既存basis規則を使う。

```yaml
physics:
  charge:
    model: fixed
  thermophoresis:
    model: waldmann_gallis
    revision: waldmann_gallis_free_molecular_single_species_heat_flux_v1
    gas_velocity_field: gas_velocity
    gas_temperature_field: gas_translational_temperature
    gas_translational_heat_flux_field: gas_translational_heat_flux
    gas_mean_free_path_field: gas_mean_free_path
    gas_molecular_mass_kg: 6.6335209e-26
    applicability: error
```

temperatureはscalar `K`、mean free pathはscalar `m`、velocityはvector `m/s`で、supported nodeのtemperatureと
mean free pathは正でなければならない。Epstein dragと併用する場合、velocity、temperature、mean-free-pathの
field名とmolecular massを完全一致させる。Stokes--Cunninghamとの併用、未知key、revision省略、適用外modelへの
fallbackは拒否する。Fourier熱流束の生成とgradient品質はcanonical producerの責務であり、このYAMLへgradientや
conductivityの隠れた既定値を持たせない。

P18-R effective-gas thermophoresisは同じcategory/modelと既存field形を使い、revisionと必須速度比だけを明示的に
分けます。

```yaml
physics:
  charge:
    model: fixed
  thermophoresis:
    model: waldmann_gallis
    revision: waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1
    gas_velocity_field: effective_gas_velocity
    gas_temperature_field: effective_gas_temperature
    gas_translational_heat_flux_field: effective_gas_translational_heat_flux
    gas_mean_free_path_field: effective_gas_mean_free_path
    gas_molecular_mass_kg: 6.6335209e-26
    maximum_speed_ratio: 1.0
    applicability: error
```

`gas_translational_heat_flux_field`はproducer-owned `q_eff [W/m^2]`であり、total/convective/radiative/electron/ion heat
fluxではない。producerはone-effective-Maxwellian/pseudogas reductionをprovenanceへ保存し、coreはtemperature gradient、
species配列、mixture ruleを回復しない。`maximum_speed_ratio`は有限正値かつ`<=1`で省略不可、P16 single-species
revisionはこのkeyを受けず上限`0.1`を維持する。両effective-gas revisionは`lambda/a>=10`も全stage/pathで要求する。

単一正イオン種の有限相対driftを扱うP15-D revisionは、同じfield契約に明示的なrun-wide drift上限を一つだけ
追加します。この値は
`|positive_ion_velocity-particle_velocity| / sqrt(8*k_B*T_i/(pi*m_i))`の上限であり、速度を変更する
clampではありません。

```yaml
physics:
  charge:
    model: plasma_continuous
    revision: oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1
    electron_number_density_field: electron_number_density
    positive_ion_number_density_field: positive_ion_number_density
    electron_temperature_field: electron_temperature
    positive_ion_temperature_field: positive_ion_temperature
    positive_ion_velocity_field: positive_ion_velocity
    positive_ion_mass_kg: 6.6335209e-26
    maximum_ion_drift_ratio: 2.0
    applicability: error
```

trajectory frameを保存する場合だけ、`trajectories`を次のmappingにします。

```yaml
output:
  trajectories:
    selection: all
    schedule:
      explicit_times_s: [0.0, 0.25, 1.0]
```

選択粒子だけのstate probeを保存する場合は次のmappingにします。particle IDと時刻はいずれもstrictly increasing、
時刻は閉run区間内でなければならず、IDはrealized sourceに存在するものへprepare時に解決します。

```yaml
output:
  trajectories: null
  probes:
    particle_ids: [7, 42]
    schedule:
      explicit_times_s: [0.0, 0.25, 1.0]
```

P07で実装済みのsurface sourceとwall lawの最小例は次である。

```yaml
sources:
  - name: wall_release
    type: surface
    boundary_group: source_wall
    count: 100
    particle_id_start: 1000
    particle:
      charge_number: 0.0
      mass_kg: 1.0e-18
      drag_diameter_m: 1.0e-7
      electrostatic_radius_m: 5.0e-8
      displaced_volume_m3: 0.0
      model_weight: 1.0
      material_id: 0
    position: {model: uniform, measure: line_length}
    velocity: {model: normal, direction: into_domain, speed_m_s: 1.0}
    release: {model: fixed, time_s: 0.0}
boundaries:
  - boundary_group: mirror
    priority: 10
    law: specular
  - boundary_group: collector
    priority: 20
    law: probabilistic_stick
    probability: 0.4
    otherwise:
      law: restitution
      normal_restitution: 0.8
      tangential_restitution: 0.6
```

RZのuniform位置では`measure`を`meridional_length`または`revolved_area`とする。3-D回転面上の一様fluxは
`revolved_area`を使い、`meridional_length`は2-D断面上の一様分布である。単一facet上の決定論的な
verificationには`position: {model: edge_fraction, fraction: 0.5}`も使える。固定vector速度は
`velocity: {model: fixed, value_m_s: [v0, v1]}`で表す。

共通構造の規則は次のとおりです。

- `start_s`、`end_s`、`dt_s`とfloat64で評価した`end_s-start_s`はfinite、
  `end_s > start_s`、`0 < dt_s <= end_s - start_s`。
- `motion.mode`は`cartesian_xy`または`axisymmetric_rz_meridional`。data座標との組合せは
  `engine.prepare`が一度だけ検査する。
- `solver.integrator`は`rk4_fixed`、`exponential_midpoint`、または`ou_langevin`、`solver.backend`は現行`cpu`だけを許可する。
  dragに対する`maximum_dt_over_tau < 2.5`は`rk4_fixed`だけのprepare gateである。線形dragでは最大rate、
  finite-speed Epsteinでは速度Jacobianの最大固有値上界を使う。`exponential_midpoint`は各stageで凍結した
  正のrelaxationを解析更新するため同じ上限を課さない。両continuous-charge revisionは
  `rk4_fixed`では同じ4 stage、`exponential_midpoint`では同じpredictor時刻のexplicit midpointにより運動と
  同時更新する。どちらも`dt * charge_lipschitz <= 0.5`、charge invariant、continuous applicabilityを必須とし、
  別integratorへのfallbackやcharge-only subcycleを行わない。
- 全ion-drag revisionは両integratorで`explicit_acceleration`として同じstage評価を使う。加速度包絡と、
  Barnesおよびrelative-flow revisionの連続drift gateは
  既存proposal/enclosureへ統合するが、現revisionはmodel固有の時間step設定を追加しない。利用caseは`dt`系列で
  trajectory収束を確認し、適用域違反をstep subdivision、floor、scaleで隠さない。
- seedはunsigned 64-bit範囲、memory上限は正の整数である。最小slabがmemory planへ収まらないcaseは
  `simulate`のprepareで運動開始前に拒否する。schema v2は粒子内thread数を入力契約にせず、旧
  `resources.threads`を未知keyとして拒否する。独立caseのprocess並列はsolver外の運用で行う。
  `memory_limit_mb`はsolver-owned predicted peakの上限であり、OS process RSSのhard capではない。`load_case`はcanonical
  numeric bytesだけをearly gateに使い、geometry/source/physics/outputを含む完全なplanは`simulate`のprepareが解決する。
- `solver.event`はscale-awareなgeometry相対許容差、float64丸めのULP係数、event-drivenな局在・残時間
  再分割のrefinement上限、各residual-work intervalを再分割するまでの境界interaction閾値、corner policy
  revisionを明示する。子intervalではinteraction countを0へ戻すがrefinement depthを引き継ぐため、
  `max_interactions_per_step`はmacro step全体のhard hit総数ではない。絶対的な長さ・時間budgetは
  後続の単一numerics resolverがgeometry、速度、時刻scaleから導く。`geometry_rtol`は局所facet長に対する
  相対値であり、例の`1e-12`は製品defaultではない。scale選定は[`numerics.md`](numerics.md)の第5節に従う。
- `0 < geometry_rtol < 1`とし、`roundoff_ulps`、`max_refinements`、
  `max_interactions_per_step`は正整数とする。`corner_policy`は非空revision IDとしてloadし、対応可否は
  event algorithmを所有する`engine.prepare`が粒子配列を確保する前に拒否する。
- physicsは`charge`を必須とし、他categoryは設定が存在するときだけ有効。無効を`null`で表さない。
  `charge: {model: fixed}`は`dZ/dt=0`だけを意味し、初期`charge_number`を所有しない。
- fixed chargeのmappingは正確に`{model: fixed}`だけを許す。model固有parameterを追加して第二の初期値や
  hidden optionを作らない。
- P18-Rの二revisionだけが`maximum_speed_ratio`を必須とし、有限正値かつ`<=1`を要求する。既存linear/P16 mappingへ
  このkeyを追加して上限`0.1`を変更することも、省略時の既定値も許可しない。producer-certified pseudogasは入力provenanceで
  あり、solver caseへspecies array、gradient recovery option、COMSOL profileを追加しない。
- `ou_langevin`と`physics.noise`は必ず同時に指定する。noise mappingは正確に`model`、`revision`、
  `interval_tree_depth`だけを持ち、modelは`inertial_langevin_fdt`、depthは整数`0..10`とする。
  `inertial_langevin_fdt_epstein_linear_frozen_start_v1`は`motion.mode=cartesian_xy`、fixed charge、
  `epstein_linear_v1`だけを許し、noise以外のoptional physics categoryを設定しない。
  `inertial_langevin_fdt_epstein_linear_rz_meridional_projected_v1`は`axisymmetric_rz_meridional`、対応する
  native/effective-gas線形Epstein、fixed/continuous charge、既存additive forceを許す。どちらも全boundary groupは
  parameterなしの`stick`、`escape`、`hold`だけを許す。FDT温度はdragの`gas_temperature_field`から
  `theta=k_B*T_g/mass_kg`として導き、noise側の温度、反射・確率wallへのfallbackを許可しない。
- continuous chargeは上記いずれかの正確なmappingだけを許し、`model=plasma_continuous`、対応する明示
  `revision`、`applicability=error`を必須とする。shifted-Maxwellian revisionだけが正かつfiniteな
  `maximum_ion_drift_ratio`を必須とし、stationary revisionへこのkeyを追加することも、省略時の既定値も許可しない。
  P15二式の`positive_ion_mass_kg`は正かつfiniteである。4本のscalar fieldはそれぞれ
  electron/positive-ion number densityが`unit=1/m^3`、temperatureが`unit=K`で、
  `components=(value), stored_basis=scalar`かつ全nodeで正とする。positive-ion velocityは`unit=m/s`で、
  XYなら`components=(x,y), stored_basis=cartesian_xy`、RZなら
  `components=(r,z), stored_basis=axisymmetric_rz`とし、既存のaxis regularity検査を受ける。
  選択された全source粒子の`electrostatic_radius_m`は正でなければならない。shifted-Maxwellian revisionは
  初期`charge_number<=0`、全宣言rangeで非正平衡を認証可能、全pathで宣言drift比以下であることも要求する。
  aggregate revisionだけは、2本のdensity fieldに加えてthermal voltage `V`、有効正イオン質量`kg`、screening長`m`の
  正値scalar fieldと、正かつfiniteな`maximum_relative_ion_speed_m_s`を必須とする。P15用のtemperature key、scalar mass
  parameter、drift-ratio keyを混在させず、全stage/pathで宣言相対速度以下であることを要求する。
- 各physics category、surface sourceのposition/velocity/releaseは非空の`model`を持つmapping。
  model固有parameterは後続の`engine.prepare`が一度だけ解釈する。
- sourceは少なくとも1件必要で、source名は一意。
- table sourceはHDF5のrealized tableを一度だけ参照する。
- surface sourceは`boundary_group`、正の`count`、非負の`particle_id_start`、particle authority、
  position、velocity、releaseを持つ。生成IDは
  `[particle_id_start, particle_id_start + count)`の連続したsigned-int64範囲で、table sourceおよび他の
  surface sourceと重複させない。
- table sourceの初期電荷はHDF5の`charge_number`、surface sourceの初期電荷は
  `particle.charge_number`だけが所有する。
- particleの`mass_kg`、`drag_diameter_m`、`model_weight`は正、`charge_number`はfinite、`electrostatic_radius_m`と
  `displaced_volume_m3`は非負、`material_id`は非負整数。
- boundary lawはHDF5に存在する全groupへちょうど一つ割り当て、非負の`priority`を明示し、
  未知groupを許可しない。priorityの数値が小さいgroupほどcornerで優先する。
- `boundaries[].law`は入力model selectorである。PreparedRunの非公開dense codeや結果schemaの列名ではない。
  eventには選択されたtop-level lawの安定semantic IDを`law_id`として保存し、compound lawの分岐結果は
  `outcome`と必要なdraw referenceで表す。`law_name`はv1 schemaに設けない。
- `stick`、`escape`、`hold`、`specular`は`boundary_group`、`priority`、`law`以外のmodel parameterを持たない。
  `hold`はhit位置、hit時速度、hit時電荷を保持するinactiveな`held` terminalであり、paused particleや再飛散を表さない。
  `specular`は法線係数・接線係数がともに1の完全鏡面を意味し、反発係数を指定した入力は拒否する。
  `restitution`は`normal_restitution`を`(0,1]`、`tangential_restitution`を`[0,1]`でともに必須とし、
  それ以外のparameterを持たない。
  `probabilistic_stick`は定数`probability`を`[0,1]`で必須とし、これと`otherwise`以外のparameterを持たない。
  `otherwise`はparameterなしの`{law: specular}`、または上記二つの反発係数をともに持つ
  `{law: restitution, ...}`だけを受理する。moving wall、diffuse law、速度依存確率は未対応である。
- P07のsurface positionは`edge_fraction`または`uniform`だけである。XYのuniform measureは
  `line_length`、RZは`meridional_length | revolved_area`。velocityは`fixed` vectorまたは
  `normal + into_domain + speed_m_s`、releaseは`fixed + time_s`だけを受理する。角度・速度・時刻分布は
  対応modelを追加するまで拒否する。
- surface releaseは外向き法線`n`に対する`v·n`を先に分類する。roundoff budgetより明確な負値はdeparture、
  明確な正値はzero-time impactとする。budget内でも非零なら曖昧として拒否し、float64で厳密に0の時だけ、
  証明済み一定加速度pathの`a·n`を同じscale-aware規則で判定する。負ならsource facetからのdeparture、正なら
  zero-time impact、決定不能ならfail-closedである。後者のimpactはterminal `stick`/`escape`を適用できるが、
  反射後にdomain内向きdepartureを作れない応答は失敗する。内向き加速度のdeparture tokenは別wall hitまで保持し、
  各intervalでeventsが再証明するため、fixed releaseの意味はmacro partitionに依存しない。
- 一般RK4のsurface releaseは明確に負の`v·n`だけをdepartureとして受理し、tangent、facet端点/corner、またはroundoff幅内の曖昧な
  contactを加速度推定で補わない。surfaceまたはsingle-facet activeな壁応答後は、event v8がstart-contactのfacet内点性と
  区間速度包絡の厳密内向きを証明したfacetだけを候補から外す。証明不能なcontactは候補を保持して細分化し、
  budget内で決定できなければ失敗する。
- table release timeは閉区間`[start_s, end_s]`内に入る。
- trajectory時刻は空でなくfinite、狭義単調増加、重複なしで、閉区間`[start_s,end_s]`内に入る。
  release前の粒子はframeへ含めず、release時刻ちょうどのframeにはrealized sourceの初期状態を入れる。
- trajectory frameはboundary eventに対して右連続である。hit時刻ちょうどのstickは速度0の
  post-event状態、holdはhit位置・hit時速度・hit時電荷を保持したheld状態、activeな反射は反射後速度、
  escapeは行なしとし、以後のframeにもescaped粒子を含めない。
- HDF5 pathはYAML fileのdirectoryを基準に解決する。
- YAML mappingの重複key、非文字列key、非有限数を拒否する。

## `case.h5`

v1のdata表現は2D Cartesian XY（`cartesian_xy`）と軸対称RZ（`axisymmetric_rz`）を対象にします。
粒子のmotion modeはYAMLが別に所有します。数値datasetのdtypeは
little-endian固定、文字列は可変長UTF-8です。

```text
/meta/schema_version                         int32 scalar = 1
/meta/coordinate_system                      UTF-8 scalar
/meta/coordinate_units                       UTF-8 scalar = "m"
/meta/provenance_json                        canonical UTF-8 JSON scalar

/geometry/nodes_m                            float64 [Nnode,2]
/geometry/node_external_id                   int64 [Nnode] optional
/geometry/cells/tri3                         int64 [Ntri,3] optional
/geometry/cells/tri3_domain_id               int32 [Ntri] with tri3
/geometry/cells/quad4                        int64 [Nquad,4] optional
/geometry/cells/quad4_domain_id              int32 [Nquad] with quad4
/geometry/boundary/line2                     int64 [Nedge,2]
/geometry/boundary/external_id               int64 [Nedge] optional
/geometry/boundary/boundary_id               int32 [Nedge]
/geometry/boundary/group_id                  int32 [Nedge]
/geometry/boundary/material_id               int32 [Nedge]
/geometry/boundary/owner_cell_type           uint8 [Nedge]
/geometry/boundary/owner_cell_local_index    int64 [Nedge]
/geometry/boundary/orientation               int8 [Nedge]
/geometry/groups/names                       UTF-8 [Ngroup]

/layouts/<name>/kind                         UTF-8 scalar: regular|p1_tri|q1_quad
/layouts/<name>/regular/axes/axis0_m         float64 [N0], regular only
/layouts/<name>/regular/axes/axis1_m         float64 [N1], regular only
/layouts/<name>/regular/cell_support         uint8 [N0-1,N1-1]
/layouts/<name>/unstructured/nodes_m         float64 [Nnode,2], P1/Q1 only
/layouts/<name>/unstructured/connectivity    int64 [Ncell,3|4]
/layouts/<name>/unstructured/cell_support    uint8 [Ncell]

/fields/<name>/layout                        UTF-8 scalar
/fields/<name>/association                   UTF-8 scalar: node|cell
/fields/<name>/components                    UTF-8 [C]
/fields/<name>/stored_basis                  UTF-8 scalar
/fields/<name>/values                        float64 [N,C]
/fields/<name>/unit                          UTF-8 scalar

/sources/<name>/particle_id                  int64 [N]
/sources/<name>/release_time_s               float64 [N]
/sources/<name>/position_m                    float64 [N,2]
/sources/<name>/velocity_m_s                  float64 [N,2]
/sources/<name>/charge_number                 float64 [N]
/sources/<name>/mass_kg                       float64 [N]
/sources/<name>/drag_diameter_m               float64 [N]
/sources/<name>/electrostatic_radius_m        float64 [N]
/sources/<name>/displaced_volume_m3           float64 [N]
/sources/<name>/model_weight                  float64 [N]
/sources/<name>/material_id                   int32 [N]
```

geometryは`tri3`または`quad4`を少なくとも一要素持ちます。tri3はCCW、quad4はQ1参照節点
`(-1,-1),(+1,-1),(+1,+1),(-1,+1)`に対応する非反転順です。v1の全cellは粒子が移動できる
domainであり、solid volumeを混在させません。wallなし解析ケースのためboundaryは0件を許可します。

`owner_cell_type`は1=tri3、2=quad4です。`orientation=+1`は`line2`がownerのCCW edge順、
`-1`は逆順であることを示します。線分`t=(dx,dy)`の外向き法線は
`orientation * (dy,-dx)/|t|`です。canonicalな`facet_id`は`line2`の0始まりrow indexで、event候補の
identityにはこれを使います。`boundary_id`はproducer側の物理boundary labelであり、複数facetで同じ値を
共有できます。group IDは0からのdense IDで、`names[id]`が名称です。

regular axisはstrictly increasingで、field rowはC-order flattenです。supportは0/1のcell maskです。
点のsupportは、support=1であるcellの**閉包の和**とします。したがってsupported cellとmasked cellの
共有面はsample可能で、候補中の最小supported cell IDを決定論的なownerとします。候補がすべてmaskedなら
support外です。layoutとfieldはballistic caseでは0件を許可します。
fieldは既存layoutを参照し、成分数とnode/cell associationに対応するshapeを持ちます。
required vector fieldのcanonical metadataはXYで`["x","y"] / cartesian_xy`、RZで
`["r","z"] / axisymmetric_rz`です。scalarは両座標で`["value"] / scalar`です。P06-RZはこの既存field datasetを
使い、schema versionやRZ専用datasetを追加しません。

realized tableは空を許可せず、`particle_id`を全tableを通して一意な非負signed-int64整数とします。
実行caseでは、参照されたtable IDとYAML surface sourceが予約するID範囲も相互に一意とします。
`mass_kg`、`drag_diameter_m`、`model_weight`は正、`electrostatic_radius_m`、
`displaced_volume_m3`、`material_id`は非負です。全実数値はfiniteでなければなりません。producerにNaNで
表現された材料側値がある場合、adapterは明示supportを先に確定し、masked-only DOFだけを決定論的な
有限placeholderへ正規化して方法と件数をprovenanceへ残します。NaN自体からsupportを推測せず、supported
cellが参照するDOFや物理補間へplaceholderを使いません。

provenance JSONは次のtop-level keyを必須とし、追加producer metadataを許可します。

```json
{
  "producer": "producer-neutral name",
  "producer_version": "version",
  "source_sha256": "sha256:<64 lowercase hex>",
  "field_semantics_revision": "revision",
  "producer_metadata": {}
}
```

HDF5 attribute、soft/external link、hard-link alias、external storage、VDS、未知objectを拒否します。
layout、field、sourceの名称は`[A-Za-z][A-Za-z0-9_]*`です。

## Logical content hash

hashはHDF5のchunk、compression、object address、作成順序に依存しません。

1. SHA-256へdomain header `chamber-particles-case\0v1\0`を投入する。
2. 全dataset recordをabsolute pathのUTF-8 byte順でsortする。
3. 各recordについて、pathとlogical dtypeを`uint64 little-endian byte length + bytes`で投入する。
4. `uint64 rank`、続いて各dimensionを`uint64`で投入する。
5. numeric値は検証済みC-order little-endian bytesを投入する。文字列は各値を
   `uint64 UTF-8 byte length + bytes`で投入する。
6. `sha256:<lowercase hex>`として返す。`-0.0`は正規化せず、hash自身はHDF5へ保存しない。

provenance JSONはduplicate keyと非有限数を拒否し、sorted key、compact separator、UTF-8の
canonical表現にしてからhashします。tupleの入力順ではなくdataset path順がidentityを決めます。

## Writer publication

`case_format.write()`は同じdirectoryへ完全な一時HDF5を書き、closeとfile fsync後にhard linkを
確定pathへ作成し、一時名を削除します。既存pathは上書きしません。hard linkを提供しないfilesystemでは
非atomicな代替処理を行わず失敗します。成功時はschema versionとlogical content hashを返します。

## v1の範囲外

時間依存field、revolved fieldを使うCartesian 3D軌道、完全3D geometry、tet4、surface triangleは
後続schemaで追加します。v1 readerへ未知datasetを先行投入せず、意味が変わる変更はschema versionを
更新します。
