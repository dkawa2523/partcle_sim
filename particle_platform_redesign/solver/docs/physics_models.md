# Physics model revisions

この文書はproductionで実装したversioned modelを扱う。model選択とfield参照はYAMLへ明示し、field名の自動探索、
適用域外での別modelへの自動切替、backend固有の係数を認めない。runtime contributionはdragを、stageで
凍結した正の`linear_relaxation(rate, target_velocity)`、その他の力を`explicit_acceleration`として返す。
モデルの選択・required field・単位・成分・基底は`physics/catalog.py`が一つのownerである。
`chamber-particles models`は同じdescriptorを表示し、stageの実行bindingも同じfield仕様を使う。
field値の条件は`value_constraint: finite | positive | nonnegative`で宣言する。`finite`は符号自由、
`positive`は厳密に正、`nonnegative`は0を含む。負イオン密度やprescribed chargeの傾き・ion rateは非負、
electron exponent offsetは符号自由である。この宣言を公開情報、prepareのprimitive extrema検査、
stageの粒子局所検査が使い、runtimeがmodel名を見て後から符号規則を追加しない。
非負fieldの全節点・全snapshotがconstant zeroである証明がある場合、prepareはzeroの区間を検査とmodel boundsの
両方へ渡す。0の外向きroundoff enclosureを負の物理値と解釈しない。正の定数と非constantのenclosureは保持し、
weight和の丸めを無視して定数値へ縮めず、負下端をepsilonやclampで受理しない。
現行方程式と適用域を以下に記す。過去revisionの導入順はGit履歴と保存receiptに保持する。
実行に採用した正確なcatalog/runtime/compiled revisionは各`run.json`を参照する。
continuous modelは有限の`J<=0`、fixed chargeは`J=0`を返し、運動と同じstageで評価する。

選択時のdescriptorと具体的なfrozen family parameter recordを`SelectedModel`が保持する。
field role、callerのfield名、解決した既定値・物性authorityを同じrecordから投影し、実行の
`resolved_physics_models`へ残す。model名をparameter型から推測せず、field roleをkeyの綴りから推測しない。
各モデルは自身のroleを共通primitive表へ直接結線する。背景場の同一性を要求する組合せはcatalogの
composition検査が所有し、dragやchargeの有無で隣のモデルの入力bindingを借りない。
stageにはdescriptor、dict、callbackを渡さず、
`compiled.py`が所有する固定のfamily binding、独立粒子物性、primitive slab、出力recordを渡す。
charge → drag → thermophoresis → ion drag → DEP → lift → electric → gravity のstage評価・加算順を維持する。

各descriptorは、stage入力と区別してlocal applicabilityとlocal additive boundが参照するfield role、
constant accelerationを証明できるfamilyかを明示する。prepareはそのroleをcallerのfield名とoffsetへ解決し、
global family bounds、local証明入力、constant証明を一つの`PreparedPhysicsCertificates`へ保持する。
実際に準備したglobal force boundが選択された全forceを覆わなければprepareを失敗させる。
local/continuousのproof family codeはstage算術のselectorと別であり、各対応revisionについてprepareで
stageとproofを照合して固定する。未登録revision・codeを残りのfamilyへ解釈する分岐は持たない。
noiseも`none / FDT / prescribed force diffusion`と温度・friction・diffusionの各offsetを明示する。
offsetの負号や別categoryの型からnoise方式を推測しない。
全modelがconstant証明を支持し、実際のfieldと全contributionがconstantと確認できた場合だけexact specializationを使う。
local/constant証明の`None`は利用不能であり、ゼロforceではない。familyの適用域gateが不要な場合の`None`とも
区別する。point式、global enclosure、local enclosure、適用域は
異なる数学的役割を保ち、point式からboundを自動推定しない。

線形・finite-speed EpsteinとStokesのcoefficient、Talbot・Waldmann・Saffman・rarefied liftの係数、
Barnes・relative-flow・image ionのpoint law、stationary/shifted OML・aggregate二/三電流・prescribed chargeの
収集電流は、それぞれ`forces.py`または`charge.py`のscalar bodyが所有する。array入口はshape/domain検査と
例外を持ち、compiled stageは同じbodyと行別statusを使う。
array入口は有限primitiveから派生したunderflow・overflow・math domain errorも`PhysicsEvaluationError`へ変換する。
負電位専用revisionの正電位など、非対応の状態は指数評価の前に拒否する。力の加算順とstageの算術順は保持する。
electric、DEP、gravity/buoyancyもconstant証明とstageが同じpoint bodyを使う。prepareで必要な係数はPython bodyを
使い、checkのための追加stage JITを起動しない。Pythonのarray検査でsqrt/hypotや丸めが変わる箇所は、共有bodyの
相互比較だけで認めず、独立速度積分・Decimal・解析極限・boundの包含検査で確認する。これらのoracleはverificationに残す。
共有body同士の一致は結線検査であり、独立した科学的資格の代用にはしない。

同じcategoryのmodel追加では、catalogのdescriptorとconcrete parameter解決、familyのpoint式とbounds、
直接role binding、revisionごとのproof宣言、既存stage内の当該family、公開例と独立検証を更新する。
電子・イオン収集のような連成条件はcomposition検査へ明示し、integratorのmodel名分岐を増やさない。
削減はこれらの当該revisionと呼出例を同時に撤去する。result表示やworkerのために別のmodel一覧を更新する必要はない。新categoryには合成順と必要certificateの
実装が必要であり、metadataの宣言だけでは新しい科学モデルを実装したことにならない。

## Particle authority

- 慣性はsourceの`mass_kg`だけが所有する。
- dragは`drag_diameter_m`、電気的な有限寸法は`electrostatic_radius_m`、浮力は
  `displaced_volume_m3`を使い、相互に再構成しない。
- Barnes ion dragの粒子半径も`electrostatic_radius_m`だけを使い、drag径から再構成しない。
- 初期電荷数`Z0`はrealized internal/surface tableの粒子rowにある`charge_number`だけが所有する。
- `charge: {model: fixed}`は`dZ/dt=0`だけを意味する。電荷Coulomb値は常に`q=Z*e`から求める。

## Explicit coefficient reference revisions

以下は外部指定の数学的方程式を既存stageで評価するreference modelである。係数の生成・native式との対応は
producerが所有する。通常のEpstein/FDT/OML/kinetic modelの適用域や物理認定を変更しない。

| category / model | revision | 必須係数・式 |
|---|---|---|
| drag / `prescribed_linear_friction` | `prescribed_linear_friction_v1` | `gas_velocity_field` [m/s]、`friction_coefficient_field` beta [kg/s]>0。F=beta(u-v)、gamma=beta/m |
| noise / `prescribed_force_diffusion` | `prescribed_force_diffusion_midpoint_2d_v1` | `force_spectral_density_field` D [N²s]>0、`interval_tree_depth`、optional `adaptive_max_depth`。theta=D/(2 beta m)を同じjoint OUへ渡す |
| charge / `prescribed_exponential_linear` | `prescribed_exponential_linear_collection_v1` | `electron_rate_reference_number_s` Ae>=0、`electron_exponent_offset_field` c [1]、`electron_charge_slope_field` alpha [1]>=0、`ion_rate_field` Ai [1/s]>=0、`ion_charge_slope_field` betaZ [1]>=0 |
| ion_drag / `prescribed_charge_polynomial` | `prescribed_charge_polynomial_v1` | `constant_force_field` F0、`linear_charge_force_field` F1、`quadratic_charge_force_field` F2（各[N]）。F=F0+Z F1+Z² F2 |
| lift / `prescribed_rotational_lift` | `prescribed_rotational_lift_v1` | `gas_velocity_field` [m/s]、`rotation_coefficient_field` c_rot [kg/s]。F=c_rot*(u_y-v_y,-(u_x-v_x)) |

chargeは`G=-Ae*exp(c+alpha*Z)+Ai*(1-betaZ*Z)`、
`J=-alpha*Ae*exp(c+alpha*Z)-betaZ*Ai`を運動と同じstageで評価する。
有限のforward-invariant charge範囲をprepareし、連続pathのforce/charge boundsへ同じ範囲を用いる。
電子rateへfloorを足してoffsetを作らず、外部producerは元rateとの節点再構成とoff-node転送差を検証する。
RZで到達可能な軸のrotationは厳密に0である。各vector係数も通常の基底・axis規則に従う。

各reference modelは次の`particle_properties`を必須とする。prepareは全source rowの同名authorityと完全一致を
要求し、同時に選んだreference modelsのbindingも一致させる。このrevisionは均一物性cohortに限定する。
同じ物性のsource subsetは使える。元の実現ID集合・solution・mesh・primitive・生成式・定数の由来は
producer provenanceへ保持し、数値bindingだけをnative真正性の証明にはしない。

```yaml
particle_properties:
  mass_kg: 2.0
  drag_diameter_m: 1.0e-6
  electrostatic_radius_m: 5.0e-7
  displaced_volume_m3: 5.235987755982989e-19
```

上は構文を示す独立な検証用の値で、装置caseのdefaultではない。contact radiusはこのbindingから再構成しない。
prescribed noiseはprescribed linear frictionと組み合わせ、FDT noiseは既存のlinear Epstein条件を維持する。
ion polynomialはfixed chargeまたはprescribed chargeと組み合わせる。速度・電荷の保存履歴を再生しない。

`electric_coulomb_charge_cap_v1`は`model: coulomb`、`electric_field`に加えて正かつfiniteな
`maximum_abs_charge_number`を必須とし、`F=e*clip(Z,-cap,cap)*E`を評価する。
capはCoulomb寄与だけに作用する明示model parameterで、resident Z、charge rate、ion dragを変更しない。
`electric_coulomb_v1`にcap parameterを与える入力は拒否する。

## Neutral-gas background authority

drag、thermophoresis、lift、gravity/buoyancyを同時に選ぶ場合、二つ以上のmodelが宣言する同じ
中性気体primitiveは一つのcanonical authorityを共有する。`gas_velocity_field`、`gas_density_field`、
`gas_temperature_field`、`gas_dynamic_viscosity_field`、`gas_mean_free_path_field`はfield名を、
`gas_molecular_mass_kg`は値を完全一致させる。liftを選ばない場合もこの規則を適用し、例えばdragと
gravity/buoyancyが異なるdensity fieldを参照するcaseを受理しない。modelが使わないprimitiveを新たに要求せず、
同じfield名で異なるproducer意味を隠すことも許可しない。

linear EpsteinとWaldmann--Gallisを併用する場合は、native single-species同士またはeffective-gas sensitivity同士に
限定する。field名が同じでもnative/effectiveを混在させない。有限速度Epsteinは一つの局所Maxwellianを入力とするため、
single-speciesまたはproducer認証済みpseudogasのどちらとも組み合わせられるが、選択したWaldmann revisionと同じ
背景へのfield bindingとprovenanceが必要である。coreはspecies配列やmixture closureを推測しない。

## Coordinate basis and RZ regularity

vector required fieldはcaseのdata座標と完全一致させる。Cartesian XYでは
`components=(x,y), stored_basis=cartesian_xy`、axisymmetric RZでは
`components=(r,z), stored_basis=axisymmetric_rz`である。gas velocityとelectric fieldはこの同じ規則を使い、
producer名や成分順を推測しない。scalar fieldは両座標で`components=(value), stored_basis=scalar`のままである。

`epstein_linear_v1`、`electric_coulomb_v1`、`gravity_buoyancy_standard_v1`の式はsample済みcanonical成分に対して
component-wiseに評価するため、XY/RZで別model revisionを作らない。ion velocityとion-drag accelerationも同じ
vector basis規則に従う。RZのRK4 trialだけはengineがsigned radial
chartを使い、stage sampling前にcanonical `(r,z)`位置・速度へ写し、評価後のradial加速度をsigned chartへ戻す。
符号反転はvelocity norm、相対速度norm、component絶対boundを変えないため、Epsteinの
`lambda/a`と`|u-v|/c_bar`のapplicability規則もXYと同一である。

axis accessibilityは`fields.py`が一度だけ決める。RZ geometryが`r=0`へ接する場合に加え、boundarylessで
fully-supported regular support boxの`r_min=0`である場合も、粒子が軸へ到達可能なのでtrueである。その時は
全required RZ vector fieldのlayout axis nodeでradial成分が厳密に0でなければならない。regular layoutはaxisを含み、
P1/Q1は`r=0` nodeを直接検査する。非零値をclamp、平均、toleranceで隠さずprepareで拒否する。material boundaryで
軸から隔離され、geometryも軸へ接しないcaseにはこのvector fieldのaxis regularity条件を課さない。
`gravity_buoyancy_standard_v1`のradial成分制約はこの`axis_accessible`判定を共有せず、下記のとおり全RZ domainへ課す。

## `epstein_linear_v1`

球半径`a=drag_diameter_m/2`、気体密度`rho_g`、平均分子速度

\[
\bar c=\sqrt{8k_BT/(\pi m_g)}
\]

に対し、subsonic free-molecularの線形形を

\[
\boldsymbol F_d=\beta(\boldsymbol u_g-\boldsymbol v),\qquad
\beta=\frac{4\pi}{3}a^2\rho_g\bar c\,\delta,qquad
\tau=m_p/\beta
\]

とする。`delta`は表面反射・accommodationを表す無次元係数で、caseが必ず明示する。独立microcase C02/C03は
数値式の切分けのため`delta=1`を使うが、これは実材料への推奨値ではない。必要field/parameterは次である。

```yaml
drag:
  model: epstein_linear
  revision: epstein_linear_v1
  gas_velocity_field: <vector field>
  gas_density_field: <scalar field>
  gas_temperature_field: <scalar field>
  gas_mean_free_path_field: <scalar field>
  gas_molecular_mass_kg: <positive scalar>
  delta: <positive scalar>
  applicability: error
```

mean free pathは適用域判定に使い、式の係数へ二重に入れない。このrevisionの保守的な運用範囲は
`lambda/a >= 10`、`|u-v|/c_bar <= 0.1`、`1 <= delta <= 13/9`である。これは普遍的なregime境界ではなく、
最初の製品model revisionが明示的に受理する範囲である。範囲外ではrunを失敗させ、
Stokes–Cunninghamへ黙って切り替えない。`count` policyはaccepted proposal単位とresult意味論が
確定するまで追加しない。

速度比上限を外れる参照caseに、このlinear revisionを無条件で適用しない。
有限相対速度は次の別revisionを使い、保存点の適用率だけで連続pathを認証しない。

根拠はEpsteinの原論文
[On the Resistance Experienced by Spheres in their Motion through Gases](https://authors.library.caltech.edu/records/3dxz2-4rt66)
である。同論文はMaxwell分布の分子による基本係数と、表面反射条件により係数が変わることを示す。

## `epstein_linear_effective_gas_sensitivity_v1`

単一の平均質量からspecies sumを回復できるとは限らない。等温・等数密度で質量比1:9、同じaccommodationなら、
number-weighted平均質量を使うpseudogasのdrag係数はspecies別Epstein和の`√5/2 ≈ 1.118`倍になる。
単一species極限では一致する。この製造例は平均化規約の差を示すもので、実混合気体の普遍補正係数ではない。

明示的なoptional reference/sensitivity revisionである。式、`delta`、field単位、粒子authorityは
`epstein_linear_v1`と同じだが、producerが混合気体を一つの有効Maxwellian/pseudogasへ畳み込み、その近似を
provenanceで認証した入力だけを受ける。species-resolved mixture modelや推奨defaultではない。

```yaml
drag:
  model: epstein_linear
  revision: epstein_linear_effective_gas_sensitivity_v1
  gas_velocity_field: <vector field>
  gas_density_field: <scalar field>
  gas_temperature_field: <scalar field>
  gas_mean_free_path_field: <scalar field>
  gas_molecular_mass_kg: <positive scalar>
  delta: <positive scalar>
  maximum_speed_ratio: <positive scalar at most 1>
  applicability: error
```

全stage・連続pathで`lambda/a>=10`と`|u-v|/c_bar<=maximum_speed_ratio`をfail-closedに要求する。
`maximum_speed_ratio`は必須の有限正値で`<=1`、速度clampではない。既存`epstein_linear_v1`は引き続き`0.1`固定で、
このkeyを受けない。formula、compiled evaluator、global bound、XY/RZ、両積分器とexisting enclosureを共有し、
species配列、mixture rule、自動blend、COMSOL branchを追加しない。

## `epstein_finite_speed_maxwell_mixed_equal_temperature_v1`

自由分子流中の孤立した非回転球について、局所shifted-Maxwellian気体とMaxwell混合表面則を使う。
`sigma_R=diffuse_reflection_fraction`は完全熱適応した等温拡散再放出の割合、残りは鏡面反射である。
`T_w=T_g`をrevisionへ固定し、一般のenergy accommodation係数とはしない。

\[
S=\frac{|\boldsymbol u_g-\boldsymbol v|}{\sqrt{2k_BT_g/m_g}},\qquad
\nu_0=\frac{4\pi}{3}\frac{a^2\rho_g\bar c}{m_p},\qquad
\boldsymbol a_d=\nu_0\left[G(S)+\sigma_R\frac{\pi}{8}\right](\boldsymbol u_g-\boldsymbol v),
\]

\[
G(S)=\frac{3}{16}(2+S^{-2})e^{-S^2}
+\frac{3\sqrt{\pi}}{32}(4S+4S^{-1}-S^{-3})\operatorname{erf}(S).
\]

`S<=0.1`では`G=1+S^2/5-S^4/70+S^6/630-S^8/5544`を使い、`S=0`を解析的に扱う。
低速極限はlinear revisionの`delta=1+sigma_R*pi/8`へ一致し、高速極限は`C_D->2`である。

```yaml
drag:
  model: epstein_finite_speed
  revision: epstein_finite_speed_maxwell_mixed_equal_temperature_v1
  gas_velocity_field: <vector field>
  gas_density_field: <scalar field>
  gas_temperature_field: <scalar field>
  gas_mean_free_path_field: <scalar field>
  gas_molecular_mass_kg: <positive scalar>
  diffuse_reflection_fraction: <number in [0, 1]>
  maximum_speed_ratio: <positive scalar>
  applicability: error
```

適用域は`lambda/a>=10`かつ全stage・連続pathで`S<=maximum_speed_ratio`である。速度比上限は
clampではない。RK4の安定性gateには速度Jacobianのradial係数`K=G+S G'`の上界、加速度と
exponential enclosureには`G`の上界を分けて使う。`T_w!=T_g`、CLL、気体混合種の総和、非球形、
回転、transition-regime blend、near-wall補正へ自動拡張しない。有限速度式は
[AERADE ARC-CP-0523](https://reports.aerade.cranfield.ac.uk/bitstream/handle/1826.2/536/arc-cp-0523.pdf?sequence=1)
と[Livi, Eq. 2.50](https://pure.tue.nl/ws/portalfiles/portal/292770290/20230419_Livi_hf.pdf)を相互確認した。

## `stokes_cunningham_allen_raabe_air_v1`

連続域Stokes dragへair用slip correctionを適用する明示modelである。

\[
\boldsymbol F_d=\frac{3\pi\mu d_p}{C_c}(\boldsymbol u_g-\boldsymbol v),
\]

\[
Kn_a=\lambda/a_p=2\lambda/d_p,\qquad
C_c=1+Kn_a\left[A_1+A_2\exp(-A_3/Kn_a)\right]
\]

とし、`A1=1.142`, `A2=0.558`, `A3=0.999`をrevision名へ固定する。ここで`Kn_a`は半径基準であり、
直径基準`Kn_d=lambda/d_p`の2倍である。曖昧な`Kn`名へ置換しない。係数とKn定義は
[NIST Journal of Research 111(4)](https://nvlpubs.nist.gov/nistpubs/jres/111/4/V111.N04.A01.pdf)
に記載されたAllen–Raabe相関に従う。この相関はair/calibration条件に基づくため、CF4/O2や低圧plasmaへ
そのまま一般化しない。さらに係数は平均自由行程の定義と不可分である。このrevisionを選ぶcase producerは、
`gas_mean_free_path_field`が同相関と整合するairの平均自由行程規約で作られたことを保証する。solverは別規約の
lambdaを推定補正しない。gas-specificな相関や平均自由行程規約を追加する場合は別revisionと独立根拠を要求する。

入力は次を必須とする。

```yaml
drag:
  model: stokes_cunningham
  revision: stokes_cunningham_allen_raabe_air_v1
  gas_velocity_field: <vector field>
  gas_density_field: <scalar field>
  gas_dynamic_viscosity_field: <scalar field>
  gas_mean_free_path_field: <scalar field>
  applicability: error
```

`gas_density_field`は粒子Reynolds数の適用域判定に使い、drag係数へ重複して入れない。productionで受理する
明示範囲は`0.03 <= Kn_a <= 7.2`、`Re_p=rho_g*d_p*|u_g-v|/mu <= 0.1`、`mu>0`である。
Kn範囲はこの相関revisionの受理範囲、`Re_p <= 0.1`は有限Re補正を持たないStokes式に対するsolver側の
保守的creeping-flow運用gateであり、Allen--Raabe係数のfit範囲と混同しない。連続包絡は外向き丸めするため、
名目上ちょうどKn/Re上限・下限にある入力は認証できないことがある。production caseは任意toleranceで救済せず、
物理的不確かさを含めて範囲の内側に余裕を持たせる。これは相関を任意気体へ一般化する主張ではない。このmodelの
`deterministic_physics_runtime_v1`がsample済みprimitiveからrate、連続適用域、global rate/acceleration boundを
一度だけ組み立てる。engineはfield sampling、event順序、accepted stateだけを所有し、drag式を複製しない。

独立oracleでは一定primitiveに対する線形緩和の解析解をXYで検証し、RZ away-axis caseが同じcanonical解へ
退化することを検証する。Epsteinとの自動blendは独立したversioned composite modelが必要であり、この二つを
暗黙に切り替えない。実行中に一粒子へ局在できる連続適用域逸脱は`model_applicability` failure、有限な
particle stateから導出値だけがoverflowする場合は`nonfinite_physics` failureとする。一方、入力時点の
共有field layout、非有限・非正のmodel係数、global bound不成立はprepare errorであり、別相関へ読み替えない。

## `electric_coulomb_v1`

\[
\boldsymbol a_E=Ze\boldsymbol E/m_p,
\qquad e=1.602176634\times10^{-19}\ {\rm C}
\]

とする。`electric_field` parameterが参照するvector fieldを実際のintegrator stage位置・時刻でsampleする。
電荷符号をfield側へ吸収せず、質量除算をintegratorや別forceへ重複させない。

```yaml
electric:
  model: coulomb
  revision: electric_coulomb_v1
  electric_field: <vector field>
```

## `gravity_buoyancy_standard_v1`

\[
\boldsymbol a_{gb}=\left(1-\rho_gV_{disp}/m_p\right)\boldsymbol g
\]

とする。`gas_density_field`とcase座標basisで表した`gravity_m_s2`を明示する。低密度気体で浮力が小さくても
式を別経路へせず、`Vdisp=0`または感度評価で省略を表す。

```yaml
gravity_buoyancy:
  model: standard
  revision: gravity_buoyancy_standard_v1
  gas_density_field: <scalar field>
  gravity_m_s2: [g0, g1]
```

`g0,g1`はXYでは`gx,gy`、RZでは`gr,gz`である。`gravity_buoyancy_standard_v1`はannular domainを含む
全RZ domainで`gr=0`を必須とし、`axis_accessible`に依存しない。非零値をclamp、平均、toleranceで隠さず
prepareで拒否する。一定非零`gr`は向きが方位とともに回転するradial body accelerationであり、3-D軸対称空間の
一様重力ではないため、このmodelへ読み替えない。必要なら別のmodel ID、式、適用域、oracleを追加する。
Cartesian XYの`gx`にはこの制約を課さない。

## Stage acceleration decomposition

同じcompiled model passは、合成加速度に加えて

```text
linear_drag_rate_s_inv
target_velocity_m_s
additive_acceleration_m_s2
```

を返す。加速度のauthorityは常に

\[
\boldsymbol a=-\lambda(\boldsymbol v-\boldsymbol u)+\boldsymbol a_{add}
\]

であり、Epstein/Stokes--Cunninghamの`lambda`やtarget gas velocityをintegrator側で再計算しない。
`rk4_fixed`は合成`acceleration_m_s2`を使う。`exponential_midpoint`と対応する`ou_langevin`は同じ評価から得た
`(lambda,u,a_add)`を、それぞれ指数更新とjoint OU更新へ渡す。dragなしでは`lambda=0`、targetは0、全非drag contributionが
`a_add`となるため、別のno-drag physics runtimeを作らない。
この分解はEpstein専用ではない。Stokes--Cunninghamの一定primitiveも同じ指数経路で閉形式の
linear-relaxation解へ一致する公開scenarioを持ち、model別integrator分岐を追加しない。

連続pathの証明用にrate上限、target速度と加算加速度の成分絶対上限も準備する。
point lawとboundは数学的役割が異なり、評価できたstage値からpath全体の安全性を推定しない。
運動と連続電荷を同じstageへ渡す更新、invariantの受理、hit prefixと残時間の処理は
[numerics](numerics.md#deterministic-exponential-midpoint)が所有する。

## `oml_stationary_maxwellian_debye_huckel_v1`

これは静止Maxwellianの電子と単一価正イオンを
吸収する球形・等電位粒子の連続平均電荷modelであり、one-way couplingだけを扱う。参照datasetにある
相対速度から有効イオン温度を作る近似は、このrevisionの式にも真値にも使わない。COMSOL等のproducerが
出力した電荷履歴は外部V&Vの比較対象であって、coreの式を決める入力ではない。

### 状態、入力、単位

権威状態は無次元の実数電荷数 \(Z\)、電荷は \(q=eZ\,[\mathrm C]\) である。

\[
\frac{dZ}{dt}=R_Z=\Gamma_i-\Gamma_e \quad [\mathrm{s}^{-1}]
\]

必要なcanonical量は次に限定する。

- 電子・正イオン数密度 \(n_e,n_i\,[\mathrm{m}^{-3}]\)
- 電子・正イオン温度 \(T_e,T_i\,[\mathrm K]\)
- 正イオン速度 \(\boldsymbol u_i\,[\mathrm{m\,s}^{-1}]\)
- 単一価正イオン質量 \(m_i\,[\mathrm{kg}]\)。空間fieldではなくrevisionの
  `positive_ion_mass_kg`定数parameterとする
- 粒子の`electrostatic_radius_m` \(a>0\)。revision 1ではcollection radiusと同一とする

電子質量、電気素量、真空誘電率、Boltzmann定数はsolver定数を使う。密度、温度、質量、半径は有限正値を
必須とし、adapter固有の名前や温度eV表現をcoreへ持ち込まない。

### 電位と収集率

\[
\lambda_D^{-2}=\frac{e^2}{\epsilon_0 k_B}
\left(\frac{n_e}{T_e}+\frac{n_i}{T_i}\right),\qquad
C_D=4\pi\epsilon_0a\left(1+\frac{a}{\lambda_D}\right),\qquad
\phi=\frac{Ze}{C_D}
\]

\[
A_s=\pi a^2n_s\sqrt{\frac{8k_BT_s}{\pi m_s}},\qquad
V_s=\frac{k_BT_s}{e}
\]

ここで \(A_s\) は無帯電球へのnumber collection rate \([\mathrm{s}^{-1}]\)、\(V_s\) はvoltである。
電子とイオンのrateを次のbranchで定義する。

\[
\Gamma_e=
\begin{cases}
A_e\exp(\phi/V_e), & \phi\le0\\
A_e(1+\phi/V_e), & \phi>0
\end{cases}
\]

\[
\Gamma_i=
\begin{cases}
A_i(1-\phi/V_i), & \phi\le0\\
A_i\exp(-\phi/V_i), & \phi>0
\end{cases}
\]

電流を出す場合だけ \(I_i=+e\Gamma_i\)、\(I_e=-e\Gamma_e\) とする。斥力側の指数引数は常に0以下で
あり、overflow clipや任意のrate floorは置かない。有限入力から非有限値が生じた場合は既存の
`nonfinite_physics`、適用域を外れた場合は`model_applicability`としてfail-closedにする。

### 明示する適用域

このrevisionは次をすべて満たす場合だけ有効とする。

\[
\frac{a}{\lambda_D}\le0.1,\qquad
M_i=\frac{|\boldsymbol u_i-\boldsymbol v|}
{\sqrt{8k_BT_i/(\pi m_i)}}\le0.1
\]

\(\boldsymbol u_i\) はrate式には入れず、静止Maxwellian近似のdrift gateだけに使う。球形、完全吸収、
等電位、衝突なし、非磁化、Maxwellian、単一価正イオン、放出なし、粒子が背景場を変えないことも
revisionの前提である。sheath内の強いdrift、collisional charging、磁化、二次電子・光電子・熱電子放出、
複数イオン種、離散捕獲は別revisionとする。適用域を緩めるための暗黙の相関切替は行わない。

XYではion velocityを`(x,y)/cartesian_xy`、RZでは`(r,z)/axisymmetric_rz`として読む。axisへ到達可能な
RZ domainではaxis nodeの \(u_{i,r}=0\) を既存vector regularityで認証する。さらに、Saffmanおよび
rarefied-vorticity liftが使う方位角渦度は軸上で厳密に0でなければならず、選択されたvorticity fieldを
required-field preparationで認証する。signed radial chartの変換は
\(|\boldsymbol u_i-\boldsymbol v|\) とscalar charge rateを変えない。

### 単調性、平衡、有限invariant

各branchで

\[
\frac{\partial R_Z}{\partial Z}<0
\]

であり、平衡は一意である。局所primitiveに対して \(A_i<A_e\) なら

\[
\phi_L=V_i(1-A_e/A_i)<0,\qquad
Z_{eq}\in[C_D\phi_L/e,0]
\]

\(A_i>A_e\) なら

\[
\phi_U=V_e(A_i/A_e-1)>0,\qquad
Z_{eq}\in[0,C_D\phi_U/e]
\]

で、\(A_i=A_e\) なら \(Z_{eq}=0\) である。平衡計算はこのbracket内の独立scalar oracleで検証するが、
production trajectoryを平衡値へ置換しない。

全field範囲の外向きextremaを上付き \(+/-\) で表す。次をrun-wideの保守的な電位bracketとする。

\[
\phi_-=-V_i^+\max(A_e^+/A_i^- -1,0),\qquad
\phi_+= V_e^+\max(A_i^+/A_e^- -1,0)
\]

\(C_D^+\) を許容範囲の最大容量とし、全初期電荷を含めて

\[
\mathcal I_Z=
\left[
\min\left(\min Z_0,C_D^+\phi_-/e\right),
\max\left(\max Z_0,C_D^+\phi_+/e\right)
\right]
\]

を有限invariantとする。左端では全許容primitiveに対して \(R_Z\ge0\)、右端では \(R_Z\le0\) である。
rateはZについて単調なので、\(\mathcal I_Z\) の両端をinterval評価して \(B_R\ge\sup|R_Z|\) を作る。
微分上界は

\[
L_Z\ge\sup\left|\frac{\partial R_Z}{\partial Z}\right|,
\qquad
L_Z=\frac{e}{C_D^-}
\left(\frac{A_i^+}{V_i^-}+\frac{A_e^+}{V_e^-}\right)
\]

を外向き丸めで用いる。electric accelerationは

\[
|a_{E,j}|\le
\frac{e}{m_p}\max(|Z_{min}|,|Z_{max}|)\,|E_j|_{max}
\]

で囲う。初期電荷だけから作った一定加速度certificateは動的電荷には使用しない。

### 数値更新への契約

`physics/charge.py`は式、平衡bracket、finite invariant、rate/derivative boundを所有する。
電荷は運動と同じ位置・速度・時刻のstageで評価し、初期ZだけをCoulombやion dragへ渡し続けない。
RK4のexplicit gateと、midpoint-frozen affine exponential chargeの更新・dense law・失敗の分類は
[numerics](numerics.md#coupled-charge-update)が所有する。clip、charge-only subcycle、平衡置換や
別modelへのfallbackでinvariant違反を隠さない。

基礎となるOML収集則は[Mott-Smith and Langmuir (1926)](https://doi.org/10.1103/PhysRev.28.727)、
有限Debye長で単純な容量関係を無条件に拡張できない点は
[Tang and Delzanno (2014)](https://doi.org/10.1063/1.4904404)を参照する。revision 1のsmall-grain/drift gateは
適用範囲を狭く保つsolver policyであり、これらの文献が閾値0.1を規定したという意味ではない。

## `oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1`

有限相対drift用revisionである。電子は静止Maxwellian、正イオンは速度
\(\boldsymbol u_i\)だけshiftした単一・単価Maxwellianとし、stationary revisionと同じDebye--Hückel容量、状態`Z`、
collection amplitude \(A_e,A_i\) とthermal voltage \(V_e,V_i\) を使う。正電位側、負イオン、複数イオン種、
放出、衝突・磁化補正はこのrevisionへ混ぜない。

設定で用いる相対drift比と、解析式のshift変数を次のように区別する。

\[
M_i=\frac{|\boldsymbol u_i-\boldsymbol v|}
{\sqrt{8k_BT_i/(\pi m_i)}},\qquad
s=\frac{|\boldsymbol u_i-\boldsymbol v|}{\sqrt{k_BT_i/m_i}}
=\sqrt{\frac{8}{\pi}}M_i .
\]

\(\phi\le0\)について

\[
P(s)=\frac12\exp(-s^2/2)
+\sqrt{\frac{\pi}{8}}\left(s+\frac1s\right)
\operatorname{erf}\left(\frac{s}{\sqrt2}\right),
\]

\[
H(s)=\sqrt{\frac{\pi}{2}}\,
\frac{\operatorname{erf}(s/\sqrt2)}{s},
\]

\[
\Gamma_i=A_i\left[P(s)-\frac{\phi}{V_i}H(s)\right],\qquad
\Gamma_e=A_e\exp(\phi/V_e),\qquad R_Z=\Gamma_i-\Gamma_e .
\]

zero-drift limitは`P(0)=H(0)=1`でstationary revisionの非正電位branchへ一致する。`s=0`を速度floorで
置換せず、小さい`s`では

\[
P(s)=1+s^2/6-s^4/120+O(s^6),\qquad
H(s)=1-s^2/6+s^4/40+O(s^6)
\]

を評価する。したがって停止近傍で`1/s`を直接使わず、物理rateを変更するregularizer、energy floor、指数clipを
導入しない。負電位branchでは

\[
\frac{\partial R_Z}{\partial Z}=-\frac{e}{C_D}
\left(\frac{A_iH(s)}{V_i}+\frac{\Gamma_e}{V_e}\right)<0
\]

なので局所平衡は一意である。

caseは有限正値`maximum_ion_drift_ratio`を必須とする。これは全accepted pathで
`M_i <= maximum_ion_drift_ratio`を認証するための適用上限であり、rateの調整値ではない。prepareは
初期`Z<=0`に加え、全field rangeと宣言drift envelopeで

\[
A_i P(s)\le A_e\quad (\phi=0)
\]

を認証し、有限invariantを`[Z_min,0]`に固定する。認証できなければ正電位branchへ切り替えずcaseを拒否する。
`a/lambda_D<=0.1`、球形・等電位・完全吸収、collisionless、unmagnetized、isolated-particle、背景非摂動も
stationary revisionと同じ前提である。

runtimeはactual stageの`M_i`と非正平衡を、engineはvelocity/path enclosureから全短縮区間の`M_i`を検査する。
現行path gateはion fieldとparticle velocityの成分絶対上界を加える安全側評価なので、強いco-flowでは適用内でも
拒否し得る。この偽拒否を解消する場合は符号付きvelocity intervalとRZ変換を一緒に改訂し、modelだけで上界を
緩めない。

RK4とcharged exponential midpointは同じ連続電荷stateとevent経路を使う。
共通の数値更新・dense law・invariantの扱いは[numerics](numerics.md#coupled-charge-update)を参照する。

shifted-Maxwellian ion collectionの形と適用上の注意は
[Alexandrov et al. (2008)](https://nano.uantwerpen.be/nanorefs/pdfs/OA_10.1088_1367-2630_10_9_093025.pdf)
および[Douglass et al. (2011)](https://doi.org/10.1063/1.3624552)を参照した。文献式の採用は、
このrevisionがsheath・collisional・multi-species caseを覆うことを意味しない。

## `aggregate_relative_drift_regularized_two_current_v1`

optionalなaggregate reference revisionであり、stationary/shifted OMLを置換しない。設定は次のstrict keyだけを持つ。

```yaml
charge:
  model: plasma_continuous
  revision: aggregate_relative_drift_regularized_two_current_v1
  electron_number_density_field: <positive scalar, 1/m^3>
  positive_ion_number_density_field: <positive scalar, 1/m^3>
  electron_thermal_voltage_field: <positive scalar, V>
  positive_ion_thermal_voltage_field: <positive scalar, V>
  positive_ion_velocity_field: <coordinate vector, m/s>
  effective_positive_ion_mass_field: <positive scalar, kg>
  screening_length_field: <positive scalar, m>
  maximum_relative_ion_speed_m_s: <positive finite scalar>
  applicability: error
```

半径`a=electrostatic_radius_m`、相対速度
\(\boldsymbol w=\boldsymbol u_i-\boldsymbol v\)、fieldが与えるthermal voltageを
\(V_e,V_i\) とする。screening fieldを粒子半径より小さくして別の容量式へ暗黙に切り替えず、revisionの式として

\[
\lambda_{eff}=\max(a,\lambda_{screen}),\qquad
\phi_1=\frac{e}{4\pi\epsilon_0a(1+a/\lambda_{eff})},\qquad
\phi=Z\phi_1
\]

\[
v_{eff}^2=|\boldsymbol w|^2+\frac{8eV_i}{\pi m_i}+u_\epsilon^2,\qquad
V_{i,eff}=\max\left(\frac{m_iv_{eff}^2}{2e},V_{floor}\right),
\]

\[
A_i=\pi a^2n_iv_{eff},\qquad
A_e=\pi a^2n_e\sqrt{\frac{8eV_e}{\pi m_e}}
\]

を使う。\(\phi\le0\)では

\[
C_i=1-\phi/V_{i,eff},\qquad
C_e=\exp(\operatorname{clip}(\phi/V_e,-50,50)),
\]

\(\phi>0\)では

\[
C_i=\exp(\operatorname{clip}(-\phi/V_{i,eff},-50,50)),\qquad
C_e=1+\phi/V_e,
\]

として、\(dZ/dt=A_iC_i-A_eC_e\)を運動と同じstageで評価する。固定model定数は
`u_epsilon=1 m/s`、`V_floor=0.01 V`、指数引数範囲`[-50, 50]`であり、設定parameterにはしない。
これらはrevisionの式であって、非有限入力を修復するfallbackではない。

`maximum_relative_ion_speed_m_s`は全accepted pathで
\(|\boldsymbol u_i-\boldsymbol v|\)を囲う有限applicability envelopeであり、速度clampやfitting係数ではない。
actual stageと連続pathのいずれかが超過すればfail-closedにする。正負電位branchを含む共通の有限charge invariant、
rate/derivative boundを準備し、RK4とexponential midpointそれぞれの[連続電荷数値契約](numerics.md#coupled-charge-update)を使う。

この式は電子と全正イオンを一つの有効密度・速度・質量・thermal voltageへ集約した平均二電流modelである。
stationary Maxwellianのattractive collectionに対して、trace-electron・zero-driftかつenergy floor外の
独立極限では、現行ion-current増分の比は`(π/4)/√(1+v_reg²/c̄_i²)`となる。
regularizationが無視できる極限でも`π/4`であり、species-resolved OMLの厳密解とは一致しない。
係数を自動補正せず、reference/sensitivity modelとしての適用scopeへこの差を含める。
species-resolved OML、負イオン収集、放出、離散電荷、collisional/magnetized sheathを表さず、普遍的な推奨modelとは
しない。Case P/AやCOMSOL名をcoreへ持ち込まず、uniform producerも有効質量を定数fieldとして書く。既存の
単一正イオンBarnes ion dragとは背景species authorityが異なるため併用を拒否し、次節のaggregate ion dragだけが
同じ背景fieldとstage電荷を共有できる。

独立Decimal式、全branch/floor/clamp、有限bound、stage結線、XY/RZ、両methodの収束、event/resumeを検査する。
指数clip境界ではsaturated側の導関数寄与を0とするが、有限Lipschitz boundは両側勾配を包含する。
producer定数規約との一致とcoreの独立式検証は別の主張であり、外部の式再生だけで連成軌道を認定しない。

## `aggregate_relative_drift_regularized_three_current_v1`

optional extensionである。前節の二電流revisionを変更せず、aggregateな単一価負イオン収集電流を一つだけ追加する。
設定は二電流のstrict keyに次の4 fieldを加えた正確なmappingを使う。

```yaml
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

負イオン密度は有限非負値で、0を許す。thermal voltageと有効質量は有限正値、速度はcase座標のvector fieldである。
`maximum_relative_ion_speed_m_s`は正負イオンそれぞれの`|u_s-v|`を囲う一つのrun-wide envelopeであり、clipではない。
一方でも超えれば同じapplicability failureにする。

二電流部分の`phi`、capacitance、正イオン・電子rateをそのまま使い、負イオンについて

\[
v_{eff,-}^2=|\boldsymbol u_--\boldsymbol v|^2+\frac{8eV_-}{\pi m_-}+u_\epsilon^2,
\quad
V_{eff,-}=\max\left(\frac{m_-v_{eff,-}^2}{2e},V_{floor}\right),
\]

\[
A_-=\pi a^2n_-v_{eff,-},\qquad
C_-=\begin{cases}
\exp(\operatorname{clip}(\phi/V_{eff,-},-50,50)),&\phi\le0,\\
1+\phi/V_{eff,-},&\phi>0
\end{cases}
\]

として

\[
\frac{dZ}{dt}=A_+C_+-A_eC_e-A_-C_-
\]

を同じstageで評価する。負イオン項のrate derivativeも既存のaffine exponential charge couplingへ渡し、全branchで
`dR_Z/dZ<0`、有限invariant、rate/Jacobian boundを要求する。`n_-=0`ならrate、derivative、global boundは前節の
二電流revisionへ厳密に一致する。ただし三電流revisionの入力field検証と`|u_--v|` applicabilityは密度0でも実行する。
従ってpublic APIの最終payloadまで二電流caseと一致するという主張は、両revisionの適用域条件が共に成立する場合に限る。

このmodelはaggregate singly-negative-ion collectionであり、species-resolved modelではない。複数負イオン種を一つの
`n_-`,`u_-`,`V_-`,`m_-`へ集約する規則はproducerが所有し、coreはspecies arrayや反応modelを持たない。screening lengthも
明示`screening_length_field`だけをauthorityとし、負イオンprimitiveからDebye長を再構成しない。元Case-P COMSOL `auxq`は
負イオンを診断量にだけ使う意図的な二電流式なので、既存Case-P same-form結果はこのrevisionのV&Vではない。

既存の共同stateとRK4/exponential couplingを使う。独立oracleは負イオン密度0での
rate/Jacobian/global-boundの二電流への退化、有限bound、公開state更新を含む。
外部companionの資格は元sourceと問題に限定し、元Case-P二電流、native FE、Brownian、
species-resolved物理、任意条件の資格へ拡張しない。実行証拠の判定は
[V&V methodology](../../vv_methodology.md)が所有する。

## Aggregate ion-drag sensitivity revisions

集約正イオン場を使う二つのmodel-form sensitivityであり、Barnes modelを変更しない。
両者を同時選択したり自動blendしたりせず、fixed chargeまたは前節のaggregate chargeとだけ組み合わせる。
共通入力は`n_i`、`T_iV`、`u_i`、行別`m_i`、capacitance用screening lengthであり、粒子mass、
`electrostatic_radius_m`、実際のstage電荷`Z`を既存stateから使う。aggregate chargeと併用する場合、これらのfield名を完全一致させる。

相対流revisionの設定は次である。

```yaml
ion_drag:
  model: screened_collection_orbital
  revision: relative_flow_screened_collection_orbital_aggregate_ion_v1
  positive_ion_number_density_field: positive_ion_number_density
  positive_ion_thermal_voltage_field: positive_ion_thermal_voltage
  positive_ion_velocity_field: positive_ion_velocity
  effective_positive_ion_mass_field: effective_positive_ion_mass
  screening_length_field: screening_length
  ion_neutral_mean_free_path_field: ion_neutral_mean_free_path
  maximum_relative_ion_speed_m_s: <positive finite scalar>
  applicability: error
```

\[
\boldsymbol w=\boldsymbol u_i-\boldsymbol v,\quad
s^2=|\boldsymbol w|^2+\frac{8eT_{iV}}{\pi m_i}+(1\,\mathrm{m/s})^2,
\]

\[
\phi_1=\frac{e}{4\pi\epsilon_0a[1+a/\max(a,\lambda)]},\quad
b_s=\max[a,\min(\lambda,\lambda_{in})],\quad
b_{90}=\frac{\sqrt{Z^2+10^{-20}}e^2}{4\pi\epsilon_0m_is^2},
\]

\[
b_{col}^2=\min\left[b_s^2,a^2\max\left(0,1-\frac{2eZ\phi_1}{m_is^2}\right)\right],
\]

\[
\ln\Lambda=\max\left[0,\frac12\ln\frac{b_s^2+b_{90}^2}{b_{col}^2+b_{90}^2}\right],\quad
\boldsymbol F_i=n_im_i\sqrt{s^2}(\pi b_{col}^2+4\pi b_{90}^2\ln\Lambda)\boldsymbol w .
\]

`maximum_relative_ion_speed_m_s`は速度clipでなく、actual stageと連続pathの`|w|`をfail-closedに認証する上限である。

electric-field-directed image sensitivityの設定は次である。

```yaml
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

\[
U=|\boldsymbol u_i|,\quad s^2=U^2+\frac{8eT_{iV}}{\pi m_i}+(1\,\mathrm{m/s})^2,\quad
\lambda_{img}=\sqrt{\frac{\epsilon_0T_{eV}}{en_i}},
\]

\[
A_{col}=\pi a^2\max(0,1-Z\phi_1/T_{iV}),\quad
b_{img}=\frac{e^2Z}{2\pi\epsilon_0m_is^2},\quad
A_{orb}=\pi b_{img}^2\ln[\max(1+10^{-12},\lambda_{img}/a)],
\]

\[
\boldsymbol F_i=n_im_i\sqrt{s^2}U(A_{col}+A_{orb})
\frac{\boldsymbol E}{\sqrt{|\boldsymbol E|^2+1\,(\mathrm{V/m})^2}} .
\]

従ってzero ion flowまたはzero electric fieldで力は0となる。Coulomb electric forceと併用する場合は同じ
electric-field名を要求する。保存Case P/Aはscalar ion-speed authorityが互いにもproduction式にも一致しないため、
coreはproducer別分岐を持たず、差は外部V&Vで明示する。両revisionはXY/RZ、RK4/explicit midpointの既存stage評価、
event経路を共有し、専用integratorを持たない。

## `quasistatic_spherical_gradient_e2_v1`

外部producerが形成した
`gradient_mean_e_squared_field`をXYまたはRZのcanonical vectorとして各stage位置でsampleし、球形粒子の準静的dipole力を
既存のadditive accelerationへ加える。

\[
\epsilon_m=\epsilon_0\epsilon_r,\qquad
\boldsymbol F_{DEP}=2\pi\epsilon_m a^3K_{CM}\nabla\langle |\boldsymbol E|^2\rangle,\qquad
\boldsymbol a_{DEP}=\boldsymbol F_{DEP}/m_p .
\]

`a=electrostatic_radius_m`、`m_p=mass_kg`をresident authorityとし、drag diameterやdisplaced volumeから再構成しない。
caseは有限正値`medium_relative_permittivity`、`[-0.5,1]`内の有限
`real_clausius_mossotti_factor`、有限正値`maximum_point_dipole_radius_m`を明示する。全粒子の
`electrostatic_radius_m`が認証上限以下でなければprepareで拒否する。独立に導出・直列化した同一境界値だけを安定に
扱うため、coreは`maximum_point_dipole_radius_m`の直後のfloat64値を上限の外向き表現として一つだけ受理する。
二つ目のsuccessorまたはそれ以上は拒否し、相対許容差、producer別例外、係数clamp、別model fallbackを使わない。

field単位は`V^2/m^3`である。DCでは`mean_E_squared=|E|^2`、周期場ではproducerが一つのfrequency/solutionについて
形成した物理時間平均、すなわちRMS二乗を意味する。DC/RF区分、solution/frequency、peak/RMS変換、時間平均、gradient
recovery、元field hash、point-dipole認証法と誤差基準はproducer provenanceへ保存する。coreはmetadataのunit/basisと
半径上限だけを実行契約として検査し、節点Eの微分やproducer固有provenanceの完全性判定を所有しない。

初回適用域は球形、線形・等方・一様媒質、準静的dipole、dilute one-way粒子、実数CM factorである。複素・周波数依存CM、
travelling-wave DEP、非球形、多極子、粒子間相互作用は別revisionとする。
2-D midpoint Brownianとは、線形EpsteinとDEP双方の適用域を満たす時だけ合成する。

component-wise global boundはfield extrema、`abs(K_CM)`、`epsilon_r`、`a^3/m`から外向きに丸めて作り、既存の
support/event enclosureへ加える。Cartesian XYでgradientが厳密定数かつ他の力も既存条件を満たす場合は
`quadratic_exact`を再利用し、それ以外はRK4またはexplicit midpointの同じstage passで評価する。scalar式、zero/sign、
`a^3/m` scaling、2-D回転共変性、乱択bound、pure/compiled parity、XY/RZ basis、および解析的調和振動子に対するRK4 3.5次以上・
exponential midpoint 1.8次以上の独立smooth-case oracleを使う。一般の場やevent後の次数保証ではない。

## `rarefied_vorticity_sensitivity_rz_v1`

axisymmetric RZ/no-swirl専用のfree-molecular lift感度revisionである。一般Saffman liftや
推奨defaultではない。meridional相対速度を`w=u_g-v`、方位vorticityを`omega_phi`とすると、

\[
\boldsymbol F_L=K(\omega_\phi\boldsymbol e_\phi)\times\boldsymbol w,
\qquad K=C_L\pi\rho_g\lambda_g a^2,
\qquad a=\frac{\texttt{drag\_diameter\_m}}{2},
\]

すなわち`F_r=K*omega_phi*w_z`、`F_z=-K*omega_phi*w_r`である。加速度は`mass_kg`で一度だけ除算する。
慣性は`mass_kg`、gas-surface寸法は`drag_diameter_m`をauthorityとし、electrostatic radiusやdisplaced volumeから
再構成しない。`lift_coefficient=C_L`は有限正値をcaseへ明示し、1をdefaultまたは普遍相関として補完しない。

必須fieldはRZ vector gas velocity `[m/s]`、正scalar gas density `[kg/m^3]`、正scalar gas mean free path `[m]`、
signed scalar azimuthal gas vorticity `[1/s]`である。方位vorticityの符号、元velocity solution、回復規則はproducerが
所有し、coreは速度fieldを微分しない。drag/thermophoresis/gravityと併用する場合は共有するneutral-gas field名を
catalogで完全一致させる。

適用域は球形、dilute one-way、RZ meridional、no-swirl、`lambda_g/a>=10`で、`applicability: error`だけを許す。
static lower boundと各actual stageでfail-closedに検査し、別lift、continuum blend、zero-force fallbackへ切り替えない。
Stokes--Cunninghamとは適用域が重ならないため併用を拒否する。現行2-D midpoint Brownianとは、RZ、線形Epstein、
上記free-molecular適用域を同時に満たす時だけ合成する。Cartesian XY、Cartesian 3-D、方位粒子速度は別revisionである。

liftは速度へ直交結合するため、固定external-acceleration配列へ入れない。prepared global extremaから粒子別coupling-rate
上界、二成分gas-velocity上界、static applicabilityを作り、runtimeはcomponent-wise particle-velocity上界を受ける
全非drag加速度のboundへ合成する。RK4とexponentialのpath enclosureは、開始速度とhalf predictorを含む速度boxで再評価する。

3-D cross-product射影oracle、zero vorticity/comoving、符号、`rho*lambda*a^2/m` scaling、相対流への直交性、
乱択global-bound包含、Kn拒否、pure/compiled parityを検査した。一様vorticity shearのRZ公開caseはRK4 3.5次以上、
exponential midpoint 1.8次以上で収束する限定oracleを持つ。指数法の全短縮stateも保守的enclosureへ含める。
resolved manifestはfield bindingと明示lift_coefficientを保存する。COMSOL式parity・軌道一致は別の外部評価である。

## `saffman_unbounded_creeping_shear_v1`

壁から十分離れた、球形・非回転・表面no-slip粒子の低Re連続体shearを対象とするoptional lift revisionである。
相対速度を`w=u_g-v`、流体vorticityを`omega=curl(u_g)`、`L=w cross omega`とすると、

\[
\boldsymbol F_L=6.46a^2\boldsymbol L
\sqrt{\frac{\mu\rho_g|\boldsymbol w|}{|\boldsymbol L|}},
\qquad a=\frac{\texttt{drag\_diameter\_m}}2
\]

で評価する。ゼロslipまたはゼロvorticityではepsilonで除算せず、連続極限の厳密なゼロを返す。平面問題では
`C=6.46*a^2*sqrt(mu*rho_g*abs(omega))/mass_kg`として、Cartesian XYは
`a=(C*w_y*sign(omega_z), -C*w_x*sign(omega_z))`、RZ no-swirlは
`omega_phi=d(u_r)/dz-d(u_z)/dr`に対して
`a=(-C*w_z*sign(omega_phi), C*w_r*sign(omega_phi))`である。

必須fieldはgas velocity `[m/s]`、density `[kg/m^3]`、dynamic viscosity `[Pa*s]`、mean free path `[m]`、
signed面外vorticity `[1/s]`である。vorticityはproducerが同じvelocity solutionと明示した回復規則から形成し、
coreは速度を微分しない。XYでは`omega_z`、RZでは上記`omega_phi`を供給する。Stokes--Cunninghamまたはdragなしとだけ
組み合わせ、Waldmann--Gallis、高Kn lift、Epstein dragとは適用域不一致として拒否する。

半径基準で`Kn_a=lambda/a`、`Re_s=rho*a*|w|/mu`、`Re_G=rho*a^2*|omega|/mu`とし、全stage/pathで

```text
Kn_a <= 0.1
Re_s <= 0.1
Re_G <= 0.1
omega == 0 or Re_s <= 0.1*sqrt(Re_G)
```

をfail-closedに要求する。最後の`0.1`は原式の`Re_s << sqrt(Re_G)`を実行可能にした保守的revision policyであり、
変更時はmodel revisionを上げる。区間vorticityがゼロを跨ぎ、非ゼロslip上界を持つ場合は階層を証明できないため、
局所細分化しても閉じなければ受理しない。near-wall lift、粒子回転、finite-Re補正へfallbackしない。

3-D cross-product oracle、XY/RZ符号、zero条件、半径scaling、hierarchyを含む適用域、pure/compiled parity、
線形回転解析解へのRK4収束、公開API fail-closedを検証した。保存済みCOMSOL datasetはこのmodelを選択していないため、
COMSOL実行値とのpoint-force/軌道parityは`NOT_TESTED`である。

## `barnes_collisionless_effective_speed_single_positive_ion_negative_debye_huckel_v1`

単一・単価正イオン、球形粒子、完全吸収collection、
局所shifted-Maxwellian ion、linear two-species Debye screening、Debye--Hückel表面電位、collisionless・
unmagnetized・dilute one-way backgroundだけを扱う。力の方向は電場でなく相対流
\(\boldsymbol w=\boldsymbol u_i-\boldsymbol v\) とし、neutral dragのrelaxationへ混ぜず
`explicit_acceleration`として加える。

\[
\bar c_i=\sqrt{\frac{8k_BT_i}{\pi m_i}},\qquad
v_s^2=|\boldsymbol w|^2+\bar c_i^2,
\]

\[
\lambda_D=\left[\frac{e^2}{\epsilon_0k_B}
\left(\frac{n_e}{T_e}+\frac{n_i}{T_i}\right)\right]^{-1/2},\qquad
\phi_p=\frac{Ze}{4\pi\epsilon_0a(1+a/\lambda_D)},
\]

\[
b_{90}=\frac{|Z|e^2}{4\pi\epsilon_0m_iv_s^2},\qquad
b_c^2=a^2\left(1-\frac{2e\phi_p}{m_iv_s^2}\right),
\]

\[
\ln\Lambda=\frac12\ln\frac{\lambda_D^2+b_{90}^2}{b_c^2+b_{90}^2},\qquad
\boldsymbol F_i=n_im_iv_s\left(\pi b_c^2+4\pi b_{90}^2\ln\Lambda\right)\boldsymbol w .
\]

必要fieldは`electron_number_density`、`positive_ion_number_density`、両温度、positive-ion velocity、
ion-neutral mean free pathであり、caseがpositive-ion massとrun-wide maximum drift ratioを明示する。
continuous chargeと併用する時は、密度・温度・ion velocity・ion massのauthorityを完全一致させる。同じstageで
一度sampleした配列と、そのstageで更新中の実電荷`Z`を両modelが共有する。別のplasma backgroundや遅延電荷copyは作らない。

適用域は`Z<=0`、`a/lambda_D<=0.1`、`b_90/lambda_D<=0.1`、`b_c/lambda_D<=0.1`、
`lambda_in/lambda_D>=10`、`ln Lambda>0`、宣言drift比以内である。0.1と10はsharpな文献境界でなく、この
revisionを弱結合・collisionless範囲へ狭く固定するsolver policyである。Coulomb log、相対energy、chargeを
floor/clampせず、image force、collisional correction、電場方向化、経験scale、別modelへのblend/fallbackを行わない。

prepareはfield extrema、宣言drift envelope、粒子の認証済みcharge区間から速度に依らない適用条件と
加速度絶対上界を保守的に構築する。runtimeはactual stageで全条件を再評価し、連続pathのdrift gateを既存enclosureへ
委ねる。continuous-charge invariantが広い場合、安全な実軌道でもprepareで偽拒否し得るため、これを局所clampで
緩めない。必要なら共通charge interval enclosureの独立revisionとして改良する。

ion dragは新しいstateや専用step制御を持たず、RK4とexplicit midpointの既存固定stepに入る。model固有の普遍的な
安定限界を捏造せず、利用caseは時間step収束を確認する。独立Rutherford impact-parameter積分、zero-flow/zero-charge、
global bound、compiled/reference parity、fixed/continuous chargeの同一stage結合、XY/RZ parityを検証した。
一様場公開oracleではRK4 3.5次以上、exponential midpoint 1.8次以上の収束を検査する。
このsmooth-case条件を任意の不連続場やevent後へ拡張しない。

collection＋orbital formは[Barnes et al. (1992)](https://doi.org/10.1103/PhysRevLett.68.313)を出発点とする。
このrevisionのeffective-speed collectionはshifted Maxwellianの速度積分そのものではない。
neutral sphere・weak driftの独立極限では、厳密Maxwellian係数`(4/3)c̄_i`に対して現行係数は`c̄_i`で、
力の比は`3/4`へ近づく。これは限定したcollectionのmodel-form差であり、orbital項や全域へ乗じる補正を意味しない。
Barnes型screeningの適用限界は[Khrapak et al. (2002)](https://doi.org/10.1103/PhysRevE.66.046414)、
collisional sheathが別modelを要することは[Ono et al. (2020)](https://doi.org/10.1103/PhysRevE.102.063212)を参照する。

## `waldmann_gallis_free_molecular_single_species_heat_flux_v1`

球形粒子、単一中性気体、dilute one-way、自由分子・
低相対driftだけを扱い、局所質量平均の中性気体座標に対する並進伝導熱流束を直接入力する。

\[
\bar c=\sqrt{\frac{8k_BT_{tr}}{\pi m_g}},\qquad
\boldsymbol F_{th}=\frac{32}{15}\frac{a^2}{\bar c}\boldsymbol q_{tr},\qquad
\boldsymbol a_{th}=\frac{\boldsymbol F_{th}}{m_p}.
\]

ここで`a=drag_diameter_m/2`、慣性は`mass_kg`である。`q_tr`は`W/m^2`のvector、gas temperatureは`K`、
gas velocityは`m/s`、mean free pathは`m`のfieldで、XY/RZとも既存のcanonical vector/scalar basis規則を使う。
軸到達可能RZ domainでは`q_tr`とgas velocityのaxis radial成分を既存field regularity検査が0へ限定する。
total heat flux、対流・放射・電子・イオン熱流束、固体熱伝導を`q_tr`として受理しない。

全stage/pathで`lambda/a>=10`かつ`|u_g-v|/c_bar<=0.1`を要求する。10と0.1はsharpな文献境界でなく、
このrevisionを狭い範囲へ固定する保守的policyである。global boundは`T`下限、`lambda`下限、`|q_tr|`成分上限を
使い、局所stageと連続速度enclosureの両方をfail-closedに検査する。適用外で力を0へしたり、Talbot/continuumへ
切り替えたりしない。

Epstein dragと併用する場合はgas velocity、temperature、mean free path、molecular massを完全一致させ、同じstage
sampleを共有する。Stokes--Cunningham revisionの認証Kn上限と本revisionの下限は重ならないため、同一caseでは拒否する。
混合気体、粒子内温度偏り、photophoresis、near-wall・accommodation補正、negative thermophoresisは別revisionである。
Fourier域のproducerは`q_tr=-kappa_tr grad(T)`を外部で形成し、translational conductivityと
`lambda |grad(T)|/T<=0.01`をproducer側で認証する。coreはgradient回復を所有しない。

独立3-D Gauss--Hermite積分で一次Chapman--Enskog分布の熱流束momentと運動量transfer momentを評価し、
`xi_1=32/(15*pi)`から具体的なproduction加速度まで比較した。zero heat flux、方向、`a^2/m`・`T^-1/2` scaling、
global bound、local/path gate、compiled parity、両積分器のaffine-field収束、XY/RZ parityを検証した。

自由分子式は[Waldmann (1959)](https://doi.org/10.1515/zna-1959-0701)、局所熱流束形の背景は
[Gallis, Rader, and Torczynski (2004)](https://doi.org/10.1080/02786820490490001)を参照する。

## `waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1`

optional reference/sensitivity revisionで、single-species revisionと同じWaldmann--Gallis heat-flux式を使う。producerが
one-effective-Maxwellian/pseudogasを認証し、その有効並進伝導熱流束`q_eff [W/m^2]`をcanonical vector fieldとして
供給する。`q_eff`はtotal/convective/radiative/electron/ion heat fluxではなく、coreはtemperature gradient、species配列、
mixture conductivity/accommodation ruleを回復しない。

```yaml
thermophoresis:
  model: waldmann_gallis
  revision: waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1
  gas_velocity_field: <vector field>
  gas_temperature_field: <scalar field>
  gas_translational_heat_flux_field: <q_eff vector field>
  gas_mean_free_path_field: <scalar field>
  gas_molecular_mass_kg: <positive scalar>
  maximum_speed_ratio: <positive scalar at most 1>
  applicability: error
```

全stage・連続pathで`lambda/a>=10`と`|u-v|/c_bar<=maximum_speed_ratio`を要求し、上限は必須の有限正値で
`<=1`である。single-species revisionの`0.1`は維持し、このkeyを受けない。XY/RZ、両積分器、formula/compiled evaluator、
global boundを既存ownerで共有する。設定できることはmixture truthを意味しない。
外部比較にはproducerの有効熱流束・集約規約と独立した適用域の証拠が必要である。

## `talbot_cross_regime_radius_knudsen_v1`

広いKn範囲を一つの明示式で扱うoptional thermophoresis revisionである。原著の半径基準に合わせ、
`a=d_drag/2`、`Kn_R=lambda/a=2*lambda/d_drag`、`Lambda=k_g/k_p`とし、

\[
\boldsymbol F_{th}=-\frac{12\pi a\mu^2}{\rho_g}
\frac{C_s(\Lambda+C_tKn_R)}
{(1+3C_mKn_R)(1+2\Lambda+2C_tKn_R)}
\frac{\nabla T_g}{T_g},
\qquad \boldsymbol a_{th}=\frac{\boldsymbol F_{th}}{m_p}
\]

を使う。[Talbot et al. 著者preprint Eq. (15)](https://escholarship.org/content/qt22f5r6cz/qt22f5r6cz.pdf)が
式・半径Kn・係数の根拠であり、原著の推奨値は`Cs=1.17`、`Cm=1.14`、`Ct=2.18`である。
caseは三係数を必ず明示し、manifestへ保存する。pure evaluator、boundにもdefaultやpresetを持たせない。
COMSOL記載の`1.17,1.146,2.2`を選ぶ場合はcoefficient variantとして扱い、原著の推奨値と同一とは呼ばない。
径Knで同じ式を表すには`Cm_d=2*Cm_R`、`Ct_d=2*Ct_R`が必要だが、productionは半径Knの一経路だけを持つ。
旧径Kn revisionは拒否し、自動変換や旧checkpointの再解釈はしない。

原著のmean free pathは粘性基準`lambda_old=2*mu/(rho*c_bar)`であり、原著係数を未換算で使うproducerは
この定義に対応するprimitiveを用意する。他のmodelと一つの`lambda_common`を共有する場合、選択した全domain・時刻で
`alpha=lambda_old/lambda_common`が一定である根拠を持つ時だけ、caseの明示係数を
`Cm_common=alpha*Cm_old`、`Ct_common=alpha*Ct_old`、`Cs_common=Cs_old`とすることで同じ力式を表せる。
これは`Cm*Kn`、`Ct*Kn`を保存する入力規約換算であり、fitでも別物理modelでもない。両規約、比の根拠と元・換算係数を
producer provenanceへ残し、point-forceを独立に確認する。例えば同じ`mu,rho,c_bar`から
`lambda_common=mu/(0.491*rho*c_bar)`を作る場合だけ`alpha=0.982`となる。Allen–Raabeの校正`lambda0`を含む
実際の温度・圧力lawでもこの比が一定か別に確認し、0.982を汎用defaultにしない。比が空間・時間で変わる場合、現在の
scalar係数では原式を全域で保存できず、明示した近似または非対応scopeとして扱う。coreはsample済みlambdaを変更せず、
Knを計算する。prefactorの`12*pi*a`は`6*pi*d_drag`と同値で、
mass、electrostatic半径、contact半径をdrag径から再構成しない。

必須fieldはtemperature `[K]`、temperature gradient `[K/m]`、density `[kg/m^3]`、dynamic viscosity `[Pa*s]`、
gas thermal conductivity `[W/(m*K)]`、mean free path `[m]`で、particle thermal conductivity
`[W/(m*K)]`はcase scalarである。原著の理論背景の脚注は多原子気体の`k_g`に並進熱伝導率を指定するため、原著再現には
`gas_thermal_conductivity_field`をその意味で用意する。単純な希薄気体のkinetic closure
`k_tr=(15/4)*mu*(k_B/m_molecule)`による再構成もproducer側の仮定・適用域・誤差を明示して行い、混合気体の汎用ruleには
しない。[COMSOL Thermophoretic Force](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.48.html)
等のnative式が全熱伝導率を使う場合、そのnative式の再現と原著Talbotの再現を区別する。coreは正の入力conductivityを
そのまま式へ使い、並進成分を推定しない。material property名だけで両者を等価と呼ばず、実際のfield定義とmodel-form差を
provenanceへ残す。gradientはproducerが同じ温度solutionと明示した回復規則から形成し、coreは温度を
微分しない。gradientは有限値、scalar primitiveと係数は有限正値に限定し、XY/RZ basisのgradientを
stage位置・時刻でsampleする。

本revisionは正の有限scalar primitiveに対してTalbot式を評価し、`Kn_R`の数値cutoffを設けない。これは広いKn範囲の
model-formを明示選択する契約であり、任意のKnで実験的に検証済みとは主張しない。near-wall、negative
thermophoresis、粒子内温度偏り、photophoresis、混合気体closureを認定せず、WaldmannとのblendやKnによる
自動切替はしない。categoryで一つだけを明示選択する。
原著の半径式によるscalar oracle、径式への同値係数変換、連続体/自由分子漸近、zero gradientと符号、point boundと
wide-Kn乱択bound、非default係数の設定伝播、XY/RZのpure/compiled parityを検証した。公開APIではTalbot＋Stokes＋
Saffmanのcompositionとmanifest係数に加え、affine温度場のTalbot単独運動を独立解析解と照合した。
保存済みCOMSOL datasetはTalbotを選択していないため、COMSOL実行値とのpoint-force/軌道parityは`NOT_TESTED`である。

## `inertial_langevin_fdt_epstein_linear_midpoint_2d_v2`

対応座標はCartesian XYとaxisymmetric RZ meridionalであり、RZはr/zへ投影した2自由度で方位自由度を持たない。
native/effective-gas線形Epstein、fixed/continuous charge、および選択した線形Epsteinと他のcatalog制約を同時に
満たすadditive forceを一つの`ou_langevin` proposalへ合成する。finite-speed Epstein、Stokes--Cunningham、
適用域が交わらないSaffman等はprepare時にfail-closedとなる。

fluctuation--dissipationの温度authorityは、組み合わせるEpstein mappingの
`gas_temperature_field`だけであり、noise mappingに第二の温度fieldを持たない。
粒子慣性はsourceの`mass_kg`で、各成分を

\[
d v=[-\gamma(v-u_g)+a]\,dt+\sqrt{2\gamma\theta}\,dW,
\qquad d x=v\,dt,
\qquad \theta=\frac{k_B T_g}{m_p}
\]

として扱う。additive accelerationを含む平均、連続電荷、joint位置・速度covariance、
conditional split、有限Hermite path、wall/axisでの再始動は
[numerics](numerics.md#inertial-brownian-integration)が所有する。

これは凍結係数の慣性Langevin modelであり、overdamped Brownian、finite-speed nonlinear frictionに
同じFDTを付けたmodel、等方3-D Brownianを表さない。XY/RZの有限数値path上でeventを扱うため、
連続OUのexact first-passage、zero miss probability、一般state-dependent SDEのstrong order/weak 2次を
このrevisionから主張しない。独立force diffusionは上記の
[explicit coefficient reference revisions](#explicit-coefficient-reference-revisions)を使い、FDT温度を倍率調整しない。

## Wall response laws

壁との交差・接触半径・corner候補集合は[numerics](numerics.md#material-contact-and-wall-events)が所有し、
ここでは局在されたhitへ一回適用する応答の意味を定める。静的geometry上のXY/RZ二成分stateを扱う。
数値的な停止を実機の接着認定へ読み替えない。

| law | 応答と制限 |
|---|---|
| `stick` | hit位置・電荷でinactiveな`stuck`へ移り、速度を0にする |
| `escape` | hitでinactiveな`escaped`へ移る。欠損から推定する結果ではない |
| `hold` | hit位置・hit速度・hit電荷を保持するinactiveな`held`。時間発展、再放出、paused状態ではない |
| `specular` | 法線反発係数1、接線反発係数1の完全鏡面。係数parameterを持たない |
| `restitution` | 必須の法線・接線反発係数を使う。係数の暗黙defaultはない |
| `maxwell_thermal` | 指定確率で壁frameの鏡面反射／完全熱適応拡散branchを選ぶ |
| `isotropic` | 静止壁で速さを保存し、内向き半円の角度を一様に選ぶ |
| `probabilistic_stick` | `probability`でstickを選び、非stick時は必須の`otherwise` lawを適用する |

外向きunit normalをn、接線をt=(-n_y,n_x)、相対入射速度をw=v-v_wallとする。
restitutionは `v_post=v_wall-e_n*(w·n)*n+e_t*(w·t)*t` である。
静的geometryのwall velocityはfacetに接線方向だけを許し、moving wallの法線運動を偽装しない。

Maxwell拡散branchは、`sigma²=k_B*T_wall/mass_kg`、独立なopen uniform Uと標準正規Nから

```text
c_n = sigma*sqrt(-2*log(U))
c_t = sigma*N
v_post = v_wall-c_n*n+c_t*t
```

を作る。法線は正のhalf-range flux分布、接線はGaussianである。
`diffuse_reflection_fraction`はbranch確率であり、一般のenergy accommodation係数ではない。
RZ則はmeridional二成分であり、完全3-D方位Maxwell分布ではない。
Maxwellの根拠と熱再放出との関係は[技術調査](../../technical_research.md#7-境界数値)を参照する。

isotropicは静止壁の入射速さsに対し

```text
phi = pi*(U_angle-0.5)
v_post = -s*cos(phi)*n+s*sin(phi)*t
```

を使う。cosine-weighted diffuseやthermal reemissionとは異なる。
stick判定とangle drawは独立streamであり、全incident facetへ内向きでなければ再抽選せずfailureを残す。
乱数identity、同時facetへの応答の一回性、hit後の残時間、terminal時刻の右連続性はnumericsが定める。

## 実装・未実装と検証範囲

上記はcatalogで選択可能な現行modelである。各runが採用したmodel/parameterと実行algorithmは
当該`run.json`をauthorityとし、この文書へruntime revision rosterを複製しない。
共有scalar body同士の一致は結線検査であり、独立の科学的正しさは解析極限、Decimal、速度積分、
Green-kernel、bound包含等のoracleで確認する。限定caseの成功を全regimeの物理validationへ拡張しない。

採用済みだがproduction未接続の方針は[技術調査](../../technical_research.md#adopted-unimplemented-gas-interaction)と
[implementation plan](../../implementation_plan.md)に保持する。RZ場＋Cartesian XYZと完全3-D mesh、
nonlinear gas noiseと粒子熱状態、全材料／浮遊VQ、表面charging／再飛散は現行modelへ黙って追加しない。
実装済みfinite-speed平均dragや独立collision oracleを、これらの未接続輸送と一緒に未実装へ戻さない。
