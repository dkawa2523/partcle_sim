# Physics model revisions

この文書はproductionで実装したversioned modelを扱う。model選択とfield参照はYAMLへ明示し、field名の自動探索、
適用域外での別modelへの自動切替、backend固有の係数を認めない。runtime contributionはdragを、stageで
凍結した正の`linear_relaxation(rate, target_velocity)`、その他の力を`explicit_acceleration`として返す。
現行physics catalog revisionは`inertial_langevin_rz_catalog_v17`、physics runtime revisionは
`signed_ion_compiled_physics_runtime_v19`である。runtime v16はP19-Lの局所range認証を追加した実装・性能snapshot、
v17はDEP認証上限の直列化丸めを一つのfloat64 successorまで外向きに扱い、v18は各charge stageへ
`charge_rate_derivative_s_inv=J`を追加し、v19はoptional aggregate three-currentを同じstage payloadへ統合した。
continuous modelはfiniteな`J<=0`、fixed chargeは`J=0`を返す。

## Particle authority

- 慣性はsourceの`mass_kg`だけが所有する。
- dragは`drag_diameter_m`、電気的な有限寸法は`electrostatic_radius_m`、浮力は
  `displaced_volume_m3`を使い、相互に再構成しない。
- Barnes ion dragの粒子半径も`electrostatic_radius_m`だけを使い、drag径から再構成しない。
- 初期電荷数`Z0`はtableの`charge_number`またはsurfaceの`particle.charge_number`だけが所有する。
- `charge: {model: fixed}`は`dZ/dt=0`だけを意味する。電荷Coulomb値は常に`q=Z*e`から求める。

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

とする。`delta`は表面反射・accommodationを表す無次元係数で、caseが必ず明示する。P02のC02/C03は
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

mean free pathは適用域判定に使い、式の係数へ二重に入れない。P06 revision 1の保守的な運用範囲は
`lambda/a >= 10`、`|u-v|/c_bar <= 0.1`、`1 <= delta <= 13/9`である。これは普遍的なregime境界ではなく、
最初の製品model revisionが明示的に受理する範囲である。範囲外ではrunを失敗させ、
Stokes–Cunninghamへ黙って切り替えない。`count` policyはaccepted proposal単位とresult意味論が
確定するまで追加しない。

`model_dataset`の参照caseには速度比上限を外れる保存行が存在するため、このrevisionだけでtarget plasma caseを
覆うとは主張しない。P15出口のM3-V評価では、保存行の適用率がcaseごとに0.542834～1となり、全軌道比較を
`NOT_APPLICABLE`とした。閾値は緩めず、有限相対速度を扱う別Epstein revisionを独立した式・適用域・oracleで
追加する。M3-Vの保存点判定はcontinuous-path certificationの代わりではない。

根拠はEpsteinの原論文
[On the Resistance Experienced by Spheres in their Motion through Gases](https://authors.library.caltech.edu/records/3dxz2-4rt66)
である。同論文はMaxwell分布の分子による基本係数と、表面反射条件により係数が変わることを示す。

## `epstein_linear_effective_gas_sensitivity_v1`

P18-Rのoptional reference/sensitivity revisionである。式、`delta`、field単位、粒子authorityは
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

連続域Stokes dragへair用slip correctionを適用する、P06-Sで実装済みの明示modelである。

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
物理的不確かさを含めて範囲の内側に余裕を持たせる。これは相関を任意気体へ一般化する主張ではない。P06-Sの
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

## P10 compiled physics runtime

P10完了時点の`deterministic_compiled_physics_runtime_v2`は、上記のversioned model式と加算順を変えず、sample済みprimitiveの
tile配列からdrag rate/target velocityとexplicit accelerationをNumbaで組み立てる。Numba 0.67、NumPy `<2.6`、
`fastmath=False`、`parallel=False`を固定する。`deterministic_physics_runtime_v1`はP06-Sで式・連続bound・
applicabilityを確定した履歴revisionとしてverification oracleに残るが、compiled failure時のproduction fallbackではない。
field support、連続applicability、非有限行の局在、wall/residual/output経路はengineが従来どおり所有するため、
physics catalogと各model revision、case/result schemaは変更しない。

## P11 linear-relaxation decomposition

P11で確立した`deterministic_compiled_physics_runtime_v3`は同じcompiled model passから、合成加速度に加えて

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
`rk4_fixed`は従来どおり合成`acceleration_m_s2`を使い、`exponential_midpoint`だけが同じ評価から得た
`(lambda,u,a_add)`を指数更新へ渡す。dragなしでは`lambda=0`、targetは0、全非drag contributionが
`a_add`となるため、別のno-drag physics runtimeを作らない。
この分解はEpstein専用ではない。Stokes--Cunninghamの一定primitiveも同じ指数経路で閉形式の
linear-relaxation解へ一致する公開scenarioを持ち、model別integrator分岐を追加しない。

連続path enclosure用のglobal boundも同じruntimeがrate上限、target速度の成分絶対上限、加算加速度の
成分絶対上限として返す。field sampling、model applicability、RZ基底変換、particle-local failureのownerは
変えない。P11 closeout時点のcharge modelはfixedだけであり、指数stageに非零`charge_rate_number_s`が現れた場合は
無視、陽更新、後付けsubcycleのいずれにもせず未対応として拒否する。

## P15 continuous charge revision 1（production実装完了）

P15で実装したmodel revisionは
`oml_stationary_maxwellian_debye_huckel_v1`とする。これは静止Maxwellianの電子と単一価正イオンを
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
RZ domainではaxis nodeの \(u_{i,r}=0\) を既存vector regularityで認証する。signed radial chartの変換は
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

### 数値解法と失敗意味

最初に受け入れたproduction sliceは既存RK4を使い、\((\boldsymbol x,\boldsymbol v,Z)\) を同じ4 stageで進める。
charge-only caseもstage evaluatorを通し、動的電荷ではlinear/quadratic exact specializationを無効化する。
step \(h\) に対して

\[
hL_Z\le0.5
\]

をRK4のexplicit stability/accuracy gateとする。全stage、midpoint、endpoint、event再積分、`state_at()`のZが
\(\mathcal I_Z\) 内にあることも必要である。違反時はclip、charge-only subcycle、平衡置換、暗黙fallbackを
行わず、未対応のcharge stiffnessまたはintegrator accuracy failureとして拒否する。

native`exponential_midpoint`への現行continuous charge接続は`charge_stable_exponential_midpoint_v3`である。
start predictorから得たmidpointで`G_mid=R_Z(Z_mid)`と`J_mid=dR_Z/dZ<=0`を凍結し、root始点へ
`A=G_mid+J_mid(Z_0-Z_mid)`と平行移動したaffine lawを

\[
Z(t_0+h)=Z_0+\frac{\operatorname{expm1}(J_{mid}h)}{J_{mid}}A
\]

で進める。`J_mid=0`では連続極限`Z_0+hA`を使う。exponential midpointは`hL_Z<=0.5`を
stability gateとして使わず、invariantとapplicabilityは引き続きfail-closedに検査する。運動は同じmidpoint chargeを
使うため一つのcoupled proposalのままである。clip、charge-only subcycle、第二engine、精度目的のhidden subdivision、
scalar implicit fallbackは追加しない。accuracyは別runの`h,h/2,h/4`で選ぶ。

`physics/charge.py`は式、平衡bracket、invariant、rate/derivative boundを所有し、catalogは選択とrequired field、
runtime/compiled passは同じ式のstage評価、integratorは共同状態更新、engineはdispatchとfailure集約を所有する。
P15 stationary slice時点はcatalog v5、runtime v4、RK4/exponential midpointと両enclosure v2である。state、boundary event、
frame/probe、checkpointには既にZがあるためschemaや第二のstate表現は追加しない。

基礎となるOML収集則は[Mott-Smith and Langmuir (1926)](https://doi.org/10.1103/PhysRev.28.727)、
有限Debye長で単純な容量関係を無条件に拡張できない点は
[Tang and Delzanno (2014)](https://doi.org/10.1063/1.4904404)を参照する。revision 1のsmall-grain/drift gateは
適用範囲を狭く保つsolver policyであり、これらの文献が閾値0.1を規定したという意味ではない。

## `oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1`

P15-Dで追加した有限相対drift用revisionである。電子は静止Maxwellian、正イオンは速度
\(\boldsymbol u_i\)だけshiftした単一・単価Maxwellianとし、P15と同じDebye--Hückel容量、状態`Z`、
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

RK4とnative explicit midpoint、材料wall、frame/probe、XY/RZ、checkpoint/resumeはP15の既存経路をそのまま使う。
追加したのはcatalog選択、純粋rate/bound、compiled row branchだけで、resident state、result schema、engineは増やさない。
P15-D完了時点のrevisionはcatalog v6、runtime v5、compiled tile v8、engine v28である。

shifted-Maxwellian ion collectionの形と適用上の注意は
[Alexandrov et al. (2008)](https://nano.uantwerpen.be/nanorefs/pdfs/OA_10.1088_1367-2630_10_9_093025.pdf)
および[Douglass et al. (2011)](https://doi.org/10.1063/1.3624552)を参照した。文献式の採用は、
このrevisionがsheath・collisional・multi-species caseを覆うことを意味しない。

## `aggregate_relative_drift_regularized_two_current_v1`

P18-Cのoptionalなaggregate reference revisionであり、P15/P15-Dを置換しない。設定は次のstrict keyだけを持つ。

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
rate/derivative bound、`h*L_Z <= 0.5`も既存continuous-charge数値契約のまま要求する。

この式は電子と全正イオンを一つの有効密度・速度・質量・thermal voltageへ集約した平均二電流modelである。
species-resolved OML、負イオン収集、放出、離散電荷、collisional/magnetized sheathを表さず、普遍的な推奨modelとは
しない。Case P/AやCOMSOL名をcoreへ持ち込まず、uniform producerも有効質量を定数fieldとして書く。既存の
単一正イオンBarnes ion dragとは背景species authorityが異なるため併用を拒否し、次節のaggregate ion dragだけが
同じ背景fieldとstage電荷を共有できる。

実装revisionはcatalog v11、runtime v10、compiled tile v12で、engine/state/schemaは変更していない。独立Decimal式、
全branch/floor/clamp、有限bound、compiled parity、XY/RZ、RK4・explicit midpointの時間収束、event/resumeを検証した。
指数clip境界ではsaturated側の導関数寄与を0とするが、有限Lipschitz boundは両側勾配を包含する。保存済み外部primitiveの
式再生はPASS、coreの標準`epsilon0`を含む厳密provider一致は定数規約差によりFAILであり、詳細は
[`../evidence/p18c/README.md`](../evidence/p18c/README.md)が所有する。この外部結果は式の層別診断だけで、連成電荷または
軌道精度の証明ではない。

## `aggregate_relative_drift_regularized_three_current_v1`

P21のoptional extensionである。前節の二電流revisionを変更せず、aggregateな単一価負イオン収集電流を一つだけ追加する。
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

実装は既存のcharge plan、compiled pass、resident `Z`、RK4/exponential couplingを共有し、第二engineやCase-P分岐を作らない。
production受入はcatalog v17、runtime `signed_ion_compiled_physics_runtime_v19`、compiled tile v18で完了した。独立oracle、
zero-densityでのrate/Jacobian/global-bound退化、finite bound、compiled/public-API回帰を含む標準verification/scenario suiteと、Ruff、Pyrefly、
import-linter、complexity、lock gateを通過した。engine、state、schemaは変更していない。外部Case-P派生companion入力監査は
canonical負イオンprimitive authority不足により`BLOCKED / NOT_EVALUATED`で閉じ、物理modelの`NOT_APPLICABLE`とは扱わない。
軌道は未実行で元Case-P二電流anchorも不変である。この任意物理の外部coverageはP21の出口から分離し、三電流外部同等性を
非認定のまま、P21/M3-C3は`CLOSED_ACCEPTED_WITH_LIMITATIONS`とする。詳細statusは
[`../../vv_methodology.md`](../../vv_methodology.md)と
[`../evidence/m3c3/caseP_three_current_companion_v1/`](../evidence/m3c3/caseP_three_current_companion_v1/README.md)が所有する。

## P18-I aggregate ion-drag sensitivity revisions

P18-IはP15-F Barnesを変更せず、集約正イオン場を使う二つのmodel-form sensitivityを`ion_drag` categoryへ追加する。
両者を同時選択したり自動blendしたりせず、fixed chargeまたは前節のP18-C aggregate chargeとだけ組み合わせる。
共通入力は`n_i`、`T_iV`、`u_i`、行別`m_i`、capacitance用screening lengthであり、粒子mass、
`electrostatic_radius_m`、実際のstage電荷`Z`を既存stateから使う。P18-Cと併用する場合、これらのfield名を完全一致させる。

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
event、checkpoint、result schemaを共有し、新しいresident stateや専用integratorを持たない。実装revisionはcatalog v12、
runtime v11、compiled tile v13で、engine v30と各schemaは不変である。

## `quasistatic_spherical_gradient_e2_v1`

P18-Dで追加した最初のDEP revisionである。外部producerが形成した
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
扱うため、runtime v17は`maximum_point_dipole_radius_m`の直後のfloat64値を上限の外向き表現として一つだけ受理する。
二つ目のsuccessorまたはそれ以上は拒否し、相対許容差、producer別例外、係数clamp、別model fallbackを使わない。

field単位は`V^2/m^3`である。DCでは`mean_E_squared=|E|^2`、周期場ではproducerが一つのfrequency/solutionについて
形成した物理時間平均、すなわちRMS二乗を意味する。DC/RF区分、solution/frequency、peak/RMS変換、時間平均、gradient
recovery、元field hash、point-dipole認証法と誤差基準はproducer provenanceへ保存する。coreはmetadataのunit/basisと
半径上限だけを実行契約として検査し、節点Eの微分やproducer固有provenanceの完全性判定を所有しない。

初回適用域は球形、線形・等方・一様媒質、準静的dipole、dilute one-way粒子、実数CM factorである。複素・周波数依存CM、
travelling-wave DEP、非球形、多極子、粒子間相互作用は別revisionとする。B02のBrownian subsetは追加決定論力を受理しないため、
DEPとの併用をprepareで拒否する。

component-wise global boundはfield extrema、`abs(K_CM)`、`epsilon_r`、`a^3/m`から外向きに丸めて作り、既存の
support/event enclosureへ加える。Cartesian XYでgradientが厳密定数かつ他の力も既存条件を満たす場合は
`quadratic_exact`を再利用し、それ以外はRK4またはexplicit midpointの同じstage passで評価する。scalar式、zero/sign、
`a^3/m` scaling、2-D回転共変性、乱択bound、pure/compiled parity、XY/RZ basis、および解析的調和振動子に対するRK4 3.5次以上・
explicit midpoint 1.8次以上を検証した。実装revisionはcatalog v13、runtime v12、compiled tile v14で、engine v30、
resident state、memory plan、case/result/checkpoint schemaは不変である。

## `rarefied_vorticity_sensitivity_rz_v1`

P18-Lで追加した、axisymmetric RZ/no-swirl専用のfree-molecular lift感度revisionである。一般Saffman liftや
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
Stokes--Cunninghamとは適用域が重ならず、B02 Brownianは追加決定論力を受理しないため、いずれもplan解決時に併用を
拒否する。Cartesian XY、Cartesian 3-D、方位粒子速度は別revisionである。

liftは速度へ直交結合するため、固定external-acceleration配列へ入れない。prepared global extremaから粒子別coupling-rate
上界、二成分gas-velocity上界、static applicabilityを作り、runtimeはcomponent-wise particle-velocity上界を受ける
単一の全非drag加速度bound callbackとして評価する。RK4は既存callbackを使い、exponential enclosure v3は開始速度と
half predictorの速度boxで再評価する。exponential midpoint v2、engine v30、proposal v7は維持する。

3-D cross-product射影oracle、zero vorticity/comoving、符号、`rho*lambda*a^2/m` scaling、相対流への直交性、
乱択global-bound包含、Kn拒否、pure/compiled parityを検査した。一様vorticity shearのRZ公開caseはRK4 3.5次以上、
explicit midpoint 1.8次以上で収束し、指数法の全短縮stateもenclosure v3内に入る。実装revisionはcatalog v14、
runtime v13、compiled tile v15、exponential enclosure v3で、integrator v2、engine v30、proposal v7、resident state、
memory plan、case/result/checkpoint schemaは不変である。resolved result manifestはfield bindingと明示`lift_coefficient`を
保存する。COMSOL pointwise式parityと軌道一致はM3-C1まで`NOT_TESTED`である。

## `barnes_collisionless_effective_speed_single_positive_ion_negative_debye_huckel_v1`

P15-Fで追加した最初のion-drag revisionである。単一・単価正イオン、球形粒子、完全吸収collection、
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
一様場公開caseではRK4 3.5次以上、explicit midpoint 1.8次以上の収束を確認した。P15-F完了時点のrevisionはcatalog v8、
runtime v7、compiled tile v10、engine v28であり、proposal、event、memory plan、case/result/checkpoint schemaは変更していない。

collection＋orbital formは[Barnes et al. (1992)](https://doi.org/10.1103/PhysRevLett.68.313)を出発点とする。
Barnes型screeningの適用限界は[Khrapak et al. (2002)](https://doi.org/10.1103/PhysRevE.66.046414)、
collisional sheathが別modelを要することは[Ono et al. (2020)](https://doi.org/10.1103/PhysRevE.102.063212)を参照する。

## `waldmann_gallis_free_molecular_single_species_heat_flux_v1`

P16で追加した最初のthermophoresis revisionである。球形粒子、単一中性気体、dilute one-way、自由分子・
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
P16完了時点のrevisionはcatalog v9、runtime v8、compiled tile v11、engine v28で、integrator、event、memory plan、
case/result/checkpoint schemaは変更していない。

自由分子式は[Waldmann (1959)](https://doi.org/10.1515/zna-1959-0701)、局所熱流束形の背景は
[Gallis, Rader, and Torczynski (2004)](https://doi.org/10.1080/02786820490490001)を参照する。

## `waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1`

P18-Rのoptional reference/sensitivity revisionで、P16と同じWaldmann--Gallis heat-flux式を使う。producerが
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
`<=1`である。P16 single-species revisionの`0.1`は維持し、このkeyを受けない。XY/RZ、両積分器、formula/compiled evaluator、
global boundを既存ownerで共有する。catalog v15、runtime v14、compiled tile v16へ更新したが、engine v30、proposal v7、
integrator v2、resident state、memory plan、case/result/checkpoint schemaは不変である。保存auditではnative linear Epstein
replayだけが約`1.1e-15`でPASSし、既存P15-E/P16 applicabilityは12/12 `NOT_APPLICABLE`、PPRに`q_eff`が無いため
本revisionのpointwise replayは`NOT_TESTED`である。COMSOL studyは再実行しておらず、設定可能性はmixture truthを意味しない。

## Brownian numerical foundation（B01履歴）

B01は線形Langevin系の数値primitiveだけを実装した。`stochastic.py`が局所固定係数のjoint `(x,v)` OU更新と
conditional half-split、`rng.py`が物理interval-tree Philox normalを所有した。B01単独ではphysics catalogへ
`noise` modelを登録せず、既存dragのfluctuation--dissipation対応、field sampling、event、output、checkpointへも
接続していなかった。production接続は次節のB02が所有する。

B01の平均更新はdrag-onlyであり、任意の`F_other/m`を含まない。この制約はB02でも維持し、他の力を黙って
落としたり、別のdragへ自動切替したりしない。

B01はfinite-speed非線形drag、RZ meridional、overdamped近似、材料wall first-passageを対応済みとは扱わない。

## `inertial_langevin_fdt_epstein_linear_frozen_start_v1`（B02）

B02の`physics.noise`はmodel `inertial_langevin_fdt`、revision
`inertial_langevin_fdt_epstein_linear_frozen_start_v1`である。Cartesian XY、fixed charge、
`epstein_linear_v1`だけを受理し、electric、gravity/buoyancy、ion drag、thermophoresis、DEP、liftなどの追加力、
continuous charge、finite-speed Epstein、Stokes--Cunningham、RZをprepare時に拒否する。材料lawもterminalな
`stick`と`escape`だけで、specular、restitution、probabilistic stickを含む反射経路へ接続しない。

各macro-rootの開始位置・速度・時刻でEpstein rate `gamma`と平衡速度`u_g`を評価し、そのroot内で固定する。
fluctuation--dissipationの温度authorityは同じEpstein mappingが参照する`gas_temperature_field`だけであり、
noise mappingに第二の温度fieldを持たない。粒子慣性は従来どおりsourceの`mass_kg`で、各成分を

\[
d v=-\gamma(v-u_g)\,dt+\sqrt{2\gamma\theta}\,dW,
\qquad d x=v\,dt,
\qquad \theta=\frac{k_B T_g}{m_p}
\]

として進める。`interval_tree_depth`は整数`0..10`、prepare時に`gamma*dt<=1e6`、全macro-rootで有限な
`0<gamma*h<=1e6`を要求する。OU endpointは
凍結係数に対してjointに厳密だが、各leafのevent/replay pathはそのendpoint位置・速度から構成するcubic Hermite
numerical pathである。この有限depth path上のfirst hitを認証するのであり、連続OU trajectoryのexact first-passage、
overdamped Brownian limit、miss probability 0を主張しない。

resolved modelにはnoise model/revisionを記録し、manifestにはBrownian RNG、joint OU、conditional splitの各revision、
tree depth、root/split draw stream、`macro_root_frozen_start_v1`、`path_kind=cubic_hermite`を残す。checkpoint resume
identityはRNG/OU/split revision、coefficient policy、depth、resolved modelを含み、mutable RNG cursorを保存しない。
B02の上記model/revisionとpayloadは現行engine v36でもbitwise不変である。現行全体revisionはengine v36、proposal v10、
catalog v17、runtime v19、runtime layout v6、memory plan v13である。
root covariance、conditional split、mean updateがfloat64で表現不能なrowは`nonfinite_physics`となり、同一batchの
正常rowは継続する。`gamma*h`上限をこの実表現可能性検査の代用にはしない。
COMSOLまたは`model_dataset`との比較は物理式のauthorityでなく、core外のV&Vだけが所有する。

## `inertial_langevin_fdt_epstein_linear_rz_meridional_projected_v1`（B03）

B03はaxisymmetric RZ meridionalのr/zへ投影した2自由度revisionである。native/effective-gas線形Epstein、fixed/continuous
charge、既存additive forceを一つの`ou_langevin` proposalへ合成する。root始点からのnoise-free predictorでmidpointを作り、
そこで`gamma,u,T,a,G=dZ/dt,J=dG/dZ`を一度評価して`macro_root_frozen_midpoint_v1`として凍結する。
`J<=0`を要求し、`u_eff=u+a/gamma`のjoint exact OUと`macro_root_affine_exponential_v2`のdense chargeを
同じproposalが所有する。root基準のrateは`G_mid+J(Z_root-Z_mid)`であり、各leaf charge intervalをprepared invariantへ
照合して逸脱・証明不能をfail-closedにする。

conditional OU treeはrootの凍結係数とendpointを保つ。axis hitではaccepted prefixをcommitしてradial stateをfoldし、残時間を
fresh `root_stochastic_interval`として再評価・独立drawで再開する。元root remainderはfold/restrictしない。terminal lawは
`stick`/`escape`/`hold`だけである。等方3-D Brownianでも一般state-dependent SDEのstrong order/weak 2次でもない。
COMSOL再実行なしに解析・manufactured・identity gateを閉じ、現行revisionはengine v36 / proposal v10 / catalog v17 /
event v16 / runtime v19 / compiled tile v18 / memory plan v13である。正式characterizationは24/24実行を全粒子active・
failure 0で完了し、静的なB03 path-array上限`648 B/row`が`2048 B/row`以内であることも確認した。計時とRSSは
[`evidence/b03/`](../evidence/b03/README.md)の初回closeout（engine v34 / proposal v9 / runtime v17 / tile v16）に
属するmachine-local・non-gating観測であり、物理妥当性の証拠ではない。

## Stage gate

P06 revision 1はXY、fixed charge、上記Epstein範囲、Coulomb電気力、重力・浮力、
`applicability=error`の物理式と共通RK4 proposalを実装した。ただしpost-reviewで、RK stage点とendpointの
判定だけでは曲線全体のfield supportを証明できず、frame scheduleがrunの成否を変え得ると判明した。
前版engine v3はEpstein dragと非一様fieldの`rk4_reintegrated`を一律拒否し、revision 3aで証明済みの
boundaryless regular subsetだけを公開runへ移した。

revision 2は物理式を増やさず、dragなし・厳密一様な必要fieldから証明した一定加速度だけを
`quadratic_exact`で実行する。topology-completeな材料境界では放物線first hit、boundarylessでは
fully-supported regular field boxと各proposalの解析的な座標極値判定が連続supportを保証する。C04/C05はこの
公開production経路で実行する。P02のmanufactured値を実チャンバー全体の妥当性範囲として引用しない。

revision 3aは物理式を増やさず、boundaryless Cartesian XY、fixed charge、全cell supportedな
`RegularLayout`だけを対象にする。canonical field extremaとmodel係数から、Epstein/electric/gravityを合成した
RK4内部stageの速度・加速度をglobalにboundし、全ての短縮RK4評価を含む外向き位置enclosureがregular support boxへ
包含されることを要求する。Epsteinは`lambda/a`下限と`|u-v|/c_bar`上限も同じfield・速度boundから全区間で
証明する。supportまたはapplicabilityを証明できないcaseはhidden subdivisionせずfail-closedにする。

revision 3aは着手前hardening後の`coupled_rk4_engine_v5` / `coupled_rk4_proposal_v3`として完了し、C02/C03の公開API解析時系列、
frameなし・疎・密のoutput schedule不変性、step途中release、安全な非一様regular field、
support/applicabilityのfail-closed反例を検証した。
revision 3bは`coupled_rk4_engine_v6`として、各local始点からboundを再構築するsequential accepted RK4 pieces、
一般材料壁面の離散tube、event-before-validityを追加した。同一macro proposalのparameter区間は流用しない。
物理model自体とproposal v3は変えず、Cartesian XY・fully-supported `RegularLayout`・fixed chargeの同じ
Epstein/electric/gravity subsetをtopology-completeな材料boundaryと連成する。P06-Uは物理式を変えず、particle
volume meshと完全一致するfully-supported P1/Q1を同じ材料domain subsetへ追加した。P06-RZは物理式を変えず、
上記RZ metadata/axis regularityとsigned-stage変換を完成し、regular fieldのboundaryless motionおよび
topology-complete材料domainの一般RK4を解禁した。解析Epstein axis crossing、away-axis XY退化一致、軸上不変状態、
frame schedule identityで検証する。P06-Sはdrag責務を`deterministic_physics_runtime_v1`へ集約し、
`stokes_cunningham_allen_raabe_air_v1`をXY/RZの同じ一般RK4経路へ追加した。解析緩和解に対する4次収束、
frame schedule不変性、away-axis XY/RZ一致、Kn適用域と混在粒子の連続包絡overflow局在を公開scenarioで確認した。
Re/Kn境界の純粋式判定はunit verificationで固定している。
P10はこれらの式と意味論を`deterministic_compiled_physics_runtime_v2`へ移し、field/physics/RK4のcompiled array
passを同じproduction engineへ接続した。P11はmodel式を変えず、同じpassへ線形緩和の分解値を追加した
`deterministic_compiled_physics_runtime_v3`を`exponential_midpoint_v1`へ接続した。scalar runtimeは
verification oracleだけであり、指数法も別physics runtimeやCOMSOL専用係数を持たない。
boundaryless P1/Q1、`count` policy、暗黙のdrag相関切替は引き続き未解禁である。continuous chargeはP15の
stationary revisionとP15-Dの単一正イオン・非正電位shifted-Maxwellian revisionの明示範囲だけを解禁した。
