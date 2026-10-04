# 技術調査：半導体チャンバー内の粒子追跡

本書は理論と数値方式の補足資料である。製品範囲、実装順、coreと外部toolの境界は
[product_specification.md](product_specification.md) を正とする。COMSOLは利用可能なモデルと
外部V&V条件を確認する参照であり、製品標準の数値方式を決める権威ではない。

## 1. 適用対象

対象は、外部の流体・熱・電磁場・プラズマ計算を読み、一方向連成で固体微粒子の軌道と
内部状態を追うLagrangian point-particle modelである。粒子が背景場を変えるtwo-way coupling、
粒子間衝突・凝集・破砕、表面粗さを直接解く接触力学は初期範囲に含めない。

`model_dataset`は2D軸対称r-z、定常場、10/30/100 nm、希薄気体、動的電荷、複数外力、
Brownianを含む。この粒径と圧力域では、dragの速度緩和、電荷の緩和、チャンバーを横断する
運動が大きく異なる時間scaleを持つ。従って「一般ODEを小さな固定stepで解く」だけでなく、
stiff drag、確率過程、境界event、場補間を一体として設計する必要がある。

COMSOL固有の設定は公式の
[Particle Tracing Module User's Guide](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/ParticleTracingModuleUsersGuide.pdf)
と対象modelの保存設定を根拠にし、未公開の内部処理を推測して互換と呼ばない。

---

## 2. 状態と支配方程式

決定論的な基本系は

```text
dx/dt = v
m_p dv/dt = F_drag + F_electric + F_gravity/buoyancy
              + F_thermophoresis + F_DEP + F_lift + F_ion + ...
dy/dt = G(x, v, y, fields, t)
```

である。`y`は電荷数Zなどの内部状態である。Brownianを含む場合、右辺へ時刻ごとに任意乱数を
足すODEではなく、Langevin型の確率微分方程式として扱う。

粒子ごとの権威ある入力は少なくとも次である。

- `mass_kg`
- `drag_diameter_m`
- OML/DEP等に使う`electrostatic_radius_m`
- 浮力に使う`displaced_volume_m3`
- 初期位置・速度・release time
- 初期電荷または電荷分布
- material ID、`model_weight`、付着model用の表面/粒子parameter

質量を密度・径からruntimeで再生成すると、非球形や有効drag径を表現できず、入力との不整合を
隠す。球形builderだけが密度と物理径からcanonical属性を一度生成する。runtimeでは再構成しない。
物理式は力で記述しても、kernel interfaceは`linear relaxation rate/target velocity`または
加速度へ統一し、mass除算をmodel間へ分散させない。

### 2.1 座標系

Cartesian 2D/3Dでは上式を直交basisで解く。axisymmetric RZ no-swirlでは運動自由度は
`r,z,v_r,v_z`であり、3D場の方位成分を軌道へ入れない。`model_dataset`ではBrownianの
`F_phi`が診断列に存在し合力magnitudeにも含まれるが、実solverの運動DOFはr/zだけである。
従ってRZ軌道比較にはr/z成分を使い、3D magnitudeを加速度の権威にしない。

ただし、等方Brownian運動は物理的には方位速度も生成する。したがって`r,z,v_r,v_z`だけの
meridional kernelは、決定論的か方位力ゼロの高速近似であり、3D Brownian軌道と等価ではない。
Brownianや方位運動を扱う標準形は、2D RZ場を参照しつつ粒子をCartesian 3Dで積分する
`axisymmetric_field_cartesian3d`とする。

RZ swirlでは`v_theta`を加え、円柱座標の幾何項を含む別kernelが必要である。これは理論catalog上の
将来候補で、初期製品modeには含めない。`r=0`は物理壁ではなく座標の継ぎ目であり、軸横断の
basis変換をwall reflectionと混同しない。

2D軸対称surfaceから3Dの粒子fluxを復元するとき、線素`ds`は回転面積`2πr ds`を表す。
一様surface releaseは2D edgeを等確率に選ぶのではなく、この面積で重み付けし、方位角を
一様に生成する。現行v0.1 schemaは単一facet上の`edge_fraction`と、XYの`line_length`またはRZの
`meridional_length | revolved_area`を明示した`uniform`を持つ。P14-Uで実測した分布品質はXY
`line_length`だけである。RZ回転面の妥当性を同じ証拠から主張しない。現時点で
`realized_surface_table`を追加せず、これらで表せない代表用途が確認された時だけ再検討する。

---

## 3. 力学module

### 3.1 希薄気体drag

dragは粒子と気体の相対速度`u_g-v_p`に作用する。連続域ではStokes/Cunningham系、高Knでは
Epstein/free-molecular系を用いる。必要な局所量は気体密度、温度、粘度、平均自由行程、組成、
accommodation parameterである。

```text
Kn_d = lambda_g / d_p
Kn_a = lambda_g / a_p = 2 Kn_d
Re_p = rho_g d_p |u_g-v_p| / mu
```

`Kn_d`は直径基準、`Kn_a`は半径基準であり、各model revisionはどちらを使うか明記する。Epstein
`epstein_linear_v1`の適用域は`Kn_a=lambda_g/a_p`、Allen–Raabe Stokes–Cunningham候補の相関は
`Kn_a=2 lambda_g/d_p`を使う。曖昧な`Kn`名だけを式やmanifestへ保存しない。KnとReは背景CSVの固定列ではなく、
粒子径・速度と局所primitive fieldから毎回計算する。
`model_dataset`の外部field bundleでは粒径派生列が全て10 nm相当であるため、この分離は正しさに
直結する。

線形化できる区間は

```text
dv/dt = (u_g-v)/tau_p + a_other
```

となる。`tau_p << dt`では明示RKの安定stepがチャンバー移動時間より極端に小さくなる。native
modeではdrag部分を指数的に厳密更新し、残りの場変化・非線形力だけを数値積分する。

COMSOLの希薄気体drag設定とKnの定義は公式
[Drag Force](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.37.html)
を基準にcaseごとに固定する。model適用域を跨ぐ自動blendは、独立したversioned modelとして
検証する。

P18-Rの`epstein_linear_effective_gas_sensitivity_v1`は既存線形式と同じformula ownerを使うが、producerが混合気体を
一つの有効Maxwellian/pseudogasへ畳み込んだ入力だけを受けるreference/sensitivity revisionである。caseは
`0 < maximum_speed_ratio <= 1`を明示し、`lambda/a>=10`とともにactual stageと既存runtime enclosureによる連続pathで
fail-closedに認証する。これは速度をclipするparameterでなく、species-resolved mixture truthでもない。
`epstein_linear_v1`の保守的上限`0.1`は変更しない。

#### finite-speed free-molecular sphere

`epstein_finite_speed_maxwell_mixed_equal_temperature_v1`は、局所Maxwell分布、孤立した非回転球、
自由分子流、粒子表面温度と気体温度が等しい条件に限定する。Maxwell表面則の
`sigma_R in [0,1]`を完全熱適応した拡散再放出の割合、残りを鏡面反射と定義する。一般のenergy
accommodation係数へ読み替えない。元の低速結果は
[Epstein (1924)](https://authors.library.caltech.edu/records/3dxz2-4rt66)、有限速度の球の係数と
Maxwell混合項は
[AERADE ARC-CP-0523](https://reports.aerade.cranfield.ac.uk/bitstream/handle/1826.2/536/arc-cp-0523.pdf?sequence=1)
および式を明記する
[Livi, Eq. 2.50](https://pure.tue.nl/ws/portalfiles/portal/292770290/20230419_Livi_hf.pdf)
を相互確認する。

```text
c0    = sqrt(2 k_B T_g / m_g)
c_bar = sqrt(8 k_B T_g / (pi m_g)) = 2 c0 / sqrt(pi)
S     = |u_g-v| / c0
nu0   = (4 pi / 3) a_p^2 rho_g c_bar / m_p
nu(S) = nu0 [G(S) + sigma_R pi/8]
a_d   = nu(S) (u_g-v)
```

鏡面成分のdrag coefficientとrate係数は

```text
C0(S) = (2 S^2 + 1) exp(-S^2)/(sqrt(pi) S^3)
      + (4 S^4 + 4 S^2 - 1) erf(S)/(2 S^4)
G(S)  = 3 sqrt(pi) S C0(S) / 16
```

である。`S>0.1`では相殺を減らした

```text
G(S) = 3/16 (2 + 1/S^2) exp(-S^2)
     + 3 sqrt(pi)/32 (4S + 4/S - 1/S^3) erf(S)
```

を使い、`S<=0.1`では

```text
G(S) = 1 + S^2/5 - S^4/70 + S^6/630 - S^8/5544 + O(S^10)
```

を使う。したがって`S->0`で`nu/nu0=1+sigma_R pi/8`となり、既存
`epstein_linear_v1`の`delta=1+sigma_R pi/8`へ連続に一致する。高速極限は`C_D->2`である。

非線形relaxationの速度Jacobianをboundするには係数`G`だけでなくradial固有値係数
`K(S)=G(S)+S G'(S)`を使う。

```text
K(S) = 3/16 [(4 - 2/S^2) exp(-S^2)
             + sqrt(pi) (4S + 1/S^3) erf(S)]
K(S) = 1 + 3 S^2/5 - S^4/14 + S^6/90 - S^8/616 + O(S^10)
```

caseは有限正値`maximum_speed_ratio`を宣言し、`lambda_g/a_p>=10`と
`S<=maximum_speed_ratio`を全stage・受理pathで認証する。これは式の値をclipするparameterではない。
加速度・exponential enclosureの係数上界には`G(S_max)+sigma_R pi/8`、RK4安定性には
`K(S_max)+sigma_R pi/8`を使う。

独立oracleはspecular分子について`rho_g pi a_p^2 E[|w| w]`を3-D Gauss--Hermite積分し、productionの
closed formを再利用しない。加えて低速極限、高速`C_D`極限、Jacobianの有限差分、一定primitive下の
scalar nonlinear ODE収束を検査する。`T_w!=T_g`、CLL、非球形、回転、transition-regime blend、
near-wall補正は別model revisionであり、この式へ隠れて追加しない。

### 3.2 電気力とDEP

電気力は`F_E=qE=ZeE`である。動的電荷では、RK stage内のZとEを同じstage状態で評価しないと
積分方法が変わる。

球形誘電粒子の準静的DEPは一般に粒子体積、媒質誘電率、Clausius–Mossotti factor、
`grad(|E|^2)`へ依存する。重要なのは係数だけでなく、`E`とgradientがどのsolution、時間平均、
回復処理から得られたかである。節点Eから数値gradientを作った値と、COMSOL内の回復gradientを
同一視しない。

P18-Dの最初のoptional revisionは`quasistatic_spherical_gradient_e2_v1`とし、runtime式を次へ固定する。

```text
K_CM = (epsilon_p - epsilon_m) / (epsilon_p + 2 epsilon_m)
F_DEP = 2 pi epsilon_m a^3 Re(K_CM) G_E2
G_E2 = grad(mean_E_squared)  [V^2/m^3]
```

DCでは`mean_E_squared=|E|^2`、RFではproducerが一つのfrequency/solutionについて物理時間平均（RMS二乗）を形成し、係数へ
二重の`1/2`を掛けない。caseは`epsilon_m=epsilon_0*medium_relative_permittivity`と`[-0.5,1]`内の実数`K_CM`を
一意に指定し、同じrunで直接Kと粒子誘電率を重複指定しない。coreが消費する空間量はproducerが形成した`G_E2`だけであり、solution ID、
時間平均、gradient recovery、mesh/topology IDをprovenanceへ残す。

初回適用域は球形、dipole/quasistatic、dilute one-way、実数CM factorとする。producerが同じfield/recoveryと所定誤差基準から
認証した`maximum_point_dipole_radius_m`をcaseへ一度だけ書き、`a`がこれを超える場合または認証できない場合は拒否する。
認証方法と誤差基準はprovenanceへ残す。加速度boundは有限な`max |G_E2|`から作る。一様Eで0、解析的な線形
`mean_E_squared`、Kの正負、`a^3/m` scaling、座標回転を独立oracleにする。複素周波数分散、travelling-wave DEP、
粒子間polarizationは別revisionである。

### 3.3 重力・浮力

```text
F_gb = (m_p - rho_g V_p) g
```

座標basisと重力方向をcase manifestへ明記する。`gravity_buoyancy_standard_v1`は、すべての
`axisymmetric_rz` domainで`gravity_m_s2[0] = g_r = 0`を要求する。非零一定`g_r`は一様重力ではなく、
標準modelへ読み替えない。`cartesian_xy`では第1成分`g_x`を非零にできる。低密度気体では浮力が
小さくても、式の削除ではなく感度または無次元比で省略を決める。

### 3.4 熱泳動とlift

P16のproduction revisionは、自由分子領域の単一気体についてWaldmann--Gallisの局所熱流束形を使う。

```text
c_bar = sqrt(8 k_B T_tr / (pi m_g))
F_th  = (32/15) a^2 q_tr / c_bar
a_th  = F_th / m_p
```

`q_tr [W/m^2]`は局所質量平均の中性気体座標に対する**並進伝導熱流束**であり、力は`q_tr`と同方向である。
Fourier域ではproducer側で`q_tr=-kappa_tr grad(T)`と形成でき、古典Waldmann勾配式へ戻る。coreは節点温度を
微分せず、total/convective/radiative/electron/ion heat fluxや固体熱流束を代用しない。Fourier producerは
translational conductivityを用い、少なくとも`lambda |grad(T)| / T <= 0.01`をproducer側で認証する。

v1は`a=drag_diameter_m/2`、慣性は`mass_kg`をauthorityとし、球形、単一気体、dilute one-way、
`lambda/a >= 10`、`|u_g-v|/c_bar <= 0.1`に限定する。閾値は文献上の不連続境界でなくrevision policyで、
全integrator stageと連続path enclosureでfail-closedに検査する。Talbot/continuum、混合気体、粒子内温度勾配、
photophoresis、near-wall・accommodation補正、negative thermophoresisへ自動blendしない。

P18-Rの`waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1`は同じWaldmann--Gallis heat-flux式を
再利用し、producerが認証した一つの有効Maxwellian/pseudogasだけを入力authorityとする。`q_eff [W/m^2]`はproducer所有の
有効並進伝導熱流束であり、total/convective/radiative/electron/ion heat fluxではない。coreはtemperature gradientを
回復せず、species配列やmixture ruleを持たない。適用域は球形、dilute one-way、`lambda/a>=10`、case明示の
`0 < maximum_speed_ratio <= 1`で、actual stageと連続pathをfail-closedに検査する。P16単一気体revisionの上限`0.1`は維持する。
このrevisionはreference/sensitivityであり、species-resolved mixture truthやCOMSOL専用branchではない。

保存成果物auditではnative linear Epstein式parityが最大相対残差約`1.1e-15`でPASSしたが、既存P15-E/P16の
physical applicabilityはmixture/model authority不一致により12/12 caseで`NOT_APPLICABLE`だった。PPRには必要な
`q_eff` primitiveが無く、thermophoresis pointwise replayは`NOT_TESTED`である。保存frameは連続pathを認証せず、
COMSOL studyの再実行も行っていないため、新revisionの追加は物理的真値や軌道一致を確立しない。

free-molecular liftは別modelであり、velocity gradient/vorticityと相対速度の定義が重要である。axisymmetric
no-swirlでは、診断上の方位成分と実際のr-z運動への投影を分ける。

現在の比較対象で明示された感度式は、P18-Lの`rarefied_vorticity_sensitivity_rz_v1`として次へ固定する。

```text
omega_phi = d(u_r)/dz - d(u_z)/dr
C = C_L pi rho_g lambda_g a^2
F_L,r =  C (u_z - v_z) omega_phi
F_L,z = -C (u_r - v_r) omega_phi
```

`omega_phi [1/s]`はproducerが同じgas-velocity solutionと回復規則から出し、coreは速度を微分しない。初回revisionは
axisymmetric RZ、no-swirl、球形、dilute one-way、free-molecular範囲だけを受理し、`C_L`は隠れたfit値でなくmanifestへ
明示する。zero-vorticity、comoving、符号、`rho lambda a^2/m` scaling、解析的shearと3-D cross-productの独立oracleで
検証する。この式はreferenceのmodel-form感度であって、一般liftまたは実験的真値ではない。

Waldmannの自由分子式は[Waldmann (1959)](https://doi.org/10.1515/zna-1959-0701)、局所熱流束形と低圧・
壁近傍への一般化の背景は[Gallis, Rader, and Torczynski (2004)](https://doi.org/10.1080/02786820490490001)
を参照する。入力不足または適用域外で別modelへ黙って切り替えない。

### 3.5 ion drag

ion dragはcollection項とorbital scattering項、イオンと粒子の相対速度、イオン温度、screening
length、衝突性の近似に依存する。方向を相対速度に取るか電場方向に取るかでも軌道が大きく変わる。

最初のproduction revisionはBarnes型のcollisionless collection＋orbital modelに限定する。

```text
w = u_i - v
v_s^2 = |w|^2 + 8 k_B T_i / (pi m_i)
lambda_D = [e^2/(eps_0 k_B) (n_e/T_e + n_i/T_i)]^(-1/2)
phi = Z e / [4 pi eps_0 a (1 + a/lambda_D)]
b_90 = |Z| e^2 / (4 pi eps_0 m_i v_s^2)
b_c^2 = a^2 [1 - 2 e phi / (m_i v_s^2)]
ln Lambda = 0.5 ln[(lambda_D^2 + b_90^2)/(b_c^2 + b_90^2)]
F_i = n_i m_i v_s [pi b_c^2 + 4 pi b_90^2 ln Lambda] w
```

screening authorityは電子と単一正イオンのlinear two-species Debye lengthである。外部field名や別のscreening
closureへ暗黙に切り替えない。revisionは単一・単価正イオン、球、Debye--Huckel capacitance、非正表面電位、
非磁化・collisionless background、相対流方向だけを扱う。`a/lambda_D<=0.1`、
`b_90/lambda_D<=0.1`、`b_c/lambda_D<=0.1`、`lambda_in/lambda_D>=10`、`ln Lambda>0`と、
caseが宣言する有限drift範囲をcontinuous-path gateにする。0.1と10はsharpな文献境界でなく、弱結合・
collisionless範囲を狭く固定するrevision policyである。Coulomb log、charge、
relative energyをfloorまたはclampせず、適用外は失敗させる。

orbital項はRutherford偏向`tan(theta/2)=b_90/b`に対する
`integral 2 pi b [1-cos(theta)] db`を`b_c`から`lambda_D`まで積分すると上式へ一致する。この独立積分を
verification oracleとし、production閉形式を再利用しない。Barnes modelは大きな粒子電位や非線形screening、
collisional sheathで普遍ではないため、それらはKhrapak型またはcollisional revisionとして分離する。

`model_dataset`のtheory-consistentとimage-minimal-correctedはsolver誤差の上下限ではなく、floor、方向、
screening、image近似まで異なるmodel-form sensitivityである。production truthとして複製せず、比較は外部V&Vで
force vectorと時間軌道を評価する。

Stage 3では比較を再現可能にするため、これらをP18-Iの二つのoptional revisionとして独立に再定義する。

- `relative_flow_screened_collection_orbital_aggregate_ion_v1`：相対流方向、collection＋orbital、collision-limited
  screeningを明示する。
- `electric_field_directed_image_orbital_sensitivity_v1`：電場方向、image-form collection/orbital、zero-field方向規則を
  明示する感度modelとする。

M3-C0aで全`min/max/floor`、zero limit、方向、単位、scale=1を保存式から固定した。relative-flow式はCase P/Aで
同一だったため、`u_eps=1 m/s`、`sqrt(Z^2+1e-20)`、collection cap、非負Coulomb logをそのrevisionの式として
採用する。一方、image式のscalar ion speedはCase Pの`sqrt(|u_i|^2+u_eps^2)`とCase Aのproducer固有
`AS_ui_mag`で一致しなかった。この差をcoreのCase分岐や重複scalar fieldへせず、production sensitivity revisionは
`U=|u_i|`、`s^2=U^2+8eT_iV/(pi m_i)+u_eps^2`へ一意化し、zero ion flowでzero forceとする。
`lambda_img=sqrt(eps0*T_eV/(e*n_i))`もrevision内で構成し、derived fieldを入力しない。

両revisionは既存Barnesと排他的に選択し、blend、自動fallback、上下誤差幅、Case名を持たない。保存Case固有式の
formula replay、production式との差、弱結合・衝突性等のphysical applicabilityは外部V&Vで別statusとして判定する。

P18-I実装はこの区分をcatalog v12 / runtime v11 / compiled tile v13の単一stage passへ接続して完了した。
COMSOLを再実行せず保存済み12 package・397,820 active rowを再生し、producer保存式は最大正規化残差
`1.7901e-15`でPASSした。一方、production relative-flow式は保存COMSOLと標準SIの`epsilon0`定数規約差により
strict比較をFAILのまま保持し、image式はCase P/Aのscalar speed定義差を非FAILのmodel-definition differenceとして
保持する。この外部結果から統合軌道精度、連続path適用性、境界、Brownian一致は主張しない。

### 3.6 near-wall physics

壁近傍drag、sheath、image force、粗さ、再飛散はbulk forceとは別moduleである。これらには正確な
壁距離、法線、material、適用距離が必要で、境界捕捉toleranceを物理厚さの代用にしてはならない。

---

## 4. 電荷model

### 4.1 連続平均電荷

プラズマdustでは、電子・イオンfluxから

```text
dZ/dt = Gamma_i(Z, plasma state) - Gamma_e(Z, plasma state)
q = Ze
```

を解く。OMLはMaxwell分布、球形孤立粒子、sheath/衝突性等に仮定があり、全条件で普遍ではない。
指数項のoverflow対策を単なるclipで隠すのではなく、無次元化、安定な数式、model適用域を示す。

P15の最初のmodelは`oml_stationary_maxwellian_debye_huckel_v1`とする。半径`a`の球について
Debye--Hückel capacitanceと表面電位を

\[
C_{DH}=4\pi\epsilon_0 a\left(1+\frac{a}{\lambda_D}\right),\qquad
\phi_p=\frac{Ze}{C_{DH}}
\]

とし、species `s`の電荷を`q_s`、密度を`n_s`、温度を`T_s`、質量を`m_s`として、stationary
Maxwellian OMLの収集率を

\[
\Gamma_s=\pi a^2 n_s\sqrt{\frac{8k_B T_s}{\pi m_s}}\,g(\chi_s),\qquad
\chi_s=\frac{q_s\phi_p}{k_B T_s},
\]

\[
g(\chi)=
\begin{cases}
\exp(-\chi), & \chi>0,\\
1-\chi, & \chi\le 0
\end{cases}
\]

と固定する。charge rateは`R_Z=sum_s (q_s/e) Gamma_s`であり、単価正イオンと電子なら
`R_Z=Gamma_i-Gamma_e`である。ion driftをこの収集率へ補正式として入れない。resolved ion-drift
Mach数`M_i <= 0.1`をstationary近似のapplicability gateにだけ使い、超過時は別式へ切り替えず拒否する。
Debye--Hückel/isolated-particle近似の初回範囲は`a/lambda_D <= 0.1`とする。

P15-Dでは、単一・単価正イオンの有限bulk driftを扱う負電位限定revisionを追加する。設定上の

\[
M_i=\frac{|u_i-v|}{\sqrt{8k_BT_i/(\pi m_i)}}
\]

と、shifted-Maxwellian momentの変数
\(s=|u_i-v|/\sqrt{k_BT_i/m_i}=\sqrt{8/\pi}M_i\)を区別する。\(\phi_p\le0\)で

\[
\Gamma_i=A_i\left[P(s)-\frac{\phi_p}{V_i}H(s)\right],
\quad
P(s)=\frac12e^{-s^2/2}+\sqrt{\frac{\pi}{8}}(s+s^{-1})\operatorname{erf}(s/\sqrt2),
\]

\[
H(s)=\sqrt{\frac{\pi}{2}}\frac{\operatorname{erf}(s/\sqrt2)}{s},
\quad \Gamma_e=A_e e^{\phi_p/V_e}
\]

とする。`P(0)=H(0)=1`でstationary OMLへ一致し、小さい`s`はTaylor級数で評価する。速度floor、
energy floor、指数clipで`1/s`を隠さない。caseは有限正値`maximum_ion_drift_ratio`をrun-wide envelopeとして
明示し、初期`Z<=0`と全primitive/drift rangeで非正平衡が存在する時だけ`[Z_min,0]`を受理する。
正電位、負イオン、複数species、emission、collisional/magnetized chargingは別revisionである。
shifted-Maxwellian collectionの理論と適用上の注意は
[Alexandrov et al. (2008)](https://nano.uantwerpen.be/nanorefs/pdfs/OA_10.1088_1367-2630_10_9_093025.pdf)と
[Douglass et al. (2011)](https://doi.org/10.1063/1.3624552)を参照する。

`model_dataset`由来の帯電heuristic、保存済みcharge履歴、COMSOL側のmodel名は、この式や適用域を
定義するcore truthではない。外部V&Vで軌道・電荷時系列の感度と差を調べる入力に限り、coreは
canonical plasma primitiveと上記versioned式だけに依存する。

ただし同条件比較を可能にするP18-Cでは、保存されたaggregate式を独立に再定義した
`aggregate_relative_drift_regularized_two_current_v1`をoptional reference revisionとして追加する。これは正イオンを
species-resolvedに解くmodelではない。M3-C0で次を一つのspecへ固定する。

- `w=u_i-v`、正則化された相対speed、正イオン密度・局所有効質量・有効温度、電子基準flux
- 総表面電位`phi=Z*phi1 <= 0`と正側のion/electron current branch
- speed regularization、ion-energy floor、指数評価範囲を含む全ての数値定数と単位
- `Z` invariant、`|R_Z|`と`|dR_Z/dZ|`の有限bound、およびbranch境界の連続性

局所有効正イオン質量、電子・正イオンthermal voltage、背景screening長はcanonical scalar field一つずつに統一し、
uniform producerも定数fieldを書く。screening長はproducerの背景量をauthorityとし、revision内で
`lambda_eff=max(a, lambda_screening)`を適用する。caseは正則化前の`|u_i-v|`に有限正値
`maximum_relative_ion_speed_m_s`をrun-wide envelopeとして必須指定し、actual stageと連続path enclosureの双方で
超過を拒否する。この値は速度clipやfit parameterではなく、有限なcharge invariant/rate boundの証明条件である。floor/clampは
overflow repairやsilent fallbackではなく、このrevisionの明示的なmodel-formである。P15/P15-Dの式と適用域は変更せず、
同じstageの`x,v,Z`で積分し、有限boundを認証できない場合は比較のために安全gateを緩めず実装を停止する。

P21では、この二電流revisionを変更せず、aggregateな単一価負イオン収集電流を一つだけ加える
`aggregate_relative_drift_regularized_three_current_v1`を別revisionとして定義する。charge rateは

\[
R_Z=\Gamma_+-\Gamma_e-\Gamma_-
\]

である。負イオンについて、producerが与える密度`n_-`、thermal voltage`V_-`、速度`u_-`、有効質量`m_-`から

\[
v_{eff,-}^2=|\boldsymbol u_--\boldsymbol v|^2+\frac{8eV_-}{\pi m_-}+u_\epsilon^2,
\qquad
V_{eff,-}=\max\left(\frac{m_-v_{eff,-}^2}{2e},V_{floor}\right)
\]

を作る。`phi<=0`では負イオンを反発種として
`C_-=exp(clip(phi/V_eff,-,-50,50))`、`phi>0`では引力種として
`C_-=1+phi/V_eff,-`を用い、`Gamma_-=pi*a^2*n_-*v_eff,-*C_-`とする。正負イオンの
相対speedは同じ明示`maximum_relative_ion_speed_m_s`でfail-closedに囲い、既存のcharge invariant、rate/Jacobian bound、
RK4またはaffine exponential couplingを同じsingle engineで使う。`n_-=0`ならrate、Jacobian、global boundは二電流revisionへ
厳密に退化する。ただし三電流revisionを選んだcaseでは負イオンfieldの有限性と`|u_--v|` applicabilityを密度0でも検査するため、
public run全体の同一性は両revisionの適用域条件も共に成立する場合に限る。

このrevisionは「aggregate singly-negative-ion collection」までを責務とし、species-resolved current、複数負イオン種の
個別反応、emission、collisional/magnetized sheathを主張しない。複数speciesをまとめる場合の集約規則と不確かさはfield
producerのprovenanceが所有する。screening lengthは従来どおり明示入力fieldだけがauthorityであり、負イオン密度からcoreが
Debye長や別の寄与を再計算しない。

元Case-PのCOMSOL `auxq`は電子と正イオンの二電流を意図的に使い、負イオン密度は診断量に留めている。したがって既存の
Case-P same-form anchorは二電流式の比較として解釈し、三電流revisionをその不具合修正や遡及的な再認定に使わない。三電流の
物理・軌道比較は、設定意味を揃えた別のCase-P派生companionで評価する。production revisionはcatalog v17、runtime v19、
compiled tile v18として同じsingle engineへ統合済みで、標準verification/scenario suiteと品質gateを通過した。外部companion入力監査は
canonical負イオンprimitive authority不足により`BLOCKED / NOT_EVALUATED`で閉じ、物理modelの`NOT_APPLICABLE`とは扱わない。
軌道は実行しておらず、元Case-P二電流anchorは不変である。この任意物理の外部coverageはP21の出口から分離し、三電流の
COMSOL軌道同等性を非認定のまま、P21/M3-C3は`CLOSED_ACCEPTED_WITH_LIMITATIONS`とする。外部statusのauthorityは
[`vv_methodology.md`](vv_methodology.md)と
[`solver/evidence/m3c3/caseP_three_current_companion_v1/`](solver/evidence/m3c3/caseP_three_current_companion_v1/README.md)である。

連続電荷の最初のreference解法は、位置・速度と同じstageで`Z`を評価する既存の古典RK4とし、
modelは有限なcharge invariant`[Z_min,Z_max]`、その区間上のrate上界`B_Z >= |R_Z|`と
derivative上界`L_Z >= |dR_Z/dZ|`を外向きに認証する。RK4の各区間は`h L_Z <= 0.5`を必須とし、
全stageとaccepted endpointがinvariant内にある場合だけ受理する。認証不能、非有限、またはgate超過を
clip、平衡値、hidden subcycleで隠さない。動的`Z`は運動と同じRK4 stateなので、加速度が見かけ上一様でも
linear/quadratic exact pathへ退化させない。RK4のexplicit `hL_Z<=0.5`は現行でも維持する。

native exponential midpointとB03は、midpoint predictorで`G_mid=R_Z(Z_mid)`と
`J_mid=dR_Z/dZ<=0`を一度評価し、root始点へ平行移動した
`A=G_mid+J_mid(Z_root-Z_mid)`を用いるaffine exponential updateをproductionへ接続済みである。

```text
Z(root + dt) = Z_root + expm1(J_mid*dt)/J_mid * A
J_mid = 0 では Z_root + dt*A
```

`expm1`と連続な`J=0`極限により強い負のJacobianでも平衡へ単調に近づく。同じmidpoint chargeが運動の
accelerationにも入るためoperator splitではない。exponential pathへRK4の`hL_Z<=0.5`をstability gateとして
流用せず、accuracyは別runの`h,h/2,h/4`で選ぶ。clip、charge-only subcycle、第二engineは作らない。

scalar implicit midpointを「剛性帯電の標準解法」とはしない。線形緩和

```text
Z' = -k (Z - Z*)
G_implicit_midpoint = (1 - h k / 2) / (1 + h k / 2)
```

ではA安定でもL安定ではなく、`h k > 2`で平衡を跨ぎ、`h k -> infinity`で`|G| -> 1`となる。
scalar implicit update、全状態bounded dyadic、quasistatic-equilibriumへの暗黙切替は現行pathへ追加しない。
chargeだけを暗黙更新して運動へ渡す一次operator splitも用いない。

### 4.2 離散電荷

10 nm級では電子1個の変化が無視できない場合がある。連続Z ODEと、整数Zの確率jump processは
別modelとして用意する。後者は電子/イオン到来率をhazardとしてイベント生成し、Brownianとは
別RNG streamを持つ。

### 4.3 COMSOLのcharge modelとの区別

COMSOLのCharge Accumulationに含まれる空気aerosol向けmodelは、プラズマOMLと同じではない。
公式[Charge Accumulation](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.33.html)
のmodel名を、そのままプラズマ帯電の証拠にしない。新基盤では`fixed`、`plasma_continuous`、
`plasma_discrete`、`aerosol_relaxation`を別typeにする。

---

## 5. Brownianと確率過程

fluctuation–dissipationが成立する線形Langevin系は概念的に

```text
dv = -(v-u)/tau dt + sqrt(2 k_B T / (m tau)) dW
dx = v dt
```

である。overdamped近似では位置拡散へ落とせるが、壁へのfirst passageや短時間速度を比較する
場合は近似の選択を明示する。

COMSOLのBrownian forceはsolver time stepに依存して評価されるため、異なるdt・乱数割当で同じseed
だけを揃えても同じpathにはならない。公式
[Brownian Force](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.08.html)
と[Brownian Motion example](https://doc.comsol.com/6.4/doc/com.comsol.help.models.particle.brownian_motion/brownian_motion.html)
を参照する。

Case Aのbuilt-in Brownianへ与えるeffective viscosityは、線形Epstein/FDTと係数形を一致させるための量である。
保存式`S_ep=36(lambda/d)/(8+pi*sigma)`、`mu_B=mu/S_ep`をStokes係数
`beta=3*pi*mu_B*d`へ代入し、`lambda=mu*sqrt(pi/(2*p*rho))`と理想気体関係を使うと、

\[
\beta=\frac{4\pi}{3}a^2\rho\bar c\left(1+\frac{\pi\sigma}{8}\right),
\qquad a=d/2
\]

となり、candidateの線形Epstein係数と一致する。従ってM3-C2 companionはcustom random-forceを追加せず、
built-in `bf1`へ`mu_B`と同じgas temperatureを渡す。これはproducerが採用したeffective-gas closure内の代数的一致であり、
species-resolved mixtureの物理妥当性を認定しない。

実装原則は次である。

- pathwise互換性を明示的に検証するmicrocaseだけ、同じ内部stepと乱数割当てを指定する。M3-C2 ensembleは
  COMSOLとcandidateで独立にstep/tolerance収束を確認し、共通出力時刻のobservableを比較する。
- native modeはOUのjoint position/velocity更新を用いる。
- adaptive分割時はBrownian bridgeで親incrementを子へ分ける。
- `seed, particle_id, macro_interval, root_stochastic_interval, tree_level, tree_index, component, stream`
  をcounter keyにし、accepted-step番号やevent ordinalへ依存させない。
- CPU thread、GPU warp、tile、inactive粒子の並べ替えで乱数pathを変えない。
- pathwise比較ができないcaseは、MSD、平均、共分散、分布距離、first-passageをreplicaで比較する。

B01ではGillespieのOU processとその積分のexact joint simulationに対応する平均・共分散、親endpointを保存する
conditional half-split、物理interval-tree Philoxまでを実装した。これはcaseから選べるsolver能力ではなく、
event/replay接続前の数値基盤である。[Gillespie (1996)](https://doi.org/10.1103/PhysRevE.54.2084)は
OU速度とその時間積分を任意step幅で同時にexact simulationできることを示している。

壁へ接続する最初のrevisionは、geometryやoutput要求で乱数pathが変わらない固定depthのdyadic node集合を作り、
nodeの位置・速度から各leafのcubic Hermite numerical pathを一意に定める。この有限解像度pathのfirst hitを
既存event interfaceで求め、depthを増やした時のfirst-passage分布収束を検証する。単なる`stochastic RMS`距離を
決定論的clear certificateとして使わない。後にadaptive clearを導入する場合は、Gaussian bridgeのcrossing確率から
全runのmiss-probability budgetを明示的に配分できる時だけ別revisionとする。

counter-based RNGのstatelessなkey/counter構造は
[Random123 official documentation](https://random123.com/releases/docs/)を参照する。
本実装のidentityは`seed, particle_id, macro_interval, root_stochastic_interval, tree_level, tree_index,
component, stream`で固定する。B01の平均はdrag-onlyであるため、最初のproduction接続はCartesian XY、Epstein linear
drag、fixed charge、terminal stick/escapeに限定し、それ以外を暗黙に無視しない。

B03では線形Epstein/FDTのjoint OUを残し、決定論力とcontinuous chargeを一つの
macro-root stochastic exponential-midpoint proposalへ合成する。

```text
noise-free deterministic exponential-midpoint predictor: root start -> root midpoint
freeze gamma, u, T, additive acceleration a, G=dZ/dt, and J=dG/dZ <= 0 there
joint exact OU over the full root with u_eff = u + a/gamma
root-owned affine exponential Z(t), using G_mid + J*(Z_root-Z_mid)
```

予測子は係数評価点を作るためだけで、stateを先にhalf-step commitするoperator splittingではない。
nativeとeffective-gas sensitivityの線形Epsteinだけを受理し、既存の加算加速度を同じmidpointで一度だけ
評価する。固定深度のconditional treeはrootと同じ凍結係数で分割し、root endpointを保存する。
Zはfixed、またはroot内で上記の一つのaffine-exponential dense stateとし、chargeだけのsubcycleを作らない。

RZ axis hitは壁反射ではない。cubic pathのaccepted prefixをhitまでcommitしてradial stateをfoldし、
macro-stepの残時間は次の`root_stochastic_interval`として係数を再評価し、独立なroot drawから再開する。
元rootのcubic remainderをfoldまたはrestrictして再利用しない。一方、axis hitのない同一root内では親Gaussianを
conditional splitし、geometryやoutput要求で引き直さない。

受入で主張するのは、凍結した定数係数に対するmeanとfull covarianceのexactness、noise-off決定論極限の
2次収束、manufactured charge-electric caseのweak mean観測次数`>=0.9`、tree-depth first-passage収束である。
一般のstate-dependent SDEのstrong orderまたはweak 2次は主張しない。terminal boundaryは
`stick`/`escape`/`hold`だけとし、反射や確率壁は独立な数値契約が完了するまで解禁しない。

現在のCOMSOL RZ比較向けrevisionはr/zへ投影した2自由度SDEで、等方3-D過程ではない。P17後のCartesian 3-D revisionを
物理的なisotropic Brownianの経路とし、RZ projectedの合格を3-Dの根拠にしない。pathwise再現が証明できない比較は、
12 packageごとの独立seed ensembleでmean、covariance、MSD、occupancy、fate、first-arrival分布を評価する。

このB03 compositionの初回closeoutはengine v34 / proposal v9 / catalog v16 / event v15 / runtime v17 / compiled tile v16 /
memory plan v13としてcore受入を完了した。現行supersessionはengine v36 / proposal v10 / catalog v17 /
event v16 / runtime v19 / compiled tile v18 / memory plan v13である。COMSOL再実行はcore受入に不要だった。
P20 performance closeoutとmeaning-matched common-P1 Case-A/Case-P 100 nm external V&V/M3-C2は完了した。Case-Pは
元COMSOL `auxq`どおりの二電流same-form比較であり、後続three-currentまたはspecies-resolved物理を認定しない。受理済みCase-P seed 3件の
287粒子owner discoveryも、受理済み科学payload・work・case identity・revisionの完全一致を保って完了した。支配ownerは
3 seedとも`integrators`（自己時間比42.58--42.86%）だったが、事前登録済みbounded ownerではないため最適化は未承認である。
M3-C2A anchorは`CLOSED_ACCEPTED_WITH_LIMITATIONS`である。10,000粒子以上の性能は製品SLAを先に定義した
独立work packageだけで評価し、COMSOL fittingは行わない。
後続のbounded follow-upでは、この発見をchord算術一箇所へ限定してserial compiled batch化し、accepted science/work identityを
保ったままend-to-end中央値を12.29%短縮した。追加の数値model、診断、runtimeは導入せず、次ownerへ進まず終了した。
正式characterizationは
2,000/20,000粒子×4構成×3反復を全粒子active・failure 0で完了し、静的なB03 path-array上限`648 B/row`も
`2048 B/row`以内である。計時/RSSは[`solver/evidence/b03/`](solver/evidence/b03/README.md)のmachine-local・
non-gating観測である。characterizationで検出した反復加算由来の終端tailはengine v34で修正し、macro timeを
補償積和による`start + n*dt`のindexed gridから構築してfloat64構築roundoff内だけendへsnapする。

---

## 6. 場、mesh、再mesh

### 6.1 geometryとfield support

geometry topologyが粒子domainの内外を決め、fieldごとのsupportが値を評価できる領域を決める。
この二つは同じとは限らない。`inside_model_domain=1`や「どれかの物理列が有限」をdomain maskに
使わない。

さらに`GeometryDomain`と`FieldLayout`自体も同一とは限らない。geometryは衝突facetと粒子
domain、fieldはregular/unstructured layoutとbasis/supportを所有する。同じFE meshの時だけmesh IDと
locatorを共有し、STL境界＋CFD mesh、RZ geometry＋regular cacheを不一致として拒否しない。

試行stepを評価するためsupport外にnearest valueを返すことは可能だが、値とsupport statusを
同時に返す。値が有限であることをinside判定に使わない。

### 6.2 COMSOL mesh

外部再現に必要なのは座標だけではない。

- stable node/element/facet ID
- element type/order/local node order
- domain/boundary selection
- element adjacency、boundary owner
- field DOFとbasis、またはCOMSOL評価oracle
- smoothing、recovery、evaluation location

現在のquad CSVは正しい周回順が`node1,node2,node4,node3`であり、列順のままpolygon化すると
2,426要素中2,019がbow-tie/zeroになる。canonical importerでorderingを明示し、面積・向き・
Jacobianを検査する。

P1 triangleはbarycentric、Q1 quadはbilinear isoparametric補間を使う。quadを暗黙tri分割すると
補間面が変わる。P2を頂点P1として読むことも禁止する。

COMSOL結果exportのsmoothing/recoveryは
[Interpolation](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_ref_results.37.224.html)
と[Mesh API](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_api_mesh.49.047.html)
を参照し、versionと設定をmanifestへ残す。

### 6.3 高速cache

規則格子への再meshは高速化cacheであり、reference fieldではない。生成時に次を測る。

- scalar/vectorのL-infinity、L2、分位誤差
- gradient誤差
- wall/sheath近傍の誤差
- supportのfalse-positive/false-negative
- 軌道・eventへの誤差伝播

solidを跨ぐ補間やNaN fillは禁止する。cache cellは元element/supportへ追跡できるようにする。

### 6.4 時間依存場

時刻knots、補間規則、範囲外policy、discontinuity、mesh固定/移動を明示する。実行時は前後2時刻の
fieldだけをdouble bufferへ置き、discontinuity/knotでstepを分ける。各integrator stageの実時刻で
空間・時間を同時評価する。時間範囲外を無条件に端点clampしない。snapshot自体の不足はparticle
stepで修復できないため、外部preprocessorが間引き検証やsecond differenceで解像度を評価する。

---

## 7. 境界数値

step終点だけの内外判定では、高速粒子が薄い壁を通り抜ける。積分器が返すendpoint、
`state_at(theta)`、path error estimateから十分に制御されたpiecewise segmentを作り、最小正時刻の
boundary intersectionを求める。

```text
trial path
  → candidate facetをBVHで絞る
  → robust intersection
  → first eventを局在
  → wall law
  → 残りstepを新状態で継続
```

材料境界はsegmentのAABBからboundary BVHで候補を絞り、exact intersectionを最終authorityとする。
P1/Q1 field locationは、受理済みstrict-interior cell hintを最速経路とし、hintなし/missは
supported-containment BVHで候補を絞る。outside/masked provisionalだけは物理最近傍とtieを守るfull scanへ戻す。
現行経路へneighbor cell-walkを重ねない。toleranceはgeometry scale、facet scale、速度、dt、float ULPから
一度解決する。固定mのepsilonで位置を内側へずらすrepairは、collision時刻とrelease位置を変えるため用いない。

v0.1はpoint-particle衝突であり、粒子中心が境界へ達した時をeventとする。有限半径offset surfaceを
geometry toleranceで代用しない。`axisymmetric_field_cartesian3d`ではCartesian pathを
`(r(t),z(t))`へ写した曲線とRZ断面境界を交差させ、RZ法線を衝突点thetaでCartesianへ戻す。

壁物理はintersectionから分離する。現行revisionは`point_wall_laws_v5`である。

- stick/absorb
- escape
- parameterを持たない非deposition terminal `hold`
- parameterを持たない完全鏡面`specular`
- 法線・接線反発係数を持つ`restitution`
- 定数確率の`probabilistic_stick`と、`otherwise`へ明示する`specular | restitution` fallback

以上をv0.1対象とする。`hold`はhit位置、hit時速度、hit時電荷を保持するinactiveな`held` lifecycleであり、
その後の力、電荷、kinematicsを時間発展させない。paused/restart/resuspensionを意味しない。diffuse reflection、
速度依存rebound、resuspensionは後続catalogである。
`specular`は係数入力を受けず、法線成分を厳密に反転して接線成分を保存する。`restitution`は
`normal_restitution`と`tangential_restitution`をともに必須とし、確率lawの非stick側もdefaultで補わない。
COMSOL固有のfreeze/disappearは外部adapterがraw stateを保持する。boundary 37/35へ確実に到達する正例microcaseで、
実際に出力されたhit後位置、速度、保存statusを確認して外部manifestの対応を定める。現referenceには
電荷列がないためCOMSOL電荷保持は認定せず、generic `hold`の非零電荷保持は解析・公開scenarioで独立に検証する。Freezeをstuckへ写像せず、
製品利用caseにも必要な非deposition終端はprovider非依存の`hold/held`として実装し、material deposition、escapeと
統計を分ける。COMSOL名やraw status codeはwall lawへ入れない。

event logは時刻、点、boundary ID、法線、pre/post速度、電荷、結果、乱数counter、局在残差を
保存する。保存時刻のstatus変化からeventを推測しない。

COMSOLの壁設定とwall accuracy orderは公式
[Wall](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.43.html)
を参照し、`tools/vv/comsol`の比較manifestへ写す。canonical caseには一般化したwall lawだけを
置き、corner、同時hit、step内最大interaction数も比較条件として外部に明示する。

---

## 8. 積分法

### 8.1 固定step reference mode

現在の`model_dataset`は固定10 µsの古典RK4を宣言しているので、外部V&Vでは一般的な
`rk4_fixed`へ同じstepを指定する。運動と電荷を同じstageで連成し、stageごとに場を再評価する。
Brownian乱数をどのstageで更新するかは推測せず、Brownian-off micro-traceとCOMSOL側stage
exportで確定する。

既知の線形dragに対しては保守的に`h/tau_min < 2.5`を要求する。これを満たさないstiff caseを
有限値のまま進めず、明示的にstepを小さくするかexponential methodを選ぶ。
一般RK4の数学的なHermite/state pathと受理endpointは固定し、dense位置評価と部分区間の位置認証を
root始点相対のBernstein制御点で行う。座標差はTwoDiffで丸め値と厳密残差に分け、評価時は両方を足して
world座標へ戻す。始終点は保存済みendpointで厳密に上書きし、enclosureの下限と上限はそれぞれ
`-inf` / `+inf`方向へ丸める。物理的な曲率とenclosureは局所変位scaleで計算するが、公開chord-deviation
boundは最終のworld座標変換、両endpoint、endpoint chord式の丸めを覆う狭い`8*eps`絶対座標termを
別に加えるため、戻り値自体を完全な平行移動不変とはしない。これをdense path v3とし、v2の広い絶対座標paddingが
二分後も十分に縮まずevent certificateが閉じなかった問題を解消する。first-hit/event algorithmと
RK4 global enclosureは変更しない。

COMSOLの一般既定法は方程式次数・version・設定で異なるため、coreに`comsol_default`や
version別profileを作らない。COMSOL adapterが通常のsolver設定へ変換し、比較条件を外部manifestへ
保存する。

### 8.2 native deterministic mode

`dv/dt=-(v-u)/tau+a`をmidpointで局所凍結し、`E=exp(-h/tau)`、`A=1-E`として

```text
v1 = u + E(v0-u) + tau A a
x1 = x0 + u h + tau A(v0-u) + tau{h-tau A}a
```

を使う。`A=-expm1(-h/tau)`と小引数級数で相殺を避ける。start係数でhalf-step predictorを作り、
midpointのfield、charge、drag、加算加速度を評価する。continuous chargeは同じmidpointで`G,J<=0`を凍結し、
上記affine exponential root pathを使う。一定係数では解析解を再現し、可変場で2次をverificationする。
非線形dragを黙って線形緩和へ入れない。

Stage 1はfixed stepと別runのstep-halvingだけにし、field knot/global source discontinuity、wall eventで
必要な区間を切る。個別releaseは粒子別残時間work、出力時刻はaccepted pathから評価する。Stage 2以降で
必要になった場合の別methodは別decisionとし、現行charge pathへsubcycleを追加しない。accuracyの
step reject、field support、geometry indeterminate、wall event refinementを別statusにし、accuracy
budgetが尽きたことを壁到達に変換しない。

### 8.3 reference solver

小粒子数の原因分析には、高精度な一般ODE solverとevent機能をreferenceとして使える。
[SciPy solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
はdense outputとeventを提供する。ただし本番の100万粒子backendにそのまま使うのではなく、
解析case・kernel検証のoracleとする。

### 8.4 収束

`dt,dt/2,dt/4`、必要ならmesh系列を計算し、位置・速度・電荷・event時刻/点の観測収束次数を
調べる。COMSOLとの差だけを見て係数を調整せず、各solverの自己離散化誤差、field export誤差、
event局在誤差を合成して許容帯を作る。

---

## 9. Python高速化

### 9.1 CPU/Numbaを最初にする

初期製品はSoA配列、Numba compiled batch、cache-friendlyな固定容量slabを使う。Numbaの公式
[performance tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html)に従うが、P14-Pの実測判断により
製品runtimeはsingle-thread compiled engine一つとする。外側の`ThreadPoolExecutor`、Numba thread mask、
multiprocessing、第二schedulerは併存させない。Python object、可変list/dict、例外、callbackを粒子loop内へ
持ち込まず、独立caseのprocess並列はsolver外で行う。

v20のouter ThreadPoolはP12/P14の履歴実装である。large regular 1Mでは正のscaleを示したが、P1/Q1 10kは
ほぼ伸びず、event-heavyは遅化した。さらにfield sampling、physics、RK stageごとの一時配列、Pythonの曲線event/
exact residual調停、worker wave barrier、thread数倍のscratchが残るため、worker数調整では汎用的な改善にならない。
P14-Pでこの方式を置換した。P14-P closeoutのv27はouter pool、future wave、worker別scratchとruntime thread maskを
削除し、bounded slab、preallocated field/physics/integrator workspace、stackless BVH、
同期single-owner writerまで統合した。linear/quadratic exactと一般曲線eventのflat SoA wavefront、compiled
wall/axis locator、boundary/Philox、row numerical status、batch surface release、direct columnar replay、bounded
event/failure stagingもengine接続済みである。

1. 一つのbounded tile slabをprepare時に確保し、field、physics、integratorはpreallocated
   `into` bufferへ書く。最初はowner別passを保ち、profileで配列往復が支配的な場合だけ同じtile内で融合する。
2. exact/curved event、残時間、境界応答、Philox drawをflat SoA work queueのroundへ変換する。各行は固有particleが
   所有し、行ごとのtarget time、refinement path、interaction/event ordinal、statusを数値配列で保持する。
3. 各roundは一行あたり高々一event/failureを固定columnar bufferへ書き、count→prefix→fillまたは固定row slotと
   stable compactionで次roundを作る。boundary identityは`(particle_id,event_ordinal)`、公開時のcanonical順は
   `(time_s,particle_id,event_ordinal)`とし、thread完了順に依存しない。
4. boundary BVHはper-row可変stackを持たないstackless skip traversalとし、boundary lawとRNGのscalar authorityを
   同じmodule内のcompiled leafから呼ぶ。Python per-particle fallbackは作らない。

このwavefrontはevent数を小さいと仮定せず、roundごとの有界bufferだけを使う。行ごとのtarget timeによってrelease、
hit、split後の異なる残区間を同じcompiled batchで処理できる。accepted state/hintだけをcommitし、simultaneous
facetはdeterministicなcount/prefix/fillで保持する。これによりscratch peakは概ね`O(tile slab)`でthread数に比例せず、
P15/P15-Dのcontinuous chargeもこの実行原理を使い、時間依存場、3-Dへ同じ原理を拡張できる。
deferred event depthのworst-caseは一般scratchへ隠さず、P14-P時点のmemory plan v10で
`event_work_bytes_per_particle = 24 * (max_refinements + 1)`、component `slab_event_work`として計上する。
現行memory plan v13もこのevent-work計上を維持する。
候補、event/failure staging、surface release、direct replayはnamed componentへ分離し、pack時だけのgatherは12.5%
safety marginが所有する。正確なbyte式は
[`solver/docs/parallel_execution_plan.md`](solver/docs/parallel_execution_plan.md)が所有する。

P14-Pのfocused correction後、regular 1Mの4-thread speedupは0.923x、1-threadはv20履歴比23.7%退行した。
field locator microkernelの約3.75xとは対照的に、proposal/enclosureのPython/NumPy調停がend-to-endを支配した。
Amdahl上も限定修正ではgateへ届かず、巨大融合kernelは物理ownerの分離を損なうため、case schema v2から
`resources.threads`を削除しcompiled single-thread engineへ一本化した。具体的な測定と再検討条件は
[`solver/docs/parallel_execution_plan.md`](solver/docs/parallel_execution_plan.md)を権威とする。

### 9.2 active粒子管理

- stateはSoA
- source scheduleを時刻順batchでactivate
- active indexを保持
- inactive比が閾値を超えたときだけstable compaction
- wall hit粒子はevent queueへ
- outputもtile/chunk単位

release cohortごとに全N boolean maskを生成する方式は、release時刻数に比例して悪化する。

### 9.3 JAX/GPU

JAXはregular grid、固定shape、wall分岐が少ないcaseから評価する。公式文書が示すように、既定
dtypeはfloat32なのでCOMSOL比較ではX64を明示する。

- [JAX X64](https://docs.jax.dev/en/latest/101/default_dtypes.html)
- [JAX control flow](https://docs.jax.dev/en/latest/201/control-flow.html)
- [JAX PRNG](https://docs.jax.dev/en/latest/random-numbers.html)
- [persistent compilation cache](https://docs.jax.dev/en/latest/501/compilation-cache.html)

compile時間を除いた数字だけを報告せずcold/warmを分ける。CPU/GPUでbitwise一致を一般要求
しないが、float64誤差、event順序、RNG identity、統計分布のbackend parityを試験する。

### 9.4 memory/output

現在状態だけなら100万粒子を数百MBに収められるが、全量時系列は巨大になる。
`1e6 particles × 1001 times × 73 values × 8 byte ≈ 584 GB`である。

標準成果物はfinal、event、選択probe、整数count/fluxとし、v0.1はJSON manifestとHDF5 epoch
segmentへchunked streamする。閉じたsegmentをatomic commitし、checkpointと`LATEST`で再開境界を
定める。Parquet/CSVは外部analysisのexport形式に限定し、coreで二つのstorage stackを保守しない。
現行cadenceは`W=macro_step_count+accepted_particle_pieces+candidate_queries+refinements`、
`T=max(2^20,128N)`で、engineがaccepted macro barrierでcommit要否を決める。cadenceはmanifest/resume identityへ入り、
output scheduleとslab幅には依存しない。output ownerは同期single-writerとしてsegment、A/B checkpoint、`LATEST`の
atomic persistenceだけを行う。cadence revisionは`cumulative_solver_work_v1`、resultは
`durable_segmented_result_v5`で、result/checkpoint schema 2は不変である。threadingや第二writerは再導入しない。
HDF5のchunked/resizable datasetは
[h5py dataset documentation](https://docs.h5py.org/en/stable/high/dataset.html)を参照する。

---

## 10. 内部静電場builder

熱流体場と少数のプラズマparameterだけから電場は一意に決まらない。少なくともPoisson方程式、
space-charge closure、material、電位/flux境界条件が必要である。

```text
-div(epsilon grad V) = rho_charge(V, plasma parameters, thermal-flow fields)
E = -grad V
```

nonlinear SASS等のclosureはsolver coreではなく独立`electrostatic_builder`に置く。成果物は完成した
potential/E field、入力closure、境界条件、非線形残差、mesh、version/provenanceである。外部
plasma場を読むCase Pと内部生成するCase Aは、その後は同じ`FieldSet` interfaceを使う。

builderとparticle solverを分けることで、Poisson収束失敗をparticle積分の問題と誤診断せず、
異なるclosureを比較可能にする。

---

## 11. 技術選択の要約

| 問題 | 推奨 |
|---|---|
| 外部solver照合 | 一般的な固定RK4等へ同じ条件を指定し、比較処理は外部toolで実施 |
| stiff drag | native modeで指数/ETD更新 |
| Brownian | SDE、counter RNG、Brownian bridge、統計V&V |
| 場 | primitive mesh-native reference＋誤差検証済みcache |
| mesh | topology ID、P1/Q1 shape function、strict-interior hint＋supported-containment BVH |
| 境界 | continuous first-hit、event log、wall law分離 |
| RZ→3D | 2πr surface weightとbasis変換をcoordinate layerに集約 |
| charge | plasma continuous/discreteとaerosol modelを別type化 |
| 内部電場 | 独立Poisson field builder |
| CPU | Numba SoA bounded slab＋flat event wavefrontのsingle-thread compiled engine |
| GPU | regular/static caseから後段追加、JAX X64明示 |
| 100万粒子 | streaming output、全N×T保持禁止 |

この選択は「最速の単一kernel」ではなく、物理を増やしても正しさの責務と性能経路を追えることを
優先している。高速化は同じV&V gateを通過した計算同士でのみ評価する。
