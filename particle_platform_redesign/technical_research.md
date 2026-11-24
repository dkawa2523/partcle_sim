# 技術調査：物理・数値方式の根拠と拡張設計

本書は採用した式・近似・数値方式の導出、文献、代替案と拡張条件を保持する。
現行modelと適用域は[physics models](solver/docs/physics_models.md)、数値受理規則は
[numerics](solver/docs/numerics.md)、製品範囲と残件は[product specification](product_specification.md)・
[implementation plan](implementation_plan.md)が所有する。本書の研究条件を未実装のruntime契約へ読み替えない。

| 状態 | 保持する内容 |
|---|---|
| 実装済み | XY/RZ meridional、現行force/charge、midpoint joint OU、固定topology linear-time、field/cache producer、process実行 |
| 独立referenceとして実装済み | gas collisionのquadrature/jump/frozen oracle。非線形輸送・熱状態のproduction実装とは異なる |
| 採用済みの拡張方針・未実装 | RZ場＋Cartesian XYZ、別途の完全3-D mesh、nonlinear gas diffusion/熱状態の同state/event経路への接続 |
| 接続前に閉じる条件 | 適用域、熱・衝突仮定、保存則、高次moment、独立reference、座標/source/wall/output/checkpoint/capacityの縦切り |

過去の実行・最適化・資格は元source付きreceiptに限定する。現行revision roster、性能表、品質件数は本書へ複製しない。
COMSOLは対象問題と外部V&Vの参照であり、製品標準の式・数値方式を自動的に決める権威ではない。

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
<a id="adopted-unimplemented-3d"></a>

Brownianや方位運動を扱う拡張の採用方針は、2D RZ場を参照しつつ粒子をCartesian 3Dで積分する
`axisymmetric_field_cartesian3d`である。採用済みだがproduction未実装で、現在選べるmode/flagではない。
完全3-D geometry/field meshは別の縦切りである。現在のmeridional二自由度を物理3-Dと呼ばず、
採用方針を現在の実装範囲だけに縮めて取り消さない。

RZ swirlでは`v_theta`を加え、円柱座標の幾何項を含む別kernelが必要である。これは理論catalog上の
将来候補で、初期製品modeには含めない。`r=0`は物理壁ではなく座標の継ぎ目であり、軸横断の
basis変換をwall reflectionと混同しない。

2D軸対称surfaceから3Dの粒子fluxを復元するとき、線素`ds`は回転面積`2πr ds`を表す。
一様surface releaseを外部でrealizeする場合、2D edgeを等確率に選ぶのではなく、この面積で重み付けし、
必要なら方位角も一様に生成する。現行schemaは結果を粒子ごとのfacet ID、strict interiorのfacet内座標、速度、
release時刻、物性としてcanonical surface tableへ保存する。P14-Uの旧generator evidenceはXY line-length分布だけであり、
RZ回転面の分布品質を同じ証拠から主張しない。

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

`epstein_linear_effective_gas_sensitivity_v1`は既存線形式と同じformula ownerを使うが、producerが混合気体を
一つの有効Maxwellian/pseudogasへ畳み込んだ入力だけを受けるreference/sensitivity revisionである。caseは
`0 < maximum_speed_ratio <= 1`を明示し、`lambda/a>=10`とともにactual stageと既存runtime enclosureによる連続pathで
fail-closedに認証する。これは速度をclipするparameterでなく、species-resolved mixture truthでもない。
`epstein_linear_v1`の保守的上限`0.1`は変更しない。

複数の中性気体modelを合成する時、各modelが重複して宣言する速度、密度、温度、粘度、平均自由行程、分子質量は
一つの背景authorityへ一致させる。linear EpsteinとWaldmann--Gallisを組み合わせる場合、native single-species同士または
effective-gas sensitivity同士だけを許し、同名fieldによる意味の混在を認めない。有限速度Epsteinは一つの局所Maxwellianを
仮定するため、producerが同じsingle-speciesまたはpseudogas背景を認証したWaldmann revisionと組み合わせられる。

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

採用したoptional revisionは`quasistatic_spherical_gradient_e2_v1`とし、runtime式を次へ固定する。

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

single-speciesのproduction revisionは、自由分子領域の単一気体についてWaldmann--Gallisの局所熱流束形を使う。

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

`waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1`は同じWaldmann--Gallis heat-flux式を
再利用し、producerが認証した一つの有効Maxwellian/pseudogasだけを入力authorityとする。`q_eff [W/m^2]`はproducer所有の
有効並進伝導熱流束であり、total/convective/radiative/electron/ion heat fluxではない。coreはtemperature gradientを
回復せず、species配列やmixture ruleを持たない。適用域は球形、dilute one-way、`lambda/a>=10`、case明示の
`0 < maximum_speed_ratio <= 1`で、actual stageと連続pathをfail-closedに検査する。単一気体revisionの上限`0.1`は維持する。
このrevisionはreference/sensitivityであり、species-resolved mixture truthやCOMSOL専用branchではない。

外部producerの有効熱流束が無ければ、このmodelを保存データから自動的に回復できない。
適用域外の保存観測を理由にcoreの閾値を緩めず、[data quality assessment](data_quality_assessment.md)のdataset別制限を使う。

明示的な感度式は、`rarefied_vorticity_sensitivity_rz_v1`として次へ固定する。

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

適用域が異なる二つのmodelを、既存categoryの明示optionとして区別する。Talbotは原著の半径Kn規約へ
修正した`talbot_cross_regime_radius_knudsen_v1`を使う。`a=d_drag/2`、`Kn_R=lambda/a=2*lambda/d_drag`、
`Lambda=k_g/k_p`として、

```text
F_th = -(12*pi*a*mu^2/rho)
       * Cs*(Lambda + Ct*Kn_R)
       / ((1 + 3*Cm*Kn_R)*(1 + 2*Lambda + 2*Ct*Kn_R))
       * grad(T)/T
```

とする。これは`6*pi*d_drag`をprefactorに使う形と同値であり、慣性mass、electrostatic半径、contact半径は
再構成しない。原著のmean free pathは粘性基準`lambda_old=2*mu/(rho*c_bar)`であり、原著係数を未換算で使うproducerの
primitiveはこの定義に対応する。中性気体model間で一つの`lambda_common`を共有する時、選択した全domain・時刻で
`alpha=lambda_old/lambda_common`が一定であることを証明できれば、caseの明示係数を`Cm_common=alpha*Cm_old`、
`Ct_common=alpha*Ct_old`、`Cs_common=Cs_old`とすることで`Cm*Kn`と`Ct*Kn`を保存し、同じ力式を表せる。
両規約・元と換算係数・比の根拠・独立point-force確認をproducer provenanceへ残す。比が空間・時間で変わる場合は現在の
scalar係数で原式を全域保存できず、明示した近似または非対応scopeとして扱う。coreはlambdaを二倍に加工せず、sample済み値を
そのまま半径Knへ使用する。Allen–Raabeの固定係数を使う時はその校正air規約を共通primitive側で保持し、Talbot側を
必要に応じて換算する。`mu=0.491*rho*c_bar*lambda_common`と同じprimitiveから作る場合だけ比は0.982であり、別の
校正`lambda0`や温度・圧力lawへ無条件に適用しない。
`Cs,Cm,Ct`はcase必須で、pure evaluatorにもdefaultを持たせない。原著の推奨値は`1.17,1.14,2.18`である。
`1.17,1.146,2.2`はCOMSOL記載のcoefficient variantとして選択できるが、原著の推奨値と同一とは呼ばない。
原著の理論背景の脚注に従い、多原子気体の原著再現では`k_g`を並進熱伝導率として用意する。native式が全熱伝導率を使う
場合はその式の再現と原著Talbot再現を区別し、実際のfield定義・model-form差をproducer provenanceへ残す。coreは並進成分を
推定せず入力値を使う。producerが同じ温度solutionから`grad(T)`を形成する。v1は有限gradientと正の有限scalar primitive・係数を要求し、
`Kn_R`の数値cutoffは設けない。これはcross-regime model-formの明示選択であり、任意のKnでの物理検証を
主張するものではない。旧径Kn revisionの自動変換や再解釈は行わない。
Saffmanは`w=u-v`、`omega=curl(u)`、`L=w cross omega`として
`F_L=6.46*a^2*L*sqrt(mu*rho*|w|/|L|)`を平面XY/RZへ射影する。ゼロ`L`は極限のゼロである。
原式の連続体・creeping-flow階層を実行時に明確化し、`Kn_a<=0.1`、`Re_s<=0.1`、`Re_G<=0.1`に加え、
非ゼロshearでは`Re_s<=0.1*sqrt(Re_G)`を要求する。vorticity intervalがゼロを跨ぐ場合、非ゼロslip pathは
証明済みと扱わない。gradient/vorticity回復、far-wall・非回転・no-slip仮定の妥当性はproducer/case責務であり、
coreは微分、blend、near-wall補正を行わない。

Waldmannの自由分子式は[Waldmann (1959)](https://doi.org/10.1515/zna-1959-0701)、局所熱流束形と低圧・
壁近傍への一般化の背景は[Gallis, Rader, and Torczynski (2004)](https://doi.org/10.1080/02786820490490001)
を参照する。Talbotの式と半径Knは[Talbot et al. 著者preprint, Eq. (15)](https://escholarship.org/content/qt22f5r6cz/qt22f5r6cz.pdf)
をauthorityとする。[COMSOL Thermophoretic Force](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.48.html)
は係数variantの参考であり、説明図だけから実行RHSとのparityを主張しない。Saffmanは
[COMSOL Particle Motion in a Shear Flow](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.42.html)
および原論文を参照する。入力不足または適用域外で別modelへ黙って切り替えない。

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

比較可能な式を保持するため、これらを二つのoptional revisionとして独立に定義する。

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

stationary OMLのmodelは`oml_stationary_maxwellian_debye_huckel_v1`とする。半径`a`の球について
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

shifted OMLは、単一・単価正イオンの有限bulk driftを扱う負電位限定revisionである。設定上の

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

同条件比較には、保存されたaggregate式を独立に定義した
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
overflow repairやsilent fallbackではなく、このrevisionの明示的なmodel-formである。stationary/shifted OMLの式と適用域は変更せず、
同じstageの`x,v,Z`で積分し、有限boundを認証できない場合は比較のために安全gateを緩めず実装を停止する。

三電流modelは、この二電流revisionを変更せず、aggregateな単一価負イオン収集電流を一つだけ加える
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
Case-P same-form anchorは二電流式の比較として解釈し、三電流revisionをその不具合修正や遡及的な再認定に使わない。
負イオンを含めたcompanionは、元の意図的な二電流referenceとは異なる問題である。
外部の有限trajectoryの対応を、species-resolved物理、native FE、Brownian、任意条件の普遍資格へ拡張しない。
実行source・参照・resolutionと判定は[V&V methodology](vv_methodology.md)に従う元receiptへ残す。

deterministic/stochastic exponential midpointは、midpoint predictorで`G_mid=R_Z(Z_mid)`と
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
別modelとして扱う採用方針であり、整数jumpはproduction未実装である。接続時は電子/イオン到来率をhazardとしてイベント生成し、Brownianとは
別RNG streamを持たせる。

### 4.3 COMSOLのcharge modelとの区別

COMSOLのCharge Accumulationに含まれる空気aerosol向けmodelは、プラズマOMLと同じではない。
公式[Charge Accumulation](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.33.html)
のmodel名を、そのままプラズマ帯電の証拠にしない。現行fixed/continuous平均Zと、未実装の
discrete plasma/aerosol relaxationを別のmodelとして設計し、後者の名称を現在のselectorとして受理するとはしない。

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

速度とその時間積分を同時に扱う理由は、位置分散とx–v共分散が短時間輸送とfirst passageを左右するためである。
[Gillespie (1996)](https://doi.org/10.1103/PhysRevE.54.2084)は凍結線形OUのjoint simulationを与える。
mean/covariance、安定評価、条件付きGaussianの式は[numerics](solver/docs/numerics.md#inertial-brownian-integration)が所有する。
速度へnormal kickを加えて旧速度で位置を進めるだけでは正しいjoint lawにならない。
同一bath FDTと独立prescribed force spectral densityは異なるmodel authorityであり、観測軌道へfitした温度倍率で代用しない。

細分時に独立noiseを引き直すとevent探索や観測要求が別pathを作るため、親joint incrementを条件とするGaussian half-splitを使う。
右子を代数的残差で作って親を保存する。stateless key/counterの背景は[Random123 documentation](https://random123.com/releases/docs/)を参照する。
実装の物理identity、root/tree/stream割当はnumericsが所有し、配列順・worker数・保存頻度をauthorityにしない。

有限depth Hermite numerical pathのearliest hitと、sampleしていない連続OUのfirst passageは別である。
base depthは全rootの解像度、adaptive depthは壁・axis・証明不能候補の追加解像度であり、前者の不足を自動補償しない。
RMS距離やprobability thresholdを決定論的clear証明として使わない。

state-dependent fieldではnoise-free coupled predictorの実midpointでgamma,u,a,G,Jを凍結し、joint OUとaffine chargeを同じrootへ適用する。
これはhalf stateを先にcommitするoperator splittingでも、任意nonlinear SDEのexact法でもない。
wall/axis/periodic hitではaccepted prefixをcommitし、post-stateから残時間をfresh rootとして再開する。
同一root内のconditional splitと、物理event後のrestartを区別する。

独立Green-kernel、Gaussian conditioning、行列指数、Kramers referenceは、凍結係数exactness、
時間/位置依存の限定manufactured case、有限pathの内部観測と到達分布を検査する。
resolution・統計幅と資格限界は[numericsの独立検証](solver/docs/numerics.md#independent-verification-and-limits)と元receiptへ残す。
shared arithmetic parityだけで独立精度を認定せず、一般SDEのstrong order/weak 2次、zero-miss、任意壁や物理3-Dへ拡張しない。

RZ meridionalは二自由度投影であり、等方3-D Brownianではない。XYZ拡張は前述の未実装方針である。
COMSOLの内部step/乱数割当を対応させるmicrocaseと、独立seedの集団observable比較を使い分ける。
同seedだけで同pathとせず、用途に適したmean/covariance/MSD/occupancy/fate/到達分布と不確かさで判断する。

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

規則格子への再meshは高速化cacheであり、reference fieldではない。field preprocessorのproducer、validation/support
具体的なalgorithm/configの契約は[field preprocessor README](solver/src/chamber_particles_tools/field_preprocessor/README.md)が所有する。staticと固定topologyのlinear-time nodal P1/regular/exact affine Q1から、
全cellが元supportで覆われるregular格子を認証する。sourceとtarget cellのcommon partitionを作り、
XYではdx dy、RZでは2πr dr dzで値と実際の補間basisの座標gradientの
relative L2を積分する。support境界の値誤差はdsまたは2πr dsで測り、RZ axisの測度は0とする。
これはcomponentごとの座標gradientであり、3Dの共変vector derivativeではない。
bilinear gradientの座標丸めは混合二階差による変化もallowanceへ含める。大きい座標原点でこの項を
落とすとtight budgetへ誤ってpublishできる反例を、独立なgradient norm sqrt(1/3)で検証・拒否した。

sample absolute maximumは記録するが、連続L-infinity上限とは呼ばない。正の微小patch、coverageの穴・
重複をepsilonで消さず、float演算で解像できないpatchはunresolvedとして公開しない。warped Q1 source、
部分supportのtarget、coverage不足、未解決reference下界は公開しない。validationのmemory上限、workspace row数、
patch work上限を事前指定し、超過時にも公開しない。solidを跨ぐ補間やNaN fillは禁止する。
coreのspatial gradientはstatic nodal fieldだけを扱い、値と同じcell/basis/Jacobianを使う。
外部preprocessorは各snapshotを同じownerのstatic viewとして評価し、時間Gramを外部で構成する。
時間区間の二乗error/reference normはquadraticとなるため、丸めを囲ったBernstein係数で正の分母下界と
比の上界を認証する。未解決区間は内部52段または資源budgetまで分割し、解決できなければ公開しない。
stationary pointはreport-onlyであり、合否gateを持たない。第二locatorや時間gradient engineは追加しない。

cacheの用途採用には、これらのfield指標とは別に、壁/sheath近傍、代表軌道・eventへの誤差伝播と
実測費用を評価する。field normだけで到達・付着精度やspeedupを認証しない。

### 6.4 時間依存場

現行revisionは固定topology、線形補間、範囲外errorに限定する。全snapshotをresident保持し、knotで
macro stepを分け、各integrator stageの実時刻で空間・時間を同時評価する。時間範囲外を端点clampしない。
snapshot自体の不足はparticle stepで修復できない。外部preprocessorは明示指定
`--source-time-diagnostic`でinterior knot omissionの感度をreportする。3点以上のsource snapshotについて、
各内部knotを除いた隣接時刻の不均一時間線形補間と元の値・gradient・support境界traceを、同じruntime samplerと
XY/RZ物理測度で比較する。2点はINSUFFICIENT、静的場はNOT_APPLICABLE、未指定はNOT_REQUESTEDである。
計算は既存資源budget内のreport-onlyであり、cache認証を先に完了する。診断の資源不足で適格cacheを拒否しない。
source knot間に未観測のpulseがあるかはNOT_TESTEDで、この感度から連続原場の時間adequacyを自動認証しない。
現行のtime cache認証は既存snapshot間の線形場に対する空間cache誤差を全区間で判定するものであり、
元データのsnapshot不足を認証・修復しない。hold/discontinuity、streaming/double buffer、moving topologyは
利用caseと独立referenceを揃えた別revisionとする。

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

現行は粒子ごとの独立`contact_radius_m`を持ち、0なら粒子中心のpoint contact、正なら静的2-D XYのdiskまたは
RZのsphere中心とmaterial `line2` segmentのcapsule first contactをeventとする。半径をgeometry tolerance、
drag径、静電半径で代用しない。finite-radius broad phaseはpath enclosureを半径だけ膨張し、narrow phaseは
segment本体と端capの最初の接触を求める。接触法線は中心からcandidate segmentの最近点への方向であり、cornerの
同時candidateを保持する。採用済み未実装の`axisymmetric_field_cartesian3d`ではCartesian pathを
`(r(t),z(t))`へ写した曲線とRZ断面境界を交差させ、RZ法線を衝突点thetaでCartesianへ戻す。

現行2-D/RZではgroupの`contact_geometry=particle_center`を明示した境界だけ中心通過で判定する。
physical contact radiusをゼロへ置換せず、材料端capと中心開口のfirst-hitを同じ候補/BVHで比較する。
exact linear/quadraticの接触起点では、convex capsuleの最近点（segmentへclamp）から得るsupporting-plane normalを使い、
quadratic normal displacementの導関数が区間両端で非正なら全区間の離脱を証明できる。これを一facetごとに適用し、
別facetの後続impactを抑制しない。反射後の解析originと有限contact-set離脱状態をmacro/checkpoint間で保持する。
曲線法の認証は従来の包絡に限定し、このexact証明で連続OUのfirst-passageや全curved endcapの離脱を認定しない。

壁物理はintersectionから分離する。現在の応答式・適用域は[physics models](solver/docs/physics_models.md#wall-response-laws)、交差・corner・離脱は[numerics](solver/docs/numerics.md#material-contact-and-wall-events)が所有する。以下は熱再放出と接触の採用根拠である。

- stick/absorb
- escape
- parameterを持たない非deposition terminal `hold`
- parameterを持たない完全鏡面`specular`
- 法線・接線反発係数を持つ`restitution`
- 壁温度、拡散混合率、接線wall frameを持つ`maxwell_thermal`
- 定数確率の`probabilistic_stick`と、`otherwise`へ明示する
  `specular | restitution | maxwell_thermal | isotropic` fallback

`hold`はhit位置、hit時速度、hit時電荷を保持するinactiveな`held` lifecycleであり、
その後の力、電荷、kinematicsを時間発展させない。paused/restart/resuspensionを意味しない。速度依存reboundと
resuspensionは後続catalogである。
`specular`は係数入力を受けず、法線成分を厳密に反転して接線成分を保存する。`restitution`は
`normal_restitution`と`tangential_restitution`をともに必須とし、確率lawの非stick側もdefaultで補わない。
`maxwell_thermal`はMaxwellの鏡面／完全熱適応拡散混合を瞬時wall responseとして実装し、有限半径でも
candidate固有接触法線を使う。rolling/slidingや持続接触を意味しない。
外向きunit normalを\(\boldsymbol n\)、\(\boldsymbol t=(-n_y,n_x)\)、
\(\sigma^2=k_B T_w/m\)とすると、拡散branchは独立な\(U_n\in(0,1)\)、\(Z_t\sim N(0,1)\)から

\[
c_n=\sigma\sqrt{-2\log U_n},\qquad c_t=\sigma Z_t,\qquad
\boldsymbol v'=\boldsymbol v_w-c_n\boldsymbol n+c_t\boldsymbol t
\]

を得る。従って\(p(c_n)=c_n\sigma^{-2}\exp[-c_n^2/(2\sigma^2)]\)、\(c_n>0\)で、接線は
Gaussianである。`diffuse_reflection_fraction`はこのbranchを選ぶ確率であり、一般のenergy accommodation係数ではない。
残りは同じwall frameで鏡面反射する。静的geometryでは`wall_velocity_m_s`をgroup内全facetへ接線方向だけに
制限し、法線wall motionをmoving geometryなしに偽装しない。XYとRZ meridionalの既存2成分stateを扱い、RZ則を
完全3-D方位Maxwell分布とは解釈しない。

wall RNGは`philox4x32_10_v2`で、付着、混合branch、open-uniform法線、standard-normal接線を独立streamへ
分け、`seed, particle_id, physical_boundary_event_ordinal, stream`から再構成する。scalar変換
`half_range_maxwell_flux_velocity`とNumba batch変換`half_range_maxwell_flux_velocity_batch`はstream選択policyを
持たない。surface sourceは粒子ごとのfacet、facet内座標、初速度、release時刻、物性をcanonical HDF5へ
realizeして与え、coreでsource RNGを引かない。入力作成側は同じpolicy-free変換を利用できる。

Maxwell混合則の原典はJ. C. Maxwell,
[On Stresses in Rarified Gases Arising from Inequalities of Temperature](https://doi.org/10.1098/rstl.1879.0067)
である。熱再放出と壁frame速度の比較対象はCOMSOL 6.4公式
[Thermal Reemission](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.05.html)とするが、
COMSOL bitwise parityや非公開sampling algorithmの一致は主張しない。Boltzmann定数はNIST CODATAの
[exact value](https://physics.nist.gov/cgi-bin/cuu/Value?k%7Csearch_for=Boltzmann)を使う。

有限半径接触は即時反射lawと別のgeometry/event処理である。surface chargingはwall electrical
stateとcharge-transfer event、resuspensionはwall-resident lifecycle/dwell/adhesion/release schedulerを必要とするため、
この即時反射lawへplaceholderを置かない。各後続revisionはstate/output/checkpoint ownerと独立referenceを先に固定する。
COMSOL固有のfreeze/disappearは外部adapterがraw stateを保持する。boundary 37/35へ確実に到達する正例microcaseで、
実際に出力されたhit後位置、速度、保存statusを確認して外部manifestの対応を定める。現referenceには
電荷列がないためCOMSOL電荷保持は認定せず、generic `hold`の非零電荷保持は解析・公開scenarioで独立に検証する。Freezeをstuckへ写像せず、
製品利用caseにも必要な非deposition終端はprovider非依存の`hold/held`として実装し、material deposition、escapeと
統計を分ける。COMSOL名やraw status codeはwall lawへ入れない。

境界応答は「最初のhit時刻」と「その時刻にincidentなfacet集合」を分けて扱う。各facetを同じpathと
scale-awareな局在budgetで独立に認証し、最初のhitと同時で共有nodeへ接続するfacetを全て保持する。
exact linear/quadratic、一般RK4、exponential midpoint、Brownian cubic Hermiteで同じcandidate tableと
priority/combined-normal則を用いる。曲線pathだけ最初に列挙された一面へ丸めず、候補の認証が閉じなければ
分割し、budgetを使い切った時はfail-closedとする。

event logは時刻、点、boundary ID、法線、pre/post速度、電荷、結果、乱数counter、局在残差を
保存する。保存時刻のstatus変化からeventを推測しない。

COMSOLの壁設定とwall accuracy orderは公式
[Wall](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.43.html)
を参照し、`tools/vv/comsol`の比較manifestへ写す。canonical caseには一般化したwall lawだけを
置き、corner、同時hit、step内最大interaction数も比較条件として外部に明示する。

---

## 8. 積分法

### 8.1 固定step reference mode

固定RK4は一般ODEのstage結線と時間収束を独立に検査しやすい。
候補のstepは自身のaccuracy/stability/applicabilityから選び、COMSOLの保存間隔をコピーしない。
元datasetが宣言した刻みと、後から同定した別native memberの内部刻みも区別する。
運動と電荷を同stageへ連成し、Brownian乱数のstage更新を推測して互換と呼ばない。

既知の線形dragに対しては保守的に`h/tau_min < 2.5`を要求する。これを満たさないstiff caseを
有限値のまま進めず、明示的にstepを小さくするかexponential methodを選ぶ。
一般RK4の数学的なHermite/state pathと受理endpointは固定し、dense位置評価と部分区間の位置認証を
root始点相対のBernstein制御点で行う。座標差はTwoDiffで丸め値と厳密残差に分け、評価時は両方を足して
world座標へ戻す。始終点は保存済みendpointで厳密に上書きし、enclosureの下限と上限はそれぞれ
`-inf` / `+inf`方向へ丸める。物理的な曲率とenclosureは局所変位scaleで計算するが、公開chord-deviation
boundは最終のworld座標変換、両endpoint、endpoint chord式の丸めを覆う狭い`8*eps`絶対座標termを
別に加えるため、戻り値自体を完全な平行移動不変とはしない。広い絶対座標paddingが二分後も十分縮まない問題を避ける。first-hit/event algorithmと
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

固定stepと別runのresolution評価を使い、field knot、event、個別release、観測clockの意味を分ける。
accepted path・outward enclosure・affine chargeと各methodのgateは[numerics](solver/docs/numerics.md)が所有する。
charge-only subcycle、支持外trialの受理、accuracy不足の壁到達への変換で不成立を隠さない。

### 8.3 reference solver

小粒子数の原因分析には、高精度な一般ODE solverとevent機能をreferenceとして使える。
[SciPy solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
はdense outputとeventを提供する。ただし本番の100万粒子backendにそのまま使うのではなく、
解析case・kernel検証のoracleとする。

### 8.4 収束

`dt,dt/2,dt/4`、必要ならmesh系列を計算し、位置・速度・電荷・event時刻/点の観測収束次数を
調べる。COMSOLとの差だけを見て係数を調整せず、各solverの自己離散化誤差、field export誤差、
event局在誤差を分けて評価する。誤差の説明と独立した用途許容を区別し、不確かさで合格幅を緩めない。

---

## 9. 実行・memory設計の根拠

SoA resident stateとbounded scratchにより、場・geometryを共有しながら多数粒子へ同じ数値ownerを適用する。
sourceは時刻順、active rowはstable ID順とし、release cohortごとに全N maskを作る仕事を避ける。
particle loopへPython callbackや可変object queueを置かず、flat wavefrontの各rowへtarget/depth/interaction/ordinalを保持する。
accepted state/hintとcandidate順を維持し、各roundで高々一event/failureをbounded stagingへ渡す。

一つのcompiled engineをserial/process共通transportで使う。processは粒子rowを排他的に担当し、親はclock/release/single writer/checkpointを所有する。
outer thread poolや第二schedulerを併存させない。配分、同期、memory peak、handoff/ACKと測定条件は
[CPU execution](solver/docs/parallel_execution_plan.md)が所有する。Numbaの背景は
[performance tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html)を参照する。

全N×T保持を標準にしない。例えば `1e6*1001*73*8 byte` は約584 GBであり、resident stateだけの容量とは桁が違う。
final/event/選択probeをstreamし、出力のため数値pathを変えない。schema、commit、hash、reader/recoveryは
[result format](solver/docs/result_format_v5.md)が所有する。owned-array planはOS RSSのhard capではない。

GPU/NUMA/out-of-coreは将来評価対象だが未実装である。JAX等を評価する場合は
[dtype/X64](https://docs.jax.dev/en/latest/101/default_dtypes.html)、
[control flow](https://docs.jax.dev/en/latest/201/control-flow.html)、
[PRNG](https://docs.jax.dev/en/latest/random-numbers.html)、
[compilation cache](https://docs.jax.dev/en/latest/501/compilation-cache.html)の意味を明示する。
compile・転送・起動・I/Oを除いたkernel値だけで用途性能を判断せず、新backendの精度・event順・RNG identityと分布を別途検証する。

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
plasma場の外部読込みと内部生成のどちらも、その後は同じcanonical field interfaceを使う。
現行builderの範囲は[builder README](solver/src/chamber_particles_tools/electrostatic_builder/README.md)が所有し、gas-only縮約を全材料／浮遊導体VQ solverと呼ばない。
全材料epsilon、界面flux、未知浮遊Vと総Q拘束、gas-side回復、charge/ion/DEP依存再生成は引き継ぐ未実装範囲であり、独立電場の採用目標から削除しない。

builderとparticle solverを分けることで、Poisson収束失敗をparticle積分の問題と誤診断せず、
異なるclosureを比較可能にする。

---

## 11. 技術選択の要約

| 問題 | 推奨 |
|---|---|
| 外部solver照合 | actual equation/field/source対応、独立数値計画、用途observableを外部toolで評価 |
| stiff drag | native modeで指数/ETD更新 |
| Brownian | SDE、counter RNG、Brownian bridge、統計V&V |
| 場 | primitive mesh-native reference＋誤差検証済みcache |
| mesh | topology ID、P1/Q1 shape function、strict-interior hint＋supported-containment BVH |
| 境界 | continuous first-hit、event log、wall law分離 |
| RZ→3D | 採用済み未実装のRZ場＋XYZ。2πr source measureとbasis/normal変換を保持 |
| charge | 現行continuous平均Zと未実装discrete/emission/熱状態を区別 |
| 内部電場 | 独立Poisson field builder |
| CPU | 一つのserial compiled transportと排他的particle process、親clock/single writer |
| GPU | 未実装。実用途・容量・転送・精度の条件を閉じて評価 |
| 100万粒子 | streaming output、全N×T保持禁止 |

この選択は「最速の単一kernel」ではなく、物理を増やしても正しさの責務と性能経路を追えることを
優先している。速度は入力・数値resolution・sourceと用途条件を対応させて評価し、科学精度の変更と実装効率を分ける。

<a id="adopted-unimplemented-gas-interaction"></a>

## 12. 採用済み未接続の非線形gas interactionと熱状態

本節は採用した拡張方針とproduction接続前に閉じる研究条件であり、現行catalogの受理条件を変更しない。
利用caseは有限gas-relative速度、微粒子の熱的揺らぎ、混合gas/非一様温度である。
元COMSOLにあるlinear dragとBrownianを再現する数学問題、物理的に妥当なcollision model、
実験/製造予測のvalidationは別に扱う。作業順とownerは
[implementation plan](implementation_plan.md)が所有する。

### 12.1 平均抵抗と速度拡散は一つのcollision仮定から求める

粒子質量をM、gas分子質量をm、粒子速度をv、gas分子速度をc、g=c-vとする。
半径aの球、表面の外向き単位法線n、数密度n_g、正規化速度分布f_g(c)に対して、
入射collision rateのmeasureは次である。

```text
dR = n_g*a^2 * (-(c-v) dot n)_+ * f_g(c) d^3c dOmega
A(v) = integral dR * E[Delta v | c,n]                 [m/s^2]
Q(v) = integral dR * E[Delta v Delta v^T | c,n]       [m^2/s^3]
```

これは衝突過程のfirst/second increment momentを定義する研究出発点である。
Qは速度covarianceの時間rateであり、Fokker–Planckのdiffusion tensor D=Q/2、
位置の拡散係数[m²/s]とは異なる。mixtureでは各speciesのn,m,T,drift,kernelを分け、
rateをsumする。単一effective massで常に同じA/Qになるとは仮定しない。

specular sphereの独立oracle候補では、reduced mass mu_red=mM/(m+M)とし、
Delta v=2*(mu_red/M)*(g dot n)*nを使う。nの解析積分から次を独立に導出した。

```text
A_spec = n_g*pi*a^2*(mu_red/M) * E_g[|g| g]
Q_spec = (n_g*pi*a^2/3)*(mu_red/M)^2 * E_g[|g|^3*I + 3*|g|*g*g^T]
```

この形はg=0でもintegrandが0となり、未定義のunit directionやepsilon補正を必要としない。
この解析積分とgas速度quadrature/collision sampleを照合する。
diffuse成分は粒子surface温度、outgoing flux law、recoil/accommodationのauthorityを要する。
current finite-speed mean lawと同じ仮定ならそのmeanを再現し、equal-temperature・低drift・heavy limitで
Q→2*gamma*k_B*T/M*Iを確認する。gammaは速度の線形緩和rate[1/s]、抵抗係数beta/Mである。
finite-speedでQが一般にlongitudinal/transverseに分かれるため、
抵抗係数をlinear FDTのscalarへ置き換えるだけの実装を採用しない。
式の詳細・独立導出・前提は
[physics設計レビュー](analysis/generic_improvement_2026_10_14/physics_design.json)に記録する。

### 12.2 SDEとして採用できる条件は別に検証する

一般のItô速度過程`dv=A dt+B dW`でQ=B B^T、D=Q/2とすれば、
Fokker–Planckは`-partial_i(A_i f)+partial_i partial_j(D_ij f)`である。
一様温度・gas rest frameでMaxwellianにzero probability currentを要求した場合に限り、
次の関係が必要になる。

```text
theta = k_B*T/M
A_i = partial_{v_j} D_ij - D_ij*w_j/theta,  w=v-u_g
```

これは本節でFokker–Planckから導いた整合性条件であり、物理的Qを一意に決めない。
isotropic D=theta*gamma(v)*Iならgammaのvelocity derivative項が必要になる。
PSD、平衡moment、上式の成立だけでは非平衡high-speed軌道、tail、rare arrivalを認定できない。
collision first/second momentが定義できることと、second-order Kramers–Moyal truncationによる
連続diffusion近似が十分であることは別である。有限mass ratioでもexact detailed balanceを無条件に主張しない。
この区別と高速粒子に対するFokker–Planck適用制限は一次論文でも扱われる。
[Trigger et al., velocity-dependent friction/diffusion, §III](https://arxiv.org/pdf/cond-mat/0401327)

production追加前に、独立jump/collision（必要ならDSMC）referenceと以下を比較する。

- mass ratio、drift、温度比、collision頻度/field変化時間、single-collision jumpと高次moment。
- A/Qのquadrature収束、低速漸近、energy relaxation、covariance、確率tailと目的のfirst arrival。
- 採用した連続近似の観測量別誤差、gas分布/熱流束closureの不確実性。

妥当な範囲を示せなければnonlinear SDEは未対応のままにする。平均抵抗だけを検証して使う
決定論的scopeは別に有用だが、揺らぎ無視の物理影響を認定済みとしない。
高drift熱泳動はgas分布のnonequilibrium correctionを必要とし、heat-flux一つから
任意速度の一意なforceが決まるとは仮定しない。Waldmannのcutoff拡大や熱泳動/noiseの暗黙削除で解決しない。

### 12.3 座標・積分・評価への接続条件

state-dependent Qには明示Itô/Stratonovich conventionと対応するdriftが必要である。
現在のmidpoint frozen-linear OUをnonlinear processのexact methodと呼ばない。
候補SDE schemeは同じstage/evaluation/accepted-state/event残時間/RNG ownerに統合し、
weak convergenceとboundary sensitivity、coupled noise incrementによる必要なstrong検証を行う。
sourceのthermal実現はruntime forcingとは別ownerである。

現行RZの二速度成分はmeridional dynamicsである。3D等方collision noiseや非零phi速度は
方位運動を生み、円柱慣性式ではdv_rにv_phi²/r、dv_phiに-v_r*v_phi/rが現れる。
単にphi成分を捨てると同じ問題ではない。採用したP17方針のRZ場/Cartesian XYZ粒子では、
軸で特異な円柱stateを追加せずfield基底・回転面normalを変換できる。
軸上・平均0のisotropic transverse Gaussian displacementについて、3Dのradial分布はRayleigh、
meridional投影はfolded-normalとなり、radial二次momentが異なる。
これは数学的microcaseであり、COMSOLの2DOF保存結果との一致からfull3D transportを認定しない。

source extractionのactive selector、frame、momentum/energy/角度samplingは
[COMSOL Release](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.17.html)、
[Inlet](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.19.html)を版付きで確認する。
これらを製品のsource分布の自動defaultにはしない。

### 12.4 独立衝突referenceの実装範囲と熱状態

外部`solver/tools/vv/gas_collision/`はMaxwellian incident gas、球面入射、outgoing fluxを
独立quadratureと衝突jump sampleで検証する。A/Pの574初期状態に対し、選択した
one-pseudogas・heavy sphere・Maxwell mixed・prescribed Tw=Tgの平均抵抗を確認した。
数値結果と入力hashのauthorityは
[scientific_conclusion.json](solver/evidence/generic_improvement_2026_10_14/gas_collision/scientific_conclusion.json)である。
方向別Q、frozen compound-Poissonのatom/tail、観測量scale別generator差も記録するが、
非線形軌道、連続diffusion、壁first passage、物理3Dのqualificationには拡張しない。

Tw=Tgは仮定であり、native `ComputeParticleTemperature=0`から導けない。
同じheavy diffuse kernelの内部heat input候補は次となる。

```text
P_internal = sigma*n_g*pi*a^2*(m/2)*E_g[|g|*(|g|^2 - 4*k_B*Tw/m)]
```

入射energyとflux-weighted出射energyを差し引いた式であり、zero slip/Tw=Tgでは0、
high slipのequal-temperature bathでは正となる。finite recoilや内部stateの補正は未検証である。
初期相対運動energy/massは熱への全変換量ではなく、熱分配前の利用可能energyである。
粒子heat capacity、初期温度、energy accommodation、熱交換closureが無ければ温度上昇を計算しない。
密度2200からmaterialやCpを推定しない。COMSOLのauxiliary温度計算にもCpとheat source/sinkが必要である
([COMSOL 6.4 temperature theory](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.52.html))。
prescribed温度下の決定論的平均軌道は有用な限定問題だが、熱的に自己整合した実粒子予測とは区別する。
