# 物理モデルと数値契約

この文書は、現行solverが実際に解く範囲とfail-fast条件をまとめたものです。入力列と
成果物は [`input_artifacts.md`](input_artifacts.md)、COMSOL adapterは
[`comsol_vv.md`](comsol_vv.md) を参照してください。

## 対象範囲

外部から与えた場の中で粒子centerを追跡する、一方向連成のLagrangian
point-particleモデルです。粒子は場を変更せず、粒子間衝突、粒子間Coulomb力、有限半径
での壁接触は解きません。壁との交差は粒子centerのtrajectoryで判定します。

状態は位置 `x [m]`、速度 `v [m/s]`、必要な場合は電荷 `q [C]` です。粒子数を
`N`、空間次元を `D` とすると、solverの位置と速度は `float64 (N, D)`、scalar状態は
`float64 (N,)`、particle IDは `int64 (N,)` です。maskは `bool`、field-support statusと
停止理由codeは `uint8` を使います。

有効な座標系は次の3つだけです。

| 座標系 | `D` | 状態のaxis順 | 備考 |
|---|---:|---|---|
| `cartesian_xy` | 2 | `(x, y)` | 2D Cartesian |
| `axisymmetric_rz` | 2 | `(r, z)` | no-swirl、deterministic、`r >= 0` |
| `cartesian_xyz` | 3 | `(x, y, z)` | 3D Cartesian |

## 運動方程式と抗力

solverは次の形をETD2で進めます。

```text
dx/dt = v
dv/dt = (u - v) / tau_eff + a_external
```

慣性質量の正本は常に `mass_kg` です。Stokesの基礎緩和時間は

```text
tau_stokes = mass_kg / (3 pi mu drag_diameter_m)
```

であり、慣性質量をdensityとdiameterから再構成しません。`drag_diameter_m` は抗力と
Brownian mobilityに使う流体力学径です。正の `density_kgm3` が明示される場合は、質量と
密度から材料等価球径

```text
physical_diameter_m = cbrt(6 mass_kg / (pi density_kgm3))
```

を一度だけ導出し、thermophoresis、DEP、lift、capacitance、charge collection areaへ使います。
densityが未指定の場合だけ、従来互換としてdrag径を物理径にも使います。`density_kgm3` は
buoyancy、pressure-gradient、virtual-massにも使う独立した材料物性です。

run準備時の `tau_stokes` はscalar gas粘度による参照値です。連続流則を使う各stageでは、
その位置・時刻の `mu` と同じ `mass_kg`、`drag_diameter_m` から基礎緩和時間を再構成します。
したがって粘度fieldがある場合はStokes、Schiller–Naumann、Stokes–Cunninghamの全てで
局所粘度が基礎項にも反映され、一定粘度なら参照値と同じ式・演算順になります。

COMSOLと同様、内部では連続流のdrag lawとrarefaction correctionを独立な因子として
合成します。既存case schemaは変更せず、公開名を次の組合せへ一意に写像します。

- `none`: 連続流則なし・希薄化補正なし。`tau_eff = +Inf` とするballistic分岐。
- `stokes`: Stokes則・希薄化補正なし。`tau_eff = tau_stokes`。
- `stokes_cunningham`: Stokes則・Cunningham補正。`Kn = lambda / diameter` とし、
  `Cc = 1 + Kn * (2.514 + 0.8 exp(-0.55 / Kn))`、
  `tau_eff = tau_stokes * Cc`。
- `schiller_naumann`: Schiller–Naumann則・希薄化補正なし。
  `Re = rho * diameter * |v-u| / mu`、
  `correction = 1 + 0.15 Re^0.687`、`tau_eff = tau_stokes / correction`。
  `Re >= 800` はruntimeでもエラーです。
- `epstein`: Stokes則・high-Kn Epstein補正。COMSOL defaultのdiffuse accommodation
  `delta = 1 + pi / 8` と
  `v_th = sqrt(8 k_B T / (pi m_g))` を使い、
  `tau_eff = 3 mass / (delta pi diameter^2 rho v_th)`。これはStokes緩和時間と
  Epstein slip factorの積を代数的に簡約した、粘度に依存しない同値式です。

内部因子はSchiller–Naumann則とCunningham補正も合成できますが、既存の公開名にはその
組合せを宣言する入力がありません。`Re` や `Kn` から自動選択するとcaseが指定した物理を
暗黙に変えるため、現行schemaでは自動適用しません。

壁面近傍のStokes drag correctionは未実装です。COMSOL相当の補正には各stageでの粒子中心から
壁までの距離、壁法線、物理半径と、法線・接線方向へ分解した相対速度が必要ですが、現在の
compiled motion backendは距離と法線を受け取りません。衝突判定用のpoint-particle geometry
だけから等方的な係数を推定することはしません。

必要なmass、diameter、密度、粘度、温度、分子質量は有限かつ正でなければなりません。
不正値をepsilonへ切り上げたり別dragへ切り替えたりしません。
宣言済みのgas density、viscosity、temperature fieldはbackend compile時に有効support全体を
一度検証し、2D/3D、fast/preciseのどの実行経路でも不正値をscalar gasへ補完しません。
quantity自体が宣言されていない場合だけ、明示されたcontext gas値を使用します。

`validate_case()` は実際に `t_end` より前にreleaseされる粒子についてrelease位置・時刻の
場をsampleし、`Re`、`Kn`、明示された場合だけrelative Machを調べます。このためactive
dragのpreflightを完全に通すには、式が直接使う最小入力に加えて
`density_kgm3`、`dynamic_viscosity_Pas`、`temperature_K`、
`molecular_mass_amu` がscalarまたはfieldとして必要です。音速は
`sound_speed`、`speed_of_sound`、`c_sound` のいずれかがfieldにある場合だけ使い、標準
空気から推定しません。

適用域の判定値は次のとおりです。modelを自動選択する閾値ではありません。

- Stokes / Stokes–Cunningham: `Re >= 0.1` warning、`Re >= 1` error。
- Schiller–Naumann: `Re >= 800` error。
- rarefaction補正のないStokes / Schiller–Naumann: `Kn >= 0.01` warning、
  `Kn >= 0.1` error。
- Stokes–Cunningham: `Kn >= 10` でEpstein適用検討のwarning。
- Epstein: `Kn <= 1` error、`1 < Kn < 10` warning。
- 音速が明示されたdrag: `M_rel >= 0.3` warning、`M_rel >= 1` error。

preflightの範囲は `initial_release_state` です。全trajectoryの適用域を保証するものでは
ありません。runtime hard guardはSchiller–Naumannの `Re < 800` です。

## ETD2、時刻、release

積分器はETD2に固定され、runtime selectorはありません。各substepの開始、half-stage、
終端のscheduleは [`integrator_common.py`](../particle_tracer_unified/solvers/integrator_common.py)
が共通に生成します。線形dragは指数関数で進め、`dt/tau` が小さい場合は
`expm1` と級数を使って桁落ちを避けます。非正または非有限な `tau_eff` は計算継続せず
非有限結果として検出されます。明示的な `drag: none` の正の無限大だけがballisticです。

各leafでは、開始値と中点値からtarget velocityと外力を時間線形に再構成し、そのforcingを
指数関数内で解析積分します。これにより `dt/tau` が大きい時間変化場でも、中点値を凍結する
旧方式のstiff order reductionを避けます。定数係数では従来のexact exponential分岐を同じ
演算順で使います。係数を評価する予測中点と、衝突判定へ渡す補正済みhalf-stage軌道点は
別に保持し、それぞれの責務を混在させません。

adaptive substepは1 full stepと2 half stepsの差から位置・速度のlocal truncation errorを
評価し、必要な場合だけschedule全体を二分します。分割回数の上限は `time.max_substep_splits`
で宣言し、既定は4、すなわち最大16 substepです。上限は0以上12以下で、substep予算は
`2 ** max_substep_splits` になります。上限schedule自体も誤差評価します。上限でも未収束なら、
その終端を衝突判定や状態commitへ渡さずnumerical stopとします。debugの
`adaptive_substep_limit_reached_count` は上限scheduleの使用数であり、精度達成を意味する値では
ありません。

滑らかな自由飛行を解像する分割数と、シース通過や壁接近を解像する分割数は同じではないため、
この予算はcaseごとの明示入力です。既定値を暗黙に引き上げることはしません。

substep長は補間セルの大きさでも制約します。regular gridの最小軸間隔、triangle meshの
candidate-grid cellが `interpolation_resolution_m` であり、1 leafがこの長さを超えると
セル・要素境界をまたぎます。そこで補間はC0でしかなく加速度の微分が跳ぶため、
step doublingが前提とする `O(h^3)` の滑らかさが成立しません。したがってleafがこの長さに
収まるようsubstep数を引き上げます。ただしこれは精度要求であり、valid-mask islandの探索
（安全要求）とは別扱いです。予算を使い切っても粒子を停止させず、既存のLTE判定と
geometry/support判定だけがfail-closedの根拠です。

名目上のglobal step境界はrelease eventがあっても変えません。step途中でreleaseされた
粒子はrelease時刻から同じstep終端までを別cohortとして積分し、既にactiveな粒子は名目
step全体を積分します。このため、一方向couplingのcaseへ無関係な遅延release粒子を追加
しても既存粒子のfield sampling時刻は変わりません。release前の粒子は移動・帯電せず、
Brownian乱数も消費しません。`t_end` でreleaseされた粒子は積分しません。

transient fieldは、実際に積分される最早release時刻から `t_end` までが、すべての
transient quantityの共通support内にあることを `load_case()` と `simulate()` で確認
します。samplerのendpoint clampを範囲外物理の代用には使いません。

## field samplingとshape

solverがfieldへ渡す位置は `float64 (N, D)`、時刻は有限scalarです。sample結果の先頭axis
は必ずparticle数 `N` で、代表的なsemantic shapeは次のとおりです。

- scalar: `(N,)`
- vector: `(N, D)`
- velocity gradient: `(N, D, D)`
- vorticity: `(N, 3)`
- support: `bool (N,)`

force pipelineは必要なquantityごとにshapeとfinite/positive ruleを検証します。例えば位置、
速度、electric field、gradientはfinite、mass、diameter、必要な密度・粘度・温度はfinite
かつstrictly positiveです。shape不一致、`NaN`、`+/-Inf` をbroadcastやzero置換で修復
しません。

例外は省略可能なparticle overrideです。CSV列自体を省略した
`dep_particle_rel_permittivity` と `thermophoretic_coeff` は、内部では `NaN` を
「overrideなし」のsentinelとして使えます。列を明示した場合は有限値必須で、0、負値、
無限大はsentinelではありません。

regular gridは補間stencilの全nodeがvalidな `CLEAN` だけをsupportとします。pointがmask内
でもstencilにinvalid nodeを含む `MIXED_STENCIL` は、fill値を混ぜずfield-support停止へ
送ります。範囲外とhard-invalidも同様です。triangle fieldはmesh外をunsupportedとして
返し、zero fieldへ置換しません。

support判定と値の取得は別の問いです。support判定は厳密で、mesh外の点は
`HARD_INVALID` のままです。一方、値の取得はmesh外で最近傍要素へ落とし、barycentric
weightをsimplexへclampして正規化します。壁を越える試行stepは必ずmesh外へ出るため、
値がNaNになるとその壁hitを局所化する前に軌道が非有限になり、壁へ到達したはずの粒子が
numerical stopになります。regular gridが範囲外で軸補間を端点clampして試行を有限に保つのと
同じ意味論を、meshにも与えたものです。clampされたweightは凸結合なので、値はその要素の
値域を出ず、外挿にはなりません。物理状態がそこでcommitされることはなく、壁hitか
support停止のどちらかが必ず終端を決めます。最近傍探索はacceleration cellの近傍数リングに
限定するため、meshから十分離れた点は解決されずhard support失敗のままです。

accepted substepはhalf-stageとendpointをordered traceとして保持します。曲線のsagittaや
valid-mask islandを現在の分割で解像できなければrefineします。上限までrefineしてもriskが
残る場合はfail-closedとし、geometryは `numerical_boundary_stopped`、supportは
`invalid_mask_stopped` になります。

## 外力

fieldが存在するだけではforceを有効化しません。dynamic chargeを有効にしてもelectric
forceは自動で有効になりません。実装済みの追加forceはelectricとgravity、experimental
扱いのthermophoresis、dielectrophoresis、Saffman lift、pressure-gradient、virtual-mass
です。native/COMSOLの入口で同じimmutable force modelへ変換し、solver loopは設定辞書を
再解釈しません。

forceの入力不足はpreflightまたは最初の評価で失敗します。主な依存は次のとおりです。

- electric / DEP: 次元分のelectric-field component。
- thermophoresis: temperature fieldと必要なthermal property。
- lift: flow velocity、gas density/viscosity、vorticity。
- pressure-gradient: flow velocityまたはfluid material accelerationとgas density。
- virtual-mass: flow velocity、time derivative、velocity gradient、gas/particle density。

continuum thermophoresisは `Lambda = k_gas / k_particle` としてCOMSOLと同じ
`-6 pi d mu^2 Cs Lambda / (rho (2 Lambda + 1) T) grad(T)` を使用します。
Saffman liftのslipは `u_fluid - v_particle`、vorticityは通常の `curl(u)` とし、
向きを `(u_fluid - v_particle) x vorticity` で定めます。

AC DEPの `electric_field_amplitude` は `rms` または `peak` を明示できます。既存caseの
defaultは `rms` で、`peak` を指定した場合は同じ `grad(|E|^2)` 入力をRMS相当へ換算する
ため係数を1/2にします。解決した振幅規約はforce runtime summaryへ保存されます。
- buoyancy: 正のgas densityとparticle density。

COMSOLとのfaithful比較で数値微分誤差を避けたい場合、gradient、vorticity、fluid
acceleration、time derivativeをsemantic quantityとして直接exportする必要があります。
regular/triangle上で再微分したfallbackは、元のFEM導関数と同一とはみなしません。

## 境界と衝突

2D edgeまたは3D triangleとの最初のcrossingをtrace全体から選び、hit時刻の位置・速度へ
wall lawを適用します。非terminal hitの残り時間は同じsegment primitiveで再生し、複数hit
も時刻順に処理します。1 step当たりのhit上限は `physics.wall_interaction.max_hits_per_step`
で宣言し、既定は5、範囲は1以上64以下です。残り時間があるまま上限へ達した場合は
numerical stopです。多数のbounceを想定するcaseは、切り詰めたstepを受け入れるのではなく
この予算を引き上げてください。

同一壁部品への反射が1 macro step内で2回起きた粒子は、既定ではその壁へ固定され接線方向へ
進む持続的接触状態へ移ります。これはwall lawではなく、壁へ押し付けられた粒子のZeno的な
反射列を有界にするための数値的装置です。COMSOLの粒子追跡には点粒子の接触modelがなく、
個々のbounceを解き続けます。したがって
`physics.wall_interaction.contact_sliding: false` を宣言すると、この装置を使わずbounceを
予算まで解き、予算切れは黙って別modelへ切り替えるのではなく可視のnumerical stopになります。
`comsol build-case` が生成するcaseは既定で `false` です。

runtimeのstickingとdiffuse reflection乱数は、公開設定のseed、particle ID、macro step、
粒子ごとのwall-response cohort、step内wall event番号、draw種別で分離します。無関係な
粒子の追加や入力行の並べ替えは既存粒子のwall outcomeと反射速度を変えません。確率0または
1の判定と、stickingを持たないspecular responseは乱数を生成しません。

境界の数値長さはgeometryから一度だけ解決します。`L_res` は正のgrid spacing、2D edge
長、3D triangleの辺長・高さの最小値です。座標の大きさを `X` とすると、

```text
roundoff = 64 * spacing(max(L_res, |X|))
classification_tolerance = max(1e-10 * L_res, roundoff)
contact_offset = max(1e-8 * L_res, 8 * classification_tolerance)
```

同じtoleranceをRZのaxis判定にも使います。`contact_offset >= 0.01 * L_res` になる条件の悪い
座標は、人工的な大移動を入れずにエラーとします。解決値とpolicy versionは
`execution.numerics.boundary` に保存されます。

hit localizationの時間roundoffは
`64 * eps(float64) * max(abs(reference_time), abs(interval))` を基礎に、segment相対tolと
境界位置tolを併用します。固定秒の絶対tolは使いません。

## Dynamic charge

modeは `te_relaxation` と `oml_linearized_relaxation` です。electric forceを使わない場合は
half-charge / ETD2 motion / half-charge のStrang分割で結合します。dynamic chargeとelectric
forceを同時に使う場合は、各accepted leafで補助ODEの `q` と運動の `(x, v)` を同じ時刻列上で
2次連成します。1 full stepと2 half stepsの差で `x / v / q` を一緒に制御し、未収束のleafは
状態へcommitしません。

OMLは負電位区間でcurrent-balance rootをbracketして二分探索し、正規化current residual
`<= 1e-10` を必須とします。linearized relaxation timeは平衡点のcurrent derivativeと
capacitanceから求め、数値floorや設定上限へ丸めません。field backgroundではelectron
densityとion densityを別quantityとして要求し、`ni = ne` を仮定しません。ion-temperature
quantityを指定した場合、そのfieldが欠けても設定constantへfallbackしません。
charge fieldがunitを宣言している場合、electron temperatureは設定した `eV` または `K`、
electron/ion densityは `1/m^3`、ion temperatureは `eV` と一致しなければ積分前に失敗
します。unit metadataを持たないlegacy/native NPZだけは設定unitをauthorityとして扱います。

terminal event、valid-mask prefix、壁反射後の残時間も同じaccepted charge-motion traceから
評価します。このためwall hitでは、位置・速度・電荷を同一のevent時刻で確定します。現在の
連成経路はCartesian 2D regular fieldのdeterministic free-flight/collisionに限定し、Brownian、
persistent contact sliding、axisymmetric RZとの同時使用は、電荷を凍結した経路へ黙って戻さず
入力時に拒否します。

## Brownian motion

実装は `underdamped_langevin` です。fluctuation–dissipationと線形OUが整合する
Stokes、Stokes–Cunningham、線形Epsteinだけを許可し、速度依存Schiller–Naumannは乱数を
消費する前に拒否します。各accepted leafで係数を評価した同じ予測中点から温度、密度、
決定論ETD2の `tau_eff` を取り、
区間内constant-coefficientのintegrated Ornstein–Uhlenbeck過程を進めます。virtual mass時は
`m_eff = m_p (1 + C_vm rho_g / rho_p)` として速度分散を `k_B T / m_eff` にするため、長時間
拡散係数は追加質量で不当に増えません。決定論軌道が静止していても温度・密度・粘度・緩和
時間が変化する場合を見落とさないよう、integrated-OUの遷移と共分散を1 full scheduleと
2 half scheduleで比較し、必要な粒子だけleaf数を増やします。上限scheduleでも未収束なら乱数を
生成せずnumerical stopとします。refine、valid-mask prefix、wall後のpartial replayでは保存した
同じ確率pathを再利用します。

`temperature_source: field_T_then_gas` はtemperature quantityが存在する場合、そのfield値
を唯一の温度源にします。field値が非有限または0以下でもgas constantへfallbackしません。
quantity自体が存在しない場合だけgas temperatureを使います。`temperature_source: gas` は
常に明示されたgas temperatureです。

Brownian乱数は公開設定のseed、particle ID、macro step、粒子ごとの確率path番号、accepted
leaf、空間成分、draw種別で分離します。このため同じcaseとseedでは再現でき、無関係な粒子の
追加や入力行の並べ替えは既存粒子のstreamを変えません。restart機能は未提供です。将来
restartを追加する場合はmacro stepと粒子ごとのpath番号も状態として復元する必要があります。
Brownian wall crossingは、粒子IDに結び付いたquery-order independentなdyadic OU bridgeを
壁近傍だけ再帰分割します。各区間は端点・中点のclearanceがgeometry toleranceと条件付き位置
標準偏差の8倍を上回る場合だけ安全とし、それ以外は最初の壁eventを時間順に探索します。
探索点はvalid-mask判定にも含め、最大深さでも安全性を確定できなければnumerical stopとします。
これは固定8 nodeのpolylineより壁通過の見落としを大幅に減らしますが、有限回で数学的に厳密な
continuous first-passageを保証するものではありません。

## axisymmetric RZ

RZは `(r, z, vr, vz)` のdeterministic no-swirlに限定します。BrownianとCartesian liftは
設定時に拒否します。axis横断はsigned radial chartで積分し、chart radiusが負なら
`(r, vr) -> (-r, -vr)` でcanonicalな状態へ写像します。両端点がaxis tolerance内の
`r=0` primitiveはwallではなく座標軸としてcollision対象から外し、`r>0` のmaterial
boundaryには従来のwall lawを適用します。
実壁上のpersistent contactがaxis端点へ達する場合は、axis時刻での再積分が未実装のため
対称継続せず既存のendpoint holdとなります。

`2 pi r` のring weightはreport用helperとして存在しますが、source粒子数やforceへ暗黙適用
しません。RZの入力shapeは2Dですが、Cartesian 2Dと列名・物理意味を混在させません。

## tolerance、NaN/Inf、再現性

数値契約は用途ごとに分け、repository全体へ一つの `rtol/atol` を適用しません。

- serialized float64の同値判定: shape一致、全値finite、各値64 ULP以内。geometry/field
  axisとCOMSOL time-support端点に使用します。
- 境界分類: geometry scaleと64 coordinate ULPから上式で解決します。
- 時刻roundoff: 64 machine-epsilon相当を時刻scaleに掛けます。
- OML current balance: 正規化残差 `<= 1e-10`。
- 物理V&V: quantityごとの `rtol/atol` をtest側で明示します。shape、dtype、unit、axis順の
  不一致をtoleranceで吸収しません。

入力の非有限値、force入力のshape不一致、clean support内の非有限field、非正の物性は
fail-fastします。低level式が返した非有限値も最終状態へ黙って流さず、例外、停止理由、
safety counterのいずれかで表面化します。JSON writerは `NaN` / `Infinity` を出力しません。

互換性の正本は、schema version 2、SI単位、axis順、array shape/dtype、artifact列、解決済み
numerical policyです。数値式や物理結果を変える修正は単なるrefactorと分け、既存の契約test、
reference比較、境界条件、複数seed testで意図を固定してから行います。

## 現在のV&V境界

- experimental forceの再微分fallbackはCOMSOL解析導関数との一致を保証しません。
- Brownian wall first-passageは8-sigmaの確率的解像基準です。有限深さで未解決なら停止しますが、
  数学的に厳密な連続first-passage samplerではありません。
- OMLの適用性そのものにはDebye length、collisionalityなど装置条件の別V&Vが必要です。
- densityが明示された球形粒子は流体力学径と材料等価球径を分離します。ただし
  capacitance径、熱泳動径、分極体積径を個別には持たないため、それらが材料等価球径とも
  異なる非球形・多孔質粒子は対象外です。
- drag regimeの通常preflightはrelease stateのみです。全trajectory envelopeは比較・V&V側で
  評価する必要があります。
