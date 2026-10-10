# 物理モデル・数値計算・COMSOL比較の再調査レビュー

2026年10月9日。結論は、**現行の2自由度粒子輸送ソルバーには妥当な数値基盤があるが、すべての物理モデルを原著と同等、または実チャンバーを定量予測できるものとして認定する根拠は揃っていない**、である。一次資料の再調査により、Talbot熱泳動のKnudsen数と係数の規約不一致、COMSOL比較の軸境界の意味不一致、補間後のBrownian摩擦係数の不一致を追加で確認した。これらは前回評価の限定を強める具体的な根拠である。

一方、shifted Maxwellian OMLの速度積分、連成RK4、固定係数のjoint OU、条件付き確率分割、Philoxの基本式には整合した独立証拠がある。モデル近似の誤差、数値積分の誤差、外部比較の意味の違いを分けて扱う必要がある。

## 1. 対象・根拠・判定方法

対象は`particle_platform_redesign/solver/`の現在の作業ツリー。HEADは`fa51c2e10f32153afbb08d54012ec676f70508c4`、package 0.2.0候補、engine v44、case/data/result schema 3、checkpoint schema 2を基準にした。既存の未コミット変更を含む。`old_code/`の環境・packageは使用していない。production code、既存の仕様文書、datasetを変更していない。

本レビューは[前回レビュー](expert_review_2026-10-09.md)と[改良設計](improvement_design_2026-10-09.md)を補足する調査記録であり、仕様の正本を置き換えない。採用する修正は`technical_research.md`、`physics_models.md`、`numerics.md`、`vv_methodology.md`などの各ownerへ反映する。

物理、数値、COMSOLの三分野を分担し、現行関数・compiled処理・prepare gate・検証コード・保存済み比較証跡を調べた。論文の著者原稿、大学の原資料、COMSOL公式6.4文書を再調査した。COMSOLの新規solve、実機との比較、現在の候補による100万粒子計測は行っていない。既存のCOMSOL証跡は、記録されたrevisionとscopeの結果として扱う。

| 判定対象 | 必要な証拠 | このレビューでの扱い |
|---|---|---|
| 式の実装 | 原著への同値変換、独立積分・解析解、compiled一致 | 個別に確認する |
| 物理近似 | 分布、collision、Kn/Re、表面則、対象の時間・空間scale | runtimeの数値gateだけでは認定しない |
| 数値解法 | 安定性、収束、確率moment、dense/event精度 | 各証拠の対象族に限定する |
| COMSOLとの一致 | 同じ場、粒子物性、力、電荷、release、座標、境界 | 設定と保存結果を照合する |
| 装置予測の妥当性 | 背景場、発生源、表面モデル、測定との比較 | 未認定 |

`PASS`はこの五つを一括して意味しない。式を正しく計算していても、採用した近似が対象条件に十分正確とは限らない。

## 2. 判断を更新する重要事項

| 優先度 | 確認事項 | 分類 | 結果への影響 |
|---|---|---|---|
| 高 | Talbotの半径Knと直径Knの係数bindingが原著と同値でない | 確認済みのモデル規約不一致 | 同じ入力でも熱泳動力が大きく変わる。対象軌道への寄与は未測定 |
| 高 | Case-A/PのCOMSOL軸境界はFreezeを継承、比較契約は座標通過を宣言 | 確認済みの外部比較設定不一致 | 軸接触が起きるcohortへ比較認定を拡張できない |
| 高 | P1補間後のBrownian側摩擦係数とcandidate dragが一致しない | 確認済みの宣言式の不一致 | 保存H5の3779重心sampleで最大約0.461%。集団・軌道への影響は未測定 |
| 中 | Barnes effective-speed近似は無電荷・弱流速のcollectionを約25%過小評価 | 近似モデルの確認済み誤差 | applicabilityが真でも厳密なMaxwell平均ではない |
| 中 | aggregate帯電は静止OMLへ厳密には還元しない | 感度モデルの確認済み性質 | 零流速の引力増分係数がOMLのπ/4 |
| 中 | RK4 bulk、dense、wall event区間、Brownian first passageの保証が異なる | 数値精度のscope | 一つの次数・toleranceで全誤差を表せない |

### 2.1 Talbot熱泳動：原著との規約不一致

[現行処理](../solver/src/chamber_particles/physics/forces.py:259)と[現行仕様](../solver/docs/physics_models.md:1022)は、`Kn_d=λ/d`に`Cs=1.17, Cm=1.146, Ct=2.2`をそのまま適用する。pure/compiled、設定伝播、同じ式のscalar oracleの整合は確認されている。しかし、これだけでは原著Talbotの式への同値性を確認したことにならない。

Talbotらの著者preprintの式(15)は、粒子半径`R`、`Kn_R=λ/R`を用いて

\[
F_{th}=-12\pi\frac{\mu^2R}{\rho}
\frac{C_s(\Lambda+C_t\lambda/R)}
{(1+3C_m\lambda/R)(1+2\Lambda+2C_t\lambda/R)}
\frac{\nabla T}{T},\qquad \Lambda=k_g/k_p
\]

と書かれる。原著の係数は`Cs=1.17, Cm=1.14, Ct=2.18`。`R=d/2`へ変換すると、prefactorは現行と同じ`6πd μ²/ρ`になる一方、補正関数の引数は`2Kn_d`になる。同じ式を直径Knで書くには`Cm,Ct`も2倍する必要がある。原著の平均自由行程は`λ=2μ/(ρ c̄)`であり、理想気体のCOMSOL公式の粘度基準λと一致する。平均自由行程の定義差によってこのfactor 2を消すことはできない。[Talbotらの著者原稿、PDF index 7の脚注・index 16の式(15)、閲覧表示8/17頁・式の印刷頁12](https://escholarship.org/content/qt22f5r6cz/qt22f5r6cz.pdf)、[COMSOLの平均自由行程の定義](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.37.html)

係数の小さな数値差とKn規約の差を分けるため、現行の同じ係数を両規約へ使い、`Λ=0.01`で独立に再計算した。

| Kn_d | 現行力／同じ係数の半径規約力 |
|---:|---:|
| 0.01 | 0.637609 |
| 0.1 | 0.835316 |
| 1 | 1.611329 |
| 10 | 1.949842 |

極限は`Kn_d→0`で比1、`Kn_d→∞`で比2となる。これは丸め誤差やRK stepの問題ではない。[再計算script](theory_independent_checks_2026-10-09.py)と[結果](theory_independent_checks_2026-10-09.json)に保存した。

さらにCOMSOL6.4のTalbotの[公式式画像](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/images/particle_ug_fluid_flow.08.48.4.png)は分母の`1−2Λ`を表示し、原著と同ページのcontinuum式の`1+2Λ`と整合しない。**公式文書の内部不整合であり、COMSOL実行コードの不具合と断定していない**。保存datasetはTalbotを選択しておらず、COMSOLの実point-force parityは未検証である。[COMSOL Thermophoretic Force](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.48.html)

判断：現在の明示revision内での式実装は整合しているが、原著Talbotとしての物理bindingは修正判断が必要。原著準拠なら半径Knまたは同値変換済みの係数へ一本化する。COMSOL準拠を必要とする場合は、低Kn・遷移域・高Kn、複数`Λ`のpoint-force read-backで実行式を確かめてから意味を固定する。既存revisionの値を黙って変更せず、モデルの科学的意味が変わるrevisionとして扱う。

### 2.2 COMSOL軸境界：座標通過とFreezeを同一視できない

COMSOLのAxial Symmetry featureは、境界選択が固定されることを除きWallと同様の応答を選べる。[公式 Axial Symmetry](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.18.html)

保存されたCase-A/Pの`axi1`は`WallCondition=[Freeze]`である。[Case-P source設定](../../model_dataset/cf4_o2_etch_caseA_nonlinear_sass/cases/formal_iondrag_theory_consistent/caseP_100nm/external_reproduction/config/particle_physics_feature_settings.csv:6)、[Case-A source設定](../../model_dataset/cf4_o2_etch_caseA_nonlinear_sass/cases/formal_iondrag_theory_consistent/caseA_100nm/external_reproduction/config/particle_physics_feature_settings.csv:6)

C2/C3 companion生成処理はmaterial wallとinlet/outletを変更するが、`axi1`はactiveを確認するだけで、Freezeを継承する。[C2設定処理](../solver/tools/vv/comsol/comsol/RunM3C2StochasticCampaign.java:357)、[C3設定処理](../solver/tools/vv/comsol/comsol/RunM3C3CasePThreeCurrent.java:357)

対してcandidateのRZ軸は座標の継ぎ目で、通過時にradial basisの符号を変える。[candidate座標処理](../solver/src/chamber_particles/coordinates.py:11)。[Case-P比較契約](../solver/tools/vv/comsol/cases/m3c2_caseP_100nm_stochastic_pilot_v1.json:62)も座標通過を宣言しており、source設定と整合しない。

別の[critical-boundary microcase](../solver/tools/vv/comsol/comsol/RunM3CCriticalBoundaries.java:155)は`axi1`を明示的にBounceへ変える。そこでの合格を、Freezeを継承したchamber companionへそのまま移せない。Case-P C2 campaignはterminal event 0で、この差が結果に現れたことは確認されていない。しかし、軸通過の一般的なparityを証明する証拠にはならない。

修正はadapter/V&V側の責務である。軸を通過する同じno-swirl物理を比較するなら、companionの軸設定を明示し、read-backとaxis crossing microcaseをreceiptへ保存する。元のFreezeモデルを再現する比較は別scopeとして扱う。また、[normalizer](../solver/tools/vv/comsol/normalize_m3c2_comsol_pilot.py:90)の`held`というstatusだけからinletと断定せず、接触したsource boundary identityを保持する。

### 2.3 Brownian：節点でのFDT一致は補間後の一致を保証しない

candidateは同じEpstein dragの`γ=β/m`と同じ気体温度を使い、noiseの強さを`2γ k_B T/m`に結び付ける。このcoreのFDT所有は妥当である。[現行仕様](../solver/docs/physics_models.md:1068)

外部companionのBrownian用粘度は、P1補間されたprimitiveから

\[
\mu_B=\frac{\mu_{P1}}{36(\lambda_{P1}/d)/(8+\pi\sigma_R)}
\]

と作られる。[C2のbinding](../solver/tools/vv/comsol/comsol/RunM3C2StochasticCampaign.java:214)

節点で`μ/λ`と`ρ√T`が一致しても、個別のP1補間後には非線形関係を保持しない。保存canonical fieldの3779三角形の重心で宣言係数を再評価すると、`β_Brownian/β_candidate`は`0.9953912683～0.9999999995`、最大相対差`0.460873%`、点ごとの非重み付きRMS差`0.0599099%`となった。287 release点では最大`0.0535537%`。Case-A/Pの双方に同じ係数不一致がある。これは宣言式の再計算であり、COMSOLの新規solveやnoise実現値の測定ではない。また、全domainで証明された最大値・面積積分norm・軌道誤差ではない。

[再計算script](theory_comsol_coefficient_checks_2026-10-09.py)と[入力hash・計算結果](theory_comsol_coefficient_checks_2026-10-09.json)を保存した。historical accepted Case-P HDF5のSHA-256は`c54b4b658213230e82307ca89018538c0240bfcc83e793206d5093f4f08908e9`。Case-Aもその保存hashと照合した。現在の別fieldへ数値を自動拡張しない。

COMSOLのBrownian力もdragと対応する粘度・時間stepからnoiseを構成するので、same-fieldを主張するには同じ評価位置で摩擦係数を揃える必要がある。[COMSOL Brownian Forceの理論](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.43.html)

改良案は、companion内でcandidateと同じ補間済み`ρ,T`から`β_canonical`を一度求め、`μ_B=β_canonical/(3πd)`を設定することである。nodeだけでなくcell内probeでもread-backを照合する。[前回改良設計](improvement_design_2026-10-09.md)のCase-P drag binding課題は、Case-Aを含むこの補間後FDT一致まで拡張する必要がある。

## 3. 力モデルの妥当性

| モデル | 評価 | 適用条件・残る限界 |
|---|---|---|
| linear Epstein | 条件付きで妥当 | 自由分子、局所Maxwellian、低相対速度、指定表面則 |
| finite-speed Epstein | 条件付きで妥当 | 単一Maxwellian、球、指定Maxwell表面則、表面と気体の等温 |
| effective-gas Epstein | 感度・reference modelとして妥当 | 混合気体から一つのpseudogasへの縮約精度は別検証 |
| Stokes–Cunningham | 条件付きで妥当 | 低Re、相関に合うλ定義、airの係数を任意process gasへ一般化しない |
| Waldmann–Gallis | 条件付きで妥当 | 自由分子、translational熱流、局所gas mass frame、単一種または明示縮約 |
| Talbot | 原著bindingの再定義が必要 | 半径／直径Knと係数、COMSOL文書の内部不整合 |
| Saffman | 条件付きで妥当 | 連続域の低Re・弱せん断。高Kn用途へ適用しない |
| rarefied-vorticity lift | 明示係数の感度モデルに限定 | 現行の正確な係数形を一般的な原著相関へbindingする根拠は未確認 |
| Coulomb、gravity/buoyancy | 基本式は妥当 | 電荷・電場・質量・排除体積のauthorityが必要 |
| quasistatic DEP | 条件付きで妥当 | 球・準静的dipole・小粒子。DCまたはproducerの周期平均。travelling-wave・多極子・近壁を含まない |
| Barnes ion drag | 明示した近似式として妥当 | 有効速度、collisionless、screening、velocity distributionの近似誤差が残る |
| aggregate ion drag二種 | 感度モデルとしてのみ評価 | 元のcase式の再現と一般的なプラズマ物理の認定を分ける |

### 3.1 中性気体drag

linear Epsteinは

\[
\beta=\frac{4\pi}{3}a^2\rho\bar c\delta,
\quad F=\beta(u_g-v),\quad\bar c=\sqrt{8k_BT/(\pi m_g)}
\]

を計算する。現行linear revisionは独立の`δ∈[1,13/9]`を受け取る。等温Maxwell鏡面・拡散混合表面則に対応させる場合には`δ=1+πσ_R/8`となる。`σ_R`はこの表面則における熱適応を伴う拡散反射の割合であり、任意のenergy accommodation係数とは別である。native revisionの`λ/a≥10, |u_g-v|/c̄≤0.1`は保守的な適用制限として合理的である。[Epstein原著](https://authors.library.caltech.edu/records/3dxz2-4rt66)、[現行係数の処理](../solver/src/chamber_particles/physics/forces.py:1099)

finite-speed revisionを分子速度の独立積分と比較する方法は妥当である。effective-gas revisionは低速制限を最大1まで広げるが、その拡張が高精度を保証するわけではない。上記の等温Maxwell混合表面則にδを合わせた比較では、`|u_g-v|/c̄=1`でfinite-speed／linearの力比が、鏡面反射で約`1.23435`、完全拡散で約`1.16827`となる。速度範囲の選択は、許容する近似誤差と結び付ける必要がある。[有限速度球の原資料](https://reports.aerade.cranfield.ac.uk/bitstream/handle/1826.2/536/arc-cp-0523.pdf?sequence=1)

混合種を`ρ,T,m_g`一組へ畳み込むと、各種の`ρ_s/√m_s`などの重みを失う場合がある。solverの数値gateはproducerの混合縮約を証明しない。実装を増やす前に、代表組成でspecies別の運動量交換和と現行pseudogasの差を確認する方が具体的である。

Stokes–Cunninghamの半径／直径Kn変換は現行では明記されている。Allen–Raabeのair相関を用いた結果は、その平均自由行程規約と相関gasに限定する。低ReはStokes近似の条件であり、熱・せん断・プラズマの他モデルの適用条件を代替しない。[COMSOLのCMD係数変換の公式説明](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.37.html)

### 3.2 熱泳動・lift・DEP

Waldmann–Gallisの`32/15`係数は、translational heat fluxを使った運動論momentに対応する。total heat flux、対流熱輸送、電子熱流をそのまま代入できない。producerはgasの局所mass frameにおける`q_tr`を宣言する必要がある。nativeの低速・高Kn gateと独立速度moment検証は適切だが、effective-gasの多種縮約は別の物理closureである。[Gallisらの原著、式(1)](https://doi.org/10.1080/02786820490490001)、[独立CE moment検証](../solver/tests/verification/test_thermophoresis.py:27)

Saffmanの`w×ω`に沿う式と係数6.46、XY/RZの符号はCOMSOL公式の低Re shear-lift式に整合する。原著とcorrigendumの本文アクセスに制限があるため、係数の全文照合は公式式に依拠する。[COMSOL Saffman Liftの理論](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.42.html)、[原著corrigendum](https://doi.org/10.1017/S0022112068999990)

高Kn用途でSaffmanを使わず適用域不一致を拒否する設計は適切。ただし、現行`rarefied_vorticity`は明示`C_L`の感度モデルであり、一般的なfree-molecular sphere liftの確立した相関として認定する原著bindingは今回確認できなかった。回転球の原著を非回転粒子へそのまま流用できない。[現行の感度revision](../solver/docs/physics_models.md:835)

quasistatic spherical DEPは`2πε_g a³ Re(K_CM) ∇〈|E|²〉`の形で、均一場ゼロ、半径の三乗依存、電気特性の符号を確認するscalar oracleがある。DCでは`〈|E|²〉=|E|²`、周期場ではproducerが一つのfrequencyで形成した物理時間平均、すなわちRMS二乗を受け取る。peak phasorとRMSのfactor 2、solution、frequency、実Clausius–Mossotti係数はproducerが所有する。[現行DEP仕様](../solver/docs/physics_models.md:794)、[COMSOL公式DEPのRMS規約](https://doc.comsol.com/6.3/doc/com.comsol.help.mfl/mfl_ug_modeling.05.14.html)

field producerは`E`と`∇〈|E|²〉`の整合した生成規則を保持する必要がある。独立に補間した勾配は、補間した`E`を微分した値と一般には同じではない。これは補間契約で定める事実であり、coreが場を勝手に再構成する理由にはならない。certified-radius上限が記入されていても、粒径と場の変化長さに対するdipole近似の物理的証明はproducerに残る。

gravity/buoyancyの`(1−ρ_g V_displaced/m)g`、Coulombの`ZeE/m`は基本式と整合する。massとdisplaced volumeを独立authorityにし、慣性をruntimeで粒径から再構成しない点は妥当である。浮力式を任意の非静水圧pressure-gradient forceまで含むものとは扱わない。[重力・浮力処理](../solver/src/chamber_particles/physics/forces.py:3053)、[独立物性検証](../solver/tests/verification/test_physics.py:886)

### 3.3 Barnes ion drag：近似誤差を実装検証と分ける

[実装](../solver/src/chamber_particles/physics/forces.py:1271)はrelative flowを使い、collectionとCoulomb orbitalの断面積を計算する。`a/λ_D, b90/λ_D, bc/λ_D≤0.1`、`λ_in/λ_D≥10`などは、採用したcollisionless binary-scattering近似を制限する妥当なgateである。

[impact-parameter求積検証](../solver/tests/verification/test_ion_drag.py:29)は、Rutherford散乱の断面積積分を独立に確認する。これは有効速度でのcross sectionの検証であり、Maxwellian全体のvelocity averageの検証ではない。

無電荷粒子を吸収球とし、イオン速度を粒子frameのshifted Maxwellianとすると、collection forceの独立期待値は

\[
F_{col}=\pi a^2 n_i m_i\mathbb E[|w|w],
\qquad F_{col}\simeq\frac43\pi a^2 n_i m_i\bar c_i U\quad(U\to0).
\]

現行effective-speed式の同じ極限は`πa² n_i m_i c̄_i U`であり、比は`3/4`。独立した球座標Gauss–Legendre128/256点求積とproduction関数を比較し、`U/c̄=.001`で`0.750000184`、`.1`で`0.751827896`を得た。いずれも現行applicabilityは真だった。[再計算結果](theory_independent_checks_2026-10-09.json)

したがって、**Barnes effective-speedは厳密なshifted-Maxwell momentum modelではなく、低driftでも一定のモデル誤差がある**。この結果を、全荷電条件のion dragが一律25%ずれるという主張へ拡張してはいけない。原著名に対応する近似の実装と、対象物理への近似精度を分ける必要がある。

ion dragには、screeningの非線形性、ion velocity distribution、collisionによる力の変化がある。力で駆動されたcollisional flowはshifted Maxwellianとは限らない。λによるcutoffを設けるだけで一般的なcollisional ion dragを再現したことにはならない。[Hutchinson–Haakonsenの原著](https://arxiv.org/abs/1305.6944)、[Khrapakらの原著](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.66.046414)

### 3.4 aggregate ion drag

[relative-flow screenedモデル](../solver/src/chamber_particles/physics/forces.py:1417)は相対速度を使う点で基本的なframe整合を持つ。しかし、screening長の上下限、`1 m/s` regularizer、断面積上限、非負Coulomb logは、versioned sensitivity closureである。collision mean free pathでcutoffすることを、一般的なcollisional theoryと呼ばない。

[electric-field-directed imageモデル](../solver/src/chamber_particles/physics/forces.py:1547)は粒子速度を受け取らず、力の向きを電場に向ける。一般的なion momentum transferは相対速度・分布によって決まるため、`v_p=u_i`のcoflowや`u_i⊥E`を普遍的に扱うモデルにはならない。「粒子がlabで遅い」「ion transportと電場方向が対応する」など、元caseの前提を必要とする。名称だけから一般的な壁のimage-charge forceと解釈してもいけない。

この二種の式再生・compiled parity・COMSOL式との一致は有効なreference evidenceだが、kinetic derivationや実機validationの代わりにならない。新しい汎用modelを増やすより、既存sensitivity revisionの利用条件と、species/distribution別の小referenceとの誤差を先に定量化する。

## 4. 帯電モデルの妥当性

### 4.1 stationary OML

[stationary OML](../solver/src/chamber_particles/physics/charge.py:650)は、吸収球のelectron/ion collection rateの引力・斥力branchを正しく使い、`dZ/dt=Γ_i−Γ_e`とする。`d(dZ/dt)/dZ<0`は負のfeedbackで、平衡探索とglobal charge envelopeに利用できる。符号branchのため不要なexp overflowも避けられている。

必要な物理前提は、単一のsingly charged positive ion、局所Maxwellian、collisionless、unmagnetized、球のequipotential表面、emissionなし、背景への粒子feedbackなしである。`a/λ_D≤0.1`と`|u_i−v_p|/c̄_i≤0.1`は保守的な利用制限だが、誤差1%などの認定値ではない。静止electron分布のframeもproducerが明示する必要がある。

`C=4πεa(1+a/λ_D)`はlinearized Debye–Hückel／generalized Whippleのcharge–potential closureである。OML collection currentの式そのものと、このcapacitance近似を混同しない。有限粒径のself-consistent plasma responseを解いたものではない。[Tang–Delzanno原著、式(35)・(38)](https://arxiv.org/html/1503.07820v1)

### 4.2 shifted Maxwellian OML

[shifted revision](../solver/src/chamber_particles/physics/charge.py:776)は、`s=|u_i−v_p|/√(k_BT_i/m_i)`として

\[
P(s)=\frac12e^{-s^2/2}+\sqrt{\pi/8}(s+1/s)\operatorname{erf}(s/\sqrt2),
\quad H(s)=\sqrt{\pi/2}\operatorname{erf}(s/\sqrt2)/s
\]

を使う。負電位のion rateは`A_i[P−(φ/V_i)H]`、electron rateは`A_e exp(φ/V_e)`となる。原著のnormalized flow `v=s/√2`を変換すると同値である。[Thomas–Coppins原著、式(2)〜(4)](https://arxiv.org/pdf/1305.5763)

今回、閉形式をreferenceへ転記せず、3D shifted Gaussianから`E|w|`と`E(1/|w|)`を球座標で独立に積分した。`s=0, 10⁻⁶, .5, 2, 5`でproductionのP/Hとの最大相対差は約`4.32×10⁻14`。この係数実装は強く支持される。[再計算script](theory_independent_checks_2026-10-09.py)、[既存の速度求積検証](../solver/tests/verification/test_shifted_charge.py:91)

認定は負電位branch、単一shifted Maxwellian、宣言drift上限、charge envelopeの範囲に限定する。collisional force-driven ion distribution、正電位への一般拡張、非球形、多種・多価イオンはこの一致から認定できない。

### 4.3 aggregate二電流・三電流

[二電流](../solver/src/chamber_particles/physics/charge.py:154)と[三電流](../solver/src/chamber_particles/physics/charge.py:328)は、背景の有効温度・有効速度・clamp/floorを宣言した感度modelである。三電流でnegative-ion密度をゼロにした時の数値還元を検証することは妥当だが、多種プラズマを一組のnegative-ion primitiveへまとめる物理精度を認定しない。

二電流の零流速極限で、`1 m/s` regularizerと`0.01 V` floorが無視できる場合、`v_eff=c̄_i`、`V_i,eff=4V_i/π`。従ってionの電位による引力増分は厳密なstationary OMLの`π/4=0.785398`となる。`−φ/V_i=10`ではion rate全体の比は約`0.804907`。**aggregateをstationary OMLの単なる高速実装と扱うことはできない**。[独立極限計算](theory_independent_checks_2026-10-09.json)

Decimalで同じ式を再評価するoracleは、浮動小数点・compiled・設定伝播の検証として役立つ。物理的な速度分布積分やspecies別電流和の検証とは別である。

### 4.4 continuous chargeの限界

電荷数Zを連続変数として積分するのはmean charging近似である。実際の吸収は±1 electronの離散過程で、極小粒子ではcharge fluctuationや正電荷へのexcursionが重要になり得る。[Thomas–Coppinsの離散帯電モデル](https://arxiv.org/abs/1305.5763)

例えばvacuum capacitanceの目安では、electron温度1 eV、半径5 nmで一電子当たりの電位変化は約0.288 V、半径50 nmでは約0.0288 Vとなる。これはdatasetの実温度・電荷分布を評価した値ではなく、連続近似を粒径だけで無条件に認定できないことを示すscale例である。

ただし、今回の範囲で直ちにdiscrete charge modelをproductionへ追加する必要があるとは判断しない。まず代表caseで平衡電荷、charge relaxation time、fluctuation scaleと対象の到達・付着感度を比較し、mean approximationが目的精度に足りるか判断する。

## 5. 数値処理の妥当性

### 5.1 coupled RK4と安定性

[RK4](../solver/src/chamber_particles/integrators.py:3658)は`(x,v,Z)`を同じ四stageで評価し、帯電だけを別subcycleして後から運動へ渡していない。[連成解析解の検証](../solver/tests/verification/test_integrators.py:1622)は、`Z'=-1.5Z, v'=Z, x'=v`でglobal四次を確認する。滑らかなRHSのclassical RK4として妥当である。[大学のRK導出資料](https://math.unm.edu/~nitsche/courses/471/classnotes.pdf)

[prepare](../solver/src/chamber_particles/engine.py:3990)にはdragの`hL_v<2.5`、chargeの`hL_Z≤0.5`という安定性gateが実在する。scalar decayのRK4負実軸限界約2.785より保守的だが、全coupled Jacobianの安定性やtrajectory errorの上限を証明する条件ではない。

空間P1のcell境界、Q1のcell間微分、cell-associated場ではRHSの滑らかさが低下する。全chamberで四次という主張は実際のfieldとtrajectoryによる収束を要する。時間snapshotのknotは[macro gridへ併合される](../solver/src/chamber_particles/engine.py:1307)ので、「時間kinkを未分割で跨ぐ」という指摘は現行には当たらない。

### 5.2 exponential midpoint

[実装](../solver/src/chamber_particles/integrators.py:3104)は半step predictorからmidpointのdrag・target・additive force・charge lawを評価して凍結する。frozen linear dragの解析更新と`φ1,φ2`、chargeのaffine exponential更新は整合する。[exponential integratorの原著review](https://na.math.kit.edu/download/papers/acta-final.pdf)

[80桁Decimal oracle](../solver/tests/verification/test_integrators.py:1934)と[可変係数manufactured case](../solver/tests/verification/test_integrators.py:2010)は、固定係数の正確性と滑らかな可変係数での二次収束を分けており適切である。`J≤0`のcharge affine安定性を、任意の非線形帯電が厳密、またはaccuracyのためのstep制約不要という意味にしない。

### 5.3 dense出力とfirst-hit

[RK4 dense](../solver/src/chamber_particles/integrators.py:3545)は、位置をendpoint位置・速度のHermite cubic、速度とchargeをstage rateから別のcubicとして形成する。[dense chargeの検証](../solver/tests/verification/test_integrators.py:3127)は、単step interior error約`O(h⁴)`を確認する。endpoint局所誤差`O(h⁵)`とは異なる。

独立に`x''=−x, h=.2, θ=.37`を確認すると、dense velocityは`−0.07393246266666667`、dense position曲線の解析微分は`−0.07397286666666722`で、差は`4.0404×10⁻5`。これは別々の近似補間の性質であり、直ちにcore bugとは判定しない。ただし、interiorで`dx/dt=v_dense`が厳密に成立する軌道と説明してはいけない。

[restricted Bernstein enclosure](../solver/src/chamber_particles/integrators.py:1226)と[event判定](../solver/src/chamber_particles/events.py:2182)は、control pointの凸包性で表現された数値曲線を囲う合理的な手法である。未確定区間は分割し、budget不足をfailureにする処理も適切である。[MITのBézier原資料](https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node12.html)

ここでのfirst-hit保証は、**表現された数値曲線に対する保証**である。`geometry_rtol`は真のODE解とのtrajectory errorを直接認証しない。step幅の収束、denseの収束、event toleranceの収束を分ける必要がある。`AGENTS.md`の「曲線state_atは同じintegrator規則で再評価」という一律説明も、現行のRK4 dense・OU Hermiteの意味に合わせて明確にする余地がある。

### 5.4 OU、FDT、条件付き分割

固定係数のinertial Langevin

\[
dv=-\gamma(v-u)dt+\sqrt{2\gamma\Theta}\,dW,\quad dx=vdt,
\quad\Theta=k_BT/m
\]

について、[joint OU covariance](../solver/src/chamber_particles/stochastic.py:34)は

\[
Q_{vv}=\Theta(1-e^{-2z}),\quad
Q_{xv}=\Theta(1-e^{-z})^2/\gamma,
\quad Q_{xx}=\Theta(2z-3+4e^{-z}-e^{-2z})/\gamma^2,
\quad z=\gamma h
\]

と一致する。small-z series、`expm1`、PSD検査は桁落ち対策として合理的で、位置と速度をjointで更新する設計が必要である。[Gillespie原著](https://doi.org/10.1103/PhysRevE.54.2084)

[conditional half-split](../solver/src/chamber_particles/stochastic.py:321)はGaussian conditioningで左childを生成し、右childを残差としてparent endpointを保持する。refineで保存済みparent incrementを破棄して独立に引き直さない設計は妥当。conditional residualに必要な新しいnormal drawは生成する。[Gaussian conditioningの原資料](https://gaussianprocess.org/gpml/chapters/RWA.pdf)

[既存oracle](../solver/tests/verification/test_stochastic.py:333)はproduction閉形式を写すだけでなく、Green核のGauss積分と行列solveを使っている。今回さらに、`γ(t)=1.7(1+0.8t), Θ=.7, t=1`の時間変動摩擦・一定温度の線形no-wall系を独立に確認した。

| 区間数N | midpoint-frozen合成のcovariance最大絶対誤差 |
|---:|---:|
| 4 | 0.00189175 |
| 8 | 0.000474614 |
| 16 | 0.000118761 |
| 32 | 0.0000296970 |
| 64 | 0.00000742468 |

観測次数は`1.99489～1.99992`。非ゼロnoise covarianceの決定論的moment伝播であり、乱数ensembleや公開`simulate`を実行した計測ではない。noiseを伴うこの限定族で二次のcovariance収束を支持する。productionの公開workflow全体、空間依存係数、一般的なcharge連成、wall first passageまで認定する証拠ではない。[独立数値check script](theory_numerics_independent_checks_2026-10-09.py)、[OU・RNG・denseの結果](theory_numerics_independent_checks_2026-10-09.json)

### 5.5 有限depthのBrownian path

macro rootで凍結した係数のjoint OU endpointを条件付き分割し、有限depthのHermite pathで境界を調べる処理は、明示された数値模型として整合している。しかし、未標本の連続OU bridge excursionを排除したことにはならない。base depthとevent時のadaptive depthを別authorityにする現在の規則は、この制約を正しく認めている。[現行説明](../solver/docs/numerics.md:1138)

既存検証の意味には、次の限定がある。

- `weak mean`と名付けたtestの一部はjoint noise incrementをゼロにしており、deterministic limitの証拠である。
- adaptiveとfixed-depthの比較の一部はmass `10³⁰ kg`でnoiseが極小。一般の有noise barrier精度を閉じる証拠ではない。
- 恒係数の公開ensemble testはexpected covarianceにproduction関数を使う。pipeline検証として有効だが、式の独立性はGreen核oracleとの組合せで評価する。
- depth 6/7/8で結果が安定しても、独立first-passage oracleやmiss probabilityの上界ではない。

次に必要なのは、実noiseを伴う空間依存場の独立moment reference、非dyadic出力時刻のdepth収束、独立PDEまたは十分精細な外部referenceによる到達確率の確認である。無条件の「Brownian二次」や「壁missゼロ」の表現は避ける。

### 5.6 RNG・field・wall・RZ

[Philox](../solver/src/chamber_particles/rng.py:233)はRandom123の定数、round、word順と整合する。公式KATのzero・全ffff・π-wordの三組を直接再計算し、全て完全一致した。particle ID・physical interval・tree node・componentによるaddressは、slabや処理順に依存しない再現性の設計として妥当である。[Random123原著](https://www.cs.tufts.edu/~nr/cs257/archive/john-salmon/parallel-random.pdf)、[公式KAT](https://raw.githubusercontent.com/DEShawResearch/random123/main/tests/kat_vectors)

[時間補間](../solver/src/chamber_particles/cpu.py:1332)はrange外を拒否し、同じ空間basisでlinear snapshot interpolationを行う。[独立解析trajectory test](../solver/tests/scenarios/test_time_dependent_field_run.py:25)もある。補間された場を正しく解くことと、snapshotが元の物理時間変動を十分解像することは別である。

有限半径接触のsegment offset＋endpoint circleは、球の中心と壁のMinkowski offsetによる接触として妥当。[接触処理](../solver/src/chamber_particles/events.py:6214)、[CGALのoffset定義](https://doc.cgal.org/latest/Minkowski_sum_2/group__PkgMinkowskiSum2Ref.html)

Maxwell wallのnormal flux分布とtangent Gaussianを分ける実装は、拡散再放出のwall lawとして妥当。粒子付着率・restitutionなどの材料parameterが実機から認定されたことは意味しない。[wall処理](../solver/src/chamber_particles/boundaries.py:680)、[SPARTA公式モデル](https://sparta.github.io/doc/surf_collide.html)

RZはsigned-meridional no-swirlの2自由度模型として整合する。3D等方Brownianでは方位速度が生まれ、radial運動へ幾何項が影響する。2D RZのnoiseを、そのまま真の3D radial distributionと見なすことはできない。[COMSOLのout-of-plane設定](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.02.html)

## 6. COMSOL比較から言える範囲

### 6.0 Case-Pのdrag sourceを値欄だけで認証しない

前回確認したC2 Case-Pのsource selector問題は今回の公式調査でも解消されていない。COMSOLのDrag Forceには速度・圧力等のsource physicsと、各値の指定方法がある。P1の値欄を書き換えても、実際のsource selectorを変更・read-backしたことにはならない。[COMSOL Drag Forceの設定](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.06.html)

[C2 runner](../solver/tools/vv/comsol/comsol/RunM3C2StochasticCampaign.java:327)は値式を設定するが、sourceの`u_src=root.comp1.u`等を明示的に更新・検査する証拠がない。receiptへ意図した式を文字列として書くことと、実使用設定をread-backすることは別である。

C3はnative dragを無効にし、[explicit Epstein force](../solver/tools/vv/comsol/comsol/RunM3C3CasePThreeCurrent.java:280)へ変更して、この問題を閉じた比較である。C3の成功はその修正済みscopeに有効だが、BrownianをoffにしたC3からC2のnoise/drag一致を認定できない。また、[C3のtotal force診断](../solver/tools/vv/comsol/comsol/RunM3C3CasePThreeCurrent.java:517)は宣言成分の再構成和であり、実assembled ODE RHSの独立read-backとは呼ばない。

### 6.1 設定されたRK4とevent区間の次数は別

公式文書によれば、`WallAccuracyOrder=1`は衝突の前後をforward Eulerで進める設定であり、単なるhit時刻の局在精度ではない。step途中のreleaseにも同じ設定が作用する。C2/C3はOrder 1の設定を確認するため、bulkがclassical RK4でも、wall/releaseを含む全区間が四次とは言えない。[COMSOL公式のWall Accuracy Order](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.02.html)

Brownianでは公式がOrder 1を推奨しているため、この設定自体を誤りとはしない。eventありの決定論、eventなしの決定論、Brownian populationで別の収束・比較scopeを定める必要がある。COMSOLのNewtonianFirstOrder＋auxiliary variableへexplicit RKを使えることも公式と保存probeで確認でき、auxiliary chargeがあるだけでRK4不正とする指摘は成立しない。[first-orderとauxiliary variablesの公式説明](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.02.html)

### 6.2 tlistは内部step履歴ではない

C2 Case-Pの[保存receipt](../solver/evidence/m3c2/caseP_100nm_final_campaign_v1/final_result_receipt.json:11)はfixed RK4 `h=20 μs`、30 ms、121の出力時刻を記録する。C3三電流referenceは`h=5, 2.5, 1.25 μs`、`rtol=10⁻8`を記録する。[fine reference receipt](../solver/evidence/m3c3/caseP_three_current_companion_v1/reference_dt_1p25us_run_receipt.json:96)。これは保存された設定の証拠である。`tlist`は通常補間された保存時刻であり、同じ間隔の内部integration stepやwall-event historyを意味しない。`rtol`の値だけでもtrajectory errorを認定できない。

現在のreceiptには全internal step履歴をread-backした証拠がない。保存済みの別revisionのglobal charge Jacobian上界をこれらのstepへ代入して、不安定と断定することもできない。外部比較では、実際のlocal charge scale、fixed-step refinement、output方式を分けて確認する。[COMSOL公式Time APIの`tout`・`tlist`・`rktimestep`](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_api_solver.51.51.html)

### 6.3 保存済み合格の保持と限定

| 保存済み比較 | 有効な証拠 | 拡張できない範囲 |
|---|---|---|
| same-P1の決定論trajectory | 揃えたcanonical field上での粒子solver agreement | native COMSOL FE field生成全体、実機validation |
| critical-boundary microcases | 明示したwall設定・cohortのevent agreement | Freezeを継承したchamber軸の一般parity |
| Case-A Brownianの独立seed population | 登録したmetricと同等性幅での集団比較 | 同じseedの軌道一致、3D Brownian、厳密FDT一致 |
| Case-P C2のterminal population | 登録した4曲線、5%幅の限定comparison | event 0から境界認証、式不一致の影響ゼロ |
| Case-P C2のRZ＋fate分布 | 80 active bins＋3 terminal区分、TV15%幅のcomparison | 他のbin定義、3D分布、全physicsへの認定 |
| aggregate charge/ion-dragの式再生 | 元モデルの式・設定のreference再現 | 一般的なプラズマ物理の正解 |

保存されたC2 Case-Pでは287粒子×参加者ごと32replica、30 msの比較で、empirical TV最大`0.0106707317`、平均位置差最大約`0.000287793 m`、terminal差ゼロを観測している。[C2 final receipt](../solver/evidence/m3c2/caseP_100nm_final_campaign_v1/final_result_receipt.json:50)。C3の100 nm、287粒子、Brownian off、三電流＋修正済みexplicit dragの限定caseは、141 terminal eventsのparticle/outcome/semantic一致、event time差RMS約`4.702×10⁻9 s`、最大約`2.753×10⁻8 s`を保存する。[C3 scopeと結果](../solver/evidence/m3c3/caseP_three_current_companion_v1/README.md:39)

これらの観測結果を今回破棄する根拠はない。入力の意味不一致は比較主張の範囲を狭める問題であり、指標が既に不合格だったという意味ではない。

同じ乱数seedの番号を設定しても、COMSOLとcandidateのRNG・draw allocation・time stepが違えば同じBrownian pathにはならない。shared incrementを明示しない比較は独立seed ensembleで行う設計が適切である。

意味を揃えた七層preflight、initial state、local RHS、boundary response、time-resolved trajectoryを分ける既存[方法論](../vv_methodology.md)は妥当である。ただし、軸のFreeze継承と補間後FDT不一致を実設定で閉じるまで、same-physicsの包括的認定はできない。ゼロeventは境界`NOT_TESTED`、未比較modelは`NOT_TESTED`として保持する。

## 7. 優先する改良と完了条件

| 順 | 具体的な変更 | owner | 最小の独立証拠・完了条件 |
|---:|---|---|---|
| 1 | Talbotの原著conventionを固定し、半径Knまたは同値変換係数へ一本化 | `technical_research`、physics式・catalog・compiled・model revision | continuum/free-molecular極限、原著規約のscalar oracle、複数Kn/Λで一致。旧説明・旧bindingを同じ変更で除去 |
| 2 | Case-A/P軸設定のread-back、contract修正、同じseam物理のcompanion | COMSOL adapter/V&V | source→companion変更receipt、axis crossing case、statusだけでfacetを同定しない |
| 3 | Brownian係数を補間後の同じβへ結び付ける | 外部Java companion/binding | nodes、cell interior、release点でβとnoise covarianceの一致。Case-A/P双方を再比較 |
| 4 | Barnes・aggregateの近似精度を独立kinetic referenceで可視化 | 既存verificationと外部V&V | drift・電位・screeningの小matrix、collection/orbital/totalの誤差を分ける |
| 5 | 非linear・空間依存の実noise moment、barrier、denseを別系列で検証 | stochastic/integratorの既存verification・scenario | h/h2/h4、base depth系列、seed confidence interval、独立reference |
| 6 | field輸出・cacheの品質認証を閉じる | field preprocessor／既存品質gate | source×targetの共通分割、cell内gradient、局所feature反例、time interval budget |
| 7 | 物理の適用性を代表caseの誤差budgetへ結び付ける | case producer＋外部V&V | mixture縮約、source weight、背景場、wall parameter、実機測定による感度・validation |

新しい汎用framework、第二solver、COMSOL専用core、常設diagnostic subsystemは不要である。既存ownerの式・binding・小referenceを閉じる。数値結果やモデルの科学的意味を変える修正はrevisionを更新し、同じchangeで置換された設定・文書・testを除去する。

field cacheの問題は[前回改良設計](improvement_design_2026-10-09.md)で独立反例が確認済みであり、今回の理論再調査でも優先度を下げない。正しいforce lawであっても、局所featureを失った場を読めば目的とするtrajectoryを正しく計算できない。

## 8. 今回の実行確認と未実行事項

今回新たに実行した確認は次のとおり。

1. shifted OML係数の独立速度積分。最大相対差約`4.32×10⁻14`。
2. Barnes無電荷collectionの独立Maxwell momentum積分。弱drift極限の比`3/4`を確認。
3. Talbotの同一係数によるKn規約比較。高Knで力比2へ向かうことを確認。
4. aggregate零流速極限の解析比較。引力増分`π/4`、`−φ/V_i=10`でrate比約0.804907。
5. 時間変動摩擦の独立Green核covarianceと現行OU更新合成。限定族で二次収束。
6. Random123公式Philox KATの三組。bitwise一致。
7. RK4 dense位置微分とdense velocityの差を解析microcaseで確認。
8. 保存HDF5のcell内部におけるBrownian／drag係数の独立再評価。
9. 現行帯電・ion-dragのfocused verification、**35 passed in 0.73 s**。

実行commandはsolver directoryで

```powershell
uv run --locked python ../reviews/theory_independent_checks_2026-10-09.py
uv run --locked python ../reviews/theory_comsol_coefficient_checks_2026-10-09.py
uv run --locked python ../reviews/theory_numerics_independent_checks_2026-10-09.py
uv run --locked pytest tests/verification/test_shifted_charge.py tests/verification/test_aggregate_charge.py tests/verification/test_signed_ion_charge.py tests/verification/test_ion_drag.py tests/verification/test_aggregate_ion_drag.py -q
```

前回のcore739件合格と外部tool suiteの失敗記録は、前回実行の結果として保持する。今回full suite、static quality gate、COMSOL solver、現在の100万粒子を再実行したと主張しない。数値確認は有限個の対象族の証拠であり、すべての組合せの形式的証明ではない。

現時点で認定できるのは、**明示された2D模型と条件内で、独立根拠を持つ数値基盤と多数の限定比較が成立していること**である。Talbotのbindingと外部比較の意味を先に修正し、その後にモデル近似・場・確率到達の誤差を目的精度へ結び付けるのが、実チャンバー予測へ進む順序として妥当である。
