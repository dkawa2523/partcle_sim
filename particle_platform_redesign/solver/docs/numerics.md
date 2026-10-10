# Numerical decisions and core microcases

この文書は、production数値処理が従うP02 oracle、P03 field監査、P04 ballistic実装、P05 geometry/event、
P06 revision 1・2のcoupled RK4と一定加速度event、revision 3aのboundaryless連続enclosure、
revision 3bの一般RK4材料boundary経路、およびP06-RZのsigned-meridional axis経路を
まとめ、P11のnative exponential midpoint、P15のcontinuous charge連成、B02のinertial Brownian経路を加える。
C01～C10のmachine-readableな入力と
期待値の権威は`tests/verification/microcases.py`であり、この文書は式、意味、合格条件を説明する。
COMSOL出力と旧solverは正解として使用しない。

## 1. P02の範囲

P02は数値solverの実装ではなく、後続実装が合格すべき独立oracleを固定する段階である。各caseは
`case_format.write()`でcanonical HDF5を作成し、公開`load_case()`を通す。期待値は別の
`expected.json`へ書き、production codeから読まない。HDF5 golden binary、汎用fixture framework、
test専用physics modelは作らない。

各能力が有効になるpackageは次のとおり。

| case | 固定する能力 | production比較を開始するpackage |
|---|---|---|
| C01 | ballistic、非macro時刻のframe | P04 |
| C02 | 一様linear drag、RK4収束 | P06 revision 3aのboundaryless公開APIとdirect RK4 verification |
| C03 | linear drag＋一定加速度 | P06 revision 3aのboundaryless RK4とP11 exponential midpointの公開API、direct verification |
| C04 | fixed charge、電荷符号、mass authority | P06の証明済み一定加速度で公開API |
| C05 | 重力・浮力、mass/排除体積authority | P06の証明済み一定加速度で公開API |
| C06 | P1と非affine mapped Q1補間/support | P03 |
| C07 | 平面first hit | P05 |
| C08 | surface departure、反射、再衝突 | P07 |
| C09 | 薄いgap、複数反射、残時間 | P07 |
| C10 | 非直交cornerの同時facet集合 | P07 |

C03の一定係数指数更新はP11で実装済みである。P02の独立oracleとproduction methodを同じ実装にせず、
COMSOL結果も合格値のauthorityにしない。

## 2. C01～C05：運動方程式

### C01 ballistic

```text
x0 = (0.125, -0.275) m
v0 = (0.4, -0.2) m/s
t = 0 ... 0.8 s, dt = 0.2 s
frame = 0, 0.05, 0.15, 0.35, 0.8 s
```

粒子ごとの一般形は
`x(t)=x_release+(t-t_release)*v_release`, `v(t)=v_release`である。macro境界でない
0.05、0.15、0.35 sはaccepted proposalから評価し、frame出力や個別release時刻のためproduction stepを
分割しない。

P04のproposalは各macro stepの丸め済み累積位置を次の解析原点にせず、tableのrelease時刻・位置を保持して
frameとendpointを評価する。異なる`dt_s`とoutput scheduleでrelease event、final、共通frameをbitwise同一に
保ち、release時刻と一致する評価では演算結果でなく入力初期値をそのまま返す。有限入力からelapsedまたは
位置がoverflowした場合はclampや成功resultにせず明示的に失敗する。

### C02 linear drag

```text
x0 = (-0.4, 0.2) m
v0 = (0.9, 0.35) m/s
u  = (0.1, -0.05) m/s
tau = 0.5 s, end = 1.0 s
```

方程式と解は

\[
\dot{\boldsymbol v}=(\boldsymbol u-\boldsymbol v)/\tau,
\quad
\boldsymbol v=\boldsymbol u+e^{-t/\tau}(\boldsymbol v_0-\boldsymbol u),
\]

\[
\boldsymbol x=\boldsymbol x_0+\boldsymbol u t+
\tau(1-e^{-t/\tau})(\boldsymbol v_0-\boldsymbol u).
\]

RK4は`dt=[0.2,0.1,0.05,0.025] s`で位置と速度を別々に評価し、後二組の観測次数を3.5以上、
誤差を各刻みで単調減少とする。比較は終端`1.0 s`のcomponent-wise L-infinity normで行い、観測次数は
`(0.1,0.05)`と`(0.05,0.025)`の各隣接刻み対で評価する。最細で位置`1e-8 m`、速度`2e-8 m/s`以下とする。
`expected.json`の
`analytic_reference_precision`は閉形式値を記録する精度であり、有限刻みRK4の合格差ではない。methodの
合格条件は`method_acceptance.rk4_fixed`だけが所有する。入力は
`epstein_linear_v1`の一様primitiveを、解析上`tau=0.5 s`になるよう構成している。RK4自身の単体検査では
同じODEをstage evaluatorへ直接与え、test専用drag modelをcatalogへ追加しない。
revision 3aでは、同じ入力をboundaryless・fully-supported regular layoutの公開`simulate`時系列として実行する。
全短縮RK4を覆うsupport/applicability enclosureを先に受理し、出力frameは受理済みproposalを読むだけである。

### C03 linear drag＋一定加速度

```text
x0 = (-0.25, -0.1) m
v0 = (0.6, -0.5) m/s
u  = (-0.2, 0.3) m/s
tau = 0.25 s
a = (0.8, -0.4) m/s^2
end = 0.75 s
```

`E=exp(-t/tau)`, `A=1-E`として

\[
\boldsymbol v=\boldsymbol u+E(\boldsymbol v_0-\boldsymbol u)+\tau A\boldsymbol a,
\]

\[
\boldsymbol x=\boldsymbol x_0+\boldsymbol u t+
\tau A(\boldsymbol v_0-\boldsymbol u)+\tau(t-\tau A)\boldsymbol a.
\]

P06ではRK4を`dt=[0.125,0.0625,0.03125,0.015625] s`で検査し、終端`0.75 s`のcomponent-wise
L-infinity誤差の単調減少、最後の二つの隣接刻み対で観測次数3.5以上、最細の位置`5e-9 m`、
速度`2e-8 m/s`以下を要求する。P11の
`dt=[0.75,0.375,0.125] s`の一定係数exponential midpointが丸め誤差内で同じ終端へ到達することを検査する。
C03もrevision 3aのboundaryless regular support公開時系列として実行する。一定係数の解析解との時系列比較と
direct RK4 verificationをP06、指数methodのreference/production検証をP11が所有する。

### C04 一様電場

`Z=-5`, `E=(4,-2) V/m`、同じ初期状態で質量だけを
`[1.602176634e-18, 3.204353268e-18] kg`とする。SI exactの
`e=1.602176634e-19 C`を用い、両粒子の力は
`(-3.204353268e-18, 1.602176634e-18) N`、加速度はそれぞれ`(-2,1)`と`(-1,0.5) m/s^2`である。
径や排除体積から質量を再構成してはならない。

C04の公開API caseは一様電場であり、証明済み`quadratic_exact`を使う。`E_x=x, E_y=0`のような
非一様manufactured fieldは4つのRK stageでの再評価とfield補間を分離検査し、revision 3aのboundaryless・
fully-supported regular caseでは`rk4_reintegrated`を使う。公開回帰は安全な非一様fieldの成功と、短縮pathの
support逸脱がframe有無に依存せず失敗することを検査する。

### C05 重力・浮力

`rho_g=1.2 kg/m^3`, `g=(0,-10) m/s^2`とし、

\[
\boldsymbol a=(1-\rho_gV_{disp}/m)\boldsymbol g
\]

を三組の`(m,Vdisp)`で検査する。期待加速度は`(0,-10)`, `(0,-7)`, `(0,-8.5) m/s^2`。
これは式とauthorityを分離するmanufactured caseであり、実チャンバーの適用条件を表すものではない。

### P06 revision 2：一定加速度の厳密放物線event

fixed charge、dragなし、`cartesian_xy`で、使用するfieldの全canonical node値がcomponentごとに厳密一致する時だけ、
粒子別合成加速度pathをこのspecializationへ送る。近似的に一様な入力を
暗黙のtoleranceで一定扱いしない。補間後の値は加重和の丸めで最下位bitが変わり得るため、certificateはcanonical
DOFをauthorityとし、実stage samplingはsupportとapplicabilityの確認に使う。

区間幅を`h`、正規化時刻を`theta in [0,1]`として

\[
\boldsymbol x(\theta)=\boldsymbol x_0+\boldsymbol L\theta+\boldsymbol Q\theta^2,
\quad
\boldsymbol L=h\boldsymbol v_0,
\quad
\boldsymbol Q=\tfrac12h^2\boldsymbol a
\]

を解析評価する。endpoint chordとの差は`Q(theta^2-theta)`なので、全区間で

\[
\|\boldsymbol x(\theta)-\boldsymbol x_{chord}(\theta)\|
\le \|\boldsymbol Q\|/4=\|\boldsymbol a\|h^2/8
\]

である。BVH broad phaseはendpoint chord AABBをこの偏差と既存の最大facet budgetで膨張する。速度normは一定加速度
区間で凸なので、budgetに使う厳密上界は`max(||v0||,||v0+a h||)`である。

facet始点`f0`、edge`e`に対し`cross(x(theta)-f0,e)=c0+c1 theta+c2 theta^2=0`を局所scaleで
正規化して解く。判別式が丸めbudget内、pathがfacet lineと共線、根または残差をfloat64で認証できない場合は
no-hitへ丸めず`EventLocationError`とする。候補facetごとのposition/time budget、同時hitの対称結合、
positive-time規則はballistic locatorと共通である。

公開API回帰は`x(t)=0.5+2t-t^2`のright wall first hit
`t=1-1/sqrt(2)`、衝突直前速度`sqrt(2)`を`dt=2,0.5,0.25 s`で比較する。これはendpoint chordが始終点を
同じ`x=0.5`へ結んで壁を見落とすcaseでもある。独立event testはpaddingだけではhitにしないturning no-hit、
`2^-20`だけ接線の両側に置いたnear-grazing hit/miss、接線・共線の明示failureを含む。trial endpointが
field support外でも先行hitを確定できるため、engine順序は`proposal → event → no-hit行validity → commit`とする。
完全な材料boundaryがある場合、no-hitは連続pathがfield coverageを証明済みのparticle domainから退出しないことも
意味する。boundaryless caseは全cell supportedな共通regular layoutに限り、各座標の極値を始点、終点、
`t_turn=-v0/a`がproposal開区間にある場合の転回点から解析的に求め、閉じたregular support box内であることを
検査する。`x(0),x(0.5),x(1)`がsupport内でも頂点がoutsideになる反例はこの極値判定で拒否し、
frame有無で成否を変えない。

`coupled_rk4_engine_v5`で導入しv6でも維持するこの極値区間は、`integrators.py`が所有する。始点・終点に加え、内部転回点を
dense stateと同じ`x0 + t v0 + 0.5 (t t) a`の演算順で評価する。転回値を1 ULPだけ広げる方法では、解析転回時刻の
隣接float時刻でdense stateが区間外へ出る反例がある。このため
`|x0| + h |v0| + 0.5 h^2 |a|`へ、6回のdense式演算、scale構築、転回時刻除算を覆う`16 eps`を掛ける。
さらに`t*t`または`0.5*(t*t)`のunderflowが後続の大きな`|a|`で増幅されるため、
`16 * smallest_subnormal * (1 + |a|)`を加えた絶対roundoff paddingを作り、区間全体を最後に`nextafter`で
外側へ丸める。engineは返された区間とregular
support boxを比較するだけで、別の代数的再編成による頂点を再計算しない。proposal endpointとdense stateの値は
変わらないため`coupled_rk4_proposal_v3`を維持する。

### P06 revision 3a：boundaryless RK4の連続enclosure

revision 3aは、対象をboundaryless `cartesian_xy`、fixed charge、全cell supportedな
`RegularLayout`、既存のEpstein/electric/gravityへ限定する。canonical node値のcomponent-wise extremaは、
support内のregular bilinear補間値をboundするauthorityである。Epsteinの最大緩和rate、gas velocityの絶対上限、
electricとgravity/buoyancyの加算加速度絶対上限を粒子propertyと組み合わせ、RK4の`a1,v2,a2,v3,a3,v4,a4`を
順に保守的に包絡する。粒子ごとのproposal幅を`h`、成分別の絶対上限を`V_i`,`A_i`、globalな加速度bounderを
`B(V)`とすると、実装は非負量について

\[
V_1=|\boldsymbol v_0|,\quad A_1=B(V_1),
\quad V_2=V_1+\tfrac h2 A_1,\quad A_2=B(V_2),
\]

\[
V_3=V_1+\tfrac h2 A_2,\quad A_3=B(V_3),
\quad V_4=V_1+hA_3,\quad A_4=B(V_4)
\]

を順に評価する。そして

\[
R_x=\max\left(\tfrac h2V_1,\tfrac h2V_2,hV_3,
\tfrac h6(V_1+2V_2+2V_3+V_4)\right),
\]

\[
R_v=\max\left(\tfrac h2A_1,\tfrac h2A_2,hA_3,
\tfrac h6(A_1+2A_2+2A_3+A_4)\right)
\]

から`x0 ± R_x`と`v0 ± R_v`を作る。`0 <= s <= h`で係数が単調に増えるため、このboxは任意の短縮RK4の
四内部stageとaccepted endpointを含む。

実装では各非負演算と最終下限・上限をfloat64の外側へ丸め、overflow、非有限、roundoff内で
順序を認証できない値を成功扱いにしない。このbox全体がclosed regular support box内にある時だけ、full proposalと
`StepProposal.state_at()`による全短縮proposalのfield samplingを許可する。stage/endpointの実sample verdictは
実装整合性の検査として残すが、連続証明の代替にはしない。

Epstein applicabilityも有限個のstageだけでは判定しない。node extremaから`lambda_min`、`T_min`、
gas velocity boundを作り、上記全stage・endpoint速度boundと粒子半径から

```text
lambda_min / particle_radius >= 10
relative_speed_upper / mean_thermal_speed(T_min) <= 0.1
```

を全区間で要求する。いずれかを証明できないproposalは、frameの有無や時刻を見る前に同じfailureになる。
revision 3aのboundaryless経路はhidden time subdivisionを行わない。revision 3bの材料boundary経路は元の
macro proposalのparameter部分区間を使わず、各local始点から再積分するsequential dyadic RK4 piece列を
accepted numerical pathとする。分割はgeometry/support/applicabilityだけで決まり、frame scheduleでは変えない。

verificationではC02/C03の公開API解析時系列、frameなし・疎・密scheduleでのfinal position/velocity/chargeの
bitwise同一性、step途中releaseに対する粒子別proposal幅、安全な非一様regular fieldの実行、full-stepの有限sampleは
insideでも短縮pathだけがsupportを外れる反例、Epstein applicability反例のschedule非依存failureを固定した。
`rk4_reintegrated`と材料boundaryの組合せはrevision 3b、exact-mesh P1/Q1材料domainはP06-Uでのみ解禁した。
boundaryless P1/Q1、RZ、continuous chargeはrevision 3a/P06-Uの合格によって解禁しない。

### P06 revision 3b：一般RK4と材料boundary

各local pieceについて、integratorが返す外向き位置enclosureをAABBとしてBVHへ問い合わせる。AABB候補には
facet支持線に対するtubeの法線方向intervalを外向き丸めで求め、facet別event budgetを含めても厳密に分離する
facetだけを除外する。候補がなければ連続pathと全facetの分離が証明された`clear`、候補があり時間・位置budget内で一意なfirst crossingを確定できれば
`hit`、それ以外は`split`とする。`split`は左区間から時間順に二分し、no-hit leafのendpointだけを次pieceの
始点へcommitする。最大refinement内に一意性を証明できない区間をno-hitへ丸めない。revision 3b当時は
run-fatalな`indeterminate_geometry`だったが、P08のengine v14は当該粒子へ局在できる場合だけ
`indeterminate_event`としてfailed terminalへ変換し、他粒子を継続する。

event v5では、`integrators.py`がRK4 velocity enclosureからcomponentwiseなendpoint-chord deviationと
float64 roundoff幅を外向きに与える。`events.py`は候補が単一facetで、chordがstrictに外向き横断する場合だけ、
normal方向進捗とdeviationから、報告hit時刻を中心とするtime error radiusを作る。比較するのは
この誤差半径とtime budgetであり、bracketのfull幅ではない。そのtime shiftによるchord位置差とcomponentwise deviationを
Euclidean radiusにまとめ、facet tangent方向の両endpointから必要clearanceがあることを要求する。certificateを
作れなければ`split`し、time/tube budget内まで細分したpieceは従来のfull-tube判定へfallbackする。
engine v7、proposal v3、enclosure v1、case/result schema、公開APIはこのevent変更では変更しない。

RK stageまたはendpointの非有限値は即時failureとするが、field supportとmodel applicabilityはprovisional flagで
保持する。piece判定を先に行い、先行hitならhit時刻まで同じRK4で再積分してvalid prefixだけを検査する。clearなら
piece全体のsupport/applicabilityを検査してからendpointをcommitする。このevent-before-validity順序により、壁外の
trial tailが先行する正当なwall hitを隠さない。frameは確定済みaccepted pieceを再生し、保存時刻のためにpieceを
分割しない。

公開scenarioは`x''=-4x`の転回を含む軌道と平面壁を使い、`h/h/2/h/4`でhit時刻・hit速度の4次収束、解析値との
一致、frame有無でevent/final/refinement集計がbitwise同一であることを検査する。独立event verificationは始終点を
結ぶchordが静止して見えてもtubeが壁を横断するcaseと、transverse certificateの外向き横断、
time-shift位置radius、facet endpoint clearance、fallbackを含む。

### P06-U：exact-mesh P1/Q1材料domain

P06-Uは新しい積分式を追加しない。required fieldのP1/Q1 layoutがparticle volume meshと完全一致し、全cell
supportedで、材料境界が全外周を所有するCartesian XY caseだけを一般`rk4_reintegrated`へ追加する。始点はstrict
insideであり、既存RK4 tubeがfacet AABBまたは候補facet支持線の認証済み法線intervalから分離したpieceだけを
clearとするため、accepted pathはfield supportから
退出しない。候補pieceはrevision 3bのfirst-event判定へ渡し、壁外trial tailより先のhit prefixだけを検査・commitする。

engine v8はこのcapability arbitrationに加え、要求frame時刻を含むaccepted proposal rowだけをmacro-step末まで
replay用に保持する。proposal v3、enclosure v1、event v5、case/result schemaは不変である。二三角形P1と非affine
mapped Q1の調和振動子wall hitでorder 3.5以上とframe schedule identityを検証する。boundaryless P1/Q1はunion
supportの連続包含証明がないためprepare時に拒否する。

### P06-RZ：signed meridional RK4とaxis event

resident状態とfieldはcanonicalな`r >= 0`のRZ基底を使う。一方、軸を横切るRK4 pieceだけは滑らかな
signed radial chart `(rho,z)`で積分する。signed chartの速度を`w=d rho/dt`、向きを
`s=+1 (rho >= 0), -1 (rho < 0)`とすると、各stageで

\[
r=|\rho|,\qquad v_r=s w
\]

へ写してfieldを`(r,z)`でsampleし、既存physicsへcanonical速度を渡す。得られたcanonical加速度は

\[
a_\rho=s a_r
\]

としてsigned chartへ戻す。`rho=0`のtrial orientationは`+1`とし、accepted stateの内向き速度だけを
axis event後に右連続なcanonical基底へfoldする。physics式、加算順、classical RK4、proposal v3、
global-abs enclosure v1は変更しない。

regular support boxとの照合はsigned enclosureをそのまま使わない。radial区間`[l,u]`のcanonical像は

```text
l <= 0 <= u : [0, max(-l, u)]
otherwise   : [min(abs(l), abs(u)), max(abs(l), abs(u))]
```

であり、axial区間は不変である。速度・加速度の絶対値boundとEpsteinの相対速度normはradial符号反転で
変わらないため、revision 3a/3bのcontinuous applicability certificateを同じauthorityとして使う。
軸上でradial位置・速度がともに0のRK4 stateは不変状態としてclearとし、偽のcrossingを作らない。
`fields.py`の`axis_accessible`は、geometryがaxisへ接する場合だけでなく、boundaryless fully-supported regular
support boxの`r_min=0`も含む。この判定はrequired vectorのaxis regularityだけを所有する。
`gravity_buoyancy_standard_v1`はこれと無関係に、annular domainを含む全RZ caseでradial gravityを厳密に0とする。
regularity検査済みのRZ vectorでも、skew P1/Q1の補間加算によりcanonical `r==0` sampleへ微小なradial丸め値が
生じ得る。`fields.sample`は`axis_accessible`かつquery radiusがfloat64で厳密に0の場合に限り、そのrequired RZ
vector radial sampleをexact `+0.0`へ復元する。これは近軸値を許容差でclampする処理ではなく、axis nodal radial値が
すべて厳密0であるとprepare時に証明したfieldの数学的regularityを補間後も保存する処理である。

axis locatorは既存RK4 pieceの位置・速度enclosureとevent budgetを使う。pieceが軸から確実に離れる場合は
`clear`、全区間でradial速度が厳密に内向きかつ終点がsigned chartの非正側へ進み、chord deviation・丸め残差・
時間誤差がbudget内なら`hit`、それ以外は`split`とする。hit prefixは同じRK4で時刻まで再積分し、support/
applicabilityと`|rho_hit| <= position_budget`を検査してから`r=0`へcommitする。残時間はfold後stateから
同じtargetまで進める。wallとaxisを別々に判定し、片方でも`split`なら分割する。双方が局在済みなら、wallの最早時刻と
axisの最遅時刻の認証区間が重なる場合を含めwallを優先し、axisが明確に早い時だけfoldする。axis端点とmaterial
cornerの完全tieは一般RK4 cornerとして局在不能ならfail-closedとし、wall優先で一面へ丸めない。

解析oracleはradial gas velocity 0のEpstein系である。

\[
w(t)=w_0e^{-t/\tau},\qquad
\rho(t)=r_0+\tau w_0(1-e^{-t/\tau})
\]

`w0 < -r0/tau`なら軸到達後の公開値は`r=|rho|`, `v_r=sign(rho)w`となる。このcaseを`h/h/2/h/4`で
位置・速度とも観測次数3.5以上、frame有無でfinal/event-refinement/axis集計がbitwise同一、axis eventが
boundary event/RNG ordinalを作らないことまで検査する。away-axisではC02～C05のdrag、drag＋一定外力、electric、
gravity/buoyancyをXYとRZへ平行移動し、座標を戻した軌道が一致すること、axis→material wallの順序、軸上radial
不変状態も公開scenarioで固定する。geometryが軸へ接しなくてもboundaryless regular supportが`r_min=0`なら
axis regularityを要求する回帰により、accessibility判定の漏れを防ぐ。

## 3. C06：field補間

P1は二つのCCW三角形`[(0,1,2),(1,3,2)]`からなる長方形mesh上のaffine vector

\[
F_x=1+200x-300y,\qquad F_y=-2+50x+400y
\]

を使う。`q=(0.005,0.0025) m`のbarycentric座標は`(0.5,0.25,0.25)`、期待値は
`(1.25,-0.75)`である。共有edge中央`(0.01,0.005) m`は両cellから評価し、双方が厳密に
`(1.5,0.5)`となることでcell境界連続性も固定する。

Q1は参照順`(-1,-1),(1,-1),(1,1),(-1,1)`に対応する
`[(0,0),(0.02,0),(0.03,0.02),(0,0.02)] m`を使う。写像は非affineであり、quadを三角形へ分割しない。
参照点`(0.2,-0.4)`は物理点`(0.0138,0.006) m`、shape weight
`(0.28,0.42,0.18,0.12)`、field期待値`(1.76,-2.84)`となる。物理点`(0.028,0.01) m`は
参照点`(1.24,0)`でsupport外である。provisional値を返しても、insideへclampして受理しない。

well-conditionedな本caseではfield relative/absolute toleranceを各`2e-12`、座標再構成を`2e-12 m`とする。
supportとcell IDは離散値として完全一致させる。

### P03 `field_location_v2`の実装規則

- `cartesian_xy`の位置・vectorはそのまま扱う。`axisymmetric_rz_meridional`のaccepted stateは`r < 0`を
  `r <- -r`、radial vector成分を反転して正準半平面へfoldする。`r = 0`で内向きradial成分も同じ基底変換を
  行うが、wall eventや位置nudgeにはしない。force-coupledなtrial stageは上記signed chartを使い、stageごとに
  canonical field basisへ写す。
- regular node fieldはC-orderのbilinear補間、regular cell fieldは選択cell値、P1はbarycentric補間、
  Q1は参照要素上のisoparametric bilinear補間を使う。Q1を三角形へ分割しない。
- P1/Q1のsupport包含は、CCW physical polygon各辺へのsigned distanceを、座標ULPと局所element径から
  解決した物理誤差budgetと比較して判定する。参照座標だけの固定許容差は使わない。field locationのbudgetは
  event局在budgetと分離する。
- 逆写像は共通offsetを除いた局所座標で行い、物理誤差を局所`||J^-1||_inf`で参照空間へ写す。
  `cond_inf(J) <= 1/sqrt(eps)`、参照不確かさ`<= 8 sqrt(eps)`、局所再構成残差を全て満たす時だけ
  補間basisを認証する。境界と認証済みの場合だけP1 weightをsimplexへ、Q1座標をreference squareへ
  丸め、outsideをclampでinsideへ変えない。
- required fieldのprepareは全P1/Q1 cellへ同じJacobian条件を先に適用する。regular layoutも各axis cellで
  `64*max(axis ULP, eps*axis span)/cell width <= 8*sqrt(eps)`を要求する。共有layout自体が解像不能なら
  particle-local support failureへ落とさずprepare-fatalとする。
- supportはsupported cellの閉包の和である。同一点を含むcandidateのうちsupported cellが一つ以上なら
  `support_inside=true`とし、最小supported cell IDをownerにする。全candidateがmaskedなら
  `masked_cell`とするが、値ownerは全supported cellのうち物理閉包へのEuclidean距離が最小のcellとする。
  layout外も同じ射影を使い、同距離は最小cell IDで決める。masked placeholderと外挿weightは使わない。
  この規則により、固体側がmaskedでも流体側wall面から放出する粒子が`t=0`に有限な場を評価できる。
- `cell_hint`は探索高速化のhintに限り、cell ID、support、補間値を変えない。共有面上のcell-associated値は
  上記owner規則で決定論的に選ぶ。
- 共有面連続性は、P1の両三角形と、少なくとも一方が非affineな隣接Q1の各cellを一要素layoutとして
  同じ物理点でproduction samplerへ与え、解析field値とsupportが両側で一致することを検査する。
- layout外でもtrial判定用のprovisional値と`support_inside=false`を返せるが、そのproposalを受理しては
  ならない。supported cellがない、float64でcellを解像できない、Q1逆写像が収束しない、または補間が
  非有限になる場合はoutsideへ偽装せず`FieldLocationError`にする。

### P03 revision 2の検証範囲

実装済みverificationは次を含む。

- 同じP1/Q1 elementとqueryを大きく平行移動してもowner/support/補間値が変わらない。
- high-aspect elementの厳密なedge pointをinside、scale-aware budget外をoutsideと判定する。
- 共有面の片側だけがsupportedでも、そのsupported cellの閉包上をinsideとする。
- outside provisionalは物理距離が最小のsupported cellへlocal coordinateを射影して構成し、masked cellの
  placeholderを物理値として補間しない。
- 有限入力から非有限なsampleを返さず、計算可能でなければ明示errorにする。
- canonical orientation計算自体がoverflowまたはNaNになるlayoutをschema validationで拒否する。

平行移動不変性は、移動後も局所geometryがfloat64誤差budget内で保存される範囲の離散owner/supportについて
保証する。微小cellが座標量子化で解像不能になる任意の巨大移動まで保証せず、その場合は明示errorにする。
revision 1は削除済みで、runtime selectorはない。

field spatial-gradient revision v2は外部cache検証で要求された時だけ、同じsampleのcell ID/basisから
static P1/regular/Q1 nodal fieldを解析微分する。preprocessorは隣接snapshotのstatic viewを同じsamplerで評価し、時間Gram momentsを外部で構成する。第二locator/inverse map/常時stage-gradient scratchを作らない。
stored componentのXY/RZ偏微分であり、3D共変微分やCOMSOL recoveryではない。
preprocessor v3はP1/regular/exact affine Q1→full regularのstatic/linear-time cacheを共通partitionで認証する。
weighted value/gradient/support-boundary normにroundoff allowanceを加え、全時間区間のerror/reference二次norm比を
bounded Bernstein上界で判定する。stationary rootはreport-only。warped Q1/partial target、解像不能patch、coverage
不整合、reference下界不足、資源超過は無公開。詳細はtools/field_preprocessor/README.mdを参照する。

### Stage 4A fixed-topology linear time interpolation

canonical data schema v3ではfieldごとにoptionalなsnapshot軸`time_s[T]`を持てる。時刻軸がない
`values[N,C]`は静的場、時刻軸がある`values[T,N,C]`は固定layout/topologyの時間依存場である。
`T>=2`、時刻はfiniteかつ狭義単調増加とし、全snapshotは同じassociation、components、basis、unitを共有する。

stage時刻を \(t\in[t_k,t_{k+1}]\)、既存regular/P1/Q1/cell空間補間作用素を \(S_x\) とすると、

\[
\alpha={t-t_k\over t_{k+1}-t_k},\qquad
F(x,t)=(1-\alpha)S_x(F_k)+\alpha S_x(F_{k+1}).
\]

locator、owner、空間weightはstage位置で一度だけ求め、二snapshotへ同じ値を使う。したがってP1/Q1の
conditioning、support、共有面owner、RZ basis/axis regularityは`field_location_v4`をそのまま使い、時間専用の
第二samplerを作らない。prepare時のrun-global primitive boundは引き続き全snapshotのcomponent extremaを
外向きに含む。一方、local continuous-path certificateは各proposal部分区間`[t_a,t_b]`と重なる線形時間区間を
挟むsnapshotだけを使う。時間補間と空間形状関数の凸性により、そのsnapshot extremaは区間内の補間値を包含する。
snapshot範囲外を外挿して証明せず、区間が範囲外ならfail-closedにする。

snapshot端点は閉区間として受理し、範囲外は外挿・clampせず`FIELD_KERNEL_TIME_OUTSIDE`でfail-closedにする。
required fieldはprepare時にrunの閉区間全体を覆う必要がある。積分器はactual stage timeをsamplerへ渡し、
required temporal fieldのrun内部snapshot knotの和集合を固定`dt_s` gridへ併合する。これにより係数の傾きが
変わるknotを一つのmacro intervalが跨がない。gridと一致するknotは重複境界にせず、静的場だけのcaseは従来の
macro partitionを保つ。

verificationはregular/P1/Q1の解析的space-time affine field、全snapshotを含むglobal bound、interval-overlapping
bracketing snapshotだけを含むlocal bound、範囲外status、および非grid knotを持つ区分線形一様電場の解析trajectoryを
含む。20×20 regular grid、4,096 local row、256 snapshotのwarm characterizationでは、同じmethod/candidateに対する
all-snapshot local bound `0.010164 s`からinterval-bracketing `0.001303 s`へ`7.80x`だった。これは同一machine上の
非gating測定であり、portableまたは普遍的なspeedupを主張しない。zero-order hold、discontinuity、moving topology、
streaming/double buffer、RZ fieldを回転展開するCartesian 3-Dはこのrevisionの数値契約外である。

## 4. C07～C10：境界event

- C07：unit square内の`x0=(0.25,0.25) m`, `v=(0.5,0.25) m/s`。最初のhitは
  `t=1.5 s`, `(1,0.625) m`、right facet、外向き法線`(1,0)`。dimensionless charge numberは
  `Z=3`のままstick前後で不変とする。
- C08：surface sourceが明示的に予約した`particle_id=801`を使い、left wall中央からdomain内向き
  `v=(1,0) m/s`でreleaseする。`t=0`をhitにせず、
  `t=1 s`でrightへ反射、`t=2 s`でleftへ再衝突してstickする。固定位置nudgeを使わない。
- C09：gap `g=2^-10 m`、初期`x=g/2`、`v_x=1 m/s`、一macro step
  `21g/4 s`。right/leftへ5回反射し、終端は`x=g/4`, `v_x=-1 m/s`。event budget枯渇時に途中結果を
  正常受理しない。interaction上限へ達した後、残区間に次のhitが存在すると判明した場合だけ残時間を
  二分して再試行し、残区間がevent-freeならそのまま受理する。成功caseの上限8とは別に、
  interaction上限1かつrefinement上限2のsuccess probe、上限1かつrefinement上限1のfailure probeを同じ
  oracleへ持つ。必要な再分割を完了できない時は`numerical_event_budget`とする。上限到達だけで直ちに
  失敗させず、未処理eventを黙って捨てない。
- C10：一つ目の三角形`(0,0),(3,0),(1,1)`の非直交cornerへ`(1,0.25)`から`v=(0,1)`で到達する。
  `t=0.75 s`でright/leftのcandidate集合を保持し、後述policyの完全反射後は
  `v=(1/sqrt(10),-3/sqrt(10)) m/s`となる。平行移動した二つ目の三角形では、同時hitした異なるgroupを
  priority 5の`stick`とpriority 20の`specular`に分け、前者だけが一回適用されることを固定する。
  同priorityでlawが矛盾する場合のdirect policy failure probeも持つ。

engine v9ではC08～C10を`load_case -> simulate -> open_result`の公開経路で実行する。対象はforce-freeな
`linear_exact` pathであり、wall上release、specular/probabilistic stick、複数hit、corner、
cap-driven residual splitを一つのproduction loopで扱う。hit位置はcanonical wallへcommitし、release時にも
reflection後にもposition nudgeを使わない。engine v10 / event v6はCartesian XYの証明済み一定加速度surfaceへ
同じexact-path意味論を拡張した。engine v11 / event v7でCartesian XY一般RK4へ、厳密内向きsurface
departureとsingle-facet activeな壁応答後の残時間継続を拡張した。接触facetの除外を証明できない区間は近似的にclearとせず、
細分化しても決定不能なら失敗する。

oracleの`boundary_events`は完全logそのものではなく、標準event logをboundary行へ絞った正規化比較projection
である。標準列の物理量はそのまま比較する一方、`boundary_event_ordinal`は粒子ごとのboundary行だけで
0から振り直し、candidate tableの`facet_id`はlistへ展開する。resolved lawは標準logと同じ`law_id`で表す。
comparatorはこれらを標準logから明示的に構成し、標準`event_ordinal`へ直接突き合わせない。release/failure行を
含む完全logの順序とschemaは後続の公開API scenarioが別に検査する。候補集合、件数、順番、law ID、lifecycleは
完全一致させる。
eventの`normal`はlaw適用に実際に使ったeffective response normalである。単一facetではその外向き法線、
combined-normal反射では選択subsetの正規化合成法線を保存する。
局在値は要求budgetの4倍以内を受入条件とし、normalと衝突前後速度はoracle記載の絶対差、電荷とweightは
完全一致で比較する。C09/C10の終端状態は各`final_acceptance`だけを使う。解析期待値との比較とruntime
局在budgetを混同しない。

## 5. event budgetの単一解決規則

YAMLは`geometry_rtol`、`roundoff_ulps`、`max_refinements`、`max_interactions_per_step`、
`corner_policy`を明示する。P02 caseは順に`1e-12`, `64`, `48`, `8`,
`priority_then_combined_normal_v1`を使う。これらはmicrocase値であり、全製品caseへ無条件に適用する
defaultではない。

決定論のexact／RK4／exponential経路では、`max_interactions_per_step`はmacro step全体のhard hit総数ではなく、一つのresidual-work intervalを
二分するtriggerである。子intervalのinteraction countは0から始め、refinement depthは引き継ぐ。したがって
macro step全体ではこの値より多い物理eventを処理できるが、再分割depthを使い切れば
`numerical_event_budget`になる。

上限に達したintervalでは、残区間にさらにhitがあるかをfirst-hit queryで判定する。hitがなければ
event-freeな残区間を受理し、追加depthを消費しない。次のhitが存在する場合だけ、現在のaccepted boundary
stateを起点に残時間を等分し、depthを一つ増やして前半、後半の順で処理する。次の分割depthが
`max_refinements`を超えるなら、最後にaccepted
された状態と処理済みevent prefixを保持したままrunを失敗させる。C09の上限1・depth上限1 probeでは
`0.5g`、`1.5g`の二eventを処理した時点で失敗し、次の未処理hitは`2.5g`である。depthを十分にした同じ
上限1 probeはbase caseと同じ5 eventと終端へ到達しなければならない。これにより「上限到達で即失敗」と
「未処理eventを捨てて成功」の両方を検出する。

engine v11の一般RK4も同じ規則を使う。interaction countは一つのparticle residual stateがsurfaceのzero-time
interactionと後続hitを通して保持する。上限値に達しただけでは分割せず、現在stateから提案した区間で次hitが
実際に局在された場合だけ、その区間をleft-firstに二分して子stateのcountをリセットする。event-freeな残区間は
余分なrefinement depthを使わず受理する。

Brownian経路では同じ設定値の役割が異なる。active surface、材料wall、RZ axis、periodic transfer後の
fresh stochastic rootへのrestartをmacro step内で数え、上限を超えたcountを持つ次rootの入口で
`numerical_event_budget`とする。直前の応答とaccepted prefixは保持する。
intervalの二分でこのrestart countをリセットせず、countが上限と等しいrootは処理できる。
support／applicabilityに伴うguard restartは`max_refinements`で別に有界化する。
axis foldはrestartには数えるが、材料wall eventやwall RNG ordinalを生成しない。
失敗reasonをphysical escapeへ読み替えず、このcapとfirst-hitの局在精度を区別する。

P05の単一resolverは、局所facet長さ`l>0`、geometry bbox対角`L`、評価位置`x`、速度scale`V`、
区間幅`h>0`、時刻`t`、float64 epsilonを用い、位置と時刻の丸め幅を

\[
\rho_x=\max\left(
U\epsilon\max(L,\|x\|,l),\;
U\max(\operatorname{ulp}(x_0),\operatorname{ulp}(x_1),\operatorname{ulp}(L),\operatorname{ulp}(l))
\right),
\]

\[
\rho_t=\max\left(
U\epsilon\max(|t|,h),\;
U\max(\operatorname{ulp}(t),\operatorname{ulp}(h))
\right)
\]

として、

\[
x_{budget}=\operatorname{nextafter}(r l+\rho_x,+\infty),
\]

\[
t_{budget}=\max\left(
\frac{x_{budget}}{\max(V,l/h)},\;
\rho_t
\right)
\]

と解決する。物理項`r l`とarithmetic項`rho_x`はmaxではなく加算し、その和を正の無限大方向へ
1 float64 step丸める。ULP項はsubnormalやゼロ近傍でも正のbudgetを失わないための下限である。
boundary、integrator、test helperへ別のmagic toleranceを置かない。最大反復またはinteraction
budgetを使い切った場合は、no-hitやstickへ丸めず、accuracy/safetyを区別した明示failureにする。

`geometry_rtol=r`は局所facet長`l`に対する相対値であり、製品caseでは対象の物理・mesh scaleから選ぶ。
`r l`が`rho_x`を十分に上回らない場合、event判定はarithmetic-limitedであり、保守的な包絡が
boundary bandと重なると`indeterminate_event`でfail-closedし得る。時刻二分の中点が隣接float64間で
潰れた後は`max_refinements`の増加では解決しない。`dt`半減も位置budgetのarithmetic床を下げず、
確率軌道では離散経路自体を変えるため、tolerance未解決の修復手段にしない。

外部V&Vでevent toleranceを設定適格化する場合は、隣接decadeの`geometry_rtol`を事前宣言して比較する。
facet identity、law、outcome、最終fateの一致と、位置・時刻差を両runの記録済みbudgetの大きい方で
正規化した比が事前宣言上限内であることをgateとする。COMSOLまたは他の外部結果へ合わせて
`geometry_rtol`をfitしない。

P05のballistic pathでは`V=||v||`が区間全体の厳密なspeedであり、pathは直線segmentそのものである。
P06以降の曲線pathでは`V`をcandidate facetごとに作らず、同じStepProposal path piece全体に対する
保守的なspeed upper boundをintegratorが提供する。endpoint速度だけで代用しない。facet `i`にはその長さ`l_i`から個別budgetを作り、
候補`i,j`の同時性比較には位置・時間とも`max(budget_i,budget_j)`を使う。この対称結合によりBVH走査順や
どちらをprimaryと見たかで候補集合を変えない。P06 revision 2では上記の厳密放物線だけを実装済みであり、
`V=max(||v0||,||v1||)`が全区間の厳密上界になる。revision 3aではboundaryless regular supportで一般RK4の
連続field-supportとmodel applicability enclosureを完了した。revision 3bでは各local始点からglobal boundを再構築する
sequential accepted RK4 pieces、一般壁面の離散tube、event-before-validity、証明不能時のfail-closedを追加した。
同一macro proposalのparameter区間は局所tubeとして流用しない。
full-step/two-half-step差だけを証明済みの軌道またはsupport包含boundとして扱わない。

最大facet長から解決したbudgetはBVH broad phaseの候補漏れ防止だけに使う。初期位置のfacet距離、
near-parallelなline separation、交差parameterは各候補facetの長さから解決したbudgetで判定する。無関係な
巨大facetを追加しても局所的なinside/no-hit判定を変えてはならない。またsegment区間外の交点を端点へ丸める
場合は、無次元parameter slackを`min(position_budget/path_length, time_budget/interval)`とし、空間・時間の
両方を満たさない交点を早いeventとして受理しない。

## 6. corner policy revision 1

現行wall-law algorithmは`contact_wall_laws_v7`である。corner selector
`priority_then_combined_normal_v1`は次で固定する。

1. 最早hitとの差がresolved time budget内で、局在点がposition budget内の全facetをcandidateに残す。
   candidateのidentityは`boundary.line2`のcanonical row、すなわち`facet_id`であり、複数facetが共有できる
   `boundary_id`で代用しない。
2. 全candidateを`(priority, boundary_id, facet_id)`の昇順に並べ、小さい`priority`のsubsetを選ぶ。
   HDF5の作成順、BVH返却順、thread順は使わない。
3. 選択subsetのresponse signatureが互換でなければ`ambiguous_boundary_law`とし、任意の一面を選ばない。
   決定論的な`specular`と`restitution`は、解決済み`normal_restitution`と`tangential_restitution`が
   ともに同じ場合に限り互換である。このためparameterなしの`specular`は`restitution`の1/1と互換になる。
   `maxwell_thermal`同士は`wall_temperature_K`、`diffuse_reflection_fraction`、`wall_velocity_m_s`がすべて
   同じ時だけ互換である。`probabilistic_stick`はtop-level law、`probability`、fallback law、そのfallbackの
   解決済み反発係数またはMaxwell parameterを含む完全なsignatureを一致させる。`stick`と`escape`も互いに
   別signatureである。
4. 互換なlaw responseは一回だけ適用する。反射responseでは、安定sort済みfacet順に入射中の外向き
   unit normalを加算し、正規化したeffective normalへ解決済み反発係数またはMaxwell transformを一回だけ適用する。
5. normal和が丸め範囲で0、または非terminal lawの応答後速度がpriorityで除外した面も含む
   **全candidate facet**のいずれかへ外向きなら
   `indeterminate_boundary_policy`とする。
6. priorityで選ばなかったものを含む全candidate集合をstable順でeventへ保存し、physical boundary event
   ordinalは一回だけ進める。`primary_facet_id`は選択subsetの先頭とする。

これは非滑らかcornerの唯一の物理法則を主張するものではなく、初期製品の明示的で再現可能なpolicyである。
別policyが必要になった場合はrevisionを分け、同名の意味を変更しない。

### 6.1 Static XY pure-translation periodic topology

`translation_periodic_xy_v1`は材料wall lawではなく、static Cartesian XY geometry上のtopology transferである。
prepareは各pairのfacet数、translation後の端点・長さ、反対向き外向き法線、一意なgroup所有、required nodal fieldの
seam一致を検査する。regular fieldは一軸方向のsupport-box対向面だけを受理し、時間依存field、RZ、回転・反転・scale、
moving topologyを拒否する。

正の`contact_radius_m`を持つ粒子では、材料facetのsegment-capsule contactと周期facetのparticle-centre crossingを
別々に局在し、認証済みの早い方だけをcommitする。両時刻のuncertainty intervalを順序付けできなければ曲線pathは
refineし、budget内で解けなければfail-closedにする。候補集合が材料facetと周期facetを混在させる場合、または周期facetが
異なるtranslationを要求する場合も任意の一面を選ばない。同じtranslationを持つsplit-node facet群だけは一回のtransferとする。

transferは出発面と到着面のcanonical projectionを使い、固定nudgeを加えず、位置だけをtranslationする。速度、電荷、
model weight、残時間は維持し、field cell hintは無効化して到着側で再探索する。logical `event_ordinal`と一intervalの
interaction budgetは一回進めるが、physical wall ordinal、wall RNG、wall counterは進めない。Brownianではhit prefixだけを
commitし、到着stateから残時間をfresh stochastic rootとして再開する。topologyを設定しないcaseは従来の候補packingと
wall hot pathをそのまま使う。

C10の`policy_failure_probes`はcanonical full-run variantではなく、P07で`boundaries.py`のpolicy数値境界へ
直接与えるunit inputである。materialized C10本体はcombined-normal成功と異priority terminal選択を
`load_case -> simulate`で検査する。synthetic probeは同priority law矛盾に加え、低priority subsetへlawを
適用した後の速度がpriorityで除外したcandidateへ外向きになる場合の`indeterminate_boundary_policy`も固定する。

## 7. P07 source、RNG、RZ axisの実装範囲

H2の`contact_geometry`はboundary groupごとに解決し、未指定は`particle_surface`とする。
`particle_center`では候補の有効半径だけを0とし、粒子の物理半径は保持する。source offset、clearance、
candidate normal、canonical hit残差も同じfacet maskを使う。一つのBVH上で中心eventと材料capsule eventを局在し、
早い方をcommitする。時刻順が局在budget内で未解決な混在候補は既存split/failureにし、lawやpriorityから方式を推測しない。
exact contact-setの離脱証明は端capを含むconvex supporting planeへ一般化した。反射originを再利用するmacroでは、
実接触の有限contact-set状態をcheckpointと共に保持し、元sourceのfacet/modeで再解釈しない。
曲線capsuleの保守的な離脱認証、finite Hermiteのfirst-arrival制約、wall RNGの物理ordinalは維持する。

tableとsurfaceは`ParticleSchedule/realize_sources`へ一度だけ変換し、resident rowはparticle ID順、release workは
`(release_time, particle_id)`順とする。runtimeはcanonical surface rowのfacet IDとfacet内座標から位置を導出し、
速度、release時刻、物性をそのまま使う。外部preprocessorがRZ回転面上の一様fluxを必要とする場合は、
facet weightに定数`2π`を除く

\[
w_f=L_f(r_0+r_1)
\]

を使い、選択facet内もradiusへ比例させる。例えばopen uniform \(u\in(0,1)\) から

\[
r=\sqrt{(1-u)r_0^2+u r_1^2},\qquad
s=\frac{u(r_0+r_1)}{r_0+r}
\]

でline parameterをrealizeできる。この標本化はsolver coreでは実行しない。canonical surface scheduleはfacet IDとstrict interiorのline parameterを保持し、
position、owner、normalはprepared geometryをauthorityとして一度だけ導出する。velocity、release時刻、粒子物性は
すべて粒子ごとの入力rowであり、solver内では分布を標本化しない。

counter RNGは`philox4x32_10_v2`とし、wall lawの独立streamにだけ使う。wall drawは
`seed, particle_id, physical_boundary_event_ordinal, law_stream`から決める。physical ordinalは実際に応答を適用した
hitだけで進み、first-hit refinement、cap-driven split、frame出力では進めない。既知vectorとstream分離を
verificationし、mutable global RNGは持たない。sourceを確率的に作る必要がある利用者は外部preprocessorでrealizeし、
その結果をcanonical HDF5へ固定する。

wall streamは`wall_probabilistic_stick`、`wall_maxwell_diffuse`、`wall_maxwell_normal`、
`wall_maxwell_tangential`を分離する。法線drawはstrict open uniform、接線drawは同じPhilox blockの二つのopen
uniformをBox--Muller変換したstandard normalである。Maxwell transform自体はRNG policyを持たず、
`boundaries.half_range_maxwell_flux_velocity`が\(\boldsymbol t=(-n_y,n_x)\)と
\(\boldsymbol v'=\boldsymbol v_w-c_n\boldsymbol n+c_t\boldsymbol t\)だけを所有する。batch wrapperは粒子mass、
外向きnormal、二つのdrawをrow-wiseに受け、scalar wall temperature/velocityを共有する。この変換は入力作成側が
明示速度をrealizeする際にも再利用できるが、source RNG streamやsource selectorはcoreに持たない。surface releaseの
速度はcanonical rowをそのまま使い、zero velocityをnudgeで補わない。

RZ meridional pathが`r=0`へ到達した場合は、最初の材料hitとaxis時刻を同じbudgetで比較する。
材料hitが先ならwall lawを適用し、axisが先ならwall eventを作らず`r`とradial velocityをfoldして残時間を進める。
axis crossingはmanifest集計にだけ加え、boundary ordinal、wall RNG draw、interaction countを消費しない。
force-coupled pathではP06-RZのsigned stage mapping、axis regularity、canonical enclosure像、event v8の
clear/split/hit判定を使い、force-free exact pathと同じright-continuousなfold意味論へ合流する。

一定加速度surfaceのdepartureは外向き法線`n`に対し、次の順で分類する。

1. `v·n`がscale-aware budgetより負なら内向きdeparture、正ならzero-time impact。
2. `v·n`がbudget内でも非零なら、加速度で符号を上書きせずfail-closed。
3. `v·n`がfloat64で厳密に0の時だけ`a·n`を同様に分類する。負なら内向きdeparture、正ならzero-time
   impact、budget内ならfail-closed。

内向き加速度の時、engineの分類だけでsource facetを検索対象から外さない。departure tokenは別wall hitで速度が
変わるまでmacro intervalをまたいで保持する。event v6は各intervalで、開始点がsource supporting lineの内側または
境界bandにあること、法線速度が厳密tangentまたは明確に内向きであること、法線加速度が明確に内向きであることを
同じbudgetで再証明し、成立するintervalだけsource facetを候補から除外する。他facetは常に通常のfirst-hit検索へ
残し、別wall応答後はsource facetも通常検索へ戻す。この寿命規則によりreleaseがmacro終端なら次macroで再証明し、
微小`dt`を含む時刻partitionで軌道/eventを変えない。releaseがrun終端なら運動queryはなく、right-continuousな
release frameだけを保存する。外向き加速度のzero-time impactではterminal stick/escapeを適用できるが、反射応答が
内向きdepartureを作れなければ失敗する。

一般`rk4_reintegrated`のsurface/single-facet active-boundary departureはCartesian XYだけを対象とする。初回surface releaseは
瞬間の`v·n`がscale-awareに厳密な負値の時だけ受理し、tangentまたは曖昧な値を局所加速度で推定しない。surfaceの
zero-time反射または通常hitのactive応答後も、post速度が明確に内向くことを要求する。event v7は各RK4 pieceで、
開始点がcandidate facetのsupporting line上かつ端点budgetから離れ、区間全体の速度包絡から法線速度上界が厳密に
負であるfacetだけをstart-contact候補から除外する。facet端点/cornerはこの証明対象外である。他facetと後刻の再衝突は通常どおり検索する。証明不能なら候補を
保持して細分化し、refinement budgetまで曖昧ならfail-closedとする。

非終端応答ではhit状態へcanonical wall projectionを適用した後、そのpost-stateから元のtarget時刻までRK4を
再積分する。accepted pre-hit pieceはtrajectory replayに残し、同時刻のactive state jumpがpost-stateで上書きする。
terminal stateも同時刻から有効なので、surface release、stick、escape、hold、reflectionのframe意味論は右連続である。
holdはhit状態を保持したinactive terminalで、残時間や後続macroで物理更新しない。

## 8. P08 particle-local failureと出力schedule

engine v14で導入し、現行engine v46が維持する規則は、有限な開始stateから一粒子へ再現可能に局在できる
event budget枯渇、event/boundary/departure不決定、
動的field support/model applicability逸脱、particle physicsの非有限導出値だけをfailed terminalへ変換する。
field、physics、integrator、enclosureは同じbatch pass内のrow statusへ最初のfailure ownerを書き、有限placeholderで
後段配列のshapeを保つ。phase barrierが失敗rowだけをparticle-local failureへ変換し、productionでscalar再実行しない。
共有shape、model revision、field layout/conditioning、global coefficient/bound、topology、I/O不変条件は局在させず
run-fatalである。support不成立行をmodel評価へ流さず、accepted wall prefixは後続policyが失敗してもframe replayと
`accepted_particle_pieces`集計の双方へ一度だけ残す。

failure、trajectory frame、probeはproduction stepを増やさない。frame/probeは確定済みaccepted pieceを同じ
integrator規則でpreallocated columnar slotへ直接再生し、同時刻のaxis/boundary response、terminal stateの順に
右連続で上書きする。`_AcceptedPiece` / `_StateJump`のPython objectは保持しない。したがって
frame/probe scheduleの追加はsurvivorのfinal、event、failure理由、refinement partitionを変えない。

## 9. P09 runtime work partition

P09は運動方程式、RK4 stage、first-hit、wall law、RNG keyを変更しない。resident position/velocity/property
列のparticle ID対応はrun中固定し、別のresident-row active indexだけをparticle ID順に保って
stable compactする。各macro intervalのactive rowはmemory planが決めたmicrotileへ分けるが、粒子ごとの
proposal/event/refinement順は不変である。tile-local結果は物理時刻、particle ID、ordinalの既存keyで
stable mergeし、tile幅からscientific resultを独立にする。

`solver_owned_memory_plan_v1`は数値algorithmの許容誤差ではなく、solver-owned配列のload/prepare/run
phase peakを予測するruntime planである。最小1粒子のscratchも入らないcaseは運動を開始せず拒否する。
process RSSはPython/native allocatorを含むため別に測定し、plannerと同じ値だと主張しない。
output replayはframe/probeのmacro内最大時刻数を使い、path分割境界が両端inclusiveであるため前後2 pieceまで
計上する。分割不能なboundaryless XY pathは時刻数にかかわらず1 proposalだけを計上する。
P09時点のNumPy engineのevent stagingは設定interaction capの一窓と必要なoutput replayを予測する。これは
pathologicalなresidual split/axis eventまで含むprocess RSSのhard capではない。P12はworker-local
residual/event/statisticsと全worker scratchをmemory planへ分離した。P13はevent/failureをworker waveごとに
容量1 writer queueへ渡し、前waveを次waveまで保持しない。memory plan v4はその最大payloadを
`worker_output_staging`へ分離して計上する。accepted replayは要求frame/probeに必要なpieceだけを保持する。

## 10. P10 compiled CPU parity

P10は数値式、stage順、proposal、event判定、accepted-state規則を変更せず、field location/interpolation、
sample済みprimitiveからのphysics、classical RK4算術をNumba array passへ移した。Numba 0.67、NumPy `<2.6`、
`fastmath=False`、`parallel=False`を固定し、scalar evaluatorはverification oracleだけに残す。compiled passが失敗した
行をscalar production経路へ黙ってfallbackしない。

regular layoutはsupported containing-cell common pathだけ軸indexからO(1)個の候補を評価する。outside/masked
provisionalは`field_location_v2`の物理最近傍規則を守るためcompiled全cell走査を使う。P1/Q1はresidentの
previous-cell hintがstrict interiorを含む場合だけそのcellを採用し、初回sample、hint miss、共有面は同じNumba
kernel内のfull searchで最小supported ownerを選ぶ。このhintは非物理stateであり、accepted full endpoint、
wall/axis prefix、residual pieceのaccepted endpointだけが更新する。trial stage、`StepProposal.state_at_rows()`、
frame/probe/output sampleはhintを変更しない。

したがってwall hitまでの再積分、hit後residual、trajectory replay、frame/probe/finalは従来と同じproduction
proposal/event経路を通る。engine v16、compiled CPU tile v1、runtime layout v2、memory plan v2、physics runtime
v2はこのwork partitionを表し、proposal v3、event v8、field location v2、case/result schemaの意味を変えない。
P1/Q1の初回stageは全粒子がhintなしfull searchになるため、large-mesh性能はP10の完了条件に含めない。
realistic cell count、initial localization、cross-cell motionはP14で測り、支配的だったsupported containmentへBVHを追加した。

## 11. P11 native exponential midpoint

P11は`dv/dt=-lambda(v-u)+a_add`を一step内でmidpoint係数へ凍結し、線形drag部分を解析的に
更新する。physics runtimeが同じcompiled model passから`lambda`、`u`、`a_add`と合成加速度を返すため、
RK4と指数法でdrag式、field sampling、model applicabilityを二重実装しない。

step幅を`h`、`z=h lambda`、`A=1-exp(-z)`とすると、一定係数更新は

\[
v_1=e^{-z}v_0+A u+h\phi_1(z)a_{add},
\]

\[
x_1=x_0+h\phi_1(z)v_0+h(1-\phi_1(z))u+h^2\phi_2(z)a_{add},
\]

\[
\phi_1(z)=\frac{1-e^{-z}}{z},\qquad
\phi_2(z)=\frac{z-1+e^{-z}}{z^2}
\]

である。実装はstart係数によるhalf-step predictorを一度作り、その位置・速度・時刻でfield/physicsを
再評価して得たmidpoint係数でfull stepを計算する。`z`が小さい時は`phi_1`、`1-phi_1`、`phi_2`の級数を
使い、通常域は`expm1`を使う。`lambda=0`では`x=x0+h v0+h^2 a_add/2`、`v=v0+h a_add`へ連続に退化し、
大きい`z`でも`tau=1/lambda`を先に作らない。

`exponential_midpoint_step`も現行`coupled_fixed_step_proposal_v10`の`StepProposal`を返す。
`state_at()`、wall/axis hit時刻までの短縮再積分、hit後の残時間、trajectory/frame/probeは同じmethodを
proposal始点から再評価する。現行`exponential_midpoint_local_stage_enclosure_v4`はv3のrun-global包絡を
常にfallbackとして保持し、start half-step predictorが通る局所field rangeで認証したstage force boundとの
共通部分だけを採用する。局所range、force算術、stage enclosureのいずれかが非有限・未証明ならglobal包絡を維持し、
狭めるために安全性を弱めない。連続applicabilityもglobal-firstで、未証明rowだけ局所primitive rangeと
prepared charge invariantで再認証する。event locatorへ渡す曲線/chord偏差はposition box全幅ではなく、
全短縮位置secantを含むvelocity enclosureに対して成分ごとに
`h * (v_upper - v_lower) + float64 roundoff padding`で包絡する。このmethod-neutral boundを
RK4と指数法の共通first-hit/refinement loopで用い、材料壁とRZ axisを調停する。第二event engine、
endpoint chordだけのno-hit判定、出力時刻によるproduction step分割は追加しない。

受入条件は次のとおりである。

- C03一定係数を`dt=[0.75,0.375,0.125] s`で解析解の丸め誤差内に再現する。
- `z`の0近傍から非常に大きい値までposition/velocityがfiniteで、`lambda -> 0`の極限が連続である。
- smoothな可変係数問題でstep半減時の位置・速度の観測次数を1.8以上とする。
- frame/probe scheduleを変えてもfinal/event/failure/refinement identityを変えない。
- Cartesian材料first hit/反射残時間とRZ axis→wall順序を既存event経路で再現する。
- surface releaseが同じfacetへ再衝突するcaseでzero-time departureと後刻hitを区別する。
- Stokes--Cunningham一定primitiveも同じ指数更新で閉形式解へ一致し、C03は要求全frameで解析解へ一致する。

RK4の`maximum_dt_over_tau < 2.5`は陽的`rk4_fixed`だけの安定性gateであり、指数法へ適用しない。
一方、P11はcontinuous chargeの局所解析更新をまだ定義しないため、stageの`charge_rate_number_s`が非零なら
明示的に拒否する。chargeだけのsubcycle、fixedへの読み替え、RK4への暗黙fallbackは行わない。

P11完了時点のrevisionはengine v17、compiled tile v2、proposal v4、event v9、physics runtime v3、
`exponential_midpoint_v1`、指数enclosure v1である。case/result schemaとphysics model revisionは変えない。

## 12. P12 event-heavy worker partition

P12は粒子ごとの数値順序を変えず、active rowをworkerあたりのmicrotileへ分け、最大`W` tileを一waveとして
実行する。一つのparticle IDは一wave内外を通して同時に複数workerへ渡さない。field、prepared geometry、physics
runtimeはread-only共有し、各workerはproposal scratch、BVH stack/candidate buffer、residual work、event/failure、
statisticsを局所に所有する。main threadはfutureを完了順でなく元のtile順に回収し、その後に既存の物理keyでstable
mergeする。writer、resident active compact、macro-step lifecycle確定はmain threadだけが行う。

`resources.threads=T`はworker数の上限で、粒子ありなら`W=min(T,N)`、空なら1とする。memory plannerは
`W * microtile_particles * scratch_bytes_per_particle`と、workerごとのBVH traversal stack/facet candidate bufferを
同時に計上する。指定上限へ入らない時に`W`を黙って減らさず、運動開始前に拒否する。microtile幅だけを安全な範囲で
縮めるため、memory設定が科学結果や並列度を暗黙変更しない。

材料pieceの共通pathは次の三段で処理する。

1. `line_boundary_bvh_v3`のNumba kernelがread-only flat BVHをworker-local stackで走査し、inclusiveなAABB候補を
   canonical facet ID順で返す。
2. event v9の保守的batch事前分類は、候補なしのclear、または曲線/chord偏差とsupporting-line straddleから必ずsplitと
   証明できるpieceだけを分類する。hit、corner、departure、曖昧pieceの物理判断は既存scalar locatorが所有する。
3. 同一float64 target timeとposition budgetを持つwall hit prefixだけを、選択済みintegratorのcompiled proposalで
   batch再積分する。boundary law、projection、RNG ordinal、残時間継続は従来のengine state transitionを一度だけ通る。

これらは最適化のcertificateであり、数値event定義ではない。1/2/4 threadで科学payload、RNG、accepted piece、candidate
query、refinement、最大深さを完全一致させる。512粒子×4 macro stepの固定caseでは後四値が
8,704 / 19,968 / 11,264 / 21である。case/result schema、proposal v4、event v9、physics runtime v3、座標・integrator
意味論は変えない。P12完了時点のrevisionは`deterministic_particle_engine_v18`、`compiled_cpu_tile_v3`、
`line_boundary_bvh_v3`、`resident_soa_worker_microtile_v3`、`solver_owned_memory_plan_v3`である。

## 13. P13 durable epochと再開不変性（履歴）

P13当時は運動方程式、step partition、event判定、RNG keyを変更しなかった。epoch boundaryはaccepted macro-step countだけで
決まり、`K=64`として

\[
  n \bmod K = 0 \quad\text{または}\quad t_n=t_{end}
\]

の時だけcommitする。trajectory/probe時刻、worker数、microtile幅、event数でepochを移動しない。checkpointへは
accepted stateだけをcaptureし、trial proposalや未commit writer bufferを混ぜない。再開後の最初のmacroは
checkpointの`macro_time_s`から始まり、release/frame/probe cursor、logical/physical event ordinal、exact origin、
surface-contact token、P1/Q1 cell hint、event aggregateを同時に復元する。これによりrelease、wall RNG、frame/probeの
inclusive境界を重複させない。特に確率wall drawは復元したzero-based physical boundary ordinalをcounter RNG keyへ
使うため、resume後も中断なし実行と同じdrawになる。

P13完了時点では、各worker waveをtile順にstable mergeし、そのwaveのboundary/failureをwriterへ渡して
ack後に破棄していた。resident stateとaccepted replayのcommitは従来のmain-thread規則を維持した。
queue容量は1であり、非同期writerが遅い時はmain threadを
blockしてbackpressureする。eventをdrop、coalesce、順序変更しない。checkpoint/final HDF5も同じsingle-owner threadが
書き、main threadは完了後だけA/Bやfinalの確定名へreplaceする。この同期ackはbounded memoryのためで、compute/I/O
overlapを利用する性能設計ではない。

`LATEST`はsegment/checkpoint pairの唯一のcommit pointである。segmentだけ、またはcheckpointまで確定して
`LATEST`更新前に失敗した場合、再開は旧`LATEST`を使う。`LATEST`のreplace後に失敗した場合は新commitを使う。
それより新しいorphanは読まず、同じindexの次epochで置換する。参照checkpoint SHA-256、checkpoint内の最新segment
SHA-256、全closed segmentのschema/countを検証し、破損を古いgenerationへsilent fallbackしない。最初の`LATEST`
以前のfailureは同じ初期stateからepoch 0を再実行する。古いsegmentは値全体のhashを持たないため、同shape・同countの
値改変は検出契約外である。

resume identityはcase/data hash、case/result/checkpoint schema、engine/result/compiled/runtime/memory/geometry/event/
physics/source/RNG/boundary revision、座標、integrator/backend/model、particle countとID digestを含む。全一致だけを
許し、migrationを持たない。通常実行とsegment/checkpoint/LATEST各境界からのresumeは、final、全event、series、
frame/probeと科学manifestをraw identityで一致させる。P13完了時点のrevisionはengine v19、result v3、checkpoint schema 1、
memory plan v4であり、case/result schema 1、proposal v4、event v9、physics runtime v3は不変である。
最終化はfinal、complete manifest、`_SUCCESS`、directory publicationの順で、各途中状態は次の同一identity runが
検査して安全に再完了する。これはlocal filesystem上のprocess failure回復であり、power-loss durability、remote
filesystem、同じOUTへの複数process同時書込みを保証しない。

## 14. P14 indexed containmentと性能採用条件

P14は物理式や積分順序を変えず、profileで支配的と確認した二つのbroad phaseだけをindex化した。production engineは
一つのままで、P14 closeout revisionはengine v20、compiled tile v4、field location v3、geometry v4、runtime layout v3、
memory plan v6である。proposal v4、event v9、physics catalog/runtime v3、result v3、case/result schema 1は不変である。

P1/Q1 field sampleは次の順に決定する。

1. accepted endpointのstrict-interior hintが有効ならそのcellを使う。
2. hintなし/miss/共有面はfield所有のread-only stackless cell AABB BVHからsupported containment候補を得る。
3. 候補をcell ID昇順に従来のconditioning-aware exact predicateで検査し、最小supported IDをownerとする。
4. containing supported cellがなければ、outside/masked provisionalの物理最近傍・同距離最小IDを変えないため全supported
   cellを走査する。このfallbackだけは意図的にO(cell数)で、index化済みとは主張しない。

BVHのAABBはcell/node座標ULPを外向きに含むが、AABBだけでinsideを確定しない。field indexは数値意味を所有せず、
既存predicateを呼ぶ候補生成だけを所有する。direct warm測定ではP1 10,000 cell×256 pointのinitial/hint/crossが
0.000395/0.000160/0.000356 s、Q1 5,000 cellでは0.000777/0.000330/0.000889 sだった。

geometryのtable start包含はboundary-first分類を保ち、その後だけvolume-cell BVHを使う。mixed tri/quad、共有内部edge、
hole/disconnected union、RZ axis seamに対し、候補cellへ従来のCCW half-space predicateを適用する。scalarの
`math.hypot`とcompiled計算の1 ULP差が境界判断を変えないようedge長をprepare時にCPython `math.hypot`で固定する。
各edgeから全非incident vertexまでの最小正規化clearanceが局所diameter由来のfloat64 floorを超えることを要求し、
未解像cellはprepareで拒否する。絶対座標の大きさだけを拒否条件にせず、AABB pruningはnode spacingも外向きに含む。

memory plan v6はfield/geometry indexの正確なresident `ndarray.nbytes`に加え、field 256 B/cell、geometry
1,024 B/cellのbuild transient上限と65,536点以下のtable-start batchを数える。これはsolver-owned予測でありRSS hard capではない。

23行×3観測matrixではregular 100k/1Mが20 workerで1.8796x/4.7965x、P1/Q1 10k crossingが
1.1737x/1.1473x、event 10k×20 hitが0.8882xだった。よって1 workerを保守的な既定とし、
host幅は大規模regular/event-lightを同機で
測定した場合だけ選ぶ。秒数をCI gateやCOMSOL比較にせず、output byte rateもend-to-end artifact rateとして扱う。

## 15. P14-P serial convergence（完了）

上記P12～P14はv20までの数値意味と履歴実装を記録する。現行engine v46 / compiled tile v21 / proposal v10 /
event v22 / CPU runtime layout v6 / memory plan v16 / geometry v7では、力の式、積分順、first-event意味、
accepted-state規則を変えず、外側`ThreadPoolExecutor`、future wave、worker別scratch、thread maskを削除した。
field/physics/integratorはbounded slabとpreallocated workspaceへの`*_into` pass、boundary BVHはpreorder + skipの
stackless traversalを使う。writerは同期single-ownerである。

linear/quadratic exactと一般曲線pathはrow別target/depth/interactionを持つflat SoA wavefrontへ統合し、compiled
wall/axis locator、boundary/RNG、row numerical status、batch surface release、direct columnar replay、bounded
event/failure stagingまでengineへ接続済みである。proposal v5はtarget timeをrow別に持ち、event v11は同時刻候補に
共有canonical nodeがないrowを`indeterminate_event`へfail-closedする。14系統のsmoke/部分matrix harnessはあるが、
focused correction後のregular 1Mは1/2/4 threadで9.32/10.09/10.10 sだった。4-thread speedup 0.923x、
1-threadもv20比23.7%退行したため、multithreadingを製品機能として不採用とした。
memory plan v11は各bounded bufferをnamed componentへ分離し、pack時だけのgatherを12.5% safety marginへ含める。
正確なbyte式は[`parallel_execution_plan.md`](parallel_execution_plan.md)が所有する。

field locator microkernelは約3.75xにscaleしたが、proposal/enclosureのPython/NumPy調停がend-to-endを支配した。
enclosureだけの理想並列化もAmdahl上約1.47xであり、巨大融合kernelは責務分離を損なう。case schema v2は
`resources.threads`を拒否し、single-thread compiled engineだけを残す。測定と再検討条件は
[`parallel_execution_plan.md`](parallel_execution_plan.md)を権威とする。

## 16. P15 continuous charge（完了）

P15は`oml_stationary_maxwellian_debye_huckel_v1`を既存の二つのexplicit integratorへ順に接続した。最初に
`rk4_fixed`が位置・速度・電荷数を同じ4 stageで評価する経路を受け入れ、その後
`exponential_midpoint`のstart half-step predictorで得る同じmidpoint時刻に、運動と電荷のexplicit midpoint評価を
接続して受け入れた。charge-only caseもstage evaluatorを通り、動的電荷ではlinear/quadratic exact pathを使わない。

両methodは`physics/charge.py`とruntimeが返すfinite invariant、rate bound、derivative bound、driftと
`a/lambda_D`のapplicabilityを共有する。各accepted intervalで`h L_Z <= 0.5`を要求し、stage、midpoint、endpoint、
event prefix、`state_at()`の電荷がinvariant内にあることを検査する。認証済みcharge区間の`max |Z|`をelectric
accelerationとposition/velocity enclosureへ渡し、初期電荷だけの一定加速度boundを動的電荷へ流用しない。
精度目的のstep subdivision、charge-only subcycle、平衡置換、implicit fallbackは行わない。共有global
coefficient/boundまたは`h L_Z`不成立はrun-fatal、粒子別support/applicability/nonfinite/invariant不成立は既存の
particle-local failure意味でfail-closedにする。

これはP15 closeout時点のexplicit policyである。現行ではRK4だけが`h L_Z<=0.5`を維持し、
deterministic exponential midpointとB03は後述するmidpoint-frozen affine exponential chargeへ置換された。

P15 stationary slice完了時点はengine v28、compiled tile v7、proposal v6、physics catalog v5、physics runtime v4、
RK4 enclosure v2、exponential midpoint/enclosure v2、memory plan v11である。runtime layout v5、event v11、
geometry v5、case/result/checkpoint/event schemaは変更せず、XY/RZ、wall、右連続frame/probe、checkpointの既存経路を使う。
外部M3-V applicability/relevance評価は完了した。現datasetへP15 stationary OMLを無理に適用せず、
relative-drift charge、有限速度Epstein、versioned ion drag、P16 Waldmann、Brownianを別model変更として扱う。
Case-A相当field builderとP17は、それぞれ独立workstreamである。

## 17. P15-D shifted-Maxwellian relative-drift charge（完了）

P15-DはP15の共同状態・bound・applicability経路へ、単一正イオン種のshifted-Maxwellian OMLを追加した。
設定上の`maximum_ion_drift_ratio`は
`|u_i-v|/sqrt(8 k_B T_i/(pi m_i))`のrun-wide上限であり、収集率式内部では
`s=sqrt(8/pi) M_i`へ変換する。`s=0`はstationary OMLへ一致し、小さい`s`を解析級数で評価するため
速度floorやhidden regularizerはない。

このrevisionは`phi<=0`だけを解き、prepareで初期`Z<=0`と全primitive/drift rangeでの非正平衡を認証する。
P15-D closeout時点のcharge invariantは`[Z_min,0]`、rate derivativeは厳密に負であり、当時は
`hL_Z<=0.5`を両explicit methodへ課した。現行supersessionは22.2節を参照する。
actual stageに加えて全短縮pathのdrift比と`a/lambda_D`を検査し、違反時は別branchへ切り替えない。

検証は独立Maxwell速度quadrature、zero-drift parity、解析微分、primitive corner bound、compiled/reference parity、
両積分器の材料wall/frame、XY/RZ、checkpoint/resumeを一つずつ再利用した。変更はcatalog v6、runtime v5、
compiled tile v8であり、engine v28、proposal/enclosure、memory plan、schemaは不変である。現行の成分絶対path boundは
co-flowを安全側に偽拒否し得るが、符号付きintervalはRZを含むenclosure全体の後続revisionとし、局所的に緩めない。

## 18. P15-E finite-speed Epstein（完了）

P15-Eは`epstein_finite_speed_maxwell_mixed_equal_temperature_v1`を追加した。球、局所Maxwell気体、
自由分子流、鏡面と完全熱適応・等温拡散再放出の明示混合だけを扱う。速度比
`S=abs(u_g-v)/sqrt(2 k_B T_g/m_g)`を使い、`S<=0.1`は解析級数、それ以外は`erf`を含む閉形式で
瞬時rateを評価する。speed floor、任意`delta`乗算、linear/Stokesへの自動切替はない。

係数上界`G(S_max)+sigma_R*pi/8`は既存の加速度・exponential enclosureへ使う。一方、RK4の
`maximum_dt_over_tau`は非線形速度Jacobianのradial係数`K=G+S G'`の上界を使う。これによりrateだけでは
見逃す有限速度域のstiffnessを拒否しつつ、指数法のenclosureを不要に膨らませない。全stage・受理pathで
`lambda/a>=10`とcase宣言の`S_max`を検査する。

独立3-D Gauss--Hermite分子速度積分、低速linear極限、高速`C_D`極限、Jacobian有限差分、global bound、
compiled/reference parityを検証した。一様気体の非線形減速を公開APIで進め、RK4は3.5次以上、
explicit midpointは1.8次以上を確認した。変更はcatalog v7、runtime v6、compiled tile v9だけで、engine v28、
integrator/proposal/enclosure、event、memory plan、case/result/checkpoint schemaは不変である。

この時点の次項目だったversioned ion dragは、続くP15-Fとして完了した。

## 19. P15-F collisionless Barnes ion drag（完了）

P15-Fは`barnes_collisionless_effective_speed_single_positive_ion_negative_debye_huckel_v1`を追加した。
単一・単価正イオン、非正粒子電位、linear two-species Debye screening、Debye--Hückel表面電位、球形完全吸収粒子、
collisionless・unmagnetized backgroundだけを扱う。collectionとorbital momentum-transfer cross sectionを相対流方向へ
加え、neutral dragの`linear_relaxation`とは分離した`explicit_acceleration`にした。

全stageで実電荷、相対速度、Debye長、collection/orbital impact parameter、Coulomb logを評価する。prepareは
field extrema・宣言drift・charge区間から加速度絶対上界と速度非依存の適用条件を外向きに認証する。runtimeは
`Z<=0`、`a/lambda_D`、`b_90/lambda_D`、`b_c/lambda_D`、ion-neutral mean-free-path比、正のCoulomb log、
宣言drift上限をfail-closedで再検査する。continuous charge併用時は同じplasma field、ion mass、stage `Z`を共有する。

Rutherford偏向の独立impact-parameter quadrature、zero-flow/zero-charge、global bound、compiled/reference parity、
fixed/continuous charge coupling、XY/RZ parityを検証した。公開一様場caseの観測次数はRK4 3.5以上、
explicit midpoint 1.8以上である。変更はcatalog v8、runtime v7、compiled tile v10だけで、engine v28、
integrator/proposal/enclosure、event、memory plan、case/result/checkpoint schemaは不変である。無効時memory planは不変で、
既存warm simulate時間の回帰も観測しなかった。この時点の次項目だったP16は続くsliceで完了した。

## 20. P16 Waldmann--Gallis thermophoresis（完了）

P16は`waldmann_gallis_free_molecular_single_species_heat_flux_v1`を、drag以外の既存
`explicit_acceleration`合成へ追加した。sample済み`T_tr`から
`c_bar=sqrt(8 k_B T_tr/(pi m_g))`を計算し、`a=drag_diameter_m/2`として
`a_th=(32/15) a^2 q_tr/(mass_kg c_bar)`を各実stageで評価する。`q_tr`はlocal mass-average neutral frameの
並進伝導熱流束であり、coreは温度gradientを回復しない。

prepare時のcomponentwise加速度boundは`T`下限と`abs(q_tr)`上限を用いる。粒子別static gateは
`lambda_lower/a>=10`、連続path gateは速度enclosureとgas-velocity絶対上限から
`abs(u_g-v)/c_bar_lower<=0.1`を外向きに認証する。actual stageでも同じ二条件を再評価するため、局所値だけが
適用内でも経路を認証できなければfail-closedとなる。これらの保守的global extremaは偽拒否し得るが、局所clamp、
zero-force fallback、Talbot/continuum blendでは緩めない。

熱泳動はRK4では4 stage、exponential midpointでは既存midpointのadditive accelerationとして評価する。新しいstate、
subcycle、dt gate、integrator、event分岐はない。一次Chapman--Enskog分布の独立3-D Gauss--Hermite momentから
係数と具体的加速度を検証し、affine heat-flux fieldに対する公開caseでRK4 3.5次以上、exponential midpoint
1.8次以上の時間収束を確認した。XY/RZ parity、global bound、local/path適用域、compiled/reference parityも合格した。

変更はcatalog v9、runtime v8、compiled tile v11だけで、engine v28、proposal/enclosure、event、runtime layout、
memory plan、case/result/checkpoint schemaは不変である。thermophoresis-only prepared boundは`17N+24 B`で、既存の
external acceleration boundを除く固有増分は`N+24 B`である。この時点の次のtrajectory-physics項目はBrownianで、P17の
state-dimension変更は独立workstreamのままとする。

## 21. B01 inertial Brownian数値基盤（履歴：完了時点はproduction未接続）

局所区間で線形緩和率`gamma`、平衡速度`u`、`theta=k_B T/m`を固定すると、各Cartesian成分は

```text
dv = -gamma (v-u) dt + sqrt(2 gamma theta) dW
dx = v dt
```

である。`a=exp(-gamma h)`、`q=1-a`とすると、B01の平均は

```text
E[v_h] = u + a (v_0-u)
E[x_h] = x_0 + u h + q (v_0-u)/gamma
```

であり、確率incrementのjoint covarianceは

```text
Q_vv = theta (1-a^2)
Q_xv = theta q^2/gamma
Q_xx = theta (2 gamma h - 3 + 4a - a^2)/gamma^2
```

である。`stochastic.py`は2個の独立標準正規数をこの2変量GaussianへCholesky写像する。小さい
`z=gamma h`では`Q_xx`の差引きが桁落ちするため、
`(2z-2q-q^2)/z^2 = (2/3)z-(1/2)z^2+(7/30)z^3-...`を使う。
これにより短時間極限`Q_xx~(2/3)theta gamma h^3`、`Q_xv~theta gamma h^2`、
`Q_vv~2 theta gamma h`を直接保つ。長時間では`Q_vv→theta`、
`Q_xv→theta/gamma`、`Q_xx~2 theta h/gamma`となる。

この平均は線形dragだけの厳密解であり、`F_other/m`を含まない。B01は数値primitiveなので任意のadditive forceを
近似的に混ぜない。最初のproduction接続もEpstein linear dragとfixed chargeだけに限定し、他の決定論力を設定した
caseは黙って力を落とさずprepare時に拒否する。

二分では親increment`e`を独立な左右increment`e_L,e_R`の合成
`e=A_(h/2)e_L+e_R`とみなす。`Q`を上記covariance、`A`をOU状態遷移行列として、

```text
E[e_L | e] = Q_L A^T Q_h^-1 e
Cov[e_L | e] = Q_L - Q_L A^T Q_h^-1 A Q_L
e_R = e - A e_L
```

を使う。実装は`x/(sqrt(theta) h), v/sqrt(theta)`とcovarianceを`z`で割った有限な無次元量で解き、
短い区間での逆行列と条件付き共分散の桁落ちを避ける。右incrementを代数的剰余にするため、子を合成した
親endpointは丸め誤差内で保存される。

乱数は既存Philox4x32-10を使うが、mutable cursorを追加しない。counterはparticle ID 64 bitとmacro interval
64 bit、派生keyは`root_stochastic_interval/tree_level/tree_index/component/root-or-split stream`から作る。
`root_stochastic_interval`は同じmacro interval内で新しく始まる独立OU区間のordinalで、最初の区間は0とする。
したがってidentityのauthorityは
`(seed, particle_id, macro_interval, root_stochastic_interval, tree_level, tree_index, component, stream)`である。
algorithm authorityは
`philox4x32_10_brownian_interval_tree_v1`であり、particle配列順、slab、将来のthread分割に依存しない。
Box--Muller変換の超越関数によるplatform間最下位bitは科学的identityとして保証しない。B02のmanifestは
algorithm revisionとstream IDを保存するが、machine fingerprintを科学的同一性の根拠として追加しない。

検証は20万replicaの平均・full `(x,v)` covariance・XY独立性、16万replicaの左右half covariance・左右独立性・
親endpoint保存、`z=1e-8`から`1e6`の短長時間極限を含む。production式から独立したcomposite
Gauss--Legendre Green-kernel積分と、無次元化した一般2x2 Gaussian conditioningを`z=1e-12`から`1e3`で照合した。
二段tree再構成とmacro/root/tree indexの64 bit上位wordも検査する。1,000,000行の開発機実測は、2成分normal生成
約14.98 M particle/s、joint increment約4.70 M particle/s、conditional split約1.46 M particle/sだった。
Philox JIT warm後の単発記述値であり、release性能gateではない。

B01完了時点では`integrators.py`、engine、case/result/checkpoint schema、memory planへ接続していなかったため、
`noise` categoryや`ou_langevin`は選択不可であり、finite field support、trajectory/probe、材料壁、first passage、
RZ Brownianを対応済みとはしなかった。次の縦切りとして定めたCartesian XY、Epstein linear drag、fixed charge、
terminalなstick/escapeだけのproduction接続はB02で完了した。一般曲面で未sample crossingを決定論的に除外できない
`RMS safety band`はB02でもproduction certificateに使わない。確率的clear boundを後に追加する場合は、明示した
全runのmiss-probability budgetを持つ別revisionとする。

## 22. Inertial Brownian production path

現行は`solver.integrator=ou_langevin`と、model `inertial_langevin_fdt`、revision
`inertial_langevin_fdt_epstein_linear_midpoint_2d_v2`の`physics.noise`を同時に選ぶ。Cartesian XY / RZ meridional、
fixed/continuous charge、native/effective-gas線形Epstein、適用域が交わる既存additive forceを同じ経路で扱う。
線形緩和率`gamma`、平衡速度`u_g`、`theta=k_B T_g/mass_kg`、additive accelerationとcharge affine係数は、
deterministic half-stepで得た各macro-root midpointで一度評価して固定する。`T_g`のauthorityはEpstein dragが参照する
`gas_temperature_field`だけであり、noise専用温度は持たない。support/applicabilityはRNG draw前にfail-closedで確定し、
prepare時に`gamma*dt<=1e6`、全rootで有限な
`0 < gamma*h <= 1e6`を要求する。

設定depthを`D in [0,10]`とすると、root joint OU incrementをB01のconditional Gaussian half-splitで深さ`D`まで
二分し、`2^D`個のleafを物理時刻順に処理する。counter identityは
`(seed, particle_id, macro_interval, root_interval=0, tree_level, tree_index, component, root-or-split stream)`で、
slab幅、output schedule、checkpoint/resumeから独立する。treeはdepth-firstに走査し、各levelで未処理right childを
一つだけ保持するため、stochastic tree workは一slab粒子あたり`32*(D+4)` byteである。memory plan v12はこれを
`stochastic_tree_work_bytes_per_particle`と`slab_stochastic_tree_work`へ明示し、runtime layout v6のslab幅決定へ含める。

各leafのOU endpoint `(x_1,v_1)`を得た後、`integrators.py`は始終点位置と速度から一つのcubic Hermite polynomialを
作る。proposal v7の`state_at()`、外向きBezier enclosure、既存event loopはこの同じpolynomialをtrajectory/frame/probe
replayと材料first hitへ使う。B02初回sliceのwall subsetはterminal `stick`/`escape`だけだったが、現行は
後述B04が同じnumerical pathへactive wallを追加している。event時刻は各leafの
Hermite numerical pathについて認証されるが、sampleされていない連続OU trajectoryのexact first-passageではない。
finite depthを精度parameterとして固定し、一般のcontinuous OU crossingを解いた、またはmiss probabilityを0にしたとは
主張しない。

受入ではjoint OU平均・共分散、frameなし/ありのfinal identity、checkpoint中断後の全公開payload identity、
Hermite wall event、平面first-passageのdepth 6/7/8安定化、slab幅不変性、surface releaseの右連続性を検査した。
非terminal wallのprepare拒否はB02初回sliceの履歴gateであり、現行B04のactive-wall gateがこれを置き換える。COMSOLはこの
数値pathのauthorityでなく、比較はcore外のV&Vだけが所有する。

`gamma*h<=1e6`は指数関数の範囲だけを制限し、極端な`theta`、状態、分割正規化のfloat64表現可能性を保証しない。
engine v30はroot covariance、各conditional split、deterministic mean適用を実際に検査する。通常経路は従来どおり
slab vector演算とし、数値例外が出た時だけrow別再評価で不良粒子を同定する。不良粒子はそのleaf開始、すなわち最後の
accepted stateで`nonfinite_physics`となり、同じslabの正常粒子と独立なright subtreeは継続する。shape不一致や
内部不変条件破損まで粒子failureへ隠さない。

現行revisionは`particle_engine_v46`、`coupled_fixed_step_proposal_v10`、
`inertial_langevin_2d_catalog_v23`、`signed_ion_compiled_physics_runtime_v22`、
`resident_soa_serial_slab_v6`、`solver_owned_memory_plan_v16`である。result algorithm v6、canonical case/data/result schema 3、
checkpoint schema 2、compiled tile v21、event `line_quadratic_curved_capsule_periodic_first_hit_v22`、
field location v4、field time v2、required field v6、geometry v7、boundary v7、
source `realized_internal_surface_contact_schedule_v5`を使う。event v22はevent v20のcontact-set継続とevent v19の周期first-event意味と
event v18 / geometry v6の独立`contact_radius_m`を維持する。有限半径exact残時間では同時接触集合を暗黙に保持し、
各candidateについて接触位置と離脱方向を証明できた時だけzero-time再接触を除外する。位置nudgeは使わず、
後続impactと材料/周期混在cornerは通常のfirst-event判定へ残す。0半径ではpoint意味を維持する。Brownian manifestは
`philox4x32_10_brownian_interval_tree_v1`、`inertial_joint_ou_v1`、
`conditional_gaussian_half_split_v1`、`conditional_boundary_refinement_v1`、`macro_root_frozen_midpoint_v1`、base/max depth、root/split stream、resolved noise model/revision、
`path_kind=cubic_hermite`を記録する。resume identityはBrownian RNG/OU/split/tree-policy revision、coefficient policy、base/max depth、
resolved noise model/revisionを含み、stream IDとpath kindを同名keyで重複させない。

### 22.1 現行2-D midpoint charged/forced Brownian

現行`inertial_langevin_fdt_epstein_linear_midpoint_2d_v2`はXY/RZで、native/effective-gas線形Epstein、
fixed/continuous charge、適用域が交わる既存additive forceを一つの
`stochastic_exponential_midpoint_v1`で合成した。係数方針は`macro_root_frozen_midpoint_v1`である。root始点からnoise-free決定論
exponential-midpoint predictorをmidpointまで進め、そこで`gamma,u,T,a,G=dZ/dt,J=dG/dZ`を1回評価する。
`J<=0`を要求し、`u_eff=u+a/gamma`に対するjoint exact OUをroot全幅に適用する。chargeはroot基準の
`A=G_mid+J(Z_root-Z_mid)`に対し

\[
Z(t)=Z_{root}+\frac{\operatorname{expm1}(J\Delta t)}{J}A
\]

とし、`J=0`では厳密に`Z_root+A*Delta t`を使う。predictorは評価点のみを
作り、stateをhalf-step commitするStrang/K-O-K分割ではない。

同一rootのconditional treeは固定係数で親joint Gaussian endpointを保存する。axis hitではHermite prefixを
hitまでcommitしてRZ stateをfoldし、macro残時間は次の`root_stochastic_interval`ordinal、再評価した係数、
独立root drawで再開する。元rootのcubic remainderはfold/restrictしない。B03初回closeoutのwall subsetは
terminal `stick`/`escape`/`hold`だけだったが、現行はB04が同じprefix/fresh-root原則でactive wallを追加する。
fixed chargeは同じaffine charge式の`G=J=0`へ厳密に退化し、XY専用の別係数分岐を持たない。

凍結定数係数のmean/full covariance exactness、noise-off決定論極限の2次収束、manufactured charge-electricの
noise-zero決定論極限の観測次数`>=0.9`、axis restart、optional-force compiled parity、tree-depth first-passage、
slab/output/checkpoint identityを受け入れた。continuous chargeのaffine-exponential dense pathはprepare済みinvariant内だけを許可し、
証明不能ならfail-closedとする。一般state-dependent SDEのstrong orderまたはweak 2次、等方3-D Brownianは主張しない。
公開OU試験の期待共分散はproduction式から独立したGreen-kernel積分を共用する。時間線形のγ(t)・u(t)は
16,000粒子、T=0.2 s、32 macroで非自律Green-kernelの平均・full covarianceへ照合した。
位置依存のmanufactured flow u_x=-3x、u_y=0、一定γ,Tは16,000粒子、T=0.4 s、32 macro、depth 0で
独立2×2行列指数・Green積分へ照合した。各caseの試験内で固定したMonte Carlo幅と離散化budgetを使い、
noise変位を流速評価へ反映しない反例を識別する。これは当該resolutionの精度であり、一般SDEの次数、
連続first-passage、背景流体modelの物理validationへ拡張しない。
現行revisionはcatalog v23 / engine v46 / proposal v10 / event v22 / runtime v22 / compiled tile v21 / memory plan v16である。
B03 path arrayの静的な保守上限は`648 B/row`で、`2048 B/row`上限内にある。正式characterizationは24/24実行を
全粒子active・failure 0で完了し、計時、process RSS、solver plan、stochastic/event workを
[`evidence/b03/`](../evidence/b03/README.md)へmachine-local・non-gating証跡として保存した。これは初回B03
closeout（engine v34 / proposal v9 / runtime v17 / tile v16）の履歴証拠である。
performance characterizationで反復加算由来の終端tailを検出したため、engine v34はmacro targetを補償積和による
`start + n*dt`のindexed gridから構築する。endへのsnapはfloat64構築roundoff内だけに限定し、意味のある残時間は
proposal/eventへ渡す。これにより丸めtailだけの余分なOU rootを作らない。

event v15は物理position budgetとroundoff budgetを加算し、facet-local offset dotを補償演算で、Hermiteの評価・包絡を
root-relative TwoDiffで扱った。event v16はvalidなRK4 dense rowだけ、integrator所有のroot-relative position
Bernstein control enclosureをfacetの外向きhalf-spaceへinterval射影する。4制御点の外向き上限がすべて既存position
budgetの負側へ厳密に入る時だけ、convex-hull性から全区間の非接触を認証する。不正・非有限・証明不能なcontrolは候補を
残してsplit/fail-closedする。monotone clearはcubic Hermite derivative-Bernstein enclosureの明示opt-inだけに限定し、
exponential、scalar pathはv16 certificateを使わない。first-hitとterminal意味、tolerance、endpointは変更しない。
control payloadは`144 B/row`、構築時追加peakは`128 B/row`で、保守的な同時live見積り約`1.76 KB/row`は既存
`2048 B/row`一般stage scratch上限内に収まるため、event v16導入時点ではmemory plan v13を維持した。memory plan v14は
exponential local-stage/applicability範囲探索にも最大64 cellの候補arena 544 B/rowを明示計上する。broad AABB pre-countはこの
facet-local clearより前の保守的なperformance work量であり、accepted event数を表さない。
現行event v22はevent v19 / v18の曲線path認証を維持し、最初のhit時刻と局在budget内で同時かつ共有nodeへincidentな
facet集合を保持する。exact、一般RK4、exponential midpoint、Brownianの材料hitは同じcandidate table、priority、
combined-normal応答を使い、証明不能なcornerを任意の一面へ丸めない。周期hitは上記6.1のcentre-crossing arbitrationを使う。
exact有限半径pathの反射後残時間では、接触集合を追加配列へ複製せず、各broad candidateについて接触位置と離脱方向を
証明できたzero-time rootだけを除外する。位置nudgeは使わず、後続impactは同じroot探索へ残す。

event v22 / engine v46は、monotone approachの非接触証明でgeometry position budgetを壁の厚さとして使わない。
外向きnormal velocityの区間下限がroundoff込みで厳密に正、endpoint signed-distanceの区間上限が厳密に負なら、
単調性により全区間がsupporting lineの内側にある。従来の`distance_upper < -position_budget`は、endpoint接触の
`distance <= position_budget`との間にsigned-distance roundoff幅の隙間を作り、単調な接近でも時間の表現限界まで
二分する場合があった。この条件を`distance_upper < 0`へ修正する。非単調・符号不確定な経路はclearにせず、
既存の包絡、局在position/time budget、corner集合、split/fail-closedを維持する。許容誤差、RNG、壁応答は変えない。

### 22.2 B04 Brownian active-wall restart（完了）

B04はB02/B03のOU endpoint、conditional tree、cubic Hermite leaf pathを変更せず、各leafのfirst hitへ
現行boundary v7のterminal `stick`/`escape`/`hold`とactive `specular`/`restitution`/`maxwell_thermal`/
`probabilistic_stick`を適用する。`probabilistic_stick`のactive側は設定済みfallback lawを同じphysical boundary
event ordinalで解決する。wall drawはBrownian tree drawとstreamを分け、mutable RNG cursorは持たない。

active hitではHermite prefixだけをhitまでcommitし、boundary response後のstateからmacro残時間を次の
`root_stochastic_interval`ordinalと独立root drawで再開する。元rootの未使用cubic tailは反射、fold、restrictせず
破棄する。RZ axisも材料wallもこのprefix/fresh-root意味論を共有するが、axis foldはwall eventや
wall RNG ordinalを作らない。これは有限depthの新しいnumerical rootであり、continuous OUのexact first-passageや
reflection-conditioned bridgeを主張しない。

active stochastic restart回数は既存`max_interactions_per_step`で有界化し、first-hit refinement depthと別に
計数する。低noiseのXY/RZ複数hit、restitution、probabilistic fallback、Maxwellの鏡面branch、B03のcharge/
force、axis→wall順序、実noiseのcontainment/depth/slab不変性、checkpoint/resume identityを公開APIで検査する。
surface sourceは入力でrealize済みの初期速度を使う。Brownian surface releaseでは明確に内向く速度を要求し、
零速度や接線速度をposition nudgeで通さない。
任意の一般SDE、等方3-D Brownian、surface charging、resuspension、rolling/slidingはこのrevisionの対象外である。
有限半径接触はevent v18で導入したHermite leaf locatorをevent v22でも維持し、同じfresh-root active-wall規則を使って、連続OU trajectoryのexact
first-passageを追加で主張しない。

### 22.3 B05 conditional Brownian boundary refinement（完了）

`interval_tree_depth=D_base`は全stochastic rootへ一様に適用するnumerical-path精度のauthorityであり、
省略も自動低減もしない。base leafのcubic Hermite proposalを既存locatorがwallなし・axisなし・数値的にclearと
証明したrowはそのままcommitする。wall、RZ axis、surface contact、または証明不能なrowだけ、保存済み親endpointを
B01のconditional Gaussian half-splitで`adaptive_max_depth=D_max`まで時間順に二分する。active wallまたはaxisで
prefixをcommitした後は、B04と同じfresh root規則を使う。fresh rootのrow・開始時刻・post-event state・ordinalは
同一slab幅の数値SoAへstable順で積み、現在のgenerator/tree/proposal frameが全て解放された後のiterationだけが
次rootを開始する。反射回数に比例するroot再帰や第二schedulerは持たない。

`D_base=D_max`は追加事前分類を通らず従来固定depth pathへ退化し、同じRNG key、科学payload、output/slab/resume
identityを維持する。`D_max>D_base`でもroot/split counter keyを変えず、独立乱数の引き直しをしない。この選択は
有限depth numerical pathの計算配置だけを変える。sampleしていない連続OU軌道のexact first-passage、zero miss
probability、またはuniform `D_max`と同じ全空間path解像度は主張しない。精度は`D_base`の収束で評価し、adaptive
効果は同一build・同一caseでcandidate query、hit数、到達depth、wall timeを固定depth退化と対にして測る。

memory plan v16は実行時の候補率に依存せず、到達可能な`D_max`からstochastic tree workを
`128*(D_max+4) B/row`として保守的に計画する。tree policy revisionは
`conditional_boundary_refinement_v1`で、base/max depthとともにmanifest/resume identityへ入る。新しいscheduler、
第二Brownian engine、確率的clear診断は追加しない。

### 22.4 Charge-stable deterministic pathとdurable cadence（完了）

deterministic `charge_stable_exponential_midpoint_v3`もB03と同じmidpoint-frozen affine lawを使い、
運動のmidpoint accelerationは同じ`Z_mid`を参照する。`state_at()`、event prefix、残時間proposalはroot始点から
同じ更新を再評価し、保存済み始終点を厳密に戻す。RK4は従来どおりexplicit `hL_Z<=0.5`を要求する一方、
exponential pathはこの値をstability gateとして使わない。clip、charge-only subcycle、第二engine、runtime method
switchingはなく、精度は別runの`h,h/2,h/4`自己収束で選ぶ。

durable epochは累積work

\[
W=n_{macro}+n_{accepted}+n_{query}+n_{refine},\qquad
T=\max(2^{20},128N)
\]

で決める。engineはaccepted macro barrierでepoch開始時からの`W`差が`T`以上か、最終macroかを判定する。
`cumulative_solver_work_v1`、resolved `T`、4 components、barrierはmanifestとresume identityの両方に入る。
output scheduleとslab幅はこの判定へ入らない。`output.py`の同期single-owner writerはengineの決定を受け、
segment、inactive A/B checkpoint、`LATEST`をatomic順で永続化する。result schemaは3、checkpoint schemaは2、result
algorithmはv6である。threadingや第二writerは再導入しない。

### 22.5 D1/D2の独立分布・first-arrival検証（限定scope）

非dyadic frameはmacro内の固定時刻`0.37h, 0.61h`をdepth 0/2/4で照合する。
独立OU Green積分によるleaf両端のjoint covarianceとcross covarianceへ、Hermite位置とその導関数の
線形写像を掛けて有限pathのGaussian lawを得る。12,000粒子の公開frameを、このlawの平均と
正規化したX、V、X+V、X−Vの分散へ照合し、family error 0.001のGaussian/χ² concentrationを使う。
真のOU内部時刻lawと有限Hermite lawは別である。登録した二時刻でdepth 4のcovariance相対差は
最大約1.12%まで縮小したが、任意時刻や任意SDEの次数を認定しない。

RZの零radial flow/force、一定係数、meridional 2DOFでは、signed OU位置を`r=|X|`へfoldした
Gaussian CDFと12,000粒子・20 macro・depth 2の公開runを比較する。axis crossingを含めて
全粒子active、failure 0を要求し、CDFの同時DKW幅と登録probability budget 0.025を使う。
一般3D等方Brownianのradial processへ拡張しない。stationary OML負電荷branchの一様plasmaでは、
独立scalar charge ODEをRK4 2048/4096区間で解き、Z(t)をOU Green kernelへSimpson積分する。
12,000粒子・32 macroのcharge、Coulomb連成平均、OU共分散へ照合する。chargeを凍結する反例を
登録mean gateで識別できる。velocity依存shifted/aggregate chargingや一般連成SDEの次数は認定しない。

D2のfirst-arrivalは`dX=V dt, dV=-V dt+sqrt(2)dW`、`X(0)=-0.5, V(0)=0`、平面`X=0`の
terminal吸収に限定する。独立test計算はKramers方程式を位置upwind有限体積、velocityの正な
exponentially fitted fluxとimplicit tridiagonal solveで解く。吸収は流入velocityのzero-inflowであり、
流出velocityを零へ固定しない。この半範囲boundaryは
[Hwang–Jang–Velázquez, equations 1.3–1.4](https://arxiv.org/pdf/1311.4635)に対応する。
frictionとdiffusion項は選択したOU SDEから導く。production OU/tree/event関数はreferenceへ呼ばない。

位置格子128/256/512、velocity格子256/512、Δt=0.001/0.0005、位置domain長4/8、
velocity上限6/9を固定した七系列で実収束を調べる。初期deltaは位置・velocityの隣接cellへ
momentを保存して配分し、dx/dv系列でその幅も縮める。最大resident mass/workは各131,072 cellで
合計約2 MiB。正値性、到達mass、残存mass、人工position/velocity端流出massの和を検査する。
観測massのずれは10⁻¹⁰未満、人工position流出は2×10⁻⁵未満、velocity流出は2×10⁻⁸未満を要求する。
最細差の二倍と他軸の差・tailを合計した経験的indicatorを0.005以下とする。これは実測された
収束のoperational allowanceであり、数学的なPDE誤差保証上界ではない。

公開solverはγh=1.2でD=0/2/4、D=4でγh=1.2/0.6/0.3の二軸を分ける。
16,000粒子、seed 53619、γt=0.4/0.6/0.8/1.0/1.2の到達CDF/fateを登録し、五run同時の
family error 0.001からDKW幅0.01696535を得る。depth 2以上の登録resolutionは
`max CDF差 + MC幅 + reference allowance <= 0.03`を満たした。depth 0はcoarse観測として残し、
自動認証しない。h/depth系列間の差はMC幅以下であり、観測次数は`NOT_RESOLVED`である。
証拠は[独立基準と集約到達数](../../reviews/d2_kramers_first_arrival_2026-10-09.json)へ記録する。
refinementが登録allowanceへ収まらない場合はfirst-arrival accuracyも未解像とする。
active/repeated wall、RZ first-arrival、time/state-dependent coefficients、連続軌道のzero-miss、
実source/adhesionの物理validationへこの合格を流用しない。

## 23. P18-I aggregate ion-drag sensitivity（完了）

P18-Iの二式は新しいODEやoperator splitを作らず、既存のadditive accelerationへ各integrator stageで加算する。
relative-flow式だけが粒子速度へ依存し、prepare時のion-velocity成分絶対上界とproposalのparticle-velocity包絡から
`|u_i-v| <= maximum_relative_ion_speed_m_s`を連続path全体で認証する。actual stageも同じ上限を検査し、超過を
速度clip、step subdivision、Barnes fallbackで隠さない。断面積は`b_col^2<=b_s^2`および
`x*log(1+a/x)<=a`からcollection＋orbital全体を`3*pi*b_s^2`で保守的に囲い、既存global acceleration enclosureへ渡す。

image式は粒子速度へ依存せず、ion-flow speedと正則化した電場方向からexplicit accelerationを作る。field extrema、
charge invariant、particle mass/radiusから有限global boundを構築し、速度path gateや新しいstateを追加しない。
P18-C continuous charge併用時は同じstageでsample済みの`n_i,T_iV,u_i,m_i,lambda`、imageでは`T_eV`も共有し、
そのstageの実電荷`Z`を使う。charge lagやcharge-only subcycleはない。

公開scenarioはrelative-flowをCartesian XYで4段階の`dt`へ収束させ、RK4はorder 3.5以上、explicit midpointは
order 1.8以上を確認した。imageはRZ一様場の定加速度関係を両積分器で、relative-flowは材料eventを両積分器で検証した。
checkpointは新state/schemaを追加しないため既存の全payload resume identity gateを再利用する。純粋式はDecimal oracle、
compiled式は同じsample配列を使うpure evaluatorとの全成分parity、
global boundは乱択包含で検査した。変更はcatalog v12、runtime v11、compiled tile v13だけで、engine v30、proposal、
event、memory plan、case/result/checkpoint schemaは不変である。

## 24. P18-D quasistatic spherical DEP（完了）

P18-Dはproducerが与える`grad(mean_E_squared)`を既存field samplerで各stage位置に補間し、
`2*pi*epsilon_0*epsilon_r*a^3*K_CM/mass_kg`を掛けた明示加速度として、既存の単一compiled stage passへ加える。
fieldのgradient recovery、DC/RF平均、point-dipole上限の認証はtrajectory core外に残し、core内でEを微分しない。

prepareは全粒子の`electrostatic_radius_m <= maximum_point_dipole_radius_m`をfail-closedに検査する。別々に直列化・導出した
同一境界値の丸めだけを許すため、runtime v17は認証上限の直後のfloat64値を外向き表現として一つだけ受理し、二つ目の
successorまたはそれ以上を拒否する。相対許容差やproducer別例外は使わない。field component extrema、
`abs(K_CM)`、`epsilon_r`、粒子別`a^3/m`から外向きcomponent boundを作り、既存のexternal-acceleration boundへ一度だけ加える。
従って非一様fieldはRK4/exponential midpointの既存support・event enclosureをそのまま使い、Cartesian XYの厳密定数gradient
だけが既存constant-acceleration経路へ退化できる。新しいresident state、専用integrator、operator split、mesh、
checkpoint payloadは持たない。現行Brownianは線形EpsteinとDEP双方のfield authority / applicabilityが成立する場合に
同じmidpoint additive-force経路で合成し、成立しない組合せをplan解決時に拒否する。

pure式ではzero gradient/zero CM、係数符号、`a^3/m` scaling、2-D回転共変性を確認し、乱択したgradientに対して
global boundの包含を検査した。
catalog/runtime/compiledは同じsample値に対してXY/RZで一致する。公開API scenarioは負のCMと線形
`grad(mean_E_squared)`から調和振動子を作り、Cartesian XYのRK4で3.5次以上、axisymmetric RZのexplicit midpointで
1.8次以上を確認した。これは積分器と座標基底を対にした最小coverageであり、不要な直積testを増やさない。

変更はcatalog v13、runtime v12、compiled tile v14だけである。engine v30、proposal/event、runtime layout、memory plan、
case/result/checkpoint schemaを変更せず、manifestの既存resolved physics-model mappingへ
`quasistatic_spherical_gradient_e2_v1`を記録する。

## 25. P18-L rarefied-vorticity lift sensitivity（完了）

P18-LはRZ/no-swirlだけに
`F=K (omega_phi e_phi) x (u_g-v)`、`K=C_L*pi*rho_g*lambda_g*a^2`、
`a=drag_diameter_m/2`を追加した。`C_L`は有限正値をcaseへ明示し、gas velocity `[m/s]`、density `[kg/m^3]`、
mean free path `[m]`、producer所有のsigned azimuthal vorticity `[1/s]`を各stageでsampleする。core内でvelocity fieldを
微分せず、`lambda_g/a>=10`をstatic lower boundとactual stage/pathの双方でfail-closedに検査する。

速度へ直交結合するため、liftを速度非依存のprepared external-acceleration配列へ入れない。runtimeの全非drag
加速度boundは`particle_index`とcomponent-wise particle-velocity上界を受ける一つのcallbackである。coupling-rate上界を
`k_max=C_L*pi*rho_max*lambda_max*a^2*abs(omega)_max/m`とすると、callbackは
`B_r=k_max*(U_z+V_z)`、`B_z=k_max*(U_r+V_r)`を外向きに返す。

exponential enclosure v3はstartで`A0=B(abs(v0))`を評価し、
`Vhalf=max(abs(v0),U)+h*A0/2`で`Ahalf=B(Vhalf)`を再評価する。`A=max(A0,Ahalf)`から
`abs(v)<=max(abs(v0),U)+h*A`、`abs(x-x0)<=h*max(abs(v0),U)+h^2*A/2`を作り、全短縮predictor/stateを囲う。
field samplingやlift dispatchをintegratorへ持ち込まず、exponential midpoint v2、engine v30、proposal v7を維持する。
RK4は同じvelocity-aware callbackを既存enclosureから使う。

3-D cross-product射影oracle、zero/comoving、符号、`rho*lambda*a^2/m` scaling、直交性、乱択bound、Kn拒否、
pure/compiled parityを確認した。一様vorticity shearのRZ公開caseはRK4 3.5次以上、explicit midpoint 1.8次以上で収束し、
指数法の全短縮stateもv3 enclosureへ包含された。現行BrownianはRZ、線形Epstein、同じgas authorityと
rarefied-vorticity liftの適用域が成立する場合に同じmidpoint経路で併用する。

変更はcatalog v14、runtime v13、compiled tile v15、exponential enclosure v3だけである。integrator v2、engine v30、
proposal v7、event/runtime layout/memory plan、case/result/checkpoint schemaは不変である。100,000-row warm direct stageの
medianはdisabled `0.023354 s`、enabled `0.0255142 s`、比`1.0925`、prepared bound増分`900016 B`だった。
これはmachine-localな非gating観測であり、COMSOL pointwise/trajectory parityはM3-C1まで`NOT_TESTED`である。

## 26. P18-R effective-gas sensitivity revisions（完了）

P18-Rは新しい力式やoperator splitを追加せず、既存linear EpsteinとWaldmann--Gallis heat-flux evaluatorへ
revision別のapplicability envelopeを渡す。`epstein_linear_effective_gas_sensitivity_v1`と
`waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1`は、producer認証済みone-effective-Maxwellian/
pseudogasに限って`0 < maximum_speed_ratio <= 1`をcaseから受ける。既存`epstein_linear_v1`とP16 single-species
thermophoresisは固定上限`0.1`のままである。

actual stageでは既存pure/compiled evaluatorが`lambda/a>=10`と速度比を検査し、RK4 enclosure v2とexponential
enclosure v3が既存速度boxから全短縮pathを認証する。保存dataset frameはこの連続path判定を置換しない。thermophoresisの
`q_eff`はproducer-owned translational conductive heat fluxであり、数値coreはgradient recoveryやspecies配列を持たない。
したがって追加配列、resident state、専用workspace、integrator、step subdivisionはない。

旧revisionの`0.1`保持、必須上限の範囲拒否、pure/compiled parity、XY/RZ、両積分器とpath逸脱のfail-closedを検証した。
変更はcatalog v15、runtime v14、compiled tile v16だけで、engine v30、proposal v7、integrator v2、event、runtime layout、
memory plan、case/result/checkpoint schemaは不変である。保存auditのEpstein式PASSは約`1.1e-15`だが、既存P15-E/P16の
physical applicabilityは12/12 `NOT_APPLICABLE`、PPR `q_eff`欠損によりthermophoresis replayは`NOT_TESTED`である。
COMSOL studyは再実行しておらず、新revisionの数値受入をCOMSOL軌道一致へ昇格しない。

## 27. P19-L localized continuous-path applicability certificate（完了）

一般RK4 proposalは現行`rk4_position_hermite_state_extension_v3`を一度だけ構築する。位置の四つのBezier制御点は
`x0, x0+h*v0/3, x1-h*v1/3, x1`で、event locatorが使うcubic Hermite pathと同一である。速度と電荷はRK4の
stage-rateから得るcubic extensionを使い、始点と受理endpointを変えない。harmonic position/velocityでは
θ=0.37,0.61でexact-startの内部一step O(h^4)と、固定終時刻0.7 sのglobal O(h^4)を別々に解析解へ照合した。
endpointの局所O(h^5)とは別の確認であり、速度extensionが位置polynomialの導関数であることは要求しない。
非滑らかなfieldや任意のdense interiorへの一律の次数保証にはしない。

v3のdense位置評価は位置制御点をroot始点相対に変換し、TwoDiffが返す丸め差と厳密残差の
両方をBezier評価してroot始点へ足す。root始終時刻では保存済みの位置・速度・電荷endpointを厳密に
上書きする。したがって保存する制御点と数学的なpolynomialは不変だが、dense位置のfloat64評価経路はv3で
原点相対になった。

certificateはこの不変なroot pathのparameter部分区間をBezier subdivisionで制限する。位置enclosureは上記の
原点相対制御点を使い、TwoDiff残差の絶対値をroundoff paddingに加える。下限は`-inf`、上限は`+inf`
方向へ丸めながらroot始点を足し、world座標へ戻す。位置の微分制御点と物理的なchord deviationも同じ相対差と
TwoDiff残差を使い、等しい制御点の差は厳密に0として扱う。これにより物理的な曲率とenclosure幅は絶対座標ではなく
pathの局所変位に追従する。

ただし公開`rk4_dense_chord_deviation()`の戻り値は、最終のworld座標変換、両endpoint、endpoint chord式を
float64で評価する丸めを覆うため、`8*eps*max(abs(root_position_controls))`と対応するsubnormal termを別に加える。
この狭い絶対座標termのため、公開chord bound自体は完全な平行移動不変ではない。v3が解消するのは、v2の広い
roundoff幅が絶対座標に比例して膨張し、dyadic制限しても十分に縮まずevent certificateが閉じなかった問題である。

位置tubeと交差するfield cellから局所primitive rangeを作り、physics runtimeがrange上のcharge、force、
速度依存applicabilityを外向きに評価する。電荷intervalは全制御点が非正または非負ならその符号を保存し、
丸めで不変条件を跨がせない。v3は受理endpoint、数学的なimmutable state path、first-hit/event algorithmを変えない。
v3導入時のengine v32、proposal v8、event v14、`rk4_global_abs_enclosure_v2`も不変だった。

まず既存global applicability certificateを試し、それで証明できないrowだけをlocal certificateへ送る。P19-L完了時点ではglobal
`rk4_global_abs_enclosure_v2`が全短縮RK4のsupportとrev3b材料/RZ event geometryを所有した。P19-Lはevent pathを
置換せず、locator-before-validity、同じintegratorによるhit prefix再積分、fresh residual proposalというsequential
rev3b意味論を維持する。

局所cell候補は一row最大64件である。overflowは適用域違反でなく「この区間ではrangeを構築できない」ため、共有する
event `max_refinements` budget内で区間を二分する。実stage/sampleまたは局所rangeで確認した違反は
`field_support`または`model_applicability`、全区間を証明できないままbudgetを使い切った場合だけ
`indeterminate_applicability_certificate`とする。accuracy不足を物理違反または成功へ畳み込まない。
同じ実sampleでsupportとmodel評価の両方が成立しない場合は、場を定義できない`field_support`を先に分類する。

memory plan v13はdense path 176 B/row、split budget 2の現設定でinterval stack 72 B/row、最大64 cellの候補arena
544 B/rowを別componentとして計上する。4,096粒子・20 stepの公開API観測ではglobal fast path `0.9276381 s`、local fallback
`0.9345506 s`、比`1.0074517`で、final/release/boundary/failure/lifecycle payloadはbitwise一致した。これは同一machineの
非gating観測であり、portableな速度保証ではない。P19-L完了revisionはengine v31、proposal v8、event v12、field location v4、
memory plan v13、dense path `rk4_position_hermite_state_extension_v2`である。これは履歴的なP19-L完了revisionである。
P19-L性能artifactはruntime v16を記録し、v17との差は上記DEP上限の1 ULP境界意味論だけである。
現行runtime v22はv20のcharge Jacobian、optional aggregate three-current、relative-flow ion dragのinterval boundと
prepared charge invariantを維持し、同じstage passへ選択式のTalbot熱泳動とSaffman liftを追加する。

Case-A 100 nm・287粒子・1 stepの後続profileでは、P19 certificateは80 call・13,120 rowをすべて分割なしで局所証明し、
元の`2.497 s` profile中約`0.285 s`だった。支配したevent refinementに対し、generic velocity-box
chord-deviationの同じfloat64算術だけをrow batchへ移した。P19-L完了時点のevent algorithm v12とwork形状は不変で、event query
`25,953` row、SPLIT `12,833`、locator 127 call、proposal rebuild 126回のままである。warmed medianは
`1.647475 s`から`1.304153 s`（`1.263x`、`-20.84%`）となり、final＋frameのSHA-256は両方
`8dafaf295c7ca47efd7b86a79410996e3e8fda38a9924f66d9a082f3006bb0c3`でbitwise一致した。これは算術batch化の
machine-localな歴史的観測である。このevent-query結合は後続v14で閉じた。
再実行可能な条件とprofileは
[`candidate_hot_path_v1.json`](../evidence/p19l/candidate_hot_path_v1.json)をauthorityとする。

## 28. M3-C1 material eventとevent-query v14（完了）

event v13はglobal absolute safety enclosureをevent BVH queryにも流用したため、global安全域が広いCase-A 100 nmで
artificial subdivisionを発生させた。450 us checkpointまでに`7,623,460 / 7,792,306` refinement、すなわち
`97.8331703092769%`が既に累積していた。最終query/refinement/accepted/depthは
`16,427,517 / 7,792,306 / 8,635,211 / 16`である。

M3-C1 `line_quadratic_rk4_axis_first_hit_v14`は、global supportが独立に証明済みでdense boundがvalidな`rk4_dense` rowに限り、
current pieceのdense Bernstein位置・速度boundをevent broad-phase query authorityとする。global enclosureはshortened-stage、
field support、applicability、acceptance safetyのauthorityとして残す。dense boundが非有限または不正ならglobal boundを
保持してsplitし、global supportを証明できないrowではevent queryもglobalへfallbackする。first-hit ordering、wall law、
endpoint、schemaは変えない。

同じCase-A 100 nm、common canonical exact-connectivity P1、Brownian off、287粒子をfirst wafer stickまで進めたv14は
query/refinement/accepted/depth `842,927 / 11 / 842,916 / 11`、failure 0だった。material comparison v5は20/20、
pre-event prefixは9/9 PASSで、event時刻、hit位置、terminal chargeの絶対差は
`2.157542807607049e-13 s / 2.683964162031316e-14 m / 4.7283812421028415e-09 e`である。COMSOL Freezeとsolver stickは
terminal velocityの保存意味が異なるため、velocityをcross-solver material gateにしない。

operator-observed shell wall-timeは同じlocal環境で約`14m13s`（約853 s）から`36.5s`（約`23.4x`）へ短縮した。
これはmachine-local、概算、非gatingで、solver-reported runtimeまたはmanifest値ではない。

v14 solver-only 0.625/0.3125/0.15625 us再実行はrefinement 0で、query/acceptedはそれぞれ
`206,927 / 206,927`、`413,567 / 413,567`、`826,847 / 826,847`だった。位置・速度・電荷のRMS観測次数は
`2.029875353701904 / 2.0816971911764033 / 2.044084026475049`、fine-pair relative L2は
`6.099791486063973e-8 / 8.321356016032579e-8 / 1.3796067752988052e-8`、全3量`ORDER_EVALUATED`で自己収束PASSである。
この約2次はpiecewise P1場とmesh crossingを含む本caseの経験値で、RK4の形式4次を証明も否定もしない。旧v13の3刻みは
precision-stabilityの履歴として有効だが、artificial subdivisionでaccepted-piece countが同一となったため独立時間収束には使わない。

v14はsolver側event BVH broad phaseだけを変更し、hash-lock済みcommon-P1 COMSOL input/reference/source MPHは不変なので、
COMSOL studyを再実行せず既存referenceへcandidateを再比較した。compact authorityは
[`case_a_100nm_material_event_v1`](../evidence/m3c1/case_a_100nm_material_event_v1/)である。この限定anchorをnative-field
parity、物理妥当性、Brownian、30 ms、他case/size/variantへ一般化しない。P18-Hも解析・公開回帰と外部Freeze
candidate 15/15で完了した。B03 core、charge-stable coupling/work-scaled cadence、P20 performance closeoutも完了した。
同じ物理・field・boundary・stochastic意味を持つ外部V&V/M3-C2はcommon-P1 Case-A/Case-P 100 nm finalまで完了した。
Case-Pは登録済みR-Z/fate gateをPASSした。これは元COMSOL `auxq`どおりの二電流same-form結果で、後続three-currentや
species-resolved物理を認定しない。optional three-current productionはP21 priority 1で完了した。受理済みCase-P seed 3件の
287粒子owner discoveryも、科学payload・work・case identity・revisionを完全一致させて完了した。支配ownerは`integrators`
（自己時間比42.58--42.86%）だったが、事前登録済みbounded ownerではないため最適化は未承認でproduction変更はない。
M3-C2A anchorは`CLOSED_ACCEPTED_WITH_LIMITATIONS`である。10,000粒子以上の性能は製品SLAを先に定義した
独立work packageだけで評価する。COMSOL fittingは行わず、M3-C1 event v14の既存evidence/evaluatorは履歴として変更しない。
後続の明示指示によるbounded follow-upでは、chord certificateの意味を変えず、Python row loopと重複式を一つの
serial compiled batchへ統合した。accepted 3 seedの科学payload/work/revisionは完全一致し、end-to-end中央値は12.29%短縮した。
この実装効率改善で作業を閉じ、step、Brownian depth、event処理、次ownerは変更していない。authorityは
[`../evidence/m3c2/caseP_100nm_chord_optimization_v1/`](../evidence/m3c2/caseP_100nm_chord_optimization_v1/README.md)である。


### 22.6 反復鏡面壁とRZ到達の独立参照（0.2候補の限定scope）

一定係数・native線形Epstein・fixed chargeのOUに対し、平面反復鏡面反射を自由OUのjoint Gaussianの
`(|X|, sign(X)V)`として独立に導出した。位置CDFと位置/速度joint probabilityを比較し、有限人工遠方壁の
到達確率上界も別に加える。RZ meridionalの半径到達はsigned OUをfoldした問題として、左右二つの物理的吸収端を
固定した独立Kramers finite-volume解の両outward fluxと比較する。これは球対称3-D radial diffusionではない。
referenceの位置/速度mesh、時間刻み、速度領域を独立に細分し、正値とmass保存を確認する。
Kramersの`0.005`reference allowanceは測定した収束に基づく運用値で、厳密なPDE誤差上界ではない。

観測前に固定したN=16,000、family-wide MC幅、probability budget `0.03`を維持する。
engine v46 / event v22で反復壁の全4rowは安全failure 0となり、登録したD>=2の3rowは適格だった。
RZの`gamma*h=1.2, D=2`は誤差＋MC幅＋reference allowance `0.031120052`で元基準を満たさず、
**元登録の全D>=2条件はNOT_MET**のまま保存する。これは許容biasの超過を証明した結果でもない。
D=4の`gamma*h=1.2/0.6/0.3`は同じ上限が`0.027449620/0.025853256/0.024876762`で適格だった。
このD4資格は観測後に選んだscopeであり、元キャンペーン全PASSへ読み替えない。
元のD2未達は有限MC幅を含むため、別seed `53649`、N=64,000、同じ`gamma*h=1.2, D=2`で追加確認を
観測前に登録した。単run DKW alpha `0.0005`、budget `0.03`とreference allowance `0.005`を固定し、
CDF差 `0.0060591451`＋MC幅 `0.0080496747`＋reference allowanceの上限 `0.0191088198`で適格となった。
これは観測後に計画した独立確認の結果であり、元キャンペーンNOT_METを取り消さない。異なるcampaignの
family alphaを一つの同時保証へ合算せず、scopeと信頼幅をそれぞれ保存する。

単調壁接近の修正は元2粒子ID・seed・input hashを保持した公開API最小回帰でfailure 2から0へ解消した。
非単調な壁越え/再入経路はclearにせず、同じevent queryのnegative回帰で検査する。
原登録、初回failure、修正版raw result、追加確認の観測前登録・scopeは
[壁精度検証receipt](../../reviews/brownian_wall_accuracy_extension_2026-10-10.json)が所有する。
一般可変係数SDE、任意wall law、連続OUのexact first-passage/zero-miss、一般弱/強次数はこの資格に含めない。
