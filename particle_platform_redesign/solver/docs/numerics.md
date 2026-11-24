# Numerical methods and accepted paths

本書は現在の積分、field supportの証明、数値path、first hit、axis、RNG、accepted stateと観測の規則を所有する。
物理式・適用域・wallの応答式は[physics models](physics_models.md)、入力の構文・SI・hashは
[case format](case_format_v3.md)、永続形式とreader/recoveryは[result format](result_format_v5.md)を参照する。
process資源・容量・性能は[CPU execution](parallel_execution_plan.md)、外部比較の資格は
[V&V methodology](../../vv_methodology.md)が所有する。以下のmicrocase値や限定oracleを一般defaultへ流用しない。
過去の導入順、旧gate、revision roster、時間/RSSはGit履歴とsource付き実行receiptに残し、現在の受理規則へ混ぜない。

## State, stages and data flow

最小の動的stateは位置x、速度v、連続平均電荷数Zである。固定ID、release時刻、lifecycle、logical event ordinal、
physical wall ordinal、最後のaccepted時刻、探索hintはそれぞれの役割を分ける。
慣性はmass_kg、drag径・電気半径・排除体積・接触半径は各独立authorityを使う。

一つの粒子区間は次の順で処理する。

1. accepted始点から選択methodのproposalを作る。
2. 実際のstage位置・速度・Z・時刻でfieldと全選択physicsを評価する。
3. 数値path上のearliest material/axis/periodic eventとsupport/applicabilityを認証する。
4. accepted prefixだけをcommitし、eventがあれば同じpathのhit x/v/Zへ応答を一回適用する。
5. activeならpost-event stateから残時間を進める。観測は確定したpathから読む。

field/physicsの同一stageを使い、chargeだけを先にsubcycleして運動へ後付けしない。
physicsはcharge→drag→thermophoresis→ion drag→DEP→lift→electric→gravityの順で合成し、
linear relaxationのrate/targetとadditive accelerationを同じ評価から返す。式は積分器側で再計算しない。

親prepareは入力/provenance、解決済み実行facts、出力時刻、field knot、境界説明、全memory planを保持する。
transport/workerは固定numeric binding、prepared proof/bounds、場・geometry・scheduleと必要なcapacityだけを受ける。
point値を評価できることと、全pathのsupport/applicabilityやconstant specializationの証明は別である。

## Clocks and numerical intervals

要求macro dt、field snapshot knot、粒子release、coefficient root、event/safety再積分、OU leaf、観測clockを区別する。
出力時刻のためproduction stepを切らず、一粒子のreleaseのため全粒子macroを切らない。

この親時計の唯一の所有者は`time_grid.MacroGrid`であり、そのpolicyをrun manifestへ記録する。固定gridの補償積とrun終端のroundoff snapping、非grid knotの併合を
engineと外部評価が同じpure規則で扱う。`run.json`とresume identityには`macro_grid_revision`を保存する。
nominal dtがrun両端の時刻をbinary64で進められない場合、または必要な固定grid indexが連続整数の表現範囲
`2**53`を超える場合は、区間の反復・件数照会を開始せずprepareで失敗させる。
完成runの件数は解決済み全gridの件数と照合でき、選択時間窓の区間は全runのboundary列を保存せずに復元できる。
観測を含む最後のmacro intervalは観測時刻へ切り詰めない。releaseが区間内部なら最初の粒子区間だけが短くなる。

親macro分割が復元できても、粒子別rootや安全性の再積分が同じ解像度であるとは限らない。
全区間件数から特定cohort/windowの収束を推定しない。比較器固有の十分条件は
[外部tool契約](../tools/vv/comsol/README.md#deterministic-pre-event-comparison)を参照する。

## Field sampling and support

required fieldは一つの共通layoutへ束縛する。regularは全cell supportとparticle-domain nodeのbox内包、
P1/Q1はparticle volume meshとのnodes/connectivity完全一致と全cell supportをprepareで要求する。
boundaryless stage-evaluated pathはfully-supported regular support boxを必要とし、unstructuredのunion supportを
推測して受理しない。geometryとfield layoutが同じという一般前提を置くのではなく、実装済みの対応条件を明示検査する。

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


`cell_hint`はaccepted endpoint/prefixだけが更新し、trial、dense読取り、frame/probeは更新しない。
P1/Q1のstrict-interior hintだけをfast pathにし、miss/共有面はsupported owner規則へ戻る。
空間gradientは同じcell/basisの解析微分を外部producerが必要時に使う。stored componentの偏微分であり、
3-D共変微分やnative recoveryとは異なる。cacheの認証は[field preprocessor](../src/chamber_particles_tools/field_preprocessor/README.md)が所有する。

### Fixed-topology time fields

stage時刻を \(t\in[t_k,t_{k+1}]\)、既存regular/P1/Q1/cell空間補間作用素を \(S_x\) とすると、

\[
\alpha={t-t_k\over t_{k+1}-t_k},\qquad
F(x,t)=(1-\alpha)S_x(F_k)+\alpha S_x(F_{k+1}).
\]

locator、owner、空間weightはstage位置で一度だけ求め、二snapshotへ同じ値を使う。したがってP1/Q1の
conditioning、support、共有面owner、RZ basis/axis regularityは同じ空間ownerを使い、時間専用の
第二samplerを作らない。prepare時のrun-global primitive boundは引き続き全snapshotのcomponent extremaを
外向きに含む。一方、local continuous-path certificateは各proposal部分区間`[t_a,t_b]`と重なる線形時間区間を
挟むsnapshotだけを使う。時間補間と空間形状関数の凸性により、そのsnapshot extremaは区間内の補間値を包含する。
snapshot範囲外を外挿して証明せず、区間が範囲外ならfail-closedにする。

snapshot端点は閉区間として受理し、範囲外は外挿・clampせず`FIELD_KERNEL_TIME_OUTSIDE`でfail-closedにする。
required fieldはprepare時にrunの閉区間全体を覆う必要がある。積分器はactual stage timeをsamplerへ渡し、
required temporal fieldのrun内部snapshot knotの和集合を固定`dt_s` gridへ併合する。これにより係数の傾きが
変わるknotを一つのmacro intervalが跨がない。gridと一致するknotは重複境界にせず、静的場だけのcaseは従来の
macro partitionを保つ。


## Deterministic integration

### Exact ballistic and constant acceleration

force-free pathは `x(t)=x_release+(t-t_release)*v_release`、`v(t)=v_release` である。
macroごとの丸め済み位置を新しい解析originへ積み上げず、releaseまたは物理応答後のoriginを保持する。
release時刻の読取りは入力stateを厳密に返す。有限入力からelapsed/位置が表現不能なら成功形へclampしない。

fixed charge、dragなし、`cartesian_xy`に加え、全選択contributionについてprepareのconstant acceleration証明が成立した場合だけ、
粒子別合成加速度pathをこのspecializationへ送る。field値の一様性だけでは速度依存lift等を排除できず、十分条件ではない。
使用するfieldの全canonical node値は全snapshotを通じてcomponentごとに厳密一致することを要求し、近似的に一様な入力を
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

完全な材料boundaryがある場合、no-hitは連続pathがfield coverageを証明済みのparticle domainから退出しないことも
意味する。boundaryless caseは全cell supportedな共通regular layoutに限り、各座標の極値を始点、終点、
`t_turn=-v0/a`がproposal開区間にある場合の転回点から解析的に求め、閉じたregular support box内であることを
検査する。`x(0),x(0.5),x(1)`がsupport内でも頂点がoutsideになる反例はこの極値判定で拒否し、
frame有無で成否を変えない。

この極値区間は、`integrators.py`が所有する。始点・終点に加え、内部転回点を
dense stateと同じ`x0 + t v0 + 0.5 (t t) a`の演算順で評価する。転回値を1 ULPだけ広げる方法では、解析転回時刻の
隣接float時刻でdense stateが区間外へ出る反例がある。このため
`|x0| + h |v0| + 0.5 h^2 |a|`へ、6回のdense式演算、scale構築、転回時刻除算を覆う`16 eps`を掛ける。
さらに`t*t`または`0.5*(t*t)`のunderflowが後続の大きな`|a|`で増幅されるため、
`16 * smallest_subnormal * (1 + |a|)`を加えた絶対roundoff paddingを作り、区間全体を最後に`nextafter`で
外側へ丸める。engineは返された区間とregular
support boxを比較するだけで、別の代数的再編成による頂点を再計算しない。proposal endpointとdense stateは同じ算術順の値を使う。

### Classical RK4

共同state y=(x,v,Z)へ同じ4 stageでf(y,t)を評価し、

```text
k1 = f(y0,t0)
k2 = f(y0+h*k1/2,t0+h/2)
k3 = f(y0+h*k2/2,t0+h/2)
k4 = f(y0+h*k3,t0+h)
y1 = y0+h*(k1+2*k2+2*k3+k4)/6
```

を使う。endpointでの追加評価はdense state/pathと安全性のためのもので、別の第5 RK stageへ係数を変更しない。
explicit RK4だけがprepare済み全域boundに対する `dt*drag_velocity_lipschitz < 2.5` と
`dt*charge_lipschitz <= 0.5` を要求する。これらは精度保証ではない。
exact specializationはfixed chargeと必要な全contributionのconstant証明がある場合だけであり、
charge-onlyを含む動的Zをlinear/quadratic pathへ送らない。

### Deterministic exponential midpoint

deterministic exponential midpointは`dv/dt=-lambda(v-u)+a_add`を一step内でmidpoint係数へ凍結し、線形drag部分を解析的に
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

積分器は同じ`StepProposal`を返す。
`state_at()`、wall/axis hit時刻までの短縮再積分、hit後の残時間、trajectory/frame/probeは同じmethodを
proposal始点から再評価する。局所stage enclosureはrun-global包絡を
常にfallbackとして保持し、start half-step predictorが通る局所field rangeで認証したstage force boundとの
共通部分だけを採用する。局所range、force算術、stage enclosureのいずれかが非有限・未証明ならglobal包絡を維持し、
狭めるために安全性を弱めない。連続applicabilityもglobal-firstで、未証明rowだけ局所primitive rangeと
prepared charge invariantで再認証する。event locatorへ渡す曲線/chord偏差はposition box全幅ではなく、
全短縮位置secantを含むvelocity enclosureに対して成分ごとに
`h * (v_upper - v_lower) + float64 roundoff padding`で包絡する。このmethod-neutral boundを
RK4と指数法の共通first-hit/refinement loopで用い、材料壁とRZ axisを調停する。第二event engine、
endpoint chordだけのno-hit判定、出力時刻によるproduction step分割は追加しない。

### Coupled charge update

exponential midpointとBrownian midpointは、実midpointの `G=dZ/dt`、`J=dG/dZ<=0`、Z_midを使う。
root accepted始点へ戻したaffine lawは

```text
A = G_mid+J*(Z0-Z_mid)
Z(t0+tau) = Z0+expm1(J*tau)/J*A       (J != 0)
Z(t0+tau) = Z0+tau*A                 (J == 0)
```

である。運動のCoulomb/ion等も同stage Z_midで評価する。predictorは評価点を作るだけでhalf stateをcommitしない。
fixed chargeはG=J=0へ退化する。stage、dense時刻、endpoint、hit prefixでfinite invariantを守り、
clipやquasistatic equilibrium、scalar implicit fallback、charge-only subcycleで不成立を隠さない。
exponential pathはRK4用のhL_Zをstability gateにしない。精度は別resolutionの実行で評価する。

### Dense state and immutable root path

一般RK4 proposalはrootごとに一つのdense state pathを構築する。位置の四つのBezier制御点は
`x0, x0+h*v0/3, x1-h*v1/3, x1`で、event locatorが使うcubic Hermite pathと同一である。速度と電荷はRK4の
stage-rateから得るcubic extensionを使い、始点と受理endpointを変えない。harmonic position/velocityでは
θ=0.37,0.61でexact-startの内部一step O(h^4)と、固定終時刻0.7 sのglobal O(h^4)を別々に解析解へ照合した。
endpointの局所O(h^5)とは別の確認であり、速度extensionが位置polynomialの導関数であることは要求しない。
非滑らかなfieldや任意のdense interiorへの一律の次数保証にはしない。

dense位置評価は位置制御点をroot始点相対に変換し、TwoDiffが返す丸め差と厳密残差の
両方をBezier評価してroot始点へ足す。root始終時刻では保存済みの位置・速度・電荷endpointを厳密に
上書きする。保存する制御点と数学的なpolynomialはそのままに、dense位置のfloat64評価を原点相対で行う。

certificateはこの不変なroot pathのparameter部分区間をBezier subdivisionで制限する。位置enclosureは上記の
原点相対制御点を使い、TwoDiff残差の絶対値をroundoff paddingに加える。下限は`-inf`、上限は`+inf`
方向へ丸めながらroot始点を足し、world座標へ戻す。位置の微分制御点と物理的なchord deviationも同じ相対差と
TwoDiff残差を使い、等しい制御点の差は厳密に0として扱う。これにより物理的な曲率とenclosure幅は絶対座標ではなく
pathの局所変位に追従する。

ただし公開`rk4_dense_chord_deviation()`の戻り値は、最終のworld座標変換、両endpoint、endpoint chord式を
float64で評価する丸めを覆うため、`8*eps*max(abs(root_position_controls))`と対応するsubnormal termを別に加える。
この狭い絶対座標termのため、公開chord bound自体は完全な平行移動不変ではない。原点相対評価は、広い
roundoff幅が絶対座標に比例して膨張し、dyadic制限しても十分に縮まずevent certificateが閉じなかった問題を避ける。

## Continuous support and applicability proofs

catalogのfield-role/value条件と、選択revisionごとの閉じたproof familyをprepareで対応させる。
全選択forceに実global boundがなければ準備を失敗させる。unknown stage/proof codeを他familyへ読み替えない。
非負constant zeroの証明と、正の定数を含む外向きroundoff enclosureを区別する。

stageの有限sampleだけでは短縮pathの包含を証明できない。primitive extrema、粒子物性、charge invariantと
速度boundから全選択forceの保守的bounder B(V)を作る。

粒子ごとのproposal幅を`h`、成分別の絶対上限を`V_i`,`A_i`、globalな加速度bounderを
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
順序を認証できない値を成功扱いにしない。boundaryless regular caseではこのbox全体がclosed support box内にある時だけ、full proposalと
`StepProposal.state_at()`による全短縮proposalのfield samplingを許可する。stage/endpointの実sample verdictは
実装整合性の検査として残すが、連続証明の代替にはしない。

model applicabilityも同じ全区間へ要求する。例えばlinear Epsteinはlambda/aと相対速度比のboundで認証し、
実際のforce式を計算できたことから適用域内と推定しない。

### Local certificates

位置tubeと交差するfield cellから局所primitive rangeを作り、physics runtimeがrange上のcharge、force、
速度依存applicabilityを外向きに評価する。電荷intervalは全制御点が非正または非負ならその符号を保存し、
丸めで不変条件を跨がせない。数学的なimmutable root pathを証明区間だけで変更しない。まずglobal applicabilityを試し、未証明rowへlocal certificateを使う。

局所cell候補は一row最大64件である。overflowは適用域違反でなく「この区間ではrangeを構築できない」ため、共有する
event `max_refinements` budget内で区間を二分する。実stage/sampleまたは局所rangeで確認した違反は
`field_support`または`model_applicability`、全区間を証明できないままbudgetを使い切った場合だけ
`indeterminate_applicability_certificate`とする。accuracy不足を物理違反または成功へ畳み込まない。
同じ実sampleでsupportとmodel評価の両方が成立しない場合は、場を定義できない`field_support`を先に分類する。

### Exponential predictor support

局所加速度boundを採用する前に、transportはfrozen accepted-start predictorの全短縮半区間を囲うboxがfield supportに含まれることを要求する。
predictor sampleのsupport flagは係数を評価したstartの情報であり、予測先の支持証明として使わない。
RegularLayoutでは既存の完全支持boxへの包含を検査する。unstructured layoutは準備時に全cell支持とparticle-volume meshとの完全一致を要求するため、
支持startを含む連結boxが全exterior boundaryから離れていれば、box全体が同じ支持領域内にある。
geometry準備はholeを含む全exterior edgeを監査する。RZは既存coordinate ownerでboxをcanonical r>=0へ写すので、省略可能なaxis seamを支持外への出口としない。
canonical boxの中心から各成分の最大到達距離とhypotを外向きに丸め、boxを囲うdiskを作る。
既存の`centers_respect_contact_radius`へ`allow_contact=False`と既存event budgetの正の位置余裕を渡し、disk全体が全segmentから厳密に離れていることを要求する。
最大facet長と全中心の座標scaleから解いたbudgetはclearance条件を強めるために使い、衝突判定を緩めない。
particle-center/periodic faceがcapsule viewで無効でも、全facetが有効なevent geometry viewを使う。完全inventory・全facet有効・clearanceのいずれかを証明できないrowは既存global enclosureを保持する。
この距離証明は斜めsegmentのAABB false positiveを回避し、実際に境界へ届くboxを許可しない。支持外nearest-cell projectionを局所boundの証明へ流用しない。

支持された全短縮midpointで補間に使うcellは既存local range queryに含まれ、P1/Q1の節点範囲が係数を囲う。
これによりlocal boundは数値proposalのendpointとsame-integratorの全短縮prefixを囲う。global fallbackは同じ既存boundを保ち、
first-hit探索と実際の支持/適用判定がaccepted prefixを決める。支持外trialを受理する変更ではない。
独立verificationは二つの離れた完全支持P1領域を使う。左領域では一定weak drag、右領域では逆向き電場とし、粗いstepの半区間trialが右領域へ投影されても、
左領域のx=1への最初のstickと衝突前frameが解析的drag解に一致することを公開APIで検査する。

## Inertial Brownian integration

物理的FDTと独立prescribed force diffusionの係数authorityは[physics models](physics_models.md)が所有する。
両者とも同じjoint OU/tree/eventへ接続する。FDTはtheta=k_B*T/m、独立force diffusionは
`gamma=beta/m, theta=D/(2*beta*m)`を使い、後者のDをFDT温度へ黙って置換しない。
各rootのnoise-free coupled predictorでmidpointを作り、gamma,u,a,G,J,Z_midを一回評価して凍結する。
`u_eff=u+a/gamma`と上記affine chargeを使い、support/applicabilityはroot draw前に確定する。
prepareのgamma*dtと各rootのgamma*hは有限な正値で1e6以下を要求するが、これだけで表現可能性は保証されない。

### Joint position and velocity increment

局所区間で線形緩和率`gamma`、有効平衡速度`u_eff`、有効variance `theta`を固定すると、各Cartesian／signed-meridional成分は

```text
dv = -gamma (v-u_eff) dt + sqrt(2 gamma theta) dW
dx = v dt
```

である。`a=exp(-gamma h)`、`q=1-a`とすると、平均は

```text
E[v_h] = u_eff + a (v_0-u_eff)
E[x_h] = x_0 + u_eff h + q (v_0-u_eff)/gamma
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

### Conditional tree

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

設定base depth `interval_tree_depth` は全rootに一様に適用する数値path精度のauthorityで、範囲は0..10である。
`adaptive_max_depth`はwall/axis/証明不能候補だけを条件付きでさらに分割する上限であり、base depthの代替ではない。
base=maxなら固定depthへ退化する。depth-first traversalでは各levelの未処理right childだけを保持し、
観測時刻やevent探索のため親drawを捨てて独立な子drawを引き直さない。

### Hermite path, absolute target and restart

leafのjoint OU endpoint位置/速度からcubic Hermite pathを作る。event、frame、probeはこの同じ数値pathを使う。
OU endpointの凍結係数exactnessと、有限Hermite pathの内部時刻lawは別である。
連続OUのexact first-passage、zero miss probability、任意SDEのstrong order/weak 2次を主張しない。
RMS safety bandや確率的thresholdを決定論的clear証明に使わない。

root durationはOU transitionと等分conditional tree、absolute targetはleaf時計のauthorityである。
guardで短縮しなければmacro終端、短縮rootはstart+guarded_durationをtargetへ保持する。
最後のdyadic nodeだけをabsolute targetで閉じ、内部nodeはfraction時刻を使う。
wall/axis/release後の丸めtailを余分なnoise root、frame clamp、近似時刻比較で吸収しない。

active wall、axis、periodic transferでは認証済みprefixをcommitし、post-event stateから残時間を
次root ordinalのfresh drawと再評価した係数で始める。元rootの未使用cubic tailを反射/fold/restrictしない。
surfaceのzero-time responseはroot係数を作る前に解決する。root/guard/base/DFSの数値非有限は
最後のaccepted stateで粒子failureへ局在させ、正常rowや独立right subtreeを継続する。

## Material contact and wall events

first hitはproposalの数値path上で求め、endpoint chordだけの内外判定で代用しない。
linear/quadraticの解析交差、一般曲線のenclosureとtransverse certificate、Hermite/Bernsteinの凸包を使い分ける。
一意性を証明できなければleft-firstにsplitし、no-hitへ丸めない。
RK4/exponentialは各accepted local始点からproposalを再構成する。immutable dense pathの証明細分を、
物理的に再積分したsequential pieceと混同しない。

`proposal → earliest event → no-hit行のvalidity → commit`の順を守る。
trial tailのsupport/applicabilityはprovisionalで保持し、先行hitなら同じmethodでhit時刻まで戻したvalid prefixだけを受理する。
fieldを定義できない実sampleはmodel評価へ通さず、非有限算術を合法なoutside値へ偽装しない。
clearなら全区間のsupport/applicabilityを確認してからendpointをcommitする。

### Point and finite-radius contact

接触は独立contact_radiusを使う。0はpoint、正はstatic XY disk/RZ sphere中心対segment-capsuleのfirst contactである。
groupのparticle_center指定は当該facetの幾何半径だけを0とし、drag/electric寸法を変更しない。
path enclosureを接触半径で膨張したbroad phaseと、segment本体/端capのnarrow phaseを使う。
各candidateのcontact normalと同時接触集合を保持し、law/priorityから接触方式を推測しない。
exact反射後は各candidateへの接触と離脱を再証明できたzero-time rootだけを除外する。
原点やsource facetを無条件に除外せず、後刻impactを通常探索へ残す。位置nudgeを使わない。

### Event budgets and bounded work

YAMLのevent設定は物理局在budgetとwork上限を区別する。microcaseの値を装置caseのdefaultにしない。
以下では `r=geometry_rtol`、`U=roundoff_ulps` とする。

prepareは既存入力値から`NumericalWorkPolicy`を一度解決する。proposal/residualのdepth上限、
immutable dense pathのcertificate subdivision数、係数midpoint guardの半減数・restart数、
決定論intervalのinteraction trigger、確率rootのmacro内restart上限は別の役割として保持する。
現行の要求値と停止規則は変えず、公開knobや別advance loopを追加しない。wavefront/certificateの容量計画と
実行はそれぞれ同じ解決済みpolicyを使い、`run.json.numerical_work_policy`へ意味と上限を保存する。
決定論のinteraction値は下記の残区間二分triggerであり、確率restartのhard limitとは異なる。

決定論のexact／RK4／exponential経路では、`max_interactions_per_step`はmacro step全体のhard hit総数ではなく、一つのresidual-work intervalを
二分するtriggerである。子intervalのinteraction countは0から始め、refinement depthは引き継ぐ。したがって
macro step全体ではこの値より多い物理eventを処理できるが、再分割depthを使い切れば
`numerical_event_budget`になる。

上限に達したintervalでは、残区間にさらにhitがあるかをfirst-hit queryで判定する。hitがなければ
event-freeな残区間を受理し、追加depthを消費しない。次のhitが存在する場合だけ、現在のaccepted boundary
stateを起点に残時間を等分し、depthを一つ増やして前半、後半の順で処理する。次の分割depthが
`max_refinements`を超えるなら、最後にacceptedされた状態と処理済みevent prefixを保持し、当該粒子をfailureにする。C09の上限1・depth上限1 probeでは
`0.5g`、`1.5g`の二eventを処理した時点で失敗し、次の未処理hitは`2.5g`である。depthを十分にした同じ
上限1 probeはbase caseと同じ5 eventと終端へ到達しなければならない。これにより「上限到達で即失敗」と
「未処理eventを捨てて成功」の両方を検出する。

Brownian経路では同じ設定値の役割が異なる。active surface、材料wall、RZ axis、periodic transfer後の
fresh stochastic rootへのrestartをmacro step内で数え、上限を超えたcountを持つ次rootの入口で
`numerical_event_budget`とする。直前の応答とaccepted prefixは保持する。
intervalの二分でこのrestart countをリセットせず、countが上限と等しいrootは処理できる。
support／applicabilityに伴うguard restartは`max_refinements`で別に有界化する。
axis foldはrestartには数えるが、材料wall eventやwall RNG ordinalを生成しない。
失敗reasonをphysical escapeへ読み替えず、このcapとfirst-hitの局在精度を区別する。

共通resolverは、局所facet長さ`l>0`、geometry bbox対角`L`、評価位置`x`、速度scale`V`、
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

ballistic pathでは`V=||v||`が区間全体の厳密なspeedであり、pathは直線segmentそのものである。
曲線pathでは`V`をcandidate facetごとに作らず、同じStepProposal path piece全体に対する
保守的なspeed upper boundをintegratorが提供する。endpoint速度だけで代用しない。facet `i`にはその長さ`l_i`から個別budgetを作り、
候補`i,j`の同時性比較には位置・時間とも`max(budget_i,budget_j)`を使う。この対称結合によりBVH走査順やprimaryの選び方で候補集合を変えない。
full-step/two-half-step差だけを証明済みの軌道またはsupport包含boundとして扱わない。

最大facet長から解決したbudgetはBVH broad phaseの候補漏れ防止だけに使う。初期位置のfacet距離、
near-parallelなline separation、交差parameterは各候補facetの長さから解決したbudgetで判定する。無関係な
巨大facetを追加しても局所的なinside/no-hit判定を変えてはならない。またsegment区間外の交点を端点へ丸める
場合は、無次元parameter slackを`min(position_budget/path_length, time_budget/interval)`とし、空間・時間の
両方を満たさない交点を早いeventとして受理しない。

### Simultaneous facets and corner response

corner selector
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
   同じ時だけ互換である。parameterなしの`isotropic`同士は互換とする。
   `probabilistic_stick`はtop-level law、`probability`、fallback law、そのfallbackの
   解決済み反発係数またはMaxwell parameterを含む完全なsignatureを一致させる。`stick`と`escape`も互いに
   別signatureである。
4. 互換なlaw responseは一回だけ適用する。反射responseでは、安定sort済みfacet順に入射中の外向き
   unit normalを加算し、正規化したeffective normalへ選択したreflection transformを一回だけ適用する。
5. normal和が丸め範囲で0、または非terminal lawの応答後速度がpriorityで除外した面も含む
   **全candidate facet**のいずれかへ外向きなら
   `indeterminate_boundary_policy`とする。
6. priorityで選ばなかったものを含む全candidate集合をstable順でeventへ保存し、physical boundary event
   ordinalは一回だけ進める。`primary_facet_id`は選択subsetの先頭とする。

応答式と適用条件は[wall response laws](physics_models.md#wall-response-laws)を使う。

これは非滑らかcornerの唯一の物理法則を主張するものではなく、初期製品の明示的で再現可能なpolicyである。
別policyが必要になった場合はrevisionを分け、同名の意味を変更しない。

### Surface source and departure

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

一定加速度surfaceのdepartureは外向き法線`n`に対し、次の順で分類する。

1. `v·n`がscale-aware budgetより負なら内向きdeparture、正ならzero-time impact。
2. `v·n`がbudget内でも非零なら、加速度で符号を上書きせずfail-closed。
3. `v·n`がfloat64で厳密に0の時だけ`a·n`を同様に分類する。負なら内向きdeparture、正ならzero-time
   impact、budget内ならfail-closed。

内向き加速度の時、engineの分類だけでsource facetを検索対象から外さない。departure tokenは別wall hitで速度が
変わるまでmacro intervalをまたいで保持する。各exact intervalでは各intervalで、開始点がsource supporting lineの内側または
境界bandにあること、法線速度が厳密tangentまたは明確に内向きであること、法線加速度が明確に内向きであることを
同じbudgetで再証明し、成立するintervalだけsource facetを候補から除外する。他facetは常に通常のfirst-hit検索へ
残し、別wall応答後はsource facetも通常検索へ戻す。この寿命規則によりreleaseがmacro終端なら次macroで再証明し、
微小`dt`を含む時刻partitionで軌道/eventを変えない。releaseがrun終端なら運動queryはなく、right-continuousな
release frameだけを保存する。外向き加速度のzero-time impactではterminal stick/escapeを適用できるが、反射応答が
内向きdepartureを作れなければ失敗する。

一般曲線pathのsurface/active-boundary departureでは、既存のfacet-interior証明を使う。初回surface releaseは
瞬間の`v·n`がscale-awareに厳密な負値の時だけ受理し、tangentまたは曖昧な値を局所加速度で推定しない。surfaceの
zero-time反射または通常hitのactive応答後も、post速度が明確に内向くことを要求する。各curved pieceでは、
開始点がcandidate facetのsupporting line上かつ端点budgetから離れ、区間全体の速度包絡から法線速度上界が厳密に
負であるfacetだけをstart-contact候補から除外する。facet端点/cornerはこの証明対象外である。他facetと後刻の再衝突は通常どおり検索する。証明不能なら候補を
保持して細分化し、refinement budgetまで曖昧ならfail-closedとする。

非終端応答ではhit状態へcanonical wall projectionを適用した後、そのpost-stateから元のtarget時刻までRK4を
再積分する。accepted pre-hit pieceはtrajectory replayに残し、同時刻のactive state jumpがpost-stateで上書きする。
terminal stateも同時刻から有効なので、surface release、stick、escape、hold、reflectionのframe意味論は右連続である。
holdはhit状態を保持したinactive terminalで、残時間や後続macroで物理更新しない。

## RZ chart and axis

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
axis event後に右連続なcanonical基底へfoldする。physicsの式・加算順を基底変換で変えない。

regular support boxとの照合はsigned enclosureをそのまま使わない。radial区間`[l,u]`のcanonical像は

```text
l <= 0 <= u : [0, max(-l, u)]
otherwise   : [min(abs(l), abs(u)), max(abs(l), abs(u))]
```

であり、axial区間は不変である。速度・加速度の絶対値boundとEpsteinの相対速度normはradial符号反転で
変わらないため、共通のcontinuous applicability certificateを同じauthorityとして使う。
軸上でradial位置・速度がともに0のRK4 stateは不変状態としてclearとし、偽のcrossingを作らない。
`fields.py`の`axis_accessible`は、geometryがaxisへ接する場合だけでなく、boundaryless fully-supported regular
support boxの`r_min=0`も含む。この判定はrequired vectorのaxis regularityだけを所有する。
`gravity_buoyancy_standard_v1`はこれと無関係に、annular domainを含む全RZ caseでradial gravityを厳密に0とする。
regularity検査済みのRZ vectorでも、skew P1/Q1の補間加算によりcanonical `r==0` sampleへ微小なradial丸め値が
生じ得る。`fields.sample`は`axis_accessible`かつquery radiusがfloat64で厳密に0の場合に限り、そのrequired RZ
vector radial sampleをexact `+0.0`へ復元する。これは近軸値を許容差でclampする処理ではなく、axis nodal radial値が
すべて厳密0であるとprepare時に証明したfieldの数学的regularityを補間後も保存する処理である。

axis locatorは数値pathの位置・速度enclosureとevent budgetを使う。pieceが軸から確実に離れる場合は
`clear`、全区間でradial速度が厳密に内向きかつ終点がsigned chartの非正側へ進み、chord deviation・丸め残差・
時間誤差がbudget内なら`hit`、それ以外は`split`とする。hit prefixは同じRK4で時刻まで再積分し、support/
applicabilityと`|rho_hit| <= position_budget`を検査してから`r=0`へcommitする。残時間はfold後stateから
同じtargetまで進める。wallとaxisを別々に判定し、片方でも`split`なら分割する。双方が局在済みなら、wallの最早時刻と
axisの最遅時刻の認証区間が重なる場合を含めwallを優先し、axisが明確に早い時だけfoldする。axis端点とmaterial
cornerの完全tieは一般RK4 cornerとして局在不能ならfail-closedとし、wall優先で一面へ丸めない。

Brownianのaxis foldはfresh-root restartに数えるが、材料wall event、physical wall ordinal、wall RNGを作らない。
決定論的axis foldも壁の付着判定を適用しない。RZ meridionalは二自由度モデルであり、
等方3-D Brownianのradial dynamicsを表さない。採用済み3-D拡張の方針は[技術調査](../../technical_research.md#adopted-unimplemented-3d)を参照する。

## Static XY translation periodicity

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

## Accepted state, failure, RNG and observation

serial/processは同じresident ParticleStateを借用する。exact/curved/surfaceのmaterial応答は
hit x/v/Z、logical/physical ordinal、lifecycle/terminal時刻、boundary ledger、wall件数を一つのcommit ownerへ渡す。
各methodのorigin、replay、残時間、departure証明、OU restartはそれぞれのownerに残す。
surfaceの物理応答をcommitした後にdeparture証明が失敗した場合、その物理ledgerを取り消さない。
axis、periodic、release、numerical failureをmaterial応答へ読み替えない。

壁drawのidentityは `(seed,particle_id,physical_boundary_event_ordinal,law_stream)`、Brownianは
`(seed,particle_id,macro_interval,root_interval,tree_level,tree_index,component,root-or-split stream)` である。
配列順、slab/worker数、保存頻度、checkpoint/resumeから独立し、mutable global cursorを持たない。
wallでは付着・混合branch・法線・接線streamを分離し、isotropicとMaxwellは同一eventで排他的なdraw bankを共有する。
coreはsource分布をsampleせず、外部でrealizeしたsource rowを使う。超越関数のplatform間最下位bitまでの一般保証はしない。

粒子に局在できるevent予算枯渇、局在/応答/departure不決定、support/applicability/invariant逸脱、非有限導出を
明示failureとして残す。有限placeholderは後段配列のshapeを保つためだけで、成功stateではない。
共有shape/revision/layout conditioning/global bound/topology/I/Oの不変条件破損はrun-fatalとし、粒子failureで隠さない。
安全性の未証明、実適用域違反、数値精度不足、physical escapeを一つのstatusへ畳み込まない。

frame/probeはaccepted pathのstate_atから評価し、residentを観測時刻へcommitしない。
同時刻のaxis/wall/terminal jumpは右連続に反映する。terminal holdは後続macroで更新しない。
保存scheduleを変えてsurvivor final、event、RNG、failure理由やproduction partitionを変えない。

親engineはaccepted macro barrierで累積work
`W=macro_steps+accepted_particle_pieces+candidate_queries+refinements` のepoch差が
`T=max(2^20,128N)`以上か最終macroの時にdurable commitを要求する。
これはoutput schedule/slab幅に依存しない。途中macroのworkをcheckpointへ混ぜない。
writer、checkpoint配列、immutable context、hash検証、回復と公開例外は[result format](result_format_v5.md)が所有する。

## Independent verification and limits

C01–C10の入力・解析期待値とmethod別budgetのauthorityは`tests/verification/microcases.py`である。
以下は独立oracleの役割であり、この文書改訂による再実行・新規資格を主張しない。

| 対象 | 独立基準と確認する内容 |
|---|---|
| ballistic / constant force | release origin、解析直線/放物線、転回点、同step内の複数hitと残時間 |
| linear drag | `v=u+exp(-t/tau)*(v0-u)`、`x=x0+u*t+tau*(1-exp(-t/tau))*(v0-u)` |
| drag＋一定a | 上式へ速度 `tau*(1-exp(-t/tau))*a`、位置 `tau*(t-tau*(1-exp(-t/tau)))*a` を加える |
| electric / buoyancy | qE/mの符号・mass authority、`(1-rho*Vdisp/m)*g`、独立物性 |
| P1 / Q1 | affine field、非affine isoparametric Q1、共有面owner、conditioning、outside provisionalとsupport分離 |
| surfaces / corners | zero-time departure、薄gapの複数反射、combined normal、priority矛盾、全candidate内向き性 |
| coupled physics | Decimal/速度quadrature/解析極限とbound包含。shared body parityは結線検査 |
| deterministic method | smooth解析caseの時間収束、dense内部時刻とendpointの別評価、h/mesh/eventの分離 |
| OU / conditional split | production式と独立なGreen-kernel積分、Gaussian conditioning、短長時間極限、親増分再合成 |
| public result identity | frameなし/あり、slab、worker、resumeでaccepted scienceを維持。container byteの同一性とは別 |

RK4のsmooth-case形式次数をP1界面・model switch・impact後へ一律に要求しない。
同じshared arithmetic同士の一致だけを解析的正しさの根拠にせず、限定oracleのPASSを任意SDEや物理validationへ拡張しない。

### Finite Hermite observations and planar first arrival

以下は保存済みの限定verification設計・結果の解釈であり、数値defaultや新しい確認的budgetではない。

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

### Repeated planar walls and RZ arrival

以下も元source/campaignに限定した証拠であり、現在の全用途資格ではない。

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
