# 粒子輸送基盤 アーキテクチャ／数値計算レビュー

本書はreview記録である。採用済みmodule責務、依存方向、runtime境界の権威は
[`architecture_proposal.md`](architecture_proposal.md) とし、両者が食い違う場合はproposalを修正してから
実装する。

## P18-C aggregate charge closeoutレビュー（2026-10-01）

P18-Cは第二のcharge engineを作らず、`physics/charge.py`がrate・導関数・有限invariant/boundを所有し、catalog、
runtime、compiled evaluatorが既存の一つのcontinuous-state passへ接続する構成で完了した。電子・集約正イオンの密度、
thermal voltage、正イオン速度・有効質量、screening長はcanonical field一つずつをauthorityとし、COMSOL名、Case P/A分岐、
parameter-or-fieldの二重入力をcoreへ入れていない。P15/P15-Dは意味論を変えず独立revisionとして残る。

数値監査では、正負/zero potential、ion-energy floor、指数clipを独立Decimal式で再現し、有限global bound、compiled/pure
parity、XY/RZ、RK4 4次・explicit midpoint 2次の連成収束、material event、checkpoint/resumeを確認した。相対速度上限は
実速度を切るparameterでなく、actual stageと連続pathの双方をfail-closedにする証明用envelopeである。新しいresident
state、subcycle、fallback、result/checkpoint schemaは追加していない。

外部V&Vは保存済み12 packageをCOMSOL再実行なしで層別評価した。export済み`phi1`を使う同式再生はPASS、coreの
screening式と標準`epsilon0`を含む厳密provider一致は定数規約差によりFAILとなった。閾値緩和やcore定数fitを行わず、
式parityとprovider完全一致を別statusとして維持する判断は妥当である。数値のauthorityは
[`solver/evidence/p18c/README.md`](solver/evidence/p18c/README.md)であり、この証拠はintegrated chargeやtrajectory一致を
意味しない。

P18-C closeout当時のrevisionはengine v30、proposal v7、compiled tile v12、catalog v11、runtime v10、runtime layout v6、memory plan v12で、
抜本的なarchitecture変更は不要である。当時の次の優先work packageは、同じstage passへ排他的な二つのion-drag revisionを
追加するP18-Iである。

## B02 Brownian production closeoutレビュー（2026-09-30）

B01の数値primitiveを第二engineへ分岐せず、既存のcase catalog、単一macro-step engine、`StepProposal`、event、
output/checkpointへ縦に接続した設計は妥当である。対応範囲をCartesian XY、fixed charge、Epstein linear drag-only、
terminalな`stick`/`escape`へ限定したことで、drag-onlyの厳密OU平均へ未実装の決定論力を黙って混入させていない。
FDT温度はEpstein mappingの`gas_temperature_field`だけが所有し、noise専用fieldやmutable RNG stateを増やしていない。

数値pathは各macro-rootで凍結した係数のjoint OU endpointをexactに生成し、物理identityを持つconditional dyadic treeを
固定depthまで処理する。各leafのcubic Hermite polynomialだけがwall eventとframe/probe replayの共通authorityである。
したがって検証済みなのは有限depth numerical pathであり、sampleされていない連続OU trajectoryのexact first-passage、
miss probability 0、単一seedのCOMSOL pathwise一致ではない。この限界は仕様・manifest・受入試験で一貫している。

最終監査では`gamma*h<=1e6`でも極端な`theta`によりcovariance、split、deterministic meanがfloat64で表現不能に
なり得ることを確認した。engine v30は通常のvector batchを維持し、数値例外時だけrow別に再評価して不良粒子を
最後のaccepted stateで`nonfinite_physics`へ移す。同じslabの正常粒子と独立なtree branchは継続し、shape不一致や
内部不変条件破損はrun-fatalのままである。極端な不良rowと正常rowの混在公開scenarioがこの規則を固定する。

B02完了時点のrevisionはengine v30、proposal v7、catalog v10、runtime v9、runtime layout v6、memory plan v12である。
compiled tile v11、event v11、geometry v5、boundary v4、case schema v2、result/checkpoint schema 1は変更していない。
抜本的な設計変更は不要である。この後、本体を凍結した外部M3-Vで理論・field・integrator・force差を分離し、
共通canonical P1場・共通3力のpre-event時系列は独立自己収束と事前登録幅によりPASSした。この限定sliceでは
COMSOLと同じ時間刻み精度を認定した。native-field空間差、boundary event、Brownian ensembleは未認定の独立gateである。

## B01 Brownian数値基盤レビュー（2026-09-30）

Stage 2Bの最初の変更はproduction integratorを急いで公開せず、`stochastic.py`のjoint OU厳密更新と
conditional half-split、`rng.py`の物理interval-tree Philoxに限定した。この責務分割は妥当である。決定論的
`StepProposal.state_at()`へrandom force callbackを差し込まず、engine/event/output/checkpoint revisionを変えて
いないため、未完成のwall crossingが既存solverへ混入しない。

数値的には`k_B T/m`、線形緩和率、平衡速度を局所固定したintegrated OUのfull `(x,v)` covarianceを使い、
小さい`gamma h`は級数、conditional splitは無次元化した2x2 covarianceで評価する。20万replicaの平均・共分散、
16万replicaの左右half covariance・独立性・親endpoint保存、短長時間極限が合格した。Philox identityは
`seed, particle_id, macro_interval, root_stochastic_interval, tree_level, tree_index, component, stream`であり、
accepted-step、配列順、slab順へ
依存しない。root stochastic intervalを独立に持つため、同じmacro interval内で将来wall後に新しいOU区間を開始しても
既存tree nodeと衝突しない。

ここでproduction接続を止めた判断も必要である。現行のcurved-path enclosureは決定論的tubeであり、Gaussian bridgeを
有限幅へ確実に囲えない。したがって「壁からstochastic RMSだけ離れていればclear」という規則は採用しない。
次のexperimental revisionはgeometry/outputと独立な固定depth dyadic treeを全対象粒子へ適用し、leaf endpointの
位置・速度が定めるcubic Hermite pathを数値pathとして固定する。このpathに対するfirst hitを既存event interfaceへ
接続し、depth増加に対するfirst-passage分布収束をgateとする。adaptive clearはrun全体のmiss確率budgetを導出できた
場合だけ別revisionで追加する。

B01完了はBrownian production対応を意味しない。caseの`noise`、`ou_langevin`、材料壁、任意時刻replay、finite field
support、checkpoint-resume、RZ Brownianは未解禁である。RZ meridionalは方位自由度を欠くため、標準の等方Brownianを
載せず、P17 Cartesian 3-Dと独立に扱う。抜本的なarchitecture変更は不要だが、次の縦切りではcase/result revision、
fixed-depth scratch memory、event path revisionを同じ変更で更新する必要がある。B01平均はdrag-onlyなので、最初の
production縦切りはCartesian XY、Epstein linear drag、fixed charge、terminal stick/escapeに限定し、それ以外の力や
反射を黙って無視しない。

## P16完了レビュー（2026-09-30）

P16はarchitecture変更ではなく、既存force catalogへ一つの明示revision
`waldmann_gallis_free_molecular_single_species_heat_flux_v1`を追加して完了した。`physics/forces.py`が局所並進
熱流束形、Kn/relative-drift適用域、global acceleration boundを所有する。catalogはrequired fieldと単一neutral-gas
authority、runtime/compiled passはsample済みprimitiveの実stage評価だけを所有し、integrator、event、geometry、
writer、checkpointへmodel固有経路や新しいresident stateを追加していない。

入力`q_tr`はlocal mass-average neutral frameのtranslational conductive heat fluxであり、coreは温度fieldを微分しない。
`F_th=(32/15)a^2 q_tr/c_bar`、`c_bar=sqrt(8 k_B T_tr/(pi m_g))`とし、全stage/pathで
`lambda/a>=10`と`abs(u_g-v)/c_bar<=0.1`をfail-closedに要求する。Epstein併用時はneutral backgroundを完全一致させ、
適用Knが重ならないStokes--Cunninghamとの併用は拒否する。Talbot/continuum、mixture、near-wall、negative
thermophoresis、accommodation fittingは別revisionであり、自動blendやzero-force fallbackはない。

数値監査では、一次Chapman--Enskog分布の独立3-D Gauss--Hermite momentから得た係数で具体的なproduction加速度まで
比較し、方向・zero・scaling、global bound、局所/path適用域、compiled parityを確認した。affine heat-fluxの公開caseは
RK4 3.5次以上、exponential midpoint 1.8次以上で収束し、XY/RZ parityも合格した。全verification/scenario 456件と
標準品質gateを受け入れる。

無効modelの代表2,000粒子smokeでは、変更前1観測と変更後3観測で科学payload SHA-256、event work、solver-owned
planned memory 60,441,944 B（none）/60,446,345 B（sample）が完全一致した。simulate秒はnoneが1.178955に対し
変更後1.196202～1.309895、sampleが1.245020に対し1.120492～1.270651で、単一before観測から一方向の回帰は断定しない。
有効100,000-row warm stageは9回median 0.014549 s（約6.87 M row/s）、prepared boundは1,700,024 Bだった。
絶対時間はmachine-localな非gating観測である。

変更revisionはcompiled tile v11、physics catalog v9、physics runtime v8だけで、engine v28、proposal/enclosure、event、
runtime layout、memory plan、case/result/checkpoint schemaは不変である。P16完了時点で抜本的な設計変更は不要で、
次工程だったBrownianは上記B02で完了した。P17はstate-dimension独立workstreamのままとする。

## P15-F完了レビュー（2026-09-30）

P15-Fはarchitecture変更ではなく、既存force catalogへ一つの明示revision
`barnes_collisionless_effective_speed_single_positive_ion_negative_debye_huckel_v1`を追加して完了した。
`physics/forces.py`がcollection＋orbital式、linear two-species Debye length、Debye--Hückel表面電位、適用域、
加速度boundを所有する。catalogは選択とfield authority、runtime/compiled passは同じstage評価だけを所有し、
integrator、event、geometry、writer、checkpointへmodel固有分岐や第二stateを追加していない。

neutral dragのrelaxationとは分け、ion dragを相対イオン流方向の`explicit_acceleration`として既存合成順へ置いた。
continuous chargeと併用する場合、electron/ion density、temperature、ion velocity、ion massは完全一致を必須とし、
同じsample配列と実stage電荷を共有する。この一所有者化により、charge用とforce用に異なるplasma背景を暗黙に使わない。

物理範囲は単一・単価正イオン、球形完全吸収粒子、非正電位、collisionless・unmagnetized・dilute background、
weak-coupling screeningに限定した。`a/lambda_D`、`b_90/lambda_D`、`b_c/lambda_D`、ion-neutral mean-free-path比、
正のCoulomb log、宣言drift比をfail-closedに検査する。0.1/10の閾値はsharpな文献境界でなく狭いrevision policyである。
外部datasetのfloor、clamp、image補正、電場方向化、scale、別modelへのblendは採用していない。

数値監査では独立Rutherford impact-parameter quadrature、zero-flow/zero-charge、global acceleration bound、
compiled/reference parity、fixed/continuous chargeの同一stage結合、XY/RZ parityを確認した。公開一様場caseはRK4 3.5次以上、
explicit midpoint 1.8次以上で収束した。model固有の根拠のないdt gateやhidden subdivisionは加えず、既存固定stepの
収束責任を維持する。広いcontinuous-charge invariantは安全側にprepareを偽拒否し得るが、局所clampで緩めず、必要なら
共通charge interval enclosureの独立変更として扱う。

無効時のplanned memoryは9,063,029 Bで不変、同一warm simulate medianは0.046734 sから0.045303 s、
公開end-to-end medianは0.059475 sから0.057424 sとなり、回帰はなかった。有効時の100k-row direct stageは
0.014814 s、約6.75 M row/s、prepared bound総量は1,700,024 B（P15-F増分100,024 B）だった。
これらはmachine-localな非gating観測である。
変更revisionはcompiled tile v10、physics catalog v8、physics runtime v7だけで、engine v28、proposal/enclosure、
event、runtime layout、memory plan、case/result/checkpoint schemaは不変である。抜本的な設計変更は不要で、次は
P16 Waldmann、その後Brownianとする。追加のcollisional/nonlinear ion dragは用途と独立referenceが得られた時だけ扱う。

## P15-E完了レビュー（2026-09-30）

P15-Eはarchitecture変更ではなく、既存drag catalogの一つの明示revisionとして完了した。
`physics/forces.py`がMaxwell混合表面則を持つ有限速度球抗力、低速級数、適用域、rate/Jacobian boundを所有し、
catalog/runtime/compiled passだけが選択とstage評価を追加した。integrator、event、geometry、writer、checkpointへ
model固有分岐や新しいresident stateを追加していないため、責務分離は維持されている。

物理範囲は孤立・非回転球、局所shifted-Maxwellian単一中性気体、自由分子流、鏡面と完全熱適応・等温拡散再放出の
混合に限定した。`diffuse_reflection_fraction`と`maximum_speed_ratio`を必須にし、後者はclampでなく全stage/pathの
fail-closed適用包絡である。`T_w!=T_g`、CLL、混合気体species総和、非球形、transition blend、near-wall補正は
別revisionへ残した。既存`epstein_linear_v1`は安価で明示的な低速modelとして残し、自動fallbackを作っていない。

数値監査では、3-D Gauss--Hermite分子速度積分、低速linear極限、高速`C_D`極限、速度Jacobian有限差分、global
bound、compiled/reference parityを確認した。非線形rateに対して加速度/enclosure用`G`上界とRK4 stiffness用
`K=G+S G'`上界を分離し、rateだけの不十分な安定性判定を残していない。公開非線形減速caseはRK4 3.5次以上、
explicit midpoint 1.8次以上で収束した。変更revisionはcompiled tile v9、physics catalog v7、physics runtime v6だけで、
engine v28、proposal/enclosure、event、memory plan、case/result/checkpoint schemaは不変である。

抜本的な設計変更は不要である。P15-E完了時点の次項目はversioned ion drag、その後P16 Waldmann、Brownianとした。
COMSOL比較や`model_dataset`差に合わせる係数・分岐は引き続き外部V&Vだけが所有する。

## F02完了レビュー（2026-09-29）

F02はarchitectureを作り直さず、既存の責務境界を実入力で閉じた。`solver/tools/comsol_adapter/`だけが
provider固有CSV、mixed triangle/Q1-quad、外部entity ID、axis seamを扱い、canonical P1を書き出す。
`solver/tools/electrostatic_builder/`は完成したthermal-flow bundleだけを読み、F01の単一解法でfieldを生成する。
既存三APIは完成fieldの由来を知らずに粒子を進め、`solver/tools/vv/comsol/`だけが外部差分を記述する。
したがってsolver core、case/result schema、engine/physics revisionの変更は不要だった。

最終監査で、境界IDの過不足とaxis facetの両端`r=0`を双方向に検査し、YAML重複key、CSV header/row幅・不正quote、
任意の非数値tokenとinfinityをfail-closedにした。全source/referenceは一度だけbytes snapshotを取り、同じbytesを
parseとhashへ使う。field node IDを持たない既存CSVは、明示tolerance内で各canonical nodeに厳密に1点かつ全体が
全単射となる場合だけ許可する。新規exportのtopology ID結合という原則は維持し、補間、丸めbucket、最近傍fallbackは
追加しない。外部比較は生成fieldのunit/components/basis/layoutと参照unitを検査し、normを`2 pi r`軸対称volumeで
積分する。

代表入力は1,987 node、2,127 triangle＋826 quadから3,779 P1 cellを生成し、193外周facetのうち33 axis facetを
除外して160物理facetを6 groupへ一意に割り当てた。最小triangle品質は`0.0423878260`、再生成hashは一致した。
builderは1,826 free node、2,821 total linear iteration、最終relative residual `1.8441e-13`、charge-balance error
`5.8498e-21 C`で完了し、GMRES storageは1,235,088 B、solve-only観測は0.689/0.737 sだった。現解法で代表gateを
満たしたため、fallback、第二linear solver、preconditionerは増やしていない。32粒子fixed-electric smokeも10 step、
3 frame/96 row、wall/failure 0で完了した。

比較結果は同一export node上の記述証拠であり、potential/Eのrelative weighted L2は約2.00%/17.05%だったが、
COMSOLを真値とする合否閾値には使わない。独立mesh convergence、COMSOL trajectory、wall/Freeze parityは未検証である。
`gas_inlet: escape`は未到達の通常lawで、COMSOL `Freeze`の翻訳ではない。小さいhash付き証跡は
`solver/evidence/f02/`が所有し、生成HDF5/resultはsource管理しない。全面的な方針変更は不要で、trajectory
physicsのspecies制約付きrelative-drift chargeをP15-Dで完了した。次は有限速度dragを優先し、P17は
state-dimensionの独立workstreamとして維持する。

## P15-D完了レビュー（2026-09-29）

P15-Dは、既存のcontinuous-charge ownershipを崩さず、有限相対driftを扱う
`oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1`を一つ追加した。rateとrun-wide boundは
`physics/charge.py`、選択と入力はcatalog、stage/path判定は既存runtime/compiled passが所有する。integrator、event、
geometry、writer、checkpointへmodel固有分岐を増やしていない。この責務分割は妥当で、別charge engineや汎用plugin層は不要である。

物理範囲はMaxwellian電子、単一・単価のshifted-Maxwellian正イオン、球形完全吸収粒子、非正表面電位に限定した。
zero-driftでstationary OMLの負電位branchへ連続に一致し、小driftは解析級数で評価する。caseが
`maximum_ion_drift_ratio`を明示し、初期`Z<=0`、全primitive/drift rangeで非正平衡、`a/lambda_D<=0.1`を
prepare時に認証する。速度floor、energy floor、指数clip、正電位branchへのfallbackは追加していない。

数値監査では独立3-D Maxwell速度積分、zero-drift一致、rate微分、global corner bound、compiled/reference parity、
continuous-path drift gateを検査した。既存のRK4・explicit midpoint、材料wall/frame、XY/RZ、64-step checkpoint/resumeも
同じ公開経路で合格した。新しいresident配列、memory component、case/result/checkpoint schemaはない。P15-D完了時点の
変更revisionはcompiled tile v8、physics catalog v6、physics runtime v5だけで、engine v28、proposal v6、両enclosure v2を維持した。

全面的な設計変更は不要である。ただし現行path gateは成分絶対上界を使うため強いco-flowを安全側に偽拒否し得る。
これを解くなら、RZ signed chartを含むvelocity interval enclosure全体を別revisionで改訂する。P15-D内だけで上界を
緩めない。Case Pの負イオン・非scalar有効質量、Case Aの一部正電位履歴、複数species、emission、sheath内surface releaseは
未対応なので、COMSOL差へ合わせる修正は行わない。その後P15-E finite-speed Epsteinを独立revisionとして完了した。

## M3-V完了レビュー（2026-09-29）

外部COMSOL/model_dataset評価をsolver本体へ混ぜず、`solver/tools/vv/comsol/`の一つの実行経路へ置いた判断は妥当である。
COMSOL 6.4からMPHを読み取り専用で直接inventoryし、12 packageの時間履歴、force relevance、charge/Epstein式parity、
ion-drag variant感度を評価した。P15 stationary OMLはion drift、Case-P species、scalar ion-mass前提により全軌道比較へ適用できず、
linear Epsteinも部分適用である。よってCOMSOLへ合わせる閾値緩和、互換mode、第二engineを追加しない。

計画は一列の優先度から三workstreamへ修正する。Case-A相当のreduced electrostatic builderはmandatoryなfirst-party
field producerであり、trajectory physicsの順位6ではない。trajectory physicsはspecies制約付きrelative-drift charge、
P15-E finite-speed Epstein、versioned ion drag、Waldmann、Brownianを別revisionで扱う。P17 Cartesian 3-Dはstate dimension
変更として独立する。この変更はsolver coreのmodule ownership、公開API、case/result schemaを変更しない。

## P15完了レビュー（2026-09-29）

P15は`oml_stationary_maxwellian_debye_huckel_v1`を一つのproduction modelとして追加し、最初に
`rk4_fixed`の`(x,v,Z)`四stage連成を受け入れ、その後`exponential_midpoint`へ同じ予測midpoint時刻の
explicit charge更新を接続して受け入れた。両methodはfinite charge invariant、rate/derivative bound、
driftと`a/lambda_D`のapplicability、`h L_Z <= 0.5`を同じownerから使う。動的電荷ではexact specializationを
禁止し、認証済みcharge区間をelectric accelerationとpath enclosureへ渡す。精度目的のhidden subdivision、
charge-only subcycle、implicit fallback、第二state/result schema、一般ODE/plugin frameworkは追加していない。

P15完了時点のrevisionはengine v28、compiled tile v7、proposal v6、physics catalog v5、physics runtime v4、
RK4 enclosure v2、exponential midpoint/enclosure v2、memory plan v11である。runtime layout v5、event v11、
geometry v5、boundary v4、field/required-field v3、result algorithm v3、case schema v2、canonical data/result
schema v1、checkpoint schema 1は維持した。XY/RZ、wall、右連続frame/probe、result、checkpointは既存stateと
単一production loopを再利用するため、architectureの抜本変更は不要である。

P15着手をexact P14-R Git baselineと初回remote CIまで禁止していた旧順序は、ユーザーの明示指示で解除した。
P14-Rのremote CIは未完の独立release trackとして正直に残す。その後M3-V外部評価を完了し、上の
M3-V完了レビューに示す三つの独立workstreamへ計画を更新した。

## P14-U完了レビュー（2026-09-29）

正式releaseは`solver/evidence/v0.1/p14u_release_v1.json`へ保存し、`release_gate_complete=true`となった。10k/100k/1M粒子、
`none`/sample各3 fresh processの18 raw観測と6 median、別実行の1M `none` profileを完了した。XYの固定空間mesh上の
時間収束、固定aspectのregular/P1/Q1 mesh収束、target first hitとfacet clearance、RZの軸横断収束と実canonical
required-field parity、失敗0、release/target-stick件数、科学payload・revision・event work identityはすべて合格した。

1M median `simulate`は`none` 550.43 s、sample 533.95 s、raw peak RSS最大は767.3 MiB、solver-owned memory planは
614.5 MiBだった。絶対秒数は当該machineだけの非gatingな製品制約であり、COMSOL比やportable性能ではない。
`none`とsampleは順次実行なので差をwriter単体costとせず、RSSはworker開始から公開`load_case`/`simulate`/
`open_result`完了までのprocess high-waterで、後続する外部検証読込みのpeakを含まない。

profile self-timeはevents 28.7%、fields 26.8%、engine 11.8%、integrators 11.7%、runtime/dependency 11.7%へ分散した。
単一owner支配は実証されていないため、局所bound、step controller、並列runtime、GPU、第二schedulerを追加しない。
P14-U完了時点のproductionはengine v27の単一compiled serial経路だった。P14-Uが検証したsurface分布はCartesian XY
`line_length`に限り、RZ `revolved_area`まで検証済みとはしない。全面的な設計変更は不要で、次はP14-Rの
再実行可能なrelease evidence・Windows/Linux wheel/clean-installとT03最小analysis/visualizationを閉じ、
P15 continuous chargeを後続とする、当時の順序を記録する。

## P14-U着手前の物理契約同期（2026-09-29）

P14-Uの性能・収束判定に先立ち、現行の物理名だけを整理した。`point_wall_laws_v4`では`specular`を
parameterなしの完全鏡面、`restitution`を法線・接線反発係数付きの別lawとし、`probabilistic_stick`は
非stick時の`specular | restitution`を`otherwise`へ必ず明示する。`gravity_buoyancy_standard_v1`はすべての
`axisymmetric_rz` domainで`g_r=0`を要求し、`cartesian_xy`の第1成分は非零を許す。schemaは
`edge_fraction`と、明示measureを持つ`uniform`を維持する。P14-Uが実測するensembleはCartesian XY
`line_length`だけであり、RZ `revolved_area`の分布品質や任意surface実現値まで検証済みとはしない。
この段階では`realized_surface_table`を追加しない。これはschema/model意味の同期であり、P14-Uの完了や
性能値を記録するものではない。

## P14後の立ち止まりレビュー（2026-09-28）

### 結論

P14までのsolver core全体に、もう一度clean-room rewriteを行う根拠はない。ただしparallel runtimeは例外であり、
P14時点のouter ThreadPoolとP14-Pの内部parallel試行を製品機能として温存しない。focused correction後も
regular 1Mの4-thread speedupは0.923x、1-threadはv20履歴比23.7%退行したため、case schema v2から
thread設定を削除し、P14-P closeoutをv27のsingle-thread compiled engineへ一本化した。三公開API、単一production
engine、sample済みprimitiveだけを受け取るphysics、integratorとgeometry/eventの分離、COMSOLをcore外へ
置く境界は維持する。`engine.py`、`events.py`、`output.py`は大きいが、現時点では責務が凝集しており、行数だけを
理由に分割すると共有stateとevent順序を複数fileへ拡散させる。新しいmanager、provider、plugin、診断frameworkは
追加しない。

### 並列化・物理的有用性の再監査

前回レビューはthread/tile間の科学的同一性、ownership、memory planとP14 matrixを確認したが、主用途である
「非一様場＋surface release＋材料壁＋多数macro step」を一つに結合した性能と、実チャンバーに対するmodel
coverageまでは閉じていなかった。このためP14を「汎用並列化完了」または「半導体プラズマ用途の妥当性完了」と
読むのは強すぎる。P14はsyntheticな性能baselineの完了と位置付け直す。

P14の3観測medianを再集計すると、20 workerのspeedupはregular 100kで1.880x、regular 1Mで4.797x、
P1/Q1 10k cross-cellで1.174x/1.147x、10k粒子×20 hitで0.888xだった。1M regularの実測RSSは
約537 MiBから1,143 MiBへ、solver memory planは約600 MiBから3,680 MiBへ増えた。P12の旧serial比6.96xは
BVH、事前認証、batch化を含む1 workerのalgorithm改善であり、thread speedupではない。このため既存multiworkerを
長期APIとして扱わず、P14-Pの否定的closeoutで削除した。

P14時点の実装はNumba `prange`ではなく`ThreadPoolExecutor`によるmicrotile並列で、外側のfield/physics/RK/geometry
kernelだけが`nogil`で並走する。一般曲線event、boundary response、residual work、stable merge、frame/probe、
writer ackにはPythonまたは直列区間が残る。またfield sample、physics、各stageは別配列を確保しており、
`technical_research.md`が目標にしていたtile slab再利用には達していない。512粒子×4 macro stepの曲線wall profileでも
約208万回のPython callを生じ、時間の大半がscalar locatorとPython refinement調停に集中した。これはworker数追加、
multiprocessing、第二scheduler、GPUでは解消しない。そこで外側ThreadPoolを廃止し、Numba内部thread team、
thread数非依存scratch slab、flat SoA event wavefrontを評価するP14-PをP14-Uより前に置いた。gate未達後は
内部thread teamも削除し、次段落の直列runtimeへ収束した。

P14-P closeoutのv27 / compiled tile v6 / proposal v5 / event v11 / `point_wall_laws_v4` / runtime layout v5 / memory plan v10 / geometry v5は、この判断に従ってouter pool、
future wave、worker別scratch、thread maskを削除し、再利用workspace、stackless boundary BVH、同期single-owner
writerまで統合した。linear/quadratic exactと一般曲線eventはflat SoA wavefront、boundary/Philoxはcompiled batch、
fieldからenclosureはrow numerical status、surface releaseはbatch、frame/probeはdirect columnar replayである。
memory plan v10はこれらをnamed componentへ分離し、pack時だけのgatherを12.5% safety marginへ含める。正確な
byte式と不採用判断は[`solver/docs/parallel_execution_plan.md`](solver/docs/parallel_execution_plan.md)が所有する。

物理modelは、宣言した狭いrevision内では式、符号、単位、適用域が整合している。Epstein linear、air向け
Stokes--Cunningham、Coulomb、重力・浮力、point-particleの理想壁則を全面的に作り直す根拠はない。一方、
`model_dataset`のtheory-consistentなP/A・10/30/100 nmケースを診断資料として集計すると、初期`Z=-1`に対して
保存履歴の電荷中央値は大きく負側へ変化し、electric、ion drag、Epstein、Brownian、thermophoresisの相対重要度も
粒径とcaseで入れ替わる。これはgolden truthではないが、fixed chargeだけでplasma代表軌道を主張できないこと、
ion drag/Brownianを自動的に遠い将来へ固定できないことを示す。P15 continuous chargeを最優先にする判断は維持し、
P15出口で累積impulse、軌道、付着感度とmodel applicabilityを外部toolで比較して、その後のmodel順を確定する。

現行`epstein_linear_v1`の`|u-v|/c_bar <= 0.1`は安全側の低速revisionであり、参照ケースの一部では外れる。
閾値を緩めず、必要性が確認された場合だけ有限速度Epsteinを別revisionとして追加する。Stokes--Cunninghamは
Allen--Raabe air revisionであり、CF4/O2低圧plasmaの自動fallbackにしない。RZ meridionalはaxisymmetric/no-swirlの
2成分運動であって完全3-D粒子運動ではない。P17とBrownianの先後はP15出口の外部評価で確定する。

数値的には、RK4の`dt/tau < 2.5`は安定性gateであって精度保証ではない。event locatorの厳しい残差も、選択した
離散path上の局在誤差であり真のODE軌道誤差ではない。global field extremaによる包絡は安全だが、局所sheathの極値を
遠方の全粒子へ課して過剰splitまたは拒否を生む可能性がある。P14-Pでparallel runtimeを閉じた後、P15前に小さい
P14-U representative-use gateを一つ置き、
surface release、非一様場、材料壁、多数stepを同時に使うcaseで次を測る。

- 固定空間mesh上の`h, h/2, h/4`による時間同期位置・速度・hit時刻/位置/facetの自己収束
- `nx=ny`固定aspect、layout別fine reference、facet endpoint clearanceによるregular/P1/Q1 mesh系列
- sourceとexact hitを含むdense path/global bound比、refinement深度、failure率、candidate数
- 10k/100k/1M、none/sample各3 fresh processのraw/median、RSS、solver plan、mode間core／mode内probe identity
- 別の1M非計時profileと、RZ軸を横切る可変場の自己収束・実canonical required-field parity

P14-Uはcore内の常設diagnosticを増やさず外部performance/V&V harnessで行う。実測したXY line sourceでは
現行入力が足りるため`realized_surface_table`は追加しない。結果が必要性を示した場合だけ、layout/BVH所有の
局所boundまたは限定的step controlを独立変更として検討する。parallel schedulerはP14-Pで削除まで閉じ、flat event
wavefrontはserial runtime内の一つのproduction pathとして固定した。P14-Uへ第二schedulerを持ち越さない。
任意の表面実現値は依然として現行table sourceでは表せないが、v0.1の要件にはしない。

一方、機能追加を続ける前に次を修正する。

1. P14のsynthetic baselineとtarget-use性能を分ける。先にP14-Pでparallel runtimeを単一方式へ収束させ、次に
   P14-Uを閉じる。その後Windows/Linuxのwheel・clean-install・
   三公開API smoke、再実行可能なP14 evidenceをP14-Rとして完了し、別trackのT03で最小analysis/visualizationを
   完了させる。既存`report_app`は設計監査資料であり、solver resultを読むT03ではない。
2. continuous chargeを単なる新しいrateとして解禁しない。当時の`force_coupled`は力の有無と「stage積分が必要」を
   兼務していたため、P15では`has_force`、`evolves_continuous_state`、`requires_stage_evaluation`を分離した。動的電荷は
   charge-only caseでも既存stage evaluatorを通し、当面linear/quadratic exact specializationを使わない。
3. electric accelerationと曲線path enclosureへ、初期値ではなく認証済みのcharge区間を渡す。charge modelが
   rate、rate/derivative bound、有限なcharge invariant、適用域を所有し、memory planはそのmicrotile scratchを
   数える。これがない状態で`dZ/dt != 0`を有効化すると、加速度上界とfirst-hit認証を過小評価し得る。
4. Stage 2Aを一括実装しない。P15 continuous chargeを独立縦切りとし、その出口のM3-V relevance gateを完了した。
   結果は冒頭のM3-V完了レビューどおり三workstreamへ分けた。これらは変更理由、schema影響、verificationが異なる。

### 帯電解法の方針修正

従来案の「強い剛性を持つ単調rateへbracket付きscalar implicit midpoint」は採用しない。線形緩和

\[
Z'=-k(Z-Z_*)
\]

に対するimplicit midpointの増幅率は

\[
G=\frac{1-hk/2}{1+hk/2}
\]

である。A安定だがL安定ではなく、`hk > 2`では平衡を跨いで符号反転し、`hk -> infinity`で
`|G| -> 1`となる。したがって電荷符号、ひいては電気力を数値的に振動させ得るため、stiff chargeの標準解法には
しない。

P15は、既存`rk4_fixed`を`(x,v,Z)`のreference pathとして使い、model固有のcharge relaxation boundから
非stiff範囲だけを最初に解禁した。native exponential-motion pathには、その受入後に同じmidpoint時刻を使うexplicit
midpoint chargeを追加して受け入れた。未対応の剛性はfail-closedとし、実ケースで必要性が確認された場合だけ、全状態bounded
dyadic、単一のL安定な全状態法、または別versionのquasistatic-equilibrium modelから一つを選ぶ。chargeだけを
暗黙更新して運動へ後付けする一次splitは作らない。

### 製品性と簡素性の判定

- 現行336件のverification/scenario、三つのimport contract、Radon gateは過剰とは判断しない。private helperや
  呼出順を固定するtestを増やさず、revision一式の重複assertは代表scenarioとrelease harnessへ集約する。
- `ResultView.read_boundary_events()`は全segmentを実体化する。T03前に同じreader primitiveを使うbatch iteratorを
  一つ追加し、100万粒子規模のanalysisでevent全量をmemoryへ要求しない。query DSLは作らない。
- このレビュー時点のGit作業treeでは旧実装の削除と新しい`particle_platform_redesign/`全体が未追跡で、P14完了状態を
  復元できるbaseline commitがなかった。この旧P15着手blockerはユーザーの明示指示で解除し、P14-R remote CIを
  独立release trackとして残した。
- P14の69観測raw JSONは一時領域だけにある。秒数をCI thresholdにはしないが、machine fingerprint、条件、
  revision、memory、semantic digestを含むrelease evidenceは再実行可能な場所へ保存する。
- P14はcore内のscalingとbottleneckを説明するが、「COMSOLより高速」をまだ証明しない。速度優位を製品上の
  主張にする前に、T02で同じ物理・出力・精度条件を固定した代表caseのend-to-end比較を行う。比較codeと
  COMSOL固有設定はcoreへ入れない。
- v0.1のsurface sourceは`uniform | edge_fraction`、`fixed | normal` velocity、fixed release timeに限定する。
  角度・速度・粒径・時刻の豊富な分布は後続能力であり、現行仕様として主張しない。table sourceが表せるのは
  任意のrealized interior初期条件であり、任意のboundary-start ensembleはまだ表せない。代表用途は
  `edge_fraction`または`line_length | meridional_length | revolved_area`を明示した`uniform`で表し、
  `realized_surface_table`は追加しない。
- 現行`point_wall_laws_v4`はparameterなしの完全鏡面`specular`と係数付き`restitution`を分離し、
  `probabilistic_stick`へ反射fallbackの明示を要求する。定数確率以外のmaterial tableは未実装である。
  `gravity_buoyancy_standard_v1`は全`axisymmetric_rz` domainで`g_r=0`を要求し、Cartesian XYの第1成分は非零を許す。
- fieldとgeometryのBVH分離、required fieldの共通layout、P1/Q1 field meshとgeometry meshの一致制約は維持する。
  異なるproducer meshは当面外部case builderで一つのcanonical layoutへ変換し、coreへmulti-mesh samplerを
  追加しない。
- 一般曲線pathの厳密corner/tangentは、budget内で一意化できなければfail-closedのままとする。実データで頻発する
  ことが測定されるまで複数facet曲線locatorを一般化しない。

実装順は、P14-Pで直列runtimeへの収束を完了した後、P14-Uで代表用途の数値・性能baselineを閉じ、T03と
P14-Rのlocal gate/evidenceを完了した。P15は旧remote-workflow blockerを解除し、RK4-firstとexplicit midpointまで完了した。
その後M3-V target applicability/relevance評価も完了し、冒頭に記録した三workstreamへ分離した。P15の物理式revision、必要field、単位、
OML branch、Debye長等の適用域、平衡bracket、rate/derivative boundはproduction code着手前に
`solver/docs/physics_models.md`へ固定済みである。T01/T02はcoreから独立した外部trackだが、
実producerからcanonical caseを作り外部V&Vまで流す製品workflowを主張する前には必要である。

## P14-P完了レビュー（serial runtime convergence）

P14-Pはouter poolをflat SoA wavefront、preallocated workspace、stackless BVH、bounded stagingへ置換した後、
内部parallelの製品価値を実測した。regular 1Mは1/2/4 threadで9.32/10.09/10.10 sだった。field locator
microkernelは約3.75xにscaleしたが、end-to-endはproposal/enclosureのPython/NumPy調停が支配し、enclosureだけを
理想並列化してもAmdahl上約1.47xだった。巨大融合kernelや第二schedulerは責務分離を損なうため採用しない。

engine v27 / compiled tile v6 / runtime layout v5は`resources.threads`、thread mask、parallel-only test/harnessを
削除し、compiled single-thread engineを唯一のproduction経路とした。bounded slab、workspace、event wavefront、
BVH、compiled boundary/RNG、direct replayは直列でも有用なので維持する。case sweepのprocess並列はsolver外に置く。
測定値、削除対象、再検討条件は[`solver/docs/parallel_execution_plan.md`](solver/docs/parallel_execution_plan.md)を
権威とする。

## P14完了レビュー（engine v20 / indexed containment / memory plan v6）

P14はarchitecture変更を性能instrumentationへ拡散させず、三公開APIをfresh childで測る外部harnessと、profileで
支配的だった二つのlocator ownerだけを変更した。23行×3観測の直交型matrixは10k/100k/1M、regular/P1/Q1、initial/
cross-cell、0/1/5/20 hit、none/sample/all、cold/warm、1/20 workerを覆い、全行でresult shape、revision、
memory-plan fit、要求JIT環境、科学payload identityをhard checkした。絶対秒数はmachine-local evidenceであり、
CI threshold、純writer帯域、COMSOLに対する速度優位を意味しない。

代表cProfileでは、修正前の1,152 cell×1,000 table startが`_validate_table_starts` 7.309 s、その内
`geometry._inside_any_cell` 7.224 sで、field prepareは0.073 sだった。hot loopへtimerを追加せず、このowner帰属を
根拠にgeometry locatorだけを変更した。

field v3はaccepted strict-interior hintを最速経路のまま保ち、hintなし/missのsupported containmentだけをfield所有の
stackless cell BVHで絞る。exact predicateと最小supported IDがauthorityである。containing supported cellがない時は
outside/masked provisionalの物理最近傍とtieを守るため従来full scanへ戻るので、この経路はO(cell数)である。
geometry v4はtable start validationのmixed tri/quad volume包含へgeometry所有のBVHを追加し、boundary-first判定と
exact half-space predicateを維持する。局所形状がfloat64で解像不能なcellだけをprepareで拒否し、大offsetだけでは
拒否しない。scalar/compiled境界parityのためCPython `math.hypot`でedge長をprepareする。

P14 closeout revisionはengine v20、compiled tile v4、runtime layout v3、memory plan v6、geometry v4、field v3、result v3、
proposal v4、event v9、physics catalog/runtime v3である。case/result schemaと粒子stateは増やしていない。memory plan v6は
両index residentに加えfield 256 B/cell、geometry 1,024 B/cellのbuild transientを計上する。336件のverification/
scenarioと標準品質gateが合格した。

3観測medianでregular 100k/1Mは20 workerで1.8796x/4.7965xだったが、P1/Q1 10k crossは
1.1737x/1.1473x、event 10k×20 hitは0.8882xだった。これはP14履歴計測である。
`threads: 1`はP14当時の推奨であり、そのthread選択自体をP14-Pで廃止した。P14でsynthetic
solver-core performance baseline、P14-Pでparallel runtime、P14-Uで代表用途、T03で最小解析・可視化を完了した。
P14-Rのlocal gate/evidenceも完了し、製品v0.1には初回remote workflow成功だけが残る。P15 continuous chargeと
その出口の外部M3-V relevance/applicability評価は完了した。T04 cache/remeshは後続profile条件付きのままとする。

## P13完了レビュー（engine v19 / durable result v3）

P13はsolver coreへ第二の実行経路や永続化frameworkを追加せず、既存`engine.py`と`output.py`の境界をdurableに
した。engineはaccepted macro barrierを固定64 stepごと、または最終macro後に作る。各worker waveの
boundary/failure payloadはtile順merge後すぐwriterへ渡すため、macro全体のevent listを保持しない。
`output.py`だけが容量1のcommand queueと単一HDF5 threadを所有し、呼出側は各commandのackを待つ。したがって
diskが遅ければcomputeへbackpressureし、eventをdropしたりworkerがfileへ直接書いたりしない。

durable commitは`closed segmentのreplace → inactive A/B checkpointのreplace → LATESTのreplace`で、`LATEST`だけを
epoch commit pointとする。checkpointは全mutable particle state、active index、release/frame/probe cursor、logical/
physical event ordinal、exact origin、surface-contact token、P1/Q1 cell hint、event aggregateを保持する。
`simulate(case, OUT)`は同一input hash、schema、physics/algorithm/backend revision、座標・method、particle identityが
一致する`OUT.partial/LATEST`を自動resumeする。不一致、破損、未知versionはfail-closedで、migrationやfallbackを
作らない。`open_result(OUT, recovery=True)`は最新segment hashと全segmentの構造・累積countを検証した
`LATEST`までのprefixだけを結合し、未完了viewの`read_final()`を拒否する。通常の完成resultはcheckpointに
依存せずlogical segmentとfinalを読む。

P13完了時点のrevisionは`deterministic_particle_engine_v19`、`durable_segmented_result_v3`、checkpoint schema 1、
`solver_owned_memory_plan_v4`である。compiled tile v3、runtime layout v3、geometry v3、proposal v4、event v9、physics
catalog/runtime、case schema 1、result schema 1は変更していない。memory plan v4はworker-wave stagingを
`worker_output_staging`へ分離し、writer reserve/output bufferとの二重計上を避ける。130 macroの三segment scenario、
最初の`LATEST`以前の初期再実行、segment/checkpoint/`LATEST`とfinal/run.json/_SUCCESS/directory publication境界の
post-replace failure injection、確率wall RNG ordinal、orphan無視、破損拒否、通常実行との全公開payload/科学manifest
identityを含むverification/scenario 322件が合格した。P13を完了とし、P14へ全規模性能matrixを送った。容量1 queueは
各commandの同期ackを待つためI/O overlapを主張しない。
また、過去segmentは構造と累積countを検証するが、hash対象は参照checkpointと最新segmentであり、同shapeの過去値改変を
検出する契約ではない。power loss、remote filesystem、同じOUTへの複数process同時実行は保証外である。

## P12完了レビュー（engine v18 / event-heavy parallel）

P12はarchitectureを増やさず、P09～P11の同じresident SoA、`StepProposal`、event loopをparallel ownershipへ
拡張した。一粒子は一workerだけが更新し、field/geometryはread-only共有、residual/event/failure/statisticsは
worker-localである。engineは最大`W`個の非重複tileをin-flightにし、main threadがfutureをtile順に回収して
stable mergeする。writerを呼ぶのもmain threadだけであり、汎用scheduler、共有atomic append、worker direct I/O、
第二engineは追加していない。threads 1/2/4でfinal/event/RNG/statisticsを含む科学出力はbitwise一致する。

event-heavy hot pathは`line_boundary_bvh_v3`のcompiled BVH query、保守的なRK4 clear/split事前認証、
同時刻wall prefixのcompiled batchへ置換した。P12完了時点のrevisionは`deterministic_particle_engine_v18`、
`compiled_cpu_tile_v3`、CPU runtime layout v3、memory plan v3である。proposal v4、event v9、physics runtime v3、
case/result schemaは不変であり、数値意味論の変更ではない。memory planはresolved worker数と
workers×proposal/geometry-query scratchを所有し、memory不足をthread数の暗黙削減で隠さない。

同一machineの512粒子×4 macro step、warm、3回medianでは、編集前serial baseline 3.066786 sに対し、
P12当時のsimulateはthreads 1/2/4で0.440460/0.528229/0.632361 s、end-to-endは
0.452938/0.542817/0.650943 sだった。1 threadはbaseline比6.96倍だが、T1/T2=0.83384、T1/T4=0.69653で
正のthread scalingはない。workはaccepted piece/candidate query/refinement/max depthが
8,704/19,968/11,264/21で、全9観測のpayload digestは一致した。したがってP12はparallel ownershipとstable mergeを
完了とし、製品規模性能と残るPython/GIL bottleneckの判断はP14へ送った。後続P13はdurable result、
checkpoint/resume、bounded writer queueを`output.py`へ追加し、P12のworker ownershipを変更せず完了した。

## P11完了レビュー（engine v17 / exponential midpoint v1）

P11はarchitecture変更を最小限に保って完了した。第二engineや汎用ODE frameworkを作らず、既存の
`StepProposal`をmethod-neutralなproposal v4へ更新した。`physics/runtime.py`と`physics/compiled.py`は一回の
model passから合成加速度と線形dragの`rate/target/additive`分解を同時に返すため、Epstein/Stokes式を
integratorへ複製していない。`integrators.py`はstart half-step predictor、midpoint-frozen解析更新、
`state_at()`、指数path enclosureを所有し、`engine.py`は従来のtime work、material/RZ event、accepted state、
writerを調停する。module ownershipと依存方向の変更は不要である。

P11完了時点のrevisionは`deterministic_particle_engine_v17`、`compiled_cpu_tile_v2`、
`coupled_fixed_step_proposal_v4`、`line_quadratic_rk4_axis_first_hit_v9`、
`deterministic_compiled_physics_runtime_v3`、`exponential_midpoint_v1`、
`exponential_midpoint_global_abs_enclosure_v1`である。CPU runtime layout/memory plan v2、case/result schema、
physics catalog/model、field semanticsは維持する。method追加を結果datasetやproducer契約の増殖へ波及させていない。

数値面ではC03一定係数を丸め誤差精度、極小/極大`h/tau`を有限、smooth可変係数を次数1.8以上で受け入れる。
RK4の`maximum_dt_over_tau < 2.5`は陽RK4固有のgateとして残し、指数法へ適用しない。現行指数法はfixed chargeだけを
対象とし、非零charge rateを黙って積分、後付けsubcycle、RK4 fallbackしない。material first hit、反射後残時間、
RZ axis→wall、output scheduleは同じproposal/event loopを通す。曲線/chord偏差はposition box全幅でなく、
全短縮secantを含むvelocity enclosureの幅からmethod-neutralに作る。材料反射とsurface departure後の同面再衝突を
過度なrefinementなしで認証し、Stokes--Cunningham一定primitiveも閉形式解へ一致する。COMSOL比較は引き続きcore外である。

抜本的なarchitecture変更は不要と判断した。P11完了時点で残る主要な性能支配候補はP10の計測で見えたevent-heavy
orchestrationであり、P12のbounded residual workとstable mergeへ送った。P13 durable result、P14全規模性能、
continuous charge、boundaryless P1/Q1、richer distribution、moving wallをP11へ取り込まない。

## P10完了レビュー（engine v16 / compiled CPU tile v1）

P10は第二engineを追加せず、P09のresident SoA/microtileを使う同じproduction loopのarray passを
Numbaへ移した。`cpu.py`がfield location/interpolation kernel、`physics/compiled.py`がsample済みprimitiveの
model加算、`integrators.py`がclassical RK4算術を所有する。`engine.py`は従来どおりtime work、event、accepted
state、writerを調停する。P10完了時点のrevisionは`coupled_rk4_engine_v16`、`compiled_cpu_tile_v1`、
`resident_soa_microtile_v2`、`solver_owned_memory_plan_v2`、
`deterministic_compiled_physics_runtime_v2`である。case/result schema、proposal、event、field semantics、
physics model revisionは変更していない。

regular fieldのsupported containing-cell common pathは軸indexからO(1)個の候補を評価する。outside/maskedの
有限provisionalは従来の最近傍意味論を守るためcompiled scanを使う。P1/Q1はprevious-cell hintのstrict interior
だけをfast pathとし、共有面、hint miss、初回sampleはNumba内full searchで最小supported ownerを選ぶ。
adjacency walk/BVHは、複雑化に見合う支配率がprofileで確認されるまで延期した。hintは非物理stateであり、
accepted full endpoint、wall/axis prefix、residual pieceのaccepted endpointだけをresidentへcommitする。
棄却trial、`state_at()`、出力sampleはhintを変更しない。

scalar evaluatorはverification oracleに限定し、compiled failure時のproduction fallbackにしない。
`state_at()`、wall hit時刻までの再積分、hit後residual、trajectory/frame/probe/finalは同じproposal/event経路を
使う。Numba 0.67とNumPy `<2.6`をlockし、`fastmath=False, parallel=False`を明示した。parallel ownershipは
P10へ混ぜず、P12へ分離した。

同一machine、trajectoryなし、1 warm-up後3回medianの非gating profileでは、regular harmonic
2048粒子×4 RK4が1.99 sから0.03058 s（65.1x）、Epstein C02 2048粒子×5 RK4が3.74 sから
0.03339 s（112.0x）、event-heavy regular wall 512粒子×4 macro steps（material-hit intervalあり）が
12.8 sから3.14395 s（4.07x）だった。
event-heavy側はaccepted pieces 8,704、candidate queries 20,480、refinements 11,776、最大深さ21で、P12の
event orchestrationが次の性能bottleneck候補だった。P11を先に完了し、P12でparallel ownershipを追加した。P10のmanual harnessは
cold/warm digest、RSS、revision、speedupを記録するが、絶対thresholdやP14の全matrix判定を持たない。

read-only synthetic local profileではP1 stripの1000 sampleをwarm実行し、hintなしfull searchは
100/500/1000/5000 cellで0.026/0.131/0.249/1.271 s、正しいstrict-interior hintは約0.0005 s
（51x～2576x）だった。初回stageは全粒子がhintなしなのでfallbackを稀とは仮定できず、large-mesh P1/Q1性能は
P10時点で未証明である。P14のunstructured matrixはrealistic cell count、initial localization、cross-cell motionを
含め、必要性が実測された場合だけadjacency/BVHを追加する。

## P09完了レビュー（engine v15 / runtime layout v1 / memory plan v1）

P09は全体architectureの変更を必要とせず、元の所有境界を実装で固定した。`case.py`/
`case_format.py`はYAMLとresource上限を先にparseし、HDF5 metadataからcanonical numeric footprintを求めて
payload展開前に拒否する。`sources.py`はrealized sourceを最終scheduleへ直接scatterし、全source配列の
追加copyを減らした。`cpu.py`は既存のN×2のposition/velocity等のresident state列とは別に、
容量Nのresident-row active index、in-place stable compact、上限付き
microtile、solver-owned memory planだけを所有し、物理modelや出力形式を所有しない。`engine.py`はそのplanを
一つのproduction loopへ適用し、tileごとのevent/failure/proposalを物理keyでstable mergeする。

memory planはload/prepare/runのphase peakとcomponent内訳、自動選択したmicrotile幅を`run.json`へ残す。
`resources.memory_limit_mb`はsolverが所有する配列の予測peakに対する上限であり、Python、HDF5、native
library、allocatorを含むOS hard RSS capではない。後者は非常駐の`p09_memory.py`がfresh/warmの
load/prepare/run別に測り、production diagnosticsやruntime dependencyは増やさない。ResultWriterのHDF5 raw
chunk cacheは有界にし、reserveを同じplanに含めた。

数値意味は変えていない。particle identityはresident rowに固定し、active indexだけをstable compactする。
microtile幅を変えてもfinal/event/RNG/outputは一致する。当初P09へ置いたper-layout cell hintは、現行の
regular locatorが消費せず未使用のO(N)配列になるため、compiled samplerを所有するP10へ
移した。これは能力の削除ではなく、消費者とstateを同じwork packageで実装する複雑化防止である。
P09 closeout時点の次gateはP10だった。現行ではP10～P14のsynthetic baseline、P14-Pの直列収束、P14-U、T03、
P14-Rのlocal gate/evidenceを完了し、remote workflowを独立trackへ残した。P15 continuous chargeはその後完了した。P12の
parallel scheduler、P13のdurable output、P14のsynthetic性能判定をP09へ遡及させない。

## P06-RZ実装後レビュー（engine v12 / event v8）

P06-RZの完了時点で、architectureの抜本変更は不要と判断する。RZ対応を第二engine、座標別physics式、axisを表す
疑似facetとして追加せず、既存の責務へ収めている。`physics/catalog.py`はXY/RZのvector metadata、
`fields.py`はaxis accessibilityとaxis node regularity、`coordinates.py`はsigned/canonical基底とsupport区間像、`integrators.py`は既存RK4と
enclosure、`events.py`はmaterial wallと独立したaxis局在、`engine.py`はfirst-event arbitrationとaccepted-state commitを
所有する。依存方向、三公開API、単一production loop、case/result schema v1は変わらない。

数値上は、resident stateを`r >= 0`へ保ったままtrial stageだけをsigned radial chartで進めるため、軸でODEをclampして
4次精度を壊す問題を避けている。field/physicsは各stageのcanonical位置・速度で評価し、radial加速度だけをsigned chartへ
戻す。global-abs enclosureのradial区間は`abs`の正しい区間像へ変換してsupportを検査し、Epstein applicabilityのnorm/
絶対boundは符号変換で不変である。axis hitは既存event budget、chord deviation、速度enclosureでclear/split/hitを
fail-closedに分類し、同じRK4でprefixを再積分してからfoldする。axis accessibilityはgeometryのaxis接触、または
boundaryless fully-supported regular boxの`r_min=0`から`fields.py`が一度だけ決め、vector regularityとradial gravityが
共有する。axis nodal radial値を厳密0と証明してもskew P1/Q1補間の加算丸めでaxis sampleが微小非零になり得るため、
`fields.sample`はaxis-accessibleかつcanonical `r==0`の証明済みRZ vector radial成分だけをexact `+0.0`へ復元する。
これは近軸tolerance clampではなくregularity preservationである。斜めfacetのAABBだけがaxis tubeへ重なる場合は、
`events.py`がtubeとfacet支持線の法線方向intervalを既存event budget込みで比較し、厳密分離を証明した候補だけを
除外する。wallとaxisの双方が局在済みで認証時間が重なる場合だけwallを優先し、axis端点とmaterial cornerの完全tieは
一般RK4 cornerとしてfail-closedにする。

品質面では、解析Epstein axis crossingの位置・速度4次収束、frame schedule identity、C02～C05のaway-axis
Cartesian退化一致、axis→wall順序、軸上不変state、regular/P1/Q1のaxis regularity、boundaryless regular support
由来のaxis accessibility、radial gravity failureを検証した。標準品質gateと
verification/scenario 230件は合格している。新resident配列、汎用contract/diagnostic framework、schema selectorは
増やしておらず、stage batchの一時変換と既存refinement workだけが追加memoryである。

P06-RZ完了時点で残った制約はboundaryless P1/Q1の連続包含、continuous charge、Stokes–Cunningham、一般RK4 surfaceのtangent/corner、
moving wallであり、P06-RZの局所fallbackで吸収しない。P06-Sはdrag責務を小さいphysics runtimeへ集約して
Stokes–CunninghamのXY/RZ schema・適用域・oracleを追加し、P08はparticle-local failure、series/probe、薄いCLIで
Stage 1Aをcloseした。その後P09のmemory/runtime layoutをengine v15、P10のcompiled CPUをengine v16、
P11のnative exponential midpointをengine v17、P12のevent-heavy parallelをengine v18、P13のdurable resultを
engine v19、P14のlocator/synthetic performance gateをengine v20で完了した。P14-Pはengine v27の
single-thread compiled runtimeへの収束として完了し、その後serial target-use gateのP14-UとT03も完了した。
P14-Rの初回remote workflowは独立trackに残り、P15はその後完了した。

## P06 revision 3b 着手前の全体checkpoint

### 結論

clean-room方針、三公開API、一つのproduction engine、canonical入力、field/physics/integrator/event/outputの
依存方向は維持できる。P06 revision 3aも、boundaryless regular subsetに対するglobal enclosureとして数値的に
妥当であり、repo全体の再設計や実装の破棄は不要である。

一方、revision 3bを「同じmacro proposalのparameter区間を細分してtubeを縮める」と解釈する案は採用しない。
現行`state_at(s)`は同一始点から短縮RK4を再実行するため、急峻な位置依存fieldではendpoint曲線のparameter方向
変化が物理速度boundを大きく超え得る。global加速度boundだけでは任意部分区間のtubeが区間幅とともに縮むことを
保証できない。これは実装前に修正すべき唯一のblockingな設計変更である。

revision 3bのaccepted numerical pathは、geometry/support/applicabilityだけで決まるsequential dyadic RK4 piece列と
する。各piece始点でrevision 3aのenclosureを再構築し、no-hit leafのendpointを次pieceへ引き継ぐ。出力時刻は
分割を変えず、確定済みpieceからframeを評価する。path tubeが包む対象はversioned integratorの離散pathであり、
真のODE解との離散化誤差は解析解と`h / h/2 / h/4`収束で別に評価する。

材料boundaryではstage内のsupport/applicability逸脱を即時例外にせずprovisional flagとして保持し、非有限値だけを
即時失敗させる。pieceごとに`first event / valid prefix → no-hitまたは残存prefixのvalidity → commit`を守る。
先行wall hitが後続のwall外trialを救済できるためである。budget内で決定できない場合はno-hitへ丸めず明示failureと
する。surface sourceのzero-time departureはP07のone-sided規則まで解禁せず、position nudgeは使わない。

### 近接する構造上の調整

- 二つ目のdrag modelより先に、sample済みprimitiveのforce評価、extrema bound、applicability、一定加速度
  certificateを小さなphysics runtimeへ集約する。field locatorとevent調停はengineに残し、framework化しない。
- revision 3b前のhardeningとして、実軌道上で非一様な`E_x ∝ -x`調和振動子の公開API収束試験と、
  放物線極値の外向きinterval化を一件ずつ追加した。大量の契約testは増やしていない。
- global boundの保守性は、候補率、refinement深さ、処理時間で測る。local bound、dense output、Numba/GPUは
  この測定が必要性を示すまで導入しない。
- P07前にseed、RNG revision、event設定、resolved boundary law/priorityをmanifestへ追加し、table/surface sourceと
  terminal/general boundaryの並行APIを残さず置換する。
- P09はYAML/resourcesを先に読み、HDF5 metadata gateとload/prepare/run memory planを完了した。
  P10 parityはendpointだけでなく`state_at`、wall hit、residual piece、output schedule不変性まで対象にする。

従って、方針は「全面刷新」ではなく「revision 3bの数値意味を実装前に修正し、責務移動を機能追加の直前に行う」
である。設計checkpoint自体ではproduction codeを変更せず、その直後のhardeningで放物線support区間の所有だけを
engineからintegratorへ移した。accepted pathやproduction loopは増やしていない。

## P06 post-reviewで発見した安全性課題（engine revision v3）

P06の最初の縦切りは、境界なしXYに限定してrequired fieldの全domain certificate、fixed charge、
Epstein/electric/gravity、classical RK4を同じproduction engineへ接続した。`physics/`はsample済みprimitiveだけを
受け、field探索は`fields.py`、stage調停とaccepted-state commitは`engine.py`、`(x,v,Z)`の積分と
`StepProposal.state_at()`は`integrators.py`が所有する。ballistic専用入口は削除し、無力場だけ同じproposalの
`linear_exact`内部specializationとしてrelease原点のbitwiseな厳密性を維持した。

区切り評価ではC02/C03の解析解と4次収束、C04/C05の一定加速度、required-field metadata/coverage、Epstein適用域、
`dt/tau` gate、位置依存加速度のdirect RK4 stage再評価をkernel/physics levelで検証した。しかしpost-reviewで、
開始・RK stage・終点という有限個のsupport判定だけでは、その間の連続曲線がsupport外へ出て戻らないことを
証明できないと判明した。さらにframe評価は同じ未証明pathを別時刻でsampleするため、frameの有無によってrunの
成功可否が変わり得る。これは材料boundaryだけの問題ではなく、boundaryなしの一般`rk4_reintegrated`にも残る。

このため前版`coupled_rk4_engine_v3`は、dragまたは非一様場を使う一般`rk4_reintegrated` production runを
一時的にfail-closedへ閉じた。C04/C05は以下の証明済み一定加速度経路で公開API scenarioとした。revision 3aの
engine v4は、この安全性判断を撤回せず、連続enclosureを証明できるboundaryless regular subsetだけを再び解禁し、
C02/C03を公開API scenarioへ移した。第二engineやfallbackは追加していない。RZ forceとStokes–Cunninghamは
各入力契約とoracleが揃うまで拒否する。

prepare safety gate追加前のNumPy kernel characterizationは正しさとbottleneck調査用であり、現行productionの
利用可能範囲や性能完成を意味しない。accepted endpoint検査を含むローカルwarm smokeでは
10,000粒子×4 stepの一様電場caseが約0.84万particle-step/s、無力場が約40万particle-step/sであり、Python point-locationが明確な
bottleneckである。数値意味を変えずにfield/physics/RK4をtile kernelへ置換する責務はP10に残す。

revision 2の`quadratic_exact`は、fixed charge・dragなし・canonical DOFが厳密一様なfieldから粒子別の一定加速度を
証明できるcaseだけを扱う。これは端点chordの近似ではなく、現RK4と一致するpathをparabola-lineの二次根で局在する
小さい縦切りである。`fields.py`は厳密一様性、`integrators.py`は解析的な位置・速度とdense evaluation、
`events.py`はchord偏差を含むBVH候補抽出と放物線first hit、`engine.py`は連続supportの証明と
event→no-hit行validity→commitの順序を所有する。topology-completeな材料boundaryがあるcaseではfirst hitが
流体domainからの退出を捕捉する。boundaryなしcaseでは全cell supportedな`RegularLayout`だけを許可し、各proposalで
各座標の端点と内部極値を解析的に求め、矩形support box内にあることを確認する。第二engine、汎用curve class、
診断subsystemは追加していない。

区切り評価では、endpoint chordが壁を見落とすturning path、paddingだけが候補を拾うno-hit、near-grazing
hit/miss、接線・共線の不定failureを独立に検証した。公開APIでは異なるmacro stepでhit時刻・位置・衝突直前速度・
stick終端が一致し、出力scheduleがevent/finalを変えず、先行hitがwall外trial endpointを安全に救済することを
確認した。一様場certificateを補間後加速度とのbit一致で二重検査しない。数学的に一定な補間でも加重和の丸めが
最下位bitを変え得るため、canonical DOFをauthorityとし、実stage samplingはsupport/applicabilityだけに使う。
boundaryなしでは放物線頂点だけがsupport外へ出る反例を、frame有無の双方で同じfailureになるよう回帰化した。

## P06 revision 3a 実装後レビュー

revision 3を一つの一般曲線frameworkとして実装すると、support証明、材料event、時間分割、unstructured containmentを
同時にengineへ持ち込み、revision 2で確立した責務とevent順序を壊す。このため最初の能力をrevision 3aとして、
boundaryless Cartesian XY、fixed charge、既存のEpstein/electric/gravity、全cell supportedな
`RegularLayout`へ限定した。実装はこの範囲を越えず、単一engineを維持している。

revision 3aの証明対象はfull macro endpointだけではない。`StepProposal.state_at()`は出力時刻ごとにproposal始点から
短縮RK4を再評価するため、global field extremaとmodel係数から、任意の短縮時間における内部stage位置・速度、
accepted endpointをすべて含む外向き丸めenclosureを実装した。Epsteinはfield supportと別に、
`lambda/a >= 10`と`|u-v|/c_bar <= 0.1`を同じ速度boundで全区間証明する。stage点だけのapplicability判定を残すと、
frame追加が新たなfailureを発生させるため、post-reviewで閉じた問題を再導入する。

責務は既存module内に留めた。`fields.py`はcanonical extremaとregular support、`physics/`はfield locatorを知らない
純粋なmodel bound、`integrators.py`はRK4 path enclosure、`engine.py`は事前証明と単一loopを所有する。
汎用certificate module、DI、第二engine、全stage診断は追加していない。証明不能なproposalはhidden
subdivisionで別の数値pathへ変えずfail-closedにする。C02/C03の公開API時系列、frameなし・疎・密scheduleでのfinal
identity、step途中releaseの粒子別残時間、安全な非一様regular field、短縮pathだけがsupportを逸脱する反例、
Epstein適用域反例をscenarioで確認した。

`rk4_reintegrated`と材料boundaryの組合せはrevision 3bへ残す。材料boundaryでは先行hitが後続trialの
support/applicability逸脱を
救済し得るため、revision 3aのwhole-proposal事前拒否をそのまま適用してはならない。piecewise proposal、
保守的path tube、時間分割と一緒に`event → no-hit/remaining行validity → commit`を維持する。
unstructured support、RZ force coupling、continuous charge、Stokes–Cunningham、反射後の残時間も各独立gateを
維持する。着手前hardening後は`coupled_rk4_engine_v5`、`coupled_rk4_proposal_v3`、
`rk4_global_abs_enclosure_v1`をresult manifestへ
記録し、既存result schemaは変更していない。対象verification/scenarioの合格をもってrevision 3aを完了とする。

## P05実装後の追補

P05は計画した責務を増やさず、大域topology監査とline BVHを`geometry.py`、数値budgetとballistic
first-hitを`events.py`、terminal lawを`boundaries.py`、状態遷移を唯一の`engine.py`、永続eventを
`output.py`へ配置して完了した。boundaryなしを別engineへ分岐せず同じqueryのno-hit caseとし、COMSOL、
`model_dataset`、旧solverへの依存は導入していない。

区切り評価では、BVH leaf内のfacet AABB再判定、boundary vertex次数と自己交差監査を追加した。最大facetの
budgetはbroad phaseだけに限定し、初期点距離・near-parallel separation・交差受理はfacet固有budgetへ修正した。
区間終端後の交点はposition/time両budgetを満たさない限り端点eventへ丸めない。これにより「交差なし」と
「浮動小数上は交差を証明できない」を分離している。event始点・endpoint・途中frameも`StepProposal`へ一本化し、
P05時点でcanonical DataBundleを含むresident-memory下限をresource gateへ加えた。この下限はP09の
phase memory planが置き換えた。C07、無効topology、multiscale facet、
escapeの右連続frameとlogical-null final、macro `dt_s`／output schedule独立性、RZ axis seamを検証した。

P05のscalar event loopは正しさの基準であり、性能完成を意味しない。複数facet応答、reflection、surface
release、RZ axis path分割はP07、compiled/batch化はP10が所有する。次のP06ではfixed charge・drag・electric・
gravityをRK4の実stageで評価し、required fieldの全domain supportと曲線path event収束を同じproduction
engine上で確立する。別physics engineや汎用plugin frameworkは追加しない。

## P04実装後の追補

P04はtable source、解析的ballistic proposal、単一production loop、同期一segment writer、最小lazy
`ResultView`という当初の責務境界内で完了した。schemaのdata座標とmotion modeを分離し、final/releaseを
必須化したため、旧key、暫定stub、第二engineは残っていない。output scheduleとmacro `dt_s`を変えたscenarioで
共通frame、event、finalの同一性を確認し、途中位置をmacro-step累積値から再出発させる1 ULP依存はrelease原点を
保持するproposalへ是正した。

P04で意図的に受理しなかった範囲は、力、canonical/configured wall、surface source、RZ軸到達・横断、
checkpoint、recovery、並列kernelである。productionのRZ axis path分割はP07が所有する。後続P05は既存の
直線proposalを変更せず、大域topology、line BVH、厳密first hit、stick/escapeだけを追加して完了した。

## P00～P03実装後レビュー

### 結論

clean-room分離、三公開API、単一engine、`DataBundle / SimulationSpec / PreparedRun`、
geometry・field・event・boundary lawの所有分離は妥当であり、architectureの作り直しは不要である。
一方、静的品質gateと既存58 testが通るだけではP03を数値的に完了とは判定できないことが分かった。
次のfeatureを積む前に必要だったP03の一度限りの補正は`field_location_v2`として完了した。P04の出力意味論は
小さく固定して実装する。

### 再現したblocking defect

旧`field_location_v1`はP1/Q1の包含判定に参照座標の固定64 ULPを使っていた。例えばP1節点
`(1,1), (1.001,1), (1.001,1.000001)`の節点0–2の厳密な中点は、barycentric weightが
`(0.5, -1.1102230246252788e-10, 0.5000000001110223)`となりsupport外へ誤分類される。同じ要素を
原点付近へ平行移動するとinsideになるため、判定が物理座標の平行移動に不変でない。Q1境界点でも
同型の誤分類を確認した。さらに、有限な入力同士でもoutside provisional補間がoverflowし得る。

このdefectは物理polygon包含、conditioning認証、最近傍supported射影を持つ`field_location_v2`で置換済みである。
補正版は物理空間の後退誤差、局所element scale、座標ULP、`||J^-1||`、再構成残差を一つの
conditioning-aware判定へまとめ、無制限にinsideを広げない。極端に悪条件なcellと非有限補間は
明示errorにし、large-offset/high-aspect P1・Q1、片側support共有面を回帰caseにする。科学的意味が
変わるためalgorithm revisionも更新する。

### 採用した計画補正

1. P03補正は完了した。別samplerやcompatibility shimは残さずP04へ進む。
2. P04はfield/geometry/physicsなしのtable-source ballisticだけとし、SoA、必須final、release event、
   `selection: all + explicit_times_s` frame、最小lazy `ResultView`を一つの縦切りで作る。
3. P04時点では一つのsegmentだけを書き、checkpoint、`LATEST`、background writer、recoveryを先行作成しない。
4. P05はballistic直線の厳密first-hitと大域topology検査だけを所有する。曲線pathの包含budgetはP06で
   physicsと一緒に検証し、未証明のstep-doubling差を保守boundと呼ばない。
5. `case_format.read`は局所schema/node-order/owner整合、`geometry.prepare`は外周完全性・一意性・
   non-manifold拒否を所有する。geometry repairは外部builderのままとする。
6. field連成前に、required fieldのparticle domain全域support、node-associated continuous field、
   unit/component/basis、masked-only DOFの有限正規化とprovenanceを一つのgateで確定する。
7. HDF5のdata座標表現とYAMLのparticle motion modeをP04前に分離し、engineが対応表を一度だけ検査する。
   これはStage 2AでRZ dataとXYZ particle stateを組み合わせてもengineを分岐させないためである。
8. cacheは利用者が明示選択した検証済みlayoutだけを使う。runtimeの暗黙hybrid fallbackは実測上必要に
   なるまで延期する。

### 後続stageへ置く判断

- P06：RZ basis parity、曲線trajectoryのwall event収束、field-driven modelの適用域。
- P07：production RZ axis path分割とsurface release、wall law。
- P07：interaction cap到達後、次hitが存在する時だけresidual intervalを分割するC09規則。
- P10：実際に消費するP1/Q1 strict-interior hint、Numba内full-search fallback、compiled parity（完了）。
  load/prepare/run RSSの独立characterizationとsolver-owned memory planはP09で完了し、walk/BVHはP14のprofileへ延期。
- Stage 2A：continuous chargeのcoupling profileはP15で完了。RZ data＋XYZ motionを含むcase/result schema revisionは未着手。
- P13：checkpoint/resume、multi-segment recovery、同じ三公開API内でのresume起動意味論（後続P13で完了）。

これらを今のP04へ一般frameworkとして先行実装しない。各stageで解析解または小さいreferenceを伴う
一つのdecisionとして追加する。

### 保守性の評価

現時点のRuff、Pyrefly、import-linter、Radon、58 testはcleanで、過剰なcontract/diagnostic subsystemも
導入されていない。`case_format.py`は長いがschema read/write/hash/局所validationという一つの変更理由に
まとまっており、行数やMIだけを理由に分割しない。result outputや大域geometry auditをここへ追加しないことが
重要である。今後も、機能追加のたびにmanager/helperを増やすのではなく、上記ownerへ一つのproduction経路を
追加し、置換した設定・実装・test・文書を同じchangeで削除する。

## 0. レビュー結論

既存の主仕様が採った「clean-room開発」「COMSOLはadapter/V&Vだけ」「mesh-nativeな原始場」
「event-first境界」「CPU batch first」「三層だけのtest」という方向は妥当である。一方、そのまま
実装すると、積分器・帯電・境界探索の責務、geometryとfield mesh、物理modelの組合せ、並列event、
長時間runの出力に曖昧さが残る。これらは局所的なhelper追加では直せず、runtimeの分岐を再発させる。

本レビューでは次の五つを基盤の固定境界とする。

1. `DataBundle / SimulationSpec / PreparedRun`
2. `PhysicsPlan`
3. `StepProposal`
4. `BoundaryEvent`
5. `ResultStore`

抽象を増やすための境界ではない。外部データ、物理選択、数値step、壁応答、永続化という、互いに
異なる変更理由を混ぜないための最小境界である。公開APIは引き続き `load_case`、`simulate`、
`open_result` の三つだけとし、上記の実行計画やstep表現は非公開にする。

---

## 1. 最終的な責務分割

```text
COMSOL / CFD / plasma code / CSV / 計測 / user mesh
                         │
                         ▼
             tools/importers + case_builder
                         │
                  case_format.write
                         ▼
          Canonical DataBundle（SI、versioned）
                         │
             SimulationSpec ──────┘
                         ▼
                    load_case
                         │
                  SimulationCase
                         ▼
                   engine.prepare
                         │
                PreparedRun（非公開）
                         ▼
              single production engine
        source → integrator/physics → event → law
                         │
                         ▼
                  canonical ResultStore
                  ┌──────┴────────┐
                  ▼               ▼
               analysis       external V&V
                  │               └─ COMSOL
                  ▼
             visualization
```

### 1.1 module ownership

| module | 所有する | 所有しない |
|---|---|---|
| `api.py` | 三つの公開操作、例外の公開境界 | 数値式、形式変換 |
| `case.py` | `SimulationSpec`、正規化済み`SimulationCase` | HDF5詳細、backend |
| `case_format.py` | canonical schemaのread/write、version、hash | source固有列、物理計算 |
| `coordinates.py` | XY、RZ、RZ-field/3D、XYZの基底・軸規則 | field補間、壁則 |
| `geometry.py` | 粒子domain、境界facet、BVH、first-hit query | stick/reflection、field補間 |
| `fields.py` | `FieldLayout`、space/time補間、support | cache生成、geometry修復 |
| `sources.py` | table/surface release、weight、発生schedule | lifecycle loop |
| `rng.py` | counter key、uniform/normal変換、stream分離 | source分布、wall outcome |
| `physics/catalog.py` | model ID/revision、要件、寄与種別 | mesh探索、積分、I/O |
| `physics/forces.py` | dragと加算加速度の純粋評価 | wall、lifecycle |
| `physics/charge.py` | fixed/continuous chargeのrateと局所solver | particle motion orchestration |
| `integrators.py` | stage連成、endpoint、path表現、局所誤差 | hit探索、wall law、file出力 |
| `events.py` | 最初のevent局在、残時間work item | 反射式、出力形式 |
| `boundaries.py` | hit後のstick/escape/reflection/probability | 幾何交差、BVH |
| `engine.py` | prepare、macro barrier、lifecycle、commit調停 | source固有変換、plot |
| `cpu.py` | SoA、tile、thread、scratch、stable active IDs | 物理の意味、schema |
| `output.py` | bounded sink、commit、checkpoint、lazy `ResultView` | 科学集計、可視化 |

依存方向を固定する。

- coreは`tools/`をimportしない。
- physicsはgeometry、I/O、boundaryをimportしない。
- integratorはstage evaluatorを呼ぶが、model registryやfileを知らない。
- geometryは最初のhitを返すが、wall outcomeを決めない。
- backendは`PreparedRun`だけを受け、未対応modelをPython callbackへ暗黙fallbackしない。
- analysis/visualization/V&Vは`ResultView`だけを読み、solver private stateをimportしない。

`models.py`のような一般型置き場は作らない。型は事実を所有するmoduleへ置く。moduleが肥大化した
場合も、helper数ではなく変更理由が二つになった時だけ分割する。

---

## 2. canonical dataとrun設定を分ける

大きなmesh/fieldをparameter sweepごとに複製しないため、入力は次の二つに分ける。

```text
DataBundle
  GeometryDomain
  FieldLayout(s)
  FieldSet
  boundary groups/materials
  optional source tables
  provenance/content hashes

SimulationSpec
  source distributions
  particle properties
  physics model selections
  boundary laws
  time/integrator/backend/resources
  output/seed
```

`load_case()`は両者を読んで正規化済み`SimulationCase`を返す。`simulate()`の開始時に一度だけ
非公開`PreparedRun`へ解決する。ここでmodel revision、必要fieldの和集合、state layout、座標kernel、
境界lawのdense map、sampler、memory planを固定する。hot loopでは文字列dispatchもcapability探索も
行わない。

COMSOL以外のproducerも正式に利用できるよう、canonical writerはsolver利用者向け公開APIとは別に
`case_format.write()`として安定させる。adapterはsource固有の単位・列・mesh意味を検査した後、
このwriterだけを介してDataBundleを作る。canonical loaderはproducer固有の事情を知らない。

### 2.1 粒子属性

canonical particleでは少なくとも次を独立して持つ。

- `mass_kg`：慣性の唯一の権威
- `drag_diameter_m`：drag相関が使う径
- `electrostatic_radius_m`：電気・DEP・帯電modelが使う半径
- `displaced_volume_m3`：浮力が使う排除体積
- `model_weight`：一つの計算粒子が代表する実粒子数

球形の簡便入力として`diameter + density`を受けてもよいが、case builderが一度だけ上記へ展開し、
導出式をprovenanceへ残す。runtimeが`mass`から密度や径を逆算しない。重力・浮力加速度は

\[
\boldsymbol a_{g+b}=
\frac{m_p-\rho_gV_{disp}}{m_p}\boldsymbol g
\]

であり、`mass_kg = rho_p V`を暗黙に仮定しない。

---

## 3. PhysicsPlanとmodel使い分け

設定はlist of arbitrary forcesではなく、意味の重ならないcategory mapとする。

```yaml
physics:
  drag: {model: epstein_linear, ...}       # 0または1
  charge: {model: fixed}                    # 必ず1。Z0は各sourceが所有
  # noise category omitted when disabled   # 0または1
  electric: {model: coulomb}
  gravity_buoyancy: {model: standard}
  # thermophoresis / dielectrophoresis / ion_drag are omitted when disabled
```

- categoryが無ければ無効であり、hidden defaultを置かない。
- 同じcategoryには一つのmodelだけを選ぶ。blend/compositeはそれ自体をversioned modelとする。
- translational DOFの`linear_relaxation` ownerは原則一つ。複数dragの単純加算は許さない。
- explicit accelerationは固定順序で加算し、順序を結果manifestへ残す。
- internal stateの各sliceには一つのownerだけを持つ。
- noiseは選んだSDE integratorの局所係数を返し、曖昧な「random forceまたはincrement」にしない。

runtimeの寄与単位は次へ統一する。

| kind | 戻り値 |
|---|---|
| `linear_relaxation` | rate `[1/s]`、target velocity `[m/s]` |
| `explicit_acceleration` | `[m/s^2]` |
| `internal_rate` | internal stateごとの`state unit/s` |
| `noise_coefficients` | 選択SDE methodが定める局所係数 |

物理説明は力[N]で記述してよいが、kernel境界では質量で割った加速度へ統一する。離散電荷は連続
`internal_rate`では表せないため、実例を得た時点でjump process用step strategyを追加する。空の
汎用event-rate frameworkを先に作らない。

各model宣言は `id/revision/category/contribution kind/required quantities/parameters/applicability/
supported coordinates/reference evaluator/compiled evaluator` だけを持つ。継承階層、DI container、
entry-point pluginは不要である。

---

## 4. 一つのengineと数値step契約

engineは電荷や速度を直接更新しない。積分strategyへ状態、時刻、step幅、stage evaluatorを渡す。

```text
engine
  → integrator.propose(state, t, h, stage_evaluator)
  → StepProposal(endpoint, error-bounded path pieces, local error, support status)
  → EventLocator.first_hit(path pieces)
  → hitまで同じintegratorで再評価
  → BoundaryLaw.apply(hit)
  → remaining timeを同じ経路へ戻す
```

`rk4_fixed`、`exponential_midpoint`、B02の`ou_langevin`は別runtimeではなくstep strategyである。
scalar evaluatorはverification oracleでありproduction fallbackではない。

### 4.1 RK4

状態`(x,v,Z)`を一つのvectorとして扱い、4 stageの各時刻・位置・電荷でfield、charge rate、forceを
再評価する。固定step modeではfield knot、global source discontinuity、boundary eventで必要な区間を
切る。個別releaseはその粒子の残時間workとして正確に処理し、出力時刻はaccepted pathから評価するだけで
stepを切らない。壁局在のための分割は物理step変更と記録して区別する。

### 4.2 exponential midpoint

局所的に

\[
\dot{\boldsymbol v}=-(\boldsymbol v-\boldsymbol u)/\tau+\boldsymbol a
\]

とし、midpointで`u, tau, a`を固定する。`E=exp(-h/tau)`、`A=1-E`とすると

\[
\boldsymbol v_1=\boldsymbol u+E(\boldsymbol v_0-\boldsymbol u)
                 +\tau A\boldsymbol a
\]

\[
\boldsymbol x_1=\boldsymbol x_0+\boldsymbol u h
 +\tau A(\boldsymbol v_0-\boldsymbol u)
 +\tau\{h-\tau A\}\boldsymbol a
\]

で更新する。`A`は`-expm1(-h/tau)`で評価する。start係数で半step predictorを作り、予測midpointで
field・charge・全寄与を評価してfull stepを計算する。smoothな問題でglobal second orderを
verificationする。非線形dragを線形緩和へ黙って近似せず、対応methodを選ぶか拒否する。

continuous chargeは既存RK4をreference pathとして最初に受け入れ、model固有のrelaxation boundが証明する非stiff範囲だけを
受理する。native指数運動にはその後、同じ予測midpointを使うexplicit midpoint chargeを接続して受け入れた。scalar implicit
midpointはL安定でないためstiff solverにしない。未対応の剛性はfail-closedとし、必要性の実測後に全状態
bounded dyadic、単一のL安定法、または別versionのquasistatic-equilibrium modelから一つを選ぶ。
modelが局所的な平衡緩和の解析形を宣言する場合だけexact relaxationを使う。charge解法はmodel revisionと
integrator profileに記録し、一次operator splitを標準にしない。

### 4.3 time control

`fixed`と`bounded_dyadic`を分ける。fixed modeは精度のためのhidden step変更を行わない。
bounded dyadic modeはstep-doublingで誤差を見積もり、2の冪levelへ粒子をbucket化する。粒子ごとの
Python adaptive objectは持たない。精度budget枯渇は`failed:numerical_accuracy`またはrun failureで
あり、stick/escapeへ変換しない。

### 4.4 StepProposalとfirst hit

RK4の始終点を一本のchordとみなすだけでは曲線軌道と薄い壁を保証できない。`StepProposal`は
endpointに加え、piecewise path、各pieceのchord偏差bound、stage support、局所誤差を持つ。
偏差boundがgeometry toleranceを超える、または拡張facet候補と交差が曖昧な場合、時間stepと
物理評価を一緒に二分する。hit状態は補間だけで済ませず同じintegratorでhit時刻まで再評価する。

P05のballistic pathは厳密な直線segmentとして判定する。P06以降の曲線pathでは、local errorや
curve–chord deviationだけを数学的な包含boundとみなさない。physics係数から保守的なswept enclosureを
構成できるcaseだけno-hitを確定し、構成できないcaseは時間分割または`indeterminate_geometry`とする。
曲線manufactured path、near-grazing hit/no-hit、turning trajectoryでevent収束を検証する。

このreview時点のv0.1境界はpoint particleだった。現行は独立`contact_radius_m`をcase schema v3へ追加し、
静的2-D XY/RZのcapsule first contactとしてgeometry/eventを一度だけ更新した。point hitへtoleranceとして半径を
足す実装にはしていない。接触rolling/slidingは引き続き別機能である。

---

## 5. geometry、field、急峻分布、3D

`GeometryDomain`と`FieldLayout`は別の事実である。同じfinite-element meshを共有できるが、同一を
前提にしない。

- geometry：粒子domain、衝突edge/facet、boundary/material ID、owner、BVH。
- field layout：regular/unstructured、basis、support domain、time axis、locator。
- 同一meshの時だけvertex storage、cell ID、locatorを共有して高速化する。
- geometry remeshは外部case builder、field resampling/cacheは外部preprocessorが所有する。
- runtimeはmeshを生成・修復しない。

v0.1ではphysics設定が参照する明示layoutをrun開始時に固定する。cacheは外部preprocessorが別fieldまたは
別DataBundleとして作り、利用者が明示選択する。場所ごとにmesh-native/cacheを切り替えるhybridは、
実ケースで必要性とspeedupが確認されるまで延期し、障害時のsilent fallbackは作らない。

cacheの採用にはsupport誤分類0、壁近傍・gradient・L-infinity/p99誤差budget、軌道/event同等性、
実測speedupを必要とする。急峻場や不連続面を跨いで平滑化しない。snapshotの時間解像度不足は
solver stepを細かくしても修復できないため、外部preprocessorが間引き検証やsecond differenceで
adequacyを判定し、warn/errorをmanifestへ残す。

時間依存場の現行revisionはsnapshot間でtopologyを固定し、全snapshotをresident保持してtime knotで
macro stepを切る。各stageの実時刻で隣接2 snapshotを線形補間し、範囲外はerrorとする。hold/discontinuity、
streaming/double buffer、moving meshや時刻ごとに異なるtopologyは未実装で、独立revisionとする。

### 5.1 座標mode

- `axisymmetric_rz_meridional`：2D RZ state。`r=0`通過は負のrを持たず、Cartesian埋込みと同値な
  速度反転規則を`coordinates.py`だけが所有する。
- `axisymmetric_field_cartesian3d`：粒子stateはXYZ、fieldは`r=sqrt(x^2+y^2)`でRZからsampleする。
  3D pathを`(r(s),z(s))`へ写した曲線とRZ断面境界の交点をbracket/localizeし、RZ法線
  `(n_r,n_z)`を衝突点のthetaで`(n_r cos theta,n_r sin theta,n_z)`へ戻す。3D三角面を複製しない。
- `cartesian_xyz`：初期完全3Dは`tet4 field + tri3 boundary`へ限定する。hex8、高次要素、moving meshは
  実データと検証caseが得られてから追加する。

---

## 6. boundary lawとstochastic crossing

event recordは最低限、`particle_id/event_ordinal/event_type/time/primary_facet_id/boundary_id/material_id/
position/normal/velocity_pre/velocity_post/charge_number_pre/charge_number_post/law_id/outcome/model_weight/localization residual`
を持つ。`release / boundary / failure`を区別し、type固有列はnullableにする。
`normal`はlaw適用に使ったeffective response normalであり、combined-normal反射では選択subsetの正規化合成
法線を保存する。raw candidate facet集合は別のcandidate tableへ保持する。
同時facet hitは候補集合を保持し、吸収系は一意、反射系は明示したcorner policyで決める。位置を
epsilonだけずらして再衝突を避けず、法線速度とpath方向からdepartureを判定する。

同時hit候補は別のcolumnar candidate tableへevent-row offset/countで保存し、単一facetへ潰さない。

OU/Langevinではjoint `(x,v)` Gaussian更新を使い、step分割時は親incrementをbinary tree pathで
条件付き分割する。RNG identityは`seed, particle_id, macro_interval, root_stochastic_interval, tree_level,
tree_index, component, stream`とし、accepted-step番号やthread順に依存させない。

source samplingは`source_id/source_particle_ordinal/draw_kind`、確率wall lawは
`particle_id/physical_boundary_event_ordinal/law_stream`を別streamでkey化する。数値refinementは
physical event ordinalを進めず、thread・tile・出力scheduleでdrawを変えない。

一般曲面でのinertial OU first passageをexactと主張しない。最初のrevisionはgeometry/outputと独立な固定depthの
dyadic treeとcubic Hermite leaf pathを数値pathとして固定し、平面解析caseと一般曲面のdepth収束・first-passage
統計で品質を規定する。stochastic RMSを決定論的clear certificateには使わない。adaptive clearは全runの
miss-probability budgetを有限に配分できる別revisionだけで許す。B02はこの固定depth形式を
限定production sliceとして完了したが、adaptive clearと連続OU first-passageは引き続き非対応である。

---

## 7. CPU実行、event-heavy実行、memory

粒子の物理SoAはID対応を固定し、並べ替えるのはactive resident-row indexだけとする。単一のcompiled passが
各粒子行を一度だけ更新し、solver内にcompute schedulerを持たない。P12のouter ThreadPool、
worker-local scratch、future mergeは履歴であり、P14-P closeoutのv27では内部thread teamも含め削除済みである。linear/quadratic exactと一般曲線eventの
row-target flat SoA work、direct columnar replayもengine接続済みであり、第二production schedulerは存在しない。

hit後もactiveな粒子は、行ごとのtarget time、refinement path、interaction count、physical event ordinalを持つflat
SoA queueの次roundへ残時間を戻す。`max_interactions_per_step`へ達したら、残区間に次hitが存在する場合だけ残時間を
二分し、event-freeなら追加depthなしで受理する。splitした子intervalのinteraction countを0へ戻して再試行し、
refinement depth budgetを超えた時だけ`failed:numerical_event_budget`とする。物理的stickへ変換しない。

各roundは一行あたり高々一件のevent/failureを固定columnar slotへ書き、stable prefix/compactionで出力と次roundを
作る。共有list、thread-local可変container、Python dataclass mergeは使わない。boundary identityは
`(particle_id,event_ordinal)`、公開時のcanonical順は`(time_s,particle_id,event_ordinal)`であり、thread順や
物理行の到着順へ依存しない。整数countと固定binだけをcore集計とし、近似分位はanalysisへ置く。

通常利用者にはtile数を公開せず、case schema v2は`resources.memory_limit_mb`だけを持つ。prepare時に

```text
resident state + active/pending index + resident field snapshots
+ serial bounded tile slab + bounded event/output buffers + safety margin
<= memory limit
```

を満たすtile幅を決める。予測memoryを超えるcaseは開始前に拒否する。deterministic CPUでは
fastmathを標準無効とし、stable prefix/reduction順でslab幅、出力schedule、checkpoint/resumeから
粒子別結果を独立させる。case sweepのprocess並列はsolver外で行う。
上式はsolver-owned配列の予測peakでありOS hard RSS capではない。P12/P13のworker数倍のscratch/stagingは
P14-P closeoutのv27 serial slabへ置換・削除した。P09はphase peak/component内訳をmanifestに永続化し、入力HDF5はmetadataで上限超過を検出して
payload展開前に拒否する。

---

## 8. ResultStore、checkpoint、analysis、可視化

v0.1はJSONとHDF5の二形式だけを使う。Parquet/CSV exportは外部analysisが必要時に作る。長時間runを
一つのappend fileへ賭けないため、閉じたepoch fileをatomic commitする。

```text
run.partial/
  run.json                 # status=running、schema/model/input hash
  segments/epoch-000000.h5 # close後renameした確定segment
  checkpoints/A.h5
  checkpoints/B.h5        # 二世代交互
  LATEST                   # 最後にcommitしたepoch
  _SUCCESS                 # 正常終了時だけ作成
```

各segmentは同じschemaで`events`、選択trajectory frame、online countを持つ。完了時にfinal particle表を
HDF5へ書き、`run.json.status=complete`と`_SUCCESS`を最後にcommitし、`OUT.partial`を要求された`OUT`へ
同一volume内でrenameする。通常の`open_result`は
`_SUCCESS`のないrunを完成結果として開かないが、明示的なrecovery modeは最後の`LATEST`まで読める。

checkpointは出力scheduleから独立した固定64 accepted macro-step上のepoch barrier、または最終macro後だけに作り、state SoA、時刻、
active IDs、pending source cursor、
RNG/refinement counters、event ordinal、subcycle level、最後のcommit IDを含む。同一input hash、case
schema、physics revision、algorithm/backend revisionでのみresumeする。初期版にmigrationを作らない。

durable commitは`closed segmentのrename → inactive checkpointのreplace → LATESTのreplace`の順とし、
`LATEST`だけをepoch commit pointにする。crash後はそれより新しいorphanを無視する。最終化は
`final.h5 → complete run.json → _SUCCESS → OUT directory rename`の順で、各境界へfailure injectionを
行う。resumeはcheckpointのevent/frame countから続け、重複・欠落を許さない。

trajectory設定は選択と時刻を直交させる。

```yaml
output:
  trajectories:
    selection: sample      # none | sample | ids | all
    count: 1000
    schedule:
      interval_s: 2.5e-4   # または explicit_times_s
```

HDF5はtime/frameでchunkし、全粒子×全時刻をmemoryへ積まない。main threadの同期single-owner writerが
backpressureし、eventを捨てない。P13の即ack queueはcompute/I/Oを重ねなかったためP14-P closeoutのv27では削除済みである。
`ResultView`はsegment順にlogical datasetを結合し、frame/probeをlazyに走査する。

外部`analysis`は付着率、source-to-target matrix、重み付きdeposition、到達時刻・入射energy/angle、
residence、ensemble信頼区間を計算する。`visualization`はResultViewまたはanalysis tableだけを描画し、
物理を再計算しない。COMSOL比較plotは`tools/vv/comsol`に置く。

---

## 9. verification、性能、複雑化budget

test treeは`verification / scenarios / performance`の三層だけとする。数値的な最低合格項目は次である。

- RK4のsmooth no-event問題でglobal fourth order。
- exponential midpointでglobal second order。
- P1 affine、Q1 mapped bilinear、tet4 affine fieldの補間再現。
- step/mesh半減によるevent時刻・点・facet IDの収束。
- continuous chargeの平衡値、rate残差、RK4/native midpointの刻み収束とstiffness拒否。
- reduced electrostatic builderのclosure/Jacobian、affine Laplace、annular logarithmic Laplace収束、
  非線形mesh self-convergence、global charge balance、canonical field round-trip。
- OUの平均・共分散・MSDと平面first-passage統計。
- 0/1/5/20 hit per particleで時間・memoryが処理step/event数に概ね比例。
- slab幅、出力schedule、checkpoint/resumeでdeterministic event sequenceが一致。
- 10^4/10^5/10^6粒子で予測RSSと実測、cold/warm、compute/I/Oを分離して記録。
- 保存frame数を増やしてもpeak memoryが増加しない。
- cacheはaccuracy budgetと実測speedupの両方を満たす時だけ採用。

性能の絶対合格値は測定前に捏造しない。同一hardware、同一physics、同一出力量でbaselineを作り、
回帰とscalingを判定する。「COMSOLより高速」はこの条件が揃ったcaseだけに限定する。

禁止事項を短く保つ。

- private helper、directory tree、call順をtestで固定しない。
- production hot loopにPython callbackやsilent scalar fallbackを入れない。
- model組合せごとのkernelを先回りして増殖させない。
- v0.1でMPI/Dask、adaptive octree、一般CAD repair、高次要素、checkpoint migrationを作らない。
- 一つのinvariantをadapter、loader、runtime、writerで重複実装しない。
- 新経路を追加したreleaseでは、置換される旧経路を削除する。

---

## 10. 段階計画の修正版

| stage | 実装範囲 | 出口条件 |
|---|---|---|
| 0 | canonical format、解析解、boundary microcase、format writer | COMSOLなしで基本式とeventを検証可能 |
| 1A | static RZ、table/surface source、fixed charge、drag/electric/gravity、RK4、first-hit | scalar oracleと公開API scenarioが合格 |
| 1B | exponential midpoint、Numba CPU、bounded output、checkpoint、10^4–10^6 synthetic benchmark | 主要物理がcompiled経路を通りmemory planとsynthetic baselineが合格 |
| 1P | Numba内部thread team、flat SoA event wavefront、thread非依存scratchのP14-P評価 | 否定的closeoutでmulti-thread能力を削除し、serial runtimeへ収束済み |
| 1U | surface＋非一様場＋壁の代表use case | 時間/mesh収束、bound/event cost、serial end-to-end時間とmemoryを同一caseで説明可能 |
| 1R | P14 evidence、Windows/Linux配布、T03 analysis/visualization | local gate/evidence完了。初回remote CIは独立release trackに残る |
| 2A | P15 continuous charge完了 → M3-V relevance gate → workstream分離 | 完了。元の12 package全軌道は適用外とし、field production、trajectory physics、state dimensionを独立化。後続の別成果物exact-P1 pre-event companionは限定的な時間離散parityにPASS |
| P15-D | species-constrained shifted-Maxwellian charge | 完了。単一正イオン・非正電位範囲を既存両積分器へ接続した |
| P15-E | finite-speed Epstein drag | 完了。Maxwell混合球抗力、明示Kn/速度比包絡、rate/Jacobian別bound、独立oracleと両積分器収束を受け入れた |
| P15-F | collisionless Barnes ion drag | 完了。単一正イオン、collection＋orbital、weak-coupling/collisionless gate、独立impact-parameter oracleを既存stage passへ接続した |
| F01 | static RZ/P1 reduced electrostatic builder | 完了。C2 Boltzmann--Bohm Poisson、semantic BC、Newton/GMRES、canonical field/provenanceを粒子engine外へ実装 |
| F02 | provider adapter、representative integration、external field V&V | mixed meshをadapterでP1化し、代表規模linear-solve gate、fixed-charge/electric run、field operator比較を完了 |
| 2B | joint OU Brownian、conditional split | 完了。限定B02 production sliceのstochastic統計と平面first-passageが合格 |
| 3 | DEP、用途が確定した追加ion-drag revision | model uncertaintyと数値誤差を分離 |
| 4A | time-dependent fixed-topology field | knot/discontinuity、prefetch、時間収束が合格 |
| 4B | tet4/tri3 full 3D | 3D release/event/scalingが合格 |
| 5 | 適合caseだけGPU、packaging/productization | CPU parityとend-to-end speedupを実測 |

DEP、continuous charge、Waldmann、Brownianの「初期対応」という旧表現は上表へ統一する。製品の
中心はCOMSOL一致ではなく、各stageが解析解、scenario、performanceの三つを通過することである。

---

## 11. 実装開始前の最終gate

次が決まるまでproduction実装を広げない。

1. particle propertyのauthorityとsurface sourceのweight/分布。
2. v0.1のdrag式revision、適用域、wall law、point-particle制限。
3. GeometryDomain/FieldLayoutのcanonical schema、P1/Q1 node order。
4. RK4とexponential midpointのstage、StepProposal、hit再積分規則。
5. physics category mapとmodel revision。
6. result segment、checkpoint、resume、commit順。
7. CPU memory plannerとevent-heavy benchmark matrix。

この七点を小さなmicrocaseで固定すれば、後続の電荷、時間依存、3D、GPUは同じ実行意味論へ追加できる。
逆に未確定のまま大量の物理modelや診断を実装すると、過去と同じ「局所修正の積み重ね」に戻る。
