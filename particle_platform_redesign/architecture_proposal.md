# 半導体チャンバー粒子軌道計算基盤：ゼロベース再設計案

## 0. この文書の位置づけ

本書はmodule責務、依存方向、runtime sequenceの詳細設計である。製品目的、実装範囲、
coreと外部toolの境界は [product_specification.md](product_specification.md)、具体的な実装順とwork packageは
[implementation_plan.md](implementation_plan.md) を正とする。本製品はCOMSOL比較基盤ではなく、
外部場を利用する独立した高速粒子輸送solverである。`model_dataset` は外部V&V資産として用い、
不備があればCOMSOL再実行・再抽出・version更新を行う。

Python環境、static quality tool、CI gateは [quality_tooling_plan.md](quality_tooling_plan.md) を正とする。
architectureを機械検査へ写す時も、少数のstable import ruleだけをimport-linterへ置き、設計書全体を
contract化しない。

既存`particle_tracer_unified`は実装母体にせず、継承するのは再検証できる数式、入力の由来情報、
失敗事例、少数の設計原則だけとする。既存のディレクトリ構成、実行経路、診断フレームワーク、
テスト構造は新基盤へ持ち込まない。

principal architecture／数値計算／performance reviewの指摘と採否は
[architecture_review.md](architecture_review.md) に記録した。以後、本書で決めるものは次に絞る。

- 製品として何を「COMSOL代替」と呼ぶか
- 粒子、場、メッシュ、境界、電荷、乱数、結果の標準表現
- 一つのengine内で複数のintegrator strategyを扱う数値方式
- 2D、軸対称、3D、定常場、時間依存場への拡張経路
- 10,000～1,000,000粒子を想定したCPU/GPU・メモリ戦略
- 機能追加によって再び複雑化しないコード配置と開発規律

データ品質の詳細は [data_quality_assessment.md](data_quality_assessment.md)、比較手順は
[vv_methodology.md](vv_methodology.md)、機械可読な監査結果は `evidence/` を参照する。

---

## 1. 経営・開発上の結論

### 1.1 推奨判断

新しいトップレベルディレクトリに、別パッケージとして実装を開始する。現行solverを
段階的に整理する方式は採らない。理由は、現行実装の中心経路がCOMSOL評価ケースの
主要物理を同じ方法で計算できず、境界、補間、物理、積分、診断の分岐が強く絡み合って
いるためである。

最初の製品版は「何でも解ける粒子追跡」ではなく、次の狭い垂直スライスを完成させる。

> SI単位の2D軸対称meshと定常場を読み、パーツ表面から粒子を発生させ、Epsteinまたは
> Stokes–Cunningham drag、電気力、重力、最初の壁イベント、stick・escape・鏡面反射・
> 確率付着を、一つのCPU batch経路で計算するsolver。

解析解と収束でこの経路を確立した後の機能追加は、一つの総順序へ混ぜず独立workstreamで進める。
trajectory physicsでは連続電荷、有限速度drag、collisionless ion drag、P16 Waldmann--Gallis熱泳動に加え、
P18-Cの集約連続帯電、P18-Iの二つの集約ion-drag感度revision、P18-Dの球形準静的DEP、P18-LのRZ lift感度、
P18-Rの二つのeffective-gas drag/thermophoresis感度revision、P22のTalbot熱泳動とSaffman liftまで完了し、
BrownianもB01数値基盤、B02のCartesian XY production、B03のRZ meridional force/charge連成に続き、
B04でterminal/active wallを同じfirst-hit経路へ統合した。固定トポロジー時間依存場も、線形補間、
実stage時刻評価、field knotでのmacro分割まで独立sliceとして完了した。solver本体を凍結した外部M3-V
trajectory certificationの最初の決定論pre-event sliceも完了している。state dimensionのP17、完全3D、GPUは
各々の利用caseと独立referenceが整った時に別変更として扱う。内部静電場生成はfield-production trackで完了している。
COMSOL照合は並行する外部V&Vであり、実装順やcore構造を支配しない。

### 1.2 COMSOLからの置換可能性

本製品がCOMSOL利用者の実務を置換できるかは、coreの定義ではなく外部評価として次を公開する。

| 適合レベル | 意味 | 合否の根拠 |
|---|---|---|
| 設定互換 | 同じ入力物理・境界・刻み・乱数方針を表現できる | capability matrix とcase manifest |
| 数値互換 | 決定論microcaseと対象ケースで軌道・力・電荷・イベントが許容範囲内 | versioned V&V report |
| 統計互換 | Brownianや確率付着を同一分布と判断できる | replica、信頼区間、分布距離 |

adapterが変換できないCOMSOL featureは外部変換時に拒否する。coreはCOMSOL feature名を知らず、
一般的な物理modelとsolver設定だけを扱う。全COMSOL機能との一般的な互換性は主張せず、外部V&Vが
検証済みのモデル、座標系、要素次数、物理、境界、versionを表にする。

### 1.3 性能目標の扱い

「COMSOLより高速」は設計目標であり、実測前の仕様保証にはしない。同一の物理、時間
刻み、出力頻度、粒子数で、cold/warm、計算、イベント、I/O、メモリを分解して測る。
10^4、10^5、10^6粒子の実ケース性能が揃ってから製品主張に変える。

---

## 2. 監査から確定した出発点

### 2.1 `model_dataset` の構造

新しい評価軸は12ケースで構成される。

- ion drag式：`formal_iondrag_theory_consistent` と
  `formal_iondrag_image_minimal_corrected`
- 場の作り方：保存プラズマ場を用いるCase P、熱流体結果からnonlinear SASS静電場を
  構成したCase A
- 粒径：10、30、100 nm
- 各ケース：287粒子、121保存時刻、0～0.03 s
- 座標：2D axisymmetric r-z、粒子domain 3
- 積分：固定刻み10 µsの陽的RK4
- 形状：7,732頂点、726境界edge、10,481三角形、2,426四辺形、47境界ID

入力と結果の内部整合性は高い。12ケースすべてで粒子・時刻キーの重複はなく、初期表は
軌道t=0と一致し、最終表は最後の保存行と一致する。速度、電荷、半径、質量、力/質量の
関係も浮動小数点誤差内で一致した。詳細な測定値は
`evidence/dataset_audit.json` と `evidence/case_audit_summary.csv` に保存した。

### 2.2 評価軸としての限界

この12ケースだけでは製品要件を検証できない。

- 発生点はパーツ表面ではなく内部の規則格子である。
- 反射は無効で、freezeの正例もなく、確率付着もない。
- 定常2D軸対称場のみで、時間依存場と3Dを含まない。
- Brownianの内部乱数増分とRK4各stageは保存されていない。
- COMSOL有限要素の自由度、形状関数、要素次数を完全には保存していない。
- mesh-point場は33,449行だが、一意座標は7,732点で25,717行が重複する。
- 規則格子座標には微小な浮動小数揺らぎがあり、丸め前のunique数は301ではない。
- `inside_model_domain` は粒子domain 3の権威あるsupport maskではない。
- 背景ファイル中の粒子派生列は各粒径向けに更新されていない。

従って、このデータは「COMSOLが保存した軌道・局所物理・状態のsystem regression」として
利用し、境界や確率過程の正しさは別の小型ケースで検証する。

### 2.3 時間軌道を見なければ見逃す差

同じCase・粒径で2種類のion drag式を比較すると、Case Pの10/30 nmは最終状態が全287粒子
で一致する。しかし1 ms時点の位置差中央値はそれぞれ約14.8 mm、4.40 mmで、全粒子が
一度は1 mmを超えて分離する。最終状態だけなら「一致」と誤判定する例である。

Case Aでは最終状態そのものも大きく異なる。最終status不一致数は10/30/100 nmで
181/203/158粒子である。この測定は `evidence/variant_divergence_time_series.csv` と
`evidence/variant_particle_divergence_summary.csv` に再現可能な形で保存した。

### 2.4 既存コードから残すもの、捨てるもの

残す候補は、すべて独立検証後に再実装する。

- SI標準化、`mass_kg`を慣性の権威とする考え方
- 入力の由来・hash・座標系をmanifestへ残す考え方
- 場の値を返すことと、その点が厳密にsupport内かを分離する考え方
- geometry scaleと浮動小数ULPから境界数値を決める考え方
- 線形dragの安定な指数更新式、法線・接線への壁速度分解
- 粒子ID・step・componentで決まるcounter-based RNG
- load_case/simulate/open_resultの公開境界、simulate内部のatomicな成果物書込み
- COMSOL adapterと比較器をsolver coreの外へ置く原則

捨てるものは次である。

- 現行のruntime orchestration、collision/refinement/contactの経路
- 物理名による巨大な分岐とbackendごとの重複実装
- 任意物理を有効にするとPython粒子loopへ落ちる実行方式
- accuracy不足を境界停止へ変換する状態機械
- exact-keyの位置だけを比べる比較器、最終snapshot中心の診断
- private helperの配置や文書ファイル数を固定するテスト
- 2～3粒子だけの性能基準、過去負債を合格扱いするbaseline運用

---

## 3. 製品境界と非目標

### 3.1 製品が所有する責務

新基盤は、外部で求めた場を受け取る一方向連成Lagrangian point-particle solverである。
製品が所有する処理は次である。

1. 標準caseの読込みと一度だけの意味検証
2. メッシュ内点・support・最初の境界交差の照会
3. 場の空間・時間補間
4. 粒子の力、電荷、確率過程の評価
5. 運動・内部状態の時間積分
6. 壁イベントと粒子lifecycle
7. 大規模粒子群のバッチ実行とstreaming出力
8. solver固有の最小限の実行統計

### 3.2 coreに入れない責務

- COMSOLのstudy名、式名、CSV quirks、version別回避策
- MPHの操作やCOMSOL Java exporter
- COMSOLとの差の診断・可視化・HTML report
- CAD修復、一般purpose mesh generator
- 流体・熱・プラズマの本格的な自己無撞着solver
- 旧設定の自動migration
- GUI固有の状態

これらはadapter、field builder、validation tool、applicationという外側のパッケージに置く。
ただし「熱流体場＋任意のプラズマパラメータから静電場を作る」機能は、solverとは別の
`electrostatic_builder` として公式に提供する。

### 3.3 P14-R `0.1.0.dev0` baselineと後続追加

| 軸 | P14-R baseline | 後続の現行追加／未実装候補 |
|---|---|---|
| 座標 | cartesian_xy microcase、axisymmetric_rz meridional | RZ場中Cartesian 3D粒子、cartesian_xyz |
| 場 | 定常、P1三角形、Q1四辺形、規則格子 | **現行追加**：固定topologyの線形時間snapshot。**未実装**：P2以上、移動mesh |
| 積分 | 一般固定RK4、native指数midpoint | 適応高次、GPU専用方式 |
| 物理 | 適用域を明示したEpstein linear / air Stokes–Cunningham、電気、重力、fixed charge | **現行追加**：continuous charge、有限速度drag、Barnesと集約場向け二revisionのion drag、Waldmann--GallisまたはTalbot熱泳動、球形準静的DEP、RZ rarefied-vorticity感度またはXY/RZ Saffman lift、effective-gas linear Epstein感度。追加model revisionは後続 |
| 確率 | なし（deterministic baseline） | **現行追加**：B02 joint OU、B03 RZ projected charged/forced Brownian、B04 active wall restart、B05の一様base depth＋境界候補だけの条件付き追加分割。**未実装**：Cartesian 3-D Brownian |
| 壁／topology | stick、escape、parameterlessな完全鏡面`specular`、係数付き`restitution`、明示fallbackを持つ定数確率stick | **現行追加**：非deposition terminal hold、壁温度・Maxwell拡散混合率・接線wall frameを持つ`maxwell_thermal`、静的2-D XY/RZの独立半径first contact、静的Cartesian XYのpure-translation periodic topology。**未実装**：material依存付着、re-entrainment、rolling/sliding、surface charging、moving geometry、RZ/3-D periodic |
| 出力 | final/event/probe/series、選択trajectory、segmented HDF5、checkpoint/resume | 分散実行・remote store |

未対応の組合せはPreparedRun作成時に具体的な理由とともに拒否する。現行engine revision
`particle_engine_v46`でproduction利用できるsubsetは、realized internal/surface schedule、fixed charge、
`oml_stationary_maxwellian_debye_huckel_v1`または
`oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1`、P18-C aggregate continuous charge、P15-E finite-speed Epstein、
P15-F collisionless BarnesまたはP18-Iの二つのaggregate ion drag、P18-D quasistatic spherical DEP、
P18-L RZ rarefied-vorticity lift sensitivityまたはP22 Saffman lift、P16 single-species、P18-R effective-gas、
またはP22 Talbot thermophoresis、無力場のballistic
退化形、静的geometryの`stick`/`escape`/`hold`/`specular`/`restitution`/`maxwell_thermal`/`probabilistic_stick`、dragなし・厳密一様fieldから証明した一定加速度、および証明済みの
Cartesian XY/RZ一般RK4/exponential midpointである。neutral dragはfinite-speed、linear、effective-gas linearの三つのEpstein revisionと明示air revisionの
Stokes--Cunninghamを選択でき、ion dragとDEPは独立したexplicit-acceleration categoryである。これに加えて
`ou_langevin`はそのB02 subsetに加え、B03のRZ meridional、fixed/continuous charge、native/effective-gas linear
Epstein、既存additive forceを受理する。B04でterminal lawとactiveな
`specular/restitution/maxwell_thermal/probabilistic_stick`を同じfirst-hit経路へ統合し、active hit後は残時間を
post-wall stateからfresh stochastic rootとして進める。
一定加速度は、topology-completeな材料boundaryによるevent confinement、またはboundaryなし・全cell supportedな
`RegularLayout`に対するproposalごとの解析的な座標極値検査のどちらかで連続field supportを証明する。
一般`rk4_reintegrated`は、既存のEpstein/electric/gravity、fixedまたは認証済みcontinuous charge、全cell supportedな共通layoutに限定する。
`rk4_global_abs_enclosure_v2`は全短縮RK4を覆うsupport、再積分、受入安全性と、材料/RZ event queryの保守的fallbackを
所有する。global supportを独立に証明済みのvalid `rk4_dense` rowだけは、現区間のdense Bernstein boundをevent broad phaseの
authorityにする。applicabilityはglobal field/model boundを先に試し、証明できないrowだけdense pathとlocal cell rangeでboundedに認証する。boundarylessは
`RegularLayout`だけ、topology-completeな材料domainではgeometryと完全一致するP1/Q1も許可する。
revision 3bはこのsubsetをtopology-completeな材料boundaryと連成し、sequential accepted RK4 piecesと
event-before-validityでterminal stick/escapeまで処理する。boundaryless unstructuredは連続包含gateが完了するまで拒否する。
M3-C1で、局所的には適用域内の強い非一様場が遠方cellを含むglobal extremaだけで時刻0拒否される反例が得られた。
完了したP19-Lは元の固定step proposal・endpoint・出力scheduleを変えず、integrator所有のdense RK4 pathをlocal cell rangeで
certificate-onlyに部分区間制限する。実際の適用域違反とcertificate budget内での証明不能を別failureにし、gate bypassや
COMSOL専用経路を作らない。rev3bのlocator-before-validity、hit prefix再積分、fresh residual proposalも変更しない。
Stokes–Cunninghamは後続P06-Sで明示air revisionとして追加済みであり、
v0.1全体のmodel selectorを先行して空interface化しない。
この数値pathはrevision 3bのengine v6で確立した。engine v7はproposal/refinement行のwork partitionだけを
deterministic wavefront batchへ置換し、event、proposal、enclosureのalgorithm revisionは変更しない。
engine v8はaccepted replayを要求frameと重なるrowだけへ限定し、exact-mesh P1/Q1 material domainをP06-Uとして
追加した。engine v9は単一のtable/surface schedule、Philox counter RNG、静止壁の
stick/escape/specular/probabilistic stick、priority/combined-normal corner応答、exact-path残時間、ballistic RZ axis
foldを追加した。engine v10 / event v6はCartesian XYの証明済み一定加速度surface departureとstart-contact
certificateを追加した。engine v11 / event v7はCartesian XY一般RK4の厳密内向きsurface departure、
single-facet active-boundary residual、右連続state jump、state-shared interaction capを追加した。tangentまたは
facet端点/cornerからの一般RK4 departureはfail-closedである。engine v12 / event v8でforce-coupled RZを
同じwork loopへ追加した。現行engine v46もその意味論を両integratorとも共有する。`coordinates.py`がsigned/canonical基底、
`fields.py`がaxis regularity、`events.py`がaxis局在、engineがwall/axis arbitrationとresidual commitを所有する。
P06-Sはdrag責務を小さいphysics runtimeへ
集約し、P08はparticle-local failure、series/probe、薄いCLIでStage 1Aをcloseした。P09は
resident stateのID対応を固定したまま使うresident-row active index、bounded microtile scratch、
solver-owned phase memory planを`cpu.py`に集約し、tile幅によらないstable mergeを固定した。P10は同じengineの
field sampling、sample済みprimitiveのphysics、RK4算術をNumba compiled array passへ置換し、accepted endpoint
だけをP1/Q1 resident hintへcommitする。scalar evaluatorはverification oracleだけでproduction fallbackではない。
P11は`physics/runtime.py`と`physics/compiled.py`が同じmodel passからdrag rate、target velocity、加算加速度を
返し、`integrators.py`がstart half-step predictor、midpoint-frozen指数更新、`state_at()`、連続path enclosureを
所有する形で追加した。`engine.py`はmethodを選ぶだけで、material first hit、wall residual、RZ axis、accepted state、
writerを従来の単一loopで調停する。RK4の`dt/tau`gateを指数法へ流用せず、非零charge rateはP11の未対応として
明示拒否する。曲線/chord偏差は`integrators.py`が全短縮secantを含むvelocity enclosureからmethod-neutralに作り、
`events.py`はRK4/指数法を区別せずcertificateへ使う。別physics式、別event engine、COMSOL専用pathを作らない。
P12は一粒子一workerの非重複tile ownershipを追加した。field/geometryはread-only共有、residual/event/failure/
statisticsはworker-localとし、最大`W`個のin-flight tile waveをmain threadがtile順にstable mergeする。workerは
fileへ書かず、writerはmain threadだけが呼ぶ。`geometry.py`はcompiled BVH query、`events.py`は保守的なRK4
clear/split事前認証、`engine.py`は同時刻wall prefixのcompiled batchとmergeを所有する。汎用schedulerや第二engineは
追加しない。
P13は同じengine/output ownerへ固定64 macro-step epoch、worker-wave event/failure stream、容量1のsingle-writer
queueとack backpressure、交互A/B checkpoint、`LATEST`を追加した。`simulate`はstrict identityが一致する
`OUT.partial`を自動resumeし、`open_result(..., recovery=True)`は`LATEST`までの確定prefixだけを読む。第二writer、
checkpoint migration、output schedule依存のcommit cadenceは追加しない。最初の`LATEST`以前はepoch 0から再実行し、
最終公開途中は次回runが安全に完了する。容量1 queueは同期ack型なのでbounded backpressureだけを保証し、I/O overlapを
性能主張にしない。power loss、remote filesystem、同じOUTへの複数process同時実行は保証外である。
固定64 cadenceとbackground queueは履歴である。現行はengineが
`W=macro_step_count+accepted_particle_pieces+candidate_queries+refinements`、`T=max(2^20,128N)`を計算し、
accepted macro barrierでcommitを決定する。cadenceはmanifest/resume identityへ入り、output schedule/slab非依存である。
`output.py`の同期single-owner writerがatomic persistenceだけを所有し、threadingや第二writerは再導入しない。
P14は同じengineを23行×3観測の直交型matrixで測り、profileで支配的だったP1/Q1 supported containmentとtable-start volume
containmentだけを、それぞれ`fields.py`と`geometry.py`所有のread-only BVHへ置換した。両indexはbroad phaseであり、
既存exact predicateが最終authorityである。field outside/masked provisionalは最近傍/tie意味論を守るfull scanなので
O(cell数)のまま残す。geometryは局所的にfloat64で解像不能なcellをprepareで拒否し、大offsetだけでは拒否しない。
`cpu.py`は両index residentとfield 256 B/cell、geometry 1,024 B/cellのprepare transientをmemory planへ計上する。

現行milestoneとalgorithm revisionの完全なrosterは`implementation_plan.md`と実行manifestだけをauthorityとし、
この文書へ複製しない。runtime v16はP19-L実装・性能snapshot、v17はDEP認証上限の1 ULP外向き境界を
変更した履歴revisionである。architecture上は、single-thread compiled engine、bounded slab、stackless boundary BVH、
単一physics stage pass、単一writerを維持する。
P18-Cはengine、proposal、event、resident state、memory plan、永続schemaを変えず、この境界内へ追加した。
M3-C1 event v14は`rk4_dense`のgeometry broad phaseだけを狭めた。現在のdense部分区間から外向きに丸めたBernstein
position/velocity boundを作り、global enclosureがfield supportを独立に証明済みのvalid rowだけevent BVH queryへ渡す。dense arithmeticが
無効なrowはglobal boundを保持してsplitし、support未証明rowはglobal event boundへfallbackする。
`rk4_global_abs_enclosure_v2`はsupport、global-first applicability、短縮RK4再積分のauthorityであり、dense boundを
physics validityやaccepted stateの代替にしない。event/result/case schema、wall law、first-hit順序は変更しない。
現行dense path v3はroot始点相対のBernstein enclosureをTwoDiff残差込みで構成し、下限は`-inf`、
上限は`+inf`方向へ外向きにworld座標へ戻す。dense位置評価も同じ相対制御点と残差を使い、
始終点では保存済みendpointを厳密に戻す。物理的な曲率とenclosureは局所変位scaleのままだが、公開chord-deviation
boundはworld座標での評価とendpoint chordを覆う狭い`8*eps`絶対座標termを別に含むため、戻り値自体は
完全な平行移動不変ではない。v3は、v2の広いpaddingが絶対座標に比例して膨張し、分割しても
event certificateが閉じなかった不具合を解消する。保存するendpoint、数学的なHermite/state path、event algorithm、
v3導入時のengine v32、event v14、RK4 global enclosure v2は変更しなかった。
surface sourceの現行形式は`realized_internal_surface_contact_schedule_v5`で、facet、strict interiorのfacet内座標、
初速度、release時刻、粒子物性、独立`contact_radius_m`を粒子ごとにcanonical HDF5へ保持する。
`particle_surface` groupではfacetの外向き法線と逆向きに半径だけoffsetし、`particle_center` groupでは
物理半径を保持したままoffsetしない。coreに分布selectorやsource RNGを持たない。
Brownian startは外向き法線成分の絶対値がroundoff幅を越えることを要求し、負ならdeparture、正なら
zero-time impactとする。zero/tangentはnudgeせず拒否する。moving geometryは対応certificateを伴う後続gateへ分ける。
P14は標準品質gateとverification/scenario 336件でsynthetic baselineをcloseした。regular大規模はhost幅で正の
scaleを示したが、unstructured 10kとevent-heavyは小さいか負で、worker数に比例するscratchも生じた。この結果から
P12のouter ThreadPoolを製品parallel runtimeとして残さない。P14-Pのv26試行はNumba内部thread team、thread数非依存の
parallel slab、再利用workspace、stackless boundary BVHへ置換し、worker wave、future merge、worker別scratchを
削除した。同期single-owner writerも選択済みである。linear/quadratic exactと一般曲線eventのflat SoA wavefront、
compiled boundary/RNG、row numerical status、batch surface release、direct columnar replay、bounded event/failure stagingは
engine接続済みである。このv26試行はP14-Pの速度・memory gateに届かなかったため、`threads > 1`の公開能力と内部thread teamを削除し、
P14-P closeoutのv27 compiled single-thread runtimeへ一本化した。詳細な実装・計測・削除条件は
[`solver/docs/parallel_execution_plan.md`](solver/docs/parallel_execution_plan.md)を権威とする。主用途を結合した
時間/mesh収束とglobal-bound/event costは、その後のP14-Uで閉じた。T03とP14-Rのlocal gate/evidenceも完了し、
2026-10-05のremote Windows/Linux workflow成功でP14-Rとv0.1配布経路を閉じた。P15はユーザーの明示指示で旧着手blockerを解除し、
RK4-firstと、その受入後のexplicit midpoint chargeまで完了した。外部M3-V applicability/relevance評価も完了し、
P15 stationary OMLと元datasetの全軌道比較は適用外、式parityと次workstreamだけを確定した。
T04の用途別cache採用は精度・profile条件付きであり、現行の生成能力は§7.2のP1/regular/affine Q1からfull regularへのstatic/linear-time変換である。

P06 revision 3aは、この能力を一度に全geometryへ広げず、boundaryless Cartesian XY、fixed charge、
既存のEpstein/electric/gravity、全cell supportedな`RegularLayout`だけに実装した。`fields.py`がcanonical node extremaと
regular support box、`physics/`がsample済み数値からの純粋なmodel bound、`integrators.py`が全短縮RK4評価を覆う
path enclosure、`engine.py`が証明と唯一のproduction loopを所有する。材料boundary、unstructured support、RZ、
continuous chargeはrevision 3aでは拒否した。revision 3bは同じowner境界のまま材料boundaryだけを追加し、
第二engineや一般certificate frameworkは作らない。

---

## 4. 論理アーキテクチャ

### 4.1 一方向の依存関係

```text
COMSOL / CFD / plasma / CSV / 計測 / user mesh
                       │
                       ▼
             tools/importers + builders
                       │ case_format.write
                       ▼
          Canonical DataBundle ── SimulationSpec
                       │
                       ▼
                  load_case
                       │ SimulationCase
                       ▼
                 engine.prepare
                       │ PreparedRun（非公開）
                       ▼
              single production engine
      source → integrator/physics → event → wall law
                       │
                       ▼
                 canonical ResultStore
                 ┌─────┴──────────┐
                 ▼                ▼
              analysis       external V&V
                 │                └─ COMSOL
                 ▼
            visualization
```

source adapterはDataBundleまで、coreはSimulationCaseからResultStoreまでを扱う。COMSOL比較器は
通常のResultViewと参照結果を扱う。core内にCOMSOL列名、study、state aliasを入れない。

### 4.2 推奨ディレクトリ

```text
particle-platform/
├─ pyproject.toml
├─ src/chamber_particles/
│  ├─ __init__.py
│  ├─ api.py
│  ├─ case.py
│  ├─ yaml_input.py
│  ├─ case_format.py
│  ├─ coordinates.py
│  ├─ geometry.py
│  ├─ fields.py
│  ├─ sources.py
│  ├─ physics/
│  │  ├─ catalog.py
│  │  ├─ forces.py
│  │  ├─ charge.py
│  ├─ integrators.py
│  ├─ events.py
│  ├─ boundaries.py
│  ├─ engine.py
│  ├─ cpu.py
│  └─ output.py
├─ tools/
│  ├─ importers/comsol/
│  ├─ importers/tables/
│  ├─ case_builder/
│  ├─ electrostatic_builder/
│  ├─ field_preprocessor/
│  ├─ analysis/
│  ├─ visualization/
│  └─ vv/comsol/
├─ tests/
│  ├─ verification/
│  ├─ scenarios/
│  └─ performance/
└─ docs/
```

これは1機能1ファイルを強制する図ではない。小さなhelperを別ファイルにせず、意味のある所有者へ
まとめる。一般型置き場の`models.py`やre-export専用moduleは作らない。moduleが二つの独立した変更
理由を持つほど大きくなってから分割する。module ownershipと依存方向は本書を権威とし、
[architecture_review.md](architecture_review.md) は判断理由と指摘のreview記録として読む。

### 4.3 公開API

```python
from chamber_particles import load_case, simulate, open_result

case = load_case("case.yaml")
summary = simulate(case, output="runs/case_001")
result = open_result("runs/case_001")
```

- `load_case`：先にYAML全体とresource設定をparseし、HDF5 metadataからcanonical numeric footprintを
  求めて上限外をpayload展開前に拒否する。その後、既にSIへ正規化されたcanonical dataの
  schema、unit metadata、局所connectivity、owner参照、静的参照整合を検査する。source unit変換はadapterの責務である。
- `simulate`：prepareでmodel requirements、coordinate/integrator/backend capability、compiled
  evaluator、大域geometry topology、memory planを解決し、数値計算をstreaming出力する。
- `open_result`：final、event、保存trajectoryを遅延読込みする。

`PreparedRun`とatomic writerは内部実装であり、公開APIにしない。独立した巨大なpreflight subsystemも
作らない。`load_case`と`simulate`内部準備で入力エラーを一度だけ検出し、詳細監査は外部tool側で行う。

### 4.4 主要データ型

```text
DataBundle
  GeometryDomain
  FieldLayout / FieldSet
  boundary groups/materials
  provenance/content hashes

SimulationSpec
  source/particle/physics/boundary/time/resource/output

PreparedRun (immutable, non-public)
  coordinate_kernel
  field_sampler_plan
  PhysicsPlan + StateLayout
  integrator_profile
  event_locator_plan
  memory_plan
  output_plan

ParticleState (SoA)
  identity: id, source_id, rng_key
  physical: x, v, mass, diameter, charge/internal state
  lifecycle: pending, active, stuck, escaped, failed, held
  geometry: cell_id, last_boundary_id
  failure_reason: uint16
```

accuracy、support、geometry判定はstep-local resultとrun counterで持ち、全粒子へ常設配列を増やさない。
失敗時だけ`failure_reason`へ確定する。精度budget枯渇を壁停止や粒子消失へ変換しない。

---

## 5. Canonical DataBundleとSimulationSpec

### 5.1 物理量は原始量だけを保存する

背景場には、粒子を変えても不変な原始量だけを保存する。

- gas velocity、temperature、pressure、density、viscosity、mean free path
- electric potential/field、density、species state、temperature gradient、vorticity
- 場を作ったsolution/time/parameter index

次は粒子状態と原始場からruntimeで求める。

- Knudsen数
- Epstein response time
- Brownian diffusivity
- Reynolds数、drag coefficient
- charge依存force、screeningに粒子情報が入る派生量

`model_dataset`では背景CSVの10/30/100 nm版が数値的に同じで、粒子派生列が実質10 nm相当
のまま複製されている。これを「入力値」として信じる設計は、粒径を変えても一部物理が
変わらないsilent errorを生む。

### 5.2 YAMLとHDF5の所有を分ける

`case.yaml`は人が変更する`SimulationSpec`と、対応するHDF5への参照だけを持つ。
文法解析は`yaml_input.parse_document(bytes)`に一元化し、全階層の重複keyとmerge keyを拒否する。
このleafはI/O、hash、domain検査を持たず、`case.py`と外部producerが各domainの検証を所有する。
通常のaliasはdomain検査へ渡し、provenanceは元のUTF-8 bytesから計算する。

```yaml
format_version: 2
case:
  name: chamber_screening_001
  data_path: case.h5
  expected_content_hash: "sha256:..."
time: {...}
solver: {...}
resources: {...}
physics: {...}
sources: [...]
boundaries: [...]
output: {...}
```

`case.h5`だけが座標系、SI単位、geometry topology、field layout/value/support、boundary
group/material、realized table source、producer provenanceを所有する。YAMLにこれらを再記述しない。
逆に、HDF5は物理model選択、時間刻み、壁面law、resource、出力scheduleを持たない。

canonical必須の由来情報はHDF5の`/meta/provenance_json`に保存する。v1のtop-level必須keyは
`producer`、`producer_version`、`source_sha256`、`field_semantics_revision`、
`producer_metadata`とする。COMSOL version、study/dataset、smoothing/recovery等は
`tools/importers/comsol`が`producer_metadata.comsol`へ条件付きで保存し、他solverや計測dataへ
必須にしない。quantityの単位・成分・座標基底は各field record、node/element orderingと
evaluation policyはversioned schemaが所有し、provenanceと二重化しない。

### 5.3 ファイル形式

- `case.yaml`：人が編集する設定、model選択、HDF5 path、少量metadata。
- `case.h5`：mesh、field、particle/source table、boundary metadata。chunk、compression、dtypeを明示。
- result：JSON manifestと閉じたHDF5 epoch segment。詳細は§11.5と主仕様§14。
- CSV：adapterの入口・人間確認用。coreの高性能native形式にはしない。

NaNを材料領域や境界の唯一の表現にしない。geometry topologyとdomain IDをsupportの権威に
し、各fieldは「このdomain/timeをsupportする」という独立maskを持つ。NaNは欠損値として
扱い、位置判定には使わない。

### 5.4 adapterでの正規化

全adapterはSI、quantity semantics、topology ID、support、provenanceを確定してから
`case_format.write()`を呼び、coreへproducer固有の曖昧さを渡さない。COMSOL adapterでは特に次を
完了する。

1. SIへ変換する。
2. COMSOL node/element/boundary IDを保持してdense local indexを別に作る。
3. 重複mesh-point行をnode IDで統合し、同一点の値衝突を検査する。
4. 新しいexportは座標の浮動小数丸めjoinではなく、topology IDで値とmeshを結ぶ。
5. 物理量名をcanonical quantityへ一度だけmapする。
6. domain/boundary selectionをmanifestへ固定する。
7. 派生物理列を除去し、必要なら監査用参照値として別namespaceへ隔離する。

既存producerがfield側のnode IDを出力できない場合に限り、adapter境界でcoordinate-only legacy入力を受けてもよい。
ただし、明示した物理長tolerance内で各canonical nodeにprovider点が厳密に1個、各provider点も厳密に1回だけ対応する
全単射を証明し、最大対応距離と入力hashをprovenanceへ残す。補間、丸めbucket、最近傍fallback、欠損埋めは許さない。
これはF02の既存CSVを移行する限定例外であり、新しいexporterの契約を弱めない。

### 5.5 設定モデル

物理ファイルとsweep設定を次へ分ける。

- `DataBundle`：geometry、field layout/value、境界タグ、由来など再利用する事実。
- `SimulationSpec.physics`：有効な力と電荷model。
- `SimulationSpec.sources`：canonical HDF5内のrealized internal/surface table参照。
- `SimulationSpec.boundaries`：境界groupごとのwall law、priority、law parameter。
- `SimulationSpec.run`：時間、積分profile、backend、resource、出力、seed。

物理moduleの選択は明示的にし、依存するfieldがなければ`load_case`または`simulate`内部準備が
拒否する。パラメータの
defaultと外部solver由来の値を混ぜない。COMSOL adapterは一般的なphysics/run設定へ明示展開し、
元のCOMSOL設定とversionは外部V&V manifestにだけ記録する。

---

## 6. 座標系、geometry、mesh

### 6.1 共通抽象と専用kernel

「すべてを3Dベクトルとして同じコードで解く」方式は、軸対称の意味論と高速化を曖昧に
する。一方で座標系ごとに製品全体を複製しても保守できない。共通にするのは状態遷移、
field/force/eventのinterface、出力schemaであり、hot loopは次の専用kernelとする。

- `cartesian_xy`
- `axisymmetric_rz_meridional`
- `axisymmetric_field_cartesian3d`
- `cartesian_xyz`

この列挙はparticle motion modeであり、入力dataの座標表現ではない。canonical HDF5はgeometry/fieldの
`cartesian_xy | axisymmetric_rz | cartesian_xyz`を所有し、SimulationSpecはparticle motion modeを所有する。
`engine.prepare`が両者の対応を一度だけ検査し、result manifestへ両方を記録する。現在の未release v1では
XY/XYとRZ/RZ-meridionalだけを有効にし、Stage 2AでRZ data/Cartesian3D motionをschema revisionとして追加する。

`axisymmetric_rz_meridional`では状態は `(r,z,v_r,v_z)` で、方位運動を持たない。`r=0`は壁ではなく
座標seamである。trial pathを軸で分け、`r <- -r`, `v_r <- -v_r`に相当する基底変換を
`coordinates.py`だけが行う。wall/reflection eventには数えない。
force-coupled RK4ではtrialだけをsigned radial chartで進め、stageごとにcanonical RZへ写してfield/physicsを
評価する。accepted stateは常に`r >= 0`である。axis accessibilityは、geometryのaxis接触、またはboundaryless
fully-supported regular boxの`r_min=0`から`fields.py`が一度だけ導き、axis nodeのvector regularityに使う。
`gravity_buoyancy_standard_v1`はaxis accessibilityと無関係に、すべての`axisymmetric_rz` domainで
`gravity_m_s2[0] = g_r = 0`を要求する。`cartesian_xy`では同じ配列の第1成分`g_x`を非零にできる。
P05当時はaxis path分割をproduction loopに接続せず、軸到達が
最初の材料hitより先になり得るpathを拒否したが、exact pathはP07、一般RK4はP06-RZで解禁済みである。
両端`r=0`のvolume外周edgeはaxis seamとして材料BVHから除外し、
材料boundaryとして登録した入力も拒否する。
`axisymmetric_field_cartesian3d`では背景場だけを`r=sqrt(x²+y²)`で評価し、粒子は
Cartesian 3D状態を持つ。等方Brownian、方位速度、磁気力が必要な場合はこちらを使う。
円柱座標swirl専用kernelは初期scopeへ入れず、実caseと独立検証が揃った後の候補とする。

### 6.2 2D軸対称から3Dを再構成する責務

軸対称場からの3D表示・source展開は、場を3D voxelへ複製する処理ではない。

- 場のsampling：`r=sqrt(x^2+y^2)`で2D場を評価し、vector成分をCartesian基底へ回転。
- sourceの3D展開：外部入力作成側が面積要素`2πr ds`でsurfaceを重み付けし、`theta~Uniform(0,2π)`を
  realizeした粒子rowをcanonical入力へ書く。
- 軌道の3D表示：meridionalなら初期thetaを保持してtrajectoryを回転表示。
- 統計：2D点数をそのまま3D面積密度とみなさず、2πr weightを使う。
- 境界：3D pathを`(r(s),z(s))`へ写した曲線とRZ境界のfirst hitをbracket/localizeし、
  RZ法線を衝突点thetaでCartesianへ回転する。

座標変換は`coordinate_kernel`、確率分布のrealizeは外部producerが所有する。geometry importer、field補間、
visualizer、solver coreの各所で同じsampling policyを別々に実装しない。

### 6.3 GeometryDomainとFieldLayout

衝突geometryとfield離散化は別の型にする。同じFE meshの時だけstorageとlocatorを共有する。

`GeometryDomain`は次を保持する。

- vertex座標とstable external ID
- 粒子domainまたはvolume connectivity
- oriented boundary facet、boundary/material ID、owner、向き
- 全体scale、AABB、局所element scale
- boundary/containment用BVHとadjacency

`FieldLayout`はregular axesまたはunstructured vertex/element、basis、support domain、time axis、
point-location indexを持ち、`FieldQuantity`がlayout IDを参照する。一つのrunに複数layoutを許せるが、
architecture上はgeometryとfieldを常に独立の事実として扱う。Stage 1の初期実装能力は同じP1/Q1 mesh、
または明示選択された検証済みregular layoutへ限定できるが、型やIDを同一視しない。同じlayout groupの
quantityだけ一回のelement lookupを共有する。この制限はtemporary capability checkであり、producerや
mesh identityの定義ではない。

三角形P1はbarycentric、四辺形Q1はisoparametric bilinear形状関数を使う。四辺形を暗黙に
三角形分割すると、COMSOL Q1補間とは別の場になるため禁止する。P2/高次要素は、実modelの
node orderingと自由度を固定したcaseが揃うまで拒否し、頂点だけをP1と偽らない。

現`model_dataset`のquadはCSV列順`[node1,node2,node3,node4]`が周回順ではなく、正しい周回順は
`[node1,node2,node4,node3]`である。列順でpolygon化すると2,426要素中2,019がbow-tieまたは
面積ほぼ0になる。adapterはこの順序をversioned import ruleとして明示し、全要素の向き、
自己交差、Jacobianを検査する。座標からの自動並べ替えでDOF対応を失わない。

### 6.4 point locationと粒子追跡

P10/P14のregular locatorはsupported containing-cell common pathだけ軸indexからO(1)個の候補を評価する。
outside/masked provisionalは物理最近傍意味論を守るためcompiled全cell走査を使う。P1/Q1はaccepted endpointのprevious-cell
hintがstrict interiorを含む場合だけfast pathで採用する。共有面は最小supported cell IDというowner規則を
守る。P14ではhint missと初回sampleのsupported containmentだけをfield-owned stackless BVHで候補化し、同じexact
predicateをID順に評価する。containing supported cellがなければoutside/masked最近傍full scanへ戻る。trial hintは
resident stateへcommitしない。

P03のscalar locatorは正しさを固定するverification oracleとして残すがproduction fallbackには使わない。
unstructured indexはP10へ入れず、初回stageのfallbackを稀と仮定しなかった。P14のrealistic cell count、initial
localization、cross-cell motionで支配的と確認したため上記一つのindexへ置換した。別samplerやruntime fallbackを作らない。
inside判定は参照座標への固定ULPだけにせず、物理空間の後退誤差、局所element scale、座標ULP、
Jacobian conditioning、再構成残差を一つのalgorithm revisionで扱う。悪条件cellのinside領域を無制限に
広げず、revisionが根拠とともに固定するmesh品質上限超過と非有限補間は明示errorにする。この上限を
利用者向けtuning knobにしない。

table sourceのstrict-interior検査はboundary-first分類を維持し、その後のvolume union containmentだけを
`geometry.py`所有のmixed tri/quad BVHで候補化する。AABBはbroad phaseで、既存half-space predicateがauthorityである。
局所diameterに対してedge/非incident vertex clearanceがfloat64で解像不能なcellはprepareでfail-closedとし、
絶対座標が大きいだけでは拒否しない。edge長はscalar/compiled parityのためCPython `math.hypot`でprepareする。
この内部index、field index、build toleranceをcase schemaやproducer設定へ露出しない。

境界交差も全edge/triangle走査を行わない。

1. 予測segmentのAABBをBVHへ照会。
2. 候補facetとrobustなsegment intersectionを行う。
3. 最小の正のhit parameterを選ぶ。
4. corner/edgeで複数facetが同時なら、versioned tie-break policyを適用する。

数値許容差は固定m値や散在する`1e-12`ではなく、mesh scale、facet scale、速度、時間、
float64 ULPから`GeometryNumerics`として一度解決する。位置を内側へ微小移動して判定を通す
repairは行わない。

### 6.5 surface release

製品の主用途なので、後付けの初期位置処理ではなく第一級のsource型とする。

```text
RealizedSurfaceSource
  particle_id[N]
  facet_id[N] / facet_parameter[N]
  release_time_s[N] / velocity_m_s[N,2]
  particle_properties[N]
```

分布、重み場、時刻schedule、粒径・質量・電荷分布、角度分布は外部入力作成側の責務とする。coreは
内部sourceの`position_m`またはsurface sourceの`facet_id + facet_parameter`と、共通の粒子ごとの物理量を
一度だけruntime scheduleへ変換する。source生成frameworkをcoreへ追加しない。

RZの3-D回転面surface samplingは`2πr ds`、2-D断面用の`meridional_length`は`ds`、3D triangleは面積で
重み付けする。canonical surface rowはfacet IDとstrict interiorのfacet内座標を保持し、runtime scheduleへ
変換する時にprepared geometryからpositionを一度導出する。runtimeはそのpositionとcanonical source facet IDを
保持し、ownerとnormalはprepared geometryが所有する。residentなbarycentric/local-coordinate authorityを
重複して持たない。壁からの出発は「接触中だが
流体側へ離れるsegment」としてevent locatorが扱い、任意のinward epsilonで位置をずらさない。
発生面の法線向きと粒子domain側をimport時に検証する。

engine v10では`v·n`のscale-aware符号を最初のdeparture authorityとする。明確な内向き/外向き速度を
加速度で上書きしない。budget内の非零法線速度は曖昧として拒否し、厳密tangentだけを証明済み一定加速度の
`a·n`で分類する。内向き加速度のdeparture tokenは別wall hitで速度が変わるまで保持する。`events.py`は各intervalで
source supporting lineの内側/境界band、tangentまたは明確な内向き速度、明確な内向き加速度を再証明し、成立時だけ
source facetを除外する。他facetは常に通常のfirst-hit authorityを保ち、別wall応答後はsource facetも通常検索へ戻す。
外向き加速度はzero-time impactであり、terminal応答は許すが反射後に内向きdepartureを作れなければ失敗する。
releaseがmacro終端ならtokenを次macroへ渡し、run終端ならright-continuousなrelease frameだけを保存する。
各intervalで再証明するため、微小`dt`を含むmacro partitionで物理eventを変えない。

surface粒子のIDはcanonical rowに明示し、tableおよび他surfaceとの重複をcase load時に拒否する。
確率分布が必要なら外部入力作成側でrealizeし、coreのseedや実行partitionへ依存させない。

### 6.6 geometryの異常判定

次は警告ではなくcase build errorにする。

- non-manifold boundary、zero-area element、自己交差
- 粒子domainにownerを持たないwall facet
- boundary selectionの重複でpriority未定義
- source facetの法線向きが不明
- support islandやholeの意味がmanifestにない

検査ownerを分ける。`case_format.read`はindex範囲、局所node順、boundary rowと宣言owner edgeの整合だけを
検査する。`geometry.prepare`はvolume incidenceから大域外周を導出し、non-manifold、重複boundary、
内部edgeのwall登録、RZ axis seam以外の外周欠落を拒否する。v0.1は内部thin baffleを未定義のまま受理しない。
修復やfacet生成は外部builderに残し、coreは判定と拒否だけを行う。
P05で一つ以上の材料facetを与える場合、このtopology-completeな`line2`を必須とする。
boundary rowが0件の明示的なcollision-free解析caseは、同じengineのno-hit profileとして残す。

geometryとfield topologyが異なること自体はerrorではない。各々のsupport、単位、座標系、
transformが未定義、または宣言したshared mesh IDの内容が一致しない場合をerrorにする。

CAD修復そのものはcoreで行わず、adapterの修復reportをcase provenanceへ残す。

---

## 7. 場の表現と補間

### 7.1 field samplerの最小戻り値

field samplerは値だけでなく、計算の根拠を返す。

```text
SampleResult
  values[N, Q]
  spatial_support[N]
  temporal_support[N]
  element_id[N]
  time_interval[N]
```

これは新しいcontract frameworkやclass hierarchyを意味しない。`fields.py`のscalar referenceと
compiled batch関数が同じ小さな戻り意味論を持てばよく、provider registryやDI containerは作らない。

trial stepを有限にするため境界外の近傍値を一時評価する場合でも、`values`と
`spatial_support`を混ぜない。値が返ったことを「粒子がdomain内」と解釈しない。support外の値は
provisional trial専用であり、そのstageを含むstepをそのまま受理しない。先行wall eventを局在して
supported区間だけ再積分するか、eventがなければ`FieldError`とする。

spatial supportはsupported cellの閉包の和とする。共有面を含むcandidateにsupported cellが一つ以上あれば
insideとし、最小supported cell IDを決定論的ownerにする。全candidateがmaskedならoutsideであり、
provisional値は物理距離が最小のsupported cellへlocal coordinateを射影した有限値に限る。regular/P1/Q1、
cell hint、探索順でsupport規則を変えない。補間が有限にならない場合は明示errorとし、値をclampしない。

v0.1で運動を駆動するrequired fieldはparticle domain全体を覆うcontinuous node-associated
regular/P1/Q1 fieldへ限定する。cell-associated quantityは保持・外部解析には使えるが、不連続面の側選択modelを
定義するまでforce/chargeへ使わない。部分supportを将来許可する時はstage点のbooleanだけでなく、path上の
最初のsupport exitをeventとして扱う。

producerのNaNはcanonical supportではない。adapterは明示supportを確定した後、masked-only DOFを決定論的な
有限placeholderへ正規化し、方法と件数をprovenanceへ残す。supported cellが参照するDOFはfinite必須であり、
placeholderを物理補間へ使用しない。

### 7.2 reference場と高速cache

一つのcaseに二種類の場を持てる。

- `reference`：元meshと形状関数で評価。V&Vの権威。
- `cache`：規則格子、texture、cell-local係数など高速化用。

現行B1/B2の外部`tools/field_preprocessor`設定format v2、producer v4は、node-associated P1・regular・exact affine Q1
sourceからfully-supported regular targetへのstatic/固定topology linear-time cacheを公開する。source/target cellの共通partitionを
流し、入力float64座標をexact binary rationalとして扱ってsource overlapとtarget coverageを別検査する。正面積sliverを
epsilonで捨てず、float64積分patchのcollapseやproduction locatorのowner不一致は`unresolved_validation`で停止する。
warped Q1、unstructured/partial targetは公開拒否。warped Q1はinverse map/Jacobian/導関数/積分の証明がないため、
coarse/fine sampled差で認証を代用しない。元runtimeのwarped Q1評価を制限するものではない。

共通patch上のvalue・stored-component gradientのweighted relative L2とtarget support boundary上のvalue relative L2を
公開gateにする。XYは`dx dy`/`ds`、RZは`2 pi r dr dz`/`2 pi r ds`を使い、RZ axisの境界measureは0である。
positive 4×4 Duffy Gauss ruleはこのpolynomial spatial scopeのdegree 5までを積分し、nodal/basis/coordinate/accumulationの
roundoff allowanceを含む`relative_l2_upper`を閾値と比較する。reference lower normが0のときはerrorとallowanceも0の場合にだけ
relative errorを0とし、それ以外は公開しない。積分が非表現可能、またはbudgetを超える場合も停止する。
linear-time場は全snapshotを保持し、隣接snapshotの共通partitionからerror/referenceのGram momentsを計算する。
時間内の二次norm比をbounded Bernstein区間とroundoff allowanceで認証し、snapshot endpointだけの比較は用いない。
stationary rootの推定はreport-onlyで、公開判定の証明には使わない。時間相殺・下界不足・52段分割で未解決なら公開拒否。
`sample_absolute_max`はsample値であり、certified L∞上界ではない。
support boundaryの誤差は材料wall-near誤差を認証せず、このgateだけでtrajectory/charge/arrival/fate精度も認定しない。

`validation.memory_limit_mb`、`workspace_rows`、`max_patch_work`を必須にする。source/candidate resident arraysとwriter/index
transientを含むowned-array planをallocation前に検査し、bounded rowsとpatchごとの小さいrational polygonでpartitionをstreamする。
work counterはcoverageと全選択fieldを共有し、AABB候補行とintersection作業を数える。これはprocess RSSのhard limitでも
out-of-core方式でもない。original fields/layouts/geometry/sourcesを保持し、生成条件・誤差・資源をoutput provenanceへ保存する。
NaNを跨ぐbilinear補間やsolid越しDelaunayは許可しない。

v0.1ではphysics設定が参照する一つのlayoutをrunのauthorityにする。cacheは外部preprocessorが別fieldまたは
別DataBundleとして作り、利用者が明示選択する。場所ごとにcacheとmesh-nativeを切り替えるhybrid samplerは、
実ケースで必要性・精度・speedupが確認されるまで延期する。runtime failure時のsilent fallbackは作らない。

### 7.3 unstructured補間

- P1 triangle：barycentric形状関数。
- Q1 quad：逆写像Newton法＋bilinear形状関数。逆写像失敗を近傍値で隠さない。
- vector/tensor：格納basisをmanifestに記載し、座標kernelで変換。
- gradient：可能ならsolverから出力された同一意味の場を使用。節点値からの数値gradientを
  COMSOLの回復gradientと同一視しない。

`fields.py`のoptional `PreparedFieldSet.spatial_gradient`はnodal P1/regular/Q1の形状関数を微分し、
直前の`sample`と同じsupported cell ID・basis weightを使う。出力はcaller-ownedの`[N,C,2]`であり、第二locatorや
逆写像、least-squares fit、通常production stageへのgradient scratch追加をしない。canonical二座標に関するstored componentの
偏微分であり、RZ vectorの3D covariant gradient、COMSOL recovery、物理primitiveの回復を所有しない。
gradient操作はstatic nodal fieldだけを受ける。preprocessorは各snapshotのstatic viewで同じ値/gradient samplerを使い、隣接snapshotのGram momentsを外部で積分する。Q1のJacobianは同じcell/basisから構成し、第二inverse mapを作らない。

COMSOL内部補間を再現するには、element order、DOF、形状関数、smoothing/recoveryを保存する。
現在のnode sampleだけで一致しない場合、COMSOL評価APIをreference oracleとしてprobe点を
追加出力し、外部補間の近似誤差と物理式の差を分離する。

### 7.4 時間依存場

現行の時間依存fieldは、固定された空間topologyに複数snapshotを持たせる最小revisionである。

```text
TemporalField v2
  time_s[T]                     # finite、狭義単調、T >= 2
  values[T, N, C]
  interpolation: linear
  outside_policy: error
  mesh_motion: fixed
```

値はsnapshot-major layoutで保存し、現revisionは全snapshotをprepare時にresident保持する。既存のregular/P1/Q1/cell
空間locationとweightを一度だけ作り、各RK/指数法stageの実時刻で隣接snapshotを線形補間する。runが参照する全fieldの
time knot和集合でmacro intervalを分割し、範囲外はclampせずprepareまたはsampling時に失敗する。`hold`、periodic、
明示discontinuity、streaming/double buffer、moving topologyは未実装であり、固定mesh samplerへ条件分岐を足さず
それぞれ独立revisionとする。

外部`tools/field_preprocessor`は設定format v2、producer v4でP1/regular/exact affine Q1→full regularの
linear-time cacheを公開できる。元snapshotとtime knotを保持し、全隣接区間のcache error/reference norm比を認証する。
元物理場のsnapshot adequacy、未解像の時間変化、discontinuityを認証するものではなく、undersampled inputを
細かいparticle stepで修復できるとはみなさない。

### 7.5 field builder

熱流体結果とプラズマパラメータから静電場を作る処理は、次の独立pipelineとする。

```text
thermal-flow DataBundle
      + plasma closure parameters
      + electrostatic boundary conditions
                    │
                    ▼
        electrostatic_builder
          Poisson equation
          charge-density closure
          nonlinear solve / convergence report
                    │
                    ▼
        augmented FieldSet + provenance
```

Poisson問題にはcharge closureと境界電位/fluxが必要であり、任意のプラズマparameterだけから
一意の電場は決まらない。builderは仮定、式、境界、収束、残差を成果物にし、solverは完成した
fieldを読むだけとする。Case Pの外部プラズマ場とCase Aの内部生成場は、同じcanonical
quantity interfaceを介して使い分ける。

`caseP` / `caseA`はdataset/COMSOL workflow名であり、trajectory engineの設定値にはしない。製品設定は
`imported_external_plasma_fields`または`reduced_electrostatic`というfield-production modeを選び、後者だけが
versioned builder model、plasma parameter、semantic electrostatic boundary group、nonlinear solve設定を持つ。
両modeは同じcanonical `FieldSet`とprovenanceを出力し、その後のcharge/force model選択と直交する。
`model_dataset`のion-drag二式は外部V&Vのmodel-form感度に留め、coreへ複製しない。将来、独立した利用caseと
referenceに基づき別式を採用する場合だけ、既存modelを変更せずphysics catalogの別revisionとして追加する。

F01ではこの境界を`tools/electrostatic_builder`として実装した。最初のcapabilityはstatic
`axisymmetric_rz`、geometry完全一致のtriangle-only P1、単一準中性bulk、C2-smoothed
Boltzmann--Bohm closureに限定する。`2 pi r` P1 FEM、analytic Jacobian、明示continuation、damped Newton、
matrix-free restarted GMRESで完成fieldをcanonical writerへ渡す。粒子engine、公開三API、case schemaは変更しない。
F02ではmixed triangle/quad参照meshをadapterが一度だけP1へcanonicalizeし、builder/runtimeに第二mesh経路を
作らず完了した。代表meshは1,987 node / 3,779 P1 cell、1,826 free nodeで、10 continuation ramp、総linear
iteration 2,821、最終relative residual `1.8441e-13`、charge-balance error `5.8498e-21 C`を得た。
GMRES basis＋Hessenbergは1,235,088 B、solve-only観測は0.689/0.737 sであり、fallback、反復上限増加、第二solverは
追加しなかった。完成fieldは既存RZ solverのfixed-charge＋Coulomb smokeへ渡した。外部比較は同一export nodeの
記述比較に限定し、独立mesh convergence、COMSOL trajectory、wall/Freeze parityを検証済みとはしない。

---

## 8. 粒子物理

### 8.1 運動方程式

基本状態は位置`x`、速度`v`、内部状態`y`（電荷など）である。

```text
dx/dt = v
dv/dt = linear_relaxation(x,v,t,y) + Σ a_k(x,v,t,y) + stochastic process
dy/dt = G(x,v,t,y)
```

runtime寄与は加速度へ統一する。診断力だけ`mass_kg * acceleration`から作る。粒子属性は
`mass_kg / drag_diameter_m / electrostatic_radius_m / displaced_volume_m3 / model_weight`を独立して
持ち、runtimeで相互に逆算しない。

### 8.2 物理moduleの形

第三者が力を追加する入口は、hot loop内の任意Python callbackではなく、最小宣言とcompiled
evaluatorの組とする。

```text
PhysicsModel
  model_id/revision
  category/contribution_kind/exclusive_group
  required_fields
  parameters
  applicability summary
  supported_coordinates/integrators
  compiled evaluator
```

`load_case`と`simulate`内部準備はrequired fieldとparameterを一度確認し、寄与を
`linear_relaxation / explicit_acceleration / internal_rate / noise_coefficients`の四種へ並べる。
primary drag、charge、noiseと各additive-force categoryは0または1 modelとし、blend/compositeは
versioned modelを明示する。同じinternal-state sliceにownerを二つ置かない。一般dependency
graphやkernel-family型階層は作らない。scalar referenceは通常関数としてverificationに使い、
production規模で未compiled modelがPython loopへsilent fallbackすることは禁止する。

`engine.py`は時間work、field sampling、event、commitの調停だけを所有する。modelが二つ目のdragへ
増える前に、展開済み`PhysicsPlan`から作る小さなphysics runtimeへ、sample済みprimitiveのreference評価、
primitive extremaからの力bound、model applicability、一定加速度certificateを集約する。このruntimeは
field locatorやgeometryをimportせず、manager/registry/DI containerにも拡張しない。`fields.py`は位置から
primitiveとそのsupport/extremaを返し、`physics/`はその数値を物理量へ変換し、engineは両者を順序付ける。

`PhysicsPlan`は`has_force`、`evolves_continuous_state`、`requires_stage_evaluation`を別の事実として保持する。
`has_force`は加速度寄与、`evolves_continuous_state`は`Z`等の連続状態rate、`requires_stage_evaluation`は
integrator stageごとの再評価を表し、Stage 1の`force_coupled`一個へ畳み込まない。
`oml_stationary_maxwellian_debye_huckel_v1`単体は`false/true/true`であり、電気力等との合成planでは
`has_force`だけが別寄与によりtrueになり得る。continuous chargeは電気力が無いcharge-only caseでもstage evaluatorを
通り、動的`Z`を持つproposalはlinear/quadratic exact specializationを使わない。proposal/event/memoryの選択は
この三特性を一度だけ参照し、capability frameworkや二つ目のengineを作らない。

### 8.3 drag

希薄気体の主経路はEpstein dragとし、relative velocity `u_g-v_p`、gas temperature、density、
mean thermal speed、accommodation parameterから計算する。Knudsen数、response time、Reynolds数は
粒子径と局所原始場から毎回求める。

連続域のStokes/Cunninghamや中間域modelを追加する場合、Kn範囲を宣言し、自動切替は
versioned composite modelとして実装する。適用範囲外で別式へ黙って切替えない。near-wall
drag補正は独立moduleであり、壁距離・法線の品質要件を持たせる。

### 8.4 決定論的な追加力

物理catalog全体では、重力/浮力、電気力、熱泳動、DEP、free-molecular lift、ion dragを扱う。
Stage 1は重力/浮力と電気力までで閉じ、P15-Fでcollisionless Barnes ion drag、P16で
Waldmann--Gallis free-molecular thermophoresisを追加した。現在の比較caseについて必要な式と入力が確認できたため、
Stage 3ではP18-Dの球形準静的DEP、P18-Iの二つのion-drag sensitivity revision、P18-LのRZ
rarefied-vorticity lift sensitivity、P22のTalbot thermophoresisとSaffman liftを追加済みである。各moduleは
acceleration vectorを返し、field samplingや粒子状態更新を
所有しない。

- 電気力：`F_E=qE`。電荷更新とstage時刻を一致させる。
- DEP：P18-DはproducerがDC/cycle-mean、solution、回復法を固定した`grad(mean_E_squared)`をcanonical vectorとして出し、
  `electrostatic_radius_m`と`mass_kg`から球形準静的dipole力を計算する。coreはEを微分せず、producer認証済みの
  `maximum_point_dipole_radius_m`をprepareで検査する。
- 熱泳動：`physics/forces.py`が局所並進熱流束形、Kn/低drift適用域、global boundを所有する。catalogは
  field authority、runtime/compiledはsample済みstage評価だけを所有し、gradient回復とproducer/COMSOL比較は外部に置く。
  P18-Rのeffective-gas revisionではproducerが一つの有効Maxwellian/pseudogasと有効並進伝導熱流束`q_eff`を認証し、
  coreはspecies配列やtemperature gradientを回復しない。P22 Talbotはproducer提供の`grad(T)`とcase明示の
  `k_p,Cs,Cm,Ct`を同じstage passで使い、Waldmannとのblendや自動切替を行わない。
- lift：P18-Lはaxisymmetric no-swirlのr/z運動だけを対象に、
  `F=K (omega_phi e_phi) x (u_g-v)`、`K=C_L*pi*rho_g*lambda_g*a^2`、`a=drag_diameter_m/2`を使う。
  `C_L`は有限正値を明示し、producer所有のsigned方位vorticityを消費してcore内で速度場を微分しない。
  `lambda_g/a>=10`をfail-closedに要求し、一般Saffman liftや3-D revisionへ暗黙拡張しない。速度依存項はruntimeの
  全非drag加速度bound callbackへ統合し、exponential enclosure v3が開始速度とhalf predictorで再評価する。
  integrator v2、engine v30、proposal v7は維持し、追加決定論力を許さないB02 Brownianとの併用は拒否する。
  P22 Saffmanは同じcategoryの別revisionとしてXY/RZのproducer提供signed面外vorticityを使い、Stokes--Cunningham
  またはdragなしだけを受理する。連続体・低Re・`Re_s<<sqrt(Re_G)`の固定policyを全pathでfail-closedに検査し、
  near-wall liftやP18-L式へfallbackしない。
- ion drag：P15-Fはrelative-velocity方向、collection＋orbital、linear two-species Debye screening、非正電位の
  一つの明示revisionである。electric-field方向、collisional/nonlinear screeningは別versionとして扱う。

`model_dataset`の2種類のion dragは、どちらかを恣意的に正解へ寄せる材料ではない。P18-Iではfloor、image補正、
方向、screeningを含む各式を独立に再定義し、optionalなversioned sensitivity modelとして実装する。既存Barnesを
変更せず、自動選択・blend・Case P/A分岐を作らない。比較式と一致することと物理妥当性は別statusで評価する。

P18-Rのauditでは、native linear Epstein式parityは約`1.1e-15`でPASSしたが、既存P15-E/P16の
physical applicabilityはmixture/model authority不一致により12/12 caseで`NOT_APPLICABLE`、保存PPRに`q_eff`が無いため
thermophoresis pointwise replayは`NOT_TESTED`だった。このため既存式ownerを共有する
`epstein_linear_effective_gas_sensitivity_v1`と
`waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1`を追加した。両者はproducer認証済みの
one-effective-Maxwellian/pseudogas reference/sensitivityで、`lambda/a>=10`と明示`maximum_speed_ratio<=1`を要求する。
既存linear/P16の`0.1` gateは維持し、species-resolved mixture truth、COMSOL branch、gradient recoveryを追加しない。

後続M3-C1では、熱泳動に必要なPPR heat-flux primitiveだけを最小COMSOL補足として再exportした。既存v6と座標が完全一致する
13,202 active saved rowについて、producer-form replayのcomponent-scale normalized residual最大は
`4.4046499933294035e-16`、global relative L2は`1.0992667494449471e-16`であり、frozen saved-state区分を8/8閉じた。
これはP18-Rのspecies/physical-applicability判断を変更せず、continuous pathまたは統合軌道を認定しない。

### 8.5 電荷

電荷moduleを一種類の「charge」へまとめない。

- `fixed_charge`
- `oml_stationary_maxwellian_debye_huckel_v1`：stationary Maxwellian OMLとDebye--Hückel capacitanceで
  `dZ/dt=Γ_i-Γ_e`を運動と連成
- `oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1`：単一・単価正イオンのshifted-Maxwellian
  OMLを非正表面電位だけで運動と連成
- `discrete_charge_jump`：整数電荷の確率jump
- `aerosol_charge_relaxation`：COMSOLの空気aerosol系モデルを必要時に別実装

プラズマ粒子へaerosol charging式を流用しない。OMLも衝突性sheathや非Maxwell分布で常に正しい
わけではないのでvalidity domainをmanifestに残す。

P15の初回modelは`a/lambda_D <= 0.1`を必須とし、ion driftは収集率へ入れず、resolved
`M_i <= 0.1`をstationary近似のapplicability gateにだけ使う。`model_dataset`のheuristicやCOMSOL側の
model名は外部V&Vの比較条件であり、core catalog、rate式、capacitance、applicabilityのauthorityにしない。
adapterは必要なcanonical primitiveへ変換するだけで、COMSOL専用evaluatorを作らない。

P18-Cの`aggregate_relative_drift_regularized_two_current_v1`も同じ`internal_rate` ownerへ置くが、P15/P15-Dを
置換しないoptional reference revisionとする。局所有効正イオン質量、電子・正イオンthermal voltage、背景screening長は
canonical scalar field一つずつに統一し、uniformなproducerも定数fieldを書く。revisionは粒子ごとに
`lambda_eff=max(electrostatic_radius, screening_length)`を適用する。parameter-or-fieldの二重authority、COMSOL式parser、
charge専用subcycleを追加しない。
正負電位branch、正則化、floor、指数範囲、有限invariant/rate/derivative boundを同じmodel revisionが所有する。
有限boundのためcaseは正則化前の`|u_i-v|`の有限正値上限を必須宣言し、actual stageとcontinuous pathで検査する。
これは速度を切り詰めるparameterではない。

P15-Dのshifted revisionは
`M_i=|u_i-v|/sqrt(8 k_B T_i/(pi m_i))`の有限正値上限をcaseへ必須とし、actual stageとcontinuous
pathの双方で検査する。初期`Z<=0`と全primitive/drift rangeでの非正平衡をprepareで認証し、invariantを
`[Z_min,0]`へ固定する。zero-driftはstationary負電位branchへ連続に一致する。負イオン、複数イオン種、正電位、
emission、collisional/magnetized chargingは別revisionであり、外部比較式のfloor/clipをcoreへ持ち込まない。

連続電荷ODEは固定RK4では各RK stageで場、速度、Zに対して評価し、explicit `hL_Z<=0.5`を維持する。
現行native exponential midpointとB03は同じ予測midpointで`G`と`J=dG/dZ<=0`を凍結し、root基準の
affine lawを`expm1`で解析更新する。運動も同じ`Z_mid`を使うので一つのcoupled proposalである。
exponential pathは`hL_Z<=0.5`をstability gateにせず、別runの`h,h/2,h/4`がaccuracyを選ぶ。
clip、charge-only subcycle、第二engine、一般IMEX、暗黙equilibrium置換は作らない。
charge数`Z`とCoulomb`q`の権威を一つにし、両方を独立入力にしない。

動的電荷modelは、局所rateだけでなく有限な`Z` invariant/bracket、rate上界`B_Z >= |R_Z|`、
derivative上界`L_Z >= |dR_Z/dZ|`、適用域を`physics/charge.py`から提供する。RK4区間は
`h L_Z <= 0.5`と全stage/accepted endpointのinvariant包含を必須とする。charge-aware path enclosureはその
charge区間から`max |Z|`を求め、electric
acceleration boundへ渡す。prepare時の初期`Z`だけで全stepの電気加速度を囲わない。stage値、区間、または
加速度上界を有限に認証できないrowは、0や平衡値へ置換せずfail-closedとする。

### 8.6 Brownian

Brownianは「各stepにランダムな力を足す」だけでは、step refinement時に別過程になる。

- 外部V&V：必要なら一般solverへ同じstepと乱数方針を指定し、比較情報を外部manifestへ保存。
- native method：速度緩和を含むOrnstein–Uhlenbeck/Langevinのjoint `(x,v)` 更新を使う。
- adaptive refinement：joint Gaussianの親incrementをbinary tree pathで条件付き分割する。
- wall近傍：一般曲面でexact first passageとは主張しない。最初のrevisionは固定depthのdyadic nodeと
  cubic Hermite leaf pathをgeometry/outputから独立に定め、depthに対するfirst-passage統計収束で規定する。

乱数identityは`seed, particle_id, macro_interval, root_stochastic_interval, tree_level, tree_index,
component, stream`から構成し、accepted-step番号やevent ordinalへ依存させない。処理順、thread数、
tileサイズ、inactive compactionで乱数列が変わらない
counter-based Philox/Threefry系を使う。

B01はjoint OU平均・full covariance、親endpointを保存するconditional half-split、上記identityの
Philox4x32-10 normal drawまでを数値基盤として実装した。B02は`noise` categoryと`ou_langevin`をengineへ接続し、
各macro-rootで係数を凍結して固定depth treeを時間順に処理し、OU leaf endpointから作るcubic Hermite pathを
材料eventとframe/probeの共通authorityにした。Cartesian XY、Epstein linear drag-only、fixed-charge state、terminal
stick/escapeだけを受理し、continuous charge、drag以外の決定論力、反射、RZ、3-Dは暗黙に近似せず拒否する。
engine v30はroot covariance、conditional split、deterministic meanの実表現可能性を検査し、例外時だけrow別に
局在して不良粒子を`nonfinite_physics`で停止し、正常粒子を継続する。Gaussian bridgeは有限tubeへ決定論的に
囲えないため、単なるRMS距離でintervalをclearにしない。B05でも`interval_tree_depth`は全rootに一様な
numerical-path精度のauthorityである。`adaptive_max_depth`は、その有限depth pathがwall、RZ axis、または
証明不能区間の候補になった時だけ保存済み親endpointから条件付き二分する上限であり、確率的clearやbase depthの
自動置換ではない。base=maxは従来の固定depth pathへ退化する。どちらもsampleしていない連続OU軌道のexact
first-passageやzero miss probabilityを主張しない。
memory plan v16は観測候補率でなく`adaptive_max_depth`からtree workを保守的に解決する。

同一COMSOL軌道を再現するには、COMSOL乱数増分と内部stepの情報が必要である。現在の
`model_dataset`だけでは粒子単位のexact replayはできないため、Brownian offの決定論比較と、
複数seed/replicaの統計比較を分ける。

現データでは`v_phi=0`かつ運動DOFがr/zだけである一方、診断用`Brownian_force_phi_N`は非zeroで
3D合力magnitudeに含まれる。RZ no-swirl kernelはr/zへ投影された成分だけを積分し、phiを含む
magnitudeから加速度を再構成しない。将来のexportもtrajectory-effective forceと3D diagnostic
sampleを別schemaにする。

B03は第二engineを作らず、既存`ou_langevin` strategyへ一つのversioned stochastic proposalを追加して完了した。比較用の
RZ revisionは2自由度meridional projectionであり、manifestのcoefficient policy
`macro_root_frozen_midpoint_v1`、composition `stochastic_exponential_midpoint_v1`を使う。
各rootでnoise-free決定論exponential-midpoint predictorからmidpointを得て、線形Epsteinの`gamma,u,T`、
全additive acceleration `a`、
continuous charge rate `G`と`J=dG/dZ<=0`を一度だけ評価して凍結する。`u_eff=u+a/gamma`に対するjoint exact OUで
`(x,v)`をroot全幅進め、Zは`macro_root_affine_exponential_v2`のdense stateとする。
このpredictorはstateをhalf-step commitするStrang/K-O-K分割ではない。

既存conditional treeは同じ凍結係数でroot Gaussianを分割し、root endpointを保存する。RZ axisに到達したら
accepted prefixをcommitしてfoldし、macro残時間を次の`root_stochastic_interval`ordinalとして係数再評価・
独立root drawで再開する。元rootのcubic remainderはfold/restrictしない。B04は同じprefix-commit/fresh-root規則を
active wallにも適用し、terminal lawと反射・確率wallを受理する。native/effective-gas線形Epstein、fixed/continuous charge、
既存additive forceを明示的に解決し、B02のXY/fixed-charge/drag-only revisionはbitwise不変である。dense chargeは
root基準`G_mid+J(Z_root-Z_mid)`のaffine exponentialとして全leaf intervalをprepared charge invariantへ照合し、
外れまたは証明不能をfail-closedにする。

検証は定数係数mean/full covarianceのexactness、noise-offの2次収束、manufactured charge-electricの
weak mean観測次数`>=0.9`、axis restart、tree-depth first-passage、slab/output/resume identityとする。
一般state-dependent SDEのstrong orderやweak 2次を主張しない。初回受入revisionはengine v34、proposal v9、catalog v16、
event v15、runtime v17、compiled tile v16、memory plan v13であり、現行supersessionはengine v46、proposal v10、
event v22、catalog v23、runtime v22、compiled tile v21、memory plan v16である。B03 path arrayの静的な保守上限は一slab rowあたり
`648 B`で、受入上限`2048 B`内に収まる。正式characterizationは2,000/20,000粒子×4構成×3反復を全粒子active・
failure 0で完了し、計時/RSSは[`solver/evidence/b03/`](solver/evidence/b03/README.md)のmachine-local・non-gating証跡とする。
characterizationで検出した反復加算由来の終端tailはengine v34で除去した。macro timeは補償積和による
`start + n*dt`のindexed gridから構築し、endへのsnapはfloat64構築roundoff内だけに限定する。

event v15は物理position budgetとroundoff budgetを加算し、facet-local offset dotを補償演算で、cubic Hermiteの
評価・enclosureをroot-relative TwoDiffで扱った。event v16はvalidなRK4 dense rowに限り、integratorが所有する
root-relative position Bernstein control enclosureをfacetの外向きhalf-spaceへinterval射影する。4制御点の外向き上限が
すべて既存position budgetの負側に厳密に入る候補だけを、Bernstein convex-hull性により非接触と認証する。入力が欠損・
非有限・不整合、または不等式を証明できない場合は従来どおりsplit/fail-closedとする。monotone-approach clearはcubic
Hermite derivative-Bernstein enclosureの明示opt-inだけに限定し、exponential、scalar pathへv16 certificateを適用しない。
position-control payloadは`144 B/row`、構築時の追加peakは`128 B/row`で、保守的な同時live見積りは約`1.76 KB/row`と
既存の一般stage scratch上限`2048 B/row`内に収まるため、memory plan v13を変更しない。broad AABB pre-countは
このfacet-local clear前の保守的なwork量であり、event数や物理結果ではない。
現行event v22は、曲線pathでも個々のfacetを独立に認証して最初のhit時刻を決め、その局在budget内で同時かつ
共有nodeへincidentなfacet集合を保持する。exact、一般RK4、exponential midpoint、Brownianは同じ候補集合、
priority、combined-normal意味論を使う。有限半径material contactとperiodic center crossingは同一のfirst-event順序で比較し、
証明不能な候補やmaterial/periodic混在cornerは一面へ丸めずsplit/fail-closedする。

P17後のCartesian 3-D isotropic Brownianは別revisionである。RZ projected revisionの合格から3-D isotropyや3-D
COMSOL parityを主張しない。どちらも同じRNG identity、event/output/checkpoint経路を使い、Python callbackや別runtimeを作らない。

---

## 9. 境界と粒子lifecycle

### 9.1 event-firstの処理

macro stepの終点がdomain内かだけを判定すると、薄い部品を飛び越す。integratorは内部
`StepProposal(endpoint_state, state_at, path enclosure, provisional validity)`を返し、各accepted numerical
pieceに対して最初の境界eventを解く。`StepProposal`が表すのはversioned integratorで定義した離散軌道であり、
path enclosureはその離散軌道を保守的に包む。真のODE解まで包むとは主張しない。ODE離散化誤差は別runの
`h / h/2 / h/4`収束、解析解、manufactured solutionで評価し、幾何判定幅へ混ぜない。

```text
integratorがtrial pathを提示
    → EventLocatorが最初のhitを返す
        → hitまで状態を積分・局在
        → BoundaryLawがoutcomeを決定
        → 残り時間を新状態で継続
```

P05は解析的なballistic直線segmentと`line2`のexact intersectionにこの流れを限定した。
scale-awareなposition/time budgetで最初hitと同時candidateを求め、candidateはfacet ID順に保持するが、
当時のproductionは単一candidateのhitだけを応答へ渡した。table初期点は材料boundaryのstrict interiorとする。
engine v9のP07 exact-path sliceはforce-free surface zero-time departureと複数candidate responseを解禁した。
engine v10 / event v6はCartesian XYの証明済み一定加速度surfaceへ拡張した。engine v11 / event v7は
Cartesian XY一般RK4について、厳密内向きsurface departureとsingle-facet activeな壁応答後の残時間継続を解禁した。
tangent、facet端点/corner、またはその他の曖昧なstart-contactはfail-closedである。
engine v12 / event v8は同じpiece列へRZ axisをmaterial wallとは別のfirst-event候補として加え、axis prefixの
再積分、basis fold、残時間継続を行う。双方が局在済みで認証時刻の不確かさがwallと重なる場合だけwallを優先し、
axis端点とmaterial cornerの完全tieは一般RK4 cornerとしてfail-closedにする。

curved pathはchordだけでno-hitを判定せず、physics係数・速度・加速度から保守的なpath enclosureを
構成できる場合だけcheap no-hitを許す。boundを構成できない区間は、区間始点から順番にdyadicな
RK4 pieceとして再積分し、各pieceで力評価とevent探索を行う。このgeometry-drivenなpiece列そのものを
accepted numerical pathとし、出力時刻は分割を変えない。hit状態は同じintegratorでhit時刻まで再評価する。
refinement budgetが尽きた場合は
`AccuracyStatus=budget_exhausted`またはrun failureであり、架空の壁停止にしない。幾何的に
交差の有無を証明できない場合だけ`IntegrityStatus=indeterminate_geometry`とする。

各pieceの`swept path tube`は、力から導いた離散path変位boundとgeometry/roundoff budgetを包含しなければ
ならない。full-step/two-half-step差、Hermite–chord偏差、midpoint差はaccuracy indicatorには使えるが、
それだけをtrajectory enclosureとはみなさない。BVHのAABB分離、または候補facet支持線に対するtubeの
法線方向intervalがfacet別event budgetを含めても厳密に分離する時だけno-hitを確定し、
重なる時は次の逐次pieceへ二分する。hit rootはtime bracket、spatial tube、candidate-facet setの三条件が
収束するまで局在する。stage・endpointが非有限なら即時失敗とする一方、support/applicability逸脱は
provisional flagとして保持し、先行eventで切り落とされなかったaccepted prefixに対してだけ確定する。

engine v7の各粒子はleft-firstなdyadic workの次の1 pieceだけをwaveに出す。wave内でfloat64の
target timeが完全に同じ行だけをstableにまとめ、候補粒子を256件ずつの有界chunkでproposalする。
`inspect_rk4_piece`とBVH candidate queryは行ごとのままであり、accepted path、event順、集計の意味を変えない。
event v5は、`integrators.py`所有のcomponentwise chord-deviation/roundoff boundを用いて、
`events.py`が単一facetの外向き横断、normal time bracket、tangent/time-shiftを含むposition radius、
endpoint clearanceを証明する。証明失敗時はsplitし、最後はfull-tube budgetへfallbackする。engine v7、
proposal v3、enclosure v1、schema/APIはこの数値変更では不変である。engine v8は数値pathを変えず、frameと
重なるaccepted rowだけをreplay用に保持した。engine v9はexact linear/quadratic pathへsurface/wall/RZ axis
意味論を追加し、C08～C10を公開経路で実行する。engine v10 / event v6は一定加速度surfaceの
start-contact certificateを追加した。engine v11 / event v7は一般RK4へinterior-facet start-contact certificateと
single-facet active-boundary residualを拡張した。interaction capはparticle residual stateで共有し、次hitが実際にある時だけ
残区間を分割する。event時刻のactive jumpとterminal transitionは右連続である。
engine v12 / event v8でこの意味論を保ったままRZ signed-stage/axis eventを追加し、現行engine v46も維持する。
P11の指数pathも同じleft-first piece列とfirst-event arbitrationを使う。現行event locatorはintegratorが渡すmethod固有の
component-wise deviationを受け、指数法では全短縮secantを含むvelocity enclosureから
`h * (v_upper - v_lower) + roundoff`を外向きに作る。position enclosure全幅を流用した過剰分割や、
指数法専用event loopは残さない。

### 9.2 event record

すべのmaterial wall interactionとperiodic topology transferをlogical canonical logへ残す。

```text
particle_id, event_ordinal, interaction_kind, time_s
primary_facet_id, destination_facet_id, candidate_offset/count, boundary_id, material_id
contact_radius_m, position_m, position_post_m, normal
velocity_pre, velocity_post
charge_number_pre, charge_number_post
law_id, outcome, model_weight
localization_residual_m, position_budget_m, time_budget_s
```

`normal`はlaw適用に使ったeffective response normalである。単一facetではその外向き法線、
combined-normal反射では選択subsetの正規化合成法線を保存し、raw candidate facet集合は別tableへ保持する。

release、boundary interaction、failureは別groupとし、boundaryの`interaction_kind`は`wall`または
`periodic_translation`に固定する。同時hit候補facetは別columnar candidate tableへoffset/countで保存し、
primary facetだけへ情報を潰さない。

現在の物理配置v1はnullableな統合tableではなく`/events/release`と`/events/boundary`を別groupにし、
group名をevent typeとする。candidateはboundary groupのoffset tableで保持する。P05で常に確定済みの
`geometry_status`と未実装failure groupを空列として先行追加しない。実装列は
`solver/docs/result_format_v3.md`を権威とする。P13のsegment拡張はlogical identityを変えず、P18-Hは
`held` lifecycle series列の追加だけでresult/checkpoint schemaを一度だけv2へ更新した。

保存時刻のstatus変化からhitを推定しない。このlogが境界比較と付着統計の権威になる。
P05はreleaseを粒子ごとのevent ordinal 0、最初のboundary eventをordinal 1とする。

### 9.3 BoundaryLaw

幾何交差と物理応答を分離する。

- `stick`：lifecycleをstuckへ。位置をhit点に固定。
- `escape`：escapedへ。以後のkinematicsはlogical nullとする。
- `hold`：heldへ。hit位置、hit時速度、hit時電荷を保持するinactiveな非deposition終端。
- `specular`：parameterを持たない完全鏡面反射`v'=v-2(v·n)n`。
- `restitution`：法線・接線反発係数`e_n`、`e_t`を必須とし、`v'=-e_n v_n+e_t v_t`。
- `maxwell_thermal`：`wall_temperature_K`、`diffuse_reflection_fraction`、`wall_velocity_m_s`を必須とし、
  完全熱適応half-range Maxwell fluxと壁frame鏡面を混合する。wall velocityは静的groupの全facetへ接線方向だけ。
- `probabilistic_stick`：定数確率でstickし、非stick時の`specular | restitution | maxwell_thermal`を`otherwise`へ明示。

以上が現行BoundaryLawである。point/finite-radiusの差はgeometry/eventが所有し、lawはevent候補に対応する
接触法線だけを受ける。有限半径は静的2-D XY disk / RZ sphere中心とmaterial `line2`のcapsule first contact、
surface sourceのinward offset、table sourceのclearanceまでを扱う。rolling/sliding、速度依存rebound、
resuspension、surface charging、moving geometryは後続catalogとする。
groupの`contact_geometry`は未指定時`particle_surface`で、明示`particle_center`だけ候補の有効半径を0とする。
`case.py`が選択を正規化し、prepareした一つのfacet surface maskをgeometry/event/source/memory planが共有する。
粒子の物理半径、mass、drag/電気半径は変更せず、一mesh/BVH上で候補固有の法線・残差・clearanceを解決する。
混在modeの時刻順が証明できない同時候補はfail-closedとし、lawのaliasや第二geometry engineを作らない。
COMSOL raw stateのfreeze/disappearは外部adapterがrawのまま保持し、正例microcaseで実際に出力された
hit後位置、速度、statusだけを比較manifestでcanonical意味へ対応付ける。P18-H初回referenceは電荷を
出力していなかったためCOMSOL電荷保持は認定せず、非零電荷保持はprovider非依存の公開scenarioが所有する。P18-Hは半導体装置内の非deposition終端という
producer非依存の利用caseに対して`hold/held`をstick/escapedと分けて追加した。COMSOL名やraw status codeはcoreへ
入れず、外部adapterが明示的に対応付ける。
P05はparameterを持たない`stick`と`escape`だけをprepareした。engine v9は静止壁の`specular`と、
定数確率でstickし非付着側を明示specularへ渡す`probabilistic_stick`を追加する。stickはpost速度0、escapeは
表面impulseなしでpost速度をpre速度と同じにする。
現行`contact_wall_laws_v7`では`specular`を係数なしの完全鏡面だけに限定し、係数を持つ反射を
`restitution`へ分離した。`probabilistic_stick.otherwise`はparameterなしの`specular`、または二つの係数を
明示した`restitution`、または完全なparameterを持つ`maxwell_thermal`だけを受理する。`maxwell_thermal`は
`v_w-c_n n+c_t(-n_y,n_x)`のpolicy-free half-range変換とcounter RNGを使い、静的geometryへ法線wall motionを
持ち込まない。`hold`はparameterを持たず、hit時payloadを保持したままinactiveにする。
hit後のchargeやkinematicsを進めるpaused状態、再開、再飛散はこのlawの責務外である。

確率を0～1へruntime clampしない。設定やmodel出力が範囲外ならerrorとする。一つのboundary groupへは
wall lawを一つだけ設定する。cornerで異なるgroupのfacetが同時候補になった場合は、manifestのpriorityで
選択し、実行順に依存させない。

確率wall lawは`seed/particle_id/physical_boundary_event_ordinal/law_stream`をkeyにする。event locatorの
二分やrejectはphysical ordinalを進めない。source、wall、Brownian、charge-jumpのstreamを分離する。

### 9.4 corner、同時hit、連続接触

corner/edge同時hitは、最小時刻だけでは法線が一意でない。候補facet集合をeventに残し、
吸収系lawなら一致する結果を適用し、反射系は明示した合成法線/逐次反射policyを使う。
同時刻の反復hitを無限loopにしないよう、hit ordinalと相対法線速度で「壁から離れる」segmentを
departureとして扱う。位置nudgeは使わない。static geometryの壁速度は法線成分0だけを許し、moving
normal wallはmoving geometry対応まで拒否する。contact slidingは点粒子追跡の基本境界ではなく、
必要なら別物理moduleとして追加する。

---

## 10. 数値積分method

### 10.1 一つのengineでstep strategyを分ける

一般性、高速、stiff drag、Brownianを一つの数値式へ押し込まない。一方で別runtimeも作らない。
production engineは一つで、決定論の二つとBrownianの一つ、計三つのstep strategyを持つ。

| method | 目的 | 方法 |
|---|---|---|
| `rk4_fixed` | 一般決定論、verification・外部solver照合 | 古典RK4、stage連成、指定固定step |
| `exponential_midpoint` | 高速・安定なnative標準 | drag解析更新＋midpoint |
| `ou_langevin` | 有限深さの慣性Langevin SDE | B02はstart-frozen drag-only、B03はmacro-root stochastic exponential-midpoint。どちらもjoint OU＋固定depth conditional tree＋cubic Hermite leaf |

COMSOL adapterは必要なcaseで`rk4_fixed`とstepを指定するだけであり、coreにCOMSOL version別profileや
`comsol_default`を作らない。

### 10.2 固定RK4 reference

運動と電荷を同じ状態vectorとして、各stageで位置、速度、電荷に対応する場と力を評価する。
決定論計算を標準とする。外部V&Vで確率式を同条件にする必要がある場合、乱数規則は外部manifestで
固定する。field time knot、global source discontinuity、wall eventで必要な区間を切る。個別releaseは
その粒子の残時間workとして扱い、保存時刻は積分刻みとは別にaccepted `StepProposal.state_at()`から
評価する。保存時刻を増やすために積分stepやRNG pathを変えない。

RK4のpathは端点位置・速度から作るcubic Hermiteを候補表現にできるが、Hermite–chord偏差だけを
保守的path enclosureとはみなさない。最初に、physics係数とfield extremaから各proposalの連続軌道全体が
required field support内にあり、選択modelのapplicability内にあることを証明する。revision 3aでは、globalな
canonical field extremaとmodel係数からRK4内部stageの速度・加速度を保守的に包絡し、float64で外向きに丸めた
位置boundが、macro終端だけでなく`state_at()`が生成する任意の短縮区間の内部stageとaccepted endpointを
すべて含むことを要求する。Epsteinでは同じ速度boundから`lambda/a`下限と`|u-v|/c_bar`上限を作る。
fully-supported regular boxへの包含または連続applicabilityを証明できないcaseはhidden subdivisionせず
fail-closedにする。これによりtrajectory scheduleは新しいvalidity判定を導入しない。

revision 3bで初めて、各piece始点から作ったpath tubeがBVH node、facet AABB、または候補facet支持線の
認証済み法線intervalから分離する時だけno-hitを確定する。
候補が残る区間は、元のmacro proposalをparameter方向に切るのではなく、時間順のdyadicな逐次RK4 pieceへ
分けて再積分する。急峻な位置依存場では、同一始点からの短縮RK4 endpoint曲線の部分区間が局所時間幅とともに
縮むとは限らないためである。分割はgeometry、support、applicabilityだけで決まり、frame scheduleには依存しない。
budget内に関係を証明できなければ成功扱いにせず、対応する明示failureとする。先行wall hitが後続trialの逸脱を
救済し得るため、材料boundaryではevent探索前にproposal全体のvalidityを拒否せず、pieceごとに
`event/valid prefix → no-hitまたは残存prefixのvalidity → commit`を維持する。frameは確定済みaccepted
pieceから評価し、frame評価のために数値pathを作り直さない。

P19-Lでは、上のgeometry-drivenなaccepted piece意味論とは分離して、強い局在場に対するapplicability certificateだけを
局所化した。P19-L完了時点の`integrators.py`は一つの固定step RK4 proposalに対応する
`rk4_position_hermite_state_extension_v2`とparameter
部分区間のstate enclosureを所有し、`fields.py`がそのtubeと重なるcellのprimitive range、`physics/runtime.py`がrange上の
model boundを所有する。certificate workは不変なroot pathをdyadicに制限してもendpointを再積分・commitせず、元proposalが
全区間で証明された時だけその元endpointを受理する。したがって固定stepの時間離散化、frame schedule、event順序を変えない。

局所rangeは一row最大64 cellで、overflowなら同じ`solver.event.max_refinements` budget内で区間を分割する。memory plan
v13はdense path 176 B/row、split budget 2のinterval stack 72 B/row、candidate arena最大544 B/rowを名前付きcomponentへ
計上する。global certificateを先に試すため一様caseへ常時local探索を課さない。4,096粒子×20 stepの公開API観測はglobal/local
`0.9276381/0.9345506 s`、比`1.0074517`、科学payload bitwise一致であり、machine-localな非gating証拠である。
代表Case-A profileでは局所certificate全rowが分割なしで閉じ、global support enclosureに連動するevent refinementが支配した。
同じgeneric chord算術のbatch化だけを残してbitwise identityを維持した。support証明とevent refinementの結合を変えるなら、
P19-Lへ条件分岐を重ねず、同じfirst-hit/support意味論を解析・回帰できる別の数値work packageとする。

actual stage/sampleまたは局所rangeが適用域外なら`model_applicability`、有限なcertificate budgetを使い切っても真偽を
証明できない場合は`indeterminate_applicability_certificate`とする。証明不能を物理違反へ畳み込まず、反対にgateを
無効化して成功扱いにもしない。一般interval framework、第二engine、比較case固有の閾値は導入しない。

P06 revision 2では一般enclosureに先行して、dragなし・厳密一様fieldから証明した一定加速度だけを解析放物線として
扱う。topology-completeな材料boundaryではfirst hitが流体domainからの退出を捕捉する。boundaryなしでは、全cell
supportedな`RegularLayout`の矩形support boxに対し、各座標の端点と内部極値をproposalごとに解析評価する。
材料boundaryの候補探索はendpoint chordを`|a|h^2/8`で膨張して漏れを防ぎ、facetとの二次根がfloat64 budget内で
決定できない場合は明示的に失敗する。一般非一様場・dragはrevision 3aで上記boundaryless regular
support/applicability enclosureを完成した。revision 3bで材料boundary用tube・sequential accepted pieces・
fail-closed判定を接続した。P06-Uでは、exact meshかつ全cell supportedなP1/Q1とtopology-completeな
材料boundaryの組合せを解禁した。boundaryless unstructured supportとRZは、それぞれの包含・basis規則が
完成するまでprepareで拒否する。
一定係数の
exponential strategyは局所解析pathを使い、どちらもhit時刻の状態は同じstrategyで再評価する。

V&V用にdebug modeでは、指定した少数粒子・stepだけ次を出力できるようにする。

- 各stageの`t,x,v,Z`
- sampled fields
- force/charge derivative
- 合成increment

全粒子全stageの常時出力はしない。

### 10.3 native deterministic profile

小粒子ではdrag relaxation timeが非常に短く、陽的methodは安定性のため極小stepを要求する。
局所的に

```text
dv/dt = (u-v)/tau + a
```

とみなせる区間は指数関数で更新する。`E=exp(-h/tau)`、`A=1-E`として

```text
v1 = u + E (v0-u) + tau A a
x1 = x0 + u h + tau A (v0-u) + tau {h-tau A} a
```

を使い、`A`は`-expm1(-h/tau)`、小さい`h/tau`の残差係数は級数で評価する。start係数の半step
predictorでmidpointを作り、そこでfield、charge、drag、加算加速度を評価する。非線形dragをこの式へ
暗黙に入れない。

Stage 1は固定macro stepと別runの`h,h/2,h/4`収束を使い、精度目的のhidden adaptivityを持たない。
field knot/global source discontinuity、wall eventだけ必要な区間を分け、個別releaseは粒子別残時間workで
処理する。Stage 2以降で必要になった場合だけ、step-doublingを用いた2の冪`bounded_dyadic`
controllerへ拡張する。

将来のcontrollerでも次を別に扱う。

- accuracy reject：dtを減らし再試行。
- event refinement：境界時刻の局在。
- support uncertainty：field/geometryの安全性。
- output sampling：accepted pathから評価し、保存要求のためにstepを分けない。

これらを一つの「substep split回数」に畳み込まない。位置[m]、速度[m/s]、電荷[個またはC]の
scale別許容差を設定し、固定の内部定数へ隠さない。

### 10.4 収束と誤差budget

methodごとに、解析解または十分細かいreferenceに対して観測収束次数を確認する。製品caseの
許容値は「COMSOLとの差を小さく見せる値」ではなく、次の和として設定する。

```text
総比較budget
  = COMSOL referenceの時間離散誤差
  + 場export/補間誤差
  + 本solverの時間離散誤差
  + event局在誤差
  + stochastic sampling uncertainty
```

dt半減で差がどう変わるかを必ず記録し、誤差plateauが場補間由来か積分由来かを分ける。

---

## 11. 実行engineと高速化

### 11.1 CPUを第一backendにする理由

初期製品はNumPy＋Numba CPU backendを推奨する。理由は、複雑なwall eventとactive/inactiveの
分岐、unstructured mesh locationとBVH query、float64、1回限りのcase準備に適し、既存Python利用者へ
配布しやすいためである。JAX/GPUは規則格子・分岐の少ない経路では強いが、最初から全機能の
共通backendにするとcompile shapeとcontrol flowが設計を支配する。

### 11.2 データ配置

粒子状態はArray of StructuresではなくStructure of Arraysとする。

```text
x_r[N], x_z[N], v_r[N], v_z[N], charge[N], mass[N], ...
cell_id[N], lifecycle[N], rng_counter[N]
```

- hot arraysは連続、dtypeをplanで固定。
- 物理SoAのparticle ID対応を固定し、固定容量の`active_particle_index`だけをin-placeでstable compact。
- v0.1は全粒子stateをresidentに置き、memory planからbounded tile slab幅を決める。全Nのstage/
  sampled-field配列やouter particle batchは、必要性が実測されるまで作らない。
- P09では未使用だったper-layout hintをP10で、実際に消費するP1/Q1のunique layout分だけ追加する。
  trial stageはscratchとして扱い、accepted endpointだけをresident hintへcommitする。regular layoutはhintを持たない。
- engine v7以降の候補粒子chunkはresident stateを分割するouter/out-of-core batchではなく、
  各粒子の次のrefinement proposal行だけを有界にまとめるscratch work partitionである。
- 一つのcompiled passは各粒子行を一度だけ更新し、field/geometryはread-onlyで共有。
- inactive粒子は毎step全N maskを作らず、active IDを管理。
- hit後もactiveな粒子はflat SoA work queueの次roundへ残時間を戻す。行ごとのtarget timeを持ち、
  release/hit/splitで異なる区間をPython粒子loopへ戻さない。
- event/failureは各roundで一行あたり高々一件を固定columnar slotへ書き、stable prefix/compactionで出力と次roundを作る。
- boundary応答とcounter RNGは各ownerのcompiled leafを呼び、slab分割順に依存しない。
- engineのmacro barrierだけがwriterへcolumnar batchを渡す。
- release scheduleは時刻順source batchとしてactivateし、cohortごとに全粒子maskを走査しない。

通常設定はslab幅でなく`memory_limit_mb`だけを公開する。resident state、field snapshot、
bounded slab、event/output、safety marginが上限内になるようprepareでslab幅を決める。
P09の上限はsolverが所有する配列の予測peakであり、Python/native libraryまで含むOS hard RSS capではない。
load/prepare/runのphase peakとcomponent内訳は`run.json`に残し、process RSSは外部performance scriptで別に測る。
P12のruntime layout v3とP13のmemory plan v4がworker数倍のproposal/query/output stagingを数えるのは履歴である。
P14-Pは一つのslabへstage、proposal、work queue、event/failure columnを明示して再利用する。最小slabが
収まらなければ開始前に拒否する。
memory plan v11はdeferred event depthを `event_work_bytes_per_particle = 24 * (max_refinements + 1)` として解決し、
三つのint64列に対応する`slab_event_work`へ独立計上する。depth依存容量を一般proposal scratchへ隠さない。
候補、event/failure staging、surface release、direct replayはnamed componentへ分離し、pack時だけのgatherは
12.5% safety marginが所有する。正確なbyte式とcapacity規則は
[`solver/docs/parallel_execution_plan.md`](solver/docs/parallel_execution_plan.md)を権威とし、容量超過のCSRを先に確保しない。
`max_interactions_per_step`へ達した場合は、残区間に次のhitが存在する時だけ、そのresidual-work intervalの
残時間を二分して再試行する。event-freeなら追加depthなしで残区間を受理する。子intervalではinteraction
countを0へ戻し、refinement depthは引き継ぐ。明示したdepth budgetを超えた
場合だけnumerical failureにする。

deterministic CPUの標準はfastmathを無効にし、model加算順、slab順、aggregate reduction順を固定する。
GPUとのbitwise一致は求めず、trajectory tolerance、event outcome、RNG identity、ensemble統計で判定する。
P10のcompiled passとP12のouter ThreadPoolは履歴である。P14-Pの実測後は`parallel=False`のsingle-thread compiled
engineを唯一のcompute runtimeとする。thread ID、thread-local可変containerへ依存しない。行ownership、
prefix/compaction順、commit順はengineと数値ownerが一度だけ所有する。
Numbaは0.67系、NumPyは`<2.6`をlockする。

### 11.3 kernel fusionと拡張性

`simulate`の内部準備は有効物理から必要fieldの最小集合を求める。1回のelement lookupで全quantityを
samplingし、同じtile内でcharge derivative、forces、integration stageを計算する。力ごとに
meshを再探索しない。

初期は選択modelごとのcompiled array passと、決定論／OUという少数のstep kernelだけを持つ。
profileで配列往復が支配的と確認できた組合せだけを融合し、物理組合せごとのkernelを先回りして
増殖させない。複数FieldLayoutがある場合、lookup共有は同じlayout group内に限定する。

### 11.4 JAX/GPU

JAXを採る場合の条件を明示する。

- 64-bitを明示的に有効化する。既定float32のままCOMSOL比較をしない。
- static shapeを保つためtile固定、`jit`内のPython分岐を避ける。
- key splittingだけでなく粒子identityに基づくcounter RNGを維持する。
- compile時間とcacheをcold/warm双方で測る。
- wall eventが少ないregular grid caseを最初の対象にする。

GPU backendはCPUと同じ`PreparedRun`と結果契約を実装するが、bitwise一致を一般要求にしない。
float64 trajectory誤差、event順序、RNG identity、統計分布でbackend parityを判定する。

### 11.5 memoryと出力

1,000,000粒子の現在状態は、必要なfieldを絞れば概ね150～400 MBで保持可能である。一方、
全粒子×全保存時刻×多数列をmemoryに保持する設計は成立しない。例えば73個のfloat64量を
1,001時刻保存すると約584 GBである。

標準出力を次の4系統に分ける。

1. `final`：粒子ごとの最終状態。
2. `events`：全boundary/release/failure event。
3. `probes`：選択粒子または層別sampleの時系列。
4. `series`：onlineの整数状態数とwall別weight/flux。

v0.1はJSON manifestとHDF5だけを使う。P05で導入した一segmentのlogical datasetは、P13でも
release/boundary/failure event、series、frame/probe、finalの意味を変えない。P13はphysical layoutを固定64 macro-stepごとの
closed segmentとA/B checkpointへ拡張し、`LATEST`を唯一のepoch commit pointにした。result v5は
固定64をwork-scaled cadenceへ置換し、現行result v6はresult schema 3で有限接触とperiodic transferを
明示列として保存する。checkpoint schemaは2を維持する。

trajectory frameはboundary eventに対して右連続とする。hit時刻ちょうどのstickは速度0の
post-event状態、holdはhit位置・hit時速度・hit時電荷を保持したinactiveなheld状態、escapeは行なしとし、
escaped粒子は以後のframeからも除外する。finalは全粒子のrun終了時刻lifecycle snapshotであり、heldは
`kinematics_valid=1`、escapedは`kinematics_valid=0`をlogical nullとする。保存する
有限な最後hit payloadは非権威値であり、最後の有効状態はboundary eventが所有する。

P13は同じlogical event/frame/final schemaのまま、閉じた`epoch-XXXX.h5`へevents/frames/series/probesをchunked
streamし、一時名からrename後に`LATEST`を更新する。二世代checkpointは出力時刻とは独立した
accepted macro-step上のepoch barrierでSoA、
active IDs、source cursor、RNG/refinement counter、event ordinal、commit IDを保存する。完了時だけ
final HDF5、`run.json.status=complete`、`_SUCCESS`を最後にcommitする。Parquet/CSV exportと分位・
occupancyは外部analysisが必要時に作る。完了時に`OUT.partial`を要求された`OUT`へ同一volume内で
renameする。`run`中にsnapshot listを溜めて最後にstackしない。

P13のepoch commit順は`segment temp→close→rename`、`inactive checkpoint temp→close→replace`、
`LATEST temp→replace`とし、`LATEST`を唯一のcommit pointにする。crash後はLATESTより新しいorphanを
無視し、checkpoint記録済みevent/frame countから再開する。最終化は`final.h5→complete run.json→
_SUCCESS→directory rename`の順である。最初の`LATEST`前とこれら全境界にfailure injection testを置き、確率wallの
physical RNG ordinalを含むstate identityを検査する。参照checkpointと最新segmentはhash、全segmentは構造と累積countを
検査する。過去segmentの同shape値改変は検出契約外である。

P12/P13のworker-local mergeは履歴であり、P14-Pではparallel roundの固定columnar rowをstable prefixで詰めて
engineへ返す。P13のbackground queueはcommand直後のackによりcompute/I/Oを重ねなかったため削除し、現行は
main threadの同期single-owner writeだけを残す。compute threadはfileへ直接書かず、eventを捨てない。将来の
output-rich profileが同期write比10%以上の改善余地を示した時だけ固定二buffer・一件in-flightを再検討する。
boundary identityは`(particle_id,event_ordinal)`、公開時のcanonical順は
`(time_s,particle_id,event_ordinal)`であり、thread順やrow到着順へ依存しない。output planは
計算開始前にmemory/disk容量を表示し、budget超過を拒否する。

### 11.6 性能計測

release gateには次を含める。

- 10^4、10^5、10^6粒子
- 定常regular、定常unstructured、壁event多発の3case
- 基本drag、製品標準の全決定論力、charge連成、Brownianの各profile
- single-thread warm repeatと粒子数・workload family別scaling
- cold compile、warm simulation、I/O込みend-to-end
- peak RSS、bytes/particle、trajectory出力速度
- field sampling、force、integration、event、outputの時間比
- 0/1/5/20 hit per particleでprocessed step/event数に対するscaling
- 予測RSSと実測peak RSS、frame数を増やした時のRSS不変性
- slab幅変更時のparticle/event/RNG identity
- output none/sample/allがfinal/event結果を変えないこと

P10のmanual harnessはこのrelease gate全体ではなく、field-heavy/event-lightと小さいevent-heavyの二caseで
cold/warm JIT、semantic digest、RSS、revision、speedupを記録する。絶対thresholdは置かない。10k/100k/1M、
各field/event/output/thread軸のsynthetic判断はP14が所有し、23行×3観測matrixで完了した。unstructured行列にはrealistic cell count、
initial localization、cross-cell motionを含める。read-only synthetic local profileではP1 stripの1000 warm sampleで、
hintなしfull searchが100/500/1000/5000 cellで0.026/0.131/0.249/1.271 s、正しいstrict-interior hintが
約0.0005 s（51x～2576x）だったため、P10はcompiled baseline完了であってlarge-mesh P1/Q1性能完了ではない。

P12の512粒子×4 macro step warm実測（3回median）では、編集前serial baseline 3.066786 sに対して現行
1/2/4 threadが0.440460/0.528229/0.632361 sだった。1 threadはbaseline比6.96倍だが、
T1/T2=0.83384、T1/T4=0.69653である。workは8,704/19,968/11,264 accepted piece/query/refinement、
最大深さ21で、全9観測のpayload digestは一致した。threads 1/2/4の科学出力はbitwise一致する。
compiled BVH query、保守的RK4 clear/split事前認証、同時刻wall prefix batchは採用するが、正のthread scalingは
まだ確認できなかった。P12の出口はparallel ownershipとstable mergeの正しさである。P14ではregular 100k/1Mで
3観測medianで20 workerがregular 100k/1Mに1.8796x/4.7965x、P1/Q1 10k crossに
1.1737x/1.1473x、event 10k×20 hitに0.8882xだった。
このためP14のouter ThreadPoolは移行前の履歴実装とし、P14-P closeoutのv27で廃止した。P14-Pは
field/physics/integratorのpreallocated `into` pass、stackless boundary BVH、linear/quadratic exactのrow-target
residual/event batch、一般曲線flat SoA state、compiled boundary/RNG、row numerical status、batch surface release、
direct columnar replay、bounded event/failure stagingまで統合した。focused correction後もregular 1Mの4-thread
speedupは0.923x、1-threadはv20比23.7%退行したため、case schema v2からthread設定を削除し直列runtimeへ収束した。
P14-Pはmultithread能力を削除し、single-thread production engineへ一本化して完了した。
P14-Uはこの単一serial runtimeだけをtarget-use caseで評価して完了した。時間収束は固定空間mesh、mesh収束は
`nx=ny`固定aspectとlayout別fine referenceで責務を分ける。10k/100k/1Mのnone/sampleを各3 fresh processで
測り、別の1M非計時profileを使う。mode間core／mode内probe payload、event work、target hit、RZ actual-field parityをhard gateとし、
production coreへtimer、第二scheduler、診断frameworkを追加しなかった。正式成果物は18 raw観測＋6 medianを
完備し、profile self-timeはevents 28.7%、fields 26.8%ほかへ分散したため、単一owner最適化も追加していない。

COMSOL比較は同一machineでなくてもよいが、hardware、COMSOL設定、solver刻み、保存量を明記する。
物理や出力を減らした計測を「COMSOLより高速」の根拠にしない。

---

## 12. 結果、可観測性、障害

### 12.1 最小限のrun manifest

各実行は次を必ず残す。

- case/config/content hash、code version、backend、hardware概要
- 展開済みphysics/integrator/output plan
- seedとRNG algorithm version
- 展開済みevent設定、boundary law/priority、algorithm revision
- 粒子数、step/event/reject数、状態別最終数
- refinement深さ・interaction数と、hit/no-hit/indeterminate/failure reasonの集計
- wall-clock内訳、peak memory、出力bytes
- accuracy/integrity warning count

これはデバッグdumpではなく、結果の再現性に必要なprovenanceである。

### 12.2 failure分類

```text
InputError       schema/unit/hash/local connectivity/reference/parameter
CapabilityError  未対応の座標×物理×backend組合せ
PhysicsError     モデル適用域外、非有限derivative
GeometryError    point location/eventが決定不能
FieldError       spatial/temporal support外
NumericalError   reject継続、非収束、accuracy budget
ResourceError    memory/disk/backend resource
```

粒子単位で継続できる障害とrun全体を無効にする障害をmanifestで明示する。`NaN`を出力して成功扱い
することも、任意epsilonへclampして続行することも避ける。escaped後の座標・速度は
`kinematics_valid=0`のlogical nullとし、有限だが非権威なpayloadと計算失敗のNaNを明確に区別する。
P08では、粒子に局在できるevent budget、不定event/boundary/departure、field support、model applicability、
nonfinite physicsを小さいfailure reasonへ変換し、他粒子を継続する。入力/schema、静的・共有model/field bound、
topology、writer/publicationなどrun全体の意味を失う問題はrun-fatalな`SimulationError`とする。

### 12.3 診断の範囲

coreはcounterと指定probeだけを出す。原因分析、COMSOL列との照合、グラフ、decision treeは
`tools/vv/comsol`が担当する。常設の巨大な診断frameworkをsolverへ組み込まない。

### 12.4 analysisとvisualization

`open_result`はsegmentを隠したlazy `ResultView`を返し、particle ID、時刻、event typeで選択読込みする。
`tools/analysis`はfate、source-to-target、deposition、arrival time、impact energy/angle、residence、
ensemble confidence intervalを計算する。`tools/visualization`はResultViewまたはanalysis tableだけを
描画し、solver private stateやCOMSOL adapterをimportしない。

---

## 13. V&V戦略の要約

詳細手順は [vv_methodology.md](vv_methodology.md) に分離する。判定順序は固定する。

1. 入力・単位・topology・provenance
2. COMSOL参照軌道上のfield sampling
3. 同じ`x,v,Z,t`で各forceとcharge derivative
4. Brownian offの1 stepとRK stage
5. wallなしの全時間軌道
6. 最初および複数のwall event
7. 決定論ensemble
8. stochastic ensemble
9. dt/mesh/cache収束
10. 製品規模performance

ある層が不合格なら、その先の総合軌道を調整して直そうとしない。たとえばfieldが違うのに
drag coefficientを変更して最終位置を合わせる行為を禁止する。

### 13.1 軌道指標

- 共通時刻へ補間した位置・速度・電荷・各forceの誤差時系列
- L∞、time-weighted L2、p50/p90/p99、最大値
- chamber scale、局所mesh size、粒子の移動距離で正規化した誤差
- 最初に閾値を超えた時刻と、その直前のoperator差
- active区間、event前後、終了後を分けた評価

最終位置・最終statusは補助指標であり、主判定ではない。

### 13.2 event指標

- hit/no-hit、boundary ID、outcomeの混同行列
- event ordinalごとの時刻差・位置差
- pre/post velocity、normal、charge
- event順序の一致
- stick/escape/reflection率と到達時間分布

COMSOL保存時刻間にeventがある場合、保存行の初回status変化ではなくevent時刻を使う。

### 13.3 stochastic指標

同一seedのparticle-wise一致は、乱数増分とstepが一致するときだけ要求する。それ以外は同一点
多数replicaと複数seedで、平均、共分散、MSD、分位、occupancy、survival、wall別到達確率、
Wasserstein/energy distanceを信頼区間付きで比較する。

### 13.4 新たに必要なbenchmark

これは対応stageで順次追加するcatalogであり、初期必須suiteではない。初期は
`model_dataset_rebuild_plan.md`のV01～V06だけを作る。その後、現`model_dataset`へ必要に応じて次を追加する。

- Brownian offの12 deterministic case
- 10 µsごとのmicro-traceとRK4全stage（少数粒子）
- 同一点からのBrownian replicaと内部random increment replay
- uniform acceleration、linear drag、一定E、charge relaxationの解析case
- 平面stick/specular/restitution/probabilistic、corner、axis、thin obstacle、複数hit
- 真のsurface release、面積重み、発生直後departure
- escape、stick、reflectionの正例、および外部adapterのfreeze/disappear対応表
- time-dependent field、2D RZ→3D、small 3D tetra/triangle case
- P1 triangle、Q1 quad、将来P2の補間oracle

---

## 14. テスト方針と複雑化防止

### 14.1 三層だけにする

```text
tests/verification
  解析解、manufactured solution、収束次数、幾何event、保存則

tests/scenarios
  surface release、wall response、field support、charge couplingの代表case

tests/performance
  10^4/10^5/10^6、memory、throughput、scaling
```

公開API、数値結果、物理不変量、schema compatibilityを検査する。COMSOL回帰は
`tools/vv/comsol`のsuiteとしcore testへ入れない。private関数の所属、helper名、
ファイル数、re-exportの形をテストしない。coverage率を物理妥当性の代替にしない。

### 14.2 一つの不変条件に一人の所有者

- unit/quantity：case adapter
- topology：geometry builder
- capability：plan compiler
- parameter range：physics spec
- runtime finite check：kernel boundary
- result schema：result writer

同じ条件をparser、preflight、runtime、writer、compareで少しずつ再実装しない。外側は内側の
typed resultを利用する。

### 14.3 complexity budget

絶対的な行数gateではなく、review時の停止条件として使う。関数complexityの観測と上限は
[quality_tooling_plan.md](quality_tooling_plan.md) のRadon規約を正とする。

- 新規・変更functionは原則CC 10以下、数値上の理由がある場合でも15以下とする。
- file分割は行数やscoreではなく、独立した変更理由が二つあるかで判断する。
- 一つのfeature追加が5層以上のswitch文変更を要求したらplan/spec境界を直す。
- production粒子loopへPython callbackを入れない。
- 同じ物理式のCPU、GPU、reference版は、共通のgolden vectorで検証する。
- 新経路を足す際は旧経路の統合・削除方針を同じPRに含める。
- 1 PRは一つの数値判断。小さなADRへ「なぜ」を残す。
- 古いbenchmarkや失敗caseを削除してgreenにしない。

### 14.4 設計判断の記録

ADRを機能ごとに増やさず、主仕様のdecision tableに理由を集約する。外部形式の破壊的変更、
物理結果を変える式変更、backend間の再現性規則など、後から戻しにくい判断だけ短い記録を残す。

---

## 15. 開発ロードマップと完了条件

### Phase 0：主仕様と解析解microcaseを固定する

成果物：

- 製品仕様、physics model revision、canonical case最小schema
- ballistic、linear drag、uniform E、first-hitの解析解
- surface release、stick、escape、specular、確率stickの小型case
- `model_dataset`再抽出仕様とknown-issue register

完了条件：COMSOLなしで基本式、補間、境界、収束を検証でき、外部datasetの不備がcore実装を
左右しない。

### Phase 1A：決定論2D RZ vertical slice

成果物：

- cartesian XY解析microcaseと単純geometry
- P1 triangle/Q1 quad field sampler
- table/source release
- Epstein/Stokes–Cunningham drag、electric、gravity、fixed charge
- 一般固定RK4
- stick、escape、specular、確率stickを含むfirst-hit
- scalar verification oracle

完了条件：解析解、step/mesh収束、surface releaseからwall eventまでがbudget内。COMSOL一致は
完了条件にしない。

### Phase 1B：高速決定論製品経路

- exponential midpoint fixed-step（P11完了）とNumba CPU batch（P10完了）
- event-heavy residual-work schedulerとstable parallel merge（P12完了）
- segmented HDF5、checkpoint/resume、bounded writer queue、lazy ResultView（P13完了）
- memory planner（P09完了）と10^4/10^5/10^6 benchmark
- P14-Pでouter worker-waveと内部parallel試行を削除しsingle-thread compiled runtimeへ収束（完了）

完了条件：主要物理がcompiled batch経路を通り、slab/output設定で数値意味論が変わらず、
memoryとsynthetic end-to-end scalingを説明できる。

### Phase 1U：代表用途closure（完了）

- surface release、非一様場、材料wall、多数macro stepを一つのcaseで結合
- `h,h/2,h/4`とregular/P1/Q1 mesh系列の軌道・hit収束
- global enclosure、refinement/failure/candidate量、serial end-to-end時間、RSS/outputの同時評価
- profileが示したownerだけを変更し、常設diagnosticまたは別runtimeを作らない

完了条件：現行model範囲で、目的用途のtrajectory精度とevent精度、入力表現、end-to-end時間・memoryを
同じcaseから説明できる。P14-U正式releaseはこの条件を満たした。秒数はmachine-localな記述値であり、
COMSOL比較またはportable thresholdではない。T03とP14-Rのlocal gate/evidenceに加え、remote Windows/Linux
workflowも完了した。Phase 2AのP15は旧着手blockerを解除して完了した。

### Phase 2A：帯電、熱、決定論3D粒子

独立した変更理由を一つのreleaseへ束ねない。M3-V外部評価は完了し、全体を一列にした候補順ではなく
field production、trajectory physics、state dimensionの三workstreamを採用する。

1. P15（完了）：`oml_stationary_maxwellian_debye_huckel_v1`のRK4-first continuous charge。
   `physics/charge.py`が式・finite invariant/rate/derivative bound、catalog/runtimeがrequired fieldと三つの
   capability、既存integratorが`(x,v,Z)`更新、engineは配線だけを所有する。RK4受入後にexplicit midpointも完了した。
2. M3-V（完了）：直接MPH inventory、12 package構造、保存時刻全体のvariant感度、charge/Epstein式parity、
   sampled applicabilityと力relevanceを評価した。元の12 package全軌道は未実装物理と適用域のため
   `NOT_APPLICABLE`。後続の別成果物exact-P1 pre-event companionは限定的な時間離散parityにPASSした。
3. field production（完了）：F01で独立reduced electrostatic builderをcanonical outputへ接続し、F02でprovider
   adapterのmixed-mesh P1化、代表case統合、同一node上の外部field V&Vまで完了した。
4. trajectory physics：species制約付きrelative-drift charge、P15-E有限速度drag、P15-F collisionless Barnes ion drag、
   P16 Waldmann--Gallis熱泳動、P18-D球形準静的DEP、P18-L RZ rarefied-vorticity lift、P18-R effective-gas
   drag/thermophoresis sensitivity、Brownian B01/B02まで完了した。
   P17はstate dimensionとして独立する。

各縦切りは解析解・収束・compiled parity・代表scenarioを個別に満たす。Phase全体の完了条件はcharge relaxationと
coupled trajectory、熱泳動modelの適用域、方位回転共変性、回転面hitがbudget内であることとする。relevance gateが
有限速度Epstein、collisionless Barnes ion drag、Brownianを含め、評価が示した優先度をowner文書と同じchangeで更新する。

### Phase 2B：確率過程

- counter RNG、joint OU/Langevin、conditional split（B01数値基盤、B02 production接続とも完了）
- Cartesian XY・Epstein linear drag-only・fixed-charge stateの固定depth dyadic node＋cubic Hermite leafによる
  平面wall first-passageと一般曲面の数値crossing、およびXY/RZ active wall fresh-root restart（完了）
- ensemble validation、first-passage depth安定化、row-local数値failure（完了）

完了条件：OU平均・joint covariance・MSD・wall first-passageが解析解と統計budget内。

### Phase 3：追加プラズマ力と外部V&V

成果物：

- M3-C0：監査済みMPH copyから、全式・parameter・derived-field意味、ion-drag以外が一致した12個のBrownian-off
  companion、admissibleな`h, h/2, h/4`系列、frozen RHS/RK情報、正例Freeze/Disappear、multi-seed入力を再構築
- P18-C（完了）：aggregate reference chargeを既存continuous-state passのoptional revisionとして追加
- P18-I（完了）：二つのaggregate ion-drag sensitivityを既存の排他的`ion_drag` categoryと単一compiled passへ追加
- P18-D（完了）：producer提供`grad(mean_E_squared)`を使う球形準静的DEPを既存の`dielectrophoresis` categoryと
  単一compiled passへ追加
- P18-L（完了）：producer提供のsigned方位vorticityを使うRZ/no-swirl rarefied-vorticity lift sensitivityを、
  速度依存bound callbackとexponential enclosure v3を含む同じ責務境界へ追加。cross-representation比較はFAILで、
  Case-A 100 nmのcommon-P1 event前same-field診断は後続M3-C1でPASS
- P18-R（完了）：既存式ownerを再利用するeffective-gas linear Epsteinとeffective-gas heat-flux thermophoresisの
  optional sensitivity revisionを追加。両者は`maximum_speed_ratio<=1`と`lambda/a>=10`を連続pathで認証し、
  producer-owned pseudogas closureをspecies-resolved truthとは呼ばない
- M3-C1：Case-A 100 nmのpre-event frozen saved-state producer-form replayは最小PPR補足後8/8完了。integrated candidateは
  exported exact-connectivity P1、参照はCOMSOL native fieldである。3 runは287粒子×46 frame、event/failureなしで完了したが、
  cross-representationの位置・速度・電荷RMS/max 6 gateはすべてFAILし、same-field agreementや物理適用性は認定しない
- P19-L（完了）：integrator-owned dense RK4 pathとlocal cell primitive rangeを使うcertificate-only restrictionで、元の固定step
  endpointを変えずにglobal envelopeの偽拒否を解消し、actual applicability違反と証明不能を分離した
- M3-C1の続き：cross-representation FAILで既定条件を満たしたため、full-physics exact-connectivity common P1を使う
  COMSOL診断を独立成果物として実行し、固定済み9 gateを全PASSした。さらに同じCase-A 100 nmを最初のwafer stickまで
  延長し、event v14 candidateに対するmaterial 20/20と再計算prefix 9/9をPASSした。認定はBrownian-off、共通canonical P1、
  最初のmaterial eventまでに限定する
- B03（完了）/M3-C2：RZ projected charge/force Brownianを一つのengineへ追加済み。common-P1 Case-A/Case-P 100 nmの
  multi-seed finalは外部V&Vとして完了し、production coreへCOMSOL経路を追加していない
- P18-H（完了）：製品利用caseにも必要なgeneric holdを単一boundary/engine経路へ統合。解析・resume・Brownianを含む
  公開回帰と、既存hash固定Freeze referenceへの外部candidate 15/15をCOMSOL再実行なしで閉じた
- F01/F02 reduced electrostatic builder/provider adapterは完了済み基盤として再利用し、common-field parityと
  exported-P1/native-field workflow comparisonを分離

P18-R完了時点ではCOMSOL studyを再実行していなかった。その後もcoreを変更せず、外部M3-C0bを段階実行した。
v5の旧「全gate PASS」は位置relative L2の分母が絶対RZ座標で原点依存だったため無効であり、履歴上の
`CHARACTERIZED`に留める。逐次確認v6はCase-A 100 nmのpre-eventについて0.625/0.3125/0.15625 us系列、各run
13,202 active recordを使い、原点不変な変位・速度・電荷のfine-pair relative L2
`3.102727085428027e-5/3.92483251084038e-5/1.3511393490811483e-6`と観測次数
`0.9041136/0.944312/1.123838`で運用上の刻み選択だけをPASSした。これはsolver agreementまたは普遍的なaccuracy
claimではない。その後のM3-C1では、最小PPR熱泳動補足を一度だけ実行し、13,202 saved row・座標完全一致・
`4.4046499933294035e-16` component normalized residual・`1.0992667494449471e-16` global relative L2で
frozen producer-form 8/8を閉じた。その後P19-Lを完了し、global certificateが不十分な局所pathを同じfixed-step endpointの
dense/local certificateで判定できるようにした。exported-P1/native-field比較は完了し、COMSOL referenceのpre-event運用自己収束PASSと
candidateのfloat64-floor内macro-step partition stabilityを確認した一方、空間表現をまたぐ6 gateはすべてFAILした。旧三つの
candidate runがglobal safety enclosure由来の人工的なevent subdivisionで同じaccepted-piece countへ細分されたため、この
安定性を独立な時間収束またはRK4次数とは呼ばない。trigger済みだった
full-physics exact-connectivity common-field COMSOL診断も完了した。287粒子×46 frameに対し、位置・速度・電荷の
RMS/max/relative L2の事前登録9 gateを全PASSし、この限定sliceのsame-field agreementを閉じた。native-field等価性、物理妥当性、
boundary、Brownian、30 ms、他case・粒径・variantは未認定である。

material-event candidate v3はevent v14で0--458.75 usを実行し、最初のwafer stickについてmaterial gate 20/20と
0--450 us prefix 9/9をPASSした。event時刻・hit位置・terminal電荷の絶対差は
`2.157542807607049e-13 s`、`2.683964162031316e-14 m`、`4.7283812421028415e-9 e`である。v13からv14で
query/refinement/accepted/depthは`16,427,517/7,792,306/8,635,211/16`から
`842,927/11/842,916/11`へ減り、failure 0を維持した。v13 refinementの`97.8331703092769%`は450 us以前に
発生しており、原因はglobal absolute safety enclosureをevent BVH boundにも流用したことだった。operator観測wall time
約`853 s→36.5 s`（約`23.4x`）はmachine-localな非gating値である。

v14のsolver-only 3刻み再実行はrefinement 0で、query=acceptedが`206,927/413,567/826,847`だった。位置・速度・電荷の
RMS観測次数は`2.029875353701904/2.0816971911764033/2.044084026475049`、fine relative L2は
`6.099791486063973e-8/8.321356016032579e-8/1.3796067752988052e-8`で、全量`ORDER_EVALUATED`の自己収束PASSである。
このfine candidateと既存common-P1 referenceの現行comparisonも9/9をPASSした。COMSOL referenceとcommon-field入力は
不変なので再実行せず、既存hash固定referenceを再評価した。P18-Hも同じ方針で既存Freeze referenceへの15/15 gateを
閉じた。B03 core、charge-stable coupling、work-scaled cadence、P20 performance closeoutも完了した。続くmeaning-matched
external V&V/M3-C2はcommon-P1 Case-A/Case-P 100 nm finalまで完了した。Case-PのR-Z/fate gateはPASSしたが、anchorは
negative-ion currentを選択しない固定二電流same-form比較であり、Case-Pプラズマ全体のphysical applicabilityは未認定である。
任意の三電流production revisionは後続P21で完了した。外部三電流軌道比較の5 primitive入力authorityも
producer companionのpriority 3で解消した。priority 4の共有three-current初期電荷を用いたcommon-P1、Brownian-off、100 nm、287粒子、
30 msの独立V&Vは、candidateと明示drag COMSOL referenceの各3刻み収束、全frameのlifecycle/finite mask exact、
共通の有限lifecycle stateの`r,z,Z`、共通active stateの`v`、141件のevent/fate identityをすべてPASSした。
これは物理model validationまたは普遍的COMSOL同等性の認定ではない。
受理済み3 seedの287粒子owner discoveryも科学payload・work・
case identity・revisionを完全一致させて完了した。支配ownerは`integrators`（自己時間比42.58--42.86%）だったが、
事前登録済みbounded ownerではないため最適化は未承認でproduction変更はない。M3-C2A anchorは
`CLOSED_ACCEPTED_WITH_LIMITATIONS`である。10,000粒子以上の性能は製品SLAを先に定義した独立work packageだけで評価し、
COMSOL fittingは行わない。
後続の明示指示によるbounded maintenanceでは、既存event/integrator責務を変えず、chord batchのPython scalar調停を
同じsingle-thread compiled ownerへ統合し、重複式を削除した。accepted 3 seedの科学payload/work/revisionは完全一致し、
end-to-end中央値は12.29%短縮した。新しいbackend、scheduler、診断層、公開設定はなく、追加ownerへ進まず終了した。
authorityは[`solver/evidence/m3c2/caseP_100nm_chord_optimization_v1/`](solver/evidence/m3c2/caseP_100nm_chord_optimization_v1/README.md)である。
既存M3-C1 compact
evidence/evaluatorはevent v14の履歴として固定し、v15へ書き換えない。
旧M3-Vの固定電荷・共通3力runnerはこの動的電荷・全力診断を代替しない。gate bypassを使わず、COMSOLは引き続きcore
dependencyではない。

将来の12-package coverage拡張条件：formula parity、numerical trajectory、physical applicability、stochastic ensembleを
別statusで記録し、各追加claimに必要なdeterministic companionとBrownian-on gateを独立work packageで完了する。
これはM3-C2A anchorの完了条件ではない。外部比較処理がcoreを公開API経由でimportする一方向依存を守り、比較のために
coreの適用域や安全gateを緩めない。

### Phase 4：時間依存場と完全3D

成果物：

- fixed-topology linear time field（完了）
- tet4/tri3 full 3D（未着手、時間依存fieldと別release）
- product-size CPU benchmark

時間依存と3Dは同じreleaseで同時に導入しない。

### Phase 5：GPUと製品化

- regular/static caseからGPU backendを追加
- installation、version migration、error guide、feature matrix
- artifact schemaの後方互換policy
- releaseごとのV&V/performance report

完了条件：CPUとの差、対応外feature、hardware条件を公開し、GUIなしでもCLI/APIで全工程が再現可能。

---

## 16. 最初の90日で実施する具体項目

### 0～30日

1. 新repoまたは新top-level packageを作る。
2. canonical YAML/HDF5とprimitive quantityを決める。
3. 解析caseとboundary microcaseを作る。
4. 外部COMSOL再抽出仕様を作るが、core実装の前提にしない。

### 31～60日

1. P1/Q1 sampler、cell ID保持、scalar RK4を実装。
2. drag、electric、gravity、fixed chargeを実装。
3. surface source、first-hit、stick/escape/specularを実装。
4. exponential midpointとNumba SoAへ移し、scalarとの差を確認する。

### 61～90日

1. 10^4/10^5/10^6のCPU benchmarkを取り、field/event/I/O別に測定する。
2. segmented output、checkpoint、memory planを実装し、failure injectionで再開を確認する。
3. COMSOL側でprimitive field、Brownian-off、event ledgerを再出力する。
4. v0.1 feature matrix、known limits、次phaseのgo/no-goを決める。

この順序なら「多くの物理が設定できるが、どこが違うか分からない」状態を避けられる。

---

## 17. 採否を決める主要設計判断

| 論点 | 採用 | 採らない案 | 理由 |
|---|---|---|---|
| 実装開始 | clean-room package | 現runtimeを整理 | 正しさと性能経路が絡み、benchmark未対応 |
| 場の権威 | mesh-native primitive | 粒径別derived CSV | 派生値の陳腐化を防ぐ |
| 補間 | shape function | solid越しDelaunay | topologyと境界を保持 |
| 四辺形 | Q1 | 暗黙tri分割 | COMSOL補間意味を変えない |
| 境界 | continuous first event | step終点inside判定 | thin-wall飛越しを防ぐ |
| 状態 | physical/safety/accuracy分離 | 単一stop code | 原因と物理結果を混ぜない |
| COMSOL比較 | 外部V&Vが通常の`rk4_fixed`設定を指定 | COMSOL名のcore profile | 方法差を物理差に見せない |
| 高速化 | fused CPU batch first | 全機能JAX first | event/unstructured/float64に適合 |
| 乱数 | counter-based | global sequential RNG | 並列順序に依存しない |
| 出力 | streaming + events | 全N×T memory | 10^6粒子へ拡張可能 |
| 比較 | operator→trajectory→event | 最終座標中心 | 最初の原因を特定できる |
| テスト | 数値・物理・公開契約 | private tree shape | 再設計可能性を保つ |

---

## 18. 技術根拠と一次資料

本設計では、COMSOL固有の挙動は公式documentation、数値実装は一次資料または公式library文書を
根拠にする。主要な参照先は以下である。

- COMSOLの対象CF4/O2 etching model：
  [ICP RF Bias CF4/O2 Silicon Etching](https://doc.comsol.com/6.4/doc/com.comsol.help.models.plasma.icp_rf_bias_cf4_o2_si_etching/icp_rf_bias_cf4_o2_si_etching.html)
- COMSOL Particle Tracing全体：
  [Particle Tracing Module User's Guide](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/ParticleTracingModuleUsersGuide.pdf)
- 粒子運動方程式と数値解法：
  [Theory for the Mathematical Particle Tracing Interface](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_introduction.02.05.html)
- 壁条件：
  [Wall](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.43.html)
- Brownian力：
  [Brownian Force](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.08.html)、
  [Brownian Motion model](https://doc.comsol.com/6.4/doc/com.comsol.help.models.particle.brownian_motion/brownian_motion.html)
- 希薄気体drag：
  [Drag Force](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.37.html)
- charge accumulation：
  [Charge Accumulation](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.33.html)
- COMSOL補間・結果設定：
  [Interpolation](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_ref_results.37.224.html)、
  [Mesh API](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_api_mesh.49.047.html)
- eventを持つreference ODE solver：
  [SciPy solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
- CPU JIT/parallel：
  [Numba performance tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html)、
  [Numba automatic parallelization](https://numba.readthedocs.io/en/stable/user/parallel.html)
- JAXのPRNG、dtype、制御、cache：
  [Random numbers](https://docs.jax.dev/en/latest/random-numbers.html)、
  [Default dtypes and X64](https://docs.jax.dev/en/latest/101/default_dtypes.html)、
  [Control flow](https://docs.jax.dev/en/latest/201/control-flow.html)、
  [Persistent compilation cache](https://docs.jax.dev/en/latest/501/compilation-cache.html)
- counter-based RNG：
  [Random123 official documentation](https://random123.com/releases/docs/)
- chunked/streaming storage：
  [h5py dataset documentation](https://docs.h5py.org/en/stable/high/dataset.html)

公式文書だけでは固定されないCOMSOLの乱数順序、内部stage、finite element recoveryは、推測で
coreへ実装せず、専用exportとmicrocaseで観測して外部V&V manifestへ閉じ込める。

---

## 19. 最終提案

成功条件は、機能数ではなく「差が出たとき最初の原因を一つの層へ特定できること」である。
そのために、新基盤は次の順序を守る。

1. 入力をprimitive、topology、provenanceへ正規化する。
2. 表面発生、field sampling、force、integration、first-hit、wall lawを一つの経路にする。
3. 解析解と収束で製品標準の数値方式を確立する。
4. 正しさを保った同じ経路をCPU batch化する。
5. memoryとoutputを最初から10^6粒子向けにstreaming設計する。
6. COMSOL固有処理、内部場builder、V&V、可視化をcore外に保つ。
7. 外部比較ではtime-seriesとevent sequenceを主判定にする。

この骨格が完成するまで、物理module、backend、診断helperを増やさない。狭いが端から端まで
説明可能なv0.1を先に作ることが、最終的にCOMSOLより柔軟で高速な製品へ到達する最短経路である。
