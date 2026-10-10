# P10--P14 Stage 1B closeout、P14-P serial convergence、P14-U gate

P10はStage 1Aで固定した数値意味を、同じproduction engineのcompiled array passへ移した。
P11はそのengineへnative `exponential_midpoint`を第二engineなしで追加した。P12は粒子ごとの数値順序を変えず、
event-heavy workをworker-localに分配した。P13はlogical resultの意味を変えず、durable multi-segment、
checkpoint/resume、bounded writerへ拡張した。P14は同じ公開経路の直交型performance matrixで製品規模を測り、
実測された二つのO(粒子数×cell数) locatorだけを同じ意味論のindexへ置換してStage 1Bをcloseした。
ここまでのP12 worker-waveとP14計測は履歴baselineであり、並列runtimeの完成を意味しない。closeout後の
深掘りreviewでP14-PをP14-Uより前の必須工程として追加した。

## P14 closeoutで確定したrevision（履歴）

- engine / compiled tile: `deterministic_particle_engine_v20` / `compiled_cpu_tile_v4`
- CPU runtime layout / memory plan: `resident_soa_worker_microtile_v3` / `solver_owned_memory_plan_v6`
- geometry: `line_boundary_volume_cell_bvh_v4`
- physics runtime: `deterministic_compiled_physics_runtime_v3`
- proposal / event: `coupled_fixed_step_proposal_v4` / `line_quadratic_rk4_axis_first_hit_v9`
- RK4 / exponential: `rk4_global_abs_enclosure_v1` / `exponential_midpoint_v1` /
  `exponential_midpoint_global_abs_enclosure_v1`
- field / physics catalog / result: `field_location_v3` / `deterministic_xy_rz_catalog_v3` /
  `durable_segmented_result_v3`
- checkpoint schema: version 1、固定epoch cadence: 64 macro steps
- case schema、result schema、proposal、event、physics model/result semanticsはP14で変更しない

現行revisionはengine `particle_engine_v45`、compiled tile
`compiled_cpu_tile_v21`、CPU runtime `resident_soa_serial_slab_v6`、memory plan
`solver_owned_memory_plan_v16`、proposal `coupled_fixed_step_proposal_v10`、RK4 enclosure
`rk4_global_abs_enclosure_v2`、charge-stable exponential midpoint v3 / enclosure v4、event
`line_quadratic_curved_capsule_periodic_first_hit_v21`、dense path `rk4_position_hermite_state_extension_v3`、field location v4、geometry
`line_boundary_capsule_contact_bvh_v7`である。P14の上記revisionと
性能値は置換前baselineとして残し、現行runtime名と混同しない。result algorithm v6、result schema 3、checkpoint schema 2である。
physics catalogは`inertial_langevin_2d_catalog_v23`、physics runtimeは
`signed_ion_compiled_physics_runtime_v22`、boundary algorithmは`contact_wall_laws_v7`、source algorithmは
`realized_internal_surface_contact_schedule_v5`である。canonical case/data/result schemaは3、checkpoint schemaは2である。
memory plan v16はdeferred event depthを
`event_work_bytes_per_particle`、Brownian tree、P19-L dense path・certificate workをそれぞれ別のnamed componentで解決し、
候補、event/failure staging、surface release、direct replayも分離する。pack時だけのgatherは12.5% safety marginが所有し、正確なbyte式は
[`parallel_execution_plan.md`](parallel_execution_plan.md)が所有する。
Brownian tree workは実際の候補率でなく`adaptive_max_depth`から保守的に解決する。
現行dense path v3は、root始点相対のBernstein enclosure、TwoDiff残差、world座標への方向付き
外向き丸めにより、座標原点に依存してevent certificateが閉じない不具合を修正した。endpoint、path、
v3導入時のevent algorithmとengine v32 / event v14 / RK4 global enclosure v2は不変だった。event v15は
geometry budgetとroundoff budgetを加算し、補償したfacet-local offset dotとroot-relative TwoDiff Hermite評価・包絡を使う。
event v16はvalidなRK4 dense rowのposition Bernstein control enclosureをfacet half-spaceへ射影し、4制御点すべての
外向き上限が既存budgetの負側に厳密に入る候補だけをclearする。証明不能ならsplit/fail-closedを維持する。
cubic Hermite derivative-Bernstein enclosureだけがmonotone clearへ明示opt-inし、exponential/scalarはfail-closedする。
現行event v21はevent v19 / v18の同時incident facet集合を曲線pathでも保持し、exact、RK4、exponential、Brownianの
材料facetへ共通のpriority/combined-normal意味論を適用する。static Cartesian XYの周期facetは同じfirst-event順序で
対応facetへpure translationし、材料lawやwall RNGを使わない。

P15は`oml_stationary_maxwellian_debye_huckel_v1`をRK4-firstで受け入れ、その後native exponential midpoint chargeも
受け入れた。現行RK4は`h L_Z <= 0.5`を維持し、exponential pathはmidpoint-frozen affine exponential
updateを使う。両methodはfinite invariant/rate/derivative bound、charge-aware electric/path enclosureを共有する。
case/result/checkpoint/event schemaとXY/RZ、wall、output、
checkpointの既存stateは変更していない。P14-R remote CIは2026-10-05にWindows/Linuxとも完了した。

Numba 0.67とNumPy `<2.6`をlockし、P14 closeoutまではfield location/interpolation、sample済みprimitiveからのphysics、
classical RK4の配列算術を`fastmath=False, parallel=False`で実行した。現行v45も同じ決定論的設定の
single-thread compiled runtimeである。regular fieldはsupported
containing-cell common pathだけ全cell走査を行わず、O(1)個の候補を使う。P1/Q1はaccepted endpointの
previous-cell hintを最速経路とし、hintなし/missのsupported containmentはfield所有のread-only cell AABB BVHで
候補を絞って従来の厳密predicateを適用する。共有面ownerは最小supported cell IDである。outside/masked
provisionalは物理最近傍とtie意味論を守るため従来のcompiled全cell走査へ戻り、ここは意図的にO(cell数)のままである。
この制約を含め、`field_location_v3`は旧v2 locatorを置換する一つのproduction経路である。

resident hintは物理stateではない。trial stageのhintをcommitせず、accepted full endpoint、wall/axis prefix、
residual pieceのaccepted endpointだけをresident stateへ反映する。`StepProposal.state_at()`、wall hit時刻の再積分、
hit後residual、trajectory/frame/probe/finalは同じproduction proposalとevent loopを通る。scalar evaluatorは
verification oracleとして残すがproduction fallbackにはしない。

## P10同一machineの非gating profile

trajectoryなし、same-processで1回warm-up後に3回測定したmedianは次のとおりである。before/afterは同じcase、
physics、dt、出力量で比較した。これはhardware-localな観測であり、絶対時間やspeedupの合格閾値ではない。

| profile | work | before | P10 warm | speedup |
|---|---:|---:|---:|---:|
| regular harmonic | 2048粒子 × 4 RK4 macro step | 1.99 s | 0.03058 s | 65.1x |
| Epstein C02 | 2048粒子 × 5 RK4 macro step | 3.74 s | 0.03339 s | 112.0x |
| event-heavy regular wall | 512粒子 × 4 macro steps（material-hit intervalあり） | 12.8 s | 3.14395 s | 4.07x |

event-heavy観測はaccepted particle pieces 8,704、candidate queries 20,480、refinements 11,776、
maximum refinement depth 21だった。field/physics/RK4 passに比べてevent orchestrationの改善幅が小さく、
P12のevent-heavy residual workとstable mergeが次のbottleneck候補である。P11の能力gateは先に完了した。

別のread-only synthetic local profileでは、P1 strip上の1000 sampleをwarm実行した。hintなしfull searchは
100/500/1000/5000 cellでそれぞれ0.026/0.131/0.249/1.271 s、正しいstrict-interior hintありは各条件で
約0.0005 s（51x～2576x）だった。これは非gatingな局所観測であり、realistic meshの製品性能を証明しない。
とくに初回RK4 stageは全粒子がhintなしなのでfallbackを「稀」とは扱えない。P10はcompiled baselineを完了したが、
large-mesh P1/Q1性能は未証明である。P14はrealistic cell countとinitial localization/cross-cell motionを
unstructured matrixへ含め、full searchが支配的ならcell adjacency/BVHを同じ意味論の最適化として判断する。

## 手動計測と受入境界

`tests.performance.p10_compiled`はfield-heavy/event-lightと小さいevent-heavy caseを別々に測る。各観測は
fresh processと空のNumba cacheを使い、`NUMBA_DISABLE_JIT=0`と1 threadを明示してcold JITと同一process
warm-up後を分離する。semantic SHA-256、
process RSS、solver memory plan、全algorithm revision、throughput、cold/warm比をJSONへ残し、cold/warm間の
semantic digest不一致を失敗にする。

```console
uv run --locked python -m tests.performance.p10_compiled \
  --field-particles 10000 --event-particles 32 \
  --repeats 3 --warmups 1 --json p10-compiled.json
```

この二caseはP10の移植確認であり、P14の10k/100k/1M、regular/P1/Q1、realistic unstructured cell count、
initial localization/cross-cell motion、0/1/5/20 hit、output量、thread数の全matrixを代替しない。
通常quality gateへ性能秒数を追加しない。

P10 closeoutではuv lock、Ruff、Pyrefly、import-linter、循環的複雑度gateが合格し、
verification/scenarioは278件合格した。

## P11 numerical acceptance

P11はphysics runtimeの一回のcompiled passを、合成加速度に加えてstage-local drag rate、target velocity、
加算加速度を返す形へ拡張した。`exponential_midpoint_v1`はstart half-step predictorでmidpoint係数を
評価し、その係数でdragを解析更新する。RK4の`maximum_dt_over_tau < 2.5`gateはRK4だけに残し、指数法には
適用しない。P11 closeout時点はfixed chargeだけを対象とし、非零charge rateを無視または後付け更新せず明示拒否した。

closeoutの数値受入は次を独立referenceと公開API scenarioで検査する。

- C03一定係数の位置・速度を複数step幅で丸め誤差精度に再現する。
- `h/tau`の0近傍から非常に大きい値まで有限で、zero-drag極限へ連続に退化する。
- smoothな可変係数problemで位置・速度の観測次数が1.8以上である。
- C03の要求全frameを解析解へ一致させ、output scheduleを変えてもfinal stateとevent identityが変わらない。
- Cartesian材料first hit、反射後残時間、RZ axis→wall順序がRK4と同じ`StepProposal`/event loopを通る。
- surface releaseのzero-time departure後に同じsource facetへ再衝突できる。
- Stokes--Cunningham一定primitiveでも同じ指数経路が閉形式解へ一致する。
- `state_at()`はproposal始点から同じexponential-midpoint規則で短縮再積分する。

`exponential_midpoint_global_abs_enclosure_v1`は全短縮predictor/stateを外向きに包絡し、event v9は
全短縮secantを含むvelocity enclosureから作る
`h * (v_upper - v_lower) + roundoff`のmethod-neutral chord deviationを既存の材料wall/RZ axis refinementへ使う。
position enclosure全幅を偏差として使わない。固定回帰caseのmaximum refinement depthは材料反射で22、
surface departure→同面再衝突で19であり、いずれも回帰上限24以内である。これはevent certificateの退行検出値で、
製品性能の秒数閾値ではない。
case/result schema、physics catalog/model revision、field semantics、resident/output layoutは変更しない。
COMSOL比較や`model_dataset/`はこの受入へ含めず、外部V&Vだけが所有する。

P11では新しい性能合格秒数を置かない。resident stateとbounded microtile方針はP10から維持し、
event-heavy改善はP12、10k/100k/1Mを含む製品性能判断はP14で同じpublic workflowを使って測った。
P11 closeoutでは標準品質gateとverification/scenario 306件が合格した。

## P12 event-heavy CPU parallel acceptance

P12は第二engineや汎用schedulerを追加しない。各particle IDを一つのworkerだけが所有し、prepared field/geometryを
read-only共有する。各workerはproposal scratch、BVH traversal scratch、residual/event/failure/statisticsを局所に持つ。
main threadは最大worker数分のtileを一waveとして起動し、完了順ではなく元のtile順に回収して既存の物理keyでstable
mergeする。resident active compact、macro-step確定、ResultWriterはmain threadだけが行う。

`resources.threads`はworker数の上限で、粒子ありなら`min(requested, particle_count)`へ解決する。memory plan v3は
全workerのproposal scratchとworkerごとのBVH stack/candidate bufferを同時に計上する。memory不足時にthread数を
黙って減らさず、microtileを縮めても最小workが入らなければ運動開始前に拒否する。

event-heavy common pathは、compiled BVH query、保守的なclear/split batch事前分類、同一時刻・position budgetの
wall-prefix compiled batchを使う。事前分類はhitを作らず、corner、departure、曖昧pieceを既存locatorへ残すため、
proposal v4、event v9、boundary/RNG意味論は変わらない。1/2/4 thread回帰はfinal、event、failure、series、frame/probe、
RNGと次のwork countを完全一致させた。

| accepted pieces | candidate queries | refinements | maximum depth |
|---:|---:|---:|---:|
| 8,704 | 19,968 | 11,264 | 21 |

同一machineの512粒子×4 macro step、same-process warm後、各3回の非gating medianは次のとおりである。9観測の
scientific payload digestは一致した。

| threads | warm simulate | end-to-end | 1-thread比 |
|---:|---:|---:|---:|
| 1 | 0.440460 s | 0.452938 s | 1.000 |
| 2 | 0.528229 s | 0.542817 s | 0.83384 |
| 4 | 0.632361 s | 0.650943 s | 0.69653 |

P12着手直前の同一machine serial baseline 3.066786 sに対し、P12当時の1-thread simulateは6.96xである。一方、thread増加は
現状負のscaleであり、並列speedupを達成したとは扱わない。残るPython/GIL調停、粒子数依存のbreak-even、realistic
P1/Q1、hit数、output量はP14の全matrixで測った。絶対秒数をpytest/CI gateにしない。

case/result schema、physics catalog/runtime、proposal、event、field意味論は不変である。後続P13はmulti-segment、
checkpoint/resume、failure injection、single-writer bounded queueを実装して完了した。
P12 closeoutでは標準品質gateとverification/scenario 309件が合格した。

## P13 durable-result acceptance

P13は第二engine、resume専用API、transaction frameworkを作らない。engineは各worker waveのboundary/failureを
tile順merge後ただちにwriterへ渡し、固定64 macro-step、または最終macro後にepoch barrierを置く。単一HDF5
writer threadは容量1 queueを所有し、main threadはcommand ackを待つため、disk遅延時はeventをdropせずcomputeへ
backpressureする。memory plan v4は最大worker-wave payloadを`worker_output_staging`へ分離し、macro-wide event
stagingを`output_buffer`へ含めない。

commit順はclosed segment、inactive A/B checkpoint、`LATEST`で、`LATEST`だけがepoch commit pointである。
`simulate(case, OUT)`はstrict resume identityが一致するpartialを自動再開する。identityはcase/data hash、
case/result/checkpoint schema、全algorithm/model/backend revision、座標・integrator、particle count/ID digestを含む。
不一致や破損をfallback/migrationせず拒否する。`open_result(OUT, recovery=True)`は最新segmentのhashと、
全segmentの構造・累積countを検証した`LATEST`までのprefixだけを結合し、未完了viewの`read_final()`を拒否する。
hashを直接保持しない過去segmentの同shape値改変は検出契約に含めない。

公開API受入caseは130 macro stepで3 segmentを作り、全segmentのrelease/boundary/failure、series、frame/probe、
finalとragged candidate offsetを検査する。segment/checkpoint/LATESTの各`os.replace`後へfailureを注入し、旧/new
`LATEST`の意味、orphan無視、auto-resume、通常runとの全公開payload・科学manifest raw identityを確認する。
最初の`LATEST`前のfailureは同一caseを初期状態から正確に再実行する。final、complete run.json、`_SUCCESS`、directory
publicationの各境界も次の`simulate`で安全に完了し、確率wall lawは復元したphysical ordinalから同じdrawを得る。
参照checkpointまたは最新segmentのhash破損と、全segmentの構造・累積count不整合をrecovery/resumeで拒否する。
P13 closeoutではuv lock、Ruff、Pyrefly、import-linter、循環的複雑度gateとverification/scenario **322件**が合格した。

容量1 queueは同期ackによるbounded memoryとbackpressureの受入であり、computeとI/Oのoverlap性能は主張しない。
process crash後の同一local volume上の再実行を対象とし、OS/filesystemのpower-loss durability、remote filesystem、
同じOUTへの複数process同時実行は保証外である。

P14は10k/100k/1M、regular/P1/Q1、realistic cell count、hit数、output量、thread数を同じpublic workflowで測り、
durable I/Oを含むthroughputとOS RSSを判断した。これは置換前runtimeの履歴評価であり、P13の小型
failure-injection caseを性能合格値へ読み替えていない。

## P14 product-scale performance acceptance

P14は全組合せの直積ではなく23行の直交型matrixを、一行ごとにfresh process/private Numba cacheと三公開APIで実行した。
10k/100k/1M粒子、regular/P1/Q1、10,368 P1/2,500 Q1 cell、initial/cross-cell、0/1/5/20 hit、
none/sample/all output、cold/warm、1/20 workerを網羅し、全行でcompletion、result shape、JIT環境設定、
revision、memory-plan fit、科学payload identityをhard checkした。絶対秒数は品質gateではなく、COMSOLとの速度比較でもない。

同一machine・3観測/行のmedian simulate値は、regular 1 workerの10k/100k/1Mが
0.095647/0.783874/7.534041 s、20 workerの100k/1Mが0.417049/1.570727 sである。20 worker speedupは
1.8796x/4.7965xだった。一方P1/Q1 10k cross-cellは1 workerで0.747215/0.894859 s、20 workerで
0.636655/0.779968 s（1.1737x/1.1473x）、event 10k×20 hitは1 worker 33.509603 sに対して
20 worker 37.728221 s（0.8882x）だった。
したがってP14時点では`threads: 1`を既定推奨とした。その後P14-Pの内部parallel試行も製品gateを満たさず、
現行case schema v3はthread設定を持たず、single-thread compiled runtimeを維持する。

このthread比較はparallel ownershipとsynthetic scalingの証拠であり、一般的な並列効率の完成を意味しない。
20 workerの並列効率はregular 100k/1Mでも約9.4%/24.0%、P1/Q1 10kでは約5.9%/5.7%である。
regular 1Mでは1/20 workerの実測RSSが約537/1,143 MiB、solver planが約600/3,680 MiBであり、speedupと
memory増加を同時に判断する必要がある。P12の旧serial比6.96xはcompiled BVHとevent algorithmを含む
1-worker pathの改善で、thread speedupとして引用しない。

またP14のevent行はexact ballistic、P1/Q1行は10k粒子であり、非一様force、surface release、材料wall、
一般曲線event、多数macro step、outputを同時に使う主用途caseではない。field/physics/RK stageも完全な一体kernelや
scratch再利用にはなっていない。従ってP14はsynthetic performance baselineを閉じたものとし、P14-Pでparallel
runtimeを不採用として削除まで閉じた。P14-Uは完成したserial runtimeについてtarget-use accuracy、utility、
throughput、memoryを確認する。

regular 10kのnone/sample/allは0.095647/0.118565/0.117372 s、artifactは約2.45/2.66/4.38 MiBだった。
これはend-to-end artifact rateでありwriter単体帯域ではない。cold/warm比はregular/P1/Q1で
25.1932x/15.8691x/12.8671xなので、cold JITとsteady throughputを分ける。regular 10k→1Mのlog slopeは
simulate 0.94818、solver plan 0.58736、RSS 0.23880で、最大RSSはregular 1Mの20 workerが1,143.2 MiB、
1 workerが537.2 MiB、event 1M×1 hitが631.1 MiBだった。全行は8 GiB設定内だが、RSSはsolver-owned planのhard capではない。

profileで支配的だったP1/Q1 initial/miss full scanは`field_location_v3`のstackless cell BVHへ置換した。
supported containmentだけをindex化し、outside/masked provisionalの物理最近傍full scanはO(cell数)のまま保つ。
table start validationのvolume全走査は`line_boundary_volume_cell_bvh_v4`へ置換し、mixed tri/quadのunionに対して
index候補へ従来の厳密half-space predicateを適用する。局所形状がfloat64で解像不能なcellはprepareで拒否し、
CPython `math.hypot`で準備したedge長によりscalar/compiled境界判断を一致させる。絶対座標が大きいだけでは拒否しない。

memory plan v6はfield/geometry indexのresident byte、field 256 B/cell、geometry 1,024 B/cellのbuild transientと、
bounded table-start batchを数える。case/result schema、粒子resident state、physics、proposal、event、result意味論は不変である。
geometry prepareのtracemallocではP1 10,368 cellのpeak-current 5,533,596 B、Q1 10,000 cellの
7,319,844 Bに対し、build上限は10,616,832/10,240,000 Bだった。field indexの10,000-cell P1/Q1も
resident差し引きpeak 1.70/1.93 MBが2.56 MB上限内だった。これはprepare一時配列のsolver-owned boundでありRSS値ではない。
metadata先行gateはcanonical payloadの下限を読む既存責務のままで、index構築前の完全なhard rejectionではない。
authoritative final planはprepared index構築後にresident/transientを含める。構築前hard capが必要になった場合はprepare順序全体の
変更として扱い、indexだけに重複preflightを追加しない。
別の最終hardened smokeは7/7行で、field rowのcompiled physics tile signature=1、exact-path event rowの0許容、
frame/probe行数、global/nested revisionを検査して合格した。release 69観測へこのdispatcher検査を遡及主張しない。
336件のverification/scenario（20.56 s）と標準品質gateが合格した。これによりP14のsynthetic solver-core
performance baselineを完了する。parallel runtimeの収束はP14-P、その完成経路で主用途を結合した
時間/mesh収束とglobal-bound/event costを測るのはP14-U、
T03 analysis/visualizationとP14-R remote Windows/Linux workflowが完了し、配布可能なv0.1を閉じた。
P15 continuous chargeと外部M3-V applicability/relevance評価は完了した。M3-Vは現datasetへの全軌道比較を
`NOT_APPLICABLE`とし、field production、trajectory physics、state dimensionを独立workstreamへ分けた。
trajectory physicsの最初の後続であるP15-D shifted-Maxwellian chargeも、単一正イオン・非正電位の明示範囲で
既存の両積分器・壁・XY/RZ・checkpoint経路へ接続して完了した。P15-E finite-speed Epsteinも同じ経路へ
接続して完了した。P15-F collisionless Barnes ion dragは同じstage passへ明示加速度として接続し、独立
impact-parameter oracle、global bound、continuous charge coupling、両積分器収束、XY/RZ parityを受け入れた。
P16 Waldmann--Gallis thermophoresisも局所並進熱流束をprimitiveとする単一気体free-molecular revisionとして、
同じstage passへ接続した。独立運動論moment、Kn/relative-drift連続適用域、global bound、両積分器収束、
XY/RZ parityを受け入れた。この時点で次だったBrownianはB02で完了した。

P18-Iでは同じcompiled stage passへ、集約場用のrelative-flow screened ion dragとelectric-field-directed image
sensitivityを二つの排他的revisionとして追加した。動的P18-C chargeとsample配列・stage電荷を共有し、relative-flowだけに
宣言最大相対速度の連続path gateを課す。engine、proposal、event、runtime layout、memory plan、case/result/checkpoint
schemaは変更していない。

P18-Dではproducer提供`grad(mean_E_squared)`を使う球形準静的DEPを同じcompiled stage passへ追加した。
`electrostatic_radius_m`と`mass_kg`をparticle authorityとし、producer認証済みの最大point-dipole半径をprepareで検査する。
component boundは既存external-acceleration enclosureへ統合し、XY/RZ、RK4/explicit midpoint、一定加速度退化形を
既存経路のまま使う。engine、proposal、event、runtime layout、memory plan、case/result/checkpoint schemaは変更していない。

P18-Lではproducer提供のsigned方位vorticityを使うRZ/no-swirl
`rarefied_vorticity_sensitivity_rz_v1`を同じstage passへ追加した。正の明示`C_L`、
`F=C_L*pi*rho*lambda*a^2*(omega_phi e_phi)x(u-v)`、`a=drag_diameter/2`、`lambda/a>=10`を固定し、
core内微分とB02 Brownian併用を拒否する。速度依存bound callbackのためexponential enclosureだけをv3へ更新し、
integrator v2、engine v30、proposal v7、schemaは維持した。後続M3-C1 common-P1複合sliceはPASSしたが、lift単独・
native-field parityと物理妥当性は未検証である。

P18-Rでは既存linear EpsteinとWaldmann--Gallis heat-flux式を再利用するeffective-gas sensitivity二revisionを同じstage
passへ追加した。producer-certified one-effective-Maxwellian/pseudogas、`lambda/a>=10`、必須の
`0 < maximum_speed_ratio <= 1`を要求し、既存linear/P16の上限`0.1`は維持する。catalog v15 / runtime v14 /
compiled tile v16だけを更新し、engine、proposal、integrator、enclosure、runtime layout、memory plan、schemaは不変である。
保存artifact auditはnative linear Epstein replayを約`1.1e-15`でPASSとしたが、既存P15-E/P16 applicabilityは12/12
`NOT_APPLICABLE`、PPR `q_eff`欠損によるthermophoresis replayは`NOT_TESTED`である。保存frameは連続pathを認証せず、
COMSOL studyは再実行していない。新revisionはreference/sensitivity設定を可能にするだけで物理的真値を確立しない。

B02初回sliceはCartesian XY、fixed charge、Epstein linear drag-only、terminal `stick`/`escape`へ限定した
`ou_langevin`を同じproduction engineへ接続した。凍結係数joint OU endpoint、conditional tree、
cubic Hermite leaf pathをevent、frame/probe、checkpoint/resumeへ通し、ensemble covariance、depth 6/7/8の
first-passage安定化、output/slab/resume identityを受け入れた。engine v30は`gamma*h` gateとは別にroot covariance、
split、mean更新の実表現可能性を検査し、例外時だけrow別に局在して不良粒子を`nonfinite_physics`へ移し、正常粒子を
継続する。これは有限depth numerical pathであり、連続OU first-passageやCOMSOL同精度の証明ではない。

B03はRZ meridional projected 2-DOFのmacro-root stochastic exponential-midpointとして完了した。
`macro_root_frozen_midpoint_v1`で`gamma,u,T,a,G,J=dG/dZ<=0`を一度凍結し、`stochastic_exponential_midpoint_v1`が
`u_eff=u+a/gamma`のjoint exact OU、root内affine exponential charge、既存conditional treeを合成する。dense chargeは
prepare済みinvariant内だけを許可し、証明不能ならfail-closedする。axis hitはprefixをcommit/foldした後、macro残時間を
新しいroot ordinalと独立drawで再開し、元cubic remainderをfold/restrictしない。受入は定数係数
mean/full covariance exactness、noise-off 2次収束、manufactured weak mean観測次数`>=0.9`、axis restart、
tree-depth first-passage、slab/output/resume identityで閉じた。fixed/continuous charge、native/effective linear Epstein、
既存additive forceを扱う。初回closeoutのwall subsetはterminal `stick`/`escape`/`hold`だけだった。
等方3-D Brownianや一般SDEのstrong order/weak 2次は主張しない。
現行revisionはengine v45 / proposal v10 / catalog v23 / event v21 / runtime v22 / compiled tile v21 / memory plan v16である。
B05では`interval_tree_depth`を全rootへ一様な精度authorityとして維持し、wall/RZ-axis/証明不能候補だけを
`adaptive_max_depth`まで条件付き分割する。base=maxは従来固定depthへ退化する。memory plan v16は候補率ではなく
max depthからtree workを保守的に計画し、連続OU exact first-passageやzero miss probabilityは主張しない。
B03の正式な公開API characterizationは24/24実行を全粒子active・failure 0で完了した。静的上限`648 B/row`は
`2048 B/row`以内で、20,000粒子までの計時とprocess peak RSSは[`evidence/b03/`](../evidence/b03/README.md)に
machine-local・non-gating証跡として保存した。この計時/RSSは初回B03 closeout（engine v34 / proposal v9 / runtime v17 /
tile v16）の履歴証拠である。
B03 coreの受入にCOMSOL再実行は不要だった。charge-stable couplingとwork-scaled durable cadenceも完了し、
直近の全品質gateはPASSした。P20 performance closeout後、M3-C2A common-P1 Case-A/Case-P 100 nmの独立pilotと
各32+32 seed final campaignを完了した。Case-Pは20 us、各287粒子×121 frame / 30 msで、83区分R-Z/fate TV
`0.010670731707317093`、同時上限`0.13119968456545308 < 0.15`を`PASS`した。終端gateはevent 0で非情報的である。
これは元COMSOL `auxq`どおりの二電流same-form結果で、後続three-currentやspecies-resolved物理を認定せず、
pathwise/native-field/boundary parityへも一般化しない。optional three-current productionはP21 priority 1で完了した。

B04は現行boundary v7のactive `specular`/`restitution`/`maxwell_thermal`/`probabilistic_stick`を
B02/B03の同じfirst-hit経路へ接続した。hitまでのHermite prefixだけをcommitし、反射後stateから
残時間を新しい`root_stochastic_interval`ordinalと独立root drawで再開し、元rootの未使用cubic tailは
破棄する。低noiseのXY/RZ複数hit、restitution、probabilistic fallback、Maxwell鏡面branch、B03 charge/force、
axis→wall、実noiseのcontainment/depth/slab、checkpoint/resume identityで受け入れた。surface sourceは
canonical rowでrealize済みのstrictly inwardな初期速度を使い、零速度や接線速度をnudgeで通さない。
B04も有限depth numerical pathであり、continuous OUのexact first-passageまたは
reflection-conditioned bridgeは主張しない。
charge-stable sliceはRK4のexplicit `hL_Z<=0.5`を残し、deterministic exponential midpoint/B03へ
midpoint-frozen affine exponential `J<=0` rootを統合した。clip、charge-only subcycle、第二engineはない。
cadenceは`W=macro_step_count+accepted_particle_pieces+candidate_queries+refinements`、
`T=max(2^20,128N)`で、engineがaccepted macro barrierと最終macroで判断する。manifest/resume identityに同じ
cadenceを持ち、output schedule/slabから独立である。同期single-owner writerはatomic persistenceだけを所有する。
T04 cache/remeshは後続profileが必要性を示すまで延期する。

## P14-P parallel runtime convergence gate

以下はcloseout後に追加した受入要約である。profile evidence、module責務、実装順、全matrixは
[`parallel_execution_plan.md`](parallel_execution_plan.md)を権威とする。

P14 closeout後のprofileでは、regular field 10^5粒子以上はcompiled field/physics/RK passが主要時間を占める一方、
exact/curved wall eventはparticle別のPython調停が支配的だった。P14のouter `ThreadPoolExecutor`は最大worker数分の
tileを起動してwaveごとに全完了を待ち、workerごとにproposal/event scratchを持つ。したがってP14で観測した
regular 1Mの4.7965倍だけを根拠に完成扱いせず、event 20-hitの0.8882倍、P1/Q1 10kの低効率、regular 1Mの
solver plan約600 MiBから約3.68 GiBへの増加も同じ設計課題として扱う。

P14-Pでは次の構造へ置換した。

- outer worker-waveを削除し、一つのbounded tileをrow単位で処理する。
- field sampling、physics、integrator、exact/curved first-hit、境界応答、RNG、残時間をcompiled numeric passでつなぐ。
- event workはPython list/dict/dataclassでなくflat SoA wavefrontとし、各rowに独立したtarget time、residual、
  refinement depth、interaction/event ordinalを持たせる。
- 一roundに一row高々一件のevent/failureを書き、stable compactionと物理key順で決定論的に出力する。
- scratchをtile幅に対して一度確保して再利用する。
- outer executor、thread mask、旧worker merge、serial/parallel feature flagを併存させない。

一回のfocused correction後もregular 1Mは1/2/4 threadで9.32/10.09/10.10 s、4-thread speedup 0.923x、
1-threadはv20比23.7%退行した。field locator microkernelは約3.75xでも、proposal/enclosure調停が支配した。
したがってmultithreading API/config、thread mask、対応test/harnessを削除し、serial compiled engineへ一本化した。
自動tuner、別scheduler、multiprocessingは代替案にしない。

`deterministic_particle_engine_v20` / `resident_soa_worker_microtile_v3`はP14-P着手前のbaselineとして保存する
履歴であり、現行v45はouter pool、future wave、worker別scratch、thread maskを削除し、
bounded slab、再利用workspace、stackless boundary BVH、同期single-owner writerへ移行した。
linear/quadratic exactと一般曲線eventはflat SoA wavefront、boundary/Philoxはcompiled batch、fieldからenclosureは
row numerical status、surface releaseはbatch、frame/probeはdirect replay、event/failureはbounded stagingへ統合済みである。
P14-Pは否定的closeoutまで完了した。続くP14-Uはsurface release＋非一様場＋材料wall＋多数stepの時間/mesh収束と
global-bound/event costを測定して完了した。

## P14-U representative-use gate（完了）

P14-Uは第二engineやproduction診断を追加せず、外部harness
`tests/performance/p14u_representative.py`から三公開APIを実行する。代表XY caseはsurface release、Epstein drag、
非affine electric field、gravity、材料target、多数fixed stepを同時に使う。時間収束は自己差分と独立fine reference、
空間収束は`nx=ny`の固定aspect比でregular/P1/Q1ごとに同layout fine referenceを使い、同期位置・速度、hit時刻・位置、
同じtarget boundary、facet内部clearanceを検査する。dense path sampleはsourceと局在hitを含め、global boundと
実経路field scaleを区別する。

RZ caseは可変かつaxis-regularなcanonical場で一回のaxis crossingを含み、時間自己差分/fine referenceに加え、
実入力fieldのscalar/axial evenness、radial-zero regularityを監査する。standard gravityのradial成分0は
axis accessibilityに依存しないsolver invariantである。wall law v4ではparameterなし`specular`を完全鏡面反射、
非単位係数を明示`restitution`として分離し、probabilistic非付着側も選択lawをmanifestへ残す。

正式releaseだけがP14-Uをcloseできる。10k/100k/1M粒子それぞれで`none`/sample出力を3回ずつfresh process/private
Numba cacheで測り、raw観測とmedianを保存する。これとは別の1M `none` runをcProfile専用に一回実行し、timing観測を
汚染しない。全performance rowは粒子ごとにrelease一件とtarget stick一件を要求し、output mode/repeat間で科学payload、
revision、event work、wall counterが一致しなければならない。RSS high-waterとsolver-owned memory planは別量として記録する。

正式releaseは`release_gate_complete=true`で完了した。local reportは
[`../evidence/v0.1/p14u_release_v1.json`](../evidence/v0.1/p14u_release_v1.json)で、10k/100k/1M×`none`/sample×3回の
18 raw観測と6 medianを含む。1M粒子のmedianは次のとおりである。

| output | simulate | public end-to-end | result artifact |
|---|---:|---:|---:|
| `none` | 550.43 s | 553.77 s | 455.1 MiB |
| sample | 533.95 s | 537.61 s | 455.3 MiB |

raw peak RSS最大値は767.3 MiB、1Mのsolver-owned planは614.5 MiBだった。全観測でfailure 0、release/target-stick
各N件と要求output utilityを確認し、共通payload digest、algorithm revision、event work、wall counterはrepeat/output
mode間で各粒子数group内において一致した。probe digestは各mode内のrepeat間で一致し、probeを持たない`none`とsampleでは意図的に異なる。
XYの時間収束とregular/P1/Q1 mesh収束、target first-hit、RZの時間収束、axis crossing、canonical入力field parity、
gravity parityも合格した。履歴generatorの受入範囲はXY line-length分布であり、RZ回転面分布はこのgateでは
検証していない。現行runtimeはgeneratorと分離したcanonical realized surface rowを入力とする。

timingから分離した1M `none` profileではowner self timeのeventsが28.7%、fieldsが26.8%で、単一ownerが支配して
いなかった。この結果だけを根拠に局所最適化や第二runtimeを追加せず、productionは一つのsingle-thread compiled
engineを維持する。`none`とsampleの観測群は逐次実行したため、両者の秒数差をsampling overheadの因果推定には使わない。
絶対秒数はmachine-localかつnon-gatingで、COMSOLとの速度比較ではない。RSSはload/simulate/openまでをsampleした
process high-waterであり、後続の外部validationは含まない。P14/P14-Uの受理済みJSONとplatform smokeは
`../evidence/v0.1/`へ保存した。T03、local audit、P14-R remote Windows/Linux workflowも完了した。remote receiptは
`../evidence/v0.1/release_remote_ci_v1.json`である。P15 continuous chargeと外部M3-V relevance評価も完了した。M3-Vの未検証項目を
production合格へ読み替えない。P15-D shifted-Maxwellian chargeは明示したspecies・非正電位範囲で完了し、
後続は三つの独立workstreamとして進める。

## M3-C1 event-query v14 post-closeout

Case-A 100 nm common-P1 material anchorで、event v13がglobal absolute safety enclosureをevent BVH queryにも流用し、
artificial subdivisionを起こしていたことを特定した。450 usまでに`7,623,460 / 7,792,306` refinement
（`97.8331703092769%`）が累積し、最終query/refinement/accepted/depthは
`16,427,517 / 7,792,306 / 8,635,211 / 16`だった。

M3-C1 event v14はglobal supportを独立に証明済みのvalid `rk4_dense` rowだけcurrent dense Bernstein位置・速度boundを
event broad-phase query authorityにする。global enclosureはshortened-stage、field support、applicability、acceptance
safety authorityのままで、invalid dense boundまたはglobal support未証明時はglobal queryへfallbackする。
first-hit、wall、endpoint、schemaの意味は変わらない。material candidateは`842,927 / 11 / 842,916 / 11`、failure 0。
evaluation v5はmaterial 20/20とpre-event prefix 9/9をPASSし、event時刻、hit位置、terminal chargeの絶対差は
`2.157542807607049e-13 s / 2.683964162031316e-14 m / 4.7283812421028415e-09 e`だった。

operator-observed shell wall-timeは約`14m13s`から`36.5s`（約`23.4x`）だが、machine-local、概算、非gatingで
solver報告値ではない。v14 solver-only 0.625/0.3125/0.15625 us再実行はrefinement 0で、位置・速度・電荷のRMS観測次数
`2.029875353701904 / 2.0816971911764033 / 2.044084026475049`、fine relative L2
`6.099791486063973e-8 / 8.321356016032579e-8 / 1.3796067752988052e-8`、全量`ORDER_EVALUATED`でPASSした。
約2次はpiecewise P1場・mesh crossingを含む本caseの経験値で、RK4の形式4次を証明も否定もしない。旧v13の3刻み結果は
precision-stability履歴として有効だが、accepted-piece countが同一なので独立時間収束authorityではない。

v14はsolver event broad phaseだけの変更で、hash-lock済みCOMSOL input/reference/source MPHは不変だったためCOMSOLを
再実行せず、既存referenceへcandidateを再比較した。このPASSはcommon canonical exact-connectivity P1、Brownian off、
287粒子、first wafer stickまでに限り、native-field parity、物理妥当性、Brownian、30 ms、他case/size/variantを認定しない。
P18-Hも解析・公開回帰と外部Freeze candidate 15/15で完了し、B03 coreと二production slice、P20 performance closeoutも
完了した。M3-C2A inventory preflight後の意味一致Case-A/Case-P 100 nm実行契約、pilot、final ensembleも完了した。
accepted Case-P seed `319032/319047/319063`の287粒子owner discoveryも、受理済み科学payload・work・case identity・revisionを
完全一致させて完了した。支配ownerは`integrators`（自己時間比42.58--42.86%）だったが事前登録済みbounded ownerではないため、
最適化は未承認で本体変更はない。M3-C2A anchorは`CLOSED_ACCEPTED_WITH_LIMITATIONS`であり、10,000粒子以上の性能、
残るpackage、process並列は利用SLAを先に定義した別work packageとする。COMSOL fittingは行わない。M3-C1 event v14の
既存evidence/evaluatorは履歴として変更しない。
このowner discoveryとは別に、後続の明示指示でchord算術一箇所だけをserial compiled batchへ統合した。accepted 3 seedの
科学payload/work/revisionを完全一致させ、end-to-end中央値を12.29%短縮したため保持し、追加ownerへ進まず終了した。
authorityは[`../evidence/m3c2/caseP_100nm_chord_optimization_v1/`](../evidence/m3c2/caseP_100nm_chord_optimization_v1/README.md)である。
