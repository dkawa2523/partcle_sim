# Implementation stage gates

長いADR treeを作らず、未確定事項と所有packageをここへ一表で残す。決定済み内容の詳細は各owner文書へ
移し、この表へ同じ説明を複製しない。

| gate | 着手前に確定または実装すること | 状態・owner |
|---|---|---|
| P03/P14/P19 fields | conditioning-aware P1/Q1 location、有限provisional、large-offset/high-aspect regression、supported-containment index、bounded local primitive range | `field_location_v4`として完了。outside/masked provisionalは意味論を守るO(cell数) full scan。局所rangeは一row最大64 cellで、超過を成功扱いせずcertificate側が区間を分割する |
| P04 ballistic | data座標表現とmotion modeの分離、typed trajectory設定、C01 frame/release/result semantics | `ballistic_engine_v1` / `ballistic_result_v1`として完了 |
| P05 geometry/event | 大域topology audit、ballistic line BVH、exact first hit、facet budgetの対称結合、failure semantics | `ballistic_terminal_event_engine_v3` / `line_boundary_bvh_v2` / `ballistic_line_first_hit_v2` / `terminal_boundary_v1`として完了。一定加速度曲線はP06 revision 2、boundaryless一般曲線boundはrevision 3a、材料eventはrevision 3b |
| P06 coupled physics | required fieldの全domain support、fixed charge、Epstein/electric/gravity、共通RK4 proposal、証明済み一定加速度の放物線event、boundaryless/material general-RK4 enclosure | 現行`particle_engine_v36` / `coupled_fixed_step_proposal_v10` / `rk4_global_abs_enclosure_v2` / `line_quadratic_rk4_axis_first_hit_v16`が意味論を維持。global enclosureはshortened-stage、field support、applicability、acceptance safetyのauthorityであり、global supportを独立に証明済みのvalid `rk4_dense` rowだけcurrent dense Bernstein boundをevent broad-phase query authorityにする。それ以外はglobal queryへfallbackする |
| P06-U unstructured material | exact-mesh fully-supported P1/Q1、Cartesian XY一般RK4、topology-complete material boundary | 完了。boundaryless unstructuredは未解禁 |
| P06-RZ force coupling | signed meridional chart、RZ field basis、axis regularity、RK4 axis event、support/applicability | engine v12 / event v8 / physics catalog v2 / required field v3として完了。case/result schema、proposal、enclosureは不変 |
| P06-S Stokes drag | 小さいphysics runtime、Allen--Raabe air revision、Kn/Re適用域、XY解析oracle、away-axis RZ parity | engine v13 / physics catalog v3 / physics runtime v1として完了。暗黙model切替なし |
| P07 boundary/source | surface measure、counter RNG、wall law、corner response、RZ axis path split、C09 residual split規則 | engine v11 / event v7で完了したsliceをengine v36 / event v16が維持。rev3bのevent-before-validityと、hit prefixを同じintegratorで再積分して残時間をfresh proposalで続ける逐次意味論は不変。分布拡張、moving wallは後続gate |
| P08 Stage 1A closure | particle-local failure、lifecycle series、state probe、三公開APIを使う薄いCLI、公開scenario | engine v14 / result v2として完了。case/result schema versionは1を維持 |
| P09 memory/runtime layout | metadata preflight、stable active index、bounded microtile、phase memory plan、fresh/warm RSS script | engine v15 / runtime layout v1 / memory plan v1として完了 |
| P10 compiled CPU | Numba field/physics/RK4 pass、accepted endpoint hint、`state_at`/wall/residual/output parity、cold/warm harness | engine v16 / compiled tile v1 / runtime layout v2 / memory plan v2 / physics runtime v2として完了。schema/model/event/field semanticsは不変 |
| P11 exponential | C03、極小/極大`h/tau`、丸め安定性、可変係数でorder 1.8以上、既存event/RZ/material経路との統合 | engine v17 / compiled tile v2 / proposal v4 / event v9 / physics runtime v3、`exponential_midpoint_v1` / enclosure v1として完了 |
| P12 event-heavy CPU parallel | 一粒子一worker、read-only共有、worker-local residual/event/statistics、bounded tile wave、stable merge、thread/memory意味論 | ownership・batch・決定論をengine v18 / compiled tile v3 / geometry v3 / runtime layout v3 / memory plan v3として完了。event-heavyの正のthread speedupは未達 |
| P13 durable result | 固定epoch、worker-wave stream、single-writer backpressure、A/B checkpoint、LATEST、auto-resume、recovery、failure injection | engine v19 / result algorithm v3 / checkpoint schema 1 / memory plan v4として完了。case/result schema 1は不変 |
| P14 performance | machine fingerprint、mesh、10^4/10^5/10^6粒子、出力量、warm-up、thread、locator bottleneck判断 | engine v20 / compiled tile v4 / field v3 / geometry v4 / memory plan v6としてsynthetic baselineを完了。23行×3観測、336 verification/scenario。当時は`threads: 1`を既定推奨し大規模regularでhost幅を実測選択したが、この運用判断は直後のP14-Pでsupersedeされ、現行productionにはthread選択がない |
| P14-P serial runtime convergence | outer worker-waveをflat SoA event wavefrontへ置換し、内部parallelを実測して採否を確定 | 完了。4-thread 0.923x、1-thread v20比23.7%退行のためmultithreadingを削除。engine v27 / compiled tile v6 / proposal v5 / event v11 / runtime layout v5 / memory plan v10 / geometry v5、case schema v2 |
| P14-U representative use | surface release＋非一様場＋材料wall＋多数stepの時間/mesh収束、global-bound/event cost、直列end-to-end性能・memory・output | 完了。正式release 18 raw観測＋6 median、別1M profileを受入れ、productionは単一の直列engineのまま維持 |
| P14-R release closure | P14 evidence保存、Windows/Linux wheel・clean install・三API smoke、current/history文書authority整理 | 完了。receipt固定のtested head/lockについてremote Windows/Linuxで620件、性能smoke 7行（cold 1＋warm 6）、wheel、runtime-only clean install、三API smokeが合格。run `37217687830`をreceiptへ固定 |
| T03 analysis/visualization | bounded boundary-event iterator、`ResultView`だけを読む最小集計・軌道・event可視化 | 完了。source resultを変更せず、tool revision・source manifest hash・parameterを派生成果物へ保存する |
| P15 continuous charge | stationary Maxwellian OML＋Debye--Hückel、rate/derivative/charge bound、state-evolving dispatch、charge-aware enclosure、非stiff RK4-first、その受入後のexplicit midpoint | production完了。engine v28 / proposal v6 / 両enclosure v2 / memory plan v11を維持 |
| M3-V external applicability / matched companion | 直接MPH inventory、12 package、式parity、sampled applicability、variant感度、別成果物の決定論exact-P1 pre-event軌道 | 完了。元datasetの比較処理はPASSだが物理認証は`NOT_CERTIFIED`。共通P1場・共通3力のCase-A 100 nm companionの時間離散parityのみPASS。native-field、boundary、stochasticへ拡張せず、core gateにしない |
| P15-D relative-drift charge | 単一正イオン種のshifted-Maxwellian OML、非正電位invariant、明示drift envelope、既存の両積分器・壁・XY/RZ・checkpoint経路 | production完了。compiled tile v8 / catalog v6 / runtime v5。engine、proposal、schemaは不変 |
| P15-E finite-speed Epstein | 文献式、散乱仮定、Kn・速度比適用域、低速極限、安定評価形、独立速度積分oracle | production完了。compiled tile v9 / catalog v7 / runtime v6。engine、integrator、schemaは不変で、linear revisionとの自動切替なし |
| P15-F versioned ion drag | 単一正イオンのBarnes型collection＋orbital scattering、相対流方向、collisionless screening適用域、独立impact-parameter積分oracle | production完了。compiled tile v10 / catalog v8 / runtime v7。engine、integrator、memory plan、schemaは不変 |
| F01 reduced electrostatic builder | static RZ/P1 C2 Boltzmann--Bohm closure、semantic BC、Newton/GMRES、canonical field | `reduced_electrostatic_builder_v1`として完了。particle engine/schemaは不変 |
| F02 field-production integration | COMSOL CSV mixed mesh adapter、代表linear solve、fixed-electric smoke、外部field比較 | 完了。core revisionは不変。同一export node比較だけTESTED、独立mesh収束・軌道/Freeze parityは未検証 |
| P16 Waldmann--Gallis | 局所並進熱流束をprimitiveとする単一気体free-molecular thermophoresis、連続適用域、独立oracle、compiled parity | production完了。compiled tile v11 / catalog v9 / runtime v8。engine、integrator、memory plan、schemaは不変 |
| B01 inertial Brownian numerics | joint OU厳密更新、物理区間木Philox、親終点を保存するconditional half-split | Stage 2B数値基盤として完了した履歴。B01単独ではcaseから選択できず、production接続はB02が所有する |
| B02 inertial Brownian production | Cartesian XY、Epstein linear、fixed charge、terminal stick/escape、固定depth OU/Hermite event・replay・checkpoint | production完了。engine v30 / proposal v7 / catalog v10 / runtime v9 / runtime layout v6 / memory plan v12。OU表現不能は粒子単位で失敗し、連続OU first-passageの厳密解とは主張しない。P18-Hで同じterminal subsetへholdを追加済み |
| B03 charged/forced RZ Brownian | RZ meridional投影、fixed/continuous charge、native/effective-gas線形Epstein、既存additive forceをmacro-root stochastic exponential-midpointへ合成 | production完了。初回closeoutはengine v34 / proposal v9 / event v15 / runtime v17 / tile v16、現行pathはengine v36 / proposal v10 / event v16 / catalog v17 / runtime v19 / tile v18。B02の物理payloadを維持し、静的path arrayは648 B/rowで2048 B/row上限内。正式characterization 24/24は初回closeoutの履歴証拠 |
| P18-C aggregate charge | 正負電位と相対driftを含む集約二電流continuous charge、有限invariant/bound、両積分器・XY/RZ・event/resume | production完了。compiled tile v12 / catalog v11 / runtime v10。engine/state/schemaは不変。外部保存式再生PASSと定数規約込み厳密provider一致FAILを分離 |
| P18-I aggregate ion drag | 集約場用relative-flow screened式とelectric-field-directed image感度式を排他的revisionとして同じstage passへ統合 | production完了。compiled tile v13 / catalog v12 / runtime v11。engine/state/schemaは不変。保存式再生とproduction式差は外部V&Vで分離 |
| P18-D quasistatic spherical DEP | producer提供`grad(mean_E_squared)`、実数CM factor、認証済みpoint-dipole半径上限を既存stage passへ統合 | production完了。compiled tile v14 / catalog v13 / runtime v12。engine/state/schemaは不変。producer provenanceとCOMSOL比較はcore外 |
| P18-L rarefied-vorticity lift sensitivity | producer提供のRZ方位vorticityと中性気体primitiveを使う、free-molecular感度revision。速度依存boundを両積分器の経路包絡へ統合 | production完了。compiled tile v15 / catalog v14 / runtime v13 / exponential enclosure v3。integrator v2、engine v30、proposal v7、schemaは不変。M3-C1 cross-representation比較はFAIL、Case-A 100 nm common-P1 event前same-field診断はPASS |
| P18-R effective-gas drag / thermophoresis sensitivity | producer認証済みone-effective-Maxwellian/pseudogasへ既存linear EpsteinとWaldmann--Gallis heat-flux式を適用するoptional revision | production完了。compiled tile v16 / catalog v15 / runtime v14。engine、proposal、integrator、schemaは不変。P18-R時点の保存audit判断は維持し、後続M3-C1の最小PPR補足でfrozen saved-state producer-form replayだけを8/8閉じた |
| P18-H generic terminal hold | nondeposition terminalをparameterなしの`hold`と独立`held` lifecycleで表し、hit payloadを有効なまま保持する | 完了。engine v32 / boundary v5 / result algorithm v4 / result・checkpoint schema 2。case schema・event locator・integrator・memory planは不変。解析・resume・Brownianを含む公開回帰と外部Freeze candidate 15/15 PASS。COMSOL再実行なし |
| P19-L localized applicability certificate | fixed-step RK4の元endpointを保つintegrator-owned dense path、local cell primitive bound、certificate-only部分区間制限 | 完了。engine v31 / proposal v8 / field location v4 / memory plan v13 / dense path `rk4_position_hermite_state_extension_v2`。P19-L完了時点はevent v12、現行はv16。actual `model_applicability`と証明不能`indeterminate_applicability_certificate`を分離し、COMSOL専用分岐を作らない |
| RK4 dense path revision 3 | 原点相対のBernstein enclosure、dense評価、roundoff certificate | 現行`rk4_position_hermite_state_extension_v3`。root始点相対差、TwoDiff残差、方向付き外向き座標変換で、v2の広い絶対座標paddingが原因で閉じなかったevent certificateを修正。公開chord boundは狭い`8*eps`絶対座標termを含み、完全な平行移動不変ではない。v3導入時のendpoint・数学的path・event algorithm、engine v32、event v14、RK4 global enclosure v2は不変だった |
| M3-C1 Case-A trajectory anchor | exported exact-connectivity P1 candidateとCOMSOL native-field referenceを分離評価し、続いて両側へ同じfull-physics common P1場を与える | cross-representation RMS/max 6 gateは全FAIL。common-P1 pre-event 9/9、material-event 20/20＋prefix 9/9、v14 solver-only自己収束を別判定でPASS。限定anchorだけを認定し、native-field等価性と物理妥当性は未認定 |
| M3-C1 100 nm・30 ms candidate-first policy | candidate自己収束を主要数値gateとし、保存COMSOLは意味一致を証明できた場合だけ外部gateに使う | candidate v3 Case A/P自己収束PASS。保存COMSOLはfixed RK4 10 us・Brownian-on・native-fieldなので`CHARACTERIZED`だけ。後続M3-C2A common-P1 Case-A/Case-P 100 nm anchorと287粒子owner discoveryは完了し、`CLOSED_ACCEPTED_WITH_LIMITATIONS`。10,000粒子以上の性能は独立work packageであり、COMSOL fittingは行わない |
| charge-stable continuous coupling | RK4のexplicit gateを維持しつつdeterministic exponential midpoint/B03へ単一のstable scalar charge pathを統合 | 完了。`charge_stable_exponential_midpoint_v3`、proposal v10、現行physics runtime v19。midpoint-frozen affine law、`J=dG/dZ<=0`、`expm1`安定評価を同じcoupled proposal/rootで使う。clip、charge-only subcycle、第二engineなし |
| P21 aggregate three-current | 既存二電流を変更せず、aggregate単一価負イオン収集を同じcontinuous-charge state/passへ追加 | priority 1 production完了。catalog v17 / runtime v19 / tile v18、engine/state/schema不変。rate/Jacobian/global-boundのzero-density退化、compiled/public-API回帰と標準品質gateを通過。priority 3外部coverageは入力authority不足で`BLOCKED / NOT_EVALUATED`、三電流軌道同等性は非認定。P21は`CLOSED_ACCEPTED_WITH_LIMITATIONS` |
| work-scaled durable cadence | 累積solver workからepoch commitを決定し、atomic persistence ownerを分離 | 完了。`cumulative_solver_work_v1`、engine v36、result v5。`W=macro+accepted+queries+refinements`、`T=max(2^20,128N)`、accepted macro barrier。manifest/resume identityへ記録し、output schedule/slab非依存。writerは同期single-owner |
| P17 RZ-field Cartesian 3-D | XYZ state、RZ mapping、回転面event、3-D normal、schema/output/memory | state-dimension独立workstream。一般可変次元frameworkは作らない |

## P05 geometry/event revision 2

| 項目 | 判断 |
|---|---|
| 解く利用case | 2-D XYまたはRZ断面のvolume mesh内を進むpoint particleについて、ballistic直線pathの最初の材料境界到達を求め、terminalなstick/escapeを適用する |
| 既存modelで解けない理由 | P04はboundaryを明示拒否し、proposal終点までを無条件で受理していたため、step途中の壁通過を検出できない |
| 所有module | `geometry.py`がtopology・法線・line BVH、`events.py`がbudget・first hit、`boundaries.py`がterminal law、`engine.py`が状態遷移、`output.py`がevent/resultを所有する |
| 必要field/state | fieldなし。prepared geometry、位置・速度・lifecycle、粒子別event ordinal、terminal時刻を追加する |
| 対応座標・integrator | `cartesian_xy`と`axisymmetric_rz_meridional`のballistic pathだけ。RZ axis seamは壁ではなく、軸到達が材料hitより先ならP07まで拒否する |
| reference | C07の`(t,x,facet,n)=(1.5,(1,0.625),1,(1,0))`、C10のgeometry-only同時candidate集合、無効topologyの最小case |
| 性能・memory影響 | geometryを一度prepareし、path AABBをflat BVHで絞る。P05時点はcanonical DataBundle、schedule、派生geometry、常駐stateの下限だけを見積もったが、P09のphase memory planが置換した。P05は正しさを固定するscalar event loopで、batch/compiled化はP10が同じ意味論を置換する |
| 置換・削除する旧経路 | P04のboundary一律拒否を削除し、boundaryなしも同じengineのno-hit caseとする。別collision engineや互換分岐は作らない |

topology auditはvolume cell edgeのmanifold性、材料外周facetの完全性・一意性、boundary graphと交差を検査する。複数の
disconnected componentとhole loopは有効であり、単一connected domainを仮定しない。RZで両端`r=0`の
incidence-1 edgeはaxis seamとして材料BVHから除外し、そのedgeをmaterial boundaryとして登録した入力は
拒否する。P05のtable releaseは材料boundaryを持つcaseではstrict interiorだけを許し、boundary上の
departure tokenとsurface sourceはP07まで拒否する。

revision 2ではboundary vertex次数、非隣接facet交差、collinear overlapもprepare時に拒否する。通常の
閉境界vertexは次数2、RZで省略したaxis seamの端点だけは次数1を許す。boundary rowが0件でもvolume
incidence監査を迂回しない。canonical schema・局所connectivityは`load_case`、この大域監査とBVH構築は
`simulate`のprepareが一度だけ所有する。

P05のevent locatorは直線segmentとline2の厳密交差だけを扱う。position/time budgetは
`solver.event.geometry_rtol`、`roundoff_ulps`、geometry/edge/速度/step scaleから`events.py`で一度だけ
解決する。同時facet判定はfacetごとのbudgetの最大値を対称に使い、candidateはfacet ID順とする。
epsilon位置押戻し、endpointだけの判定、BVH走査順によるtie breakは行わない。複数candidateのlaw/法線
解決はP07の責務なので、P05 production runは単一candidate hitだけを受理する。

revision 2では最大facet budgetをBVH broad phaseだけに用い、初期点の境界距離、near-parallel separation、
path/facet parameter受理は候補facet固有budgetで判定する。区間外交点を端点へclampするにはpositionとtimeの
両budgetを満たす必要がある。event始点、accepted endpoint、途中frameは`StepProposal`から評価し、engineへ
別のballistic式を残さない。

P05のboundary lawはparameterを持たない`stick`と`escape`だけである。stickはhit位置、速度0、
`lifecycle=stuck`へ遷移する。escapeは表面impulseを与えずeventのpost速度をpre速度と同じにして
`lifecycle=escaped`へ遷移し、以後のtrajectory frameから除外する。frameはeventに対して右連続とし、
hit時刻ちょうどではstickはpost-event状態、escapeは行なしとする。

final表はrun終了時刻のlifecycle snapshotであり、全行の`time_s`は`time.end_s`とする。escaped後の
kinematicsをNaN sentinelにせず、`kinematics_valid=0`をlogical nullとして表す。数値payloadには最後の
hit状態を保持するが、readerはvalidityが0のposition/velocityを科学値として使用してはならず、最後の
有効状態はboundary eventがauthorityである。P05時点ではactive/stuckを`kinematics_valid=1`とし、lifecycle codeを
`0=pending, 1=active, 2=stuck, 3=escaped, 4=failed`へ固定した。P18-Hは既存codeを変えず`5=held`を追加し、
active/stuck/heldをvalidとする。

releaseはordinal 0、最初のboundary eventは同じ粒子のordinal 1とする。物理boundary RNG用ordinalは
P07で別stateとして導入し、この永続event ordinalと混同しない。

P05でeventをfloat64精度内に証明できない場合はrun全体を`SimulationError`で失敗させ、no-hit、stick、
escapeへ読み替えない。粒子単位failure eventと継続policyはP08のStage 1A closureまでに追加し、P05 schemaへ
空のfailure groupを先行作成しない。

## P06 coupled RK4 revision 1: first vertical slice

P06は一つのwork packageのまま段階実装する。最初の縦切りでは「境界なしのXYで、静的primitive
fieldからfixed-charge粒子の決定論軌道をclassical RK4で解く」数値式とreference proposalを固定した。力なしcaseも同じ
`StepProposal`とproduction loopの`linear_exact`退化形を通し、第二engineは作らない。

| 項目 | revision 1の判断 |
|---|---|
| 解く利用case | `cartesian_xy`、realized table source、fixed charge、静的fieldのEpstein linear drag、Coulomb電気力、重力・浮力の数値reference。public runの有効範囲はrevision 2と3aのcontinuous-support certificateで制限する |
| 既存modelで解けない理由 | P05は無力場の解析的直線だけで、RK stage位置でfieldと力を連成評価しない |
| 所有module | `physics/catalog.py`がmodel解決、`physics/forces.py`が純粋式、`physics/charge.py`がfixed rate、`fields.py`がfield binding/coverage、`integrators.py`がRK4 proposal、`engine.py`がstage調停とcommit |
| required field | 全required fieldは同じlayoutのnode field。regularは全cell supportとgeometry全nodeのaxis box内包、P1/Q1はgeometryとのnodes/connectivity完全一致と全cell supportをcoverage certificateとする |
| field semantics | scalarは`(value)/scalar`、XY vectorは`(x,y)/cartesian_xy`。unit、association、basis、componentをprepareで完全一致させる |
| physics scope | `fixed_charge_v1`、`epstein_linear_v1`、`electric_coulomb_v1`、`gravity_buoyancy_standard_v1`だけ。加算順はdrag、electric、gravity/buoyancy |
| Epstein適用gate | 数値式の必須条件を`rho,T,lambda,m_g,d,m > 0`とする。revision 1の保守的運用範囲を`lambda/a >= 10`、`|u-v|/c_bar <= 0.1`、`1 <= delta <= 13/9`と固定し、普遍的な物理境界とは主張しない |
| applicability | revision 1は`error`だけ。`count`はaccepted proposalだけで数える単位とresult意味を固定してから追加する |
| integrator | `(x,v,Z)`のclassical RK4。各stageの実`t,x,v,Z`とaccepted endpoint（zero-durationを含む）でsample・validity判定し、既知線形dragは`dt/tau_min < 2.5`をprepareで要求する |
| reference | C02/C03の閉形式解と4段階dt収束はdirect RK4かつrevision 3aの公開API、C04の二質量とC05の三つの排除体積はrevision 2の公開API |
| 性能・memory | P06はNumPy reference。unique layoutは粒子・stageごとに一回locateしてrequired fieldを共有sampleする。compiled tileはP10 |
| 置換・削除 | ballistic専用proposal型とproduction入口を削除し、`rk4_step`と`StepProposal`を唯一のproposal経路にする。無力場の`linear_exact`はbitwiseな厳密性を保つ内部specializationとしてrelease原点から解析評価し、force-coupled caseだけcurrent accepted stateからproposal→判定→commitする |

post-reviewで、RK stage点とaccepted endpointの判定だけでは曲線の連続supportを証明できず、
要求frame時刻での再評価がrunの成否を変え得ることを確認した。前版engine v3では一般`rk4_reintegrated`を
production prepareで一律拒否した。revision 3aはboundaryless regular support/applicability包絡を追加し、
証明済みのC02/C03を公開runへ移した。材料boundaryとの一般RK4連成はrevision 3bまで拒否していた。

`stokes_cunningham_allen_raabe_air_v1`は式候補のままとし、viscosity/density/mean-free-path field
schema、air-only適用域、独立oracleが揃うまでproduction catalogへ入れない。RZのvector basis語彙とaxis
regularity、曲線pathの保守的な速度・変位包絡も後半gateとする。revision 2の証明済み一定加速度を除き、
このrevisionでは、証明対象外のforce-coupled runとforce-coupled RZ caseをprepare時に明示拒否した（RZは後続P06-RZで解禁）。不十分なendpoint chordや
midpoint差をno-hitの証明としない。

## P06 coupled event revision 2: certified constant acceleration

一般RK4曲線の前に、「固定電荷、Epstein dragなし、使用するfieldが全nodeで厳密に一定」というprepare時に
証明できる範囲だけmaterial boundaryを解禁する。電気力と重力・浮力の合成加速度は粒子ごとに一定であり、
RK4位置pathは次の放物線と厳密に一致する。

\[
\boldsymbol x(t)=\boldsymbol x_0+\boldsymbol v_0\Delta t+
\tfrac12\boldsymbol a\Delta t^2,
\qquad
\boldsymbol v(t)=\boldsymbol v_0+\boldsymbol a\Delta t.
\]

| 項目 | revision 2の判断 |
|---|---|
| 解く利用case | `cartesian_xy`、fixed charge、dragなし、一様electric/density。topology-completeな材料boundaryの`stick`/`escape`、またはfully-supported regular box内のboundaryless motion |
| 既存modelで解けない理由 | endpoint chordは壁を越えて戻るturning pathとnear-grazing contactを見落とす |
| 所有module | `fields.py`が厳密一様性、`integrators.py`がquadratic path state、`events.py`がparabola-line first hit、`engine.py`がevent→validity→commit順を所有する |
| path/broad phase | 材料boundaryではendpoint chordを`|a|h^2/8`だけ膨張し、facet交差を時間parameterの二次方程式で解く。boundarylessでは座標ごとの内部転回時刻と始終点から極値を求め、regular support boxの包含を各proposalで証明する |
| failure | 判別式、退化、残差がfloat64 budgetで決定不能ならno-hitへ丸めずrunを失敗させる |
| reference | turning/chord-miss、turning no-hit、near-grazing hit/miss、公開APIでのhit時刻・速度・output schedule不変性 |
| 未解禁 | Epstein drag、非一様field、continuous charge、force-coupled RZとmaterial boundaryの組合せ |
| 当時の後続gate | revision 3b、memory checkpoint、P06-Uを経てP07 boundary/sourceへ進む計画だった |

trial endpointやRK stageがfield support外でも、それより前のcertified wall hitがあればhit状態をcommitできる。
したがって実行順は`proposal → earliest event → no-hit行だけsupport/applicability判定 → endpoint commit`とする。
hit後frameのためにwall外trial pathを再評価せず、hit時刻の速度と電荷を同じproposalから得る。

revision 2の放物線eventは`coupled_rk4_engine_v5`、`coupled_rk4_proposal_v3`で
`line_quadratic_first_hit_v3`として維持していた。revision 3bのengine v6以降は一般RK4 piece判定を同じevent ownerへ追加し、
`line_quadratic_rk4_first_hit_v4`とした。P06 close時のevent v5は一般RK4のtransverse certificateだけを追加する。一様性のauthorityはcanonical node値の厳密一致であり、補間後の
加速度とのbit一致を二重検査しない。補間の加重和は数学的に一定でも最下位bitが変わり得るためである。
実stage samplingはsupportとapplicabilityの判定に残す。turning path、near-grazing hit/miss、接線・共線failure、
異なるmacro stepとoutput scheduleでの公開API結果を検証した。`quadratic_exact`はtopology-completeな材料boundaryが
連続pathのfield domain退出をfirst hitとして捕捉できるcaseで使える。boundarylessでは、全cell supportedな共通regular
layoutの閉じたaxis boxをsupport certificateとし、各proposalの各座標について始終点と内部転回点を解析評価する。
この極値がbox外なら、stage点がsupport内でもrunを失敗させる。frame有無で結果を変えない回帰を持つ。
revision 3aはboundaryless regular supportに限ってdrag・非一様場を解禁した。continuous charge、force-coupled
RZ、一般RK4と材料境界の組合せは引き続き拒否する。

## P06 coupled RK4 revision 3a: boundaryless enclosure

revision 3aは一つの縦切りとして完了した。一般曲線の全geometry対応ではなく、output scheduleがproductionの
成功可否を変えないための最小能力だけを固定した。

| 項目 | revision 3aの判断 |
|---|---|
| 解く利用case | boundaryless `cartesian_xy`、realized table source、fixed charge、既存の`epstein_linear_v1` / `electric_coulomb_v1` / `gravity_buoyancy_standard_v1`、全cell supportedな共通`RegularLayout` |
| 既存modelで解けない理由 | full-stepのRK stageとendpointだけでは、その間または出力用の短縮RK4がsupport・Epstein適用域を外れて戻らないことを証明できない |
| 所有module | `fields.py`がcanonical node extremaとregular support box、`physics/forces.py`がsample済み数値だけの純粋bound、`integrators.py`が全短縮RK4を覆うpath enclosure、`engine.py`が事前証明と唯一のproduction loopを所有する |
| enclosure | global field extremaとmodel係数から各stageの速度・加速度を再帰的にboundし、外向き丸めした位置下限・上限が任意の短縮時間の内部stageとaccepted endpointを含むことを要求する |
| applicability | Epsteinの`lambda/a`下限と`|u-v|/c_bar`上限を同じfield・速度boundから全区間で証明する。stage sampleだけをauthorityにしない |
| failure | regular support box包含、連続applicability、有限な外向きboundのいずれかを証明できなければ、output scheduleと無関係にrunをfail-closedにする。revision 3aではhidden subdivisionしない |
| reference | C02/C03の公開API解析時系列、frameなし・疎・密でのfinal bitwise identity、step途中releaseの粒子別残時間、安全な非一様regular field、短縮pathだけが逸脱するsupport反例、Epstein applicability反例 |
| 性能・memory | P06 NumPy referenceの同じbatch proposalにbound arrayを追加し、当時のresident/scratch下限へ含めた。P09がphase memory planで置き換え、compiled tileはP10が所有する |
| 未解禁 | `rk4_reintegrated`と材料boundaryの組合せ、P1/Q1 unstructured support、RZ force、continuous charge、Stokes–Cunningham |
| 置換・削除 | 一般`rk4_reintegrated`の一律prepare拒否を、上記証明済みsubsetだけに狭める。別engine、certificate framework、diagnostic traceは追加しない |

revision 3bは材料boundary用の保守的path tube、sequential accepted RK4 pieces、event-before-validityを同時に扱う。
先行wall hitが後続trialのsupport/applicability逸脱を救済し得るため、revision 3aのwhole-proposal事前拒否を
材料boundaryへ流用しない。unstructured supportとRZはそれぞれ別の包含・basis証明が完成するまで拒否する。

## P06 coupled RK4 revision 3b: sequential material-boundary path

着手前レビューで次を確定し、`coupled_rk4_engine_v6`として実装した。同じ始点から短縮時間だけ再実行する現行RK4の
endpoint曲線は、急峻な位置依存fieldで任意のparameter部分区間を局所時間幅とともに収縮するtubeへ安全に
変換できない。従って「一つのmacro proposalをparameter区間分割する」案は採用しない。

| 項目 | revision 3bの判断 |
|---|---|
| accepted path authority | geometry/support/applicabilityだけで決まる時間順のsequential dyadic RK4 piece列。各no-hit leaf endpointが次pieceの始点になる |
| enclosureの意味 | versioned integratorの離散pathを包む。真のODE解の離散化誤差は`h / h/2 / h/4`収束で別評価し、tubeへ混ぜない |
| event順序 | stage/endpointの非有限値は即時失敗。support/applicabilityはprovisional flagとし、`first event / valid prefix → remaining validity → commit`で確定 |
| output | frameは確定済みaccepted pieceから評価し、frame時刻でpiece partitionを変更しない |
| 最初のbound | revision 3aのglobal field/model boundを各piece始点で再構築する。local boundは候補率、分割深さ、処理時間の実測後だけ検討 |
| failure | refinement/interaction budgetで決定不能ならno-hitへ丸めず、明示的なindeterminate/failureにする |
| 未解禁 | surface source、reflection後residual、unstructured support、RZ force、continuous charge。P07のone-sided departureまで壁上初期点をnudgeしない |
| 実装形 | `events.py`が一pieceのclear/split/hit判定、integrator proposalが一piece、`engine.py`が時間順workとaccepted piece列を調停する。第二engine、再帰的巨大proposal、汎用certificate frameworkは作らない |

実装前hardeningは完了した。実軌道上で位置依存となる`E_x ∝ -x`調和振動子を公開APIで
`h / h/2 / h/4`実行し、位置・速度の観測次数3.5以上と、frameなし／内部frameありのfinal bitwise同一性を
確認した。放物線極値は`integrators.py`がdense stateと同じ演算順で評価し、式の絶対項scaleと演算回数に基づく
roundoff幅で始点・終点・内部転回点の区間を外向きに拡張して`engine.py`へ渡す。このsupport受理変更は
当時の`coupled_rk4_engine_v5`、path値が不変のproposalは
`coupled_rk4_proposal_v3`とする。

frameなし／macro終端／interior frameとboundaryless／material-hitの小型performance baselineは、三公開APIを
通る非gating測定として`tests/performance/p06_baseline.py`が所有する。revision 3bはaccepted particle-piece数、
candidate query数、refinement数、最大深さをmanifestへ集約し、frame scheduleでは値を変えない。baseline v3は
engine、proposal、event、enclosureのrevisionをすべて記録する。簡単な横断hitでも既定`geometry_rtol=1e-12`では
最大深さ40となり、scalar candidate処理はboundaryless RK4より明確に高価だった。

`coupled_rk4_engine_v7`はこの数値意味論を変えず、一般RK4材料候補を最大256粒子のchunkへ分け、
各waveで各粒子のleft-firstな未処理pieceを1件だけ出して完全一致するtarget timeごとにproposalする。
event v5は`integrators.py`がcomponentwise chord deviationとroundoff幅を所有し、`events.py`が単一facetの
外向き横断、normal time bracket、tangent/time-shiftを含むposition radius、endpoint clearanceを証明する。
証明できなけれrefinementをsplitし、十分小さいpieceは従来のfull-tube budgetへfallbackする。
engine v7、proposal v3、enclosure v1、schema/APIは変更しない。

64粒子の同一machine baselineで、material medianはevent v4の0.9666176 sからevent v5の
0.6450654 sへ短縮し（約1.50倍）、boundaryless medianは0.0471402 s、比は21.77から13.6840となった。
event v4の別再測は0.9617038 sだが、比較値には0.9666176 sを使う。初期scalarの2.3706326 sからは約3.68倍で、
accepted piece / candidate query / refinement / 最大深さは1088 / 2496 / 1408 / 21である。
これはlocal observationであり性能gateではない。同一公開scenarioの`tracemalloc`測定では、frameなしの
64/256/1024粒子が669,511/2,142,293/7,118,215 Bから356,077/913,336/2,108,319 Bへ減少した。
interior frameありは369,536/955,316/2,286,835 Bである。NumPy buffer追跡を校正済みだがprocess RSS、
`load_case`、native HDF5 memoryは含まない。P09はその後、solver-owned phase planと外部のfresh/warm
RSS characterizationに分けてこの空白を埋めた。これをもって
P06 revision 3bをcloseし、local bound/compiled kernelは独立した測定なしに先行しない。

材料壁のproduction subsetはCartesian XY、fixed charge、terminal `stick`/`escape`である。required fieldは
fully-supported common `RegularLayout`、またはparticle volume meshと完全一致するfully-supported P1/Q1を使う。
padded enclosure AABBがBVH facetから分離するか、AABB候補facetの支持線とtubeの法線方向intervalが
facet別event budgetを含めても厳密に分離するpieceだけをno-hitとして受理する。候補pieceは
左から逐次二分し、endpointがboundary/outside、candidate facetが一意、時間幅とtube径がroot event budget内の時だけ
hitにする。hit速度・電荷は同じRK4をhit時刻まで再実行して得る。cross-return/tangent/cornerが上限まで曖昧なら
`indeterminate`で失敗し、no-hitへ丸めない。調和振動子の壁到達時刻・速度は観測次数3.5以上、frame有無で公開event、
final、refinement集計がbitwise同一である。

## P06-U：exact-mesh P1/Q1 material domain

| 項目 | 判断 |
|---|---|
| 解く利用case | Cartesian XY、fixed charge、既存force、全cell supportedでparticle volume meshと完全一致するP1/Q1、topology-complete material wall、terminal `stick`/`escape`の一般`rk4_reintegrated` |
| 既存modelで解けない理由 | revision 3bはcontinuous support判定をregular axis boxへ限定していた。single-point P1/Q1 locatorだけではboundarylessな非凸supportの連続包含を証明できない |
| 所有module | `fields.py`がexact-mesh coverageとnodal extrema、`integrators.py`が既存RK4 enclosure、`geometry.py`/`events.py`がcomplete boundaryとtube、`engine.py`がcapability arbitrationとevent-before-validityを所有する |
| reference | 二三角形P1と非affine mapped Q1の調和振動子wall hit。hit時刻/速度order 3.5以上、frame schedule identity、壁外trialより先のevent、boundaryless negative rejection |
| 性能・memory | P1/Q1 locatorはこのP06-U時点ではcell走査するNumPy referenceであり、大mesh性能を主張しなかった。P10はstrict-interior hintとNumba内full searchで同じ意味論をcompiled化し、walk/BVHは追加しなかった |
| 置換・削除 | 証明済みsubsetのregular-only拒否だけを削除する。第二sampler、第二engine、boundaryless fallbackは作らない |

`coupled_rk4_engine_v8`はframeを消費しないaccepted proposalを保持せず、frameを含むmacro-stepでもその時刻と
重なるproposal rowだけをreplay用に保持する。同じrevisionで上記P1/Q1 material capabilityを解禁した。
proposal v3、enclosure v1、event v5、schema/API、trajectory/eventの数値意味は変更しない。この時点の次の能力gateはP07だった。

## P07：wall/source slice

| 項目 | 判断 |
|---|---|
| 解く利用case | table粒子のexact linear/quadratic pathとCartesian XY一般RK4を一つのengineで処理する。exact pathはcorner・複数hit・RZ axis crossing、一般RK4は厳密内向きinterior-facet surfaceとsingle-facet active responseの残時間を含む |
| 既存modelで解けない理由 | P06-Uまでは壁上release、非terminal wall response、物理event ordinalに基づく乱数、hit後残時間、RZ axis通過を拒否していた |
| 所有module | `sources.py`が単一`ParticleSchedule/realize_sources`、`rng.py`がcounter/draw kind、`boundaries.py`が応答、`coordinates.py`がaxis fold、`events.py`が残時間budget、`engine.py`が唯一のwork loopを所有する |
| source subset | fixed release time、`edge_fraction`またはuniform位置。XYは`line_length`、RZは`meridional_length`または`revolved_area`。速度は固定vectorまたは固定speedのdomain内向き法線 |
| wall subset | 静止壁のstick/escape/specular/probabilistic stick。priority最小subsetを選び、同一反射則は入射中法線の正規化合成法線へ一回だけ適用する |
| RNG | `philox4x32_10_v1`。sourceはsource ID・source内ordinal・draw kind、wallはparticle ID・zero-based physical boundary ordinal・law streamから決め、refinementや出力scheduleをkeyにしない |
| event/path | exact linear/quadraticとCartesian XY一般RK4で非terminal応答後の残時間を新しい壁状態から進める。interaction countはparticle residual stateで共有し、上限後も実際の次hitがある時だけ残区間を二分する。一定加速度の内向きdeparture tokenは別wall hitまで保持する。一般RK4はinterior start-contactと区間速度包絡から厳密内向きを証明した単一facetだけを除外する。ballistic RZ axisはwall eventを作らず基底foldする |
| reference | C08 surface departure/reimpact、C09 5反射とcap-driven split/failure、C10 combined-normal/priority、Philox既知vector、RZ回転面積sampleとaxis crossing、一定加速度surfaceのone-sided case、一般RK4 surfaceの再衝突・tangent failure・specular residual・cap split・右連続frame |
| 未解禁 | force-coupled RZ、角度・速度・時刻の確率分布、moving wall。一般RK4のtangent、facet端点/corner、または曖昧なstart-contactは証明不能としてfail-closed |
| 置換・削除 | `TableSchedule/build_table_schedule`とterminal専用boundary経路を残さず、単一scheduleと一般wall responseへ置換する。位置nudge、第二engine、RNG状態配列は作らない |

`coupled_rk4_engine_v9`はC08～C10のforce-free exact-path sliceを公開APIへ追加し、
`coupled_rk4_engine_v10` / event `line_quadratic_rk4_first_hit_v6`はCartesian XYの証明済み一定加速度surfaceへ
同じ意味論を拡張した。surfaceのrealized scheduleは位置とcanonical
`facet_id`を保持し、ownerと法線はprepared geometryを唯一のauthorityとする。壁上初期位置は動かさず、初速度が
domain内向きならdeparture、壁向きなら時刻0の通常impactとする。`v·n`の明確な符号を常に優先し、budget内の
非零値は曖昧として拒否する。厳密tangentだけは`a·n`をscale-awareに分類する。内向きdeparture tokenは別wall
hitで速度が変わるまで保持し、event側が各intervalでsource supporting lineの内側/境界band、tangentまたは明確な
内向き速度、内向き加速度を再証明した時だけsource facetを除外する。他facetは常に通常検索し、別wall応答後は
source facetも通常検索へ戻す。
外向き加速度ならzero-time impactとし、terminal stick/escapeは許すが、反射で内向きdepartureを作れない場合は
fail-closedにする。hit位置はlocatorが返したcanonical wall位置へcommitし、反射後に固定距離を足さない。

`coupled_rk4_engine_v11` / event `line_quadratic_rk4_first_hit_v7`でCartesian XY一般RK4へこのsliceを
拡張した。surfaceはfacet端点budgetから離れ、明確に内向く初速度だけをdepartureとして受理し、facet端点/corner、厳密tangentまたはroundoff幅内の曖昧な
法線速度はfail-closedとする。surfaceまたはwall応答後のactive stateは、start-contact facetの内点と区間全体の
法線速度上界が厳密内向きであることをeventsが証明した時だけそのfacetを候補から外す。証明不能な候補は保持して
left-firstに細分化し、残時間は応答後stateから同じmacro targetまで進める。interaction capは同じparticle stateで
数え、上限に達しても次hitを実際に検出した時だけ残区間を分割する。activeな速度jumpとterminal transitionは
event時刻のframeへpost-stateを上書きするため右連続である。これはP07全体の完了を意味しない。

P06-RZはこのP07のwall/source意味論を変更せず、同じwork loopへforce-coupled RZのaxis eventを追加した。

specularは法線反発係数`(0,1]`と接線反発係数`[0,1]`を必須とする。`probabilistic_stick`は定数確率`[0,1]`を
一回だけ比較し、非付着側を明示したspecularへ渡す。永続`event_ordinal`と乱数用physical boundary ordinalは別の
stateだが、現行resultではreleaseが0で境界eventが連続するため、draw referenceはmanifestのseed・streamと
`particle_id,event_ordinal-1`から再構成できる。manifestはsource/RNG/boundary revision、draw kind、解決済みlaw、
wall event・residual split・axis crossing件数を記録する。case/result schema v1のdatasetは変更しない。

## P06-RZ：force-coupled axisymmetric meridional motion

| 項目 | 判断 |
|---|---|
| 解く利用case | `axisymmetric_rz + axisymmetric_rz_meridional`で、fixed chargeと既存Epstein/electric/gravityを一般`rk4_reintegrated`として解き、材料壁と`r=0`通過の残時間を一つのengineで処理する |
| 既存modelで解けない理由 | canonical RZは`r >= 0`だが、RK4 stageを軸で単純にclamp/foldすると滑らかなODEを壊す。XY専用vector metadata、signed pathをそのままsupport判定する実装、ballistic専用axis時刻ではforce-coupled軌道を認証できない |
| 所有module | `physics/catalog.py`が座標別vector要求、`fields.py`が単一の`axis_accessible`判定とaxis node regularity、`coordinates.py`がsigned/canonical基底変換と区間像、`integrators.py`が既存RK4/enclosure、`events.py`がaxis局在、`engine.py`がwall/axis順序・commit・残時間を所有する |
| 必要field/state | RZ vectorは`components=(r,z)`, `stored_basis=axisymmetric_rz`。geometryが軸へ接する、またはboundaryless fully-supported regular boxが`r_min=0`ならaxis-accessibleであり、全required vectorの軸node radial値とradial gravityを厳密に0とする。resident stateは常に`r >= 0`、trialだけsigned radial chartを使う |
| 対応座標・integrator | `axisymmetric_rz_meridional`の一般RK4。case/result schema v1、proposal v3、global-abs enclosure v1、既存の力式は変更しない。一定加速度specializationはXYだけに維持する |
| reference | zero-radial-gas Epstein解`rho=r0+tau*w0(1-exp(-t/tau))`, `w=w0 exp(-t/tau)`をfoldした解析軌道、`h/h/2/h/4`の4次収束、frame schedule identity、軸上不変状態、C02～C05 away-axisのCartesian退化一致、axis→wall順序、boundaryless regular support由来のaxis accessibility |
| 性能・memory | 新しいresident配列や第二engineは追加しない。stage batchの一時的なcanonical位置・速度とradial sign、既存refinement workだけを使う。axis候補だけが既存left-first細分化を受ける |
| 置換・削除 | physics catalogのforce-coupled RZ一律拒否を削除する。axisを偽の材料facet、位置nudge、RZ専用integrator、別support/applicability経路として実装しない |

軸通過はmaterial boundary eventではない。局在したprefixを同じRK4で再積分してvalidityとradial残差を検査し、
`r=0`でradial velocityの基底だけをfoldして同じtarget時刻まで続行する。wallとaxisの双方が局在済みで時間不確かさ
区間が重なる時はwallを優先し、いずれか一方でも未決定ならpieceを分割する。axis端点とmaterial cornerの完全tieは
一般RK4 cornerとしてfail-closedであり、wall優先で任意の一面へ丸めない。axis foldは`axis_crossings`だけを増やし、boundary event、
boundary ordinal、wall RNG draw、interaction countを消費しない。

## P06-S：small physics runtime and Stokes--Cunningham

| 項目 | 判断 |
|---|---|
| 解く利用case | air条件を明示した`stokes_cunningham_allen_raabe_air_v1`をXY/RZの既存一般RK4経路で解く |
| 既存modelで解けない理由 | free-molecular前提のEpsteinを連続・slip域へ黙って外挿できず、気体別相関とKn定義を明示する必要がある |
| 所有module | `physics/catalog.py`がschema、`physics/forces.py`が式、`physics/runtime.py`がsample済みprimitiveの合成・適用域・bound、engineがsampling/event/state commitを所有する |
| 必要field/state | gas velocity、density、dynamic viscosity、mean free pathと粒子mass/drag diameter。新しいresident stateは追加しない |
| 適用域 | 半径基準`0.03 <= Kn_a=2 lambda/d <= 7.2`、`Re_p <= 0.1`、正のdensity/viscosity/mean-free-path。範囲外でEpsteinへ切り替えない |
| 対応座標・integrator | Cartesian XY / axisymmetric RZ meridionalの既存`rk4_fixed`、`rk4_reintegrated`。第二integratorを作らない |
| 解析解またはreference | 一定primitiveの解析的線形緩和に対するXY 4次収束、frame schedule final不変、away-axis RZ parity、公開Kn反例とunit Kn/Re反例 |
| 性能・memory影響 | 既存required field配列と粒子別rate boundだけを使い、新しい履歴配列を持たない。field samplingをmodelごとに重複しない |
| 置換・削除する旧経路 | engine内のdrag別bound/evaluate分岐を小さいimmutable runtimeへ集約し、registry、DI、汎用plugin層は追加しない |

## P08：Stage 1A closure

| 項目 | 判断 |
|---|---|
| particle-local failure | 数値event budget、event/boundary/departure不決定、動的なfield support/model applicability、非有限particle physicsを当該粒子のfailed terminalへ変換する。最後の有限局在stateを保持し`kinematics_valid=0`とする |
| run-fatal | 入力/schema、静的・共有field layout/model係数/bound、topology、writer/publicationなどrun全体の意味を失う問題は即時停止する |
| output | failure event、macro-step lifecycle count、明示ID/timeのstate probeを既存segmentへ追加する。全force、全stage、全refinement traceは保存しない |
| CLI | `check CASE`、`run CASE -o OUT`、`inspect OUT`だけを提供し、`load_case`、`simulate`、`open_result`以外の実行経路を作らない |
| determinism | failureはphysical wall ordinal/RNGを消費せず、frame/probe scheduleはproduction stepを分割しない。survivorとall-failed runのcomplete publicationをscenarioで確認する |
| revision | engine v14、physics catalog v3、physics runtime v1、result algorithm v2。未releaseのcase/result schema versionは1を維持する |

## P09：memory/runtime layout

| 項目 | 判断 |
|---|---|
| 解く利用case | 10^4～10^6粒子へ拡張できるよう、全particle stateをresidentに保ったままproposal scratchを設定memory上限内のmicrotileへ分ける |
| 所有module | `case.py`/`case_format.py`がmetadata preflight、`sources.py`がdirect schedule scatter、`cpu.py`がactive index/microtile/memory plan、`engine.py`が単一loopでのtile適用とstable merge、`output.py`がbounded writer cacheを所有する |
| state layout | position/velocity等の既存resident state列はparticle ID対応を固定する。容量Nのresident-row active indexだけをsortedに保ち、in-place stable compactする |
| memory semantics | `solver_owned_memory_plan_v1`はload/prepare/run phase peak、component内訳、microtile幅を示す。`memory_limit_mb`はsolver-owned predicted peakの上限でありOS hard RSS capではない |
| input/check | YAML/resourcesを先にparseし、HDF5 metadataからcanonical numeric footprintを求めてpayload前に拒否する。CLI `check`はこのnumeric bytes/上限だけを報告し、完全なprepared planは`run.json`が所有する |
| determinism | microtile幅によらず、tile-local event/failure/proposalを物理keyでstable mergeし、final/event/RNG/output identityを維持する |
| measurement | 手動`p09_memory.py`がfresh/warm processのload/prepare/run RSSを10k/100k/1Mで測る。production dependency、background monitor、絶対時間閾値は追加しない |
| P10へ移したもの | 現行regular locatorが消費しないper-layout cell hintは先行配列化せず、P10のcompiled samplerと同時に、実際に消費するP1/Q1 strict-interior hintだけを追加した |
| revision | engine v15、CPU runtime layout v1、memory plan v1。物理、event、proposal、case/result schemaの意味は不変 |

この節はP09完了時点の履歴である。P10は次節、P11はその次の節のとおり完了した。

## P10：compiled CPU

| 項目 | 判断 |
|---|---|
| 解く利用case | P09のresident SoA/microtileを保った一つのproduction engineで、field sampling、sample済みprimitiveのphysics、classical RK4算術をcompiled array passとして実行する |
| 所有module | `cpu.py`がcompiled field pass、`physics/compiled.py`がcompiled physics pass、`integrators.py`がcompiled RK4算術、`engine.py`がtime/event/accepted state/writerを所有する |
| locator | regularはsupported containing-cell common pathだけ軸indexからO(1)個の候補を評価する。outside/masked provisionalはcompiled全cell走査を使う。P1/Q1はaccepted endpointのprevious-cell strict-interior hintをfast pathとし、初回、hint miss、共有面は同じNumba kernel内のfull searchへ戻る |
| hint commit | hintは非物理stateであり、accepted full endpoint、wall/axis prefix、residual pieceのaccepted endpointだけをresidentへcommitする。trial、`state_at()`、frame/probe/output sampleは変更しない |
| determinism | Numba 0.67、NumPy `<2.6`、`fastmath=False`、`parallel=False`。scalar evaluatorはverification oracleだけでproduction fallbackにしない |
| parity | `state_at()`、wall hitまでの再積分、hit後residual、trajectory/frame/probe/finalを同じproposal/event経路へ通し、case/result schema、proposal v3、event v8、field location v2を変更しない |
| measurement | 手動harnessで空cacheのcold JITとsame-process warmを分離し、semantic digest、RSS、revision、speedupを記録する。絶対thresholdは置かない |
| revision | engine v16、compiled CPU tile v1、CPU runtime layout v2、memory plan v2、physics runtime v2 |
| 後続 | 初回stageのP1/Q1はhintなしfull searchなので、large-mesh性能をP10完了とは主張しない。P14でrealistic cell count、initial localization、cross-cell motionを測り、支配的だったsupported containmentへBVHを追加した。P11は次節で完了し、P12/P13/P14はP10の範囲外 |

## P11：native exponential midpoint

| 項目 | 判断 |
|---|---|
| 解く利用case | EpsteinまたはStokes--Cunninghamの線形dragが運動時間刻みより速いcaseを、RK4の安定性上限へ縛られず、固定stepの指数更新で解く |
| 既存modelで解けない理由 | `rk4_fixed`は既知の線形緩和で`dt/tau >= 2.5`を安全上拒否する。一方、線形dragを陽RK4で解くためだけに極小stepを要求すると製品の主要なnano-particle caseを非効率にする |
| 所有module | `physics/runtime.py`と`physics/compiled.py`が全加速度と同じ一回のmodel passでdrag rate、target velocity、加算加速度を返す。`integrators.py`が指数更新、`StepProposal.state_at()`、連続path enclosureを所有し、`engine.py`が既存event loopとaccepted stateを調停する |
| 必要field/state | 既存required fieldとresident `(x,v,Z)`だけを使う。physics stage値は`linear_drag_rate_s_inv`、`target_velocity_m_s`、`additive_acceleration_m_s2`へ分解し、同じdrag式をintegratorへ複製しない。現行P11は`dZ/dt=0`だけを受理する |
| 対応座標・integrator | `cartesian_xy`と`axisymmetric_rz_meridional`の`exponential_midpoint`。線形dragなしのexact specializationも同じengineに残す。材料boundary、wall残時間、RZ axis fold、output replayはRK4と同じ`StepProposal`/event経路を使う |
| 解析解またはreference | C03一定係数を`dt=[0.75,0.375,0.125] s`かつ要求全frameで丸め誤差精度、`h/tau`の0近傍から非常に大きい値までfinite、smooth可変係数で観測次数1.8以上、output schedule identity、Stokes一定primitive閉形式、material/RZ event順、surface departure後の同面再衝突を検査する |
| 性能・memory影響 | field/physicsはstart predictorとmidpointでcompiled tile評価する。resident state、active index、result schemaを増やさず、提案中のrate/target/additiveとenclosureだけを既存bounded microtile scratchへ収める。全規模throughputとevent-heavy bottleneck判断はP14/P12が所有する |
| 置換・削除する旧経路 | RK4専用engineを並存させず、engine名をmethod-neutralなv17へ更新し、曲線pathのproposal/event orchestrationを共有した。RK4の`dt/tau`gateを指数法へ適用する分岐と、physics式をintegrator側で再計算する案を採用しない |

P11完了時点のrevisionは`deterministic_particle_engine_v17`、`compiled_cpu_tile_v2`、
`coupled_fixed_step_proposal_v4`、`line_quadratic_rk4_axis_first_hit_v9`、
`deterministic_compiled_physics_runtime_v3`、`exponential_midpoint_v1`、
`exponential_midpoint_global_abs_enclosure_v1`である。case/result schema、physics catalog、field semanticsは
変更しない。非零charge rateは黙ってfixedとして扱わず明示的に拒否し、continuous charge連成の実装時に
同じ状態更新として追加する。COMSOLはこの判断にも依存せず、比較は外部V&Vだけが所有する。次の能力gateはP12の
event-heavy residual workとstable mergeであり、次節で完了した。

event v9の指数pathはposition enclosure全幅をchord偏差へ流用せず、全短縮secantを含むvelocity enclosureから
`h * (v_upper - v_lower) + roundoff`を外向きに作るmethod-neutral boundを使う。固定回帰の最大refinement深さは
材料反射22、surface departure後の同面再衝突19で、双方に上限24を置く。これは性能秒数でなくcertificate退行を
防ぐ小さい数値回帰である。

## P12：event-heavy CPU parallel

| 項目 | 判断 |
|---|---|
| 解く利用case | 材料壁へ多数粒子が到達し、first-hit refinement、反射/付着、残時間継続がfield/physics tileより支配的になるevent-heavy caseを、同じpublic APIと単一engineで複数CPU workerへ分配する |
| 既存modelで解けない理由 | P11までは要求thread数を記録してもresolved thread数は1で、BVH query、piece判定、hit-prefix再積分を粒子ごとのPython調停へ戻していた。field/physics/RK4だけをcompiled化してもevent-heavy workloadの大半が直列に残った |
| 所有module | `cpu.py`がworker数・worker scratch・memory plan、`geometry.py`がcompiled read-only BVH query、`events.py`が保守的clear/split事前分類、`engine.py`が一粒子一worker、tile wave、worker-local residual/event/statistics、tile順stable mergeを所有する。`output.py`はmain threadだけが呼ぶ |
| 必要field/state | resident物理stateは増やさない。field/geometry/physics runtimeはread-only共有し、各workerは自分のtileのproposal、geometry-query、residual、event、failure、集計だけを持つ。RNG keyとparticle-local ordinalは従来どおりparticle identityに結び付ける |
| 対応座標・integrator | Cartesian XY/RZと`rk4_fixed`/`exponential_midpoint`を既存`StepProposal`/event loopのまま扱う。compiled事前分類は「明らかなclear」または「必ずsplit」だけを認証し、hit/corner/曖昧pieceは従来のscalar locatorへ渡す。同時刻wall prefixだけを同じcompiled proposalへbatch化する |
| 解析解またはreference | 1/2/4 threadでfinal、release/boundary/failure、series、frame/probe、RNG、event work countを完全一致させる。compiled BVHはcanonical facet順、inclusive overlap、scalar oracleとの候補一致を検査する。既存C01～C10とP11解析oracleを変更しない |
| 性能・memory影響 | `resources.threads`は上限で、粒子ありなら`min(requested, particle_count)`をresolved worker数とする。`workers * microtile * proposal scratch`とworkerごとのBVH stack/candidate bufferをplanへ計上し、不足時はthread数を黙って減らさず開始前拒否する。512粒子×4 stepのwarm medianは1/2/4 threadで0.440460/0.528229/0.632361 s、着手時serial 3.066786 s比でP12当時の1 threadは6.96x。thread scaleは負なので絶対性能gateにはせず、残るPython/GIL調停と全matrix判断をP14へ渡す |
| 置換・削除する旧経路 | Python BVH traversalをproduction pathから外し、粒子ごとの明白なclear/split判定と同時刻wall-prefix再積分をcompiled batchへ統合した。別scheduler、別parallel engine、workerからのwriter呼出し、memory不足時のsilent worker downgradeは追加しない。case/result schema、proposal v4、event v9、physics runtime v3は変更しない |

P12完了時点のrevisionは`deterministic_particle_engine_v18`、`compiled_cpu_tile_v3`、
`line_boundary_bvh_v3`、`resident_soa_worker_microtile_v3`、`solver_owned_memory_plan_v3`である。
後続P13がmulti-segment、checkpoint/resume、failure injection、single-writer bounded queueを追加して完了した。
P12のmacro内accepted/event stagingは予測量であり、process RSSまたはpathological residual workを含む真のhard boundとは
主張しない。P14は10k/100k/1M、mesh、hit数、output量、thread数の全matrixとproduct-level性能判断を所有する。

## P13：durable segmented result

| 項目 | 判断 |
|---|---|
| 解く利用case | 長時間・多数粒子runを閉じたepochへdurableに保存し、process/I/O failure後も最後の確定macro barrierから重複・欠落なく同じ`simulate(case, output)`で再開する。未完了resultは明示recoveryで確定prefixだけを調査できるようにする |
| 既存modelで解けない理由 | P12までは一つのsegmentを最終化時にだけ完成扱いし、中断すると再利用できなかった。worker-local eventをmacro全体へ保持する予測stagingもhard boundでなく、checkpoint、commit point、partial readerがなかった |
| 所有module | `engine.py`が固定epoch barrier、worker-wave stream、checkpoint stateのcapture/restoreを所有する。`output.py`が単一HDF5 writer thread、容量1 queue、segment/A-B/LATEST/final commit、hash検証、`ResultView`を所有する。`cpu.py`はworker output stagingをmemory planへ計上する |
| 必要field/state | macro時刻/step、release/frame/probe cursor、position/velocity/charge/lifecycle/failure/terminal、active resident-row index、logical/physical event ordinal、exact origin、surface-contact state、P1/Q1 cell hint、event aggregate。resume identityはinput hash、全schema/revision、座標・method・model、particle ID集合を含む |
| 対応座標・integrator | P13当時のCartesian XY/RZ、`rk4_fixed`、`exponential_midpoint`を同じengine/output経路で扱った。当時のcadenceはoutput schedule・thread・microtileから独立した固定64 macro-stepで、最終macro後もcommitした。現行supersessionは後段のwork-scaled decisionを参照する |
| 解析解またはreference | 130 macroの小型公開API caseで3 segmentを作り、release/boundary/failure/series/frame/probe/finalを全segmentから統合する。最初の`LATEST`以前、segment/checkpoint/`LATEST`後、final/run.json/`_SUCCESS`/directory publication後へfailureを注入し、確率wall ordinal、orphan、segment構造/count、checkpoint/hash破損、通常runとresumeの全公開payload・科学manifest raw identityを照合する |
| 性能・memory影響 | worker waveごとにevent/failureをstreamし、前waveを次waveまで保持しない。HDF5 commandは容量1 queueへ投入してackを待つため、disk遅延はcomputeへbackpressureする。これはI/O overlapの高速化ではない。memory plan v4は`worker_output_staging`を`output_buffer`とproposal scratchから分離して計上する。製品規模I/O throughput/RSSはP14で測った |
| 置換・削除する旧経路 | 同期一segment writerとrecovery拒否を同じ`ResultWriter`/`ResultView`経路で置換し、互換reader、checkpoint migration、第二writer、再開専用API、汎用transaction frameworkを追加しない。`LATEST`より新しいorphanは無視し、参照済み破損はfail-closedにする。power loss、remote filesystem、同じOUTへの複数process同時実行は保証しない |

P13完了時点のrevisionは`deterministic_particle_engine_v19`、`durable_segmented_result_v3`、checkpoint schema 1、
`solver_owned_memory_plan_v4`である。case/result schema 1、compiled tile v3、runtime layout v3、geometry v3、
proposal v4、event v9、physics catalog/runtimeは変更していない。verification/scenario 322件が合格し、当時の次gateを
P14の10k/100k/1M、mesh、hit数、output量、thread数を含むproduct-level性能判断とした。過去segmentは構造と累積countを
検証するが、hash対象は参照checkpointと最新segmentであり、同shapeの過去値改変はP13の検出契約外である。

## P14：product-scale performanceと二つのlocator index（履歴）

| 項目 | 判断 |
|---|---|
| 解く利用case | P14当時の10k/100k/1M粒子、regular/P1/Q1、初期/cross-cell、event量、output量、cold/warm、1/host workerを公開APIで直交比較し、v0.1 solver-coreの実行特性を記録する |
| 既存modelで解けない理由 | P09～P13は局所または小型caseだけで、初期P1/Q1 locationとtable start validationがO(粒子数×cell数)で支配的だった。writer単体、thread数、RSS/solver planも製品規模で未判定だった |
| 所有module | `tests/performance/p14_matrix.py`が外部harness、`fields.py`がsupported-containment BVH、`geometry.py`がvolume-cell BVHと局所解像性、`cpu.py`がresident/build transient計上を所有する。`engine.py`/`output.py`へtimerや別実行経路を追加しない |
| 必要field/state | indexはprepared read-only arrayだけで、粒子物理stateは増やさない。accepted cell hintは既存stateを維持する。geometryはCPython `math.hypot`で準備したedge長を保持し、scalar/compiled predicateを一致させる |
| 対応座標・integrator | 現行Cartesian XY/RZ、`rk4_fixed`、`exponential_midpoint`を同じengineで扱う。indexはintegrator非依存で、case/result schemaを変えない |
| 解析解またはreference | 全23行のhard checkと科学payload identity、field旧predicate oracle、geometry scalar/compiled parity、large-offset/high-aspect/共有面/hole/disconnected/RZ seam、未解像tri/quad fail-closed、verification/scenario 336件 |
| 性能・memory影響 | P14当時の3観測medianでregular 1 worker 10k/100k/1Mは0.095647/0.783874/7.534041 s、20 worker 100k/1Mは1.8796x/4.7965x。event 10k×20 hitは20 workerが0.8882xで遅く、当時は1 workerを既定推奨とした。このworker選択は後続P14-Pで削除済みである。memory plan v6はfield 256 B/cell、geometry 1,024 B/cell build transientと両index residentを数える |
| 置換・削除する旧経路 | supported P1/Q1 initial/missとgeometry volume包含の無条件full scanをindex候補＋同一exact predicateへ置換する。field outside/masked最近傍full scanは意味論上残しO(cell数)制約を明記する。第二engine、production profiler、汎用cache framework、T04 preprocessorは追加しない |

P14はabsolute threshold、COMSOL速度比較、純writer帯域を主張しない。P14当時は20 workerが大規模regular/event-lightでのみ
大きな正のscaleを示し、unstructured 10kでは小さくevent-heavyでは負だった。この履歴判断を現行host幅選択へ使わず、
solver内部のthread幅選択そのものを後続P14-Pで削除した。
P14は各軸を直交させたsynthetic baselineであり、非一様force、surface release、材料wall、一般曲線event、
多数macro stepを同時に使うtarget workloadを測っていなかった。また当時のouter worker-wave、particle別Python event調停、
stage配列割当て、worker数比例scratchを完成形とは扱わなかった。後続P14-Pで並列runtimeを削除して単一直列engineへ
収束し、P14-Uで代表用途、T03で最小解析・可視化を閉じた。P14-Rのremote Windows/Linux workflowも完了し、
P15 continuous chargeは旧着手blockerを解除して完了した。
T04は後続profileで
accepted-state cacheまたはremesh需要が示された場合だけ再開する。

## P14-P：parallel runtime convergence

この節は採否と所有境界のdecision recordである。実測profile、wavefront構造、実装順、全acceptance matrixの権威は
[`parallel_execution_plan.md`](parallel_execution_plan.md)とする。

| 項目 | 判断 |
|---|---|
| 解く利用case | 10^5～10^6粒子の非一様場計算と、5回以上の一般曲線wall eventを含むcaseを、thread増加に比例するscratchなしで複数CPU coreへ分配する |
| 既存modelで解けない理由 | P12/P14の`ThreadPoolExecutor` worker-waveはregular 1Mでのみ大きな効果を示し、event 10k×20 hitは0.8882倍へ低下した。exact/curved eventはPython list/dict/dataclassとscalar locatorで調停され、各stageの配列確保、wave barrier、worker別microtile scratchがGIL・割当て・memory律速を残す |
| 所有module | `cpu.py`がNumba thread数、thread非依存tile/scratchとmemory planを所有する。`engine.py`はmacro/release/output調停と一つのcompute入口だけを所有する。`integrators.py`/`fields.py`/`physics`はdisjoint rowへ書くcompiled pass、`events.py`はflat SoA wavefrontとfirst-hit/refinement、`boundaries.py`/`rng.py`は同じcompiled row leaf、`geometry.py`はread-only stackless queryを所有する |
| 必要field/state | resident物理stateと公開schemaは増やさない。scratchにrow別particle ID、始点/現在時刻、target time、residual、refinement depth、interaction/event ordinal、status、accepted hint、bounded event/failure bufferを持つ。一roundに一row高々一件を書き、stable compactionして再利用する |
| 対応座標・integrator | 現行Cartesian XY/RZ、`rk4_fixed`、`exponential_midpoint`を同じwavefrontで扱う。Numba内部thread layerだけを使い、outer executor、nested parallel、multiprocessing、thread-ID private API、第二backendを追加しない |
| 解析解またはreference | thread 1/2/4/8でfinal、release/boundary/failure、series、frame/probe、RNG/event ordinal、event/refinement workをbitwise一致させる。既存C01～C10、時間/mesh収束、exact/curved first hit、corner・axis・resume identityを維持する |
| 性能・memory影響 | warm同一machineで、代表100k粒子以上は4 thread/1 threadが1.8倍以上、curved event 5 hit以上は1.4倍以上、大規模caseで8 threadが4 threadより5%以上速い、8-thread solver-owned memoryは1-thread比1.15倍以下、置換後1-threadは置換前比10%以上遅くならないことを要求する。scratchはtile幅でboundedとしthread数に比例させない |
| 置換・削除する旧経路 | outer `ThreadPoolExecutor`、worker-wave/merge、worker-local scratch計画、Python object event queueを同じchangeで削除する。新旧schedulerをfeature flagで共存させない。一回のfocused correction後も科学同一性・性能・memory gateが未達なら、`resources.threads`を含むmultithreading API/config、旧code/test/文書を削除し、serial compiled engineへ一本化する |

P14-Pは自動thread tunerや一般task schedulerを作る工程ではない。regular passだけを並列化してevent-heavy pathを
Pythonへ残す部分完成も許容しない。必要なprofilingとfeasibility spikeは外部harnessで行い、productionへ二経路を
残さず、置換または削除までを一つのgateとして完了する。

v26試行ではouter `ThreadPoolExecutor`、future/wave merge、worker別scratch計画を削除し、内部Numba thread team、
thread非依存slab、preallocated workspace、stackless boundary BVH、同期single-owner writerへ移行した。
regular 1Mの1/2/4 thread実測は9.32/10.09/10.10 sで4-thread speedupが0.923xに留まり、事前gateを満たさなかった。
そのためcase schema v2から`resources.threads`を削除し、P14-P closeoutのv27 / compiled tile v6 /
`resident_soa_serial_slab_v5`へ一本化した。flat SoA wavefront、compiled boundary/Philox、row status、batch source、
direct replay、bounded stagingは直列runtimeにも有用なため維持する。正確な測定と削除範囲は
[`parallel_execution_plan.md`](parallel_execution_plan.md)が所有する。

v20のmachine-readable exact payload digest snapshotは保存されていないため、v20とのbitwise同一を後から捏造しない。
P14の文書化値と当時の336件を履歴上の移行証拠にし、v27全公開payloadをP14-P integration anchorとして
slab/output/repeatのidentityを判定する。

## P14-U：representative-use utility/performance gate

| 項目 | 判断 |
|---|---|
| 解く利用case | 現行modelだけを使い、part surfaceから放出した粒子が非一様場を横切り、材料wallへ到達する多数stepの代表caseを一つ通す |
| 既存modelで解けない理由 | P14はregular/P1/Q1、event、output、threadを直交比較したが、主用途の結合case、時間/mesh収束、global extremaの局所sheath影響、P14-P後の直列性能を同時には判定していない |
| 所有module | `tests/performance`または外部V&V harnessが計測を所有する。coreへproduction timer、diagnostic subsystem、第二engineを追加しない |
| 必要field/state | 既存canonical case/resultだけを使う。実測範囲はCartesian XY `line_length`のedge-fraction/uniform sourceに限定し、RZ `revolved_area`や任意surface実現値を検証済みとしない。`point_wall_laws_v4`とphysics catalog v4は先行して意味を固定済み |
| 対応座標・integrator | XYは固定64 x 64 mesh上の`rk4_fixed`時間系列と、`nx=ny`固定aspectのregular/P1/Q1 mesh系列を分離する。RZは軸横断を含むregularな可変場を使う |
| 解析解またはreference | 時間同期位置/速度、hit時刻・位置・同一mesh facet、各layoutのfine reference、target facet clearance、sourceとexact hitを含むdense path、RZ自己差分と実canonical field parity |
| 性能・memory影響 | 10k/100k/1Mのnone/sampleを各3 fresh processで測り、raw/median、RSS high-water、solver plan、event work、mode間core／mode内probe digestを保存する。1M none profileは別の非計時runとし、opaque ownerから局所最適化を推論しない |
| 置換・削除する旧経路 | 測定だけならproduction変更なし。scheduler、第二engine、診断framework、realized-surface sourceを追加しない。単一ownerの支配が実証された時だけ所有責務内の限定変更を別gateにする |

正式releaseは`../evidence/v0.1/p14u_release_v1.json`へ保存し、`release_gate_complete=true`となった。10k/100k/1M粒子の
`none`/sample各3 fresh process、別実行の1M `none` profile、XY時間・mesh収束、RZ収束/parity、失敗0、
release/target-stick件数、科学payload・revision・event workのidentityを満たした。1Mのmedian `simulate`は
`none` 550.43 s、sample 533.95 s、raw peak RSS最大は767.3 MiB、solver-owned planは614.5 MiBだった。
秒数は当該machineだけの非gating値で、COMSOL比やportable性能を表さない。profile self-timeはevents 28.7%、
fields 26.8%ほかへ分散し、単一owner支配を示さないためproduction変更を行わない。RSSはworker開始から
`load_case`/`simulate`/`open_result`完了までのprocess high-waterであり、後続する外部検証読込みのpeakではない。
`none`とsampleは順次実行なので、両者の差をwriter単体costと解釈しない。成果物はmachine-local原証拠であり、
再実行可能なrelease evidenceの保存はP14-Rが所有する。

## P15：continuous charge physics/numerics decision

| 項目 | 判断 |
|---|---|
| 解く利用case | 外部plasma primitiveを受け、単一価正イオンと電子の収集で変化する連続平均電荷`Z`と、`q=Ze`による電気力を同じ粒子軌道上でone-way連成する。最初のmodel IDは`oml_stationary_maxwellian_debye_huckel_v1`とする |
| 既存modelで解けない理由 | 現行`fixed`は`dZ/dt=0`だけであり、plasma状態に応じた帯電緩和と電気力feedbackを表せない。`model_dataset`の有効相対速度heuristicをcoreの真値へ昇格すると、外部V&Vとproduction物理の責務が混ざる |
| 所有module | `physics/charge.py`がOML rate、Debye--Hückel電位、平衡bracket、finite invariant、rate/derivative boundを所有する。catalogがmodelとrequired input、runtime/compiled passが同じstage式、integratorが`(x,v,Z)`更新、engineがdispatchとfailure集約だけを所有する |
| 必要field/state | 既存resident `charge_number=Z`を唯一のcharge authorityとし、sourceが`Z_0`を所有する。fieldは`n_e,n_i,T_e,T_i,u_i`、定数parameterは単一価正イオン質量、粒子属性は`electrostatic_radius_m`を使う。全量は正かつ有限、`M_i<=0.1`と`a/lambda_D<=0.1`、finite `[Z_min,Z_max]`、`abs(R_Z)`と`L_Z>=abs(dR_Z/dZ)`のboundを必須とする |
| 対応座標・integrator | XYとRZ no-swirlの既存basis/axis regularityへ対応する。最初に古典RK4で位置・速度・Zを同じ4 stageにより進め、その後native exponential motionへchargeを接続した。RK4は`h L_Z<=0.5`と全stage/endpointのinvariant包含を維持する。現行exponential pathは後段decisionのmidpoint-frozen affine exponential `J<=0`を使い、この値をstability gateにしない |
| 解析解またはreference | 電子・イオンbranchの単位・符号・`phi=0`での連続性、一意平衡とbracket、invariantとboundの包含を純粋scalar式で検証する。一定primitiveの独立高精度ODE、test-onlyの`Z'=-k(Z-Z*)`と一様Eの解析連成解に対するRK4 4次収束、explicit midpoint 2次、drift/Debye gate反例、XY/RZ・compiled parityを使う |
| 性能・memory影響 | 既存Z state、result、boundary event、checkpoint列を再利用しschemaを増やさない。bounded slabへcharge bound用の有限scratchだけを計上し、charge区間からelectric acceleration/path enclosureを構築する。slab幅・output schedule・checkpoint-resumeで科学payloadを不変にする。独立release trackだったP14-R remote CIも後に完了した |
| 置換・削除する旧経路 | `fixed` model自体は残し、fixed-only dispatchと非零charge-rate一律拒否を選択したcontinuous modelのstage経路で置換する。clip、floor、自動model切替、平衡置換、charge-only subcycle/operator split、implicit-midpoint stiff fallback、第二engine、第二charge state、汎用plugin/ODE frameworkは追加しない |

P15着手をexact P14-R Git baselineと初回remote CI成功まで禁止していた旧順序は、ユーザーの明示指示で解除した。
RK4-first slice、その受入後のexplicit midpoint sliceの順でproductionへ入り、case/result/checkpoint/event schemaは不変である。
続く外部M3-V relevance/applicability評価は完了した。現datasetへの全軌道比較は適用外であり、
stationary OMLの閾値を緩めたり互換modeを追加したりしない。後続判断は次節に記録する。

## M3-V：外部target applicability/relevance decision

| 項目 | 判断 |
|---|---|
| 解く利用case | Case P相当の外部plasma場とCase A相当のreduced electrostatic場を使う参照caseについて、production modelを実装する前に適用可能性と物理的関連度を判定する |
| 所有境界 | `tools/vv/comsol/`がCOMSOL直接監査とdataset評価を所有し、`chamber_particles`、core test、通常PR gateは依存しない |
| field mode | 製品上は`imported_external_plasma_fields`と`reduced_electrostatic`。legacyのCase P/A名をcore dispatchへ持ち込まず、どちらも同じcanonical fieldへ変換する |
| model選択 | ion dragはfield modeから独立したversioned physics modelとする。二つのdataset variantはmodel-form感度であり、自動fallbackまたは正解選択ではない |
| 直接監査 | COMSOL 6.4で二MPHを`loadCopy` / `-nosave`し、studyを実行せず、主要feature、selection、Case-A closure式をinventoryした。前後MPH SHA-256一致を必須とする |
| 数値結果 | 12 packageの構造、relative-drift two-current式、Epstein式、全保存時刻のvariant感度を評価した。式parityはprovenanceであり、物理妥当性の認証ではない |
| 適用判断 | P15 stationary OMLは全caseで適用外。Case Pは負イオンと非scalarな局所有効正イオン質量もP15契約に合わない。linear Epsteinはsampled coverage 0.542834～1。full production trajectoryは`NOT_APPLICABLE`、boundary・stochastic・reduced builderは`NOT_TESTED` |
| 後続 | field production F01/F02、relative-drift charge、P15-E finite-speed Epstein、P15-F collisionless ion drag、P16 Waldmann、Brownian数値基盤B01に続くB02 production sliceまで完了した。B02はCartesian XY・Epstein linear drag-only・fixed-charge state・terminal stick/escapeに限定し、state dimensionはP17を独立して進める |
| 削除・禁止 | COMSOL専用core経路、Case P/Aによるsolver分岐、比較差を消すclamp、第二engine、datasetをgolden truthとするtestは追加しない |

このapplicability/relevance決定の後、production coreを変更しないdeterministic matched sliceを外部toolで実施した。
Case-A 100 nm、固定電荷、Coulomb electric＋linear Epstein＋gravity/buoyancy、Brownianと追加力を無効化し、
canonical P1のnode値とtriangle connectivityをCOMSOL sectionwise形式で共有した。287粒子、0..0.4 ms、41 frame、
10/5/2.5 usの独立自己収束後にcross幅を事前登録し、2.5 us全時系列はposition RMS `8.643e-16 m`、
velocity RMS `1.999e-14 m/s`でPASSした。これはhash固定pre-event sliceの時間離散精度だけを認定する。
native finite-element場、材料event、dynamic charge、追加力、Brownian ensembleは別gateであり、元の12 package全軌道が
`NOT_APPLICABLE`である判断を書き換えない。証拠は
[`../evidence/m3v/matched_caseA_100nm_deterministic_v1.md`](../evidence/m3v/matched_caseA_100nm_deterministic_v1.md)にある。

## P15-D：species-constrained relative-drift charge decision

| 項目 | 判断 |
|---|---|
| 解く利用case | 外部または内部field producerが与える電子と単一・単価の正イオンprimitiveから、有限なイオン相対driftを含む連続平均電荷とCoulomb運動を同時に解く。model revisionは`oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1` |
| 既存modelで解けない理由 | P15 stationary revisionは`M_i<=0.1`を超えると拒否する。一方、M3-VのCOMSOL式をそのまま移すと、負イオン、局所有効質量、速度・energy floor、指数clipを暗黙に普遍化してしまう |
| 所有module | `physics/charge.py`がshifted-Maxwellian収集率とrun-wide bound、catalogがmodel入力、runtime/compiled passがstage式とpath applicabilityを所有する。integrator、event、output、checkpointは既存の共同状態経路を再利用する |
| 必要field/state | P15と同じ`Z,a,n_e,n_i,T_e,T_i,u_i,m_i`に、caseが有限正値`maximum_ion_drift_ratio`を明示する。これは`M_i=|u_i-v|/sqrt(8 k_B T_i/(pi m_i))`のrun-wide適用上限であり、速度をclampするparameterではない |
| 適用範囲 | 球形・等電位・完全吸収、衝突なし、非磁化、背景を変えない粒子、Maxwellian電子、単一shifted-Maxwellian正イオン、放出なし、`a/lambda_D<=0.1`、全状態で`M_i`が宣言上限以下、表面電位`phi<=0`。初期`Z<=0`と、全primitive/drift範囲で非正平衡が存在することをprepareで認証する |
| 数値解法 | RK4またはnative exponential midpointで`(x,v,Z)`を同時更新し、`[Z_min,0]`と連続pathのdrift/Debye gateを使う。RK4だけがexplicit `hL_Z<=0.5`を要求し、exponential pathは後段decisionのaffine exponential `J<=0`を使う。zero-drift極限はstationary OMLへ連続に一致し、小driftは解析級数で評価する |
| referenceと統合 | 3-D Maxwell速度積分による独立moment、zero-drift一致、解析微分と有限差分、primitive cornerでのbound包含、reference/compiled parityを検査する。両積分器の材料wall/frame、XY/RZ、checkpoint/resumeは既存公開scenarioを再利用する |
| 性能・memory影響 | 既存compiled row passへ`erf`を含む一つの選択branchだけを追加し、新しいresident列・scratch・schema・engineを作らない。catalog v6、runtime v5、compiled tile v8へ更新し、engine v28を維持する |
| 既知の限界と禁止 | Case Pの負イオン・非scalar有効質量、正電位履歴、複数イオン種、emission、collisional/magnetized charging、surface-release直後のsheath物理は未対応。COMSOL差を埋めるfloor/clip、正電位branchへのfallback、model自動切替、比較専用core分岐を追加しない |

このrevisionはM3-Vの全caseを適用可能にするものではない。Case Pはspecies契約に合わず、Case Aの一部には正電位履歴がある。
比較は引き続き外部V&Vが所有し、production coreでは入力が宣言範囲にある場合だけ解く。現行の連続path drift認証は
fieldと粒子速度の成分絶対上界から作るため安全側だが、強いco-flowでは偽拒否し得る。符号付きintervalへ改善する場合は
RZ基底を含むenclosure全体の独立revisionとし、このmodel内で安全性を緩めない。

## P18-C：aggregate relative-drift continuous charge decision

| 項目 | 判断 |
|---|---|
| 解く利用case | 外部または内部field producerが与える電子と集約正イオンprimitiveから、正負表面電位、有限相対drift、保存式で明示された正則化を含む連続平均電荷を運動と同じstageで解く。model revisionは`aggregate_relative_drift_regularized_two_current_v1` |
| 既存modelで解けない理由 | P15はstationary単一正イオン、P15-Dは非正電位の単一shifted-Maxwellian正イオンに限定される。閾値を緩めたり既存modelへ正電位branch・集約質量・floor/clampを足すと、異なる物理modelと適用域を混同する |
| 所有module | `physics/charge.py`がrate、局所微分、有限invariant/rate/Lipschitz boundとrevision定数を所有する。catalogはfield authorityと相対速度envelope、runtime/compiledは既存の一つのcontinuous-state passへのstage/path接続を所有する |
| 必要field/state | 既存`Z`、particle velocity、`electrostatic_radius_m`に加え、`n_e,n_i` [1/m^3]、電子・正イオンthermal voltage [V]、正イオンvelocity [m/s]、局所有効正イオンmass [kg]、背景screening length [m]を正値fieldとして使う。uniform値も定数fieldを書く。revisionは`lambda_eff=max(a,lambda_screening)`を適用する |
| 適用域と有限証明 | caseは正則化前の`abs(u_i-v)`について有限正値`maximum_relative_ion_speed_m_s`を必須宣言する。これは速度clipでなく、全primitive rangeに共通な`[Z_min,Z_max]`、`B_Z`、`L_Z`を作るrun-wide certificateであり、actual stageとcontinuous pathの超過は拒否する。物理的推奨modelではなく、集約二電流のversioned reference/sensitivity revisionである |
| 数値式 | `u_eps=1 m/s`、ion-energy floor `0.01 V`、指数範囲`[-50,50]`をrevision定数とし、overflow修復parameterにしない。RK4は`hL_Z<=0.5`と全stage/endpointのinvariantを維持し、native exponential pathはmidpoint-frozen affine exponential `J<=0`を使う。charge-only subcycle、平衡置換、state clipを追加しない |
| 対応座標・integrator | Cartesian XYとaxisymmetric RZ meridional、既存RK4とnative explicit midpoint。field sampling、geometry、event、boundary、output、checkpoint schemaは変更しない |
| reference | producer非依存のDecimal scalar式、正負/zero/floor/clamp branch、branch連続性、行別有効質量、Galilean/rotation不変性、primitive range上のfinite bound、compiled parity、両積分器の連成収束、XY/RZ・event・checkpoint identityを検証する。保存COMSOL primitiveとの全行parityはcore testでなく外部V&Vが所有する |
| 性能・memory影響 | 既存single-thread compiled row passへ一つのcharge branchを足し、追加fieldはstage slabでsampleする。resident particle state、scratch schedule、第二engine、result/checkpoint schemaを増やさない |
| 置換・削除する旧経路 | P15/P15-Dを変更・置換しない。COMSOL式parser、Case P/A分岐、`Ge0`/`phi1`の二重入力、massのparameter-or-field分岐、自動model切替を追加しない。既存Barnesはsingle-species parameter authorityなのでP18-Iまでは本revisionとの併用を明示拒否する |

P18-C完了時点はcatalog v11、physics runtime v10、compiled tile v12で、engine v30と各schemaは不変である。
独立Decimal oracle、global bound、compiled/pure parity、XY/RZ、RK4 4次・explicit midpoint 2次の連成収束、material
event、checkpoint/resumeを検証した。外部V&Vは保存式再生をPASS、coreの標準`epsilon0`からscreening式を再構成する
厳密provider一致を定数規約差によりFAILとして分離する。数値と解釈は
[`../evidence/p18c/README.md`](../evidence/p18c/README.md)が所有し、core定数や閾値を外部datasetへ合わせない。

## P18-I：aggregate ion-drag sensitivity revisions decision

| 項目 | 判断 |
|---|---|
| 解く利用caseと選択 | 集約正イオン場から、保存モデルに由来する二つの異なるion-drag閉じ方を明示選択して軌道へ加える。`ion_drag` categoryで、`screened_collection_orbital` / `relative_flow_screened_collection_orbital_aggregate_ion_v1`と、`image_orbital_sensitivity` / `electric_field_directed_image_orbital_sensitivity_v1`を排他的に選ぶ。既存`barnes_collisionless`は第三の独立revisionとして維持する |
| 所有module | `physics/forces.py`が二式、固定正則化、局所適用判定、有限global boundを所有する。catalogがversioned field authorityと組合せを確定し、runtime/compiledが既存の一つのstage passへadditive accelerationとして接続する。field producer、integrator、event、boundary、output、checkpoint、COMSOL比較は変更しない |
| 共有primitiveとauthority | 粒子の`Z`、mass、electrostatic radiusに加え、`n_i` [1/m^3]、正イオンthermal voltage`T_iV` [V]、正イオンvelocity`u_i` [m/s]、局所有効正イオンmass`m_i` [kg]、capacitance用screening length`lambda` [m]を使う。`phi_1=e/[4*pi*eps0*a*(1+a/max(a,lambda))]`をcoreで一度だけ構成する。relative-flow式はさらにion-neutral mean free path`lambda_in` [m]、image式は電子thermal voltage`T_eV` [V]とelectric field`E` [V/m]を必要とし、`lambda_img=sqrt(eps0*T_eV/(e*n_i))`をrevision内で一意に構成する。uniform値もcanonical fieldとして与え、`phi_1`、`lambda_img`、scalar ion speed、COMSOL derived variableを二重入力しない |
| relative-flow式 | `w=u_i-v`、`s2=|w|^2+8*e*T_iV/(pi*m_i)+u_eps^2`、`b_s=max(a,min(lambda,lambda_in))`、`b_90=sqrt(Z^2+1e-20)*e^2/(4*pi*eps0*m_i*s2)`、`b_col^2=min[b_s^2,a^2*max(0,1-2*e*Z*phi_1/(m_i*s2))]`、`lnL=max[0,0.5*ln((b_s^2+b_90^2)/(b_col^2+b_90^2))]`、`F=n_i*m_i*sqrt(s2)*(pi*b_col^2+4*pi*b_90^2*lnL)*w`とする。`u_eps=1 m/s`、charge-number正則化`1e-20`、scale=1はrevision定数で、設定値や数値修復ではない |
| image感度式 | producer非依存に`U=|u_i|`、`s2=U^2+8*e*T_iV/(pi*m_i)+u_eps^2`、`A_col=pi*a^2*max(0,1-Z*phi_1/T_iV)`、`b_img=e^2*Z/(2*pi*eps0*m_i*s2)`、`A_orb=pi*b_img^2*ln[max(1+1e-12,lambda_img/a)]`、`F=n_i*m_i*sqrt(s2)*U*(A_col+A_orb)*E/sqrt(|E|^2+1 (V/m)^2)`とする。従ってzero ion flowまたはzero electric fieldで力は0になる。保存Case Pの`U=sqrt(|u_i|^2+u_eps^2)`とCase Aのproducer scalar`AS_ui_mag`は同一式でないためcoreで分岐・追加scalar authorityを作らず、その差を外部formula replayで報告する |
| charge/electricとの整合 | fixed chargeとP18-C aggregate chargeだけを許可する。aggregate charge併用時は`n_i,T_iV,u_i,m_i,lambda`を、image式ではさらに`T_eV`のfield名を完全一致させ、relative-flow式の`maximum_relative_ion_speed_m_s`もcharge側と一致させる。P15/P15-Dの単一species/Kelvin/scalar-mass authorityとは混成しない。image式とCoulomb electric forceを併用する場合は同じelectric-field authorityを必須とする |
| 適用域と有限bound | 全primitiveは有限正値、粒子mass/radiusは正値とする。relative-flow式はcaseが有限正値`maximum_relative_ion_speed_m_s`を宣言し、actual stageと連続pathで`|u_i-v|`超過をfail-closedにする。これは速度clipでなく、charge invariantとfield extremaから有限断面積・加速度boundを作るcertificateである。image式は粒子速度へ依存せず、有限field extremaとcharge invariantからglobal boundを作る。式内`min/max/floor`以外のBarnes適用域、blend、fallback、経験安全率を追加しない。両式はmodel-form sensitivityであり普遍的なsheath ion-drag精度を主張しない |
| 対応座標・integrator | Cartesian XYとaxisymmetric RZ meridionalの既存vector basis、RK4とnative explicit midpointへ対応する。動的`Z`は各stageの同じ値を使い、charge-only subcycle、force lag、第二integrator、operator splitを作らない |
| 独立referenceと受入 | production式を呼ばないDecimal/scalar reference、全floor/min/max branch、zero limit、方向、charge符号、回転・Galilean性（relative-flow式のみ）、global-bound乱択包含、compiled parity、XY/RZ、relative-flowのRK4 4次・midpoint 2次、imageの両積分器での定加速度関係、relative-flow材料eventを検査する。checkpointは新state/schemaを追加しないため既存の全payload resume identity gateを再利用する。保存primitiveの式再生、Case固有画像式との差、統合軌道は`tools/vv`が別判定し、core testのgolden truthにしない |
| 性能・memory影響 | compiled row passへ二つの選択branchだけを追加し、既存additive-acceleration列を再利用する。relative-flowのpath certificateはion velocity二成分上界だけを保持し、image式は追加resident stateを持たない。無効modelのbefore/afterと100,000-row有効stageを測り、第二engine、scheduler、診断subsystem、result schemaを増やさない |
| 置換・削除しない経路 | P15-F Barnes、P18-C charge、既存electric forceを変更・置換しない。Case P/A名、directory variant、自動model選択、二式のblend、比較差を減らすfit parameter、COMSOL式parser、`phi_1`/ion-speed二重入力をproductionへ追加しない |

この表はM3-C0aの保存式監査と、Case P/A画像式の不一致を実装前に固定したdecisionである。relative-flow式は
保存P/Aで同一だが、image式はproducer固有のscalar speed規則を含んでいたため、製品revisionでは上記の一意な
vector-norm規則へ正規化する。したがって後者のnative COMSOL式との一致は、低速差を含む外部V&V結果として扱い、
同じ式であるとは表示しない。

P18-I完了時点はcatalog v12、physics runtime v11、compiled tile v13で、engine v30、proposal、event、runtime layout、
memory plan、case/result/checkpoint schemaは不変である。独立Decimal oracle、全branch・zero limit、global bound乱択包含、
compiled parity、P18-Cとの同一stage結合、公開XY/RZ case、relative-flowのRK4 4次・explicit midpoint 2次、imageの
両積分器での定加速度関係、relative-flow材料eventを検証した。
100,000-rowのP18-C併用warm stageは、charge-only / relative-flow / imageでそれぞれmedian
`0.027057 / 0.028538 / 0.028826 s`、prepared boundは`1,600,088 / 1,600,104 / 1,600,088 B`だった。
これは同一machineの非gating記述値で、source-code before/afterの因果推定ではない。

保存済み12 package・397,820 active rowの外部式再生は最大正規化残差`1.7901e-15`でPASSした。production
relative-flow式のstrict比較は保存COMSOLと標準SIの`epsilon0`規約差と整合する差を示してFAIL、image式はCase P/Aの速度authority差を
`DOCUMENTED_MODEL_DEFINITION_DIFFERENCE`として保持した。詳細は[`../evidence/p18i/README.md`](../evidence/p18i/README.md)
が所有し、この結果から統合軌道精度や物理適用性は主張しない。

## P18-D：quasistatic spherical DEP decision

| 項目 | 判断 |
|---|---|
| 解く利用caseと選択 | 外部field producerが球形粒子位置で評価可能な`grad(mean_E_squared)`を供給できる時、準静的dipole近似のDEPを明示選択して軌道へ加える。category/model/revisionは`dielectrophoresis` / `quasistatic_spherical` / `quasistatic_spherical_gradient_e2_v1`とする |
| 既存modelで解けない理由 | Coulomb力は自由電荷`Z e E`であり、誘起dipoleが非一様場から受けるDEPとは異なる。既存electric式へ半径三乗や誘電率を混ぜると、自由電荷と分極のauthorityおよび無効化条件を混同する |
| 所有module | `physics/forces.py`が純粋なDEP加速度とcomponent-wise global boundを所有する。catalogが設定とfield metadata、runtime/compiledが既存の一つのstage passへのadditive accelerationを所有する。field gradient/recovery、COMSOL変換、比較、可視化はcore外とする |
| 方程式と粒子authority | `epsilon_m=epsilon_0*medium_relative_permittivity`、`F=2*pi*epsilon_m*a^3*K_CM*grad(mean_E_squared)`、`a=electrostatic_radius_m`、`acceleration=F/mass_kg`とする。`mass_kg`と`electrostatic_radius_m`をresident authorityとし、drag diameterやdisplaced volumeから再構成しない |
| 誘電authority | caseは有限正値`medium_relative_permittivity`と`[-0.5,1]`内の有限`real_clausius_mossotti_factor`を直接一度だけ指定する。coreは粒子誘電率、導電率、周波数から`K_CM`を再計算せず、粒子誘電率と`K_CM`の二重入力を許さない。負・零・正の係数を式どおり扱い、符号をclampしない。範囲外の有効polarizabilityは別revisionとする |
| fieldとprovenance | `gradient_mean_e_squared_field`は単位`V^2/m^3`のcanonical XYまたはRZ vectorである。`mean_E_squared`はDCでは`|E|^2`、周期場では一つのfrequency/solutionについてproducerが解決した物理時間平均（RMS二乗）を意味する。producerはDC/RF、solution/frequency、peak/RMS変換、時間平均、gradient recovery、元field hashをcanonical provenanceの`producer_metadata`へ保存する。coreはunit/basisを検査し、provenance completenessは外部import/V&V validatorが所有する。coreは節点Eを微分せず、producer固有scalarや回復設定をsolver caseへ複製しない |
| 適用域と失敗 | 球形、線形・等方・一様媒質、準静的dipole、one-way dilute粒子を仮定する。producerが同じfield/recoveryと所定誤差基準から認証した有限正値`maximum_point_dipole_radius_m`をcaseで一度だけ宣言し、全resident `electrostatic_radius_m`がこれ以下でなければprepare時に拒否する。認証方法と誤差基準はprovenanceへ残す。数値的には正のmass/radius、有限parameter、有限vector fieldをfail-closedで要求し、dipole条件をepsilon clampや任意安全率で隠さない |
| 対応座標・integrator | Cartesian XYとaxisymmetric RZ meridionalの既存vector basis、RK4とnative explicit midpointへ対応する。速度・電荷へ依存しないadditive accelerationとして各stage位置でfieldをsampleし、operator split、第二integrator、DEP専用meshを作らない |
| finite boundと最適化 | field component extrema、`abs(K_CM)`、medium permittivity、`a^3/m`から有限component boundを作り、既存support/event enclosureへ加算する。Cartesian XYでgradient fieldをexact constantと証明でき、他の力も既存constant条件を満たす場合だけ既存constant-acceleration pathを再利用する |
| 独立referenceと受入 | production式を呼ばないscalar reference、一様E相当のzero gradient、係数符号、`a^3/m` scaling、analytic quadratic-potential field、global-bound乱択包含、pure/compiled parity、XY/RZ basis、RK4 4次・midpoint 2次を検査する。保存COMSOL primitiveの式再生と統合軌道は`tools/vv`が別判定し、core testのgolden truthにしない |
| 性能・memory影響 | compiled row passへ一つのoptional branchとstage vector入力だけを追加し、既存additive-acceleration列とprepared external boundを再利用する。resident state、scratch schedule、engine、result/checkpoint schema、scheduler、診断subsystemを増やさない。100,000-row warm stageとprepared memoryを記述的に測る |
| 非目標と置換しない経路 | Coulomb、ion drag、thermophoresisを変更しない。複素・周波数依存Clausius--Mossotti、travelling-wave DEP、非球形、多極子、粒子相互作用、fieldの数値微分、自動model選択、Case P/A分岐は別revisionまたはcore外とする |

2026-10-01にこのdecisionどおりP18-Dを完了した。catalog v13、physics runtime v12、compiled tile v14で既存の単一stage
passへ接続し、engine v30、proposal/event、resident state、runtime layout、memory plan、case/result/checkpoint schemaは
変更していない。独立scalar式、zero/sign、`a^3/m` scaling、2-D回転共変性、乱択global-bound包含、XY/RZのcatalog/runtime/compiled
parityを確認し、公開APIの解析的調和振動子でRK4 3.5次以上、explicit midpoint 1.8次以上を受け入れた。B02 Brownianは
追加決定論力を許さない既存subsetのためDEP併用を拒否する。COMSOL保存primitiveと統合軌道の比較は外部V&Vに残し、
production受入から同精度を主張しない。100,000-rowのwarm stageはdisabled/enabled median `22.94/23.83 ms`
（1.039x）で、prepared boundは双方`1,600,000 B`だった。これは
[`../evidence/p18d/`](../evidence/p18d/)に保存するmachine-localな非gating観測である。

## P18-L：rarefied-vorticity lift sensitivity decision

| 項目 | 判断 |
|---|---|
| 解く利用caseと選択 | 外部field producerがaxisymmetric no-swirlの中性気体速度、密度、平均自由行程、方位vorticityを同じsolutionから供給できる時、保存referenceで使われた自由分子lift感度式を明示optionとして軌道へ加える。category/model/revisionは`lift` / `rarefied_vorticity_sensitivity` / `rarefied_vorticity_sensitivity_rz_v1`とする |
| 既存modelで解けない理由 | dragは相対流方向の散逸、thermophoresisは並進熱流束方向であり、相対流へ直交するvorticity結合を表さない。liftをdrag rateまたは固定外力へ混ぜると、物理方向と速度依存の経路boundを失う |
| 所有module | `physics/forces.py`が純粋式、Kn適用域、component-wise速度依存boundを所有する。catalogがRZ限定設定とcanonical field、runtime/compiledが既存の一stage passを所有する。`integrators.py`はmodelを知らず、runtimeが与える非drag加速度bound callbackを指数中点包絡で再評価する。vorticity生成、回復法、COMSOL変換・比較はcore外とする |
| 方程式とauthority | `w=u_g-v`、`a=drag_diameter_m/2`、`beta=C_L*pi*rho_g*lambda_g*a^2*omega_phi/mass_kg`、`a_L=(beta*w_z,-beta*w_r)`とする。これは`(omega_phi e_phi) cross w`のRZ射影である。慣性は`mass_kg`、gas-surface寸法は`drag_diameter_m`をauthorityとし、electrostatic radiusやdisplaced volumeから再構成しない |
| fieldとparameter | 必須fieldはRZ vector `gas_velocity_field [m/s]`、正scalar `gas_density_field [kg/m^3]`、正scalar `gas_mean_free_path_field [m]`、signed scalar `azimuthal_gas_vorticity_field [1/s]`。producerはvorticityを同じ速度solution・円筒座標符号・回復規則から形成する。coreは速度を微分しない。`lift_coefficient`は有限正値でcaseへ明示し、保存値1をdefaultや普遍相関にしない |
| 適用域と組合せ | 球形、dilute one-way、axisymmetric RZ meridional、no-swirl、`lambda_g/a>=10`だけを受理し、`applicability: error`でfail-closedにする。Cartesian XY、3-D、一般Saffman、continuum/transition blendを拒否する。drag併用時はgas velocity/density/mean-free-path、thermophoresis併用時はvelocity/mean-free-path、gravity併用時はdensityのfield authorityを一致させる。適用域が交わらないStokes--Cunninghamとの併用とB02 Brownianを拒否する |
| 対応integratorと有限bound | RK4は`k_max=C_L*pi*rho_max*lambda_max*a^2*abs(omega)_max/m`から`B_r=k_max(U_z+V_z)`、`B_z=k_max(U_r+V_r)`を各velocity boxで評価する。指数中点包絡v3は非drag加速度bound callbackを開始速度と半ステップ速度包絡で評価し、両者のcomponent最大をendpoint・position boundへ使う。liftをscalar drag rateへ偽装せず、固定`external_acceleration_abs_upper`へも入れない。積分器本体v2は変更しない |
| 独立referenceと受入 | 任意方位角の3-D cross-product射影oracle、zero vorticity/comoving、符号、`rho*lambda*a^2/m` scaling、相対流への直交性、乱択global-bound包含、Kn fail-closed、catalog/runtime/compiled parityを検査する。`u_r=Omega*z,u_z=0,omega_phi=Omega`の解析解を使い、RZ公開caseでRK4 3.5次以上・explicit midpoint 1.8次以上、指数中点の全短縮state包絡を確認する |
| 性能・memory影響 | 既存compiled row passへ一つのoptional branchを加え、additive-acceleration workspaceを再利用する。prepared dataは粒子別coupling-rate上界とstatic applicability、および二成分gas速度上界だけで、新resident state、checkpoint/result schema、schedulerを追加しない。100,000-row warm stage、disabled overhead、prepared bound memoryを記述的に測る |
| 置換・削除しない経路 | drag、thermophoresis、DEP、ion drag、electric、gravityを変更・置換しない。Case P/A名、`nojac`、`spf.vorticityphi`、core内gradient recovery、自動`C_L=1`、比較差を減らすfit、第二engine、lift専用integratorを追加しない。exported-P1/native-field比較はFAILであり、後続common-P1診断が限定same-field agreementをPASSしても、native-field pointwise parityとlift modelの物理妥当性は`NOT_TESTED`とする |

指数中点包絡v3の安全性は二段法の実際の評価順に合わせる。最大stepを`h`、drag targetのcomponent上界を`U`、
開始速度を`v0`、runtimeの非drag加速度boundを`B(V)`とする。`A0=B(abs(v0))`、
`Vhalf=max(abs(v0),U)+h*A0/2`、`Ahalf=B(Vhalf)`、`A=max(A0,Ahalf)`を外向き丸めで作り、
任意の短縮stepに対して`abs(v)<=max(abs(v0),U)+h*A`、
`abs(x-x0)<=h*max(abs(v0),U)+h^2*A/2`を使う。callbackはfield位置に依らないglobal primitive extremaから作り、
field sampleやmodel dispatchを積分器へ持ち込まない。

2026-10-01にこのdecisionどおりP18-Lを完了した。catalog v14、physics runtime v13、compiled tile v15から既存の
単一stage passへ接続し、速度依存する全非drag加速度boundを一つのcallbackへ統合した。exponential enclosure v3だけを
更新し、exponential midpoint v2、engine v30、proposal v7、event/runtime layout/memory plan、case/result/checkpoint
schemaは維持した。producer提供のsigned方位vorticity、有限正値`lift_coefficient`、`lambda/a>=10`をfail-closedに
検査し、B02 Brownianとの併用を拒否する。resolved result manifestのlift entryにはmodel/revision、field bindingと
明示`lift_coefficient`を保存する。

3-D cross-product射影oracle、zero/comoving、符号、`rho*lambda*a^2/m` scaling、相対流への直交性、乱択bound、
Kn拒否、catalog/runtime/compiled parity、RZ公開caseのRK4 3.5次以上・explicit midpoint 1.8次以上、短縮state包絡を
受け入れた。100,000-row warm direct stageのmedianはdisabled `0.023354 s`、enabled `0.0255142 s`、比`1.0925`、
prepared bound増分`900016 B`である。machine-localな非gating観測であり、後続common-P1診断の限定same-field PASSを
COMSOL native-field pointwise parityまたはlift modelの物理妥当性へ拡張しない。

## P18-R：effective-gas drag / thermophoresis sensitivity decision

| 項目 | 判断 |
|---|---|
| 解く利用caseと選択 | producerが混合気体を一つの有効Maxwellian/pseudogasへ畳み込み、その近似を明示認証できるreference/sensitivity runを解く。dragは`epstein_linear` / `epstein_linear_effective_gas_sensitivity_v1`、thermophoresisは`waldmann_gallis` / `waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1`を選ぶ |
| 既存modelで解けない理由 | P15-EはMaxwell mixed equal-temperature molecular model、P16はsingle-species modelであり、保存CF4/O2入力のmixture/model authorityを満たさない。既存revisionのgateや意味を緩めず、producer-owned effective closureを別revisionとして識別する |
| 所有module | `physics/forces.py`の既存linear Epstein / Waldmann--Gallis式を唯一のformula ownerとし、catalogがrevisionと必須速度比、runtime/compiledが同じstage passと既存continuous-path enclosureを所有する。effective reduction、`q_eff`形成、COMSOL比較はproducer/外部V&Vが所有する |
| field・state・authority | 既存のgas velocity、density、temperature、mean-free-path、molecular mass、drag `delta`を再利用する。thermophoresisの`q_eff`は有効pseudogasの並進伝導熱流束であり、coreはtemperature gradient、species配列、species別accommodation、mixture ruleを回復しない。新しいresident stateは持たない |
| 適用域と失敗 | 球形、dilute one-way、producer-certified one-effective-Maxwellian/pseudogas、`lambda/a>=10`、必須の有限`0 < maximum_speed_ratio <= 1`をactual stageと全受理pathでfail-closedに検査する。値をclipせず、既存`epstein_linear_v1`とP16 single-species revisionの上限`0.1`は変更しない |
| 座標・integrator | Cartesian XY / axisymmetric RZ meridional、RK4 / explicit midpointの既存経路へ対応する。engine v30、proposal v7、integrator v2、RK4 enclosure v2、exponential enclosure v3を変更せず、第二engineやoperator splitを作らない |
| reference・受入 | 旧revisionが`0.1`を維持するcatalog回帰、必須上限の拒否、pure/compiled parity、XY/RZ、両積分器、連続path超過拒否を検査する。保存auditはnative linear Epstein replayが約`1.1e-15`でPASS、既存P15-E/P16 applicabilityが12/12 `NOT_APPLICABLE`、PPR `q_eff`欠損でthermophoresis replayが`NOT_TESTED`であり、保存frameをpath certificateにしない |
| 性能・memory | 既存formula、sample配列、workspace、boundを共有するoptional catalog branchだけを追加する。100,000-row warm stageの旧/new revision pairは`0.0267117/0.0267643 s`（`1.00197x`）で、payload、applicability、prepared bound `2500040 B`、共有workspace `6600000 B`は同一だった。machine-localな非gating観測である。compiled tile v16 / catalog v15 / runtime v14へ更新し、engine、proposal、integrator、resident state、memory plan、case/result/checkpoint schemaは不変とする |
| 置換・非目標 | P15-E、`epstein_linear_v1`、P16を置換せず、自動blend、COMSOL/Case分岐、air相関流用、species-resolved mixture truthを追加しない。このturnではCOMSOL studyを再実行しておらず、新revisionは物理的真値または軌道一致を確立しない |

2026-10-01にこのdecisionどおりP18-Rを完了した。P18-R closeout時点では、M3-C1は新しいBrownian-off COMSOL studyなしでは
candidate preflightまででblockedであり、当時の次の実行可能なcore workはB03だった。

この二文はP18-R closeout時点の履歴であり、変更しない。後続M3-C0b v5の旧「全gate PASS」は、位置relative L2を
絶対RZ座標で正規化して原点依存だったため`INVALIDATED`となり、v5は履歴上の`CHARACTERIZED`に留まる。
未見の0.15625 usを使った逐次確認v6は、Case-A 100 nmの0--450 us、287粒子×46 frameについて
0.625/0.3125/0.15625 us系列を評価し、各runの13,202 recordがすべてactiveだった。原点不変な変位・速度・電荷の
fine-pair relative L2は`3.102727085428027e-5/3.92483251084038e-5/1.3511393490811483e-6`、観測次数は
`0.9041136/0.944312/1.123838`である。このPASSはpre-eventの運用上の固定step選択だけで、solver agreementや
普遍的なaccuracyを認定しない。後続M3-C1の最小PPR補足は、既存v6と座標が完全一致する13,202 active saved rowで
Waldmann producer-formをcomponent-scale normalized residual最大`4.4046499933294035e-16`、global relative L2
`1.0992667494449471e-16`で再生し、frozen saved-state producer-formを8/8閉じた。これはP18-Rの物理・適用域判断、
continuous-path認証、統合軌道を変更しない。P19-L後のexported-P1/native-field比較は完了し、6 gateすべてのFAILで
条件付きcommon-field診断をtriggerした。後続full-physics exact-connectivity common-field COMSOL診断は事前登録9 gateを
すべてPASSした。この時点では残る決定論展開、B03、M3-C2の順としていたが、後続のboundary意味確認と計算量評価により
下記「M3-C0 boundary semantics / execution-order supersession」で置換した。現行順序は`implementation_plan.md`と
`vv_methodology.md`が所有する。

## P19-L：localized continuous-path applicability certificate decision

| 項目 | 判断 |
|---|---|
| 解く利用case | 強い局在場を持つexact-connectivity P1/RZ caseで、実際の初期sampleと局所pathはmodel適用域内でも、遠方cellを含むglobal primitive extremaだけで全rowが時刻0拒否される偽陰性をなくす。M3-C1は最初の実測例であり、このpackage自体はproducer非依存とする |
| 既存methodで解けない理由 | global enclosureは安全だが局所性を持たず、実model違反と「global boundでは証明不能」を同じ`model_applicability`へ畳み込んでいた。上限緩和、非restrictive gate、比較case専用branchは物理意味と安全性を壊す |
| 所有module | `integrators.py`が元の固定step RK4 proposalに対応するdense pathとparameter部分区間のstate enclosure、`fields.py`がpath tubeと交差するcellのsupport・primitive range、`physics/runtime.py`がrange上の力・電荷・applicability bound、`engine.py`がboundedなcertificate workと最終commit/failure分類を所有する |
| 数値意味 | P19-L完了時点の`rk4_position_hermite_state_extension_v2`の位置制御点はRK4 proposalの始終位置・速度に一致し、速度・電荷は同じstage-rateから作る。certificate-onlyなdyadic restrictionはこの不変なroot pathの証明領域だけを狭め、endpointを再積分・逐次commitしない。fixed `dt`、RK4 stage式、frame schedule、accepted endpointは変更しない |
| eventとの分離 | P19-L完了時点では`rk4_global_abs_enclosure_v2`をsupportとrev3b材料/RZ event geometryのauthorityとして維持した。後続event v14は、global supportを独立に証明済みのvalid `rk4_dense` rowに限りdense Bernstein boundへevent broad phaseを移した。global enclosureはsupport、再積分、受入安全性とevent query fallbackを所有する。locator-before-validity、hit prefixの同一RK4再積分、fresh residual proposalによるsequential event意味論は維持する |
| failure | actual stage/sampleまたは局所rangeの実違反は既存の`field_support`または`model_applicability`、有限budget内で証明だけが閉じない場合はfailure code 9の`indeterminate_applicability_certificate`とする。64-cell候補上限のoverflowも区間分割の入力であり、物理違反や成功へ読み替えない |
| reference・受入 | 遠方cellに巨大な極値があっても局所pathが有効なmicrocase、pathが実際に適用域を横切る反例、certificate budget exhaustion、global成功caseのendpoint/final/event/output-schedule/tile identityを検証済み。strictなCase-A 100 nm candidateが時刻0を越えることはblocker解除でありtrajectory agreementではない |
| 性能・memory | dense pathは176 B/row、共有するevent `maximum_refinements`が2の現観測ではinterval stackを72 B/row、最大64 cellの候補arenaを最大544 B/rowで計画する。4,096粒子×20 stepのglobal/local公開API観測は`0.9276381/0.9345506 s`、比`1.0074517`、payload bitwise一致。代表Case-A 287粒子1 stepはgeneric chord-deviation算術のbatch化だけで`1.647475→1.304153 s`（`1.263x`、`-20.84%`）、final＋frame hashも同一だった。いずれもmachine-localな非gating観測である |
| revision | P19-L完了revisionは`particle_engine_v31`、`coupled_fixed_step_proposal_v8`、event v12、`solver_owned_memory_plan_v13`、`field_location_v4`、dense path v2。physics runtime v16はP19-L実装・性能snapshot、v17はDEP上限の1 ULP外向き境界を整えた。現行v18はcharge Jacobianをstage payloadへ追加する |
| 非目標・削除 | 一般interval framework、adaptive accuracy controller、COMSOL専用path、gate bypass、閾値fit、比較用第二solverを作らない。局所applicability certificateはglobal-firstのfallbackであり、global safety enclosureを置換しない。event v14のdense boundも条件付きbroad phaseだけを所有し、第二locatorや公開設定を作らない。callerのないdense subset helperは残さない |

P19-Lは完了した。続くM3-C1ではcandidateのprepare設定、canonical input、run manifest、result、source exportのhashを
一つのprovenance chainへ固定した。candidateはexported P1、参照側はCOMSOL native finite-element fieldであり、candidateを
`native-field`と呼ばない。この比較結果と次の判断は以下のdecisionへ分離する。

## M3-C1：Case-A 100 nm pre-event cross-representation decision

| 項目 | 判断 |
|---|---|
| 比較範囲 | Brownian-off、0--450 us、287粒子×46 frame。動的電荷、electric、relative-flow ion drag、effective-gas Epstein、Waldmann thermophoresis、DEP、RZ lift、gravity/buoyancyを含む。candidateはexported exact-connectivity P1、referenceはCOMSOL native finite-element field |
| 実行成立 | 0.625/0.3125/0.15625 usのcandidate 3 runは指定macro `dt`を使用し、各13,202 trajectory row、event 0、failure 0で完了。COMSOL referenceのpre-event運用上の自己収束gateは位置・速度・電荷すべてPASS |
| candidate刻み解釈 | 当時のevent v13 certificateがdyadic leafを同じ時間幅まで再積分し、全runのaccepted-piece countと最深nominal leaf幅が一致した。fine-pair差は設定済みfloat64 representation-scale floor内のmacro-step-halving production-output stabilityとして有効な履歴である。ただしartificial subdivisionのため独立時間収束、RK4次数、floor未満の誤差は主張しない。v14再実行が現行時間収束authorityである |
| cross-representation結果 | fine差は位置・速度・電荷の事前登録RMS/max 6 gateをすべてFAILした。exact値はV&V methodologyとcompact evidenceが所有する |
| 認定しないこと | 異なるfield表現をまたぐFAILであり、same-field solver agreement、全物理のCOMSOL同精度、physical applicability、boundary、30 ms、他粒径・case・variantを認定しない。frozen saved-state 8/8 parityもruntime sampling/wiringやtrajectory agreementへ昇格しない |
| 次の診断 | cross-representation FAILで既定triggerが成立したため、full-physics exact-connectivity common-fieldをCOMSOL側にも使う独立診断を実行する。旧M3-V common-field runnerは固定電荷・共通3力のreduced sliceで、動的電荷と全決定論力を覆わない。core式・安全gate・閾値を比較結果へfitしない |

判定値とprovenanceのauthorityは
[`../evidence/m3c1/case_a_100nm_pre_event_v6/`](../evidence/m3c1/case_a_100nm_pre_event_v6/)である。

## M3-C1：Case-A 100 nm full-physics common-P1 decision

| 項目 | 判断 |
|---|---|
| 比較目的 | cross-representation FAILをintegrator/runtime差とfield representation差へ分離する。candidateとCOMSOLへ同じcanonical exact-connectivity P1 node値・connectivityを与え、core式や許容値を変更しない |
| 物理構成 | Brownian-off、動的電荷、electric、relative-flow ion drag、effective-gas Epstein、Waldmann thermophoresis、DEP、RZ lift、gravity/buoyancy。Case-A 100 nm、0--450 us、287粒子×46 frame、boundary event 0 |
| 実行保全 | COMSOL原本を`loadCopy`した隔離copyへ共通tableを与え、`-nosave -np 1`で3刻みを実行した。26入力artifact、source MPH前後hash、6,314個のt=0 primitiveを検証し、原本を変更しない |
| 事前登録 | cross差を読む前にcandidate/reference自己収束、t=0初期状態の4096-ULP基準、位置・速度・電荷のRMS/max/relative L2上限を固定した。t=0は287粒子×5成分の1,435値を全PASS |
| 結果 | 初回結果は履歴として維持する。現行eval_v3のRMS/max/relative L2は、位置`4.06807283316903e-13 m` / `1.2035778717837921e-12 m` / `2.0316001855367802e-10`、速度`2.10847522277453e-9 m/s` / `3.844306466969233e-9 m/s` / `1.9630813983926987e-10`、電荷`1.1818881019499895e-7 e` / `2.3758877887303242e-7 e` / `4.670220207738041e-10`。登録budget `ab3713fb...b7b5`、result `9fa50b04...90bf`で9/9 PASS |
| 認定範囲 | このcommon-field、Brownian-off、event前sliceのsame-field solver agreementだけを認定する。native-field等価性、field producer精度、物理modelの妥当性、境界、Brownian、30 ms、他case・粒径・variant、普遍的COMSOL同精度は認定しない |
| 当時の次 | 残るM3-C0b/C1の30 ms・case・variant・boundaryを閉じ、その後B03、M3-C2へ進むとしていた。この順序は後続のboundary意味確認と計算量評価により、直後のsuperseding decisionで置換した。比較結果へのcore fitting、許容値緩和、COMSOL分岐を追加しない原則は維持する |

compact authorityは
[`../evidence/m3c1/case_a_100nm_common_p1_v1/`](../evidence/m3c1/case_a_100nm_common_p1_v1/)である。

## M3-C0：boundary semantics / execution-order supersession

このdecisionは上記M3-C1 decisionの「次」だけを置換し、過去の測定値・認定範囲を変更しない。

| 項目 | 判断 |
|---|---|
| 解く利用case | COMSOLを含む外部producerが持つ「停止して位置を保持する非deposition終端」と「領域から消えてkinematicsが無効になる終端」を、productionのstick/escapeと混同せずcanonical boundary lawへ写す |
| 外部正例 | 力なし・動的帯電なしの解析的normal impactを、boundary 37 Freezeとboundary 35 Disappearについてfixed RK4 10/5/2.5 us、0--150 us、2.5 us保存で隔離実行した。現行v2は2 scenario×3刻みのexact 6 configuration receiptをprocess logから照合し、欠落・重複・形式不正・設定差をfail-closedで拒否する。原本は`loadCopy`/`-nosave`で扱い、前後SHA-256は`3bbf08e3469758313eac5de473a7a0dd4cc9a6f72c9722393229f0b856e9b524`のまま |
| 観測結果 | 全active frameは`x=x0+v0*t` / `v=v0`に一致し、全run最大の位置/速度誤差は`4.726604209672303e-16 m` / `1.7763568394002505e-15 m/s`（各上限`1e-12`）。両scenarioのeventは73 us、最初のterminal保存時刻は75 us。Freezeはstatus 2でhit点R-Zを保持し、保存velocityは衝突前値を保持する。Disappearはstatus 4で位置・速度がNaNとなる。Freeze/Disappearのstep間event-time spreadは`3.07371315899641e-17/2.71050543121376e-20 s`。56 gate PASS、FAIL 0、velocity分類6件は`CHARACTERIZED_NOT_GATED` |
| 認定範囲 | 隔離したCOMSOL Freeze/Disappear意味だけを認定する。production solver parity、grazing/corner、多重hit、full physics、native-field/P1等価性、全12 packageを認定しない。Disappearのhit点は解析的再構成であり直接event座標ではない |
| core判断 | Freezeはvelocity 0のdepositionでもkinematics nullのescapeでもないため、producer非依存の最小`hold` law / `held` lifecycleをP18-Hで実装する。positionとhit時payloadを保持するterminal状態とし、event後もcharge等を時間発展させるpaused-particle一般modelは作らない。COMSOL名/status codeをcoreへ入れない |
| 当時の実装順 | common-P1 material-stick、field表現差局在化、P18-H、B03、段階的決定論比較、M3-C2としていた。material-stickと局在化の完了後、その時点の順序はP18-H、B03、段階的決定論matrix、M3-C2だった。現行順序は後続P18-H decisionが置換する |
| 当時の順序変更理由 | 全12×3刻み×30 ms要件は維持した。一方、event v13の長時間1 run約`3e8` accepted piece見積りはartificial subdivisionを含む履歴であり、後続v14の計画根拠には使わない |
| 所有と削除 | COMSOL runner・normalizer・evidenceは`tools/vv/comsol`、production law/lifecycleは既存`boundaries.py`/`output.py`が所有する。比較用core branch、第二engine、境界status変換framework、旧順序用runnerは追加しない |

外部probeは本体を変更していない。v1は科学的に無効ではなく、configuration receiptとactive-flight oracleの監査強度が
不足した履歴成果物としてv2にsupersedeされた。current compact authorityは
[`../evidence/m3c0/boundary_semantics_v2/`](../evidence/m3c0/boundary_semantics_v2/)とする。

## M3-C1：material-event / event-query v14 supersession

このdecisionは上記M3-C0 decisionの未完了順序と、P19-L完了時点のevent-query境界だけを置換する。過去の測定・判定は履歴として保持する。

| 項目 | 判断 |
|---|---|
| 原因と修正境界 | event v13はglobal absolute safety enclosureをevent BVH boundにも流用し、material候補探索をartificialに細分した。v14は、global supportを独立に証明済みのvalid `rk4_dense` rowだけcurrent dense Bernstein位置・速度boundをevent broad-phase query authorityにする。global enclosureはshortened-stage、field support、applicability、acceptance safetyのauthorityとして維持する。invalid dense boundまたはglobal support未証明時はglobal queryへfallbackし、必要ならsplitする |
| 履歴カウンタ | v13のquery/refinement/accepted/depthは`16,427,517 / 7,792,306 / 8,635,211 / 16`。450 usまでに`7,623,460 / 7,792,306` refinement（`97.8331703092769%`）が発生済み。v14は`842,927 / 11 / 842,916 / 11`、failure 0 |
| material判定 | Case-A 100 nm、common canonical exact-connectivity P1、Brownian off、287粒子、first wafer stickまで。evaluation v5はmaterial 20/20とprefix 9/9をPASS。event時刻、hit位置、terminal chargeの絶対差は`2.157542807607049e-13 s / 2.683964162031316e-14 m / 4.7283812421028415e-09 e`。Freezeとstickのvelocity保存意味が違うためterminal velocityはcross-solver gateにしない |
| 性能観測 | operator-observed shell wall-timeは同一local環境で約`14m13s`（約853 s）から`36.5s`（約`23.4x`）。machine-local、概算、非gatingでありsolver-reported runtimeまたはmanifest値ではない |
| v14時間収束 | solver-only 0.625/0.3125/0.15625 usのquery/refinement/acceptedは`206,927/0/206,927`、`413,567/0/413,567`、`826,847/0/826,847`。位置・速度・電荷のRMS観測次数は`2.029875353701904 / 2.0816971911764033 / 2.044084026475049`、fine relative L2は`6.099791486063973e-8 / 8.321356016032579e-8 / 1.3796067752988052e-8`、全量`ORDER_EVALUATED`でPASS。約2次はpiecewise P1場・mesh crossingを含む本caseの経験値で、RK4の形式4次を証明も否定もしない |
| COMSOL再実行 | v14はsolver event BVH broad phaseだけを変更した。common-P1 input/reference/source MPHはhash-lock済みで不変なので、COMSOL studyを再実行せず既存referenceへのcandidate比較だけを再計算した |
| 認定範囲 | この限定same-field material anchorだけを認定する。native-field parity、field producer精度、物理妥当性、Brownian、30 ms、他case/size/variant、普遍的COMSOL精度は認定しない。旧v13のroundoff-level結果はprecision-stability履歴として有効だが、独立時間収束authorityではない |
| 当時の次 | material event、field表現差局在化、P18-H、B03完了後に段階的Brownian-off決定論matrix、その後M3-C2としていた。この順序は後続の100 nm・30 ms candidate-first decisionが置換する。旧v13の約`3e8` accepted piece/run見積りを現行計画根拠にしない |

compact authorityは
[`../evidence/m3c1/case_a_100nm_material_event_v1/`](../evidence/m3c1/case_a_100nm_material_event_v1/)である。

## P18-H：generic terminal hold boundary decision

| 項目 | 判断 |
|---|---|
| 解く利用case | 半導体装置内の捕捉、隔離、計算打切りなど、材料付着ではないがhit点とhit時payloadを有効なまま保持する終端境界を扱う |
| 既存lawで解けない理由 | `stick/stuck`はdepositionかつpost速度0、`escape/escaped`はkinematics logical nullである。どちらへも畳み込むと付着統計または有効状態の意味を壊す |
| 所有module | `boundaries.py`がparameterなしの`hold`応答と`held` outcome、`engine.py`がterminal lifecycle、`output.py`が永続code・series・checkpoint/resultを所有する。geometry/event locator、COMSOL adapter、比較toolは変更しない |
| 状態意味 | first hit位置、hit時速度、hit時電荷をpre/post payloadとresident stateへ保持し、inactiveな`held`へ遷移する。`kinematics_valid=1`だが、hit後の位置、速度、電荷、力は更新しない。paused/restart/resuspension modelではない |
| 対応座標・integrator | 既存point-particle first-hit経路を使う全対応XY/RZ、ballistic、quadratic、RK4、exponential midpoint、およびB02 terminal subset。新しいintegrator、event locator、残時間workを持たない |
| 解析解またはreference | force-free normal impactで時刻・位置・pre/post速度・charge、右連続frame、最終validity、stick/escapeとの差を確認する。曲線path、output schedule、checkpoint/resumeも既存公開経路で回帰し、外部Freeze候補はcore外V&Vとして別判定する |
| 性能・memory影響 | resident particle stateとboundary event列を再利用する。seriesへ`held` uint64列を一つ追加するだけで、particle数比例state、scratch、scheduler、diagnostic subsystemを増やさない |
| schema・revision | `particle_engine_v32`、`point_wall_laws_v5`、`durable_segmented_result_v4`、result/checkpoint schema 2。case schema v2、canonical data schema、proposal、integrator、event、physics、memory planは不変 |
| 置換・削除 | 旧result/checkpoint schema用のproduction reader/writer、compatibility shim、COMSOL名/status分岐、`freeze` aliasを併存させない。比較adapterだけがproducer raw stateをcanonical `hold/held`へ明示対応付けする |

production意味論とschema更新は上記の単一経路へ統合した。解析的直線・曲線hit、右連続frame、resume、slab identity、
OU/Brownian、zero-time surface、analysis非deposition分類を公開経路で確認し、既存hash固定Freeze candidateとの比較は
COMSOL再実行なしで15/15 PASSした。candidate/COMSOL event-time差は約`4.34e-17 s`、held位置のaligned最大差は
`2.776e-17 m`である。認定は力なしnormal-impactのterminal semanticsに限定し、full physics、grazing/corner、
普遍的COMSOL一致を認定しない。compact authorityは
[`../evidence/p18h/hold_freeze_v1/`](../evidence/p18h/hold_freeze_v1/)である。その後B03 core、charge-stable coupling、
durable cadenceとP20 performance closeoutも完了した。当時の次工程だったmeaning-matched external V&V/M3-C2Aも
後続decisionで`CLOSED_ACCEPTED_WITH_LIMITATIONS`として完了した。

## RK4 dense path revision 3：coordinate-origin-independent enclosure

| 項目 | 判断 |
|---|---|
| 解く利用case | 局所pathがfloat64で十分に解像可能でgeometryから分離していても、v2の位置boundは広い絶対座標paddingが分割後も支配し、event certificateが閉じないことがあった。物理的な曲率とenclosureは局所変位scaleで認証し、world座標の不可避な丸めだけを狭い別termで扱う |
| 所有と数値規則 | `integrators.py`がroot始点を原点とするBernstein制御点を評価・制限する。座標差はTwoDiffで丸め値と厳密残差に分解し、dense位置評価は両方をworld座標へ戻し、始終点で保存済みendpointを厳密に上書きする。enclosureは残差の絶対値をpaddingへ加え、下限を`-inf`、上限を`+inf`方向へ戻す。位置微分と物理曲率も相対差と残差を使うが、公開chord-deviation boundはworld座標評価とendpoint chordを覆う`8*eps*max(abs(root_position_controls))`の狭いtermを別に加える |
| 不変条件 | classical-RK4 endpoint、保存する制御点と数学的なHermite/state path、applicabilityのglobal-first/local-fallback、first-hit・wall・residualのevent algorithmを変更しない。`state_at()`の位置float64評価経路だけはv3の原点相対算術へ更新する。v3導入時のengine v32、proposal v8、event v14、`rk4_global_abs_enclosure_v2`、memory plan v13、case/result/checkpoint schemaは不変だった |
| revisionと履歴 | 現行dense pathだけを`rk4_position_hermite_state_extension_v3`とする。P19-L完了時点のdecision、性能値、evidence、既存resultが`rk4_position_hermite_state_extension_v2`を記録するのは履歴として保持する |

## P15-E：finite-speed free-molecular drag decision

| 項目 | 判断 |
|---|---|
| 解く利用case | 球形粒子と局所shifted-Maxwellian中性気体の相対速度が、`epstein_linear_v1`の低速域を超える自由分子流軌道を解く。model/revisionは`epstein_finite_speed` / `epstein_finite_speed_maxwell_mixed_equal_temperature_v1`とする |
| 既存modelで解けない理由 | linear revisionは`abs(u_g-v)/c_bar <= 0.1`だけを認証する。閾値を緩めたり外部dataset係数をfitしたりすると、有限速度の非線形運動量輸送と低速近似を混同する |
| 所有module | `physics/forces.py`がMaxwell球抗力、安定な無次元係数、低速級数、path適用域とglobal boundを所有する。catalogがmodel入力、runtime/compiled passが同じstage式とlocal relaxation分解を所有する。integrator、event、wall、output、checkpointは既存経路を再利用する |
| 表面散乱と入力 | `diffuse_reflection_fraction=sigma_R`を`[0,1]`で必須とし、`1-sigma_R`を鏡面、`sigma_R`を完全熱適応した等温拡散再放出とする。粒子表面温度は局所気体温度`T_w=T_g`に固定する。必要primitiveは既存の`u_g,rho_g,T_g,lambda_g`と気体分子質量、粒子の質量・抗力径で、新しいresident stateを持たない |
| 適用範囲 | 孤立・非回転・凸な球、衝突なしの局所Maxwell分布、`lambda_g/a_p >= 10`、全stageと受理pathで`S=abs(u_g-v)/sqrt(2 k_B T_g/m_g)`がcaseの有限正値`maximum_speed_ratio`以下。上限は速度clampではなくfail-closedな適用域である |
| 数値式 | `nu=nu_0[G(S)+sigma_R*pi/8]`、`nu_0=(4*pi/3)a_p^2 rho_g c_bar/m_p`として`a_drag=nu(u_g-v)`を評価する。`S<=0.1`では偶数級数、それ以外は`erf`を含むexact式を使う。`S->0`で`epstein_linear_v1`の`delta=1+sigma_R*pi/8`へ一致し、高速極限でspecular球の`C_D->2`となる |
| 積分とbound | RK4は速度Jacobianのradial固有値上界を用いて既存`maximum_dt_over_tau < 2.5`を保守的に適用する。explicit midpointはstart predictorとmidpointでlocal `nu(S)`を再評価し、midpoint係数を指数更新する。加速度・path enclosureには宣言`S`上限での係数上界を使い、別substep、自動blend、第二engineを作らない |
| referenceと統合 | 3-D Gauss--Hermiteによるspecular分子運動量積分、低速linear極限、高速`C_D`極限、係数/Jacobian bound、test-only高精度scalar ODEに対するRK4 4次・explicit midpoint 2次、reference/compiled parityを検査する。XY/RZ、材料wall、frame、checkpointは既存公開経路を再利用する |
| 性能・memory影響 | compiled rowへ`erf`を含む選択branchを一つ追加する。既存rate/target workspaceとparticle別rate boundだけを再利用し、新しいresident列、scratch、schema、schedulerを追加しない。linear revisionは低速用途として独立に残す |
| 明示的な非対象 | `T_w!=T_g`、任意energy accommodation、CLL kernel、非球形、回転、intermolecular transition regime、near-wall補正、連続流との自動切替、経験blendは別revisionであり、このmodelへfallbackや隠れparameterとして追加しない |

式はEpsteinの低速極限と、Maxwell混合境界を持つ自由分子流中の球の有限速度係数に基づく。外部COMSOL結果は
式の選択・係数調整のauthorityにせず、比較は引き続き`tools/vv`が所有する。

## P15-F：collisionless Barnes ion-drag decision

| 項目 | 判断 |
|---|---|
| 解く利用case | 外部plasma場またはreduced electrostatic builderが与える、電子と単一・単価正イオンの密度・温度、イオン流速、ion-neutral mean free pathから、負または中性の球形粒子へ働くcollection＋orbital ion dragを相対イオン流方向に加える。model/revisionは`barnes_collisionless` / `barnes_collisionless_effective_speed_single_positive_ion_negative_debye_huckel_v1`とする |
| 既存modelで解けない理由 | 中性気体dragとCoulomb electric forceだけでは、イオンの直接捕集と小角散乱による運動量輸送を表せない。M3-Vの外部式をそのまま移すと、速度floor、charge floor、Coulomb-log clamp、任意scale、電場方向またはimage補正を普遍化してしまう |
| 所有module | `physics/forces.py`がBarnes断面積、局所適用判定、global加速度boundを所有する。catalogがversioned入力を解決し、runtimeがsample済みprimitiveと連続path gate、compiled passが同じstage式を所有する。field producer、integrator、event、wall、output、checkpointは変更しない |
| 必要field/state | `Z`、particle mass、electrostatic radius、`n_e,T_e,n_i,T_i,u_i`、ion-neutral mean free path `lambda_in`、単一scalar ion mass、caseが明示する有限正値`maximum_ion_drift_ratio`。`lambda_D=[e^2/(eps_0 k_B)(n_e/T_e+n_i/T_i)]^-1/2`をrevision内で一意に作る。新しいresident state、電場方向、COMSOL名は持たない |
| 対応座標・integrator | XYとaxisymmetric RZの既存vector basisを使い、RK4とnative explicit midpointの両方で`explicit_acceleration`としてstageごとに再評価する。neutral `drag`のlinear-relaxation categoryへ混ぜず、第二integrator、subcycle、operator splitを作らない |
| 方程式と適用域 | `w=u_i-v`、`v_s^2=|w|^2+8k_B T_i/(pi m_i)`、`b_90=abs(Z)e^2/(4 pi eps_0 m_i v_s^2)`、Debye--Huckel `phi=Ze/[4 pi eps_0 a(1+a/lambda_D)]`、`b_c^2=a^2[1-2e phi/(m_i v_s^2)]`、`ln Lambda=0.5 ln[(lambda_D^2+b_90^2)/(b_c^2+b_90^2)]`、`F=n_i m_i v_s[pi b_c^2+4 pi b_90^2 ln Lambda]w`を使う。全stage/pathで`phi<=0`、`a/lambda_D<=0.1`、`b_90/lambda_D<=0.1`、`b_c/lambda_D<=0.1`、`lambda_in/lambda_D>=10`、`ln Lambda>0`、`|w|/sqrt(8k_B T_i/(pi m_i))<=maximum_ion_drift_ratio`を要求する。0.1と10はsharpな文献境界でなく、このrevisionの保守的安全率である |
| 解析解またはreference | orbital項をRutherford偏向`tan(theta/2)=b_90/b`の運動量輸送断面積として`b_c`から`lambda_D`まで独立数値積分し、閉形式と比較する。zero-relative-flow、collection-only/weak-charge極限、force方向、global bound、reference/compiled parityを検査し、一定場の公開APIで両integratorの時間収束を確認する |
| 性能・memory影響 | 既存の単一compiled row passへ一つの選択branchを加え、既存additive-acceleration列を再利用する。prepare時は粒子別static applicability 1 byte、ion velocity 2 scalar、温度下限とglobal加速度boundだけを保持する。無効時の算術・memory・公開payloadを変えず、before/afterの同一field benchmarkで確認する |
| 置換・削除する旧経路 | ion dragの旧production経路はない。外部datasetの`theory-consistent`/`image-minimal-corrected`、energy/charge floor、`min/max` clamp、任意scale、電場方向化、model blend、自動fallback、COMSOL比較分岐をcoreへ追加しない。collisional、nonlinear-screening、magnetized、multi-ion、positive-potential、image/sheath modelは別revisionとする |

`maximum_ion_drift_ratio`は式の速度を切り詰めるparameterではなく、caseが採用する検証済み範囲の上限である。
continuous chargeと併用する時は`n_e,T_e,n_i,T_i,u_i,m_i`を同じspecies/background authorityへ一致させる。
prepareはfield extremaとcharge invariantから上記条件を連続path全体で保守的に認証する。認証不能なcaseを
局所値だけで成功扱いせず、強いco-flowによる偽拒否を解消する場合も共通enclosureの別変更として扱う。

P15-Fはimpact-parameter quadrature、zero-flow/neutral limit、方向、global bound、compiled parity、
fixed/continuous chargeの同一stage結合、RK4 3.5次以上、explicit midpoint 1.8次以上、XY/RZ parityで受け入れた。
変更はcompiled tile v10、catalog v8、runtime v7だけで、engine v28、proposal/enclosure、runtime layout、
memory plan、case/result/checkpoint schemaを維持する。無効modelの2,000粒子field benchmarkは編集前後で
warm simulate 0.046734/0.045303 s、公開end-to-end 0.059475/0.057424 s、solver-owned memory
9,063,029 Bで、測定上の退行を示さなかった。有効modelの100,000-row warm stage評価は同一machineで
median 0.014814 s、prepared boundは1,700,024 Bだった。

## P16：Waldmann--Gallis free-molecular thermophoresis decision

| 項目 | 判断 |
|---|---|
| 解く利用case | 外部熱流体・希薄気体計算が与える局所並進熱流束から、気体分子の平均自由行程より十分小さい球形粒子へ働く熱泳動力を計算する。category/model/revisionは`thermophoresis` / `waldmann_gallis` / `waldmann_gallis_free_molecular_single_species_heat_flux_v1`とする |
| 既存modelで解けない理由 | dragは相対運動による運動量緩和だけを表し、非平衡な熱流束による正味運動量輸送を表さない。coreが温度fieldを再微分したり、Waldmann、Talbot、continuum相関を自動blendしたりするとfield生産と軌道計算の責務が混ざる |
| 所有module | `physics/forces.py`がWaldmann--Gallis式、局所適用判定、global加速度boundを所有する。catalogはmodel入力とsingle-gas整合、runtime/compiled passはsample済みprimitiveのstage評価だけを所有する。gradient回復、Fourier熱流束生成、COMSOL比較はadapter/V&V側に置く |
| 必要field/state | 粒子massとgas-collision径である`drag_diameter_m`、gas velocity `u_g`、並進温度`T_tr`、局所mass-average frameの並進熱流束`q_tr [W/m^2]`、gas mean free path `lambda`、単一scalar分子質量`m_g`。新しいresident state、particle thermal conductivity、total/convective heat flux、producer固有名を持たない |
| 対応座標・integrator | `q_tr`と`u_g`はXYまたはaxisymmetric RZの既存vector basisを使う。RK4とnative explicit midpointの両方で`explicit_acceleration`としてstageごとに再評価し、drag、ion drag、electric、gravityと同じcompiled passで加算する。第二integrator、subcycle、operator splitを作らない |
| 方程式と適用域 | `a=drag_diameter/2`、`c_bar=sqrt(8 k_B T_tr/(pi m_g))`、`xi_1=32/(15 pi)`、`F_th=xi_1 pi a^2 q_tr/c_bar=(32/15)a^2 q_tr/c_bar`とする。全stage/pathで`lambda/a>=10`、`abs(u_g-v)/c_bar<=0.1`を要求する。10と0.1はsharpな文献境界でなくv1の保守的policyである。single species、球、dilute one-way、粒子表面温度偏りを解かない範囲に限定する |
| 解析解またはreference | 一次Chapman--Enskog分布の熱流束momentと球への運動量transfer momentを独立3-D Gauss--Hermite積分し`xi_1`を再現する。zero heat flux、方向、半径二乗・質量逆数scaling、global bound、local/path gate、reference/compiled parityを検査し、affine fieldの公開APIで両integratorの時間収束とXY/RZ parityを確認する |
| 性能・memory影響 | 既存単一compiled row passへ一つの選択branchを追加し、既存additive-acceleration列を再利用する。prepare時に粒子別static applicability 1 byte、gas velocity 2 scalar、温度下限とglobal加速度boundだけを保持する。無効時のbefore/after、100,000-row有効stage、solver-owned memoryを同一machineで測る |
| 置換・削除する旧経路 | thermophoresisの旧production経路はない。`q_tr=-kappa_tr grad(T)`を使うproducerは、translational conductivityと`lambda abs(grad(T))/T<=0.01`をproducer側で認証する。total conductivity、mixture effective mass、Talbot/continuum、near-wall補正、negative thermophoresis、accommodation fitting、自動fallbackはv1へ追加しない |

このheat-flux形は、Fourier則を満たす単一気体では古典Waldmann勾配式へ厳密に戻る。一方、coreは
`T`からgradientを作らず、外部solverが与えた非連続域の並進熱流束も同じcanonical primitiveとして読む。

完了時には、独立3-D Gauss--Hermite momentから得た`xi_1`で具体的なproduction加速度まで比較し、方向、zero、
particle/temperature scaling、global bound、局所/path適用域、compiled parityを検証した。affine heat-fluxの
公開caseはRK4 3.5次以上、exponential midpoint 1.8次以上で収束し、XY/RZ parityも合格した。変更は
compiled tile v11、catalog v9、runtime v8だけである。thermophoresis-onlyの100,000-row warm stage評価は
同一machineでmedian 0.014549 s（約6.87 M row/s）、prepared boundは1,700,024 Bだった。
複数speciesを扱う場合はspecies別`q_tr,m_g,xi`の和が必要であり、aggregate値へ暗黙変換せず別revisionとする。

## B01：inertial Brownian numerical foundation decision（履歴）

| 項目 | 判断 |
|---|---|
| 解く利用case | 線形抵抗と局所的に凍結した気体温度の下で、慣性Langevin方程式の位置・速度を同時に進めるための数値primitiveを用意する。B01はproduction case能力ではなく、Stage 2Bのevent統合前に数式・乱数・分割意味論を固定する基盤である |
| 既存modelで解けない理由 | RK4へwhite-noiseの「力」をstageごとに加えるとstep・stage・event refinementで別の確率過程になり、exponential midpointも位置と速度の相関した熱揺動を持たない。決定論的`state_at()`で再生成する方式も同一pathを保存できない |
| 所有module | `stochastic.py`がjoint OU平均・共分散・厳密更新とGaussian conditional split、`rng.py`が物理区間identityから正規乱数を生成する。`integrators.py`、`engine.py`、event/output/checkpointはB01では変更しない |
| 必要field/state | 数値primitiveの入力は粒子ごとの線形緩和率`gamma>0`、熱速度分散`theta=k_B T/m>=0`、平衡速度、区間長、2成分ごとに2個の標準正規数である。新しいresident state、RNG cursor、mutable generatorを持たない |
| 対応座標・integrator | B01 primitiveは当時のCartesian XYの2成分だけに固定した。RZ meridionalへ等方Brownianを流用しない。B01ではproduction integrator IDを追加せず、P17のCartesian 3-D状態変更とも結合しなかった |
| 解析解またはreference | Uhlenbeck--Ornsteinの平均とGillespieのOU過程・積分のjoint Gaussianを用い、平均、`Q_xx/Q_xv/Q_vv`、短時間ballistic極限、長時間拡散極限、Monte Carlo covarianceを検証する。half-splitは左右独立incrementの条件付きGaussianとして親endpointを保存し、unconditionalな左右独立性も検査する |
| 性能・memory影響 | NumPyのbounded batch演算と既存Philox4x32-10を使い、粒子別Python objectやstateful RNGを作らない。B01はhot loopへ未接続だったためengine memory planを変えず、kernel throughputと一時配列だけを計測した。B02がstochastic tree workをmemory plan v12へ追加した |
| 置換・削除する旧経路 | Brownianの旧production経路はない。accepted-step番号、thread順、chunk順をRNG identityに使わず、`(seed, particle_id, macro_interval, root_stochastic_interval, tree_level, tree_index, component, stream)`をauthorityとする。`root_stochastic_interval`は同じmacro interval内で壁後などに新しく始まる独立OU区間のordinalであり、初期値は0とする。壁での独立な引き直し、RZ 2-Dへの疑似3-D noise、Euler random kickは追加しない |

B01完了時点では、材料壁first-passage、finite field support、任意時刻のtrajectory/probe replay、case schema、
checkpoint-resumeをBrownian対応とは表示しなかった。次の縦切りとして定めたCartesian XY、Epstein linear drag、
fixed charge、terminalなstick/escapeだけのproduction接続は、以下のB02で完了した。B01の平均式はdrag-onlyなので、
B02でも他の決定論力や反射を黙って無視せずprepare時に拒否する。

## B02：inertial Brownian production vertical slice

| 項目 | 判断 |
|---|---|
| 解く利用case | Cartesian XYの球形粒子を、局所macro-rootで凍結したEpstein線形緩和とFDT熱揺動で進め、有限depthのstochastic path上でterminalな`stick`/`escape`材料壁、frame/probe、checkpoint/resumeを同じ三公開APIから扱う |
| 既存modelで解けない理由 | B01はjoint OU increment、conditional half-split、counter RNGだけを固定し、case選択、field sampling、材料event、出力replay、memory planへ未接続だった。OU endpointだけではstep内の壁交差時刻もtrajectory時刻の状態も定義できない |
| 所有module | `physics/catalog.py`が狭いmodel組合せとdepth、`physics/runtime.py`がFDT温度、`stochastic.py`と`rng.py`がOU increment・conditional split・物理interval identity、`integrators.py`がcubic Hermite proposal、`engine.py`が固定depth tree・event・replay、`output.py`が既存result/checkpointを所有する |
| 必要field/state | `charge: fixed`、`epstein_linear_v1`、noise model `inertial_langevin_fdt`のrevision `inertial_langevin_fdt_epstein_linear_frozen_start_v1`だけを組み合わせる。FDTの唯一の温度authorityはEpstein dragが参照する`gas_temperature_field`で、`theta=k_B T_g/mass_kg`とする。noise専用温度field、resident RNG cursor、新しいparticle stateは持たない |
| 対応座標・integrator | `cartesian_xy` / `ou_langevin` / CPUだけ。`interval_tree_depth`は整数`0..10`、prepare時に`gamma*dt<=1e6`、各macro-rootで有限な`0<gamma*h<=1e6`を要求する。追加のelectric、gravity/buoyancy、ion drag、thermophoresis、DEP、continuous charge、finite-speed/Stokes drag、RZ、反射・確率wallはfail-closedに拒否する |
| 解析解またはreference | joint OUの平均・共分散、output schedule不変性、checkpoint/resumeの公開payload identity、Hermite path上のterminal wall、平面first-passage統計のdepth 6/7/8安定化、slab幅不変性、surface release右連続性、表現不能OU rowと正常rowの混在継続を公開scenarioと独立B01 primitiveで検査する |
| 性能・memory影響 | treeはdepth-firstに走査し、一levelにつき未処理right child一つだけを保持する。stochastic tree workは一slab粒子あたり`32*(depth+4)` byteとしてmemory plan v12へ計上し、runtime layout v6のbounded slabを維持する |
| 置換・削除する旧経路 | B01時点の`noise`/`ou_langevin`一律拒否だけを上記subsetで置換する。第二engine、stateful RNG、Euler kick、RMS safety band、任意forceの黙示無視、連続OU pathの厳密first-passage solverを追加しない。COMSOL比較はcore外のV&Vに留める |

現行revisionは`particle_engine_v36`、`coupled_fixed_step_proposal_v10`、
`inertial_langevin_rz_catalog_v17`、`signed_ion_compiled_physics_runtime_v19`、
`resident_soa_serial_slab_v6`、`solver_owned_memory_plan_v13`である。compiled tile v18、event v16、
field location v4、geometry v5、boundary v5、case schema v2、result/checkpoint schema 2、result algorithm v5を使う。
manifestは`philox4x32_10_brownian_interval_tree_v1`、`inertial_joint_ou_v1`、
`conditional_gaussian_half_split_v1`、`macro_root_frozen_start_v1`、tree depth、root/split stream ID、
resolved noise model/revision、`path_kind=cubic_hermite`を記録する。resume identityはこのうちBrownian RNG/OU/split
revision、coefficient policy、depth、resolved noise model/revisionを含み、stream IDとpath kindを同名keyで重複させない。
leaf内はOU endpointとendpoint velocityから定まる
cubic Hermite numerical pathであり、有限depthの離散pathを認証する。連続OU trajectoryのexact first-passageは
B02の保証ではない。

engine v30は`gamma*h`の上限だけでは保証できないfloat64表現可能性を、root increment、conditional split、
deterministic mean適用で実際に検査する。通常はslab全体をvector化し、数値例外が出た場合だけrow別に再評価して、
不良rowを`nonfinite_physics`として最後のaccepted stateで停止する。正常な隣接row、乱数identity、残りtree pathは
継続し、入力shapeや内部不変条件の破損はrun-fatalのままとする。

## B03：charged/forced RZ Brownian composition decision（初回closeout履歴）

| 項目 | 判断 |
|---|---|
| 解く利用case | axisymmetric RZ meridionalの2自由度運動で、FDT Brownian、fixedまたはcontinuous charge、nativeまたeffective-gas線形Epstein、既存additive forceを一つの`ou_langevin` proposalで進める。等方3-D Brownianではなく、terminal wallは`stick`/`escape`/`hold`だけを扱う |
| 既存modelで解けない理由 | B02はCartesian XY・fixed charge・native linear Epstein drag-onlyに限定される。従来案のdeterministic half-step＋joint `(x,v)` OU full-step＋half-stepは、両者が`dx=v dt`を進めるため位置変化を二重計上する。kick-only化はHermite endpoint derivativeとdense chargeを不連続にする。よってStrang/K-O-K案を置換する |
| 所有module | 初回closeoutでは`physics/catalog.py`が組合せとnoise revision、`physics/runtime.py`が同一stageの線形緩和・加算加速度・charge rate、`integrators.py`がroot predictorとlinear-Z cubic proposal、`stochastic.py`/`rng.py`がexact OU・conditional split・root identity、`engine.py`がmacro-root・axis restart・event/output/checkpoint統合を所有した。現行charge表現は直後のsuperseding decisionを参照する。COMSOL比較は`tools/vv`から移さない |
| 必要field/state | FDT温度は選択した線形Epsteinの`gas_temperature_field`を唯一authorityとする。root midpointで`gamma,u,T`、全additive acceleration `a`、`G=dZ/dt`を1回評価し、`u_eff=u+a/gamma`を固定する。Zは`Z_root+G_mid*(t-t_root)`のlinear dense pathで、prepare済みinvariantを証明できない場合はfail-closedする。resident RNG cursor、noise専用field、B03専用particle stateは追加しない |
| 対応座標・integrator | `axisymmetric_rz_meridional` / 既存`ou_langevin` / CPU single-thread。coefficient policyは`macro_root_frozen_midpoint_v1`、composition methodは`stochastic_exponential_midpoint_v1`。noise-free deterministic exponential-midpoint predictorはroot midpointの係数評価点を作るだけでstateをhalf-step commitしない。joint exact OUでroot全幅を進め、既存conditional treeは同じ凍結係数で親root endpointを保存する。axis hitはprefix commit＋fold後、残時間を次の`root_stochastic_interval`ordinalと独立drawで再開し、元cubic remainderをfold/restrictしない |
| 解析解またはreference | 凍結した定数係数のmeanと`Q_xx/Q_xv/Q_vv`、noise-off決定論極限の2次、manufactured charge-electricのweak mean観測次数`>=0.9`、RZ axis restart、全optional force compiled parity、tree-depth first-passage、slab/output/checkpoint identityを受け入れた。これはRZ meridional projected 2-DOFであり、等方3-D Brownianまたは一般state-dependent SDEのstrong order/weak 2次は主張しない |
| 性能・memory影響 | 第二scheduler、particle別object、Python callbackを追加せず、当時のbounded slabとdepth-first treeを再利用した。初回B03受入revisionはcatalog v16 / engine v34 / proposal v9 / event v15 / runtime v17 / compiled tile v16 / memory plan v13。静的B03 path arrayは648 B/rowで2048 B/row上限内。2,000/20,000粒子×4構成×3反復の正式characterizationは全active・failure 0で、計時/RSSは[`evidence/b03/`](../evidence/b03/README.md)のmachine-local・non-gating証跡とする |
| 置換・削除する旧経路 | 未実装のStrang/K-O-K計画を本methodで置換する。B02の`inertial_langevin_fdt_epstein_linear_frozen_start_v1`と数値payloadはbitwise不変とする。第二engine、Euler kick、root内係数のleaf別再評価、反射・確率wall、COMSOL専用branchは追加しない。B03 coreの受入にCOMSOL再実行は不要で、将来のstochastic外部比較は独立seed ensembleとする |

noise revisionは`inertial_langevin_fdt_epstein_linear_rz_meridional_projected_v1`である。
case/result/checkpoint schemaは変更せず、resolved model、coefficient policy、RNG/OU/split revision、tree depthを
既存manifest/resume identityで識別する。engine v34はperformance characterizationで検出した反復加算の微小な終端tailを
除き、macro timeを補償積和による`start + n*dt`のindexed gridから構築する。endへのsnapはfloat64構築roundoff内だけに限定し、
科学的に意味のある残時間を消さない。現行順序は後続の100 nm・30 ms candidate-first decisionが置換し、
M3-C1 v14の既存evidence/evaluatorは履歴として変更しない。

event v15はgeometry budgetとroundoff budgetを加算し、facet-local offset dotを補償演算で、Hermite評価・包絡を
root-relative TwoDiffで扱った。現行event v16はvalidなRK4 dense rowで、integrator所有のroot-relative position
Bernstein control enclosureをfacetの外向きhalf-spaceへinterval射影する。全4制御点の外向き上限が既存position budgetの
負側へ厳密に入る候補だけをconvex-hull性からclearする。controlが不正・非有限または証明不能なら候補を残して
split/fail-closedする。monotone clearはcubic Hermite derivative-Bernstein enclosureの明示opt-inだけであり、
exponential、scalar pathへv16 certificateを適用しない。event/terminal意味、tolerance、endpoint、schemaは不変である。

## charge-stable coupling / work-scaled durable cadence（完了）

このdecisionは直前までのM3-C1/B03 evidenceを変更せず、二つのproduction sliceを完了状態として定める。

| 項目 | 判断 |
|---|---|
| charge-stable coupling | `rk4_fixed`はexplicit `h L_Z <= 0.5`を保持する。deterministic exponential midpointとB03はmidpointで`G`と`J=dG/dZ<=0`を凍結し、root基準のaffine law `G(Z)=G_mid+J(Z-Z_mid)`を`expm1`で解析更新する。fixed chargeの`J=0,G=0`極限も同じ式で扱う |
| charge受入 | production式を呼ばないaffine stiff-relaxation、nonlinear coupled `h,h/2,h/4`、electric/event/dense/resumeを検証済み。stabilityのためのclip、charge-only subcycle、第二engineはなく、精度選択は別runの`h,h/2,h/4`が所有する |
| durable cadence | `W=macro_step_count+accepted_particle_pieces+candidate_queries+refinements`、`T=max(2^20,128N)`。engineがaccepted macro barrierでepoch開始時との差`>=T`または最終macroを判定する。`cumulative_solver_work_v1`、resolved threshold、components、barrierをmanifestとresume identityへ記録する |
| ownership | engineが`W/T`とcommit要否を所有し、`output.py`は同期single-owner writerとしてsegment→inactive A/B checkpoint→`LATEST`のatomic persistenceを所有する。cadenceはoutput schedule、frame/probe、slab幅、wall clock、filesystem timestampから独立である |
| revision・順序 | engine v36 / tile v17 / proposal v10 / exponential midpoint v3 / runtime v18 / result v5 / event v16 / memory plan v13。614件PASS、P20 performance closeout完了。当時の次はmeaning-matched external V&V/M3-C2で、後続M3-C2A final decisionでanchorを完了した。COMSOL fitting、第二writer/engine、threading再導入は行わない |

## M3-C2A saved-inventory preflight decision

| 項目 | 判断 |
|---|---|
| 目的 | 保存済み100 nm Case A/Pを正式stochastic referenceへ誤昇格させず、最初のmeaning-matched ensembleに使うseedだけを確定する |
| 外部所有 | `tools/vv/comsol/m3c2_anchor_preflight.py`と`evidence/m3c2/anchor_preflight_v1/`が所有する。solver core、case schema、`model_dataset`を変更しない |
| 確認結果 | source MPHと選択package artifactはM3-C0 expected SHA lockへ一致し、Case A/P各32 seedは完全・一意・非重複。保存Case A/Pの設定parameter値は21/1だが、interface乱数modeが`GenerateUnique`なので実効乱数系列は未証明。各case 1 run、fieldはnative。particle interfaceはout-of-plane無効で、解かれる運動状態はR-Z 2自由度である。表の`r/phi/z`と非零`Fbphi`はreported feature componentであり第3運動自由度ではない |
| 判定 | 次元不一致ではなく、native field、単一run、実効乱数系列未証明、COMSOL stepwise BrownianとB03 exact joint OUの独立収束未評価がblockerである。statusは`BLOCKED_MISSING_MEANING_MATCHED_COHORT`、comparisonは`NOT_AUTHORIZED`、accuracyは`NOT_EVALUATED`。64行matrixはprovisional seed allocationだけでcampaign lockではない |
| 次の単位 | Case-A 100 nmについてcommon-P1 input hash、287粒子、121 frame/30 ms、release、geometry/boundary、全physics revision、COMSOL R/Z FDT recipe、B03 case、pilot/final stepを一つのexecution contractへ固定する。COMSOLはout-of-plane無効とbuilt-in `bf1`＋Epstein-equivalent effective viscosityを維持し、乱数modeを`UserDefined`として`bf1.i`をseed authorityにする。本比較と非重複のpilot seedで各solverのobservable step uncertaintyを決めた後だけ32 seedを実行する |
| 性能判断 | 意味一致と刻みを先に閉じ、同一workloadでprofileする。leaf/eventが支配した場合だけ単一engine内のbounded batchを改良する。seed/caseは外部processの1/2/4 worker実測後だけ採用し、solver threading、第二engine、効果のないparallel wrapperを残さない |
| 入力・意味契約 | `caseA_100nm_pilot_contract_v1`でsource/common-P1 identity、287粒子、30 ms/121出力、release/geometry/boundary、全決定論revision、R-Z Brownian target、pilot/final seed分離をfail-closed固定し、`PASS_INPUT_IDENTITY_AND_SEMANTICS_LOCKED`。runner未固定・pilot未実行なのでexecutionは`NOT_AUTHORIZED`、accuracyは`NOT_EVALUATED` |
| 当時の次 | no-save COMSOL runnerで`UserDefined`と`bf1.i`を適用するrecipe、candidate runner、各solver独立のstep/tolerance/observable pilot recipeを固定し、分離済みpilotを受理してから32-seed anchorへ進む。これは後続のM3-C2A final decisionで完了した |

保存modelのR-Z DOFと乱数modeのread-only authorityは
[`../evidence/m3c2/model_semantics_probe_v1/`](../evidence/m3c2/model_semantics_probe_v1/README.md)である。
入力・意味契約authorityは
[`../evidence/m3c2/caseA_100nm_pilot_contract_v1/`](../evidence/m3c2/caseA_100nm_pilot_contract_v1/README.md)である。

## M3-C2A Case-A 100 nm final stochastic decision

| 項目 | 判断 |
|---|---|
| 比較scope | common-P1 Case-A 100 nm、R-Z 2自由度、287固定発生源、30 ms、121観測時刻。COMSOL built-in `fptas` companionとproduction B03 candidateを同じsource、geometry、boundary、決定論力、charge、FDT targetで比較する |
| 数値選択 | final seedと非重複のpilotをconfiguration screeningに限定し、COMSOL classical RK4 20 us、candidate 20 us / Brownian tree depth 3 / geometry rtol 1e-8を選択した。4-seed pilotを95%推論へ使ったV2解釈は無効化した |
| final実行 | COMSOL seed 318160--318191、candidate seed 318192--318223。各32 replica、各287粒子×121時刻=34,727行。candidate failure 0、境界event 8,451、COMSOL境界event 8,448、COMSOL study isolation 32/32 PASS |
| 確認的gate | `stuck/held/escaped/any_terminal`の4曲線×121時刻をseed×固定source populationで評価するtwo-sample union-Hoeffding。最大差0.00598868、95%同時半径0.0327841、上限0.0387728で事前登録margin 0.05以内、`PASS` |
| 軌道の補助評価 | 平均位置差最大1.148 mm、R-Z占有total variation最大2.588%、分位点差最大5.928 mm。全287発生源のparticipant別seed平均R-Z overlayを保存した。これらは非判定の説明指標であり、後半のcontinuous指標は両者全seedでactiveなsourceが減る点を併記する |
| 認定範囲 | 固定済みcommon-P1 anchorの終端人口精度について「COMSOLと約5%幅内で同等」を認定する。独立RNGなのでpathwise一致は評価しない。native field、Case P、第2 ion-drag、10/30 nm、3-D、普遍的COMSOL同等性へ一般化しない |
| core境界 | evaluator、runner、図、証拠は`tools/vv/comsol`と`evidence/m3c2`が所有する。production coreへCOMSOL分岐、比較専用tolerance、第二engine、threadingを追加しない |
| 当時の次 | 同一契約を一軸だけCase P 100 nm・30 msへ展開し、meaning-matched pilot/final accuracy anchorを先に完了する計画だった。これは直後のCase-P final decisionで完了した。solver内部のthread幅選択を再導入しない原則は維持する |

compact authorityは
[`../evidence/m3c2/caseA_100nm_final_campaign_v1/`](../evidence/m3c2/caseA_100nm_final_campaign_v1/README.md)である。

## M3-C2A Case-P 100 nm final stochastic decision

| 項目 | 判断 |
|---|---|
| 比較scope | common-P1 Case-P 100 nm、R-Z 2自由度、287固定発生源、30 ms、121 frame。COMSOL/candidate各32独立seedを20 usで比較する |
| 確認的gate | policy revision 5の83区分R-Z/fate gateは最大empirical TV `0.010670731707317093`、同時上限`0.13119968456545308 < 0.15`で`PASS`。終端人口gateも`PASS`だが、全64 replicaでevent 0のため境界parityには情報を持たない |
| 認定範囲 | 元Case-P COMSOL `auxq`が意図して採用する電子＋正イオン二電流same-form数値モデルの登録人口observableだけを認定する。負イオン密度は元modelでも診断量であり、二電流採用をcandidateの欠損とは扱わない。一方、species-resolvedな物理真値、後続の三電流拡張、pathwise RNG、eventful boundary parity、native-field parity、普遍的COMSOL同等性は認定しない |
| core境界 | 比較は外部V&Vに留め、production core、公開API、solver dependencyを変更しない。final計時は`NON_AUTHORITATIVE_EXTERNAL_WORKLOAD_OVERLAP`で性能根拠にしない |
| 性能owner discovery | accepted candidate seed `319032`、`319047`、`319063`の287粒子で完了。全rerunは受理済み科学payload・work・case identity・revisionと完全一致。支配ownerは3 seedとも`integrators`、自己時間比42.58--42.86%。ただし事前登録済みbounded ownerではないため`optimization_authorized=false`で本体変更なし。authorityは[`../evidence/m3c2/caseP_100nm_owner_profile_v1/`](../evidence/m3c2/caseP_100nm_owner_profile_v1/README.md) |
| bounded chord follow-up | 後続の明示指示で`curved_chord_deviation_bounds`だけをserial compiled batchへ置換し、scalar helperとevent側重複式を削除。accepted 3 seedのpublic end-to-end中央値は45.4668 sから39.8801 sへ12.29%短縮し、科学payload・revision・workは完全一致。新runtime/設定/診断なし。ここで停止し、10,000粒子scale、次owner、COMSOL速度比は認定しない。authorityは[`../evidence/m3c2/caseP_100nm_chord_optimization_v1/`](../evidence/m3c2/caseP_100nm_chord_optimization_v1/README.md) |
| closeout | Case-A/Case-P 100 nm anchorは`CLOSED_ACCEPTED_WITH_LIMITATIONS`。追加COMSOL、追加seed、10,000粒子以上のscale、残るpackage、process並列を完了条件にしない。性能は利用SLAを先に定義した独立work packageでのみ評価し、solver内部threadingは再導入しない。終了条件のauthorityは[`../../implementation_plan.md`](../../implementation_plan.md) |

compact authorityは
[`../evidence/m3c2/caseP_100nm_final_campaign_v1/`](../evidence/m3c2/caseP_100nm_final_campaign_v1/README.md)である。

## P21 / M3-C3：aggregate three-currentとcritical 2-D closure decision

| 項目 | 判断 |
|---|---|
| 解く利用case | 外部producerが電子、aggregate正イオン、aggregate単一価負イオンのcanonical primitiveを供給する時、動的電荷とR-Z粒子運動を同じsingle engineで連成する。元Case-P二電流anchorは変更せず、拡張を明示選択した別caseだけで使う |
| 既存modelで解けない理由 | `aggregate_relative_drift_regularized_two_current_v1`は元Case-P `auxq`とのsame-form比較には正しいが、負イオン収集電流を物理選択として加えるcaseを表現できない。既存revisionへhidden branchやCOMSOL名を加えるとmodel意味が変わる |
| 所有module | 純粋rate/Jacobian/boundは`physics/charge.py`、field契約と選択は`physics/catalog.py`、samplingとcompiled評価は既存`physics/runtime.py`・`physics/compiled.py`が所有する。比較runner、COMSOL model、判定、図は`tools/vv/comsol`と`evidence/m3c0`/`evidence/m3c3`が所有し、coreからimportしない |
| 必要field/state | 既存二電流fieldに、有限非負`negative_ion_number_density`、正値`negative_ion_thermal_voltage`、座標vector`negative_ion_velocity`、正値`effective_negative_ion_mass`を加える。screeningは既存の明示`screening_length`だけがauthorityであり、負イオンprimitiveからcoreが再計算しない。resident stateは既存`Z`だけで増やさない |
| 物理式と適用域 | `R_Z=Gamma_+-Gamma_e-Gamma_-`。負イオンはaggregate singly-negative-ion collectionで、二電流revisionと同じ正則化speed、energy floor、branch、有限invariant/Jacobian boundを使う。一つの`maximum_relative_ion_speed_m_s`で正負両イオンを囲い、`n_-=0`ならrate/Jacobian/global boundは二電流へ厳密退化する。ただし三電流revisionの負イオンfieldと相対speed applicabilityは密度0でも検査する。species-resolved current、emission、collisional/magnetized sheathは範囲外 |
| 対応座標・integrator | 現行XY/R-Z、fixed RK4、deterministic/B03 exponential charge couplingを共有する。Case-P専用分岐、第二engine、charge-only subcycle、新しいstatus/diagnostic subsystemは作らない |
| 解析解またはreference | 独立式oracle、正負電位branch、zero-densityでのrate/Jacobian/global-bound退化、finite global bound、compiled parity、公開API scenarioをcore受入にする。外部V&Vは(1) critical boundary microcase、(2) 三電流を明示したCase-P派生companion一件だけとし、全12 package総当たりを行わない |
| boundary priority 2 | `evidence/m3c0/critical_boundaries_v1`はCOMSOL/public API/解析解を同じ3粒子で比較し、surface contact departure、同step残時間を含むspecular reflection、R-Z axis passageを`dt=1/0.5/0.25 ms`で132/132 PASSした。最大solver間位置差は`1.61339e-17 m`。grazing/corner、multiple/probabilistic、力・native fieldは認定しない |
| 性能・memory影響 | 追加は選択時だけ4 fieldと既存bounded slabのsamplingを使う。catalog v17 / runtime v19 / tile v18でsingle-thread compiled engineとresident state/schemaを維持した。代表runで実測上のownerが現れるまで、新scheduler、threading、model別kernel frameworkを追加しない |
| 置換・削除する旧経路 | 置換なし。二電流revisionと既存M3-C2A evidenceを保持し、三電流は別revisionにする。比較用のcore branch、負イオンからのscreening再計算、重複validator、全case診断を作らない |
| 状態 | priority 1 productionは標準verification/scenario suiteと品質gateを通過して`COMPLETE`。priority 2も`COMPLETE/PASS`。priority 3はcanonical負イオンprimitive authority不足で`BLOCKED / NOT_EVALUATED`として入力監査を終了し、軌道は未実行。元Case-P二電流anchorは不変。この任意物理の外部coverageを出口から分離し、P21/M3-C3と明示scopeの2D benchmarkは`CLOSED_ACCEPTED_WITH_LIMITATIONS` / `2D_CRITICAL_VV_COMPLETE`。三電流のCOMSOL軌道同等性は非認定 |

## F01：reduced electrostatic field builder decision

| 項目 | 判断 |
|---|---|
| 解く利用case | canonical thermal-flow field、bulk plasma parameter、semantic electrostatic boundary groupから静的なpotential/E/plasma primitiveを作り、外部plasma場と同じ`case.h5` interfaceでparticle solverへ渡す。最初のmodel IDは`boltzmann_bohm_sheath_c2_v1` |
| 既存modelで解けない理由 | 熱流体場と任意parameterだけでは電場は一意でなく、Poisson equation、charge closure、material domain、potential/flux BCが必要である。これをtrajectory engineへ埋め込むとfield収束とparticle積分の責務が混ざる |
| 所有module | `tools/electrostatic_builder/numerics.py`がclosure、軸対称P1 weak form、Newton/GMRES、field recoveryを所有する。`workflow.py`がstrict YAML、semantic BC、canonical read/write、provenanceだけを所有する。`chamber_particles`は完成fieldの生成方法を知らない |
| 必要field/state | 入力はgeometryと完全一致するfully-supported triangle-only P1 layout、node `gas_velocity(r,z)`、node `gas_temperature`、単一準中性bulk density、`Te[V]`、aggregate positive-ion mass/mobility、bulk potential、明示BC/continuation。出力は`electric_potential`、`electric_field`、`ne/ni/rho`、`Te/Ti`、positive-ion velocity/speed/mass。input field/sourceは保持する |
| 対応座標・解法 | v1は静的`axisymmetric_rz`だけ。`2 pi r`付きP1 Galerkin、triangle三点quadrature、analytic `d rho/dV`、明示continuation、damped Newton、非正定値Jacobianを許す行列フリーrestarted GMRES、mass-lumped nodal gradientを使う。`r=0`は自然対称でDirichlet/material wallにしない |
| 解析解またはreference | C2 gate/Jacobian finite difference、`V=Vp`準中性解、軸方向affine Laplace、annular `A log(r)+B` Laplace収束、非線形sheath、nested 2-D mesh self-convergence、global charge balance、axis `Er=0`、canonical round-tripをCOMSOLなしで検査する。M3-Vのclosure保存値との式parityはprovenance確認に限り、field parityは外部V&Vで行う |
| 性能・memory影響 | field buildはtrajectory hot loop外で一case一回実行する。疎local matrixとKrylov basisは`O(nodes + cells + nodes*krylov_dimension)`で、dense global matrixを作らない。solver runtime dependencyは追加せずNumPyだけを使う。大規模性能はcanonical Case-A adapter完成後に別計測する |
| 置換・削除する旧経路 | `caseA`/COMSOL tag、数値boundary ID、particle diameter依存派生場をproduction modelへ持ち込まない。自動closure/solver fallback、mixed-element第二runtime、field comparison用core分岐、汎用PDE/plugin frameworkは追加しない |

closureは`psi=(V-Vp) H2(-(V-Vp), dV)`、Boltzmann electron、collisionless Bohm-ion、
`rho=e(ni-ne)`である。`H2`は明示C2 quintic smooth Heavisideで、指数clipを最終方程式へ
入れず、density/ion-speed floorのactive数をreportする。`electron_temperature_V`はeV相当をpotential [V]
として表す量であり、canonical `electron_temperature`はKへ変換する。

ion velocityは`Gamma_i=ni ug + mu_i ni E - Di grad(ni)`の方向とBohm energy速度を組み合わせる。
`grad(ni)`はnodal densityの再微分ではなくanalytic `(dni/dV) grad(V)`を使う。これはion continuityや
species別輸送を解いたものではない。positive-ion temperatureはv1ではgas temperature equilibriumである。

共有cornerの異なるDirichlet値は明示priorityで一意にし、同priority conflictを拒否する。参照Case-Aの
triangle/quad mixed domainはadapterが品質基準付きで決定論的にP1 triangulateし、boundary ownerを再構成する。
builder/runtimeへmixed topologyを追加しない。P15 stationary OMLまたはP15-D shifted-Maxwellian OMLが生成fieldへ
適用できるかは、それぞれのspecies・電位・drift契約で独立に決まるため、F01完了はfull trajectory parityを意味しない。
F02で代表P1 meshのlinear solve、fixed charge/electricのfield-production integration、同一export node上の外部field比較まで
完了した。P15-Dはその後、field producerから独立したproduction physics revisionとして追加した。

## F02：provider adapter / representative integration decision

| 項目 | 判断 |
|---|---|
| 解く利用case | COMSOL等の外部producerが出したaxisymmetric mixed triangle/Q1-quad mesh、boundary entity、thermal-flow node fieldをcanonical P1へ変換し、F01 builderと既存particle solverへ一貫して渡す |
| 既存modelで解けない理由 | F01はcanonical triangle-only P1を前提とし、参照packageは2,127 triangle＋826 quad、producer固有ID/列/axis seamを持つ。これをbuilder/coreへ直接読ませるとprovider syntaxと数値責務が混ざる |
| 所有module | `tools/comsol_adapter`がstrict CSV/YAML、domain抽出、P1化、boundary再構築、field転記を所有する。F01 builderがfield equation、`tools/vv/comsol/compare_reduced_fields.py`が外部差分、既存三APIが完成fieldの粒子計算を所有する |
| 数値規則 | Q1 node順を明示し、二対角の最小triangle品質が高い方を選び、完全同点だけexternal node IDでtie-breakする。外周/ownerはP1 incidenceから再構築する。新規exportはtopology ID結合を原則とし、node IDがないlegacy CSVだけは設定tolerance内のexactly-one全単射を証明して許可する。補間・埋めはしない。axis radial velocityは明示上限内だけ0へ投影する |
| representative gate | 1,987 node / 3,779 P1 cell、1,826 free nodeで最終relative residual `1.8441e-13`、charge-balance error `5.8498e-21 C`、GMRES storage 1,235,088 B、solve-only 0.689/0.737 s。現GMRESを維持し、fallbackを追加しない |
| integration | wafer surfaceから32粒子をfixed charge＋Coulombだけで10 macro step進め、3 frame/96 row、failure/wall event 0。これは生成fieldが既存RZ solverで消費されることのsmokeであり、COMSOL trajectoryまたはFreeze parityではない |
| external V&V | `reduced_field_comparison_v1`はaxisymmetric lumped-volume norm、semantic boundary trace、builder charge balanceを記述する。同一export nodeだけTESTED、独立mesh convergenceは`NOT_TESTED_SINGLE_REFERENCE_MESH`、trajectory parityはN/A。差を消すcore/builder分岐を作らない |
| 性能・memory影響 | adapter/builderはtrajectory hot loop外の一回処理。particle engine、case/result schema、algorithm revision、runtime dependencyを変更しない。小さいhash付き証跡を`evidence/f02/`に保存し、生成HDF5/resultはsource管理しない |
| 置換・削除 | mixed-element builder/runtime、COMSOL boundary law自動写像、solver fallback、比較専用physics mode、dataset golden testは追加しない。Case-P直接plasma-field入力はspecies/primitive契約を固定した別T02 adapter sliceとする |

## P03 field location revision 2

| 項目 | 判断 |
|---|---|
| 解く利用case | staticなregular/P1/Q1場をXYまたはRZ粒子位置で決定論的にsampleする |
| 既存modelで解けない理由 | P00～P02はschemaとoracleだけで、productionの座標変換・point location・補間を持たなかった |
| 所有module | `coordinates.py`がRZ axis fold、`fields.py`がlocation・support・補間を所有する |
| 必要field/state | canonical layout/value/support、2成分位置、任意のsearch-only cell hint |
| 対応座標・integrator | `cartesian_xy`と`axisymmetric_rz_meridional`、integrator非依存 |
| reference | C06のtwo-cell affine P1、非affine mapped Q1、regular bilinearのmanufactured field |
| 性能・memory影響 | 初期single-point経路。配列を複製せず、一sample分の小さいlocation/valueだけ返す |
| 置換・削除する旧経路 | なし。旧solverのsamplerは移植せず、production samplerはこの一経路だけ |

supportはsupported cellの閉包の和とし、共有面では最小supported cell IDをownerにする。hintで結果を
変えない。P03ではBVH/cell walkやbatch kernelを先回りして追加せず、P10で同じ意味論をcompiled CPUへ移す。
P1/Q1の別々のpolicy layerやproducer別samplerは作らない。
P1/Q1 locatorのRadon CC 12は、全cellを一度走査してinside候補とoutside provisional候補を同時に作る
単一数値loopに由来する。CCだけを下げる薄い走査helperへ分割せず、共通のcandidate選択だけを一箇所にした。

`field_location_v2`はrevision 1を置換したP03時点のproduction locatorであり、P14の`field_location_v3`が
同じ数値意味をsupported-containment BVHへ移して置換した。CCW physical polygonの
signed-distance判定をsupportの主判定とし、P1/Q1逆写像は局所座標で行う。座標ULPとelement径から求めた
物理誤差を`||J^-1||`で参照空間へ写し、`cond_inf(J) <= 1/sqrt(eps)`、参照不確かさ
`<= 8 sqrt(eps)`、局所再構成残差を同時に要求する。これらはalgorithm定数でありcase設定にはしない。

support外のprovisionalは、全supported cellの物理閉包へ射影したEuclidean距離で選ぶ。同距離は最小cell ID、
補間weightは非負かつ総和1とする。masked cellのplaceholderや参照座標上の近さを値ownerに使わない。
supported cellがない、cellがfloat64で解像不能、またはsampleが非有限なら`FieldLocationError`とする。

P1/Q1のlarge-offset・high-aspect edge、誤差budget外、物理最近傍、masked regular、全masked、極端な有限値、
orientation overflowをverificationへ追加済みである。平行移動後も局所geometryがfloat64で解像できる範囲では
owner/supportを維持し、解像不能な移動はtolerance拡大で隠さず明示errorとする。旧revision selectorはない。

## P04 ballistic/result revision 1

| 項目 | 判断 |
|---|---|
| 解く利用case | 時刻の異なるtable particleをfield・boundaryなしでballisticに進め、finalと明示時刻frameを読む |
| 既存modelで解けない理由 | P00～P03は入力とsingle-point numericsだけで、production engineと永続resultを持たない |
| 所有module | `sources.py`がstable release schedule、`integrators.py`がballistic proposal、`engine.py`が唯一のloop、`output.py`がresultを所有する |
| 必要field/state | fieldなし。固定IDのSoA位置・速度・charge・lifecycle・release cursorだけ |
| 対応座標・integrator | 初期は`cartesian_xy`と`axisymmetric_rz_meridional`、ballisticだけ |
| reference | C01解析解と、release時刻が異なる2粒子の非macro frame scenario |
| 性能・memory影響 | 全particle stateはresident、frameは時刻単位でstreamし、全粒子×全時刻を保持しない |
| 置換・削除する旧経路 | `simulate/open_result`のfail-closed stubを同じchangeで削除し、別reference engineを作らない |

P04のtrajectory入力は`selection: all`と`schedule.explicit_times_s`だけを受ける。時刻は有限・狭義単調増加・
重複なし・閉区間`[start_s,end_s]`内とする。release前の粒子はframeに含めず、release時刻ちょうどは入力tableの
初期状態を保存する。finalとrelease eventは常に保存し、粒子行は`particle_id`順とするため、`events`と
`final_particles`のenable flagは未release schemaから削除する。

P04は`run.json`、一つのclosed frame/event segment、`final.h5`、`_SUCCESS`と最小lazy `ResultView`だけを
実装した。checkpoint、`LATEST`、background writer、multi-segment recoveryは後続P13で追加した。
P04時点は`open_result(recovery=True)`を明示的に拒否した。output scheduleを変えてもfinal stateを変えない。
P04時点の実行profileは`rk4_fixed + cpu`で、requested thread数はprovenanceへ残すがresolved thread数は1とした。
P04時点は確実にmaterializeする配列の下限がresource上限を既に超える場合だけ開始前に拒否した。
P09の`solver_owned_memory_plan_v1`がこの下限をload/prepare/run phase planで置き換えた。

release eventは`(release_time_s, particle_id)`順、frameとfinalの行は`particle_id`順とする。ballistic
proposalはmacro-step開始状態からの累積演算ではなくtableの解析的release原点を保持し、`dt_s`やframe
scheduleを変えても共通時刻の状態をbitwiseで変えない。resultのdataset、dtype、公開readerは
[`result_format_v2.md`](result_format_v2.md)が所有する。

HDF5側のdata座標表現とYAML側のparticle motion modeは別のauthorityにする。v1で有効な組合せは
`cartesian_xy + cartesian_xy`と`axisymmetric_rz + axisymmetric_rz_meridional`だけで、prepareが一度だけ
検査し`run.json`へ両方を記録する。将来の組合せを動かす互換分岐はP04で作らない。
RZの軸へ到達・横断するballistic pathはwallとして扱わず、path分割を所有するP07まで明示的に拒否する。
DataBundle内の未参照geometry/layout/fieldは再利用可能な入力資産として許す一方、設定されたboundaryや
fixed charge以外のphysicsは黙って無視せずprepareで拒否する。

## P02で行ったschema判断

- fixed chargeの初期`Z`をphysicsから削除し、sourceだけをauthorityにした。
- `solver.event`へgeometry相対許容差、roundoff ULP、refinement/interaction上限、corner policyを置いた。
- 各boundary groupへ非負`priority`を必須化した。
- `priority_then_combined_normal_v1`を初期corner policyとした。
- surface sourceへ非重複の連続粒子ID範囲を予約する`particle_id_start`を必須化した。
- `boundary_id`の重複を許し、canonical facet identityを`line2` rowの`facet_id`へ固定した。

これらは未releaseのformat v1を実装前に整合させた変更であり、compatibility shimや旧入力経路を残さない。

## 明示的な非決定

次はP02でdefaultを捏造しない。

- 実チャンバー全体へ適用できる単一drag相関
- CF4/O2向けStokes–Cunningham係数
- material別付着率・反発係数
- performanceの絶対合格秒数
- 3D、時間依存field、continuous chargeの設定値

各機能を実装するpackageで、入力case、独立reference、memory/performance影響とともに決める。
