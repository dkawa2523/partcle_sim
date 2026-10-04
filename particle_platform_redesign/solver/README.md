# Chamber Particles

半導体製造チャンバー内の粒子軌道を、外部から与えた場とgeometryから計算するための
clean-room solverです。このdirectoryは旧実装から独立したuv projectであり、COMSOLや
`model_dataset/`をruntime dependencyにしません。

## 現在の到達点

P00からP05と、P06 revision 1の物理式・RK4 reference、revision 2の証明済み一定加速度経路、revision 3aの
boundaryless RK4 enclosure、revision 3bの一般RK4材料壁経路まで実装しています。revision 3bの数値pathは
`coupled_rk4_engine_v6`で確立し、現行の`particle_engine_v36` / `compiled_cpu_tile_v18` /
`coupled_fixed_step_proposal_v10` / event v16 / physics catalog `inertial_langevin_rz_catalog_v17` /
physics runtime `signed_ion_compiled_physics_runtime_v19` /
RK4 enclosure v2 / dense path `rk4_position_hermite_state_extension_v3` /
charge-stable exponential midpoint `charge_stable_exponential_midpoint_v3`・enclosure v3 /
field location v4 / runtime layout v6 / memory plan v13 /
boundary algorithm `point_wall_laws_v5`では、
一定加速度に加えて、fixedまたはP15/P18-C continuous charge・Cartesian XY/RZ・Epstein/Stokes--Cunningham・electric・
gravity・thermophoresis・ion drag・quasistatic spherical DEP・RZ rarefied-vorticity lift sensitivityの
一般`rk4_reintegrated`を公開force-coupled runとして
扱います。boundarylessはfully-supported common
`RegularLayout`、材料domainはそれに加えてvolume meshと完全一致するfully-supported P1/Q1を利用できます。
P03ではreviewで再現したlarge-offset・高aspect P1/Q1の境界誤分類をv2で置換し、P14で同じ意味論の
supported-containment BVHを持つfield location v3へ更新し、P19-Lのbounded local rangeをv4で追加しました。
outside/masked provisionalはO(cell数)です。
Python 3.12、runtime
dependency、品質tool、build設定をlockし、利用者向け三APIと公開例外を一つのpackage境界から
importできます。

- `load_case`は厳格なYAML全体とresource上限を先に読み、HDF5 metadataからcanonical numeric
  footprintを求めて上限外をpayload展開前に拒否します。その後producer-neutralなHDF5 v1を検査し、
  logical content hash、source、boundary参照を照合します。
- importer向け`chamber_particles.case_format`はcanonical `DataBundle`のread/write/hashを所有します。
  package rootの利用者向けAPIは増やしていません。
- 入力schemaの正確な契約は[`docs/case_format_v2.md`](docs/case_format_v2.md)、resultの契約は
  [`docs/result_format_v2.md`](docs/result_format_v2.md)にあります。
- P06-S/P08の受入範囲と再実行gateは
  [`docs/stage1a_validation.md`](docs/stage1a_validation.md)にまとめています。
- P10 compiled CPU、P11 exponential midpoint、P12 event-heavy CPU並列、P13 durable result、P14 performanceのrevision、
  意味論parity、数値受入、性能観測とP14-P parallel runtime収束gateは
  [`docs/stage1b_validation.md`](docs/stage1b_validation.md)にまとめています。
- P14-Pの実測根拠、否定的closeout、直列runtimeの採用理由と再検討条件は
  [`docs/parallel_execution_plan.md`](docs/parallel_execution_plan.md)を権威とします。
- C01～C10は`tests/verification/microcases.py`で一時生成し、入力をcanonical writer、公開loaderへ
  通したうえで、productionから独立した解析期待値を`expected.json`へ分離します。式と有効化順は
  [`docs/numerics.md`](docs/numerics.md)、初期物理revisionは
  [`docs/physics_models.md`](docs/physics_models.md)にあります。
- `coordinates.py`はRZのsigned RK4 stageとcanonical field基底、軸通過、support区間像を所有し、`fields.py`はstaticな
  regular/P1/Q1 layoutのlocation、support、node/cell補間を一つの意味論で実装します。production tileはregularの
  supported containing-cell common pathでO(1)個の候補を使い、outside/masked provisionalはcompiled全cell走査へ
  戻ります。P1/Q1はprevious-cell strict-interior fast pathとNumba内full-search fallbackを使います。scalar経路は
  verification oracleだけで、production fallbackではありません。C06に加え、
  regular bilinear、共有面、mask境界、hint不変性、large-offset/high-aspect、最近傍supported射影、
  非有限failureをverificationしています。悪条件または座標分解能外のcellは明示errorにします。
- `sources.py`はtable/surfaceを単一`ParticleSchedule`へ安定順序で統合する。surfaceは固定時刻、
  `edge_fraction`または明示measureのuniform位置、固定vectorまたは固定speed inward-normal速度を受理する。
  XYは`line_length`、RZは`meridional_length | revolved_area`を明示する。`rng.py`のPhilox4x32-10は
  sourceとprobabilistic wallをparticle identityから再現する。`integrators.py`は一つの`StepProposal`で
  classical RK4、exponential midpoint、無力場の`linear_exact`退化形を所有します。`engine.py`は唯一のmacro-step loopを持ち、
  個別release時刻やframe時刻のために全粒子stepを分割しません。
- P09で`sources.py`は各sourceを最終scheduleへ直接scatterし、一時copyを減らしました。現行`cpu.py`は
  固定ID対応のresident state、resident-row active index、bounded slabとmemory planだけを所有します。
  field/physics/integratorはslab幅で一度確保したworkspaceへ書くsingle-thread compiled engineです。slab幅を
  変えてもevent/failureは物理keyで並べ、final/event/RNG/outputの意味を変えません。
- P10ではNumba 0.67（NumPy `<2.6`）でfield sampling、sample済みprimitiveのphysics、classical RK4算術を
  compiled array passへ置換しました。P14-Pで同じownerのpreallocated `*_into` passへ整理し、最終的に
  `fastmath=False, parallel=False`の単一直列runtimeへ収束しました。P1/Q1のresident hintはaccepted endpointだけでcommitし、
  `state_at()`、wall hit、residual、outputを別engineへ分岐しません。
- P11ではphysics runtimeが全加速度と同時に`linear_drag_rate_s_inv`、`target_velocity_m_s`、
  `additive_acceleration_m_s2`を返し、`exponential_midpoint`がstart half-step predictorとmidpointで凍結した正の
  relaxationを解析更新します。C03一定係数、極小/極大`h/tau`、可変係数の観測次数1.8以上を独立referenceで確認し、
  `state_at()`、材料first hit、wall残時間、RZ axis foldはRK4と同じproposal/event loopを通します。
  曲線/chord偏差は全短縮secantを含むvelocity enclosureから作り、材料反射とsurface departure後の同面再衝突を
  過度なposition-box分割なしで認証します。C03の全frameとStokes--Cunningham一定primitiveも閉形式で検査します。
  `dt/tau < 2.5`はRK4だけの安定性gateです。P11 closeout時点の指数法はfixed chargeだけを受理し、非零charge rateを
  明示拒否していました。
- P15は`oml_stationary_maxwellian_debye_huckel_v1`を最初にRK4へ、受入後にnative exponential midpointへ接続しました。
  RK4はexplicit `h L_Z <= 0.5`を維持し、exponential pathはmidpoint-frozen affine exponential chargeを使います。
  両methodはfinite charge invariant/rate/derivative bound、driftと`a/lambda_D`のapplicability、
  charge-aware electric/path enclosureを共有します。動的電荷ではexact pathと精度目的のhidden subdivisionを使いません。
  XY/RZ、wall、frame/probe、result、checkpointの既存state/schemaを再利用します。
- P15-Dは単一・単価正イオンと非正表面電位に限定した
  `oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1`を同じ両積分器へ追加しました。caseが
  run-wide drift比上限を明示し、zero-drift一致、独立moment、bound、compiled parity、XY/RZ、wall、resumeを検証します。
  正電位、負イオン、複数species、emissionへ自動切替しません。
- P18-Cは`aggregate_relative_drift_regularized_two_current_v1`を同じcontinuous-state passへ追加しました。
  電子と集約正イオンの密度、thermal voltage、正イオン速度・有効質量、screening長をfieldとして受け、正負電位branch、
  相対drift、`1 m/s`正則化、`0.01 V` ion-energy floor、有限指数範囲を明示したoptional reference revisionです。
  宣言した最大相対速度をactual stageとpath enclosureで検査し、P15/P15-Dへの自動切替やCOMSOL専用分岐は行いません。
- P18-Iは集約正イオン場向けの`relative_flow_screened_collection_orbital_aggregate_ion_v1`と
  `electric_field_directed_image_orbital_sensitivity_v1`を、既存Barnesとは別の排他的なion-drag revisionとして追加しました。
  前者は相対流方向、後者は正則化した電場方向を使い、P18-Cと同じstageの背景場・電荷を共有します。Case P/A名、
  producer固有scalar速度、自動blendは本体へ持ち込みません。
- P18-Dは`quasistatic_spherical_gradient_e2_v1`を追加しました。producerが形成した
  `grad(mean_E_squared)`をXY/RZの各stageでsampleし、`electrostatic_radius_m`、`mass_kg`、媒質比誘電率、実数CM factorから
  球形準静的DEPを計算します。point-dipole認証上限をprepareで検査し、Eの微分、DC/RF平均、gradient recovery、COMSOL比較は
  core外に保ちます。複素CM、travelling-wave、非球形、多極子、粒子相互作用へ自動拡張しません。
- P18-LはRZ/no-swirl専用の`rarefied_vorticity_sensitivity_rz_v1`を追加しました。
  `F=K (omega_phi e_phi) x (u_g-v)`、`K=C_L*pi*rho_g*lambda_g*a^2`、`a=drag_diameter_m/2`を
  既存の明示加速度passで評価します。gas velocity `[m/s]`、gas density `[kg/m^3]`、mean free path `[m]`と、
  producer所有のsigned azimuthal vorticity `[1/s]`を使い、core内で速度場を微分しません。`C_L`は有限正値を明示し、
  `lambda_g/a>=10`をfail-closedに要求します。速度依存boundは単一callbackへ統合され、exponential enclosure v3が
  start/half-predictorの速度boxで再評価します。B02 Brownianとの同時利用、Stokes--Cunningham、Cartesian/3-D、
  一般Saffmanへのfallbackは拒否します。
- P15-Eは`epstein_finite_speed_maxwell_mixed_equal_temperature_v1`を追加しました。鏡面と完全熱適応・
  等温拡散再放出の割合、最大分子速度比を明示し、低速級数と全速度閉形式を使い分けます。加速度boundと
  RK4速度Jacobian boundを分離し、両積分器の収束と独立3-D分子速度積分を検証しています。linear/Stokesへの
  自動切替、経験blend、新しいengine/state/schemaはありません。
- P15-Fは`barnes_collisionless_effective_speed_single_positive_ion_negative_debye_huckel_v1`を追加しました。
  単一正イオンのcollection＋orbital momentum transferを相対流方向の`explicit_acceleration`として評価し、
  continuous chargeとは同じplasma primitiveとstage `Z`を共有します。独立impact-parameter積分、global bound、
  両積分器収束、XY/RZ parityを検証し、collisional/image/E方向補正への自動切替は行いません。
- P16は`waldmann_gallis_free_molecular_single_species_heat_flux_v1`を追加しました。単一中性気体の局所並進伝導
  熱流束を`explicit_acceleration`へ変換し、Kn/relative-driftの全path適用域をfail-closedに検査します。独立
  Chapman--Enskog moment、global bound、両積分器収束、XY/RZ parityを検証し、gradient回復、Talbot/continuum、
  mixture/near-wall補正への自動切替は行いません。
- P18-Rは既存式ownerを再利用する`epstein_linear_effective_gas_sensitivity_v1`と
  `waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1`を追加しました。producer認証済みの
  one-effective-Maxwellian/pseudogas reference/sensitivityだけを対象とし、両revisionは`lambda/a>=10`と
  必須`0 < maximum_speed_ratio <= 1`をactual stageと連続pathでfail-closedに検査します。既存linear/P16の`0.1`は
  維持します。thermophoresisの`q_eff`はproducer-owned translational conductive heat fluxで、coreにgradient recovery、
  species配列、mixture rule、COMSOL branchを追加していません。
- Brownian B02はCartesian XY・fixed charge・Epstein linear drag-onlyの`ou_langevin`をproductionへ接続しました。
  凍結係数joint OU endpoint、固定depth conditional tree、cubic Hermite leaf pathを既存のterminal
  `stick`/`escape`、frame/probe、checkpoint/resumeへ通します。engine v30はroot/split/meanの実表現可能性を
  粒子単位で検査し、不良rowだけを`nonfinite_physics`として正常rowを継続します。連続OU first-passageの厳密解では
  ありません。
- P12の履歴runtimeでは`resources.threads`をCPU worker数の上限として解決し、各粒子を一つのworkerだけが所有しました。
  field/geometryはread-only共有、residual/event/statisticsはworker-local、最大worker数分のtileを一waveとして実行し、
  tile順にmain threadがstable mergeしてwriterを一つに保ちます。材料候補検索はcompiled BVH、一般曲線pieceは
  保守的なclear/split事前分類、同時刻wall prefixはcompiled batchを使います。1/2/4 threadで科学payload、RNG、
  event work countが一致します。case/result schemaと物理・proposal・event revisionは変更していません。これは
  P12/P14の履歴baselineであり、正の並列効果とthread数非依存memoryを満たす完成形ではありません。
  P14-Pでこの設定、worker-wave、対応harnessは削除済みで、現行schema/runtimeには残っていません。
- `geometry.py`はvolume incidenceとtopology-completeな`line2`を監査してpreorder + skipのstackless BVHを準備し、`events.py`が
  scale-aware budgetでfirst hitと残時間workを扱います。`boundaries.py`は静止壁の`stick`/`escape`/`hold`/
  `specular`/`restitution`/`probabilistic_stick`、priorityとcombined-normal corner応答を所有します。
  `specular`はparameterなしの完全鏡面反射、非単位の法線・接線係数は`restitution`だけが所有し、
  probabilistic lawの非付着側も`otherwise_law`で両者を明示的に選びます。
- `simulate`はtable/surface source、fixed charge、CPU固定stepを一つのengineで扱います。required fieldはprepare時に
  metadata、共通layout、全particle-domain supportを検査しますが、これだけで曲線経路の連続supportは
  証明しません。必須release/boundary eventとfinal、任意の明示時刻frameを
  sibling partial directoryからno-clobberで公開します。
  `open_result`は完成resultをlazyに読みます。
- `simulate`はload/prepare/runのphase peak、component内訳、slab幅を`run.json.memory_plan`に記録します。
  `resources.memory_limit_mb`はsolver-owned配列の予測peak上限であり、OS hard RSS capではありません。CLI
  `check`は公開`load_case`の範囲でcanonical numeric bytesと上限だけを報告し、完全なplanは完了runから
  読みます。最小slabが収まらないcaseは運動開始前に拒否します。ResultWriterは同期single-ownerとして
  segment/checkpoint/finalを一経路で
  永続化します。P13の即ack queueはcomputeとI/Oを重ねなかったため削除しました。bounded event/failure stagingとHDF5 raw
  chunk cacheのreserveはmemory planへ含めます。depth依存のdeferred event workは
  `event_work_bytes_per_particle`と`slab_event_work`へ独立計上し、proposal scratchへ隠しません。
  exact/curved候補、event/failure staging、surface release、direct replayはmemory plan v11のnamed componentへ
  分離し、pack時だけ生存する小さいgatherは12.5% safety marginが所有します。count passで固定capacity以下の
  stable row batchへ分け、総event数に比例するPython objectを保持しません。byte式のauthorityは
  [`docs/parallel_execution_plan.md`](docs/parallel_execution_plan.md)です。
- fixed charge、dragなし、厳密一様な電場・密度から粒子別一定加速度を証明できるXY caseは
  `quadratic_exact`を使います。topology-completeな材料boundaryと併用する場合はparabola-line first hitを
  求め、endpoint chordが見落とすturning pathも検出します。boundaryless caseは、required fieldが共通の
  fully-supported regular boxを持つ場合に限り、各proposalの始点・終点と座標成分ごとの放物線極値を
  解析的に検査して経路全体がsupport内であることを証明します。C04/C05はこの公開経路で実行できます。
- boundaryless Cartesian XY/RZのEpstein dragや非一様fieldは、global field/model boundから全短縮RK4の内部stageと
  endpointを覆う外向き位置・速度enclosureを作り、regular support box包含を証明します。applicabilityはglobal boundを
  先に試し、証明できないrowだけdense RK4 pathとlocal cell rangeでboundedに認証します。C02/C03は公開APIの
  解析時系列、frameなし・疎・密のoutput schedule不変性、step途中releaseのscenarioとして有効です。
  revision 3bでは同じsubsetをtopology-completeな材料boundaryと連成し、terminal `stick`/`escape`/`hold`まで扱います。
- 材料boundaryを持つtable初期位置をstrict interiorに限ります。surface releaseは境界上の位置を動かさず、
  内向きならdeparture、壁向きなら即時impactとします。RZの`r=0`は材料壁ではなくaxis seamであり、
  linear/quadratic pathとforce-coupled一般RK4は軸でmeridional座標をfoldして残時間を継続します。RZ vector fieldは
  `(r,z)/axisymmetric_rz`を要求します。geometryが軸へ接する場合、またはboundaryless fully-supported regular boxの
  `r_min=0`である場合をfieldsが一度だけaxis-accessibleと判定し、axis nodeのradial vector成分を0にします。
  standard gravityのRZ radial成分は座標基底の軸対称性条件なので、annular domainを含め常に0だけを許します。
- trajectory frameはboundary eventに対し右連続です。`hold`はhit位置、hit時速度、hit時電荷を保持した
  inactiveな`held`状態として以後のframe/finalにも残り、depositionやpaused particleとして扱いません。escaped後はframeから除外し、finalは
  `kinematics_valid=0`をlogical nullとします。有限なpayloadは非権威値であり、最後の有効状態はboundary eventが
  authorityです。

P06 revision 3aでは、canonical field extremaとmodel係数からRK4内部stageの速度・加速度をboundし、float64で
外向きに丸めたenclosureがfull macro stepと`StepProposal.state_at()`が作る全短縮区間の内部stage・accepted
endpointを覆うことを固定しました。Epsteinは`lambda/a`と`|u-v|/c_bar`も同じboundで連続的に検査します。
revision 3bでは、材料壁候補を持つ一般RK4区間をlocal始点から時間順に二分し、壁から分離したpieceだけを
support/applicability検査後にcommitします。最初のhitは無効なtrial tailより先に確定し、frameは確定済みpiece列から
再生します。非一様調和振動子の壁到達時刻・速度は4次収束し、frame有無でevent/final/refinement集計がbitwise同一です。
P19-L完了時点ではこのevent意味論を変更しませんでした。M3-C1 event v14で
`rk4_global_abs_enclosure_v2`をsupport、global-first applicability、hit時刻までの短縮RK4再積分のauthorityとして維持し、
`rk4_dense`のevent BVH queryだけをcurrent dense Bernstein位置・速度boundへ限定しました。event v15はこのauthorityを維持し、
geometry budgetとroundoff budgetを加算しました。現行event v16はvalidなRK4 dense rowについて、integrator所有の
root-relative position Bernstein control enclosureをfacet half-spaceへinterval射影し、全4制御点が既存budget込みで
厳密にinsideと証明できる候補だけをclearします。controlで証明できなければ候補を残してsplitします。dense boundが
不正ならglobal boundを保持し、required supportをglobalに証明できないcaseではevent queryもglobal boundへ戻します。
現行`rk4_position_hermite_state_extension_v3`はroot始点相対のBernstein enclosureをTwoDiff残差込みで作り、
下限は`-inf`、上限は`+inf`方向へ外向きにworld座標へ戻します。これは絶対座標で誤差幅が
肥大してevent certificateが閉じない原点依存の不具合だけを直します。accepted endpoint、Hermite/state path、
first-hit/event algorithmは不変で、v3導入時のengine v32、event v14、RK4 global enclosure v2も変更しませんでした。このdense pathと
local cell rangeはapplicabilityを認証します。実違反は既存
`field_support`/`model_applicability`、有限budgetで証明不能なら`indeterminate_applicability_certificate`です。一row最大64 cellの
overflowは共有するevent refinement budgetで分割します。memory plan v13はdense path 176 B/row、現設定のstack 72 B/row、
候補arena最大544 B/rowを計上します。
代表Case-A 287粒子1 stepでは、generic chord-deviation算術のbitwise同値なbatch化だけを残し、warmed medianを
`1.647475→1.304153 s`（`1.263x`）としました。これはP19-L完了時点のevent v12履歴で、query/split形状、final＋frame hashは
当時不変でした。M3-C1 event v14のevent-query精密化は後述する限定anchorで閉じています。詳細は
[`evidence/p19l/candidate_hot_path_v1.json`](evidence/p19l/candidate_hot_path_v1.json)です。
engine v7はこの数値意味を変えず、候補粒子を256件ずつのchunkで扱い、各粒子からleft-firstな次の
outstanding pieceを1件だけ取り出して、float64で完全に同じtarget timeごとにproposal行をbatch化します。
engine v7導入時のevent v4は`inspect_rk4_piece`とBVH candidate queryを行ごとに処理していました。
event `line_quadratic_rk4_first_hit_v5`は、integrator所有のcomponentwise chord-deviation boundとroundoff幅から、
eventsが単一facetの外向き横断、normal方向のtime bracket、tangent/time-shiftを含むposition radius、
facet endpoint clearanceを証明します。証明できないpieceはsplitし、最後は従来のfull-tube budgetへfallbackします。
proposal v3、enclosure v1、schema/APIは不変です。64粒子の同一machine baselineでは、material medianは
event v4の0.9666176 sから0.6450654 sへ短縮し（約1.50倍）、boundaryless medianは0.0471402 s、比は
21.77から13.6840へ低下しました。初期scalarの2.3706326 sからは約3.68倍です。accepted piece / candidate query /
refinement / 最大深さは1088 / 2496 / 1408 / 21です。engine v8は要求frameと重なるaccepted rowだけを保持し、
64/256/1024粒子のtraced-allocation checkpointを完了しました。この値はprocess RSSではありません。
小型の非gating性能基準は
[`tests/performance/README.md`](tests/performance/README.md)から実行できます。

engine v9のP07 exact-path sliceはC08～C10を公開経路で実行し、specular/probabilistic stick後の残時間、
cap-driven split、corner、ballistic RZ axis foldを扱います。manifestにはseed、RNG/source/law revision、draw kind、
resolved law、wall/residual/axis集計を残します。engine v10 / event
`line_quadratic_rk4_first_hit_v6`はCartesian XYの証明済み一定加速度surfaceを追加しました。`v·n`を優先し、
厳密tangentだけを`a·n`で分類します。内向きdeparture tokenは別wall hitまで保持し、eventsが各intervalで
source supporting line、内向き速度、内向き加速度を再証明した時だけsource facetを除外します。他facetは常に検索し、
macro partitionに依存しません。外向き加速度のzero-time impactはterminal応答を許しますが、反射で内向きdepartureを
作れなければ失敗します。engine v11 / event `line_quadratic_rk4_first_hit_v7`で、Cartesian XYの一般
`rk4_reintegrated`へ厳密内向きsurface departureと、single-facet activeな静止壁応答後の残時間継続を追加しました。接触facetを
候補から外すのは区間全体の速度包絡が厳密内向きを証明した時だけで、tangent、facet端点/corner、または曖昧な接触はfail-closedです。
interaction countは同じparticle residual stateで共有し、上限到達後も実際の次hitが確認された時だけ区間を分割します。
event時刻のframeは反射後stateを含め右連続です。対応fieldはfully-supported common `RegularLayout`、または材料domain
meshと完全一致するfully-supported P1/Q1です。engine v12 / event
`line_quadratic_rk4_axis_first_hit_v8`はRZ trialをsigned radial chartで積分し、canonical位置・速度でfield/physicsを
評価してradial加速度をsigned chartへ戻します。axis候補は既存enclosureとevent budgetでclear/split/hitを認証し、
同じRK4でhit prefixを再積分してfold後の残時間を継続します。双方が局在済みでwall/axisの不確かさが重なる時だけ
wallを優先し、axis端点とmaterial cornerの局在不能な完全tieはfail-closedにします。axis foldは
boundary event、wall RNG、interaction ordinalを消費しません。Epstein axis crossingの4次収束、frame identity、
away-axisのXY退化一致、axis→wall順序、軸上不変状態を公開scenarioで検証済みです。P06-Sは小さいphysics
runtimeと明示air revisionのStokes–Cunningham、P08はparticle-local failure、lifecycle series、明示state probe、
薄いCLIを同じengine/resultへ追加しました。case/result schema v1、proposal v3、enclosure v1は変更していません。
P09はengine v15/runtime layout v1/memory plan v1で完了しました。fresh/warmのload/prepare/runを分離して
RSSとscientific identityを測るharness、代表規模の実測、10k/100k/1M粒子のrelease/manual実行手順は
[`tests/performance/README.md`](tests/performance/README.md)にあります。P14は全規模のsynthetic性能軸を完了し、
P14-Pは直列runtimeへの収束まで完了し、その後P14-Uで主用途を結合したtarget-use判定も完了しました。
P10はengine v16/runtime layout v2/memory plan v2/physics runtime v2で完了し、実際に消費するP1/Q1 layoutの
per-particle hintだけをresident化しました。P14でinitial/miss full searchが支配的と確認されたため、field v3の
read-only BVHでsupported containment候補だけを絞ります。outside/masked最近傍full scanは意味論上残します。cold/warm JITを分離する手動harnessも同じperformance文書から
実行できます。P11はengine v17/compiled tile v2/physics runtime v3/proposal v4/event v9、
`exponential_midpoint_v1` / `exponential_midpoint_global_abs_enclosure_v1`として完了しました。
P12は`deterministic_particle_engine_v18` / `compiled_cpu_tile_v3` / `line_boundary_bvh_v3` /
`resident_soa_worker_microtile_v3` / `solver_owned_memory_plan_v3`として完了しました。
512粒子×4 macro stepの同一machine warm medianは1/2/4 threadで0.440460/0.528229/0.632361 sで、
P12着手時のserial 3.066786 sから当時の1 threadは6.96倍短縮しました。ただしこの小型caseのthread増加は負のscaleであり、
絶対時間のgateではありません。全規模のsynthetic mesh条件はP14で判定済みですが、surface＋force＋wallの
結合caseはP14-Uで判定済みです。
boundaryless unstructured、richer source distribution、moving wallは未解禁です。
P13は`deterministic_particle_engine_v19` / `durable_segmented_result_v3` /
`solver_owned_memory_plan_v4` / checkpoint schema 1として完了しました。固定64 macro-stepごと、または最終macro後に
segment、交互A/B checkpoint、`LATEST`の順でcommitします。`simulate`はstrictなresume identityが一致する
`OUT.partial`を自動再開し、`open_result(..., recovery=True)`は`LATEST`までの確定segmentだけをlazyに読みます。
最初の`LATEST`以前の初期再実行、segment/checkpoint/`LATEST`と最終公開の各境界、確率wall RNG ordinal、
orphan無視、破損拒否、通常実行との全公開payload identityを含むverification/scenario 322件が合格しました。
容量1 queueは同期ackによるbounded backpressureで、I/O overlap性能はP13の主張ではありません。power loss、remote
filesystem、同じOUTへの複数process同時実行も保証外です。

この固定64 cadenceは履歴です。現行`cumulative_solver_work_v1`は
`W=macro_step_count+accepted_particle_pieces+candidate_queries+refinements`、
`T=max(2^20,128N)`とし、engineがaccepted macro barrierで`W-W_epoch>=T`または最終macroを判定します。
resolved cadenceはmanifestとresume identityに含まれ、output scheduleとslab幅から独立です。`output.py`は
同期single-owner writerとしてsegment、inactive A/B checkpoint、`LATEST`のatomic persistenceだけを所有します。
現行result algorithmは`durable_segmented_result_v5`、result/checkpoint schemaは2です。

P14はengine v20 / compiled tile v4 / field v3 / geometry v4 / memory plan v6として完了しました。23行×3観測matrixの
identity/revision/memory checkとverification/scenario 336件が合格しています。geometry v4はtable-start volume containmentを
mixed-cell BVHで絞り、局所的にfloat64で解像不能なcellをprepareで拒否します。3観測medianでregular 100k/1Mは
20 workerで1.8796x/4.7965xでしたがevent-heavyでは負のscaleでした。配布可能なv0.1には
T03 analysis/visualizationは完了しました。P14-Rはbaseline v27/P14-U evidence保存とlocal Windows/Linuxの
wheel・runtime-only clean install・三API smokeまで合格し、初回remote workflow成功だけが残ります。

P14-Pは否定的closeoutまで完了しました。focused correction後のregular 1Mは1/2/4 threadで
9.32/10.09/10.10 s、4-thread speedup 0.923xで、v20の1-thread 7.534 sからも23.7%退行しました。
microkernelは約3.75xにscaleした一方、end-to-endはPython/NumPyのproposal/enclosure調停が支配し、Amdahl上も
限定修正では製品gateに届きません。このためcase schema v2から`resources.threads`、runtime thread mask、
parallel-only test/harnessを削除し、P14-P closeoutをv27のsingle-thread compiled engineへ一本化しました。
v27のregular 1M直列3観測は10.249/10.273/10.100 s（median 10.249 s）であり、この直列性能課題は
P14-Uで同一条件のprofileと代表用途評価により扱いました。

bounded slab、再利用workspace、flat SoA event wavefront、stackless boundary BVH、compiled boundary/Philox、
direct replay、bounded stagingは直列runtimeの有用な構造として維持します。独立caseのprocess並列はsolver外で行います。
詳細は[`docs/parallel_execution_plan.md`](docs/parallel_execution_plan.md)にあります。

P14-Uの外部harnessは、XY surface release＋Epstein＋非affine電場＋gravity＋材料targetの時間/2D mesh収束と
first-hit、可変RZ場のaxis crossing/parity、`none`/sample出力utility、event work、RSS/memory planを同じ公開API経路で
確認しました。正式releaseは10k/100k/1M粒子×2出力mode×3 fresh processと、timingから分離した1M `none` profileで、
`release_gate_complete=true`、18 raw観測/6 median、failure 0、出力utilityと科学payload/revision identityの一致を
満たしました。受理済みreportは[`evidence/v0.1/p14u_release_v1.json`](evidence/v0.1/p14u_release_v1.json)です。これは
machine-local evidenceで、他のrelease evidenceも[`evidence/v0.1/`](evidence/v0.1/)に保存します。製品runtimeは
一つのsingle-thread compiled engineのままです。P14-Rの初回remote workflowは未完の独立release trackとして残ります。
P15は旧着手blockerをユーザーの明示指示で解除して両explicit sliceを完了しました。続く外部M3-V評価、
canonical RZ/P1 reduced electrostatic builder F01、F02 provider adapter/integrationも完了しました。F02は代表meshの
linear solve、32粒子fixed-electric smoke、同一export node上の記述的field比較までをcore外で閉じています。
再実行条件と限界は[`tools/comsol_adapter/README.md`](tools/comsol_adapter/README.md)、field builder契約は
[`tools/electrostatic_builder/README.md`](tools/electrostatic_builder/README.md)、小さいhash付き証跡は
[`evidence/f02/`](evidence/f02/)にあります。species制約付きrelative-drift chargeはP15-D、finite-speed Epsteinは
P15-E、collisionless Barnes ion dragはP15-F、Waldmann--Gallis thermophoresisはP16、aggregate ion-drag sensitivityは
P18-I、quasistatic spherical DEPはP18-D、RZ rarefied-vorticity lift sensitivityはP18-L、effective-gas drag / thermophoresis
sensitivityはP18-Rで完了しました。
P18-Iの保存式再生、production式との差、性能観測は
[`evidence/p18i/`](evidence/p18i/)、P18-Dのmachine-localなstage性能・memory観測は
[`evidence/p18d/`](evidence/p18d/)へ分離しています。
P18-Lの100,000-row warm direct stageはdisabled `0.023354 s`、enabled `0.0255142 s`、比`1.0925`、
prepared bound増分`900016 B`でした。これはmachine-localな非gating観測です。P18-Rの保存artifact auditはnative
linear Epstein replayを約`1.1e-15`でPASSとした一方、既存P15-E/P16 applicabilityを12/12 `NOT_APPLICABLE`、
PPR `q_eff`欠損によるthermophoresis replayを`NOT_TESTED`としました。保存frameは連続pathを認証せず、P18-R成果物自体は
COMSOL studyを再実行していません。新revisionはreference/sensitivity runを設定可能にするだけで物理的真値やCOMSOL
軌道一致を確立しません。

後続の外部M3-C0bではCase-A 100 nmをBrownian-off、dynamic chargeと全決定論寄与で再実行しました。30 msの
10/5/2.5 us系列は非漸近で`CHARACTERIZED`に留めました。v5の旧「全gate PASS」は位置relative L2を絶対RZ座標で
正規化して原点依存だったため`INVALIDATED`であり、v5は履歴上の`CHARACTERIZED`です。逐次確認v6は全粒子が
activeな0--450 usで0.625/0.3125/0.15625 usを評価し、各runの13,202 recordがすべてactiveでした。fine pairの
原点不変な変位・速度・電荷relative L2は
`3.102727085428027e-5/3.92483251084038e-5/1.3511393490811483e-6`、観測次数は
`0.9041136/0.944312/1.123838`です。原本MPHと本体coreは不変です。このPASSはpre-eventの運用上の固定step選択だけで、
solver一致、普遍的なaccuracy、30 ms/event収束ではありません。frozen saved-state producer-form replayは8/8完了し、
P19-Lでglobal applicabilityによる時刻0の過保守blockerを解消しました。その後のCase-A 100 nm pre-event
exported-P1/native-field比較は、candidate 3 runを287粒子×46 frame、event/failureなしで完了し、referenceのpre-event運用上の自己収束はPASS、
cross-representationの位置・速度・電荷RMS/max 6 gateはすべてFAILでした。当時のevent v13では全candidate runが同じ
accepted-piece countへ細分されたため、macro-step半減時のfloat64-floor内安定性は有効なprecision-stability履歴ですが、
独立時間収束またはRK4次数を示しません。続くfull-physics exact-connectivity common-P1 COMSOL診断の現行eval_v3は、同じ
canonical場、動的電荷、全決定論力を使う287粒子×46 frameについて、位置・速度・電荷のRMS/max/relative L2の事前登録
9 gateをすべてPASSしました。位置は`4.06807283316903e-13 / 1.2035778717837921e-12 m /
2.0316001855367802e-10`、速度は`2.10847522277453e-9 / 3.844306466969233e-9 m/s /
1.9630813983926987e-10`、電荷は`1.1818881019499895e-7 / 2.3758877887303242e-7 e /
4.670220207738041e-10`です。これはCase-A 100 nm、Brownian-off、0--450 us、event前のsame-field agreementだけを閉じます。
native-field等価性、物理妥当性、Brownian、30 ms、他case・粒径・variantは未認定です。
その後の外部M3-C0 boundary semantics probeは、力なしの解析的normal impactを10/5/2.5 usで実行しました。現行v2は
2 scenario×3刻みのexact 6 configuration receiptをCOMSOL process logから照合し、欠落・重複・形式不正・設定差を
fail-closedで拒否します。全active frameを`x=x0+v0*t` / `v=v0`で検証した最大位置/速度誤差は
`4.726604209672303e-16 m` / `1.7763568394002505e-15 m/s`で、各`1e-12`上限を満たしました。boundary 37
Freezeのstatus 2・hit点R-Z保持・衝突前velocity保持と、boundary 35 Disappearのstatus 4・event後位置/速度NaNを分離しました。
両caseともeventは73 us、最初のterminal frameは75 usで、step間event-time spreadは
`3.07371315899641e-17/2.71050543121376e-20 s`、56 PASS / 0 FAIL / velocity記述6件でした。原本hashとcoreは不変です。
これは隔離したCOMSOL境界意味だけの証拠で、production parity、grazing/corner、full physicsを認定しません。v1は科学的に
無効だったのではなく、configuration receiptとactive-flight oracleの監査強度が不足した履歴成果物としてv2に置換されています。

common-P1の最初の自然なwafer stickとfield-representation差局在化は完了しました。material evaluation v5は20/20と
pre-event prefix 9/9をPASSし、event時刻、hit位置、terminal chargeの絶対差は
`2.157542807607049e-13 s / 2.683964162031316e-14 m / 4.7283812421028415e-09 e`です。
v13のquery/refinement/accepted/depthは`16,427,517 / 7,792,306 / 8,635,211 / 16`で、450 usまでに
`7,623,460 / 7,792,306` refinement（`97.8331703092769%`）が既に発生していました。v14は
`842,927 / 11 / 842,916 / 11`、failure 0です。operator-observed shell wall-timeは約`14m13s`から`36.5s`
（約`23.4x`）へ短縮しましたが、machine-local、概算、非gatingでsolver-reported runtimeではありません。

v14 solver-only 0.625/0.3125/0.15625 us再実行はrefinement 0で、位置・速度・電荷のRMS観測次数は
`2.029875353701904 / 2.0816971911764033 / 2.044084026475049`、fine-pair relative L2は
`6.099791486063973e-8 / 8.321356016032579e-8 / 1.3796067752988052e-8`、全3量`ORDER_EVALUATED`で自己収束PASSです。
この約2次はpiecewise P1場とmesh crossingを含む本caseの経験値で、RK4の形式4次を証明も否定もしません。v14はsolver側の
event BVH broad phaseだけを変え、hash-lock済みのcommon-P1 COMSOL inputs/reference/source MPHは不変なので、COMSOLを再実行せず
既存referenceへcandidateを再比較しました。

P18-Hのgeneric terminal `hold/held`はengine v32 / boundary v5 / result v4として完了しました。解析的直線・曲線hit、
右連続frame、resume、slab identity、OU/Brownian、zero-time surface、analysis非deposition分類を公開経路で確認し、
既存hash固定Freeze referenceへの外部candidateもCOMSOL再実行なしで15/15 PASSです。compact authorityは
[`evidence/p18h/hold_freeze_v1/`](evidence/p18h/hold_freeze_v1/)です。この結果をfull physicsやgrazing/cornerへ一般化しません。

B03は第二engineを作らず、既存`ou_langevin`へRZ meridional projected revisionを追加して完了しました。
noise-free midpoint predictorで`gamma,u,T,a,G,J=dG/dZ`を1回評価し、`J<=0`を確認して
`u_eff=u+a/gamma`のjoint exact OUと`macro_root_affine_exponential_v2`のroot内chargeを合成します。
同一rootのconditional treeは凍結係数とendpointを共有し、
axis hit後は元remainderを折り返さず、accepted prefix commit＋fold後の残時間を新しいroot ordinalで再開します。
native/effective-gas線形Epstein、fixed/continuous charge、既存additive force、terminal `stick`/`escape`/`hold`が対象で、
B02はbitwise不変です。これはRZ meridional projected 2-DOFであり、等方3-D Brownianでも一般SDEのstrong/weak 2次でもありません。
現行revisionはengine v36 / proposal v10 / catalog v17 / event v16 / runtime v19 / compiled tile v18 / memory plan v13です。
B03の正式な公開API characterizationは24/24実行を全粒子active・failure 0で完了しました。20,000粒子の
machine-local medianはB02 fixed `5.7722 s`、B03 fixed `11.7142 s`、continuous charge＋gravity `11.8427 s`、
axis restart `16.3002 s`で、最大process peak RSSは`226,316,288 B`です。静的path arrayは`648 B/row`で
`2048 B/row`上限内にあり、値はportable gateではありません。詳細は[`evidence/b03/`](evidence/b03/README.md)を参照してください。
これらの計時・RSSは初回B03 closeout（engine v34 / proposal v9 / runtime v17 / tile v16）の履歴証拠です。
performance characterizationで反復加算由来の終端tailを検出したため、engine v34はmacro timeを補償積和による
`start + n*dt`のindexed gridから構築し、float64構築roundoff内だけendへsnapします。科学的に意味のある残時間は消しません。

charge-stable continuous couplingとwork-scaled durable cadenceは完了しました。RK4にはexplicit
`hL_Z<=0.5`を残し、deterministic exponential midpointとB03は同じmidpoint-frozen affine exponential charge rootを
使います。clip、charge-only subcycle、第二engineはなく、精度は別runの`h,h/2,h/4`で選びます。
直近の全品質gateはPASSです。P20 performance closeoutに続き、意味を揃えた外部V&V/M3-C2A Case-A/Case-P
100 nm anchorも完了しました。
Case-A 100 nm common-P1 companionを、COMSOL 32 seedとcandidate 32 seed、各287粒子、30 ms、121時刻で実行しました。
独立pilotで選んだCOMSOL 20 usとcandidate 20 us / Brownian tree depth 3を固定し、確認的な終端人口曲線差は最大0.5989%、
95%同時上限3.877%で、事前登録5%幅を`PASS`しました。全287発生源のseed平均R-Z軌道overlayを含むauthorityは
[`evidence/m3c2/caseA_100nm_final_campaign_v1/`](evidence/m3c2/caseA_100nm_final_campaign_v1/README.md)です。
このCase-A PASSはpathwise RNG一致、native field、Case-P、第2 ion-drag、10/30 nm、一般的COMSOL同等性を意味しません。
[`evidence/p20_efficiency/`](evidence/p20_efficiency/README.md)はmanual・machine-local・non-gatingな運用効率証拠で、
portable timingまたは異なるstep間のequal-accuracyを主張しません。

続くCase-P 100 nm finalは、20 us、COMSOL/candidate各32独立seed、各287粒子、30 ms、121 frameで完了しました。
登録済み83区分R-Z/fate gateは最大empirical TV `0.010670731707317093`、同時上限
`0.13119968456545308 < 0.15`で`PASS`しました。終端gateも`PASS`ですがevent 0のため境界parityには情報を持ちません。
これは元Case-P COMSOL `auxq`が意図する電子＋正イオン二電流とのsame-form比較で、後続のaggregate three-currentや
species-resolved物理を認定しません。pathwise RNG一致、普遍的COMSOL同等性、eventful boundary parityも主張しません。
authorityは[`evidence/m3c2/caseP_100nm_final_campaign_v1/`](evidence/m3c2/caseP_100nm_final_campaign_v1/README.md)です。
optional aggregate three-currentのproduction実装はP21 priority 1で完了し、priority 2のcritical boundary microcaseも`PASS`です。
外部Case-P派生companionのpriority 3入力監査はcanonical負イオンprimitive authority不足で`BLOCKED / NOT_EVALUATED`として閉じ、
軌道は実行していません。物理modelの`NOT_APPLICABLE`ではなく、元Case-P二電流anchorは不変です。この任意物理の外部coverageは
P21の出口から分離し、P21と明示scopeの2D benchmarkは`CLOSED_ACCEPTED_WITH_LIMITATIONS`、
`2D_CRITICAL_VV_COMPLETE`です。三電流のCOMSOL軌道同等性は非認定です。authorityは
[`evidence/m3c3/caseP_three_current_companion_v1/`](evidence/m3c3/caseP_three_current_companion_v1/README.md)です。

final計時は`NON_AUTHORITATIVE_EXTERNAL_WORKLOAD_OVERLAP`です。accepted candidate seed `319032`、`319047`、`319063`の
287粒子owner discoveryは、受理済み科学payload・work・case identity・revisionを完全一致させて完了しました。
支配ownerは3 seedとも`integrators`（自己時間比42.58--42.86%）でしたが、事前登録済みbounded ownerではないため
`optimization_authorized=false`でproduction変更はありません。Case-A/Case-P 100 nm anchor benchmarkは
`CLOSED_ACCEPTED_WITH_LIMITATIONS`です。10,000粒子以上の性能、残るpackage、process並列は完了条件ではなく、
利用SLAを先に定義した独立work packageです。owner discoveryのauthorityは
[`evidence/m3c2/caseP_100nm_owner_profile_v1/`](evidence/m3c2/caseP_100nm_owner_profile_v1/README.md)です。COMSOL fitting、保存COMSOLの刻み模倣、
threadingの再導入は行いません。B03 core受入にCOMSOL再実行は不要でした。旧event v13の約`3e8`
accepted piece/run見積りはartificial subdivisionに基づく履歴で、現行v16の計画根拠ではありません。
後続のbounded chord follow-upでは、同じaccepted seed 3件の科学payload/work/revisionを完全一致させたまま、
重複chord式とPython scalar loopを一つのserial compiled batchへ統合し、end-to-end中央値を12.29%短縮しました。
ここで停止しており、10,000粒子scaleまたはCOMSOL速度比は認定していません。authorityは
[`evidence/m3c2/caseP_100nm_chord_optimization_v1/`](evidence/m3c2/caseP_100nm_chord_optimization_v1/README.md)です。
M3-C1 compact判定は
[`evidence/m3c1/case_a_100nm_pre_event_v6/`](evidence/m3c1/case_a_100nm_pre_event_v6/)と
[`evidence/m3c1/case_a_100nm_common_p1_v1/`](evidence/m3c1/case_a_100nm_common_p1_v1/)、
[`evidence/m3c1/case_a_100nm_material_event_v1/`](evidence/m3c1/case_a_100nm_material_event_v1/)です。
boundary semanticsのcompact判定は
[`evidence/m3c0/boundary_semantics_v2/`](evidence/m3c0/boundary_semantics_v2/)です。
currentな小型証跡は[`evidence/m3c0/deterministic_pilot_v6/`](evidence/m3c0/deterministic_pilot_v6/)で、v5は変更しません。
Brownian production縦切りB02まで完了しました。solver本体を変更せず、`model_dataset`の理論・設定を合わせた
COMSOL deterministic matched sliceの外部M3-V評価も、Case-A 100 nm、固定電荷、共通P1場・共通3力、
材料event前の287粒子×41時刻について完了しました。10/5/2.5 usの自己収束から事前登録した幅に対し、
position RMS `8.643e-16 m`、velocity RMS `1.999e-14 m/s`でPASSです。この旧runnerは固定電荷・共通3力のreduced sliceで、
現在必要な動的電荷・ion drag・thermophoresis・DEP・liftを含むcommon-field診断を代替しません。後続のfull-physics
common-P1診断で限定same-field agreementはPASSしました。COMSOL側の隔離boundary意味は後続probeで確認しましたが、
native-field空間収束、production boundary parity、物理適用性、
Brownian ensembleは未認定です。
詳細は[`evidence/m3v/matched_caseA_100nm_deterministic_v1.md`](evidence/m3v/matched_caseA_100nm_deterministic_v1.md)にあります。
P17のstate-dimension変更は別workstreamとして進めます。

T03の派生結果はsource result外へ出力します。

```console
uv run --locked python -m tools.analysis RESULT --output DERIVED/summary.json
uv run --locked python -m tools.visualization RESULT --output-directory DERIVED/visualization
```

analysisはtrajectory保存なしでも使えます。visualizationは保存済みprobeを優先し、なければframeを読むため、caseの
trajectory scheduleで少なくとも一つの対象時刻を保存してください。軌道を後処理側で再計算はしません。

## 開発環境

```console
uv sync --locked
uv lock --check
uv run --locked ruff format --check src tests tools
uv run --locked ruff check src tests tools
uv run --locked pyrefly check --summarize-errors
uv run --locked lint-imports
uv run --locked python scripts/check_complexity.py
uv run --locked pytest tests/verification tests/scenarios -q
```

環境、dependency、command実行にはuvだけを使います。runtimeはNumba `>=0.67,<0.68`と
NumPy `>=2.0,<2.6`をlockします。通常変更でlockを更新せず、dependencyを
意図して変更する場合だけ`uv lock`を実行します。
