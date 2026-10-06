# Particle Platform Redesign

半導体製造チャンバー向け粒子軌道計算基盤を、既存実装から独立して再設計した成果物です。
既存ソルバーは実装基盤として継承せず、失敗事例と式・要件の候補を抽出するための資料としてのみ扱っています。
`../model_dataset/` は、参照モデルの理解と外部V&Vに使う資料です。製品仕様そのものや
solver coreの依存先ではありません。旧ソルバーと旧環境は内容を実装へ流用せず、`../old_code/`へ
参照用archiveとして移動しています。

## 成果物

- [`AGENTS.md`](AGENTS.md) — 今後のcoding agentが守るclean-room境界、数値不変条件、責務、
  品質gate、test、旧処理削除の実装規約。
- [`quality_tooling_plan.md`](quality_tooling_plan.md) — uv、Ruff、import-linter、Pyrefly、Radonの
  単一責務、最小設定、command、CI適用方針。
- [`product_specification.md`](product_specification.md) — 製品目的、物理・帯電・発生源・境界理論、
  数値基盤、最小API、入力・出力、高速化、複雑化防止、段階的実装を固定する**主仕様書**。
- [`solver/docs/support_and_errors.md`](solver/docs/support_and_errors.md) — 静的2-D v0.1の対応範囲、
  明示的な非対応組合せ、公開例外、粒子単位failure、schema互換方針をまとめた利用者向け入口。
- [`architecture_proposal.md`](architecture_proposal.md) — Canonical DataBundle、実行計画、backend、
  詳細な責務境界を扱うarchitecture authority。文書間の優先順位は一律ではなく、
  [`AGENTS.md`](AGENTS.md) の責務別authority表に従います。
- [`architecture_review.md`](architecture_review.md) — principal architect、数値計算、HPC/I/O観点の
  レビュー結果と、module ownership、StepProposal、並列event、checkpointを含む反映済み判断。
- [`implementation_plan.md`](implementation_plan.md) — 新しい独立packageの配置、fileごとの責務、
  canonical schema、runtime状態、実装backlog、stage出口条件を実装者向けに具体化した計画書。
- [`solver/docs/stage1a_validation.md`](solver/docs/stage1a_validation.md) — P06-S/P08の数値受入、failure境界と
  P09 runtime/memory受入、品質gate、次段階への制約を固定したcloseout。
- [`solver/docs/stage1b_validation.md`](solver/docs/stage1b_validation.md) — P10 compiled CPU、P11
  exponential midpoint、P12 event-heavy parallel、P13 durable result、P14 synthetic baselineのrevision、数値受入、
  cold/warm計測境界、P14-Pの直列runtime closeout、P14-U代表用途gateの受入方法と完了証拠。
- [`solver/docs/parallel_execution_plan.md`](solver/docs/parallel_execution_plan.md) — P14-Pの実測、Amdahl判断、
  multithreading不採用、single-thread compiled runtimeへの収束と再検討条件。
- [`data_quality_assessment.md`](data_quality_assessment.md) — `model_dataset` の構造、整合性、
  比較基準として利用できる範囲と利用禁止事項の監査結果。
- [`vv_methodology.md`](vv_methodology.md) — 時間履歴を主対象とするCOMSOL比較、収束試験、
  イベント照合、確率モデル評価、原因切り分けのGate方式V&V手順。
- [`technical_research.md`](technical_research.md) — 力学、帯電、ブラウン運動、壁面、メッシュ、
  軸対称系、内部静電場生成、CPU/GPU高速化に関する技術調査。
- [`model_dataset_rebuild_plan.md`](model_dataset_rebuild_plan.md) — 現在の参照データの制約と、
  COMSOL外部V&V用microcase・再抽出schema・再実行順序。
- [`report_app/dist/index.html`](report_app/dist/index.html) — 主要な監査結果と設計判断を可視化した
  ローカル分析レポート。ソースは `report_app/src/`、集計値は `evidence/` にあります。
- [`analysis/`](analysis/) — 監査・比較を再実行するスクリプト。
- [`evidence/`](evidence/) — スクリプトから再生成できるJSON/CSV証拠スナップショット。

## 再現方法

現在の`evidence/`は設計監査時のreview済みsnapshotです。`analysis/`は外部証拠の監査方法を
保存するsourceであり、solver runtimeの実行入口ではありません。旧環境やplain `python`で
再生成せず、必要なdependencyと再生成commandを外部toolのwork packageで固定してから実行します。

以下のmilestone記録は変更理由を残す履歴であり、現行revisionや未完了項目のauthorityではありません。
現行状態は[`implementation_plan.md`](implementation_plan.md)のbacklog表、数値・V&Vの判断は
[`vv_methodology.md`](vv_methodology.md)、実行時の厳密なrevisionは各result manifestを参照してください。

P00～P05が完了し、`particle_platform_redesign/solver/`は独立uv projectとして作成済みです。P03では
large-offset・高aspect要素で再現したpoint-location defectを`field_location_v2`で置換し、物理空間の
包含判定、conditioning gate、最近傍supported provisional、非有限failureを一経路へ統一しました。現在は
locked環境、品質gate、三公開APIのimport境界に加え、producer-neutralなYAML schema v2/HDF5 schema v1、論理hash、
厳格な`load_case`、C01～C10の独立analytic microcase、XY/RZ coordinate規則、static regular/P1/Q1
single-point field samplingを提供します。P04では複数table source、release原点からの厳密ballistic、
必須release/final、明示時刻frame、atomic no-clobber result、最小lazy `ResultView`を一つのproduction経路へ
追加しました。P05では大域topology監査、line BVH、ballistic exact first hit、parameterなしの
stick/escape、boundary event、terminal lifecycleを同じengineへ追加しました。P06 revision 1では
required-fieldの全domain certificate、fixed charge、Epstein linear drag、Coulomb電気力、重力・浮力を
実RK stageで連成する共通`StepProposal`とkernel/physics verificationを同じengineへ追加しました。その後の
安全性reviewで、有限個のRK stage点だけでは連続軌道のfield supportを証明できず、frameの有無が成功可否を
変え得ることを確認しました。前版`coupled_rk4_engine_v3`では一般`rk4_reintegrated` runを一時的に閉じ、
revision 3aを着手前hardeningした`coupled_rk4_engine_v5`で、boundaryless Cartesian XY、fixed charge、全cell supportedな
`RegularLayout`に限って全短縮RK4評価を含む連続support/applicability enclosureを追加しました。

P06 revision 2でproduction利用できるforce-coupled範囲は、fixed charge・dragなし・厳密一様fieldから証明した
一定加速度の`quadratic_exact` pathです。topology-completeな材料boundaryがあるcaseはfirst hitで流体domainからの
退出を捕捉し、boundaryなしcaseは全cell supportedな`RegularLayout`に限定して各proposalの座標極値を解析的に
support boxと照合します。放物線first-hit、hit時刻状態、event優先のvalidity順序を含み、C04/C05は公開API
scenarioとして有効です。C02/C03もrevision 3aの証明範囲で公開API scenarioとなりました。一般RK4と
材料boundaryの連成はrevision 3bの`coupled_rk4_engine_v6`で完了し、各local始点からboundを再構築する
sequential accepted RK4 pieces、保守的な離散path tube、event-before-validityを追加しました。同一macro proposalの
parameter区間は流用しません。`coupled_rk4_engine_v7`は数値意味を変えず、各粒子のleft-firstな
outstanding pieceを1件ずつ取り出し、完全に同じtarget timeごとにproposal行をまとめる256粒子chunkの
deterministic wavefront batchに置換しました。event v5は、integrator所有のcomponentwise chord deviationと
roundoff幅を使い、events所有の単一facet外向き横断、normal time bracket、tangent/time-shiftを含む
position radius、endpoint clearanceを証明します。失敗時はsplit/full-tube fallbackとし、proposal v3、
enclosure v1、event v5、schema/APIは変えていません。64粒子の同一machine baselineはmaterial median 0.6450654 s、
boundaryless median 0.0471402 s、比13.6840です。直前event v4の0.9666176 sから約1.50倍、初期scalarの
2.3706326 sから約3.68倍で、accepted piece / candidate query / refinement / 最大深さは
1088 / 2496 / 1408 / 21です。engine v8は要求frameと重なるaccepted rowだけを保持し、
traced-allocation checkpointを完了しました。また、材料domainと完全一致するfully-supported P1/Q1を一般RK4へ
追加しました。engine v9は、単一のtable/surface schedule、Philox counter RNG、固定時刻surface、
静止壁のstick/escape/specular/probabilistic stick、corner、exact-path複数hit、ballistic RZ axis foldを追加し、
C08～C10を公開経路で実行します。engine v10 / event v6はCartesian XYの証明済み一定加速度surfaceへ
velocity優先のdeparture/impact判定とsource-facet start-contact certificateを追加しました。
`coupled_rk4_engine_v11` / event v7でCartesian XY一般RK4へ、厳密内向きsurface departure、single-facet activeな
壁応答後の残時間継続、右連続なstate jump、次hitを確認してから行うinteraction-cap splitを拡張しました。
tangentまたはfacet端点/cornerからの一般RK4 departureはfail-closedです。
現行の正確なmilestoneとalgorithm revisionは
[`implementation_plan.md`](implementation_plan.md)と各実行manifestを参照してください。現行solverは、engine v12で
同じproduction loopへ追加したforce-coupled RZを維持します。signed radial chartでRK4 stageを滑らかに進め、canonical RZ field/physicsへ
基底変換し、axisをwallではないfirst eventとして局在してfold後の残時間を継続します。RZ vector metadataと、
geometryまたはboundaryless regular supportから一度だけ決めるaxis accessibility、canonical support像をprepare/
acceptanceで検査します。standard gravityのRZ radial成分はaxis accessibilityに関係なく0だけを許し、
axis-accessible fieldのradial vector成分も軸上0を要求します。Epstein解析解の4次収束、frame不変性、
away-axis XY退化一致、axis/wall順序で検証済みです。このRZ追加時点ではcase/result schemaを変更していません。
現行はYAML case schema v2、canonical HDF5 data schema v1、result/checkpoint schema v2です。
P06-Sでは明示air revisionのStokes–CunninghamをXY/RZへ追加し、P08ではparticle-local failure、
lifecycle series、明示state probe、三公開APIだけを使う薄いCLIを追加してStage 1Aをcloseしました。
P09はYAML/resourceの先行parseとHDF5 metadata memory gate、sourceの直接schedule scatter、固定ID対応の
resident stateとresident-row active index、bounded microtile scratch、load/prepare/runのsolver-owned memory planを
実装しました。memory limitはOS hard RSS capではなく、fresh/warm RSSは外部performance scriptが別に測ります。
このP09履歴はengine v15/runtime layout v1/memory plan v1として維持します。P10は同じengineのfield sampling、
sample済みprimitiveのphysics、classical RK4算術をNumba compiled array passへ置換し、runtime layout v2 / memory
plan v2として、実際に消費するP1/Q1 previous-cell hintだけをresident化しました。regular lookupはsupported
containing-cell common pathだけO(1)候補で、outside/masked provisionalはcompiled全cell走査です。P1/Q1は
P14でinitial/missの全cell走査が支配的と確認されたため、strict-interior hintの次にfield-owned cell BVHで
supported containment候補を絞る`field_location_v3`へ置換しました。最小supported ownerとexact predicateは維持し、
outside/masked physical-nearest provisionalだけは意図的にO(cell数) full scanのままです。
accepted endpointだけがhintをcommitし、`state_at()`、wall、residual、outputは同じproduction経路です。
scalar referenceはverification oracleでありproduction fallbackではありません。Numbaは`fastmath=False`、runtime
dependencyは0.67系、NumPyは`<2.6`です。P10時点のarray passはP14-Pでpreallocated `*_into` passへ整理し、
最終的に`fastmath=False, parallel=False`のsingle-thread compiled runtimeへ収束しました。P10ではproposal、event、
field semanticsのrevisionを変更していません。

P11は同じengineと`StepProposal`へ`exponential_midpoint_v1`を追加しました。physics runtimeは各stageで
線形drag rate、target velocity、加算加速度を一度だけ組み立て、指数法はstart half-step predictorで得た
midpoint係数を用いて線形dragを解析更新します。C03一定係数の丸め誤差精度、極小から極大までの
`h/tau`の有限性、可変係数での観測次数1.8以上を独立oracleで検証し、材料first hit、残時間、RZ axis、
`state_at()`、output scheduleもRK4と同じevent経路を使います。
全短縮secantを含むvelocity enclosureからmethod-neutralなchord偏差を作り、surface departure後の同面再衝突と
Stokes--Cunningham一定primitiveの閉形式も公開scenarioで検証します。RK4の`dt/tau < 2.5`安全gateは
`rk4_fixed`だけに適用し、指数法へ流用しません。P11 closeout時点ではcontinuous chargeを指数法で拒否していました。
P15は`oml_stationary_maxwellian_debye_huckel_v1`をRK4-firstで受け入れ、その後native exponential midpointへ接続しました。
現行RK4は`h L_Z <= 0.5`を維持し、exponential pathはmidpoint-frozen affine exponential updateを使います。
両methodはfinite invariant/rate/derivative bound、charge-aware electric/path enclosureを共有し、
動的電荷のexact pathと精度目的のhidden subdivisionを禁止します。現行proposalはv10、RK4 enclosureはv2、
charge-stable exponential midpointはv3、同enclosureはv3です。
P15-Dは単一・単価正イオン、非正表面電位に限定した
`oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1`を同じ経路へ追加しました。zero-driftではstationary
OMLへ一致し、明示drift envelope外、正電位、負イオン・複数speciesはfallbackせず拒否します。
P18-Cは電子と集約正イオンの保存primitiveを使う
`aggregate_relative_drift_regularized_two_current_v1`をoptional continuous-charge revisionとして追加しました。
正負電位branch、相対drift、正則化、ion-energy floor、有限指数範囲を明示し、同じRK4／explicit midpoint stageで
運動と連成します。P15/P15-Dの置換、外部producer名による分岐、外部datasetへの定数fitは行いません。
P15-EはMaxwell鏡面/等温拡散混合を明示する`epstein_finite_speed_maxwell_mixed_equal_temperature_v1`を追加し、
有限速度rateと速度Jacobianを別々にboundして両積分器へ接続しました。低速modelやStokesへ自動切替しません。
P15-Fは`barnes_collisionless_effective_speed_single_positive_ion_negative_debye_huckel_v1`を追加し、単一正イオンの
collection＋orbital ion dragを相対流方向の明示加速度として同じstage passへ接続しました。continuous chargeとは
plasma primitiveと実stage電荷を共有し、collisionless適用域外でfloor、image補正、別modelへ切り替えません。
P16は`waldmann_gallis_free_molecular_single_species_heat_flux_v1`を追加し、単一中性気体の局所並進伝導熱流束から
自由分子熱泳動を同じstage passで評価します。Kn/relative-drift適用域、独立運動論moment、global bound、両積分器の
時間収束とXY/RZ parityを検証し、coreでの温度gradient回復やTalbot/continuumへの自動切替は行いません。
P18-Rは既存式ownerを再利用するoptionalな
`epstein_linear_effective_gas_sensitivity_v1`と
`waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1`を追加しました。producerが混合気体を
一つの有効Maxwellian/pseudogasへ畳み込んだreference/sensitivity用途だけを対象とし、両revisionは
`lambda/a>=10`とcase明示の`0 < maximum_speed_ratio <= 1`を全stage・連続pathでfail-closedに検査します。
既存`epstein_linear_v1`と単一気体P16の上限`0.1`は変更しません。熱泳動の`q_eff`はproducer所有の
有効並進伝導熱流束であり、coreは温度gradientやspecies配列を回復せず、species-resolved mixture truthや
COMSOL専用branchとも呼びません。
P18-Dはproducerが形成した`grad(mean_E_squared)`を入力とする
`quasistatic_spherical_gradient_e2_v1`を追加しました。球形準静的dipoleの明示加速度を同じstage passで評価し、
point-dipole半径上限をprepareで検査します。DC/RF平均とgradient recoveryはproducer、COMSOL比較は外部V&Vが所有し、
trajectory coreはEの数値微分やproducer別分岐を持ちません。
P18-LはRZ/no-swirl専用の`rarefied_vorticity_sensitivity_rz_v1`を追加しました。
`F=K (omega_phi e_phi) x (u_g-v)`、`K=C_L*pi*rho_g*lambda_g*a^2`、
`a=drag_diameter_m/2`を既存の明示加速度passで評価します。`C_L`は有限正値をcaseへ明示し、producer提供の
signed方位vorticity `[1/s]`だけを使ってcore内で速度場を微分しません。`lambda_g/a>=10`をfail-closedに要求し、
B02 Brownianとの同時利用は拒否します。速度依存boundは一つのcallbackへ統合し、exponential enclosure v4が
開始速度とhalf predictorの速度boxで再評価します。exponential midpoint v2、engine v30、proposal v7は維持しています。
Brownian数値基盤B01に続き、B02はCartesian XY・fixed charge・Epstein linear drag-onlyの
`ou_langevin`をproductionへ接続しました。凍結係数に対するjoint OU endpoint、物理interval-tree Philox normal、
親endpointを保存するconditional half-split、各leafのcubic Hermite pathを既存の材料event、frame/probe、
checkpoint/resumeへ統合しています。壁lawはterminalな`stick`/`escape`だけです。engine v30はOU covariance、split、
mean更新がfloat64で表現不能なrowだけを`nonfinite_physics`にし、同一batchの正常粒子を継続します。有限depthの
Hermite pathを認証するもので、連続OU trajectoryのexact first-passageは主張しません。
boundaryless unstructured、richer source distribution、moving wallは未解禁です。

B03は、RZ meridional投影とfixed/continuous charge、native/effective-gas線形Epstein、
既存additive forceを一つの`ou_langevin` proposalへ合成して完了しました。
root始点からのnoise-free predictorでmidpointを決め、`gamma,u,T,a,G=dZ/dt,J=dG/dZ`を1回評価し、
`J<=0`を要求するmidpoint-frozen affine exponential chargeと`u_eff=u+a/gamma`のjoint exact OUを
同じroot proposalで進めます。これはstateを前後half-stepでcommitする
Strang/K-O-K分割ではありません。axis hitはaccepted prefixをfoldし、残時間を新しいstochastic rootで
再開します。B02はbitwise不変、terminal wallは`stick`/`escape`/`hold`に限定します。これはr/zへ投影した
2自由度closureであり、等方3-D Brownianや一般state-dependent SDEのstrong order・weak 2次は主張しません。
現行revisionはengine v37、proposal v10、catalog v17、event v16、runtime v20、compiled tile v18、memory plan v14です。
event v15で導入した物理budget＋roundoff budgetとroot-relative TwoDiffを維持し、event v16はvalidなRK4 dense rowの
position Bernstein control enclosureをfacet half-spaceへ射影します。全4制御点の外向き上限が既存budgetの負側に
厳密に入る候補だけをconvex-hull性からclearし、証明不能、不正なcontrol、exponential・scalar経路は引き続き
split/fail-closedです。monotone-approach clearもcubic Hermiteのderivative-Bernstein enclosureだけがopt-inします。
B03 path arrayの静的な保守上限は一slab rowあたり`648 B`で、受入上限`2048 B`内です。正式characterizationは
2,000/20,000粒子×4構成×3反復の24/24実行を全粒子active・failure 0で完了しました。20,000粒子の
machine-local medianはB02 fixed `5.7722 s`、B03 fixed `11.7142 s`、continuous charge＋gravity `11.8427 s`、
axis restart `16.3002 s`で、最大process peak RSSは`226,316,288 B`です。これはportable gateではなく、詳細は
[`solver/evidence/b03/`](solver/evidence/b03/README.md)が所有します。
characterizationで見つかった反復加算由来の終端tailはengine v34で修正し、macro timeを補償積和による
`start + n*dt`のindexed gridから構築して、float64構築roundoff内だけendへsnapします。

P12は、一粒子を一workerだけが更新する非重複tile ownership、read-onlyなfield/geometry、worker-localな
residual/event/statistics、最大`W`個のin-flight tile wave、tile順stable mergeを同じproduction loopへ追加しました。
writerはmain threadだけが呼び、workerからfileへ書きません。geometry queryはcompiled BVH、曲線eventは
保守的なRK4 clear/split事前認証と同時刻wall prefixのbatch再積分を使います。CPU runtime layoutとmemory planは
v3で、workers×microtile scratchをprepare時に見積もります。threads 1/2/4で科学出力はbitwise一致します。
512粒子×4 macro stepのwarm非gating実測（3回median）では、直前serial baseline 3.066786 sに対してP12当時の1 threadが
0.440460 s（6.96倍）でした。一方、2/4 threadは0.528229/0.632361 sで正のthread scalingをまだ示していません。
accepted piece/query/refinement/max depthは8,704/19,968/11,264/21で、全9観測のpayload digestは一致しました。
P12はownershipと決定論的mergeを固定する履歴baselineであり、並列runtimeの完成形ではありません。製品規模の
scalingと残るPython/GIL bottleneckはP14で判定しました。
P13は同じlogical resultを固定64 macro-step epochへ分け、当時は単一HDF5 writer threadの容量1 queueと完了ackで
backpressureをかけました。worker-waveごとのevent/failureをmacro全体へ溜めずにstreamし、交互A/B checkpointと
`LATEST`を唯一のepoch commit pointとして中断runを自動resumeします。`open_result(..., recovery=True)`は
`LATEST`までの確定prefixだけを読み、未完了viewのfinal読出しを拒否します。resume identityはinput hash、schema、
physics/algorithm/backend revisionまで厳密一致を要求し、migrationやsilent fallbackを持ちません。
P13の固定64 cadenceは後続sliceで置換済みです。現行は累積work
`W=macro_step_count+accepted_particle_pieces+candidate_queries+refinements`と
`T=max(2^20,128N)`をengineが計算し、accepted macro barrierで`W-W_epoch>=T`または最終macroの時だけ
commitを決定します。cadence revision・resolved threshold・components・barrierはmanifestとresume identityに入り、
output scheduleとslab幅には依存しません。`output.py`の同期single-owner writerだけがsegment、inactive A/B checkpoint、
`LATEST`のatomic persistenceを所有します。現行result algorithmは`durable_segmented_result_v5`、
result/checkpoint schemaは2、memory planはv13です。
最初の`LATEST`以前の初期状態からの再実行、segment/checkpoint/`LATEST`と最終公開の各境界、確率wall RNG ordinal、
破損・orphanを含むfailure injectionを検証し、verification/scenario 322件が合格しました。容量1 queueは
bounded memoryとbackpressureの意味論でありI/Oを重ねなかったため、P14-Pで同期single-owner writerへ単純化しました。power-loss耐性、remote
filesystem、同じOUTへの複数process同時実行は保証外です。

P14では10k/100k/1M、regular/P1/Q1、initial/cross-cell、0/1/5/20 hit、none/sample/all、cold/warm、1/20 workerを
直交させた23行×3観測を三公開APIで測定し、全行のidentity/revision/memory fitを確認しました。table start validationも
geometry-owned volume cell BVHへ置換し、局所的にfloat64で解像不能なcellをprepareでfail-closedにします。
memory plan v6はfield/geometry index residentとbuild transientを数えます。3観測medianでregular 100k/1Mは
20 workerで1.8796x/4.7965xでしたが、event 10k×20 hitは0.8882xだったため、既定推奨は1 workerのままです。
verification/scenario 336件と標準品質gateが合格し、P14のsynthetic performance baselineは完了しました。ただし
20 workerの効果は大規模regular/event-lightに限られ、主用途のsurface＋非一様場＋wallを結合した並列効果は
未証明です。さらにexact/curved eventはPythonのparticle別調停、stage配列割当て、wave barrierが残り、worker数に
比例するscratchがmemoryとbatch幅を悪化させました。このため`ThreadPoolExecutor` worker-waveをP12/P14の
移行対象baselineとしました。P14-Pでouter pool、future wave、worker別scratchを削除し、再利用
field/physics/integrator workspace、bounded slab、stackless boundary BVH、
同期single-owner writerへ移行しました。linear/quadratic exactと一般曲線eventのflat SoA wavefront、wall/axis
locator、boundary/Philox、row numerical status、batch surface release、direct columnar replay、bounded event/failure
stagingはengine接続済みです。新旧schedulerとobject replayは共存させません。memory plan v11は候補、
event/failure staging、surface release、direct replayをnamed componentへ分離し、pack時だけのgatherは12.5%
safety marginが所有します。正確なbyte式は[`parallel_execution_plan.md`](solver/docs/parallel_execution_plan.md)が所有します。
P14-Pのfocused correction後もregular 1Mは1/2/4 threadで9.32/10.09/10.10 s、4-thread speedup 0.923x、
v20の1-thread比23.7%退行でした。microkernelは約3.75xにscaleしましたが、proposal/enclosureの直列調停が支配し、
限定修正では製品gateに届きません。このためcase schema v2から`resources.threads`、runtime thread mask、
parallel-only test/harnessを削除し、v27のsingle-thread compiled engineへ一本化してP14-Pを完了しました。
v27のregular 1M直列3観測は10.249/10.273/10.100 s（median 10.249 s）で、直列性能の改善余地は
P14-Uへ明示的に引き継いで評価しました。並列runtimeは戻しません。
詳細は[`parallel_execution_plan.md`](solver/docs/parallel_execution_plan.md)を権威とします。

P14-Uは同じ直列engineにXY surface release、Epstein drag、非affine電場、gravity、材料target、多数fixed stepを
結合し、時間収束、固定aspect比の2D regular/P1/Q1 mesh収束、target first-hit、RZ axis crossing/parity、
出力utility、event work、peak RSSとsolver-owned planを一つの外部harnessで判定します。正式releaseは
10k/100k/1M粒子について`none`/sample出力を各3 fresh processで測り、timingと分離した1M `none` profileを
一回取得しました。`release_gate_complete=true`、18 raw観測/6 median、時間・mesh収束、RZ parity、出力utility、
科学payload/revision identity、failure 0を確認し、single-thread compiled engineを維持してP14-Uを完了しました。
受理済みreportは[`solver/evidence/v0.1/p14u_release_v1.json`](solver/evidence/v0.1/p14u_release_v1.json)へ保存しました。
絶対秒数はmachine-localな記述証拠です。
T03 analysis/visualizationは完了しました。P14-Rはcurrent evidence保存に加え、receipt固定のtested head/lockについて
remote Windows/Linuxで620件、性能smoke 7行（cold 1＋warm 6）、wheel、runtime-only clean install、
三公開API smokeまで合格し、`0.1.0.dev0`開発baselineの配布可能性closureを完了しました。
このCI成功はCOMSOL V&Vまたはportable性能の追加認定ではありません。P15は旧着手blockerをユーザーの明示指示で
解除し、RK4-firstと、その受入後のexplicit midpoint chargeまで完了しました。続く外部M3-V評価、canonical
RZ/P1 field builder F01、provider adapterと代表統合を担うF02も完了しました。F02はmixed meshのP1化、代表規模
linear solve、既存RZ solverへのfixed-electric smoke、同一export node上の外部field比較までをcore外で閉じています。
証跡は[`solver/evidence/f02/`](solver/evidence/f02/)にあり、独立mesh convergenceとCOMSOL trajectory／Freeze parityは
未検証です。species制約付きrelative-drift chargeはP15-D、finite-speed EpsteinはP15-E、collisionless Barnes ion dragは
P15-F、Waldmann--Gallis thermophoresisはP16、集約二電流continuous chargeはP18-C、二つの集約ion-drag感度revisionは
P18-I、quasistatic spherical DEPはP18-D、RZ rarefied-vorticity lift sensitivityはP18-L、effective-gas drag / thermophoresis
sensitivityはP18-R、上記Brownian縦切りはB02、局所continuous-path applicability certificateはP19-L、aggregateな
単一価負イオン収集を加えるthree-current chargeはP21 priority 1で完了しました。
現行revisionはengine v37、compiled tile v18、proposal v10、event v16、boundary v5、result algorithm v5、
result/checkpoint schema v2、field location v4、memory plan v14、physics catalog v17、
physics runtime v20、RK4 enclosure v2、dense path v3、charge-stable exponential midpoint v3 / enclosure v4です。dense path v3は
始点相対のBernstein enclosureとTwoDiff残差を使い、world座標への戻しを外向きに丸めることで、
座標原点に依存してevent certificateが閉じない問題を解消しました。このdense-path変更当時はendpoint、path、
engine v32、event v14、RK4 global enclosure v2を変更しておらず、保存済みM3-C1証跡もその履歴revisionを維持します。本体を凍結した外部M3-Vの
最初のdeterministic matched trajectory比較も完了し、共通canonical P1場・共通3力・固定電荷のCase-A 100 nm
pre-event sliceは、独立自己収束から事前登録した幅内で全時系列PASSとなりました。P17 RZ場中Cartesian 3-Dは
別workstreamです。COMSOL比較は引き続き外部V&Vだけが所有し、native場、production boundary parity、stochasticなど証拠のない範囲を
同精度とは認定しません。P18-Rの保存成果物auditではnative linear Epstein式parityが最大相対残差約`1.1e-15`でPASSした一方、
既存P15-E/P16のphysical applicabilityはmixture/model authorityの不一致により12/12 caseで`NOT_APPLICABLE`、PPRの
thermophoresis heat-flux primitiveが無いためpointwise replayは`NOT_TESTED`でした。保存frameは連続pathを認証せず、
P18-R成果物自体は保存artifactだけの監査で、物理的真値やCOMSOL軌道一致を確立しません。その後、外部M3-C0bで
Case-A 100 nmのCOMSOL deterministic pilotを原本不変の`loadCopy`/`-nosave`で実行しました。本体coreは変更していません。
v5の旧「全gate PASS」は、位置relative L2を絶対RZ座標で正規化して原点依存だったため無効化し、v5は履歴上の
`CHARACTERIZED`に留めます。未見の0.15625 usを追加した逐次確認v6では、全粒子がactiveな0--450 usについて
0.625/0.3125/0.15625 us系列を評価し、各runの13,202 recordがすべてactiveでした。0.3125→0.15625 usの
原点不変な変位relative L2、速度relative L2、電荷relative L2はそれぞれ
`3.102727085428027e-5`、`3.92483251084038e-5`、`1.3511393490811483e-6`、観測次数は
`0.9041136/0.944312/1.123838`です。このPASSはpre-eventの運用上の刻み選択だけを確認し、solver一致や普遍的な
物理精度を認定しません。続くM3-C1 frozen saved-state producer-form replayは、不足していた熱泳動PPR primitiveだけを
最小COMSOL補足としてexportし、既存v6と座標が完全一致する13,202 active saved rowについて8/8を閉じました。
Waldmann replayのcomponent-scale normalized residual最大は`4.4046499933294035e-16`、global relative L2は
`1.0992667494449471e-16`です。integrated candidateはCOMSOL native fieldではなくexported exact-connectivity P1です。
P19-Lは元のfixed-step endpointとrev3b event順を維持し、global enclosureをsupport、global-first applicability、短縮RK4
再積分のauthorityとして残したまま、applicabilityだけをlocal fallbackでも認証します。M3-C1で使ったevent v14はglobal supportを
独立に証明済みのvalid `rk4_dense` rowのevent BVH queryだけに現在区間の外向きBernstein boundを使います。dense boundが無効ならsplitし、global enclosureでsupportを
証明できないrowはevent queryもglobal boundへfail-safeに戻します。actual `model_applicability`とfailure code 9の証明不能を分離し、
一row最大64 cellのoverflowは同じrefinement budgetで分割します。4,096粒子×20 stepのglobal/local公開API観測は
`0.9276381/0.9345506 s`、比`1.0074517`、科学payload bitwise一致でした。これはmachine-localな非gating観測です。

M3-C1 Case-A 100 nm pre-eventでは、exported-P1 candidateの3 runを287粒子×46 frame、event/failureなしで完了しました。
COMSOL native-field referenceとのcross-representation 6 gateはすべてFAILしましたが、これは空間表現差を含む別判定です。
この結果を受け、両側へ同じfull-physics exact-connectivity common P1場を与える診断を、固定済みの許容値を変更せず
隔離COMSOL copyで実行しました。初回の0--450 us、287粒子×46 frameについて、位置RMS/maxは
`2.4360183916994177e-11/2.3107811844634912e-10 m`、速度は
`1.8055229266990532e-7/1.1037594178013694e-6 m/s`、電荷は
`7.930847904707173e-7/7.105382977101726e-6 e`となり、絶対値とrelative L2の事前登録9 gateをすべてPASSしました。
これでこの狭いcommon-field・Brownian-off・event前のsame-field solver agreementだけを認定します。native-field同等性、
物理modelの妥当性、Brownian、30 ms、他case・粒径・variant、普遍的なCOMSOL同精度は未認定です。続いて本体を変更せず、
力を切った解析可能なnormal-impact正例でboundary 37 Freezeとboundary 35 DisappearのCOMSOL意味を分離しました。
現行v2は2 scenario×3刻みのexact 6 configuration receiptをCOMSOL process logから照合し、欠落・重複・形式不正・設定差を
fail-closedで拒否します。10/5/2.5 usの3刻みでeventは73 us、最初のterminal保存時刻は75 usです。全active frameは
`x=x0+v0*t` / `v=v0`に一致し、最大位置/速度誤差は`4.726604209672303e-16 m` / `1.7763568394002505e-15 m/s`
（各上限`1e-12`）でした。Freezeはstatus 2でR-Zをhit点に保持し、保存velocityは衝突前値を保持しました。Disappearは
status 4で位置・速度をNaNにしました。56 gate PASS、FAIL 0、velocity意味の記述6件は`CHARACTERIZED_NOT_GATED`です。
この結果はproduction solver parity、grazing/corner、full physicsを認定しません。原本MPH hashは不変です。v1は科学的に
無効だったのではなく、設定receiptとactive-flight oracleの監査強度が不足した履歴成果物としてv2に置換されています。

続くM3-C1 Case-A 100 nm common-P1 material-event sliceは、Brownian-offで0--458.75 usを進め、particle 57の最初の
wafer `stick`だけを比較しました。event v14 candidate v3はfailure 0で、material gate 20/20と再計算した0--450 us
prefix 9/9をPASSしました。prefixのRMS/max/relative L2は、位置
`4.0680739379876864e-13/1.2035672231612963e-12 m/2.0316007372844697e-10`、速度
`2.1084754731725404e-9/3.844306466969233e-9 m/s/1.9630816315240495e-10`、電荷
`1.1818881686693635e-7/2.37588949403289e-7 e/4.670220471379047e-10`です。event時刻、hit位置、terminal電荷の
candidate/reference絶対差はそれぞれ`2.157542807607049e-13 s`、`2.683964162031316e-14 m`、
`4.7283812421028415e-9 e`です。

旧event v13 candidateはquery/refinement/accepted/depthが
`16,427,517/7,792,306/8,635,211/16`で、450 us checkpointまでにrefinementの
`7,623,460/7,792,306`（`97.8331703092769%`）を既に費やしていました。v14はこれを
`842,927/11/842,916/11`へ減らし、failure 0を維持しました。同じshellでのoperator観測wall timeは約
`14 min 13 s`（約`853 s`）から約`36.5 s`（約`23.4x`）ですが、manifest内のsolver timingではなくmachine-localな
非gating値です。COMSOL側のcommon-P1入力・reference・原本MPHは変わらないため再実行せず、hash固定済みreferenceと
v14 candidateをevaluation v5で再評価しました。

旧3刻みcandidateのroundoff-level一致は、global safety enclosureをevent BVH boundにも流用した人工的な細分で三つの
macro stepが同じ実効pieceへ潰れていたため、独立な時間収束とは扱いません。event v14による0.625/0.3125/0.15625 usの
solver-only再実行はrefinement 0、query=accepted `206,927/413,567/826,847`で完了し、位置・速度・電荷のRMS観測次数
`2.029875353701904/2.0816971911764033/2.044084026475049`、fine relative L2
`6.099791486063973e-8/8.321356016032579e-8/1.3796067752988052e-8`を得て全量`ORDER_EVALUATED`の自己収束PASSとなりました。
このv14 fine candidateを使う現行common-P1 comparisonも、位置
`4.06807283316903e-13/1.2035778717837921e-12 m/2.0316001855367802e-10`、速度
`2.10847522277453e-9/3.844306466969233e-9 m/s/1.9630813983926987e-10`、電荷
`1.1818881019499895e-7/2.3758877887303242e-7 e/4.670220207738041e-10`で9/9をPASSしました。
これはsolver-onlyのpre-event時間収束と限定same-field agreementであり、それだけではmaterial-event gateの範囲を拡張せず、native-field等価性、物理妥当性、
Brownian、30 ms、他caseを認定しません。compact authorityは
[`solver/evidence/m3c1/case_a_100nm_material_event_v1/`](solver/evidence/m3c1/case_a_100nm_material_event_v1/)です。

P18-Hのgeneric terminal `hold/held`はengine v32 / boundary v5 / result v4として完了しました。解析的直線・曲線hit、
右連続frame、resume、slab identity、OU/Brownian、zero-time surface、analysis非deposition分類を公開経路で確認し、
既存hash固定Freeze referenceへの外部candidateもCOMSOL再実行なしで15/15 PASSです。compact authorityは
[`solver/evidence/p18h/hold_freeze_v1/`](solver/evidence/p18h/hold_freeze_v1/)です。認定は力なしnormal-impactの
terminal semanticsに限定し、full physicsやgrazing/cornerを認定しません。

100 nm・30 ms candidate v3はCase A/PのBrownian-off `h,h/2,h/4`自己収束をPASSしました。これを主要な数値判定とします。
同じ保存時刻のCOMSOL Case A/Pはmanual explicit fixed RK4 10 us、Brownian-on、native finite-element fieldの単一runなので、
candidateとの軌道差は外部characterizationでありcore gateではありません。candidateへCOMSOLの刻みを強制しません。

charge-stable continuous couplingとdeterministic work-scaled durable cadenceの二production sliceは完了し、
直近の全品質gateはPASSです。RK4のexplicit `hL_Z<=0.5`は維持し、exponential midpoint/B03は
midpoint-frozen affine exponential root pathでstability bottleneckを外します。clip、charge-only subcycle、第二engine、
threadingの再導入はありません。P20 performance closeoutに続き、meaning-matchedな外部V&V/M3-C2A Case-A/Case-P
100 nm anchorまで完了しました。
[`solver/evidence/p20_efficiency/`](solver/evidence/p20_efficiency/README.md)はmanual・machine-local・non-gatingな
運用効率証拠であり、portable timingまたは異なるstep間のequal-accuracyを主張しません。
COMSOL fittingや意味の異なる保存runへのstep/path合わせはcoreの目的にしません。将来のstochastic比較は同じ
物理・field・確率意味を持つ独立seed ensembleで行い、単一seed pathを合否に使いません。保存済みM3-C1 compact
evidence/evaluatorはevent v14の履歴として固定します。

M3-C2Aでは過去のnative-field単一履歴を使わず、common-P1 Case-A 100 nmの意味一致companionを新規実行しました。
独立pilotでCOMSOL 20 usとcandidate 20 us / Brownian tree depth 3を選択し、結果を見ずに固定した別seedで各32 replica、
287粒子、30 ms、121時刻を完走しました。確認的な4終端人口曲線の最大差は0.5989%、95%同時上限は3.877%で、
事前登録した5%同等性幅を`PASS`しました。R-Z平均軌道の重ね合わせも保存しています。これは固定済みcommon-P1 Case-A anchorの
終端人口精度だけを認定し、pathwise RNG一致、native field、Case-P、第2 ion-drag、10/30 nm、普遍的COMSOL同等性は認定しません。
compact authorityは
[`solver/evidence/m3c2/caseA_100nm_final_campaign_v1/`](solver/evidence/m3c2/caseA_100nm_final_campaign_v1/README.md)です。

続くCase-P 100 nm finalも、20 us、COMSOL/candidate各32独立seed、各287粒子、30 ms、121 frameで完了しました。
登録済み83区分R-Z/fate gateは最大empirical TV `0.010670731707317093`、同時上限
`0.13119968456545308 < 0.15`で`PASS`しました。終端gateも`PASS`ですが、全64 replicaでevent 0のため境界parityには
情報を持ちません。これは元Case-P COMSOL `auxq`が意図する電子＋正イオン二電流とのsame-form比較で、後続のaggregate
three-currentやspecies-resolved物理を認定しません。pathwise RNG一致、普遍的COMSOL同等性、eventful boundary parityも主張しません。
authorityは[`solver/evidence/m3c2/caseP_100nm_final_campaign_v1/`](solver/evidence/m3c2/caseP_100nm_final_campaign_v1/README.md)です。
別のdeterministic common-P1 companionでは、size-specificな入力とprovenanceを修正し、10/30 nmの
relative-flow ion dragと100 nmのimage ion dragについて、各287粒子・0--450 us・3刻みのcandidate/COMSOL自己収束と
cross-solver 9 gateをすべて`PASS`しました。これはevent-free、Brownian-offの限定比較で、上記M3-C2A stochastic anchorの
主張範囲を変更しません。authorityは
[`solver/evidence/m3c1/case_a_size_ion_drag_companion_v1/`](solver/evidence/m3c1/case_a_size_ion_drag_companion_v1/README.md)です。
optional aggregate three-currentのproduction実装はP21 priority 1で完了しました。priority 2のcritical boundary microcaseも
`PASS`です。外部Case-P派生companionのpriority 3はproducer-owned one-sided cacheによりcanonical負イオン5 primitiveを
全1987節点へ生成して入力blockerを解消しました。priority 4の共有three-current `Z0`を用いたcommon-P1、Brownian-off、100 nm、
287粒子、30 msの比較で、candidateと明示drag COMSOL referenceの各3刻み収束、全frameのlifecycle/finite mask exact、
共通の有限lifecycle stateの`r,z,Z`、共通active stateの`v`、141件のevent/fate identityをすべて`PASS`しました。
元Case-P二電流anchorは不変です。
この異なる任意物理の外部coverageはP21の出口から分離し、P21と明示scopeの2D benchmarkは
`CLOSED_ACCEPTED_WITH_LIMITATIONS`、`2D_CRITICAL_VV_COMPLETE`です。このscoped numerical agreementは物理model validationや
普遍的COMSOL同等性の認定ではありません。authorityは
[`solver/evidence/m3c3/caseP_three_current_companion_v1/`](solver/evidence/m3c3/caseP_three_current_companion_v1/README.md)です。
旧M3-C0 umbrellaに残る全12 package総当たり、完全derived-field provenance、RK probeは
`DEFERRED_NOT_RELEASE_BLOCKING`です。現行v0.1の完了条件へ含めず、科学revisionまたは明示的なcoverage拡張時だけ
独立work packageとして再開します。

本体coreへCOMSOL分岐や比較用toleranceを追加していません。final計時は
`NON_AUTHORITATIVE_EXTERNAL_WORKLOAD_OVERLAP`です。受理済みcandidate seed `319032`、`319047`、`319063`の
287粒子owner discoveryは科学payload・work・case identity・revisionを受理済み結果と完全一致させて完了しました。
3 seedとも支配ownerは`integrators`で自己時間比は42.58--42.86%でしたが、事前登録済みbounded ownerではないため
`optimization_authorized=false`で、本体は変更していません。Case-A/Case-P 100 nm anchor benchmarkは
`CLOSED_ACCEPTED_WITH_LIMITATIONS`です。10,000粒子以上の性能、残るpackage、process並列は完了条件ではなく、
利用SLAを先に定義した独立work packageです。owner discoveryのauthorityは
[`solver/evidence/m3c2/caseP_100nm_owner_profile_v1/`](solver/evidence/m3c2/caseP_100nm_owner_profile_v1/README.md)です。
後続の明示指示によるbounded follow-upでは、この一つのchord ownerだけをcompiled batchへ統合し、重複式を削除しました。
同じaccepted 3 seedで科学payload/workを完全一致させたままend-to-end中央値を45.4668 sから39.8801 sへ12.29%短縮し、
そこで停止しています。これは10,000粒子scaleやCOMSOL速度比の認定ではありません。authorityは
[`solver/evidence/m3c2/caseP_100nm_chord_optimization_v1/`](solver/evidence/m3c2/caseP_100nm_chord_optimization_v1/README.md)です。
終了条件のauthorityと詳細は[`implementation_plan.md`](implementation_plan.md)、
[`vv_methodology.md`](vv_methodology.md)、[`solver/evidence/m3c1/`](solver/evidence/m3c1/)、
[`solver/evidence/m3c0/boundary_semantics_v2/`](solver/evidence/m3c0/boundary_semantics_v2/)が所有します。
各caseは所有packageの安全条件が完成した時点でproduction結果との比較を
有効化します。通常の品質確認は次で実行します。

```console
cd particle_platform_redesign/solver
uv sync --locked
uv lock --check
uv run --locked ruff format --check src tests tools
uv run --locked ruff check src tests tools
uv run --locked pyrefly check --summarize-errors
uv run --locked lint-imports
uv run --locked python scripts/check_complexity.py
uv run --locked pytest tests/verification tests/scenarios -q
```

可視レポートの再ビルドは次のとおりです。

```powershell
$dataAppBuilder = Get-ChildItem "$env:USERPROFILE/.codex/plugins/cache/openai-curated-remote/data-analytics" `
  -Filter data-app.mjs -Recurse | Sort-Object FullName -Descending |
  Select-Object -First 1 -ExpandProperty FullName
node $dataAppBuilder build `
  --project-dir (Resolve-Path particle_platform_redesign/report_app) --separate-data
```

## 利用上の重要事項

`model_dataset` は保存済みCOMSOL結果を理解し、外部で回帰・軌道比較するためには利用できますが、
製品仕様全体を証明するgolden datasetではありません。とくに粒子依存の派生場、四角形要素の
節点順序、ブラウン運動の再現性、壁面反射・表面放出・帯電連成の網羅性には制約があります。
製品coreの実装は解析解microcaseを主軸に進め、COMSOL比較は `tools/vv/comsol` 相当の外部toolで
行います。詳細は主仕様書、データ品質評価、V&V手順、再構築計画を確認してください。
