# Result format v3

この文書はresult schema 3、現行engine v45、result algorithm v6、checkpoint schema 2の永続resultと
`ResultView`の契約です。軌道計算の入力契約は
[`case_format_v3.md`](case_format_v3.md)が所有し、集計、可視化、COMSOL比較はこのresultを読むcore外の
toolが所有します。

## v3成果物の物理配置

正常終了したresultは次のdurable layoutを持ちます。segment数はengineがaccepted macro barrierで判定する
`cumulative_solver_work_v1` cadenceと最終macroから決まります。

```text
OUT/
  run.json
  segments/epoch-000000.h5
  segments/epoch-000001.h5
  ...
  checkpoints/A.h5
  checkpoints/B.h5
  LATEST
  final.h5
  _SUCCESS
```

writerは同じ親directoryの`OUT.partial/`へ書きます。main threadの単一`ResultWriter`だけがsegment、checkpoint、
finalを同期的に所有するため、diskが遅い時は自然にcomputeへbackpressureし、eventを捨てません。最終epoch
commit後に`final.h5`、complete `run.json`、`_SUCCESS`の順で作成してから`OUT/`へrenameします。既存の完成`OUT/`は
上書きしません。P13で使った容量1 queueとbackground writerは各command直後にackを待っておりcompute/I/Oを
重ねなかったため、P14-Pで削除しました。永続layout、commit point、resume意味論は変更していません。

matching resume identityを持つ`OUT.partial/`があれば`simulate(case, OUT)`が`LATEST`から自動再開します。
identity不一致、未知version、参照artifactのhash/構造破損はfail-closedで、checkpoint migrationや互換readerを
持ちません。最初の`LATEST`以前に失敗した同一identityのpartialはcommit済みstateを持たないため、epoch 0を
初期状態から正確に再実行し、orphan temp/final名を置換します。`open_result(OUT, recovery=True)`は未完了partialの
`LATEST`までをread-onlyに開きますが、finalはauthoritativeでないため`read_final()`を拒否します。
final、complete manifest、`_SUCCESS`、directory renameの途中で失敗したpartialは、次の同一identityの`simulate`が
検査して最終化を再実行します。`_SUCCESS`まで確定したpartialは再計算せず完成directoryへ公開します。

## `segments/epoch-NNNNNN.h5`

root attributeは次のとおりです。

| attribute | 値 |
|---|---|
| `result_schema_version` | integer `3` |
| `segment_index` | zero-based integer `N` |
| `closed` | uint8 `1` |

release eventは物理時刻、次に`particle_id`の順で格納します。P04でevent typeはgroup名`release`が
表し、各粒子の最初のeventなので`event_ordinal`は0です。

```text
/events/release/time_s          float64 [E]
/events/release/particle_id     int64   [E]
/events/release/event_ordinal   uint32  [E]
/events/release/source_id       int32   [E]
```

boundary eventは、局在した材料壁interactionまたはperiodic topology transferを一行に保存します。releaseが
`event_ordinal=0`なので、同じ粒子のwall/periodic logical interactionは1から連続します。行の物理的identityは
`(particle_id,event_ordinal)`、一括readerの公開canonical順は`(time_s,particle_id,event_ordinal)`であり、file行順を
意味論のauthorityにしません。同時hit candidateは各event内でfacet ID順です。event v21はevent v19 / v18と同じくexact、一般RK4、
exponential midpoint、Brownianの最初のhitと局在budget内で同時なincident facet集合をここへ保持し、
primary facetだけへ潰しません。`candidate_count[i]`は重複列を持たず、
`candidate_offset[i+1]-candidate_offset[i]`です。hold eventは`law_id=hold`、`outcome=held`で、
`position_m`とpre/postの速度・電荷はhit時payloadを表します。

```text
/events/boundary/time_s                    float64 [B]
/events/boundary/particle_id               int64   [B]
/events/boundary/event_ordinal             uint32  [B]
/events/boundary/interaction_kind          UTF-8   [B]
/events/boundary/primary_facet_id          int64   [B]
/events/boundary/destination_facet_id      int64   [B]
/events/boundary/boundary_id               int32   [B]
/events/boundary/material_id               int32   [B]
/events/boundary/contact_radius_m           float64 [B]
/events/boundary/position_m                float64 [B,2]
/events/boundary/position_post_m           float64 [B,2]
/events/boundary/normal                    float64 [B,2]
/events/boundary/velocity_pre_m_s          float64 [B,2]
/events/boundary/velocity_post_m_s         float64 [B,2]
/events/boundary/charge_number_pre         float64 [B]
/events/boundary/charge_number_post        float64 [B]
/events/boundary/model_weight              float64 [B]
/events/boundary/law_id                    UTF-8   [B]
/events/boundary/outcome                   UTF-8   [B]
/events/boundary/localization_residual_m    float64 [B]
/events/boundary/position_budget_m          float64 [B]
/events/boundary/time_budget_s              float64 [B]
/events/boundary/candidate_offset           int64   [B+1]
/events/boundary/candidate_facet_id         int64   [C]
```

`interaction_kind`のcanonical値は`wall`または`periodic_translation`だけです。`wall`行は
`destination_facet_id=-1`かつ`position_post_m=position_m`で、従来どおり`law_id`/`outcome`が材料応答を表します。
`periodic_translation`行は`primary_facet_id`を出発面、`destination_facet_id>=0`を対応面、`position_m`を出発面上の
局在位置、`position_post_m`をtranslation後の位置として保存し、`law_id=""`、`outcome="transferred"`です。
速度、電荷、model weightはtransferで変えず、材料wall lawとwall RNGは呼びません。この三列を省いた旧encodingや
列値からinteraction種別を推測するcompatibility readerは持ちません。

粒子単位で局在できる数値不能はrun全体を捨てず、failure eventとして保存します。行順は
`(time_s, particle_id, event_ordinal)`のstable順で、reason codeはmanifestのversioned mappingがauthorityです。
failureは物理wall interactionではないためwall RNG ordinalを消費しません。現在のcodeは
`1=numerical_event_budget`、`2=indeterminate_event`、`3=indeterminate_boundary_policy`、
`4=indeterminate_surface_departure`、`5=field_support`、`6=model_applicability`、
`7=nonfinite_physics`、`8=integrator_accuracy`、`9=indeterminate_applicability_certificate`です。
動的に一粒子へ局在できる問題だけをlocal failureにします。入力不正、静的・共有field layout/model係数/bound、
topology、I/O失敗は引き続きrun-fatalです。

```text
/events/failure/time_s          float64 [D]
/events/failure/particle_id     int64   [D]
/events/failure/event_ordinal   uint32  [D]
/events/failure/reason_code     uint16  [D]
```

各macro-step終端には、全resident粒子をちょうど一つの状態へ数えた小さい整数seriesを一行保存します。
各行で`pending + active + stuck + held + escaped + failed = N`を満たします。

```text
/series/time_s                  float64 [S]
/series/pending                 uint64  [S]
/series/active                  uint64  [S]
/series/stuck                   uint64  [S]
/series/held                    uint64  [S]
/series/escaped                 uint64  [S]
/series/failed                  uint64  [S]
```

trajectory frameは指定時刻順のragged tableです。`offset`は長さ`F+1`で、frame `i`の行範囲は
`offset[i]:offset[i+1]`です。各frame内は`particle_id`順とし、release前の粒子を含めず、release時刻と
一致するframeにはrealized sourceの初期状態を保存します。frameが空でも時刻とoffsetは一件記録します。

```text
/frames/time_s                  float64 [F]
/frames/offset                  int64   [F+1]
/frames/particle_id             int64   [R]
/frames/position_m              float64 [R,2]
/frames/velocity_m_s            float64 [R,2]
/frames/charge_number           float64 [R]
/frames/lifecycle               uint8   [R]
```

全数値datasetはlittle-endianです。lifecycle codeは`0=pending`、`1=active`、`2=stuck`、
`3=escaped`、`4=failed`、`5=held`です。frameはeventに対して右連続です。hit時刻ちょうどのstick粒子は
hit位置・速度0・stuckとして残り、hold粒子はhit位置・hit時速度・hit時電荷を保持してheldとして残ります。
activeな反射はhit位置・反射後速度を持ち、escape/failed粒子はterminal時刻ちょうどとそれ以後のframeへ含めません。
heldはinactiveな終端状態であり、hit後の位置、速度、電荷を時間発展させるpaused-particle modelではありません。
surface上のzero-time active応答も同じpost-state jumpを使います。
terminal時刻より前はaccepted path上のactive stateです。
要求出力時刻のためにmacro stepを分割せず、受理した`StepProposal`をその時刻で評価します。無力場は
release原点からの解析的ballistic、証明済み一定加速度は解析的`quadratic_exact`で評価します。revision 3aの
boundaryless general RK4は、full macro proposalと全短縮`state_at()`評価の内部stage・endpointを覆う外向き
support enclosureと、Epsteinの連続applicabilityを先に証明した`rk4_reintegrated`で評価します。frameは
受理済みproposalを読み出すだけで、新しいvalidity判定、production step、resident stateを作りません。

probeはtrajectoryと同じragged state列と右連続event意味論を使いますが、caseで明示したparticle IDだけを保存します。
probe時刻もproduction macro-stepを分割しません。

```text
/probes/time_s                  float64 [P]
/probes/offset                  int64   [P+1]
/probes/particle_id             int64   [Q]
/probes/position_m              float64 [Q,2]
/probes/velocity_m_s            float64 [Q,2]
/probes/charge_number           float64 [Q]
/probes/lifecycle               uint8   [Q]
```

## `checkpoints/A.h5`、`checkpoints/B.h5`、`LATEST`

checkpoint schema versionは2です。commit IDが偶数なら`A.h5`、奇数なら`B.h5`をinactive generationとして
置換します。checkpointは次のaccepted macro barrierから再開するためのmutable solver stateだけを保存し、
完成resultの科学datasetを複製しません。

```text
attributes:
  checkpoint_schema_version, result_schema_version, result_algorithm_revision
  commit_id, segment_index, segment_sha256
  resume_identity_hash, macro_time_s, macro_step_count
  release_cursor, frame_cursor, probe_cursor
  accepted_particle_pieces, candidate_queries, refinements, maximum_refinement_depth
  wall_interactions, residual_splits, axis_crossings

/counts attributes              committed logical row/frame counts
/state/position_m               float64 [N,2]
/state/velocity_m_s             float64 [N,2]
/state/charge_number            float64 [N]
/state/lifecycle                uint8   [N]
/state/failure_reason_code      uint16  [N]
/state/terminal_time_s          float64 [N]
/state/event_ordinal            uint32  [N]
/state/physical_boundary_event_ordinal uint32 [N]
/state/exact_origin_time_s      float64 [N]
/state/exact_origin_position_m  float64 [N,2]
/state/exact_origin_velocity_m_s float64 [N,2]
/state/start_contact_state      uint8   [N]
/state/active_particle_index    int64   [A]
/state/last_field_cell          int64   [N]   # P1/Q1 runtimeだけ
```

`LATEST`はversioned JSONで、result/checkpoint schema、`commit_id`、`segment_index`、参照する
`A.h5 | B.h5`、そのcheckpoint SHA-256、累積logical countを持ちます。一epochの順序は次で固定します。

1. `segments/epoch-NNNNNN.tmp.h5`をflush/closeし、確定segmentへreplaceする。
2. inactive checkpoint tempをflush/closeし、`A.h5`または`B.h5`へreplaceする。
3. checkpoint hashを含む`LATEST.tmp`を`LATEST`へreplaceする。ここだけがepoch commit pointである。

resume/recoveryは`LATEST`が参照するcheckpoint hashと、checkpointが参照する最新segment hashを照合し、0から
`segment_index`までのclosed segmentだけを採用します。それより新しいsegment/checkpoint/tempはorphanとして
無視し、再開時に置換します。全segmentの構造と累積countも`LATEST`へ一致させるため、hashを直接持たない古い
segmentの行欠落も拒否します。ただし過去segmentの全値hashは保持しないため、shape/countを変えない過去値改変の
検出はP13の契約外です。通常の完成resultはcheckpointを科学datasetのauthorityにせず、closed segment、complete
manifest、`final.h5`を検査して読みます。

## `final.h5`

rootの`result_schema_version` attributeは3です。`/particles`は常に全resident粒子を`particle_id`順で
格納します。`time_s`はrun-end snapshotの時刻として全行でrun終端です。escaped/failedの物理terminal時刻は
対応するboundary/failure eventが所有します。

finalの行数はprepare済みの粒子容量、およびcomplete manifestの`counts.particles`と一致しなければなりません。
粒子IDは非negative・厳密昇順で、一意ですが、連番である必要はありません。publish前、完成resultのopen時、
`read_final()`が実際に開いたhandleで同じ検査を行います。IDと既存のcontact radius、validity、
lifecycle/reasonの値検査は固定長row blockで行い、検査用の全粒子配列を追加しません。
manifestの粒子数・macro-step数も非negative integerを要求します。この検査は列構造・容量・ID順序の整合を
保証するもので、sourceへのexact ID membership、全event/frameの参照整合や任意改変の検出は追加しません。

```text
/particles/particle_id                   int64   [N]
/particles/source_id                     int32   [N]
/particles/time_s                        float64 [N]
/particles/position_m                    float64 [N,2]
/particles/velocity_m_s                  float64 [N,2]
/particles/kinematics_valid              uint8   [N]
/particles/charge_number                 float64 [N]
/particles/lifecycle                     uint8   [N]
/particles/mass_kg                       float64 [N]
/particles/drag_diameter_m               float64 [N]
/particles/contact_radius_m               float64 [N]
/particles/electrostatic_radius_m        float64 [N]
/particles/displaced_volume_m3           float64 [N]
/particles/model_weight                  float64 [N]
/particles/material_id                   int32   [N]
/particles/failure_reason_code           uint16  [N]
```

粒子propertyは入力の独立authorityをそのまま保持し、質量や半径から別propertyを再構成しません。
`contact_radius_m`は有限かつ非負で、0はpoint particleです。boundary eventの`position_m`は接触時の粒子中心を表し、
同じ行の`contact_radius_m`が粒子の独立した物理接触半径を保持します。groupの`particle_center`判定でも
この値を0に書き換えません。方式はmanifestの`resolved.boundary_laws[].contact_geometry`で識別します。
省略時の入力は`particle_surface`として解決し、case hashとgeometry/event/source/engine revisionをresume identityに保持します。
active/stuck/heldの`kinematics_valid`は1です。heldはhit位置・hit時速度・hit時電荷を保持するinactiveな
非deposition終端で、以後の物理更新を行いません。escaped/failedは0で、有限なposition/velocity payloadには最後のhitまたは
局在できたfailure状態を保持しますが科学値として使用できません。NaNをlogical nullやfailure sentinelとして
使わず、escaped粒子の最後の有効な位置・入射/応答速度はboundary eventをauthorityとします。
`failure_reason_code`はfailedだけ非zeroで、他状態は0です。P19-Lのcode 9
`indeterminate_applicability_certificate`は有限budget内で局所証明が閉じなかった数値statusであり、実際に確認した
`field_support`または`model_applicability`違反へ畳み込みません。新datasetや診断traceは追加せず、既存mapping/countだけで
区別します。

## `run.json`

manifestは少なくとも次を記録します。

- `status: complete`、`result_schema_version`、`result_algorithm_revision`、`checkpoint_schema_version`
- `durable_commit_cadence`（revision、resolved work threshold、4 work components、accepted-macro barrier）、
  `segment_count`、`latest_commit_id`、strictな`resume_identity`とそのSHA-256
- case名、raw YAMLの`case_file_hash`、canonical HDF5の`data_content_hash`、case schema version
- data coordinate system、particle motion mode
- engine、compiled tile、step-proposal、選択methodのRK4またはexponential enclosure、一般RK4で使うdense path、physics catalog/runtime、field-location、required-field、geometry、event、boundary、topologyのalgorithm revision
- source/RNG algorithm revision。sourceはrealized rowだけでdraw kindを持たず、RNG draw kindはprobabilistic/
  Maxwell wallとBrownian root/splitに限定する
- BrownianではRNG、joint OU、conditional split、tree policyのrevision、base/max depth、root/split draw kind、
  macro-root coefficient policy
- requested/resolved integrator、backend、seed、path kind、`maximum_dt_charge_lipschitz`、
  physics model revision、required field binding
- source ID/name/type、解決済みboundary law・priority・fallback law・反発係数・付着確率
- start/end/dt、field snapshotによる追加macro split時刻、YAML順の`source_id_to_name`
- resolved event設定。一般RK4材料boundary経路ではaccepted particle-piece、candidate query、refinement、最大深さの集約
- wall event、residual split、RZ axis crossingの集約
- solver-owned memory plan revision、semantics、limit/planned bytes、slab幅、phase peak、component内訳
- particle、release/boundary/failure event、series、frame/frame row、probe/probe row、macro stepの件数
- lifecycle counts、failure reasonのname-to-code mappingとreason別件数

P06 revision 3bの数値pathはengine `coupled_rk4_engine_v6`で確立した。現行algorithm revisionは次である。

```text
engine          particle_engine_v45
compiled tile   compiled_cpu_tile_v21
runtime layout  resident_soa_serial_slab_v6
memory plan     solver_owned_memory_plan_v16
step proposal   coupled_fixed_step_proposal_v10
RK4 enclosure   rk4_global_abs_enclosure_v2
RK4 dense path  rk4_position_hermite_state_extension_v3
exponential     charge_stable_exponential_midpoint_v3
exp enclosure   exponential_midpoint_local_stage_enclosure_v4
event           line_quadratic_curved_capsule_periodic_first_hit_v21
boundary        contact_wall_laws_v7
topology        translation_periodic_xy_v1
source          realized_internal_surface_contact_schedule_v5
physics catalog inertial_langevin_2d_catalog_v23
physics runtime signed_ion_compiled_physics_runtime_v22
required field  required_field_time_linear_v6
field location  field_location_v4
field time      fixed_topology_linear_time_v2
geometry        line_boundary_capsule_contact_bvh_v7
result          durable_segmented_result_v6
checkpoint      schema version 2
```

`rk4_enclosure_revision`と`rk4_dense_path_revision`は一般stage RK4を実際に使うrunだけに記録し、ballistic、
`quadratic_exact`、exponential midpoint、OUへ実行していないrevisionを付けません。
`field_time_revision`はcanonical data schema v3の固定topology・線形snapshot契約を識別する。
required field entryは`time_interpolation`、snapshot数・範囲を、time sectionは固定`dt_s` gridへ追加した
`field_snapshot_splits_s`を記録する。静的場だけなら追加splitは空で、result dataset/schemaは増えない。
resume identityにもfield time revisionとsplit時刻を含めるため、異なる時間partitionのpartialを再開しない。

`contact_wall_laws_v7`では`specular`はparameterなしの完全鏡面で、`restitution`だけが法線・接線の反発係数を持つ。
`maxwell_thermal`はwall temperature、Maxwell拡散混合率、接線wall-frame velocityをmanifestへ記録する。
`probabilistic_stick`のfallbackはparameterなしの`specular`、両係数を持つ`restitution`、または完全なparameterを持つ
`maxwell_thermal`である。eventの`law_id`は選択されたtop-level lawを保持し、compound lawの実現分岐は既存の
`outcome`とdraw referenceで表す。v5はparameterなしの`hold`と`held` outcomeを追加し、v6は既存event列を変えず
Maxwell thermal反射を追加した。lifecycle seriesへ`held`を一列追加したためresult/checkpoint schemaを
2、result algorithmをv4へ一度だけ更新した。case schema v2は不変で、v1 result/checkpointの互換readerやmigration shimは
持たない。`realized_internal_surface_schedule_v3`ではinternal tableは`position_m`、surface tableは`facet_id`と
strict interiorの`facet_parameter`を持ち、両者が粒子ごとのvelocity、release時刻、物性を保持する。
source分布のselectorやsource RNGはresult/manifestへ持ち込まない。
`inertial_langevin_rz_catalog_v19`は`gravity_buoyancy_standard_v1`のmodel revisionを維持しつつ、annular domainを
含む全RZ caseでradial gravityを0へ限定する。Cartesian XYの第一成分はこの制約を受けない。
P15-Fのion dragも既存のresolved physics-model mapping、required-field binding、particle-local failureへ収まり、
result datasetまたはschemaを追加しない。
P16のthermophoresisも同じmapping/binding/failureを使い、`waldmann_gallis_free_molecular_single_species_heat_flux_v1`
をmanifestのresolved modelへ記録するだけである。force traceやheat-flux datasetをresultへ複製せず、result schema v1、
checkpoint schema 1、durable result algorithm v3を維持する。
P22のTalbot熱泳動とSaffman liftも既存の`resolved_physics_models`、`required_fields`、failure vocabularyを
そのまま使う。model別のresult datasetやschemaは追加しない。
P18-Iの二つのaggregate ion-drag revisionも同じresolved mappingと既存failureを使う。manifestは選択した
model/revisionを記録するだけで、stage force、producer固有速度、比較残差をresultへ追加しない。外部frozen-force比較は
`evidence/p18i/`が所有し、result/checkpoint schema 1は不変である。
P18-DのDEPも同じresolved mapping、required-field binding、既存failureへ収まる。manifestは
`quasistatic_spherical_gradient_e2_v1`を記録するが、stage force、`mean_E_squared`、gradient recovery、producer固有の
point-dipole認証をresultへ複製しない。これらの入力provenanceはcanonical caseが所有し、result/checkpoint schema 1は
不変である。
P18-Lのliftも同じresolved mapping、required-field binding、既存failureへ収まる。resolved lift entryは
`rarefied_vorticity_sensitivity_rz_v1`、gas velocity/density/mean-free-path/signed azimuthal-vorticityのfield binding、
caseで明示した正の`lift_coefficient`を記録する。stage force、vorticity recovery、COMSOL式残差をresultへ複製せず、
producer provenanceと外部M3-C1へ残す。result/checkpoint schema 1は不変である。
P18-Rの二つのeffective-gas sensitivityも既存resolved mapping、required-field binding、failure vocabularyへ収まる。
manifestは選択したrevisionとfield bindingをresolved mappingへ記録し、入力hashがcase明示の`maximum_speed_ratio`を識別するが、
species配列、mixture rule、`q_eff`の生成過程、stage force、COMSOL比較statusをresult datasetへ追加しない。producerの
one-effective-Maxwellian/pseudogas認証はcanonical case provenance、外部auditはV&V成果物が所有し、result/checkpoint
schema 1は不変である。

engine v7は一般RK4材料壁の
proposal/refinement行のwork partition、engine v8はreplay保持範囲とP1/Q1 material capabilityだけを変更し、
engine v9はsource realization、wall response、exact-path残時間、ballistic RZ axis foldを追加した。engine v10と
event v6はCartesian XYの証明済み一定加速度surface departureとsource-facet start-contact certificate、engine v11と
event v7はCartesian XY一般RK4の厳密内向きsurface departure、single-facet active-boundary residual、右連続state jump、
state-shared interaction capを追加した履歴revisionである。engine v12 / event v8はこれを維持し、RZ signed-stage basis、
axis event、残時間継続を追加した。engine v13はP06-Sの小さいphysics runtimeと明示Stokes--Cunninghamを、
engine v14/result v2はP08のparticle-local failure、lifecycle series、state probeを追加した。既存の
event/final/frameの物理意味とresult schema v1は維持し、pre-release algorithm revisionだけを更新した。
engine v15はP09の固定resident-row active index、bounded microtile work partition、stable tile mergeと
memory plan v1を追加した。物理、event、proposal、result algorithm/schemaの意味は変えていない。
engine v16はP10のNumba field/physics/RK4 array pass、accepted endpoint P1/Q1 hint、runtime layout v2、
memory plan v2、physics runtime v2を追加した。case/result schema、proposal、event、field semanticsは変えていない。
engine v17はP11のmethod-neutralなengine名、compiled tile v2、proposal v4、event v9、physics runtime v3と
`exponential_midpoint_v1`を追加した。結果datasetとschema v1、physics catalog/model revision、field semanticsは
変えていない。RK4と指数法は同じengine、event、output writerを使う。
engine v18はP12の一粒子一worker partition、worker-local residual/event/statistics、最大worker数分のtile wave、
tile順stable merge、compiled BVH/保守的piece事前分類/同時刻wall-prefix batchを追加した。結果dataset、schema v1、
proposal v4、event v9、physics runtime v3は変えていない。requested thread数はCPU worker数の上限で、粒子ありなら
resolved thread数は`min(requested, particle_count)`、空なら1である。memory不足時にsilent downgradeしない。
engine v19/result v3はP13の固定64 macro epoch、worker-wave event/failure stream、容量1 single-writer queue、
A/B checkpoint、`LATEST`、auto-resume、recoveryを追加した。case/result schema 1、compiled tile v3、runtime layout v3、
geometry v3、proposal v4、event v9、physics catalog/runtimeと全logical scientific datasetは変えていない。
engine v20/compiled tile v4はP14でfield v3のsupported-containment BVHとgeometry v4のvolume-cell BVHをprepare/
compiled経路へ接続した。outside/masked field provisionalのfull scan、logical result dataset、checkpoint schema、
proposal/event/physics/result意味論は変えていない。
engine v21/compiled tile v5はP14-Pでouter worker poolを単一Numba thread teamへ置換した履歴revisionである。
v26はthread数非依存slab、preallocated workspace、stackless boundary BVH、同期single-owner writer、
exact/curved flat SoA wavefront、row numerical status、batch surface release、direct columnar replay、bounded
event/failure stagingへ実装を収束した試行revisionである。製品gateではregular 1Mの4-thread speedupが0.923xに
留まったため、P14-P closeoutのengine v27 / compiled tile v6 / runtime layout v5はthread APIと内部thread teamを削除し、
同じ数値意味論をsingle-thread compiled runtime一つで維持する。proposal v5はrow別target timeを同じ
`StepProposal`へ持たせ、event v11は同時刻候補に共有canonical nodeがないrowをrun-fatalにせず
`indeterminate_event`へfail-closedする。YAML case schemaだけをv2へ更新し、canonical HDF5 data、
result、checkpointのschema v1とphysics/result意味論は変更していない。

engine v28 / proposal v6はP15の`oml_stationary_maxwellian_debye_huckel_v1`を、RK4-first、その受入後の
explicit midpointの順で同じengineへ接続した。P15-Dは同じengineへ
`oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1`を追加し、compiled tile v8、physics catalog v6、
runtime v5、RK4 enclosure v2、exponential midpoint/enclosure v2、memory plan v11を記録する。
`maximum_dt_charge_lipschitz`は解決済み`dt * L_Z`であり、dynamic chargeでは0より大きく0.5以下でなければならない。
既存の`charge_number` state、frame/probe、boundary event、checkpoint列を再利用するため、case/result/checkpoint/event
schemaとdatasetは変更していない。P14-R remote CIはこのruntime変更とは別のrelease trackである。

engine v29 / proposal v7はB02の`ou_langevin`とcubic Hermite proposalを同じengine/event/output経路へ追加した履歴revisionである。
当時のphysics modelはCartesian XY、fixed charge、`epstein_linear_v1`、noise model `inertial_langevin_fdt`のrevision
`inertial_langevin_fdt_epstein_linear_frozen_start_v1`だけを許し、P18-H以前はterminal `stick`/`escape`以外を拒否した。
catalog v10、runtime v9、runtime layout v6、memory plan v12へ更新したが、compiled tile v11、event v11、geometry v5、
boundary v4、result algorithm v3、case schema v2、result/checkpoint schema 1と既存logical datasetは変更していない。
engine v30はOU covariance/split/meanの表現不能を粒子単位の`nonfinite_physics`へ局在し、正常rowを継続する。
result dataset、failure reason vocabulary、proposal、physics、event、schemaは変更していない。
P18-Lはengine v30とproposal v7を維持したままcatalog v14、runtime v13、compiled tile v15へ更新し、
velocity-dependent non-drag boundを扱うためexponential midpoint enclosureだけをv3へ更新した。exponential midpoint v2、
RK4 enclosure v2、event/runtime layout/memory planと永続schemaは不変である。
P18-Rは既存式ownerを共有するeffective-gas二revisionをcatalog v15、runtime v14、compiled tile v16へ追加した。
engine v30、proposal v7、integrator/enclosure、event/runtime layout/memory plan、case/result/checkpoint schemaは不変である。
COMSOL studyは再実行しておらず、保存audit statusをproduction resultへ埋め込まない。
P19-L完了時点はengine v31 / proposal v8 / event v12 / field location v4 / memory plan v13と
dense path `rk4_position_hermite_state_extension_v2`で、global-firstの
applicability認証にbounded local fallbackを追加した。P19-L完了時点では`rk4_global_abs_enclosure_v2`をsupport/event
authorityとして維持し、
result/checkpoint schema 1とlogical datasetを変えていない。runtime v16はP19-L性能snapshot、v17はDEP上限の
1 ULP外向き境界だけを変更した後続revisionである。runtime v18はcharge Jacobianをstage payloadへ追加し、
現行runtime v22はそのpayload意味を維持する。

後続M3-C1のevent v14はresult/checkpoint schema 1、logical dataset、event/final/frame payloadを変更しない。
manifestのevent revisionだけが`line_quadratic_rk4_axis_first_hit_v14`を記録する。global enclosureはshortened-stage、
field support、applicability、acceptance safetyのauthorityとして維持し、global supportを独立に証明済みのvalid
`rk4_dense` rowだけcurrent dense Bernstein boundをevent broad-phase query authorityにする。invalid dense boundまたは
global support未証明時はglobal queryへfallbackする。query/refinement/accepted/depth等は既存manifest countersに収まり、
新しいdatasetや診断traceを追加しない。

P18-Hのengine v32 / boundary v5 / result v4は、同じfirst-hit event経路へparameterなしの`hold`を追加した。
`held`はstuckのdeposition統計にもescapedのlogical-null統計にも含めず、hit位置、incident velocity、hit時電荷を
保持するinactive terminal lifecycleである。hit後のcharge、force、kinematicsは更新しない。既存event列とresident
particle stateを再利用し、lifecycle seriesの`held`列だけを追加したためresult/checkpoint schemaを2へ更新した。
case schema、canonical HDF5 data schema、proposal、integrator、event locator、memory planは変更していない。

現行dense path v3はroot始点相対のBernstein enclosureとTwoDiff残差を使い、world座標への下限・上限を
それぞれ`-inf` / `+inf`方向へ戻す。絶対座標の大きさに比例して誤差幅が肥大し、分割しても
event certificateが閉じない不具合を修正した。v3導入時のendpoint、path、event algorithm、engine v32、event v14、
RK4 global enclosure v2は不変だった。manifestは`rk4_dense_path_revision`だけにv3を記録し、
case/result/checkpoint schemaとlogical datasetを変更しない。P19-L証拠と既存resultがv2を記録するのは履歴として正しい。

Brownian resultのtop-level manifestは`brownian_rng_revision=philox4x32_10_brownian_interval_tree_v1`、
`joint_ou_revision=inertial_joint_ou_v1`、`joint_ou_split_revision=conditional_gaussian_half_split_v1`、
`brownian_tree_policy_revision=conditional_boundary_refinement_v1`、`brownian_interval_tree_depth`、
`brownian_adaptive_max_depth`を持つ。`random_draw_kinds`は`brownian_root_normal`と`brownian_split_normal`、
`resolved`は`brownian_coefficient_policy=macro_root_frozen_midpoint_v1`、noise model/revision、
`path_kind=cubic_hermite`を記録する。resume identityはBrownian RNG/OU/split/tree-policy revision、coefficient policy、base/max depth、
resolved noise model/revisionを含むため、不一致partialを再利用しない。root/split stream IDとpath kindはmanifest provenanceであり、
resume identityに同名keyを重複させない。checkpointは既存macro countと粒子stateを使い、mutable RNG cursorや新しい
Brownian datasetを追加しない。

現行Brownianはresult/checkpoint schemaを変更せず、noise revision
`inertial_langevin_fdt_epstein_linear_midpoint_2d_v2`、coefficient policy
`macro_root_frozen_midpoint_v1`、composition method `stochastic_exponential_midpoint_v1`をXY/RZ共通のresolved
model/algorithm provenanceとresume identityで識別する。現行proposal v10の`state_at()`はmidpoint-frozen
`G,J<=0`のaffine exponential chargeをroot始点から再評価し、保存済み始終点を厳密に戻す。
prepare済みinvariantを証明できない場合はfail-closedとする。別のcharge trace datasetは作らない。
axis hitではaccepted prefixの既存axis counterに加え、残時間の新しい`root_stochastic_interval`が
従来のBrownian RNG identityに反映される。mutable RNG cursorやaxis専用datasetは追加しない。
fixed chargeは同じaffine式の`G=J=0`へ退化する。旧XY frozen-start / 旧RZ projected revisionは現行入力として
受理せず、履歴resultがそれらを記録することだけが正しい。現行supersessionはengine v45 / proposal v10 / event v21 /
catalog v23 / runtime v22 / compiled tile v21 / memory plan v16である。
CPU layoutはv6のままで、B03 path arrayの静的な保守上限は`648 B/row`、受入上限は`2048 B/row`である。
正式characterizationは24/24実行を全粒子active・failure 0で完了した。計時とprocess RSSはmachine-localな
外部performance証跡[`evidence/b03/`](../evidence/b03/README.md)が所有し、portable result契約や合否閾値へ格上げしない。

B04のBrownian active wallは新しいdatasetやschemaを追加しない。`/events/boundary/*`は決定論runと
同じ`law_id`/`outcome`、hit時刻・位置、incident/response速度、charge、logical/physical ordinalを持つ。
terminal lawは既存lifecycle/terminal timeを更新し、active lawはhit位置とresponse速度から粒子を継続する。
wall counter RNGはphysical boundary ordinalで再構成でき、active hit後の残時間は次の
`root_stochastic_interval`ordinalがBrownian root RNG identityへ反映される。元rootの未使用cubic tailは永続化しない。
restart ordinalやstochastic treeの中間値をcheckpoint/resultの追加cursorとして保存せず、accepted macro barrierの
既存particle stateから再構成するため、checkpoint/resumeは通常runと全公開payloadが一致する。
B02初回sliceとB03初回closeoutの履歴resultの意味、数値payload、schema versionは変更しない。

result v5導入時はlogical datasetとschema 2を変えず、固定64 macro cadenceだけを置換した。有限半径接触は
同じdurability algorithmを維持したままfinal粒子とboundary eventへ`contact_radius_m`を追加し、result schemaを3へ
更新した。checkpointは接触半径を変更しないruntime authorityから再構成するためschema 2のままである。engineが
`W=macro_step_count+accepted_particle_pieces+candidate_queries+refinements`と
`T=max(2^20,128N)`（`N`はscheduled particle count）を所有し、accepted macro barrierでepoch開始時との差が
`T`以上、または最終macroならcommitを決める。
同じcadence objectをtop-level manifestとresume identityへ含めるため、異なるthreshold/revisionのpartialを再利用しない。
output schedule、frame/probe、slab幅は`W/T`へ入らない。`output.py`の同期single-owner writerはengineの決定後に
segment→inactive A/B checkpoint→`LATEST`をatomic順で永続化するだけで、cadenceを再決定しない。
現行result v6はresult schema 3のcanonical encodingとして`interaction_kind`、`destination_facet_id`、
`position_post_m`を全boundary eventへ必須化する。checkpoint schemaは2のままで、これらの列を欠く旧resultを
同じschema番号の別variantとして推測読込しない。

このdurability契約は同一local volume上でatomic replace/renameが働くprocess failure後の再実行を対象とします。
power lossに対するfsync durability、remote filesystemのatomicity、同じOUTへ書く複数process間の排他は保証しません。

`philox4x32_10_v2`のwall drawはmanifestのrequested seed、`particle_id`、zero-based physical boundary ordinal、
`random_draw_kinds`の`wall_probabilistic_stick`、`wall_maxwell_diffuse`、`wall_maxwell_normal`、
`wall_maxwell_tangential`から再構成する。logical `event_ordinal`はrelease後のwallとperiodic transferの両方で進むが、
physical boundary ordinalは材料wallだけで進み、periodic transferはwall RNG drawを消費しない。したがって外部readerが
event列から再構成する場合は、同じparticleについて先行する`interaction_kind == "wall"`行の件数を使い、
`event_ordinal-1`を使わない。局在refinement、residual split、trajectory scheduleもphysical ordinalを進めない。
法線thermal drawはopen uniform、接線thermal drawはstandard normalである。draw専用datasetやmutable RNG stateを
resultへ追加しない。

sourceはcanonical HDF5の粒子rowから再構成する。surface rowの`facet_id`と`facet_parameter`から位置を一度導出し、
保存済みvelocity、release時刻、物性を変更しない。確率分布を使う入力作成はsolver外で完了させるため、manifestに
source draw kindやsource RNG identityは存在せず、source用draw datasetも追加しない。

`resolved.path_kind`は無力場の`linear_exact`、証明済み一定加速度の`quadratic_exact`、一般
`rk4_reintegrated`、`exponential_midpoint_reintegrated`、Brownianの`cubic_hermite`を区別します。
`cubic_hermite`はbase depth `0..10`のOU interval treeを全rootへ一様に生成し、wall、RZ axis、または
証明不能候補だけをoptional max depthまで条件付き分割した各leafについて、OU endpoint位置・速度から定まる
numerical pathです。base=maxは従来固定depthへ退化します。材料eventとframe/probeは同じleaf pathを読みますが、
連続OU trajectoryのexact first-passage、zero miss probability、またはuniform max-depth pathを意味しません。
`quadratic_exact`はtopology-completeな材料boundaryと併用するcaseだけでなく、fully-supported regular field
box内のboundaryless caseにも使います。`rk4_reintegrated`は、証明済みのXY/RZ・fixed chargeまたはP15 continuous charge・
fully-supported `RegularLayout` caseに現れ、revision 3b/P06-RZではtopology-completeな材料boundaryとterminal
stick/escapeを組み合わせられ、engine v11では厳密内向きinterior-facet surface departureとsingle-facet activeな静止壁law後の残時間も扱います。
この意味論はengine v30でも不変です。
P06-Uは材料domain meshと完全一致するfully-supported P1/Q1にもこの組合せを
許可します。これらは利用者が選ぶ第二integratorではなく、同じ`rk4_fixed` engine内のpath意味論です。
boundaryless unstructuredはStage 1Aでも含めません。YAML case schemaはv3、canonical HDF5 data schemaはv3、
result schemaはv3で、上記の小さい
failure/series/probe datasetだけを追加し、diagnostic treeは作りません。`exponential_midpoint_reintegrated`は
利用者が選ぶintegratorであり、fixed chargeに加えてP15 continuous chargeを扱い、同じ`StepProposal.state_at()`と
event loopで材料wall/residual/RZ axisを扱います。engine v30はstage-evaluated RZ meridional pathもaxisでfoldし、
field vector metadataとaxis regularityをprepareで検査します。

`boundary_interactions.axis_crossings`はforce-free/force-coupledを問わず、canonical RZ stateをaxisでfoldした回数です。
axis foldはmaterial boundary eventではないため`/events/boundary`へ行を作らず、永続`event_ordinal`、physical boundary
ordinal、wall RNG draw、wall event count、interaction countを進めません。frameはaxis foldにも右連続で、同時刻には
`r=0`とfold後radial velocityを持ちます。

manifestの`maximum_dt_over_tau`は、dragなしresultでは0、linear dragではprepare時のglobal rate上限、
finite-speed Epsteinでは速度Jacobianのglobal固有値上限から解決した非零値です。選択methodにかかわらず
provenanceとして記録するが、`rk4_fixed`だけは`2.5`未満を要求する。`ou_langevin`はEpstein linearについて
prepare時の`gamma*dt<=1e6`と各macro-rootの有限な`0<gamma*h<=1e6`を要求し、RK4/exponential revision keyを
すべてnullにする。指数法のrunでは`exponential_midpoint_revision`と
`exponential_midpoint_enclosure_revision`を記録し、`rk4_enclosure_revision`はnullになる。RK4 runでは逆に指数二keyが
nullになり、選択していないmethodのrevisionを実行済みと装わない。
manifestの`maximum_dt_charge_lipschitz`はfixed chargeでは0、continuous chargeではprepareしたglobal
`dt * L_Z`上限であり、選択methodにかかわらず0.5以下を要求する。
`memory_plan` mappingは次を持ちます。

- `revision: solver_owned_memory_plan_v16`と`runtime_layout_revision: resident_soa_serial_slab_v6`
- `semantics`、`limit_bytes`、`planned_bytes`
- `geometry_preparation_transient_bytes`、`field_preparation_transient_bytes`
- `slab_particles`、`scratch_bytes_per_particle`、`event_work_bytes_per_particle`、
  `stochastic_tree_work_bytes_per_particle`、
  `dense_path_bytes_per_particle`、`certificate_work_bytes_per_particle`、
  `release_work_bytes_per_particle`、`replay_work_bytes`
- `event_candidate_capacity`、`event_staging_capacity`、`event_staging_bytes_per_row`、
  `event_staging_fixed_bytes`、`failure_staging_bytes_per_particle`
- `phase_peaks.load_case/prepare/run`
- canonical data、prepared geometry、schedule、resident state、P1/Q1 cell hint、active index、compiled physics runtime、probe index、
  output buffer、replay work、writer reserve、`geometry_query_scratch`、`slab_proposal_scratch`、`slab_event_work`、
  `slab_stochastic_tree_work`、`slab_dense_path`、`slab_certificate_work`、
  `slab_release_work`、`slab_event_staging`、`slab_failure_staging`、safety marginの`components`

memory plan v16の`stochastic_tree_work_bytes_per_particle`は、実際に境界候補になったrow数ではなく
`brownian_adaptive_max_depth`から`128*(D_max+4)`として解決する。従ってadaptive runでも予測peakは
到達可能な最深treeを保守的に含み、観測された候補率で過小評価しない。active wall/axis後のfresh-root stateは
同じslab幅のbounded SoA waveだけに保持し、前rootのgenerator/tree/proposalを解放してから次waveを進めるため、
反射回数分のroot scratchを同時保持しない。

このplanはsolverが所有する配列の予測peakであり、Python、HDF5、native library、allocatorを含む
OS process RSSのhard capではありません。最小slabが収まらなければ開始前に
拒否します。writerのbounded raw-chunk cacheとreserveはplanへ含みます。event/failureは固定容量SoA/CSR stagingを
各waveでcanonical順にflushし、総event数に比例するmacro-wide object stagingを持ちません。frame/probeは要求slotへ
direct columnar replayします。
`event_work_bytes_per_particle`はdeferred target/depth/interactionのint64 stack三列と、point/surface arbitrationの行別workを表します。periodicまたはparticle_centerのpoint viewでは後者も計画します。depth依存容量をproposal scratchへ隠さず、slab幅のfit判定へ含めます。byte式はengine/cpuのmemory-plan ownerとmanifestが所有します。
P19-Lを使う一般RK4では`dense_path_bytes_per_particle=176`です。`certificate_work_bytes_per_particle`は共有する
event split budgetで決まるinterval stackと、最大64 cellの候補arenaを含みます。split budget 2の現観測では
72 B/row + 544 B/row = 616 B/rowです。
`geometry_query_scratch`、event/failure staging、surface-release work、direct replayは別componentのまま保ちます。
pack時だけのgatherは12.5% safety marginが所有します。正確なbyte式とcapacity規則は
[`parallel_execution_plan.md`](parallel_execution_plan.md)が所有し、result formatはmanifest keyと永続意味だけを所有します。
fresh/warm RSSは外部performance scriptが別に測定します。

## 読み取りAPI

`open_result(path)`は`_SUCCESS`、complete manifest、schema version、`LATEST`までのclosed segment、必須列の
dtype/shape、frame/probe offsetの先頭0・単調性・終端を検査し、粒子値本体をまだ読まない`ResultView`を返します。
`open_result(path, recovery=True)`は完成resultがあれば同じ完成view、なければsibling `OUT.partial`について、
最新segment hashと全segmentの構造・累積countを検証したcommit prefixを返します。recovery manifestは
`status: recovery`で、未commit orphanやfinal tempを公開しません。

- `read_final()`：read-onlyなfinal particle列を返す。
- `read_release_events()`：read-onlyなrelease event列を返す。
- `iter_boundary_event_batches(batch_rows=4096)`：各batchのevent行数を上限内に保ち、segment/writer保存順で
  boundary eventとbatch-local candidate offsetを返す。batch間は物理時刻の大域sortを保証しないため、順序非依存の
  集計・可視化に使う。
- `read_boundary_events()`：上のbatch iteratorだけから全行を構成し、candidate offsetをlogical全体へrebaseして
  `(time_s, particle_id, event_ordinal)`順にsortしたread-onlyなboundary event列を返す。全表がmemoryへ収まる場合の
  convenience readerである。
- `read_failure_events()`：reason codeを含むread-onlyなparticle-local failure event列を返す。
- `read_lifecycle_series()`：macro-step終端の整数state countを返す。
- `iter_frames()`：segment index順にHDF5をframe単位で遅延走査する。
- `iter_probes()`：segment index順に選択particle stateをprobe時刻単位で遅延走査する。

未完了recovery viewのevent/series/frame/probe readerは確定prefixだけを返しますが、`read_final()`は
authoritative finalがないためerrorにします。

scientific集計、CSV/Parquet export、描画、V&V差分を`ResultView`へ追加しません。それらは別toolで同じ
logical resultを読みます。

open時のboundary candidate整合性検査も同じ固定行数で分割し、巨大event表を検査だけのために一括展開しません。
T03の`tools/analysis`と`tools/visualization`はpackage rootの`open_result`が返すこのviewだけを読み、solver resultへ
派生物を書き戻しません。派生物はsource manifest SHA-256、tool revision、parameterを持ち、source result外へ書きます。
analysisのarrival timeはrelease後の飛行時間ではなく、eventに保存されたsimulation絶対時刻です。analysis JSONはbatch行数を、
visualization metadataはbatch行数・最大描画点数・要求particle IDを記録します。

## 後続stageとの境界

P07の完了sliceは同じboundary eventへ反射・確率lawの行、一定加速度surface event、一般RK4の
厳密内向きsurface/single-facet active-boundary residualを追加した。tangent、facet端点/corner、または曖昧な一般RK4 contactはfail-closedである。
P06-Sは小さいphysics runtimeと明示Stokes--Cunninghamを完成し、P08はparticle-local failure、series/probe、
薄いCLIでStage 1Aをcloseした。P09はmemory/runtime layoutとphase memory plan、P10は同じproduction engineの
compiled field/physics/RK4 passを完了した。P11は同じengineへexponential midpointを追加し、P12は同じengineの
event-heavy workをworker-localに分配して完了した。P13は同じlogical event/frame/final意味論をmulti-segment、
checkpoint、recoveryへ拡張して完了した。P14の全performance matrixは別packageである。moving wallは後続gateに残る。
現在は任意分布をsolver外でrealizeしたinternal/surface tableを受け入れるが、
core内distribution engineは持たない。P14はそのmatrixを完了したが、result schema/datasetを変更していない。
P13/P14は未実装dataset、互換reader、migration frameworkをschema v1へ追加していない。
P15も既存のZ state/result/checkpoint列を再利用し、新しいdataset、reader、schema、migration frameworkを追加していない。
P18-Hは`held`をdeposition/escapeと分離するためresult/checkpoint schemaを2へ一度だけ更新し、旧schema用の
reader、writer、migration frameworkを併存させていない。
