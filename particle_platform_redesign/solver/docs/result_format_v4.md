# Result format v4

この文書は0.4.0のresult schema 4、checkpoint配列schema 2の永続resultと
`ResultView`の契約です。軌道計算の入力契約は
[`case_format_v3.md`](case_format_v3.md)が所有し、集計、可視化、COMSOL比較はこのresultを読むcore外の
toolが所有します。

## v4成果物の物理配置

正常終了したresultは次のdurable layoutを持ちます。segment数はengineがaccepted macro barrierで判定する
`cumulative_solver_work_v1` cadenceと最終macroから決まります。

```text
OUT/
  run.json
  context.h5
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
上書きしません。workerはwriterを持たず、accepted prefixのイベントを親へ渡します。

matching resume identityを持つ`OUT.partial/`があれば`simulate(case, OUT)`が`LATEST`から自動再開します。
identity不一致、未知version、参照artifactのhash/構造破損はfail-closedで、checkpoint migrationや互換readerを
持ちません。最初の`LATEST`以前に失敗した同一identityのpartialはcommit済みstateを持たないため、epoch 0を
初期状態から正確に再実行し、orphan temp/final名を置換します。`open_result(OUT, recovery=True)`は未完了partialの
`LATEST`までをread-onlyに開きますが、finalはauthoritativeでないため`read_final()`を拒否します。
final、complete manifest、`_SUCCESS`、directory renameの途中で失敗したpartialは、次の同一identityの`simulate`が
検査して最終化を再実行します。`_SUCCESS`まで確定したpartialは再計算せず完成directoryへ公開します。

Windowsの完成directory公開に限り、renameのWinError 5/32/33は25 ms間隔・最大1 sのI/O期限で再試行します。
finalizeと完成partialの再公開は同じ処理を使い、成功は実際のatomic renameによってのみ確定します。
出力先の出現、partialの消失、他のerrorと他OSの拒否は再試行しません。恒常的な拒否は元のerrorで失敗し、
完成partialを保持します。この期限は粒子の時間刻みや数値許容値ではなく、物理計算を再実行するものでもありません。

## 必須のimmutable context

`context.h5`は既存canonical DataBundle形式のgeometry-only bundleで、layouts/fields/sources配列は空です。
完全なGeometryData（volume cells、boundary owner/orientation、group/material/external IDs）を保持します。
producer_metadataにはcontext revision、元case/dataのhash、motion/axis/SI、resolved sourceのID・名前・table・種別、
boundary groupのpriority/law/contact/parameters、periodic mappingを保存します。source所属の再採番や初期全配列の複製はしません。

親single writerが初回epochより前にatomic作成し、logical content hashとfile SHA-256をrun manifest、resume identity、checkpointへbindします。
resume/recoveryは同じcontextを検証し、必須の親case/data identityとcanonical identityの再hashも照合します。
欠落・改変・親identity不一致を拒否し、再生成や上書きで欠損を隠しません。
初回commit前でも有効なcontextが必要で、contextを書き終わる前に中断した出力は新規の出力名からやり直します。
readerの検証中に必要なgeometry一時配列もprepare/writer峰値へ計上し、writerが保持し続けるのはhash bindingだけです。

結果だけで形状付き表示・source/boundaryの意味の確認ができますが、背景場を持たないcontextだけで元計算を再実行することはできません。
旧result3のcompleted/partialは受理せず、checkpoint配列schemaが2でもresult version4を要求します。

`ResultView.manifest`はobjectをread-only mapping、arrayをtupleとする再帰的なimmutable snapshotです。
serializationや利用者の編集には`manifest_document()`で独立したJSON documentを取得します。
`manifest_sha256`は読込み時のcanonical JSON（UTF-8、sorted keys、compact separators）へbindし、
`source_identity()`は派生結果の共通receiptを返します。返されたdocument/receiptの編集は元のviewへ影響しません。
粒子lifecycleのcode/name/basic terminal分類は`chamber_particles.lifecycle`が所有し、
集計上の付着、first wall hit、field support、数値精度やrun integrityとは区別します。

## `segments/epoch-NNNNNN.h5`

root attributeは次のとおりです。

| attribute | 値 |
|---|---|
| `result_schema_version` | integer `4` |
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
意味論のauthorityにしません。同時hit candidateは各event内でfacet ID順です。event v23はevent v19 / v18と同じくexact、一般RK4、
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

永続particle列の型、shape、finite条件は`cpu.PersistentParticleState`の一つの宣言が所有し、
resident `ParticleState`、checkpointへのprojection、HDF5の列I/O・layout検査が同じ宣言を使います。
`CheckpointState`はこのparticle snapshotと親時計・cursor・統計・field hintを組み合わせます。
active maskは保存せず、sorted active indexから再構成してlifecycleへ照合します。
writerによる構造検査と、engineによるschedule/cursor/hintの意味検査は別の責務です。

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
検出はこの形式の保証範囲外です。通常の完成resultはcheckpointを科学datasetのauthorityにせず、closed segment、complete
manifest、`final.h5`を検査して読みます。

## `final.h5`

rootの`result_schema_version` attributeは4です。`/particles`は常に全resident粒子を`particle_id`順で
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
- requested/resolved integrator、backend、seed、path kind、
  physics model revision、required field binding
- contextのpath、canonical content hash、file SHA-256
- source ID/name/type、解決済みboundary law・priority・fallback law・反発係数・付着確率
- start/end/dt、親時計の`macro_grid_revision`、field snapshotによる追加macro split時刻、YAML順の`source_id_to_name`
- resolved event設定。一般RK4材料boundary経路ではaccepted particle-piece、candidate query、refinement、最大深さの集約
- `numerical_work_policy`：proposal/certificate/coefficient guard/residualの解決済みwork上限、容量、停止意味。
  既存event設定の用途を区別するもので、実際の精度や到達depthを表さない
- `prepared_stability_upper_bounds`：要求base dtと全必須field域の準備済み上界による
  `dt_times_drag_velocity_lipschitz`、`dt_times_charge_lipschitz`。
  実accepted stepで観測した最大値や精度推定ではなく、field knots等で実区間が短くても保守的gateを緩和しない
- wall event、residual split、RZ axis crossingの集約
- solver-owned memory plan revision、semantics、limit/planned bytes、slab幅、phase peak、component内訳
- 実行資源のworkers/process policy。数値設定、particle IDやRNG keyとは分けて記録する
- particle、release/boundary/failure event、series、frame/frame row、probe/probe row、macro stepの件数
- lifecycle counts、failure reasonのname-to-code mappingとreason別件数

## 読み取りAPI

`open_result(path)`は`_SUCCESS`、complete manifest、schema version、`LATEST`までのclosed segment、必須列の
dtype/shape、frame/probe offsetの先頭0・単調性・終端を検査し、粒子値本体をまだ読まない`ResultView`を返します。
`open_result(path, recovery=True)`は完成resultがあれば同じ完成view、なければsibling `OUT.partial`について、
最新segment hashと全segmentの構造・累積countを検証したcommit prefixを返します。recovery manifestは
`status: recovery`で、未commit orphanやfinal tempを公開しません。

- `read_context()`：readonlyな型付きgeometry・座標・source/boundary・topology意味を返す。
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


open時のevent構造検査もbounded行数で行い、巨大表を検査だけのために一括展開しません。
解析・export・描画は`chamber_particles_tools`が公開ResultViewを読み、source resultへ派生物を書き戻しません。
JSON/CSV/SVGは入力manifest/context hash、client revision、選択ID・分母・単位・描画上限を記録します。
全interaction、粒子ごとのfirst hit、terminal停止、保存frame占有を区別し、欠損をescapeや0へ補完しません。

## 変更と履歴

科学的な式・RNG・event順・永続列を変える時は該当ownerのrevision/schemaを更新します。
code移動だけで科学値を変えず、contextの追加と科学payloadの比較は別に確認します。
実runのmanifestが解決済みrevisionのauthorityであり、本書へ全rosterを複製しません。
旧形式・P00〜P22の経緯と資格はGit履歴と元sourceに固定したevidenceへ保持します。
