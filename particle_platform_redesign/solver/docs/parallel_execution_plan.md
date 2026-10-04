# P14-P parallel execution closeout

## 1. 結論

P14-Pは複数thread機能を不採用として完了した。P14-P closeout時点の製品runtimeは
`deterministic_particle_engine_v27` / `compiled_cpu_tile_v6` /
`resident_soa_serial_slab_v5` のsingle-thread compiled engine一つである。P15-F完了時点のrevisionは
engine v28 / compiled tile v10 / physics catalog v8 / physics runtime v7であり、後続model追加も同じruntime layout v5と
単一直列実行原理を維持する。
case schema v2は`resources.memory_limit_mb`だけを持ち、旧`resources.threads`を未知keyとして拒否する。
result schema v1、物理式、first-event意味論、RNG key、bounded-memory方針は変更していない。

独立caseの同時実行はsolver外のprocess並列で行う。solver内部にはouter pool、Numba thread mask、
multiprocessing backend、自動tuner、experimental parallel flagを残さない。

## 2. 判断の根拠

P14のouter worker-waveは一部の大規模regular caseだけで改善したが、主用途を含むevent-heavy caseでは遅化し、
worker数に比例するscratchも増えた。

| P14履歴workload | 1 worker | 20 workers | speedup |
|---|---:|---:|---:|
| regular 100k | 0.783874 s | 0.417049 s | 1.8796x |
| regular 1M | 7.534041 s | 1.570727 s | 4.7965x |
| P1 cross-cell 10k | 0.747215 s | 0.636655 s | 1.1737x |
| Q1 cross-cell 10k | 0.894859 s | 0.779968 s | 1.1473x |
| exact event 10k × 20 hit | 33.509603 s | 37.728221 s | 0.8882x |

P14-Pではouter poolを削除し、thread数非依存slab、再利用workspace、stackless boundary BVH、row-target
flat SoA event wavefront、compiled boundary/Philox、direct columnar replay、bounded event/failure stagingへ
置換した。その上で一回だけprofile根拠のfocused correctionを行った。

regular 1M、4 macro-stepの同一machine warm実測は次だった。

| active threads | simulate |
|---:|---:|
| 1 | 9.32 s |
| 2 | 10.09 s |
| 4 | 10.10 s |

4-thread speedupは0.923xで、事前gateの2.0xを満たさない。新1-threadもP14のv20履歴値7.534 sより
約23.7%遅く、許容した10% regressionを超えた。一方、1M行のfield locator microkernelは
1/2/4 threadで0.03360/0.01686/0.00897 s、約3.75xだったため、thread team自体の故障ではない。

profileでは約9.42 sのうち、proposal約8.15 s、curved-path enclosure約4.17 s、RK4 step約3.98 sを占めた。
field locator等の並列leafより、Python/NumPyでのstage調停、配列操作、upper-bound bookkeepingが支配した。
enclosureだけを理想並列化してもAmdahl上の見込みは約1.47xで、4-thread gateには届かない。さらに融合kernelや
第二schedulerを積むことは、単純さ、拡張性、物理ownerの分離を損なうため採用しない。

## 3. 維持する成果

並列化の採否と無関係に、次は直列runtimeの有用な構造として維持する。

- resident SoA stateとmemory-limit由来のbounded slab
- field/physics/integratorの再利用workspaceと`*_into` leaf
- row別target timeによるrelease、hit prefix、残時間の統一
- exact/curved/axis/residualのflat SoA wavefront
- stackless boundary BVHとbounded candidate arena
- compiled boundary responseとparticle-ID/physical-ordinal基準のPhilox
- direct columnar frame/probe replayと同期single-owner writer
- event/failureのbounded stagingとstable physical-key order

これらは10^4～10^6粒子へのmemory scaling、Python object増殖の抑制、数値責務の明確化に寄与する。

## 4. 削除したもの

- `resources.threads`とthread-bearing case schema v1
- runtime thread mask、thread-team context、threading-layer manifest
- parallel acceptance driver `tests/performance/p14p_parallel.py`
- thread identityだけを検査するverification/scenario
- thread数別memory branch、worker別scratch、outer executor、future wave
- multithreading採用を示す現行文書とCLI例

履歴P12/P14の測定値は方式を再採用する根拠ではなく、不採用判断を再現する証拠としてだけ残す。

## 5. 現行runtimeと測定

`tests/performance/p14_matrix.py`は直列専用の外部harnessである。各観測を独立process・private Numba cacheで実行し、
`NUMBA_NUM_THREADS=1`を固定する。10k/100k/1M、regular/P1/Q1、initial/cross-cell、event量、output量、cold/warmを
測り、三公開API、科学payload digest、memory plan、revision、result shapeを検査する。thread scalingや
thread manifestは測らない。

P14-Pの測定artifactは履歴証拠であり、現行製品の受入harnessではない。保存済みartifactの値を再構成したり、
存在しないv20 payload digestを捏造しない。

直列化後のv27について、同じregular 1M・4 macro-stepを独立process、各1回warm-upで3回測定したsimulate時間は
10.249/10.273/10.100 s、median 10.249 sだった。v26の1-thread観測9.32 sより約10.0%、v20履歴値
7.534 sより約36.0%遅い。これはmultithreadingを復活させる理由ではない。P14-Uの別1M profileではself-timeが
events 28.7%、fields 26.8%、engine 11.8%、integrators 11.7%、runtime/dependency 11.7%へ分散し、単一ownerの
限定変更でend-to-end改善を説明できなかった。代表用途1M median `simulate`は`none` 550.43 s、sample 533.95 sで、
当該machineの製品制約として記録し、production変更は行わない。

## 6. 再検討条件

P14-Uのprofileはこの再検討条件を満たさなかった。今後の代表用途profileで、直列runtimeでは製品目標を満たせず、かつcompiled owner内の限定変更で
end-to-end改善を説明できる場合だけ、新しいwork packageとして再検討する。その時も先に利用case、対象phase、
科学identity、memory上限、削除条件を定める。次は再導入しない。

- case設定によるthread mask
- Python worker poolとNumba内部parallelの併存
- model/layout別kernel生成framework
- parallel/serial二つのproduction engine
- benchmarkにだけ効く巨大融合kernel

case sweepのprocess並列、cluster scheduler、GPUはsolver coreと別の実行層であり、このcloseoutを変更しない。

## 7. P14-P完了時点の次工程（履歴）

順序は次で固定する。

```text
P14-P serial convergence（完了）
  -> P14-U representative accuracy/utility/performance（完了）
  -> T03 analysis/visualization（完了）
  -> P14-R local release audit（完了、remote workflow初回成功は未完了）
  -> P15 stationary continuous charge（完了）
  -> P15-D finite-relative-drift continuous charge（完了）
  -> P15-E finite-speed Epstein revision
```

P14-Uは並列方式の選定ではなく、完成した単一直列runtimeを表面発生、非一様場、複数力、材料壁、多数stepで
評価して完了した。固定mesh上の時間収束、固定aspectの2D mesh収束、event failure、peak memory、出力量を同時に測った。
10k/100k/1Mのnone/sampleは各3 fresh process、1M profileは別の非計時runとし、mode間core／mode内probe payloadとevent workを
一致させる。profileが単一のproduction ownerを支配要因として示した場合だけ責務内の限定変更を検討し、
`runtime_or_dependency`が最大の場合や費用が複数ownerへ分散する場合は並列runtime再導入の根拠にしない。

## 8. M3-C2A closeoutと条件付き性能work package

上記P14-P/P14-U値は履歴である。現行productionはengine v36 / compiled tile v18 / proposal v10 / runtime v19 /
event v16 / memory plan v13である。M3-C2A common-P1 Case-P 100 nm finalは20 us、32+32独立seed、各287粒子、30 ms、
121 frameで完了した。登録済み83区分R-Z/fate gateは最大empirical TV `0.010670731707317093`、同時上限
`0.13119968456545308 < 0.15`で`PASS`した。終端gateはevent 0のため境界parityには情報を持たない。元Case-P `auxq`どおりの
二電流same-form結果であり、後続three-current、species-resolved物理、pathwise RNG、普遍的COMSOL同等性、boundary parityは認定しない。
比較は引き続き外部V&Vである。final candidate計時は4外部processが重なったため
`NON_AUTHORITATIVE_EXTERNAL_WORKLOAD_OVERLAP`であり、性能判断やCOMSOLとの速度比には使わない。

受理済みstep、Brownian tree depth、geometry tolerance、30 ms、全力場条件を変えないcandidate seed `319032`、`319047`、
`319063`の287粒子owner discoveryは完了した。各fresh childのwarm-up後に非profile計時と別`cProfile`を行い、すべてのrerunで
受理済み科学payload・work・case identity・revisionが完全一致した。3 seedの支配ownerは`integrators`、自己時間比は
42.58--42.86%、非profile public end-to-end wall medianは45.4668 sだった。ただし`integrators`は事前登録済みbounded ownerでなく、
automatic decisionは`owner_discovery_consistent=false`、`optimization_authorized=false`である。したがって本体変更はなく、
この結果を後付けで既存25% gateの合格へ読み替えない。authorityは
[`../evidence/m3c2/caseP_100nm_owner_profile_v1/`](../evidence/m3c2/caseP_100nm_owner_profile_v1/README.md)である。

完了したdiscoveryは三公開APIを通し、次を同時に記録した。

- macro root、OU leaf、accepted piece、candidate query、refinement、wall/axis/restart数
- field locate/sample、physics coefficient、OU/RNG/charge、event broad/localize、boundary、writerの時間比
- solver-owned memory plan、process peak RSS、artifact bytes、writer throughput
- 受理済み121-frame scheduleを固定した科学payload、RNG/event identity。slab独立性は既存P14/P20受入れを参照し、
  この287粒子profileでは再主張しない

M3-C2Aの終了状態と再開条件は[`../../implementation_plan.md`](../../implementation_plan.md)のcloseout authorityが所有する。
状態は`CLOSED_ACCEPTED_WITH_LIMITATIONS`であり、10,000粒子確認、100,000/1,000,000粒子scale、外部process並列は
benchmark完了条件でも自動的な次工程でもない。

製品利用者が対象hardware、粒子数、30 msの物理条件、output mode、許容wall time、peak RSSを明示した場合だけ、独立した
`accepted-workload performance` work packageを開始する。最初のworkloadは同じaccepted-accuracy設定の10,000粒子一件に固定する。
元table 287行を順番どおりcycleし、先頭287行、粒子ID、RNG addressing、revision、step、tree depth、toleranceを不変にする。
jitter、再標本化、model weight変更、accuracy変更は混ぜない。

低オーバーヘッド計測は外部harnessから`_proposal_chord_deviation_bounds`、`_proposal_event_bounds`、
`count_curved_event_candidates`、`locate_curved_first_event_batch`の四境界だけを一つのcurved-event処理鎖として計時する。
timerをproductionへ追加せず、計時overheadをend-to-endの1%未満とする。この鎖が非profile wall timeの25%以上を説明する場合だけ、
外向きenclosure、endpoint、candidate ordering、RNG/event意味を変えない一変更を一度評価する。同条件の3反復で科学payload・work identity、
wall/process CPU、peak RSSを比較し、改善が`max(5%, 3 MAD)`を超えなければrevertして終了する。通過しても次ownerへ移らない。

100,000/1,000,000粒子、no-output対121 frame、外部1/2/4 process並列は、最初の10,000粒子work packageから自動続行しない。
それぞれを要求するSLAがある時だけ別work packageを開く。solver内部thread pool、Numba parallel mask、serial/parallel二engine、
model別kernel framework、autotuner、benchmark専用巨大融合kernelは再導入しない。

### 受理済みCase-P chord follow-up（完了）

利用者の明示指示により、製品scale認定とは分離した一回限りのbounded maintenanceを実施した。owner discoveryで既に
12,000 batch / 3,444,000 accepted pieceに現れていた`curved_chord_deviation_bounds`だけを対象に、Pythonのrow×axis loopを
serial compiled batchへ置換し、event側の重複式とscalar helperを同じ変更で削除した。新しいtimer、設定、runtime、event経路、
threadingは追加していない。

同じaccepted seed 3件の非profile public end-to-end wall中央値は45.4668 sから39.8801 sへ12.29%短縮した。全seedで科学payload、
algorithm revision、1,500 macro / 3,444,000 OU leaf / accepted piece / candidate queryが完全一致し、failure/event/refinementも不変だった。
5%と3 baseline MADの大きい方を超えたため変更を保持し、このfollow-upは終了する。authorityは
[`../evidence/m3c2/caseP_100nm_chord_optimization_v1/`](../evidence/m3c2/caseP_100nm_chord_optimization_v1/README.md)である。

これは287粒子accepted workloadの実装効率改善であり、10,000粒子以上の製品throughput、portable speedup、正式COMSOL速度比を
認定しない。したがって上記SLA-backed 10,000粒子work packageを暗黙に開始せず、次owner、broad-phase再設計、内部並列へも
自動的に進まない。
