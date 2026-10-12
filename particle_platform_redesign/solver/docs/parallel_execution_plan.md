# CPU実行・性能：0.3.0

この文書は現行v62の実行構造、用途別の実測、source別の品質・配布資格を所有する。
数値methodは[`numerics.md`](numerics.md)、revision一覧は[`result_format_v3.md`](result_format_v3.md)、
COMSOL比較の科学的意味は[`vv_methodology.md`](../../vv_methodology.md)を参照する。
版ごとの変更は[`RELEASE_NOTES.md`](../RELEASE_NOTES.md)、今回のsource・配布入口の照合は
[0.3.0 branch検証記録](../../reviews/v0_3_branch_2026_10_13/source_qualification.json)を参照する。

同じ入力でもworker数を増やすと遅くなる用途がある。既定値は`workers=1`であり、
速度は粒子数だけでなく、活動粒子数、各macroの仕事量、境界負荷、出力量、起動・通信費に依存する。
測定したscopeを超えるCPU数比例効果、総メモリ削減、portable SLAは保証しない。

## 実行資源と同期

公開APIは`load_case`、`simulate`、`open_result`の三つである。
`simulate(case, output, *, workers=1)`とCLI `--workers`でprocess数を明示する。
caseの数値設定には混ぜず、旧`resources.threads`、自動worker縮小、tunerを持たない。
複数workerを使うPython scriptにはimport可能なmain moduleとmain guardが必要である。
利用例は[`README.md`](../README.md)が所有する。

| owner | 現行の責務 |
|---|---|
| 親engine | macro時計、release、活動粒子の整理、仕事配分、単一writer、checkpoint |
| 永続spawn worker | 排他的なparticle rowの既存stage/proposal/hit/prefix/wall law/remainder |
| 共有配列 | read-only prepared factsと、担当rowだけを更新するstate/accepted hint/replay |
| worker専用scratch | bounded slab、query、Brownian tree、event/failure staging |
| output reader | 保存順と区別した公開canonical event順 |

親は全workerのconnection/sentinelを監視し、完了したworkerへ次の排他的slabを渡す。
task幅はprepare時のmemory-fit容量を上限として、現在の活動粒子数と4W task目標から決める。
要求worker数は固定で、粒子数が減ってもslab幅を縮めて仕事を分配する。
serialと複数processは同じ数値処理を使う。一つのengineを維持し、追加JIT化は停止している。

**全workerの完了待ちは1 macroステップごとに固定され、同期間隔を変える公開設定はない。**
RK4 stageやBrownian leafごとの全worker同期は行わず、担当slab内で処理する。
各macroの全task/handoffと必要なoutput消費が完了してから、親が次macroへ進む。
起動時は全子のprepared/scratch準備完了、終了時は全子の終了を待つ。
workerはevent/failureを高々一batch送信し、親writerの同期消費後のACKを待つ。
このACKは送信したworkerだけの待機で、他workerは計算を続けられる。
保存batchは到着順を許し、全macro eventを保持する並べ替えbufferやworker側HDF5 writerを持たない。

workerは一つの`simulate`内でだけ永続し、別`simulate`へpoolを持ち越さない。
checkpointは毎macro保存ではなく、既存累積work閾値を満たした完了macroと最終macroでcommitする。
突然死・親例外時に未完macroのresident stateをdurable checkpointとして採用しない。
公開出力順、particle ID/物理ordinal基準のRNG、accepted-only commit、同method hit prefixを維持する。

## メモリ容量と寿命

総予算は共有入力・resident state・replay、親prepared原本、全worker専用scratch、
移送中の元配列とserialized handoff、writer、page padding、安全reserveを含む。
複数workerのscratchはstartup時の親・共有template・全子の重なりを覆うW+2組で保守的に計画する。
全子がprivate scratchをcopyしてreadyを返した後、親はtemplate rootを解放しprepared transportをsealする。
この寿命短縮を理由にstartupの容量会計は下げない。

非空replayはrun所有の一shared rootとworkerの永続readerを使う。
macro/slabのviewを再利用し、全consumer終了後にだけ解放・上書きする。
空replayの不要なmappingを省き、親writerや全resident stateをworkerごとに複製しない。
Brownian scratchは実際に到達し得る最大adaptive depthで計画し、観測したrefine数で予算を減らさない。

byte式、slab fit、12.5% safety margin、load/prepare/runのphase peakは
[`cpu.py`](../src/chamber_particles/cpu.py)のmemory-plan ownerと各run manifestが所有する。
`process_shared_transport`、`process_private_query`、`process_private_scratch_and_handoff`、
`process_handoff_serialization`、`process_transport_page_padding`を通常componentと区別して計上する。
event/failureの元stagingとcanonical payloadが生存中のserialization、particle-index commandを省略しない。
この配列予算はPython、LLVM/JIT、native libraryを含むOS process memoryのhard capではない。

観測では次の三値を区別する。

- solver-owned array plan：予算内の保守的容量予測。
- 同時parent＋live childのsampled private commit：process private容量の観測。100 ms間のpeakを見逃し得る。
- working-set合計：共有pageを重複計上し得る。unique物理RSSではない。

個々のprocess lifetime peakの和を同時peakとは呼ばない。共有入力やbounded slabの改善から
全用途の実メモリ削減、NUMA配置、compiler/library容量削減を推定しない。

## 計測の読み方

以下の時間はWindows/Core Ultra 7 265KF、20 physical/20 logical CPU、RAM約63.6 GiBのlocal観測である。
locked Python 3.12.13、NumPy 2.5.3、Numba 0.67.0、h5py 3.16.0を用い、各processのlibrary threadを1に固定した。
roleは順番に実行し、private cacheを分けたcold一回とwarm三回を保存している。
public時間は`load_case + simulate + open_result`で、worker起動・import・compile/cache・IPC・writer・shutdownを含む。
親の初回package importは計時外である。cache再利用から全dispatcherのhitを推定しない。
callbackを受ける既存advance三入口の`cache=False`方針と、そのchild compile費も維持している。

warm中央値比W1/W4の1未満は遅化である。秒数・倍率は入力、数値設定、出力量、hostと結合した観測で、
profile時間、品質検査時間、別revisionの値を性能値へ混ぜない。
小さい用途が遅くなる理由の全量を通信・同期だけへ帰属するprofileはまだない。

### 9.27 v62：完了順配分、一batch ACK、共有memory所有と公開前資格

測定C3と最終品質C4の27 production fileは同一で、source-map SHA256は
`49906998c297d23c9be8578fb8c5a757168fa8b7c2dddecc0695290c5df263a2`である。
C4の237 bound-file mapは`1f07b777ff209f9a09844d7b1c4f4babd482d5b7706333d7f1d48b68eef58a59`。
C3→C4の差は外部toolのsynthetic fixture二件のexpected revision literal五つのみで、production/core testは同一である。
[source binding](../evidence/process_dispatch_memory_2026_10_12/raw/r64/candidate_binding_c4.json)と
[確定verdict](../evidence/process_dispatch_memory_2026_10_12/raw/r64/verdict.json)が資格を所有する。
文書cleanup後のREADME/METADATA・配布artifactは、この歴史wheelと別に照合する。

主八用途は事前policyへ固定した41 role・164結果で科学identity、件数、配列予算、観測memory、cleanupを確認した。
別scopeの時間変化場五role・20結果も完了し、主scopeへ合算して性能gateを救済していない。
全raw sample・C1/C2/C4/C8/C20・baseline比・cold値は
[保存inventory](../evidence/process_dispatch_memory_2026_10_12/inventory.json)とcase別science receiptへ保持する。
表は同じ最終C3内のpublic warm中央値である。

| 用途 | 粒子数 | W1 (s) | W4 (s) | W8 (s) | W1/W4 |
|---|---:|---:|---:|---:|---:|
| P14-U、32 macro、軌道出力なし | 100,000 | 62.221750 | 20.726649 | 13.362965 | 3.0020x |
| B03 continuous | 100,000 | 22.966331 | 10.413016 | 8.269923 | 2.2055x |
| P14-U、32 macro、軌道出力なし | 10,000 | 6.109820 | 5.930617 | — | 1.0302x |
| P14-U、4時刻全粒子frame＋32-ID probe | 100,000 | 62.067107 | 20.730623 | 14.021615 | 2.9940x |
| event負荷を前半IDへ集中 | 10,000 | 2.386702 | 2.470163 | 3.309200 | 0.9662x |
| event負荷を分散 | 10,000 | 2.519154 | 2.603649 | 3.764139 | 0.9675x |
| early hold、16 macro | 100,000 | 2.153782 | 2.486062 | 2.997173 | 0.8663x |
| regular、4 macro | 1,000,000 | 25.271948 | 10.750538 | 8.553300 | 2.3508x |
| 時間場＋非格子snapshot/出力、4 macro（別scope） | 10,000 | 0.311757 | 3.557117 | 4.382606 | 0.0876x |

元bulk二primaryのW1/W4は事前2x条件を満たし、保存v61と同数workerで比較した非劣化条件も合格した。
境界集中/分散caseは同じ100,000 eventを処理し、旧coordinator比で改善したが、W1よりW4は遅かった。
regular 1MのW20は9.866913 sでW8より15.36%遅い。全CPU数比例、境界負荷の完全均等化を認定しない。
時間変化場の全20結果は科学・非劣化条件を満たすが、並列高速化は認定しない。

memoryの総改善も認定しない。主164結果のsampled private最大はregular1M/W20の10,231,394,304 B、
同roleのarray planは1,345,378,574 Bである。事前primaryの8 GiB観測上限と追加scopeの32 GiB上限内だった。
P14-U100k/W4のarray planは旧source比14.64%増えており、より完全な同時live会計と実メモリ変化を区別する。

| local環境 | 標準品質 | 外部tool | 利用・配布入口 |
|---|---|---|---|
| Windows | C4 static六gate PASS。C3 core944 passed・1 warningをproduction/core test同一と明示して保持 | C4 full tools495 passed | Quick Start六操作、P14 smoke八row、runtime-only installed wheelのAPI/console/module CLI workers2 PASS |
| Linux/WSL | C4 static六gate、core944 passed・1 warning | portable suites＋V&V120 passed | 同じ六操作・八row・実installed-wheel workers2入口 PASS |

[Windows receipt](../evidence/process_dispatch_memory_2026_10_12/raw/r64/windows_release.json)、
[Linux receipt](../evidence/process_dispatch_memory_2026_10_12/raw/r64/Linux_release.json)、
[最終quality receipt](../evidence/process_dispatch_memory_2026_10_12/raw/r64/quality_final_c4.json)へ結合したlocal資格である。
Windows wheelはC3時点の実行receiptを、production/lock/project/core test/最終READMEの一致を明示して保持する。
両OSともsource/wheel/installedの27 Python payload、README由来METADATA、lockを照合し、
serial対workers2の全public science一致、API終了後live child0、CLI正常終了・timeout内完了を確認した。
Linux/WSLの四logical CPU機能資格をWindows20workerのscaling・NUMAへ転用しない。
local実行からremote CI成功やpackage registry公開を推定しない。

突然死回帰では第二macroの境界handoff直後に子を終了させ、未完result非公開、最後のdurable prefix回復、
全子/shared mapping cleanup、workers2→4 resumeとserial science一致を確認した。
Windows full toolsの初回492 passed・1 failed・2 errorsは保存し、revision fixture修正後の495件と区別する。
licensed COMSOL solveはこのOS/tool資格に含まれない。

### 9.28 Case A/P Brownian full-stepの同一run並列評価（R65）

**287粒子・1500 macroの両caseではworkers1を使う。** 共通canonical P1場、RZ、100 nm、
0～30 ms、dt20 µs、OU base depth3、121保存時刻、連続二電流帯電と全物理寄与を固定した。
各case32 current serial seedと保存native32 seedの人口比較は別評価で、科学的範囲はV&V文書が所有する。

| full-step case | W1 warm (s) | W2 warm (s) | W4 warm (s) | W1/W4 | cold W1/W2/W4 (s) |
|---|---:|---:|---:|---:|---|
| Case A | 13.925685 | 33.971821 | 32.248288 | 0.431827x | 89.994977 / 98.994212 / 105.491089 |
| Case P | 19.690740 | 33.967174 | 33.873763 | 0.581298x | 85.304696 / 97.837806 / 100.365778 |

warmは事前固定した各三seedの記述的中央値で、同一seed三反復の有意差試験ではない。
24性能結果と対応serial六結果の全emitted science/frame/probeを既存readerで再読して一致し、failureは0だった。
複数processのidentity確認は各三seedに限り、全32 seed並列campaignではない。
Case Pは最後まで287粒子が活動し3,444,000 accepted piece/queryを処理した。
Case Aは終端後の通常更新を停止し、選定三seedのaccepted pieceは約164万、wall eventは260～266だった。
多macroだけでは、各macroの配分・完了待ちを十分償却できるとは限らない。

sampled private最大（cold/warm込み）はAのW1/W2/W4で0.908/1.764/3.227 GB、Pで0.840/1.699/3.155 GBだった。
owned array planは両case54,235,788/263,752,671/395,655,447 Bで、並列化による実メモリ改善を認定しない。
中断したPのseed939043は公開APIで再開後、追加fresh full runと全science一致を確認した。
この時間は性能matrixへ混ぜていない。

[確定verdict](../evidence/comsol_brownian_fullstep_parallel_2026_10_12/raw/r65/verdict.json)、
[全性能比較](../evidence/comsol_brownian_fullstep_parallel_2026_10_12/raw/r65/performance_comparison.json)、
[保存証拠](../evidence/comsol_brownian_fullstep_parallel_2026_10_12/README.md)を参照する。
保存COMSOL人口gateは両case PASSだが、完全native RHS、詳細境界応答、時間発展v/Z ensemble、native FE、
continuous SDE、普遍的COMSOL同等性は未認証である。新COMSOL solveや未使用seed確認ではない。
Case Pは元の二電流same-formで負イオン電流を含まず、zero壁eventからwall law合格を主張しない。
独立seedのsolver外並列throughputは今回測定していない。

### 9.29 P14-U 1万粒子・320ステップの1/4 process実測（2026-10-13）

**この入力では4 processが1.964倍速く、約49.1%短縮した。** XY regular32×4場、固定電荷、
Epstein＋Coulomb＋重力、Brownianなし、trajectory/probeなし、seed0である。
0～4秒をdt0.0125秒で320 macro実行した。32-step入力からdtだけ変更しcanonical HDF5は同一byteである。

| workers | warm中央値 (s) | warm三回範囲 (s) | cold (s) | warm総process CPU中央値 (s) | warm最大private (GiB) |
|---|---:|---|---:|---:|---:|
| 1 | 27.361519 | 27.342829～27.426571 | 85.930094 | 31.828125 | 0.772 |
| 4 | 13.929350 | 13.804737～14.148069 | 76.004645 | 52.000000 | 0.802 |

各role空cacheのcold一回と同じseed/inputのwarm三回、全八結果の保存science・件数・workが一致した。
failure0、境界event/最終stuck10,000、accepted piece3,144,057、query3,304,830、refinement160,773である。
frame/probeは0件なので、未保存の全内部軌道まで直接照合したとは主張しない。
warm0の活動粒子を含むmacro区間は309、全粒子終端後の保持は11区間で、活動粒子×区間数は3,058,066だった。
これをintegrator stage数とは呼ばない。

cold込み最大privateはW1/W4で0.772/2.723 GiB、array planは135,443,535/432,406,016 Bだった。
warmだけの小さい増加からcoldメモリ改善を推定しない。32-step、Brownian、出力あり、他CPU数へ倍率を外挿しない。
[測定要約](../evidence/p14_10000_320step_process_2026_10_13/summary.json)、
[全八結果照合](../evidence/p14_10000_320step_process_2026_10_13/comparison.json)、
[条件と保存証拠](../evidence/p14_10000_320step_process_2026_10_13/README.md)へ結合する。

## 将来項目

2026-10-13の利用者指示により、複数macroをworker内で連続実行する同期間隔設定は将来項目に留める。
現行0.3.0には追加しない。検討時は数値dtと独立な最大連続macro数を扱い、既定1とのscience一致、
release・出力・人口集計・checkpoint、bounded replay容量、負荷偏りを同じ入力で評価する。
粒子間相互作用や場feedbackを追加する場合は、その連成周期に合わせて同期上限を制限する。
現在の速度観測だけから設定値や高速化倍率を決めない。

対象用途/hostが明確になった時に、独立seed並列、長期間/追加出力、hybrid/SMT/NUMAを別scopeで評価する。
追加JIT、旧native team、第二scheduler、model別engineを今回のrelease要件に追加しない。

## 不採用判断と履歴の保存先

P14-PのNumba内部thread試行はregular1Mで1/2/4 threadが9.32/10.09/10.10 s、4-thread 0.923xで不採用だった。
locator単体約3.75xをend-to-end改善へ変換できず、thread mask、旧outer worker-wave、`resources.threads`を削除した。
bounded SoA、再利用workspace、flat event wavefront、compiled geometry/RNGは現在の単一engineへ維持する。

旧v60 process試行もB03 20kのW1/W4 0.7747xで不採用で、後続v61/v62の大規模合格で救済していない。
v62の`count_nonzero`→`bincount`案は遅化して撤回した。
[v60否定receipt](../evidence/process_particle_slabs_2026_10_12/raw/r63/trial_v60_stop.json)、
[lifecycle否定receipt](../evidence/process_dispatch_memory_2026_10_12/raw/r64/lifecycle_rejected.json)へ保持する。

過去revisionの設計案、source別測定、profile、cache試行、段階的進捗は現行説明へ重ねず、
[凍結した旧全文](../evidence/comsol_brownian_fullstep_parallel_2026_10_12/current_docs/solver/docs/parallel_execution_plan.md)
とそのリンク先のimmutable evidence、Git履歴で保存する。保存値のrevision・入力・採否を現行sourceへ付け替えない。
