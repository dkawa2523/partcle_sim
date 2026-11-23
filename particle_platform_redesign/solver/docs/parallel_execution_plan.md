# CPU実行・性能

この文書は現行の単一engineの実行資源、同期、memory、性能計測の解釈を所有する。
数値methodは[`numerics.md`](numerics.md)、利用者の入口は[`README.md`](../README.md)、
benchmarkの選択とcommandは[`tests/performance/README.md`](../tests/performance/README.md)を参照する。

## 実行資源と責務

`simulate(case, output, *, workers=1)`またはCLI `--workers`でprocess数を指定する。
既定値は1であり、caseの物理・数値設定とは独立である。Pythonから複数workerを使うscriptは
import可能なmain moduleとmain guardを持つ。

| owner | 責務 |
|---|---|
| 親engine | prepare、macro時計、release、活動粒子整理、single writer、checkpoint |
| execution | spawn worker、排他的slab配分、共有transport、bounded IPC/ACK |
| transport | serial/process共通のstage、proposal、hit prefix、wall law、残時間処理 |
| worker | 担当particle rowのstate/hint更新、bounded private scratch |
| output reader | 保存batch順と区別した公開canonical event順 |

親は全workerのconnection/sentinelを監視し、完了したworkerへ次のslabを渡す。
task幅はprepare時のmemory-fit容量を上限に、活動粒子数と4W task目標から決める。
要求worker数は固定で、各macroの全task/handoffとoutput消費を終えて次macroへ進む。
同期は**1 macroごと**であり、RK4 stageやBrownian leafごとの全worker同期はない。

workerはevent/failureを高々一batch送信し、親writerの同期消費後のACKを待つ。
保存batchは到着順を許し、公開readerがcanonical順を解決する。worker側writerや
macro-wide event並べ替えbufferはない。process teamは一つの`simulate`内で永続する。

checkpointは累積work閾値を満たした完了macroと最終macroでcommitする。
突然死・親例外による未完macroをdurable prefixへ採用しない。particle ID/物理ordinalのRNG、
accepted-only state、同じmethodのhit状態、出力時刻に依存しないpathを維持する。

## メモリ容量と寿命

総予算は共有入力・resident state・replay、親prepared原本、全worker専用scratch、
移送元とserialized handoff、writer、page padding、安全reserveを含む。
startupのscratch重なりはW+2組として保守的に計画する。全workerのready後に親templateを解放し、
prepared transportをsealする。非空replayはrun所有の一shared rootと永続readerを使う。

容量式、slab fit、安全marginとphase peakは[`cpu.py`](../src/chamber_particles/cpu.py)が所有し、
各runのmemory planに保存する。Brownian scratchは最大到達adaptive depthで計画する。
この予算はPython、LLVM/JIT、native libraryを含むOS process memoryのhard capではない。

- owned-array planはsolver配列の保守的容量である。
- 同時parent＋live childのsampled private commitは観測間のpeakを見逃し得る。
- working-set合計は共有pageを重複計上し、unique物理RSSとは異なる。

個々のprocess lifetime peakの和を同時peakと解釈しない。

## 現行の測定経路

`tests.performance.p14_matrix`はlayout、event、出力、cold/warmと粒子数の直交presetを、
`p14u_representative`は非一様場・surface release・targetとtime/mesh refinement、RZ検証を所有する。
release presetの10k/100k/1Mと、選択行だけの実行は区別する。
`tests.performance.process_workers`は同じ生成入力でworker数を変えて公開APIを実行し、
none/sample出力を選ぶ。sampleは既存P14-UのID・時刻policyを使う。
入力作成・warmup、実測、独立したtarget/outcome検査、science比較を分離し、生成入力を残す。

計測は`load_case + simulate + open_result`で、spawn、import、compile/cache、IPC、writer、
shutdownを含む。process計測ではmanifestのpayload roster全byteの独立SHA-256読取、
outcome検査・読取、science hashing・読取、明示的`verify_contents()`を個別に記録し、
`public_end_to_end`の外へ置く。独立した入力準備も計時外である。
`open_result`自身の検証は公開API区間に含まれる。result4/5のopen・integrity policyの違いを、
kernel演算だけのspeedupへ読み替えない。
一つのserial warmupはpersistent disk cacheを温めるが、後続`simulate`のworker起動費は除外しない。
source/input hash、条件、全sample、全emitted scienceを記録する。

P14はfresh processとprivate cacheでcold/warmを区別し、P14-Uは各workerの32粒子warmup後の
公開APIを計時する。独立profileはthroughputへ含めない。P14-U clientは短いprivate cache名と
計時後の返却observationのdurable sidecarを使い、独立profileでは完了した最大粒子数none/r0の
同一source cacheを再利用する。既存warmupとfull profileは実行し、timed cacheはfreshのままとする。

全payload一致はworker数による再現性の検査である。独立した解析解・target条件・収束検査と
区別し、worker一致だけで精度を認定しない。failure0、release/terminal ledger、final、
lifecycle、全frame/probeを確認し、unsaved軌道やcontainer byteの一致は主張しない。

## 結果の読み方

小さいcohort、多macroでも仕事量の少ないmacro、eventやoutputの多い用途では、複数workerが
遅くなることがある。粒子数だけで推奨worker数を決めない。比較は同じ入力、dt、tree depth、
output、host、cache条件で行い、実測scopeを超えるCPU数比例やportable SLAを推定しない。
数値resolutionを粗くした速さを同一精度の実装speedupへ置き換えない。

現行0.5 productionの実行条件、全行の時間・memory・入力／science identity、scopeは
[`foundation performance acceptance`](../evidence/foundation_2026_11_23/performance_acceptance.md)と
[機械可読receipt](../evidence/foundation_2026_11_23/performance_acceptance.json)が所有する。
P14はbaseline各行1回／current各行3回、P14-Uは両versionとも各count/modeを3回、
processは10k・1/4 worker・none/sampleを各3回実行した。processの10k観測を大規模process matrixへ拡張しない。

これらは共有host上で他のbenchmark、A/P、wheelや検証が一部重なった観測であり、
host scheduling/loadを隔離したmatched timing実験ではない。入力・数値設定・sourceの対応と
saved scienceの一致を確認したが、時間比だけから因果的なspeedupやregressionは認定しない。
owned-array plan、単一processのhigh-water RSS、whole-team memoryを区別する。

currentの87 timed observationsは、全実行前／実行後に確認した59-file runtime/helper/configへ束縛する。
51 production sourceを含む全59値はarchived 179-file inventoryと一致し、そのinventoryの実行後値も一致する。
179-file inventory自体の採取は最初のprocess W1計時中であり、全実行前のsnapshotとは扱わない。
後続のWindows cache path／observation保存修正は180-file clientの実smoke・fault検査で確認し、
51 production sourceとwheel実装の同一byteを保持する。修正後clientでfull 1M matrixを再実行したとは扱わない。
以前の[`resolved_semantics`](../evidence/resolved_semantics_2026_11_23/README.md)は過去sourceの記録として保持し、
その実測値・資格を現行の性能保証へ転用しない。
