# 再構成前後の性能観測

8条件のcomplete smokeと、既存R66の同じ入力による1万粒子・320macroを再実行した。全8条件のcore/full digestと、全process観測の7系統science payloadは完全一致した。失敗・残存workerは0。

| 条件 | 旧simulate [s] | 新simulate [s] | 旧peak RSS [MiB] | 新peak RSS [MiB] |
|---|---:|---:|---:|---:|
| event-20 | 0.111970 | 0.165336 | 484.66 | 484.82 |
| p1-cross | 0.054852 | 0.071660 | 742.36 | 751.05 |
| p1-initial | 0.053969 | 0.055984 | 742.95 | 751.72 |
| q1-cross | 0.048719 | 0.061193 | 744.40 | 749.88 |
| q1-initial | 0.044212 | 0.059822 | 740.21 | 750.41 |
| regular-cold | 32.857075 | 34.553693 | 611.73 | 572.59 |
| regular-time | 0.050002 | 0.059508 | 596.53 | 571.60 |
| regular-warm-all | 0.048028 | 0.057891 | 608.87 | 572.45 |

短いwarm runは約44–112msから約56–165msへ増えた条件がある。result4のcontext保存・検査を含む機能差を含めた比較で、各処理の寄与は分離していない。旧matrix時には一部実装試験が並行していたため、表の比だけで性能退行・改善を認定しない。RSSはPython/native/JITを含み、owned-array予算と異なる。

| 1万粒子/320macro | 旧public [s] | 新public [s] |
|---|---:|---:|
| workers=1 | 28.487 | 29.913 |
| workers=4 | 18.310 | 18.184 |

こちらは他のsimulation/testを止め、baseline→currentを逐次実行した。時間はload+simulate（spawn/出力込み）+openで、science hashingは除く。各sourceでworkers1を一度warmup後、各worker数を1回観測した。新1-workerは約5%長く、新4-workerはほぼ同じだった。反復統計や4倍速の保証にはしない。

[数値・条件・SHA](comparison.json)、[旧matrix](baseline_matrix.json)、[新matrix](current_matrix.json)、[旧process](baseline_process.json)、[新process](current_process.json)を参照する。旧harnessのstale revision固定による最初の拒否は[結線記録](baseline_harness_binding.json)に残し、数値packageは変更していない。

generic harnessのrevisionはowner定数を読む。P10/P11はcache時にPython登録が0になり得るnested kernel counterを、実際のcompiled advance dispatcherへ変更し、report v2とした。旧campaignの固定source guardsは維持した。

matrixは現行cloneで`uv run --locked python -m tests.performance.p14_matrix --suite smoke`から再生成できる。processの元case/dataと全結果は外部`.artifacts/r66_2026_10_13/`等を必要とし、receiptのpathは配布リンクではない。完全10^5/10^6 release matrix、厳密なscaling資格、新native計算は実施していない。
