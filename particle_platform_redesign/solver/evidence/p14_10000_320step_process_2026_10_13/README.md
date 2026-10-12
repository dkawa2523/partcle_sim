# P14-U 10,000粒子・320ステップ：1/4 process実測

**4 processの方が速い。warm中央値の速度比 W1/W4 は 1.964倍。** 全8結果の科学出力は既存held readerで一致を確認済みです。

XY、regular 32×4場、固定電荷、Epstein抵抗＋Coulomb＋重力、Brownianなし。0～4秒をdt=0.0125秒で320 macrosteps積分しました。既存32ステップ入力から変更したのはdtだけで、canonical HDF5はbyte同一です。trajectory/probe出力はありません。

| process数 | warm中央値 秒 | warm3回の範囲 秒 | cold 秒 | warm総process CPU中央値 秒 | warm最大private GiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 27.362 | 27.343–27.427 | 85.930 | 31.828 | 0.772 |
| 4 | 13.929 | 13.805–14.148 | 76.005 | 52.000 | 0.802 |

各役割は別の空JITキャッシュでcold1回、その後同一seed0・同一入力をwarm3回。公開load_case＋simulate＋open_resultを計時し、worker起動・import・JIT/cache・IPC・出力・shutdownを含めました。各simulateは新workerを起動します。親の初回package importは両条件で計時外です。

warm0の人口系列では、活動粒子を含むmacro区間は309区間、全粒子終端後の尾部は11区間でした。開始時人口から数えた活動粒子×macro区間数は3,058,066で、全粒子を320区間動かす名目3,200,000とは区別します。境界終端eventの時刻範囲は3.7819374276720117～3.851308927477456秒です。これらはstage数やevent refinement数ではありません。

3回の時刻はこのマシン上の記述的実測で、一般的な高速化保証ではありません。メモリは親＋生存子processの100 msサンプルprivate commitです。瞬間peakを見逃し得て、working set合計は共有pageを重複計上し得ます。配列予算8 GiBとOS総メモリは別です。

COMSOL比較・Brownian計算・旧新solver性能比較・正式公開を今回追加評価したものではありません。既存comparerの旧新nonregression関連null/falseはbaseline不在による対象外項目です。

[測定要約](summary.json)・[全8結果の比較証跡](comparison.json)・[測定方針](policy.json)・[保存inventory](inventory.json)
