# 現行 source による A/P 粒子輸送の保存 COMSOL 比較

全体の運動粒子減少、停止総数、代表的な停止時間は、乱数軌道を一対一に合わせない大勢の比較ではよく対応している。一方、境界33の停止数と100 µsの運動粒子速度には方向の揃った局所差が残る。全観測量の厳密一致、事前に定めた用途許容に対する同等性、native solver parity を認定した結果ではない。

比較対象は A/P 各287個の固定 realized source、保存 native 各1実現、現行0.5 sourceで新たに計算した各40実現（seed21–60）である。A/Pを80独立実現へ合算していない。両caseの40件完了・全件正常・計算前後の32 coreファイルSHA一致を確認してから集計した。途中で中断された他の `ensemble*` は使っていない。

## 全体輸送と停止

候補欄は40個の「全287粒子を一組とした実現」の平均と実測最小–最大であり、平均の信頼区間ではない。粒子の物理的重み・flux・付着質量には換算していない。

| 観測量 | A native | A 現行候補 平均（実測範囲） | P native | P 現行候補 平均（実測範囲） |
|---|---:|---:|---:|---:|
| 5 ms制限平均active時間 [µs] | 200.397 | 203.489（182.211–224.206） | 187.383 | 203.055（178.605–220.079） |
| retained数 / 全287 | 251 | 249.675（235–265） | 254 | 251.775（241–262） |
| retained終端時刻 p10 [µs] | 17.486 | 15.563（13.382–19.733） | 15.950 | 15.326（12.796–19.439） |
| retained終端時刻 p50 [µs] | 153.670 | 150.717（127.490–174.933） | 142.311 | 147.672（125.441–170.582） |
| retained終端時刻 p90 [µs] | 401.550 | 436.773（363.722–504.914） | 373.983 | 450.459（373.135–531.871） |

平均active時間の平均差は A +3.092 µs、P +15.671 µs。retained総数の差は A −1.325個（全放出比 −0.462 percentage points）、P −2.225個（−0.775 pp）だった。Pの遅い終端側には差も残るため、中央値だけの一致とはしていない。

全active曲線の対称 leave-one-out 距離順位は A 0.488、P 0.610、停止先compositionの同順位は A 0.707、P 0.415。nativeの全体的な変動は候補群のばらつきと整合する記述的根拠になる。ただし既に見た差から選んだ探索的指標であり、非棄却を同等性PASSや全時刻・全指標の同時保証へ読み替えていない。

## 局所に残る差

| 観測量 | A native | A 現行候補 平均（実測範囲） | P native | P 現行候補 平均（実測範囲） |
|---|---:|---:|---:|---:|
| 境界33 retained数 / 全287 | 74 | 60.450（45–72） | 73 | 61.325（46–74） |
| 100 µs own-active速度中央値 [m/s] | 759.083 | 861.318（772.046–955.508） | 737.508 | 866.924（771.861–970.216） |
| 100 µs own-active Z中央値 [e] | −7.0149 | −7.0708（−7.2613–−6.8988） | −9.3540 | −8.4795（−9.5336–−7.5463） |

境界33は z=22 mm、r=153.5–179 mm の同定済み表面。平均停止数差は A −13.550個（−4.721 pp）、P −11.675個（−4.068 pp）。Aでは全40候補がnativeより少ない。100 µs速度中央値は A/Pとも全40候補がnativeより大きい。したがって、これらの差を「全て通常の乱数差」として消していない。

速度・電荷は各実現のその時点の生存粒子に条件付けた量であり、同一粒子集合の加速度誤差ではない。場転送、native有限刻みRK4の時間キー付きnoiseと候補weak-limit表現、境界応答後の生存群の選択、数値path/event処理等の寄与は、この40実現比較だけでは分離できない。厳密な乱数一致を新たな目的にはせず、境界33への量や100 µs速度を精密に使う場合の未解決差として残す。

## 現行計算のcoverageと履歴比較

- 全80runでfailure 0、active非有限状態 0、terminal/releaseで説明できない欠損 0。各runの初期全287 IDについて r/z、vr/vz、Z、release時刻がnativeと完全一致した。
- nativeは各96保存時刻×287行＝27,552行を比較窓に持つ。元0–30 ms CSVは不変。nativeの最終保存値は `0.004999999999999999 s`、候補はそれに加えて実行終端 `0.005 s` も保存する。曲線・空間図は96個のexact common clockのみ。時刻fit・補間はしていない。全native粒子はこの最終保存時刻より前にterminalへ到達しており、5 ms制限平均active時間はこの1ulpの観測端差に影響されない。
- nativeのretainedは A 251、P 254、未知原因のdisappearanceは A 36、P 33。後者をcandidate escapeと同義にはしていない。retainedは数値的終端であって物理的接着の認定ではない。
- 旧40の保存JSON数値と現行fresh40を同seedで比較した。RMST、全lifecycle/failure数、停止総数・全境界別停止数、終端時刻p10/p50/p90・最終terminal時刻、10/20/30/50/100/200/500 µsのactive数・速度/Z p10/p50/p90は、各seedで全て数値完全一致した。入力logical content hashとbase YAML SHAも不変。これは選んだ保存要約の維持であり、全ての未保存pathや全state payloadのbitwise一致を主張していない。旧result containerを現行readerで開いたり、schema5へ付け替えたりしていない。

図は [40実現集計](ap_population_summary.png)、[現行seed21の全域空間比較](ap_spatial_overview.png)、[source近傍](ap_spatial_source_zoom.png)。空間図は各case全287 ID、solver左/native右、RZ mm、等しいaspectと共通軸。candidate既知boundary-event区間・欠損・非activeで線を切り、native未保存反射を再構成していない。共通canonical perimeterは表示の基準であり、native geometry parityの認定ではない。

数値・source hashは [assessment](ap_population_assessment.json)、履歴要約の全項目比較は [preservation receipt](ap_population_preservation.json)、描画意味とhashは [集計図receipt](ap_population_summary_receipt.json)・[空間図receipt](ap_spatial_receipt.json) に記録した。各PNGは実見して軸、marker、凡例、coverage、文字の欠けを確認した。

`qualified native solver parity = NOT_TESTED`、`physical validation = NOT_CERTIFIED`。事前の用途許容、native側の独立resolution/ensemble、物理的源weight・実験比較は、この比較では追加していない。
