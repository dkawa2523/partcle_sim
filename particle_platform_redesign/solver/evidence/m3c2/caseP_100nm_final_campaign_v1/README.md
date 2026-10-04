# M3-C2A Case-P 100 nm final stochastic comparison

## 結論

固定済みのcommon-P1 Case-P 100 nm条件について、COMSOL 32 seedと本ソルバー32 seedを各287粒子、30 ms、121観測時刻で実行した。事前登録した終端人口gateとfull-population R-Z/fate分布gateはともに`PASS`した。

この結果が認定するのは、**negative-ion currentを含まない固定same-form数値モデルについて、登録した人口observableがCOMSOLと事前登録幅内で同等であること**である。粒子IDごとの乱数path一致、COMSOLを物理のgolden truthとする判断、Case-Pプラズマ全体の物理妥当性は認定しない。

## 確認的gate

| gate | 観測値 | 同時半径 | 同時上限 | 同等性幅 | 判定 |
|---|---:|---:|---:|---:|---|
| 4終端人口曲線 | 0 | 0.0339157 | 0.0339157 | 0.05 | PASS |
| 83区分R-Z/fate TV | 0.0106707 | 0.120529 | 0.131200 | 0.15 | PASS |

全64 replicaで終端eventは0件、30 ms時点も全粒子activeだった。このため終端人口gateは正しく計算されているが、境界一致に関しては情報を持たない。主たる非自明な結果は、10×8のactive位置binと3終端categoryからなる83区分分布のgateである。

## 実行整合性

- 両participantとも32/32 replica完了、各replicaは34,727行（287粒子×121時刻）。
- candidateは境界event 0、failure 0。COMSOLも境界event 0。
- COMSOL全32 replicaで、対象particle studyは`fpt`だけを有効化したstudy-isolation検査をPASS。
- source identityは64 replica-level stateすべてPASS。初期R/Zはfloat64完全一致し、初期速度・電荷の最大差は64 epsilon予算の13.6%以下。
- COMSOL source MPHは`loadCopy`、`-nosave`で実行し、実行前後SHA-256は一致。
- final seedはCOMSOL `319000–319031`、candidate `319032–319063`で、participant間およびpilot seedと非重複。
- 固定入力から評価器を別outputへ再実行し、manifest、CSV、2つのSVG、performance、生成READMEがbyte単位で一致。
- production solver coreは変更していない。修正したのは外部V&V評価器のparticipant別event header読取りだけで、共通必須列の欠落・重複はfail-closedで拒否する。

## 軌道分布の補助評価

以下は結果理解のための非判定指標である。

| 指標 | 最大差 |
|---|---:|
| seed×固定発生源の平均位置 | 0.2878 mm |
| 平均位置差 / geometry scale | 0.07743% |
| R/Z固定分位点 | 0.7672 mm |
| R-Z占有分布のtotal variation | 1.0671% |
| 共分散成分 | 3.59378e-5 m² |

[R-Z trajectory overlay](rz_ensemble_trajectory.png)は全287発生源について各participantのseed平均軌道を重ねた図である。橙がCOMSOL、青が本ソルバーであり、個々の乱数pathではない。

[Observable differences](ensemble_observable_differences.png)は、平均位置差をgeometry scaleで正規化した値とR-Z占有分布差の時間変化を示す。
両PNGは評価器生成SVGから作った判定非依存の閲覧用派生物であり、source SVG、PNG、rasterizer identityの対応は
`final_result_receipt.json`に固定した。

## 物理上の範囲

このCase-Pにはnegative ionが存在するが、固定したcharge revisionはnegative-ion currentを含まない。従ってstatusは`NOT_CERTIFIED_NEGATIVE_ION_CURRENT_OMITTED`であり、今回のPASSをcharging physicsや実装置軌道の物理認定へ拡張してはならない。

COMSOLのbuilt-in Saffman forceは無効だが、これはlift未評価を意味しない。COMSOLではcustom `liftfm`、本ソルバーでは`rarefied_vorticity_sensitivity_rz_v1`として、同じcommon-P1の密度、平均自由行程、ガス速度、方位渦度を用いたfree-molecular lift式を評価した。

## 性能上の位置

候補finalは4外部processを重ねて完走させたため、計時は`NON_AUTHORITATIVE_EXTERNAL_WORKLOAD_OVERLAP`であり、COMSOLとの速度比較には使わない。科学work counterは各replicaで3,444,000 accepted pieces、3,444,000 candidate queries、refinement 0で、32 replica合計110,208,000件だった。

first/middle/last seedの外部owner discoveryは後続evidenceで完了し、科学payload・work・revision identityを維持したまま
`optimization_authorized=false`と判定した。M3-C2A benchmarkは`CLOSED_ACCEPTED_WITH_LIMITATIONS`であり、10,000粒子以上の
scaleと外部process並列は完了条件ではない。明示的な製品SLAが設定された場合だけ、別の性能work packageとして評価する。
solver内部thread pool、第二engine、autotunerは追加しない。

## Authority

- 最終受領書: [`final_result_receipt.json`](final_result_receipt.json)
- 評価結果: [`evaluation_manifest.json`](evaluation_manifest.json)
- 時刻別指標: [`ensemble_metrics.csv`](ensemble_metrics.csv)
- 統合campaign: [`combined_campaign_manifest.json`](combined_campaign_manifest.json)
- participant台帳: [`candidate_campaign_manifest.json`](candidate_campaign_manifest.json)、[`comsol_campaign_manifest.json`](comsol_campaign_manifest.json)
- COMSOL実行台帳: [`comsol_run_receipt.json`](comsol_run_receipt.json)、[`comsol_artifact_hashes.csv`](comsol_artifact_hashes.csv)
- 評価器生成物: [`performance.json`](performance.json)、[`R-Z SVG`](rz_ensemble_trajectory.svg)、[`difference SVG`](ensemble_observable_differences.svg)
- 実行認可: [`selection_receipt.json`](selection_receipt.json)、[`final_registration.json`](final_registration.json)

評価器の独立再実行でbyte一致した`README.md`はraw出力側の生成READMEを指す。このdirectoryの`README.md`は、hash固定した
評価結果を説明するcurated文書であり、判定入力ではない。

## 範囲外

このPASSはnative COMSOL field、negative-ion currentを含むcharge model、第2 ion-drag式、10/30 nm、3-D、他geometry、eventful Case-P境界、一般的なCOMSOL同等性を認定しない。比較専用処理は引き続き外部V&Vに留め、本体solverの責務へ混入させない。
