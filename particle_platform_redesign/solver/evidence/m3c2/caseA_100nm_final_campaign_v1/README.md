# M3-C2A Case-A 100 nm final stochastic comparison

## 結論

固定済みのcommon-P1 Case-A 100 nm条件について、COMSOL 32 seedと本ソルバー32 seedを各287粒子、30 ms、121観測時刻で実行し、確認的な終端人口同等性判定を`PASS`した。

事前登録した唯一の確認的gateは`stuck`、`held`、`escaped`、`any_terminal`の4曲線×121時刻である。最大絶対差は`0.005988675958188083`、32×287=`9184`観測/participantに対する95%同時Hoeffding半径は`0.032784144435947335`、差と半径の和は`0.03877282039413542`で、同等性幅`0.05`以内だった。結果後の閾値変更、COMSOLへのpath fitting、本体coreの比較専用分岐は行っていない。

従って言えることは、**この固定case designの終端人口曲線について、本ソルバーはCOMSOLと約5%幅以内で同等な精度を持つ**ことである。乱数系列が独立なので粒子IDごとのpathwise一致は対象外であり、COMSOLを物理のgolden truthとはしない。

## 実行整合性

- 両participantとも32/32 replica完了、各replicaは34,727行（287粒子×121時刻）。
- candidateは境界event 8,451件、failure 0件。COMSOLは境界event 8,448件。
- COMSOL全32 replicaで、対象particle studyは`fptas`だけを有効化したstudy-isolation検査をPASS。
- source identityは64 replica-level stateすべてPASS。初期R/Zはfloat64完全一致し、初期速度・電荷の最大差は64 epsilon予算の13.6%以下。
- COMSOL source MPHは`loadCopy`、`-nosave`で実行し、実行前後SHA-256は一致。
- 固定入力から評価器を別outputへ再実行し、判定manifest、時刻別CSV、2つのSVG、performance、生成READMEがbyte単位で一致。
- このcloseoutでproduction solver coreは変更していない。外部V&V、証拠、計画文書だけを更新した。
- 最終品質gateは標準solver `614 passed`、M3-C2外部V&V `71 passed`、Ruff、Pyrefly、import-linter、complexity gateがすべてPASS。

## 軌道分布の補助評価

以下は結果理解のための非判定指標であり、確認的PASSを左右しない。

| 指標 | 最大差 |
|---|---:|
| seed×固定発生源の平均位置 | 1.148 mm |
| 平均位置差 / geometry scale | 0.4204% |
| R/Z固定分位点 | 5.928 mm |
| R-Z占有分布のtotal variation | 2.588% |
| 共分散成分 | 1.23446e-4 m² |

全観測時刻×粒子IDのうち、両participantの全seedでactiveだった組は80.62%である。30 msでは終端済み粒子が多いため、この条件を満たす発生源は11/287になる。従って後半時刻の連続位置指標は、終端人口曲線と併記して解釈する。

[R-Z trajectory overlay](rz_ensemble_trajectory.png)は、全287発生源について各participantのseed平均軌道を重ねた図である。橙がCOMSOL、青が本ソルバーで、個々の乱数pathではなく発生源ごとのensemble平均を比較する。

[Observable differences](ensemble_observable_differences.png)は、平均位置差をgeometry scaleで正規化した値とR-Z占有分布差の時間変化を示す。

## Authority

- 最終受領書: [`final_result_receipt.json`](final_result_receipt.json)
- 評価結果: [`evaluation_manifest.json`](evaluation_manifest.json)
- 時刻別指標: [`ensemble_metrics.csv`](ensemble_metrics.csv)
- 統合campaign: [`combined_campaign_manifest.json`](combined_campaign_manifest.json)
- participant台帳: [`candidate_campaign_manifest.json`](candidate_campaign_manifest.json)、[`comsol_campaign_manifest.json`](comsol_campaign_manifest.json)
- pilot訂正とfinal lock: [`pilot_method_correction.json`](pilot_method_correction.json)、[`selection_receipt.json`](selection_receipt.json)、[`final_registration.json`](final_registration.json)

V2 pilotで4 seedを95%推論に使った解釈は無効化し、V3ではconfiguration screeningに限定した。finalの32 seedはpilot seedおよびparticipant間で非重複であり、V3 policyを固定してから実行した。

## 範囲外

このPASSはnative COMSOL field、第2 ion-drag式、10/30 nm、3-D、他geometry、一般的なCOMSOL同等性を認定しない。
後続Case-P anchorも完了し、M3-C2A Case-A/Case-P 100 nm benchmarkは`CLOSED_ACCEPTED_WITH_LIMITATIONS`である。
追加coverageと製品規模性能は本benchmarkの完了条件ではなく、明示的なclaimまたはSLAを持つ独立work packageとする。
