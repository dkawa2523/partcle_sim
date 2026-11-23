# 現行入口への撤去・統合記録

開始sourceは`75100e74f1243b527e065c9e6a68121221408b39`、branchは`cleanup/current-only-platform`。
コード撤去と下記の現行確認を実施した。物理削除には外部blockが残り、全撤去完了とはしていない。
新しい基盤課題・根因・改良案は[全体調査](../../../reviews/foundation_root_cause_review_2026-11-23.md)へ分けた。

## 対象と必要な役割

現在の目的は、外部の場・形状・実現sourceから、通常のCLI/Pythonで入力作成、検査、計算、結果利用を行うこと。
公開入口、model登録、形式、動的呼出、実利用、独立した科学確認を調べ、旧名・未参照という理由だけでは削除しない。

| 領域 | 撤去・移管 | 保持する役割 |
|---|---|---|
| numerical core | 孤立したBézier bound wrapperと専用compiled helperを撤去 | 現行proposal/dense enclosure、同stage連成、event/残時間、RNG |
| core検証・fixture | 旧工程専用metadata/activation引数、旧版/thread名専用検査、診断文言の固定を整理 | 解析解、未知形式/未知設定の厳格拒否、実配列容量の検査 |
| 外部V&V | 固定12package探索、パス名による物理推測、旧式replay、廃止actual-run契約、旧F02専用case/producerを撤去。重複headerとID丸めを修正 | 明示CSV/current model診断、statusとcauseの区別、native/解析参照、selected CSVの実入力mapping |
| ベンチマーク | P11の重複CLI/worker/reportをP10へ統合し旧専用report/一時workerも撤去。重複metricsとP19 private byte式・旧worker fallbackを撤去 | C03高dt緩和、cold/warm、科学payload、独立oracle、現行plan/実配列所有検査。scenarioの残る定数は全体調査で提案 |
| 一度限りの解析 | 固定12datasetのpandas解析3scriptを撤去 | 元のnative観測表と調査記録。必要な当時sourceはGit履歴 |
| 旧生成物 | 旧Brownian tracked fixture/result text72件、旧v46candidate公開・評価層44件、旧binding比較2件を撤去 | native authority、元のpolicy/selection、nativeが実際に使ったsource |

## 残すものと理由

現在のcase/data3、result4、checkpoint2、model revision名は現行の意味を持つ。番号だけでは廃止対象にしない。
数値predictor、同じ数値bodyへの検証adapter、独立Gaussian/Kramers/解析oracleは現在必要な確認として保持する。

独立したassets監査でnative authority196ファイルの存在・hash一致を確認した。
`candidate_input_three_current_z0.h5`は、名前にかかわらずCOMSOLが使った実現source入力なので保持する。
nativeの刻み選択に使った混在pilot評価、元のpolicy/selection/seed authority、原MPH/raw CSV、実環境、
`model_dataset/`、read-onlyの`old_code/`を、この撤去から推定削除しない。

## 実行状況

- Core関連279検査、独立Brownian/複数反射/軸oracle5検査が成功。
- 全10 microcaseのcanonical hash、全設定、物理oracle・許容値はactivation metadata撤去前後で厳密一致。
- 全体初回は1 failed /1227 passed。旧診断文言のregexを更新後、clean全体1228 passed /679.713 s。
- 統合P10の3 workload×cold/warm、P06、全8行P14 smoke、P19、B03を再実行し成功。独立C03解析11検査も成功。
- 最終外部suite156 passed /39.10 s。CSV重複headerの回帰を含み、native196件のhashは不変。
- Windows/Linux双方の新wheelと、checkout外の自分のCSV→source/case→check/run→result-only解析/export/表示が成功。
- lock、Ruff182 files、Pyrefly0 errors、import6契約、CC<16、文書33/skills4、文書3検査とreport snapshot4検査が成功。
- 実行とhashの集約は[final execution](final_execution.json)。reportのbuild/表示検査は専用receiptに記録する。
- 今回追加で発見した入力nested設定・単独JSON公開・result内容identity等の改良案を、修正済みとはしていない。

## 実行できていない削除

旧Brownian生成物はtracked text72件を撤去したが、95 HDF5と28 marker、計123ファイル・153,000,925 bytesが残る。
workspace内の絶対パス、旧software test生成記録、native/model/environment資産を含まないことを確認した上で、
recursive削除2回と、旧result3を確認した9,840 bytesの単一HDF5への非再帰削除1回が自動承認審査に拒否された。
返却理由は`blocked by policy`のみであり、削除は未実行である。
別途198の旧result subtree（inventory時78,204 files /4,403,705,627 bytes）を分類したが、これらへの削除は試行していない。
分類rootの重複・包含と保護資産の混入は検出されなかった。実行時には再確認が必要である。
対象・留保は[blocked cleanup](blocked_cleanup.json)に記録する。回収容量の実績とは扱わない。

再開には、対象削除が許可される実行条件への変更が必要。その後、実際の残存・native/input/environment境界とlive processを再確認して削除し、
残存・誤削除監査を再実行する。実装やテストの成功を、この物理削除の完了へ読み替えない。
