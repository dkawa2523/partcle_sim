# 現行入口への撤去・統合記録

開始sourceは`75100e74f1243b527e065c9e6a68121221408b39`、branchは`cleanup/current-only-platform`。
コード撤去と下記の現行確認を実施した。利用者によるバイナリ削除後、記録済み撤去範囲の残存0件と保護対象の保持を確認した。
新しい基盤課題・根因・改良案は[全体調査](../../../reviews/foundation_root_cause_review_2026-11-23.md)へ分けた。

## 利用者による削除後の確認

現在の撤去範囲と検証・採点の集約は[完了記録](completion.json)、物理削除の確認は[削除後の確認](execution/post_manual_cleanup.json)を正本とする。
記録済みHDF5 77,757件と孤立bytecode17件、計77,774件・論理サイズ4,554,763,321 bytesの残存は0件。
198旧result rootsとBrownian rootの計199ディレクトリも全て空である。空ディレクトリは保持している。
[独立した保存確認](execution/post_manual_preservation.json)ではnative authority479件が全SHA一致し、
保護領域のtracked差分は0件、既知Windows環境11件の設定とinterpreterも存在した。未知資産・全環境の全byte保証とはしない。

[新しい公開操作](execution/post_manual_user_flow.json)で、自分のCSV/recipeからcreate/check/run、inspect/analyze/export/viewを実行し、全CLIがexit0。
一粒子の解析的な壁到達時刻0.0175 sとの差は3.47e-18 s、保存位置5行の最大差は0、failureは0件。
これは実行した小型ballistic利用経路の確認であり、全model・全suite・COMSOL再計算ではない。
現行実装67ファイルのSHAは既存の実行記録と一致する。役目を終えたmanual cleanup scriptも撤去した。

削除後の[現行P10 benchmark](execution/post_manual_p10.json)もfield/event/relaxation各8粒子、cold/warm計6観測を再実行し、exit0・failure0。
各scenarioの科学payloadはcold/warm間で一致し、削除前の小型確認の同じhashとも一致した。
これは現行producerの自立性と同一入力の実装identityの確認であり、cold/warmの差はJIT償却、装置予測の認定や改訂間speedupではない。

[最終コードレビュー](execution/post_manual_final_review.json)と公開操作レビューの採点は、現行alphaの設計・利用に対する判断であり、物理精度の確率ではない。
利用性82、保守性78、拡張性73、科学・数値の説明可能性80を等重みで評価し、総合78.25点、丸めて78/100とする。
コードレビュー内の77点は利用性を含まない三観点の提案集計であり、ここでは利用性を加えた四観点の集計を採用する。
R06のnested入力エラー、R07の単独JSON公開、R01の結果説明などの改良案は未実装として引き継ぐ。
今回の撤去完了と、これらの次の実装段階を混同しない。
可視レポートも現在の残存0件と保存確認へ更新した。[表示・出典検証](execution/post_manual_report.json)に公式build、4 tests、desktop/mobile確認、保護234 filesと全10行の入力・表示一致を記録した。

## 削除前の停止記録

削除前の最終残存監査は[実行停止条件と残存](execution/impasse_audit.json)。
`92c7d79`後の独立点検で、廃止済みsourceの孤立bytecode17件・196,342 bytesを追加特定した。
当時の既知の未撤去バイナリはHDF5 77,757件と合わせ77,774件・4,554,763,321 bytesだった。
合成HDF5の再確認と孤立bytecode1件へのSHA guard付き非再帰削除も実行前に拒否され、具体的拒否は計7操作となった。
同じ外部blockを確認したgoal turnは3回連続であり、操作数とは区別する。全ファイルを個別に削除試行した意味ではない。
正確な17パス・SHA・source対応は[bytecode preflight](execution/orphan_bytecode_preflight.json)に保存した。
過去のreport検証記録と実行receiptは各作成時点の根拠として保持し、現在の状態は上記の削除後確認へ対応付ける。

## 対象と必要な役割

現在の目的は、外部の場・形状・実現sourceから、通常のCLI/Pythonで入力作成、検査、計算、結果利用を行うこと。
公開入口、model登録、形式、動的呼出、実利用、独立した科学確認を調べ、旧名・未参照という理由だけでは削除しない。

| 領域 | 撤去・移管 | 保持する役割 |
|---|---|---|
| numerical core | 孤立したBézier bound wrapperと専用compiled helperを撤去 | 現行proposal/dense enclosure、同stage連成、event/残時間、RNG |
| core検証・fixture | 旧工程専用metadata/activation引数、旧版/thread名専用検査、診断文言の固定を整理。追加でforce/Brownian/wall-source scenarioのprivate byte定数を撤去 | 解析解、未知形式/未知設定の厳格拒否、実allocation対budget、予算/slab/科学identity |
| 外部V&V | 固定12package探索、パス名による物理推測、旧式replay、廃止actual-run契約、旧F02専用case/producerを撤去。重複headerとID丸めを修正 | 明示CSV/current model診断、statusとcauseの区別、native/解析参照、selected CSVの実入力mapping |
| ベンチマーク | P11の重複CLI/worker/reportをP10へ統合し旧専用report/一時workerも撤去。重複metricsとP19/B03 private byte式・旧worker fallbackを撤去 | C03高dt緩和、cold/warm、科学payload、独立oracle、現行plan/実allocation、B03独立transient上界 |
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
- 全体初回は1 failed /1227 passed。旧診断文言のregexを更新後、`8883e3a`時点のclean全体1228 passed /679.713 s。
- その後のprivate容量定数整理では、surface/exponentialとdynamic Brownian bankの所有確認を補い、関連30検査が成功（22.44 s）。全体1228の再実行とは区別する。
- B03も32粒子・全4行を再実行し、science hashは前回と各行一致。raw646/丸め648 bytesの独立transient stage上界を保持した。
- 統合P10の3 workload×cold/warm、P06、全8行P14 smoke、P19、B03を再実行し成功。独立C03解析11検査も成功。
- 最終外部suite156 passed /39.10 s。CSV重複headerの回帰を含み、native196件のhashは不変。
- Windows/Linux双方の新wheelと、checkout外の自分のCSV→source/case→check/run→result-only解析/export/表示が成功。
- lock、Ruff182 files、Pyrefly0 errors、import6契約、CC<16、文書33/skills4、文書3検査とreport snapshot4検査が成功。
- 実行とhashの集約は[final execution](final_execution.json)。reportのbuild/表示検査は専用receiptに記録する。
- 今回追加で発見した入力nested設定・単独JSON公開・result内容identity等の改良案を、修正済みとはしていない。
- 継続時の最終全体検査は1240 passed /473.52 s。R13の関連30検査、B03の4行に加え、削除後の隔離Windows/Linux wheelで13操作・5例題を再実行した。

## 物理削除の経緯（完了前の記録）

旧Brownian生成物はtracked text72件に加え、UTF-8の旧result3 checkpoint pointer/空completion marker28件・6,221 bytesを撤去した。
現時点で95 HDF5・152,994,704 bytesが残り、markerは0件。元の123件は`8883e3a`時点の記録である。
workspace内の絶対パス、旧software test生成記録、native/model/environment資産を含まないことを確認した上で、
recursive削除2回と、旧result3を確認した9,840 bytesの単一HDF5への非再帰削除2回が自動承認審査に拒否された。
返却理由は`blocked by policy`のみであり、削除は未実行である。
別途198の旧result subtree（inventory時78,204 files /4,403,705,627 bytes）から、旧engineのUTF-8 metadata571件・2,930,032 bytesを撤去した。
16件のpartialもhash検証済みnested engine identityと当時writerに照合した。これらにはHDF577,633件・4,400,775,595 bytesだけが残る。
この旧形式群の実残存はHDF577,728件・4,553,770,299 bytes。198ルートのバイナリ削除は試行していない。
形式cutoffでは拾えなかった現行形式のP10/P11 guard debugも、producer由来・合成入力の現行生成処理との完全一致・実consumerに照合した。
そのtext24件・68,989 bytes（8 YAMLを含む）を撤去し、29 HDF5・796,680 bytesは残存する。
現行P10/P06/C03と独立oracle、dated maintenance記録、実行環境を保持した。P10のfield/event/relaxation各8粒子をcold/warm計6行で再実行し、failure0・各scenario科学payload一致を確認。
`92c7d79`時点の既知残存はHDF577,757件・4,554,566,979 bytes。合成P11 fixture一つにもhash guard付き削除を試したが同じpolicyで拒否され、この時点の具体的拒否は計5操作となった。
詳細は[producerの役割による追加撤去](execution/guard_artifact_cleanup.json)。全バイナリを個別に試したという意味ではない。
分類rootの重複・包含と保護資産の混入は検出されなかった。実行時には再確認が必要である。
対象・留保は[blocked cleanup](blocked_cleanup.json)に記録する。回収容量の実績とは扱わない。

継続時には役目を終えた一時generator/audit/QA wrapper9件も撤去し、既存の実行receiptと独立oracleは保持した。
削除後の独立再確認はnative関連479ファイル（Git authority196、外部raw278、再生成metadata5）の全SHA一致。
実行内容は[検証整理・marker撤去](execution/continuation_cleanup.json)と、その後の[追加text撤去・公開操作再確認](execution/other_text_cleanup.json)へ区別して記録する。

当時は対象削除が許可される実行条件への変更が再開条件だった。
その後、利用者が限定scriptを実行し、上記の独立した残存・保存確認が成功した。
実装やテストの成功を削除完了へ読み替えず、実際の対象不在を現在の証拠としている。
