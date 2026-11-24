# 文書再構成と確認の記録

基点は`ec646c07a35d9f51fdb5ed7f3a6d9772751fe668`、branchは`refactor/foundation-usability`です。
今回の対象は現行説明、参照、表示レポートと既存の文書検査です。数値core・科学model・製品設定・lockを変更せず、
既存の実行根拠を保って本文を再構成しました。文書の完成を数値・native・実験妥当性の認定にはしません。

## 読者と正本

利用者は[利用ガイド](../../README.md)から、開発者・評価者は[文書案内](../../../README.md)から読みます。
目的は製品仕様、責務・依存・変更の受渡しはarchitecture、式はphysics models、積分・pathはnumerics、
入出力列はcase/result形式、変更と確認はquality tooling、比較の判断はV&V方法論へ統合しました。
AGENTSは安定した制約と数値不変条件を保持します。

## 統合と履歴

工程順に積み上がっていた数値説明をmethodとaccepted pathの流れへ再構成し、旧thread設定、旧result形式、
後続実装で置換されたcharge拒否gateを現行本文から除きました。物理式・導出・独立oracleと資格の限界は保持しています。
3D、非線形gas noise/SDE、粒子熱状態は採用済み未接続の方針として、実装済みmean dragやcollision oracleと区別しました。

旧architecture reviewとdataset再構築計画は有効な条件を各正本へ統合して`reviews/history/`へ移しました。
hashに束縛されたreview、元の実行receipt、native観測、参照dataset、archiveのbytesは変更しません。
可視レポートは同じappとquery/component identityを保持し、現行文書の説明と、過去の実行source・条件を分けます。

## 実行した確認

確認用のwheel・入力・結果・ログは`.artifacts/documentation_20261124_01/`へ分離しています。
既存installed workflowとREADME中の小型console/Python例を実行し、解析的hit・最終状態・failure、
全配布例と通常入口を確認しました。新しい科学解析・COMSOL solve・性能campaignは行っていません。
文書検査は維持対象のlocal path/fragmentと4技能、reportのcurrent authorityのexact-byte bindingを扱います。
文書44件・技能4件のpath/fragment、文書とsnapshotの23 tests、変更toolのRuff/Pyrefly/CC検査が成功しました。
installed wheelの13 CLI経路・5例題と、READMEの11 console/Python例を確認しました。
三観点の独立通読と実表示確認で見つかった必須指摘は全て修正し、再確認しました。
公式split/offline build、desktop/mobileの8表・Source・Copy・再生成手順、offline全10 datasetを確認済みです。
条件、保全確認、以前のsourceに限定する資格は[実行receipt](verification.json)を参照してください。

## 保存計測値と削除完了の扱い

[品質projection](quality_projection.json)は以前のfoundation品質12行を再利用し、削除対象5storeについてのみ
既存の[利用者による削除完了記録](../foundation_2026_11_23/baseline_reference_removal.json)から状態を更新しています。
それ以外の11行、assessment/performanceの値、実行source・条件を変更せず、新しい計測とは数えません。
以前の自動承認拒否と保持記録は当時の証拠として残します。
