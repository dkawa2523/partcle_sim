# chamber-particles 0.3.0

0.3.0は、既存の2-D数値処理を一つのengineのまま高速化し、同一runのCPU process並列を追加した版です。
Python 3.12、Cartesian XY/RZ meridional、canonical case/data/result schema 3、checkpoint schema 2を維持します。

## 主な変更

- `simulate(case, output, *, workers=1)`とCLI `run ... --workers 4`でprocess数を指定できます。
  既定は1で、worker数はYAMLの数値設定やRNG keyに含めません。
- 親が時刻・release・単一writer・checkpointを所有し、workerは排他的な粒子群を同じ数値処理で更新します。
  完了したworkerへ次の粒子群を渡し、境界/failureイベントは容量を制限したbatchとACKで受け渡します。
- 作業配列の再利用、stage workspaceの統合、field/geometry探索と初期包含検査の処理効率を改善しました。
  共有scratch templateは全workerの準備完了後に解放し、run内のreplay共有メモリを再利用します。
- 付着・保持後の粒子は運動更新を停止し、最終状態と要求された出力を保持します。
- 計算途中のworker異常終了では未完結果を完成として公開せず、最後のdurable checkpointからworker数を変えて再開できます。
  実workerの異常終了、再開後の保存科学出力一致と共有メモリ・process cleanupを検証します。
- 0.2.0後の数値修正として、Brownianの絶対終端時計と、有効な静止exact pathの有限半径接触判定を修正しました。
  数値revisionを更新し、実行資源だけの変更と区別しています。

## 性能と検証範囲

Windowsの登録用途では大粒子数の並列効果を確認しました。追加P14-Uの1万粒子・320ステップは
1/4 processで27.36/13.93秒、warm中央値の速度比1.96xでした。一方、COMSOL入力の287粒子・1500ステップでは
2/4 processが遅く、worker数を増やせば常に速くなる仕様ではありません。
条件、cold/warm、出力量、総CPU・メモリ、科学一致と制約は[並列性能](docs/parallel_execution_plan.md)を参照してください。

品質・公開scenario・用途別科学出力、Windows/Linux、clean-installed wheelの三APIと実process CLIを検証対象とします。
COMSOL Case A/Pは保存参照に対する登録済み位置・終端人口分布の回帰です。全native右辺、境界応答、速度・電荷分布、
実験的な製造予測精度の一般保証へ拡張しません。[評価方法](../vv_methodology.md)に比較範囲を明示しています。

## 利用上の条件と将来項目

複数processを使うscriptはimport可能なファイルに保存し、main guardを付けてください。
solver-owned配列予算とOS総メモリは異なります。wheelはsolver packageとCLIを含み、`tools/`・`examples/`はsource checkoutに付属します。
過去schemaの互換readerはありません。

全workerは活動粒子の各macroステップ末尾で同期します。複数macroをまとめて実行する同期間隔の設定化は将来項目です。
数値刻み・Brownian乱数・境界処理・出力を維持した上で、時間・メモリ・checkpoint/resumeを比較して判断します。
追加JIT化、3-D、GPU、粒子間相互作用や場へのfeedbackは0.3.0に追加していません。

利用ライセンスは設定しておらず、再利用条件は権利者の明示許諾が必要です。
過去の変更履歴は[v0.2.0](https://github.com/dkawa2523/partcle_sim/blob/v0.2.0/particle_platform_redesign/solver/RELEASE_NOTES.md)と
[v0.1.0](https://github.com/dkawa2523/partcle_sim/blob/v0.1.0/particle_platform_redesign/solver/RELEASE_NOTES.md)に保持します。
