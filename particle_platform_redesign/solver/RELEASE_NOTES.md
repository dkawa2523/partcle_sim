# chamber-particles 0.4.0

利用者がwheelだけで入力の準備から結果の再利用まで進めるように、通常clientを同じ配布物へ統合しました。

- `models`でモデルとfieldの単位・成分・基底を調べ、`init`の5例を編集し、`create`で自分の粒子源と設定を公開できます。
- CSV取込み、field cache、限定された静的RZ電場生成を`inputs`へ統合しました。COMSOL固有の抽出・比較は外部V&Vです。
- coreの4操作は`load_case`、`check_case`、`simulate`、`open_result`です。`check_case`は実行と同じ準備を使い、計算を開始しません。
- `analyze`、`export`、`view`はresult単独で使えます。源・重み・分母を保持し、接触回数、最初の材料接触、終端、最終停止を区別します。
- 必須のgeometry・源・境界contextをresult4へ保存します。result3と旧途中結果は読み込みません。入力schema3、checkpoint配列schema2は維持します。
- 親時計・writer、process実行、粒子slab更新の責務を分離しました。数値処理とRNGの意味は一つのengineに維持します。

旧経路整理時の実行済み範囲と外部条件は[整理・実行記録](evidence/retirement_2026_11_23/README.md)、
今回の修復後の確認は下記の修復記録を参照してください。
過去の認定を新しいsourceへ自動適用しません。基本操作は[README](README.md)にまとめています。

## 公開入力・出力の不具合修復

- selected CSVのnode IDを元tokenからexact int64へ解決し、scope選択と物理値の並替えを同じ行へ適用します。
  adapter revisionはv5。domain/boundary/materialのint32範囲も配列作成前に検査します。
- source設定を値の使用前に解決し、必須列と全行の幅をfilter前に検査します。numeric filterもexact decimal比較です。
- 静的電場builderは時系列の速度・温度をcheck/build共通prepareで拒否します。builder revisionはv4です。
- `analyze -o`は完成JSONを既存先へ上書きせず公開します。公開後の一時file削除失敗は、完成fileを保持して明示します。

case/data、result、checkpointの形式と数値core・物理式は維持しています。
修復の再現・正常操作・実行済み確認は[修復記録](evidence/public_io_fix_2026_11_23/README.md)にまとめています。

## 現行経路への整理

source作成はrecipeから単一publisherへ接続します。旧source facade、未使用scalar event探索、
固定native campaign、旧cadenceを再現する性能処理、専用の設定・検査・生成物を撤去しました。
必要な科学検査は現行kernelと解析解・独立native参照へ移しています。
性能測定は公開APIの同一入力をworker数別に実行し、入力・参照・評価・結果を区別します。

形式はcase/data3、result4、checkpoint2です。過去形式のreader、旧入口alias、移行dispatchは提供しません。
旧版の詳細はGit履歴へ保持し、現在のsourceや性能資格として扱いません。
複数processではmain guardを付け、worker数の効果を実際の入力・数値精度・出力量で測定してください。

通常機能はwheelへ含まれます。live COMSOLと開発V&V/benchmarkはcheckout側です。
3D、全材料/浮遊電場、GPU、場feedback等の条件付き拡張は実装計画で区別します。
利用ライセンスは設定しておらず、再利用条件は権利者の明示許諾が必要です。
