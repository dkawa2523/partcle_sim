# chamber-particles 0.5.0 alpha

入力の準備から結果の解釈・再利用まで、同じwheelのCLIとPythonで実行します。
coreの公開操作は `load_case`、`check_case`、`simulate`、`open_result` の四つです。

- `init --example rz-charged`は既存例題と同じ事実から編集可能なgeometry/field CSVとmappingを作ります。
  編集したfieldをimportし、source recipeへ渡して自分の入力で計算できます。
- `inspect`はsourceのrelease名・table・種別・粒子数と、boundary ID/group/materialの対応を示します。
  `analyze/export/view`は同じcohortを使い、`--source-names`と`--particle-ids-file`でも選択できます。
- viewの詳細表示対象は `--focus-particle-id` / `focus_particle_id` です。旧aliasはありません。
- source CSVの解釈とhashは同じbytesに束縛します。明示された派生field群は、primitive・layout・clock・
  parameter・物性・全outputの対応を入力出版前に検査します。独立prescribed fieldへ推測した依存を加えません。
- modelの解決済みidentity・field role・scalarは一つのrecordから説明と実行へ渡します。
  stage算術、全域bound、局所適用域証明、noiseの役割を分け、親の説明用planをworkerへ渡しません。
- result schema **5**は全commit prefixのhashをcheckpointへ保持します。resumeとsealは全prefixを照合し、
  readerは同じopen handleでhashとHDF5読取りを行います。`inspect --verify` / `verify_contents()`で全payloadを確認できます。
  固定identityと操作の検証範囲は別receiptです。byte integrityは科学的精度認定ではありません。
- 保存COMSOL集団比較はtime group昇順のCSVを逐次処理します。同じ時刻内のID順は自由です。
  全時刻×全粒子のhistory配列は作りません。欠損、原因不明の消失、candidate escape/failureを区別します。

case/data schema **3**、source/recipe **2**、CSV importer設定 **4**、checkpoint particle配列 **2**は維持します。
旧resultと旧partialは受理しません。必要な旧科学列は元環境でexportするか、元入力から再実行してください。
数値式・event順・RNGの意味を変える変更は、構造の再編と区別して検証します。

変更前後の科学payload、実行済みgate・benchmark・配布確認、三観点のレビューと未確認範囲は
[基盤再構成記録](evidence/foundation_2026_11_23/README.md)が所有します。
過去の資格・native観測を現行sourceへ付け替えません。Git push、remote CI、tag、Release/registry公開は
local確認と別の実施事項です。利用手順は[README](README.md)を参照してください。

3D、全材料/浮遊電場、GPU、動的wall charging、場feedbackは条件付きbacklogです。
数値stickは実付着、computational countは物理fluxの認定ではありません。
利用ライセンスは設定しておらず、再利用条件は権利者の明示許諾が必要です。
