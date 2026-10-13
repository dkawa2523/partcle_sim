# COMSOL selected-field CSV adapter

対応する mesh / field CSV を、producer-neutral な canonical H5v3 へ一方向変換する外部 tool です。
field production、粒子物理の選択、COMSOL native reader、軌道比較はこの tool の責務ではありません。
core から import されず、任意式 DSL や COMSOL 専用 engine を持ちません。

## v3 の変換範囲

adapter 設定は `format_version: 3`、revision は `comsol_selected_nodal_csv_adapter_v3` です。
v2 の thermal 5-field dialect は置換しました。設定・例・test は一つの selected mappings 経路を使います。
H5v3 と field / geometry runtime は変更していません。

- 静的 `cartesian_xy` / `axisymmetric_rz` の、明示した一つの volume domain を抽出します。
- tri3 と provider の明示的な `q1_reference` quad を扱い、quad は品質を比較して P1 へ分割します。
- axis ID は RZ だけで明示除外し、残る perimeter ID は semantic group へ過不足なく割り当てます。
- `fields.mappings` に指定した任意の有限個の nodal scalar / vector primitive を、同じ P1 mesh へ転記します。
- 原始値に有限の `scale` / `offset` を適用し、入力単位と canonical SI 単位を記録します。
- node ID がある export は `fields.node_id_column` で mesh `vertex_id` に結合し、全単射と座標を検証します。
  provider に scoped mesh node ID がない場合は null を指定し、tolerance 内の座標全単射を証明します。
  row number / probe ID を node ID と見なす操作は producer 側でも許されません。
- 全 input を一度だけ bytes snapshot し、同じ bytes を hash と parse に使います。

時間 snapshot、moving / remeshed geometry、cell association、one-sided / discontinuous field、multi-domain union は
この adapter では未対応です。これらを既存 static / continuous P1 へ無言で変換せず、設定で拒否します。
PPR / recovered derivative は、producer が選択した nodal primitive として転記できます。その値を転記しても、
元の FE operator と off-node の P1 derivative が同じになるという主張はしません。

## 入力設定

`source` は vertices、triangles、quadrilaterals、boundary_edges、field_values、field_columns の六つの path を指定します。
空の triangle / quadrilateral 表も header は必要です。vertices の header は RZ で `vertex_id,r_m,z_m`、
XY で `vertex_id,x_m,y_m` です。他の mesh CSV の header は F02 例と合成 test を参照してください。
field_values は COMSOL の `%` comment を許す数値 CSV、field_columns は
`column,COMSOL_expression,unit` の辞書です。新しい export は stable mesh node ID も数値列として出してください。

各 mapping は次を明記します。列名・値・unit は実際の export と照合し、field の name / components / basis / SI unit は
選択する solver physics catalog の required metadata と一致させます。converter の成功は、要求物理 model の
`load_case()` 検査や stage support / axis regularity 認定の代わりにはなりません。

```yaml
format_version: 3
# source / boundary_groups は chamber ごとに inspected exports から設定する。
domain:
  coordinate_system: axisymmetric_rz
  external_id: 12                 # この例の仮値。新 case の default ではない。
  layout_name: background
  quadrilateral_order: q1_reference
  excluded_boundary_ids: []
fields:
  coordinate_columns: [r_m, z_m]
  inside_column: inside_model_domain
  node_id_column: vertex_id        # ID がない export だけ null
  domain_scope:                   # 独立に inspected export scope を指定する。
    column: domain_id
    mode: equals_domain_id
    evidence: domain-side export record path and sha256 binding
  coordinate_match_tolerance_m: 1.0e-14  # export 精度と geometry scale から決める。
  mappings:
    - name: electric_field
      columns: [Er_kV_m, Ez_kV_m]
      components: [r, z]
      stored_basis: axisymmetric_rz
      association: node
      input_unit: kV/m
      unit: V/m
      scale: 1000.0
      offset: 0.0
      producer_record:
        producer: COMSOL 6.4 saved-data export
        dataset: actual model-level dataset tag and observed parent chain
        solution: actual model-level solution tag
        member: {indexing: one_based, inner: 4, outer: 1, parameter_values: {}}
        time: {representation: static, selection: steady member 4}
        domain_ids: [12]
        side: domain
        recovery: exported selected native E; actual setting from extraction record
        continuity: continuous
        authority: supplied native primitive
        topology: fixed
        support: full_selected_domain
        evidence: extraction record path and sha256 binding
      axis_transform: null
```

`domain_scope` は null（行の事前 filter なし）、または明示した column / mode / evidence です。
`evidence` はcallerが与える非空のprovenance文字列で、configuration hashに含まれます。
converterはその文字列内のpath/hashを解釈・照合せず、native selectionの真正性を認定しません。
準備者が既存inspection recordの実ファイルhashとdomain/side/export意味を独立に確認してください。
canonical node coverageの一致だけではそのnative scope証拠になりません。
`equals_domain_id` は unit `1` の domain ID 列、`finite` は選択 domain にだけ存在することを inspected record と
準備者がhashを照合したnative scope証拠が認定するdomain-anchor列に限ります。
証拠が無ければcallerの宣言に従う数値変換に留まり、native domain転送の認定は未検証です。
有限値だけから物理 support を推論する一般規則ではありません。
旧 multi-domain CSV に node/domain-side ID がない場合の explicit producer scope であり、gas-temperature fallback はありません。
いずれの mode も mesh node coverage、surviving rows の exactly-one coordinate / scoped ID 対応、選択 primitive の有限性を
別々に検査します。scope filter 後に重複が残る場合は曖昧として拒否します。

`producer_record` は各 field の producer、dataset、solution、member（indexing を含む）、time、domain IDs、side、
recovery、continuity、authority、topology、support、evidence を一つの provenance owner へ結びます。
追加 JSON 項目で reader/version、requested / actually accepted native unit、RF / phase convention、derivation input、
approximation、probe receipt の参照を保持できます。member や recovery が既存 export に記録されていない場合は
`not recorded in existing export` と明示し、完全な native qualification と見なしません。

axis correction は default null です。必要な変換だけ、`component`、canonical unit の `max_abs_correction`、物理的な
`reason` を指定してください。tool は raw axis 値・external node ID・変更 node 数・影響する P1 cell・最大値・
二乗和を記録します。補正後の incident cell 内の off-axis 値も変わります。native symmetry-consistent re-export を
優先し、変更の科学的採否は case preparation が判断します。null は値を保存し、必要な regularity は core 検査が認定します。

対象 row は volume domain の mesh node identity / coordinate coverage で決め、選択 domain の全 node で全要求値の
有限性を検査します。domain 外の plasma-only / thermal-only finite row と raw NaN は捨て値として埋めず、入力 hash と
provider / selected / unselected row 件数に結びます。domain 内の欠損、重複 ID、座標の曖昧性、unit 不一致、boundary
inventory 不一致は失敗します。有限値であることや inside column だけを domain の物理的証明にしません。

## 実行・移行例

solver project 内から、既存 file を上書きしない形で実行します。

```powershell
uv run --locked python -m tools.comsol_adapter `
  tools/comsol_adapter/cases/f02_case_a_100nm.yaml `
  _out_f02_acceptance/case_a_selected.h5 `
  --report _out_f02_acceptance/adapter_report.json
```

F02 例は thermal field production 接続を調べる参照入力です。元 export の native member / recovery は未記録である
ことをそのまま記録します。chamber ID、物性、粒径、境界、tolerance、axis correction を新 case の default にしません。
出力を `tools.electrostatic_builder` に渡す既存接続は維持します。A/P の plasma / derivative selected-field export も同じ
mapping の列を増やせば転記でき、case 専用の別 field writer は必要ありません。native heat flux / screening 等の
別 primitive を producer が生成する場合は、その生成根拠を `producer_record` に結びます。

通常 test は独立した小型合成 CSV を使います。XY affine vector の off-node 評価、SI scale / temperature offset、
node ID join と coordinate bijection、明示 axis transformation、欠損・不支持 gate を検証します。
F02 の参照 inventory は 1,987 node、3,779 P1 triangle、160 physical facet です。変換成功を mesh convergence、
COMSOL particle parity、実験的妥当性と解釈しません。旧 F02 / A/P campaign の evidence は当時の変換結果の履歴です。
