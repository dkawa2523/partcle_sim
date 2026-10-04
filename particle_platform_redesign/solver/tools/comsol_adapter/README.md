# COMSOL axisymmetric CSV adapter

この外部toolは、対応するCOMSOL mesh/field CSVをproducer-neutralなcanonical `case.h5`へ一方向変換します。
粒子engine、reduced electrostatic builder、COMSOL比較器のいずれでもありません。COMSOL固有の列名、entity ID、
mixed mesh規則はこのdirectoryだけが所有し、`src/chamber_particles/`からimportされません。

## v1の変換

- 指定した1個の2-D axisymmetric domainだけを抽出する
- providerのQ1 node順を明示的に要求し、各quadを二候補のうち最小triangle品質が高い対角でP1化する
- triangle edge incidenceから外周と唯一のownerを再構築する
- 明示した軸symmetry IDを物理boundaryから除外し、その全facetが両端`r=0`であることを検査する
- 残る全外周IDを設定中のsemantic groupへ過不足なく割り当てる
- node fieldを補間せず、設定tolerance内のexactly-one座標対応が全体で全単射となる場合だけ転記する
- gas velocity、temperature、density、dynamic viscosity、mean free pathをcanonical primitiveとして保存する
- 軸上のradial gas velocityは設定上限以下の場合だけ厳密な0へ投影し、補正件数と最大量をprovenanceへ残す

CSVの`inside_model_domain`はprovider exportの整合性確認にだけ使います。domain選択のauthorityはvolume-cellの
`domain_id`と、必要primitiveが有限なnode集合です。NaNを補間・最近傍埋めせず、座標対応が欠落、重複、曖昧なら
失敗します。quad品質が低いことだけを任意閾値で拒否せず、退化・反転・non-manifold topologyは拒否します。

新しいprovider exportはstable node IDをfield表にも含め、topology IDで結合することを原則とします。F02の既存CSVは
field node IDを持たないため、全canonical nodeと全provider pointが明示tolerance内で厳密に1対1となることを証明する
legacy経路だけを許可しています。これは丸めbucketや最近傍joinではなく、補間・欠損埋めも行いません。入力6ファイルは
一度だけbytes snapshotを取り、その同じbytesをhashとparseへ使います。YAML重複key、CSV headerの重複・順序違い、
行の列数違い、設定したboundary IDの過不足は成功扱いにしません。

境界`material_id`は設定値をそのまま保存します。参照datasetには信頼できる材料taxonomyがないため、F02例では
全groupを`0`とし、COMSOL boundary IDを`boundary_id`、export row IDを`external_id`として保持します。`gas_inlet`
と`pump_outlet`は静電位が同じでも粒子境界物理が異なるため統合しません。COMSOLの`Freeze`を別lawへ暗黙変換する
こともありません。

## 実行

solver project directoryから、既存fileを上書きしない形で実行します。

```powershell
uv run --locked python -m tools.comsol_adapter `
  tools/comsol_adapter/cases/f02_case_a_100nm.yaml `
  _out_f02_acceptance/case_a_thermal.h5 `
  --report _out_f02_acceptance/adapter_report.json
```

この出力を`tools.electrostatic_builder`へ渡すと、同じcanonical geometryとthermal primitiveを保持したまま、
potential、electric field、density、ion primitiveを追加できます。adapterはplasma parameterや境界電位を選ばず、
builderはCOMSOL CSVを読みません。

F02の代表builder設定は次で実行します。

```powershell
uv run --locked python -m tools.electrostatic_builder `
  tools/electrostatic_builder/cases/f02_case_a_100nm.yaml `
  _out_f02_acceptance/case_a_fields.h5 `
  --report _out_f02_acceptance/builder_report.json
```

## 参照caseの期待inventory

`f02_case_a_100nm.yaml`はfield-production接続を調べる代表入力であり、粒径やion-drag modelをsolverへ固定する
golden caseではありません。期待する構造量は次です。

- 1,987 node
- source triangle 2,127、source quad 826
- P1 triangle 3,779
- quad splitは対角03が561、対角12が265、最小triangle品質は`0.0423878260`
- 外周193 edgeのうちaxis 33 edgeを除外し、物理boundary 160 edge
- finite thermal row 1,987
- 軸上補正34 node、最大補正は約`1.5313e-4 m/s`

通常testは小型の合成CSVだけを使います。上記dataset実行、COMSOL field差分、代表規模linear solveは外部F02
受入であり、coreの常時testやsolver dependencyにはしません。

F02では、この出力をF01 builderへ渡した代表linear solveと、完成fieldを既存RZ solverへ渡すfixed-electric smoke、
同一export node上の記述的field比較まで完了しました。小さいhash付き証跡は`evidence/f02/`にあります。独立mesh
convergence、COMSOL trajectory、wall/Freeze parityは未検証であり、このadapterの成功をそれらの一致と解釈しません。
