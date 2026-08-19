# COMSOL からケースを組み立てる手順

この文書は手順書です。COMSOL のモデルと設定を出発点に、実行可能な case を作り、
結果が参照と比較できる状態にするまでを順に示します。各段階が強制する契約は
[`comsol_vv.md`](comsol_vv.md)、入力列と成果物は
[`input_artifacts.md`](input_artifacts.md)、物理と数値は
[`physics_numerics.md`](physics_numerics.md) にあります。

前提は一つだけです。**このコードは COMSOL を実行しません。** COMSOL 側で解いた場を
読み、その中で粒子を追跡します。したがって最初にやることは、COMSOL のモデルから
「何が解かれているのか」を確定させることです。

## STEP 0 — COMSOL 側で確定させる

export を始める前に、次を COMSOL のモデルツリーから読み取って記録します。推測で
埋めないでください。ここが違うと以降のすべてが静かにずれます。

### 0.1 解の同定

| 記録する項目 | COMSOL のどこ | 用途 |
|---|---|---|
| model 名 | ファイル名/`model.name` | provenance |
| study | Study ノードの tag（例 `std2`） | provenance |
| dataset | Results > Datasets の tag（例 `dset3`） | 評価対象の解 |
| solution | Solver の tag（例 `sol2`） | 評価対象の解 |
| solution number | パラメトリック解のどれか（1 始まり） | 評価対象の解 |
| mesh tag | Mesh ノードの tag（例 `mesh1`） | 節点座標と mphtxt の同一性 |
| parameter 名と値 | パラメトリック掃引の対象（例 `Vrf` = `20[V]`） | provenance |

**同じ mesh tag を export と mphtxt の両方に使ってください。** 節点値表は
`node_index`（mphtxt の全体頂点インデックス）で mesh に結合するため、両者が別の
mesh だと結合は成立しても意味が壊れます。

### 0.2 粒子領域の同定

粒子が動ける真空領域の **COMSOL domain 番号** を列挙します（`vacuum_domain_ids`）。
モデル全体の外周ではなく、この選択の境界が壁になります。全体外周を使うと、真空と
固体の界面（ウェハ、フォーカスリング、誘電体窓）が壁として消えます。

exporter はすべての `Interp` をこの選択に限定します。これにより真空/固体界面上の
節点が固体側から `NaN` で答えられることがなくなります。**空にできません。**

### 0.3 場の式と単位

粒子に効く量ごとに、COMSOL の式と単位を 1 つずつ決めます。

| semantic | 典型的な COMSOL 式 | 単位 |
|---|---|---|
| 流速 | `u`, `w`（軸対称なら r, z 成分） | `m/s` |
| 粘度 | `spf.mu` | `Pa*s` |
| ガス密度 | `spf.rho` | `kg/m^3` |
| 温度 | `ht.T` | `K` |
| 電場 | プラズマ/静電インタフェースが公開する電場変数 | `V/m` |

**電場は `-d(V,r)` のような微分演算子より、インタフェースが公開する電場変数を
優先してください。** 微分の段数が 1 つ減り、要素間で不連続な導関数を評価する
曖昧さが消えます。

時間平均量（`ptp.Vav` など）を使う場合、それが妥当かを明示的に判断してください。
ダストの運動時定数は RF 周期より桁違いに長いので運動には妥当ですが、帯電の緩和は
RF 周期に近く、⟨Q·E⟩ ≠ ⟨Q⟩·⟨E⟩ です。

### 0.4 境界条件

COMSOL の Wall ノードと、このコードの wall law の対応です。境界エンティティ ID
ごとに 1 行、`boundaries.csv` に書きます。

| COMSOL Wall condition | wall_law | 補足 |
|---|---|---|
| Freeze | `freeze` | 位置・速度を接触時で凍結 |
| Bounce | `specular` | `wall_restitution: 1.0` で弾性 |
| Stick | `stick` | |
| Disappear | `absorb` または `escape` | 消滅か流出かで使い分け |
| Pass through | `pass_through` | |
| Diffuse scattering | `cosine_diffuse` | 速度の大きさは保存 |
| Mixed diffuse and specular | `mixed_specular_diffuse` | `wall_diffuse_fraction` で割合 |

対応物がないもの:

- COMSOL の **General reflection**（ユーザ定義の反射速度式）、**Secondary
  emission**、**Thermal re-emission**（壁温 Maxwellian で速度を再サンプル）は
  未実装です。COMSOL 側で使っている場合、その境界は一致しません。
- このコードの `critical_sticking_velocity` は COMSOL に対応物のない追加法則です。

### 0.5 粒子物性とリリース

- **質量**: `mass_kg` が慣性の正本です。COMSOL の Particle Properties が密度と
  直径で与えられている場合、`mass_kg = ρ π d³/6` を満たす必要があります
  （preflight が 0.1% で検証します）。
- **リリース位置と初速**: COMSOL の release feature を再現する必要があります。
  `Release from Data File` を使っているなら、その表がそのまま `particles.csv` に
  なります。`Inlet` の `Mesh based` は境界メッシュ要素ごとに重心へ 1 個、
  初速は `Maxwellian` / `Constant speed, cosine-law` などの分布則です。手作業で
  CSV に写すと粒子数が同じでも配置と速度分布が一致しません。
- **電荷**: `charge_C` は固定値です。COMSOL 側でダスト帯電を解いていない限り、
  動的帯電は有効にしないでください（有効にすると乖離が増えます）。

### 0.6 力の構成

**COMSOL 側で有効な Force ノードを列挙し、それと 1:1 に合わせてください。**
実装済みは drag / electric / gravity / thermophoresis / dielectrophoresis /
lift / pressure_gradient / virtual_mass です。

イオンドラッグ力は未実装です。COMSOL 側が含んでいないなら追加不要ですし、
含んでいるなら現状では一致しません。

抗力則は COMSOL の Drag Force ノードの選択に合わせます。真空チャンバーの微粒子は
`Kn = λ/d` が 1 を大きく超えることが多く、その領域では `epstein` です
（preflight が `Kn ≤ 1` をエラーにします）。

## STEP 1 — export する

`external/comsol_icp_export/` の Java exporter を COMSOL 上で実行します。設定は
`config/*.json` で、STEP 0 で確定した値を書きます。

exporter は 1 回の評価で 2 つの表を書きます。

- `field_samples_nodes.csv` — mesh 節点座標での評価。**これがソルバの入力です。**
  `node_index` で mphtxt に結合します。
- `field_samples.csv` — 設定した `r`/`z` 格子での評価。同じ export の可読な参照用。

`export_manifest.json` が両者のハッシュと provenance を宣言します。

## STEP 2 — case を組み立てる

```console
particle-tracer comsol build-case \
  --raw-export-dir EXPORT_DIR \
  --out-dir CASE_DIR \
  --coordinate-system axisymmetric_rz \
  --diagnostic-grid-spacing-m 1e-3 \
  --release-table release.csv \
  --boundaries boundaries.csv \
  --drag-law epstein \
  --force electric \
  --gas-temperature-K 300 --gas-density-kgm3 4.3e-5 \
  --gas-molecular-mass-amu 39.948 \
  --dt-s 1e-4 --t-end-s 1.0
```

`--raw-export-dir` は manifest が `field_node_samples_sha256` を宣言していれば
mesh native、していなければ正則格子を選びます。ディスク上のファイルの有無だけでは
切り替わりません。

直接指定するなら `--mphtxt` と `--field-node-samples`（mesh native）または
`--field-bundle`（正則格子）です。両者は排他です。

境界上でリリースする場合は `--release-projection-tolerance-m` を宣言します。点は
宣言されたエンティティ上へ載せられ、そこに留まります（COMSOL の Inlet と同じ）。

生成される `run_config.yaml` は COMSOL 準拠の既定値を持ちます
（`wall_interaction.contact_sliding: false`）。

## STEP 3 — 検証する

```console
particle-tracer check CASE_DIR/run_config.yaml --full
```

`--full` は違反した粒子と境界の行を残します。よくある失敗と意味:

| issue code | 意味 | 対処 |
|---|---|---|
| `input.initial_geometry` | 初期位置が幾何の内部でも自身のリリース境界上でもない | 位置か `source_part_id` が誤り |
| `input.initial_field_support` | リリース位置で場が有効でない | mesh native なら mesh 外。正則格子なら stencil が領域外 node に触れている |
| `physics.drag.regime` | 宣言した抗力則がリリース状態で成立しない | `Kn`/`Re` を確認。真空なら `epstein` |
| `physics.particle.sphere_consistency` | `mass_kg` と `density × d³` が 0.1% 以上違う | 粒子物性の出所を統一 |
| `physics.force.field.missing` | 有効にした力が要求する場が manifest にない | 式を追加するか力を無効化 |
| `physics.gas.missing` | 抗力則が要求するガス物性がない | `--gas-*` を指定 |
| `comsol.boundary.coverage` | 幾何の境界 part と `boundaries.csv` の行が一致しない | エンティティ ID を突き合わせ |
| `comsol.time_support` | transient 場が積分区間を覆っていない | 定常場では発生しません |
| `inward_offset_m is obsolete` | 旧 manifest（manifest 検証で失敗） | キーを削除。境界上リリースは変位しません |

## STEP 4 — 実行する

```console
particle-tracer run CASE_DIR/run_config.yaml -o RUN_DIR
```

終端状態が結果の読み方を決めます。

| terminal_state | 意味 |
|---|---|
| `stuck` / `frozen` / `absorbed` / `escaped` | 壁法則が決めた終端。COMSOL と比較できる |
| `active_free_flight` | `t_end` まで飛び続けた |
| `invalid_mask_stopped` | 場のサポートを出た。正則格子なら壁の手前で消えた可能性 |
| `numerical_boundary_stopped` | 細分予算内で安全性を証明できなかった |
| `contact_sliding` | 接触正則化に入った。COMSOL に対応物なし |

`numerical_boundary_stopped` が多い場合は `time.max_substep_splits` を、
壁での bounce が多い場合は `physics.wall_interaction.max_hits_per_step` を
引き上げます。`contact_sliding` が出る場合は COMSOL ケースの設定が
`contact_sliding: false` になっているか確認してください。

## STEP 5 — 比較する

```console
particle-tracer compare field ...        # 場のサンプル値
particle-tracer compare first-step ...   # 初期加速度の内訳
particle-tracer compare trajectory ...   # 軌道
particle-tracer compare boundary ...     # 壁イベント
```

この順で切り分けます。場が合っていなければ軌道は合いません。場が合っていて初期
加速度が合わなければ力の構成が違います。両方合って軌道が違えば時間積分か境界です。

`compare trajectory` と `compare boundary` は debug 実行の成果物を要求します。

## 判断が必要な選択肢

既定値はすべて COMSOL 比較を前提に選ばれています。変更する理由は
[`README.md`](../README.md) の表にあります。特に:

- **field の保存形式**: mesh native が既定です。正則格子はモデル中で最も薄い物理層を
  自力で解像する必要があり、壁に隣接する cell は stencil が領域外 node に触れて
  粒子を停止させます。シース厚が格子刻みより薄い場合、格子では原理的に一致しません。
- **`contact_sliding`**: COMSOL ケースでは `false` です。`true` にすると、壁へ
  押し付けられた粒子が COMSOL には存在しない接触状態に入ります。

## 一致しないときの切り分け順

1. **場のサンプル値** — `compare field`。ここが違えば以降は無意味です。式・単位・
   solution 番号・座標スケールを疑います。
2. **初期加速度** — `compare first-step`。力の構成が COMSOL の Force ノードと
   1:1 か。抗力則、電荷、重力の向き。
3. **時間積分** — `time.dt` と `max_substep_splits`。COMSOL 側の相対許容誤差も
   確認します（既定は緩いことが多く、こちらの方が精密な場合があります）。
4. **境界** — `compare boundary`。wall law の対応、`contact_sliding`、
   `max_hits_per_step`。
5. **残差** — ここまで合って残るなら物理モデルの差です。未実装の力
   （イオンドラッグ、壁近傍抗力補正、thermal re-emission）を疑います。
