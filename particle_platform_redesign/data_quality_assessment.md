# `model_dataset` データ品質評価

## 1. 結論

`model_dataset/cf4_o2_etch_caseA_nonlinear_sass` は、**COMSOL の保存済み粒子軌道を時系列で調査する参照資料**、および **新しいソルバーの入出力・ジオメトリー・比較手順を設計する材料**として有用である。一方、現状のまま「外部ソルバーへ投入すれば 10/30/100 nm の各ケースを再現できる、自己完結した正解入力」とみなしてはならない。

判定を要約すると次のとおりである。

| データ群 | 判定 | 根拠と条件 |
|---|---|---|
| 粒子時系列・初期状態・最終状態 | 条件付きで基準に使用可 | 12 ケースすべてでキー、粒子数、保存時刻、状態フラグ、主要な派生量が内部整合。Brownian 項を含むため、異なる乱数実装との粒子単位完全一致を合格条件にしてはならない |
| ジオメトリー・接続・境界 ID | 条件付きで基準に使用可 | 全 12 ケースで同一ハッシュ、接続範囲正常、境界 ID の 0/1 始まり対応も整合。ただし四角形の周回順はCSV列順ではなく`node1,node2,node4,node3`である |
| 背景場の原始量（速度、温度、密度、電場等） | 条件付きで使用可 | 値そのものは利用できるが、領域支持、重複座標、格子座標ゆらぎを正規化してから使う。NaN や `inside_model_domain` から形状を復元してはならない |
| 背景場の粒子依存派生量 | **入力として使用禁止** | 30/100 nm パッケージでも `Kn`、Epstein 応答時間、Brownian 拡散係数が実質 10 nm 条件。原始量と粒子属性から実行時に再計算する必要がある |
| 境界モデルの網羅的検証 | 不可 | Stick/Disappear の一部実績はあるが、反射は未設定、Freeze 到達は 0、表面放出は存在しない。各境界 ID の正例も揃っていない |
| 確率モデルの統計検証 | 不可 | seed は記録されているが、各条件 1 realization のみ。分布、信頼区間、乱数更新規約の検証に必要な反復がない |
| 3D・時間依存場・大規模性能評価 | 不可 | 2D 軸対称・定常場・287 粒子のデータであり、対象機能を直接検証しない |

最重要の判断は、**参照軌道と外部入力場を同じ品質レベルで扱わないこと**である。軌道表内の粒径、質量、軌道上の局所 Knudsen 数は 10/30/100 nm に整合する一方、独立入力用の背景場ファイルに含まれる粒子依存列は整合しない。これは、保存済み COMSOL 軌道全体が破損していることを意味しないが、外部再計算用入力には修理が必要である。

## 2. 評価範囲と証拠の区分

対象は [データセット本体](../model_dataset/cf4_o2_etch_caseA_nonlinear_sass/) のうち、2 種類の ion-drag 定義 × Case P/A × 粒径 10/30/100 nm、計 12 ケースである。MPH 内部の Solution ベクトルを再計算・再監査したものではなく、公開された CSV、`mphtxt`、設定表、説明文書を対象とした静的監査である。

この文書では、証拠を次のように区別する。

- **測定事実**: 監査スクリプトが CSV/接続表を直接読み、件数、キー、値、式の恒等関係、ハッシュを再計算した結果。
- **データセット宣言**: COMSOL 設定表、manifest、付属文書に記載されている内容。宣言だけでは保存済み解と一致する証明にはしない。
- **推論**: 測定事実から考えられる原因。原因が COMSOL 内部状態まで追跡できない場合は断定しない。
- **品質判定**: 新しい粒子計算基盤の入力または V&V 基準として採用できるかという本評価の判断。

主な再現可能証拠は以下である。

- [全ケース監査スクリプト](analysis/audit_model_dataset.py)
- [全ケース監査 JSON](evidence/dataset_audit.json)
- [ケース別要約](evidence/case_audit_summary.csv)
- [背景場のケース間冗長性監査](analysis/audit_field_redundancy.py)
- [背景場の比較結果](evidence/field_redundancy_audit.json)
- [集団状態の時系列](evidence/population_time_series.csv)
- [ion-drag 2 方式の時系列乖離](evidence/variant_divergence_time_series.csv)
- [粒子別乖離要約](evidence/variant_particle_divergence_summary.csv)

付属説明のうち、特に [物理・数値手法](../model_dataset/cf4_o2_etch_caseA_nonlinear_sass/doc/PHYSICS_AND_NUMERICAL_METHODS.md)、[ジオメトリー・境界](../model_dataset/cf4_o2_etch_caseA_nonlinear_sass/doc/GEOMETRY_DOMAINS_AND_BOUNDARIES.md)、[外部再現手順](../model_dataset/cf4_o2_etch_caseA_nonlinear_sass/doc/EXTERNAL_REPRODUCTION_WORKFLOW.md) を宣言情報として参照した。

## 3. インベントリー

### 3.1 物理ファイル

再帰走査の測定結果は 590 ファイル、2,963,911,950 byte である。

| 種別 | 件数 |
|---|---:|
| CSV | 300 |
| PNG | 108 |
| log | 73 |
| Java source / class | 25 / 25 |
| status | 25 |
| Markdown | 18 |
| `mphtxt` | 12 |
| MPH | 2 |
| その他 | 2 |

ファイル数やバイト数はデータ品質そのものを保証しない。特に 12 ケースに同じジオメトリーとほぼ同じ背景場が複製されており、物理的に独立な情報量はファイル容量よりかなり小さい。

### 3.2 ケース構成

全 12 ケースには次が揃っている。

- `config/`: パラメータ、変数、物理 feature、solver、境界条件、列辞書
- `geometry/`: SI 頂点、境界辺、三角形・四角形接続、元 `mphtxt`
- `input_fields/`: 元メッシュ点版と 301×301 規則格子版
- `raw_comsol/`: 状態・力・局所場に分けた COMSOL 直接 export
- `results/`: 結合済み粒子時系列、初期状態、最終状態、境界状態表
- `validation/`: パッケージ作成時の構造検証結果
- `manifest.csv`: ケース識別と主要ファイル対応

軌道表は各ケース 287 粒子 × 121 保存時刻 = 34,727 行で、時刻範囲は 0–0.03 s である。Case P は 85 列、Case A は 86 列を持つ。

## 4. 信頼できる内容

### 4.1 粒子時系列の粒度とキー

全 12 ケースについて次を直接確認した。

- `particle_id,time_s` の重複は 0。
- 各粒子はちょうど 121 行、121 個の保存時刻を持つ。
- 粒子ごとの時刻は単調で、全ケースの保存時刻列は同一ハッシュである。
- `release_state_t0_tidy.csv` は時系列の `t=0` 行と完全一致する。
- `particle_final_summary_tidy.csv` は各粒子の最終保存行と完全一致する。
- 列辞書は時系列列を過不足なく被覆し、辞書内の重複列名はない。

したがって、時系列比較では `particle_id,time_s` を正規キーとして利用できる。ただし、境界イベント時刻は必ず `stop_or_event_time_s` 等のイベント情報を使い、保存時刻間のどこで衝突したかを最終保存座標から推定してはならない。

### 4.2 状態と欠損値

状態 code と Active/Freeze/Stuck/Disappear の one-hot flag の矛盾は全ケースで 0 であった。Active 行の必須値欠損も 0、数値列の無限大も 0 である。

Case A には必須値の null が存在するが、監査上はすべて `current_status_code=4`、すなわち Disappear 後の保存行に限られた。これは、消滅した粒子の後続座標・速度を NaN とする表現として整合的である。したがって null 件数だけをデータ破損と判定してはならず、状態と組み合わせて解釈する必要がある。

### 4.3 基本的な物理恒等関係

全ケースの時系列について、次が倍精度の丸め誤差水準で一致した。

- `speed = sqrt(vr^2 + vphi^2 + vz^2)`
- `charge_C = charge_number_e × e`
- `radius = diameter/2`
- `mass = 2200 × pi × diameter^3 / 6`
- 各方向の `acceleration = sum_exported_force / mass`

最大相対誤差は、速度で約 2.2×10^-14、電荷で約 3.8×10^-14、半径で約 1.3×10^-16、加速度成分で約 1.2×10^-14 以下、質量は監査式と完全一致した。これは列の結合ずれや粒径の取り違えが**軌道表内にはない**ことを強く支持する。

### 4.4 ジオメトリーと接続

全 12 ケースで、主要 7 ジオメトリーファイルの SHA-256 はそれぞれ 1 種類だけであった。直接測定した構造は次のとおりである。

| 項目 | 値 |
|---|---:|
| メッシュ頂点 | 7,732 |
| 境界辺 | 726 |
| 三角形 | 10,481 |
| 四角形 | 2,426 |
| domain ID | 1–11 |
| COMSOL boundary ID | 1–47 |
| 座標範囲 | `0 <= r <= 0.25 m`, `0 <= z <= 0.22 m` |

node ID は 0 始まりで連続し、接続表の node はすべて 0–7,731 の範囲内である。`boundary_edges.csv` の `comsol_boundary_id = mphtxt_boundary_index + 1` という関係にも違反はない。

したがって、領域 3 の三角形と四角形を明示的に読み、下記5.7の四角形順序を適用し、境界辺の COMSOL ID を利用する限り、ジオメトリーは新基盤の基準入力として採用できる。逆に、包絡矩形、規則格子の NaN、点群の凸包から移動可能領域を推定する必要も、推定してよい理由もない。

## 5. 重大な欠陥と曖昧さ

### 5.1 背景場の粒子依存列が全粒径で実質 10 nm

これは本データセットで最も重要な入力品質問題である。

背景場列辞書は、例えば次を粒径 `d0` の関数として宣言している。

- `particle_Knudsen_number = gas_mean_free_path / d0`
- `Epstein_response_time_s = rho_p × d0^2 / (18 × viscosity)`
- `Brownian_diffusivity_m2_per_s = k_B × T / (3 × pi × viscosity × d0)`

しかし [背景場比較結果](evidence/field_redundancy_audit.json) では、同じ Case の規則格子版が 10/30/100 nm の間で全数値列完全一致となった。実データの最初の有限点では、P/A と粒径表示によらず

```text
gas_mean_free_path / particle_Knudsen_number ~= 1.0e-8 m
```

となり、背景場の `particle_Knudsen_number` がすべて実効 10 nm で評価されたことを示す。`Epstein_response_time_s` と `Brownian_diffusivity_m2_per_s` も粒径間で同じ値である。本来、同じ気体場なら 30/100 nm に対し、おおむね次の倍率変化が必要である。

| 派生量 | 30 nm / 10 nm | 100 nm / 10 nm |
|---|---:|---:|
| Knudsen 数 | 1/3 | 1/10 |
| Epstein 応答時間 | 9 | 100 |
| Brownian 拡散係数 | 1/3 | 1/10 |

この一方で、軌道・初期状態表の `particle_diameter_m` はそれぞれ 1×10^-8、3×10^-8、1×10^-7 m、質量は径の 3 乗則に一致し、軌道上の `local_gas_mean_free_path_m / local_particle_Knudsen_number` から復元される径も各ケースの 10/30/100 nm に一致する。

以上からの品質判定は次のとおりである。

- **測定事実**: 独立入力用背景場の粒子依存 3 列は粒径別になっていない。
- **測定事実**: 保存済み軌道表内の粒径、質量、局所 Knudsen 数は粒径別である。
- **推論**: 背景場 export 時に、保存済み粒子 Study のパラメータではなく、10 nm に復帰した可変 global parameter が評価された可能性が高い。付属 Java に `d0` を 10 nm へ戻す処理があることとも整合するが、CSV 監査だけでは COMSOL 内部の正確な評価順序までは断定できない。
- **使用規則**: 背景場からは気体・プラズマ・電場等の**粒子によらない原始量だけ**を読み、Knudsen 数、応答時間、拡散係数、その他の粒子依存量は粒子属性と原始量から実行時に再計算する。

これらの派生列を 30/100 nm 入力としてそのまま使用すると、抗力、確率力、時間スケールを誤らせる。列名やケースディレクトリ名が正しくても、値が正しいとは限らない典型例である。

### 5.2 背景場のケース間重複

規則格子版は、同じ P/A・粒径における ion-drag 2 variant 間でも数値的に完全一致した。Case A の元メッシュ点版も variant 間・粒径間で完全一致した。Case P の元メッシュ点版には一部差があるが、多くは電場の約 10^-12 V/m、密度の 0.5–12 /m^3 など、元の値のスケールに対して極小で散発的な差である。

この微差を物理差とみなせる証拠はない。異なる export の浮動小数丸め差である可能性が高いが、それも推論である。重要なのは、ion-drag の variant は力モデルの差であり、背景場を 2 重に保存する必要はないことである。

新形式では、背景場を `field_set_id` で 1 回だけ保存し、粒径・力モデル・初期条件を別の run manifest から参照すべきである。ファイルハッシュの差だけで「異なる物理場」と判定せず、正規化後の数値比較と由来を併用する。

### 5.3 元メッシュ点版は「1 行 = 1 node」ではない

各ケースの `background_fields_mesh_points.csv` は 33,449 行だが、座標を小数 12 桁に正規化すると一意座標は 7,732、ジオメトリー頂点数と完全一致する。したがって 25,717 行は重複座標である。生の浮動小数値で数えた重複行は 25,710 であり、差の 7 行は座標表現の微小ゆらぎに起因する。

ガス速度、r 方向電場、ガス温度の 3 列については、同じ正規化座標に複数の有限値がある場合の値衝突は 0 だった。ただし、これは監査した 3 列に限る。全列が常に同一、あるいは将来の export でも同一であることは証明していない。

したがって、33,449 行を接続表の node ID と位置で対応付けたり、先頭 7,732 行を node 値として使ったりしてはならない。少なくとも次が必要である。

1. 座標をジオメトリー頂点へ許容誤差付きで照合する。
2. 同一点の複数候補について、有限値の個数と相互差を列ごとに検査する。
3. 値が一致する場合のみ 1 node 値へ縮約する。
4. 異なる有限値があれば、domain/selection/side 情報なしに平均せず export 不備として停止する。

重複が生じた COMSOL 側の厳密な理由は、現ファイルだけからは確定できない。domain・境界・要素側の評価を同じ座標へ重ねた可能性はあるが、推論を読み込み規則に埋め込んではならない。

### 5.4 301×301 格子の座標ゆらぎ

規則格子は 90,601 行で座標組の重複はない。一方、生の値で一意な r は 476、z は 462 あり、「301×301」という論理格子と一致しない。小数 12 桁へ丸めると r/z とも 301 になる。

これは微小な数値表現差である可能性が高いが、単純な `unique()` や浮動小数値の完全一致を格子 index 作成に使うと形状を誤る。12 桁丸めを恒久仕様として埋め込むのではなく、export に整数 `i,j`、原点、格子間隔、期待 shape を持たせ、座標はそれから再構成するべきである。既存データの変換時だけ、格子間隔に対して十分小さいことを検証した上で index へ量子化する。

### 5.5 `inside_model_domain` と NaN は領域マスクではない

列辞書上の `inside_model_domain` の COMSOL 式は定数 `1` であり、規則格子の 90,601 行すべてが 1 である。これは「粒子が動ける domain 3」を表していない。

また、ガス速度などは規則格子 46,530 行で有限だが、熱伝導率、熱容量、電子温度など一部列は 90,601 行で有限である。このため「全列 NaN の行」は 0 であり、行全体の NaN 判定から固体・外部・粒子領域を分類できない。ある 1 つの物理量の有限性を領域マスクに流用することも、その物理量の定義域と粒子 domain が将来一致する保証がない。

移動可能領域は `domain_triangles.csv` と `domain_quadrilaterals.csv` の `domain_id=3`、境界は `boundary_edges.csv` を唯一の幾何学的根拠とする。値の支持領域とジオメトリーの内外判定は別のデータとして保持する。

### 5.6 manifest と保存済み解の来歴は十分ではない

manifest はケース名、dataset 名、ファイル対応、粒子数、保存時刻数を記録し、各ケースの `global_parameters.csv` も 10/30/100 nm を宣言する。一方、manifest は export 時に Study/Solver を実行せず、背景場も粒子も再計算していないことを明記している。

これは非侵襲な export という長所である反面、次の連鎖を機械的には証明しない。

```text
source MPH hash
  -> COMSOL version/build
  -> solution tag / dataset / parameter point
  -> export expression / selection / evaluation context
  -> generated CSV hash
```

今回の粒径派生列問題は、この来歴不足が実害につながった例である。設定表の `d0=100 nm` と、背景場の式を 100 nm 文脈で評価したことは同義ではない。

### 5.7 四角形の局所 node 順が未記載

`domain_quadrilaterals.csv` の4列を、ファイルに並んだまま
`[node1,node2,node3,node4]` の周回順と解釈すると、2,426要素中2,019要素が自己交差または
面積ほぼ0、314要素が負向きになる。全要素が正向きになる周回順は次である。

```text
[node1, node2, node4, node3]
```

この順序での面積範囲は約`5.82e-9 ... 1.35e-5 m2`である。node ID自体や接続範囲が正常でも、
local orderingを誤ればdomain判定、形状関数、Jacobian、境界横断が破壊される。これは外部実装に
とってHigh severity、High confidenceの契約欠落である。

新しいexportには`element_type`、`local_node_ordinal`、COMSOL側のordering名を含める。現在の
bundle converterは`[1,2,4,3]`を明示し、変換時に全quadの自己交差、向き、Jacobianを検査する。
自動的に座標角度順へ並べ替える処理は、元の形状関数DOF対応を失うため採用しない。

### 5.8 軸対称運動に使われない方位 Brownian 力

`velocity_phi_m_per_s`は全有限行で0であり、実solver logの運動従属変数もr/zだけである。一方、
`Brownian_force_phi_N`は全有限行で非0で、10/30/100 nmの最大絶対値はそれぞれおよそ
`2.42e-16 / 6.86e-16 / 2.21e-15 N`である。この成分は
`sum_exported_forces_phi_N`、3D force magnitude、phi accelerationへ含まれる。

従って、保存された方位力は診断用3D random force sampleであり、axisymmetric no-swirlの
trajectory DOFへは積分されていないと判断できる。外部solverが合力magnitudeまたはphi加速度を
3自由度運動へ適用すると、COMSOLとは別の問題を解く。

比較では決定論力とBrownianのr/z成分を運動の権威とし、phiを含むmagnitudeは別名の診断量として
扱う。将来のexportは「solverへ投影されたforce」と「診断用3D force realization」を別列群にする。

## 6. ベンチマーク被覆の不足

### 6.1 初期条件は表面放出ではない

付属物理文書と初期状態表は、粒子を時刻 0 に

```text
r = 0.14 ... 0.18 m（0.001 m 間隔、41 点）
z = 0.023 ... 0.026 m（0.0005 m 間隔、7 点）
```

の 41×7 = 287 点へ配置する。これは domain 3 下部の**内部格子**であり、材料表面からの放出ではない。初速も局所ガス速度の大きさと、位置から決める ±15° の決定論的角度規則で与えられる。

したがって、このデータは表面三角形/辺の面積重みサンプリング、放出側の法線、所有 cell、初期接触の二重判定、表面フラックス重み、表面粗さ・脱離速度分布を検証しない。主用途の「パーツ表面を発生源とする粒子」に対する専用ケースが別途必要である。

### 6.2 境界条件の正例が不足する

宣言上の境界は Axis、Stick、Disappear、Freeze であり、反射 feature は存在しない。実測された最終状態では、Stick は 3 ケース、Disappear は Case A の 6 ケースで見られる一方、Freeze は全ケース 0、reflection flag の総和も全ケース 0 である。

よって本データセットだけでは次を検証できない。

- 鏡面反射、拡散反射、反発係数、付着確率の Monte Carlo 分岐
- Freeze 境界 37 の実到達時刻と位置
- 同一 step 内の複数衝突、角・頂点への衝突、接線接触
- 軸 `r=0` の座標処理
- 各 Stick boundary ID の個別 hit
- 壁面から放出直後の再衝突回避

最終状態に Stick が含まれることは、すべての境界判定が正しい証拠ではない。境界 V&V には、境界 ID ごとの単一イベント microcase と、交点・時刻・入射速度・法線・反射後速度・状態遷移を記録した event ledger が必要である。

### 6.3 確率モデルの反復がない

Brownian seed parameter と 10 microsecond の更新間隔は設定表に記録され、時系列には Brownian 力も含まれる。しかしinterfaceは`GenerateUnique` modeであり、記録されたparameter値を実効乱数seedとは認定できない。各物理条件は1 saved realizationだけで、同一条件の独立replicateがない。

COMSOL と同じ乱数生成器・粒子/component への割当・step 更新順を外部コードが再現できない限り、個々の Brownian 軌道の点ごとの一致は期待できない。現在のデータは COMSOL 側 1 realization の監査には使えるが、平均二乗変位、到達確率、付着率、分位点、分布距離の統計的不確かさを評価するには不足する。

必要なのは、Brownian 無効の決定論的 companion、固定 seed の再現ケース、複数 seed の統計ケースを分けることである。

### 6.4 対象範囲外の機能

本データセットは 2D 軸対称、定常背景場、287 粒子、0.03 s のケースである。次の将来機能に対する合否判定には使えない。

- 時間変化する 2D 場と time interpolation
- 3D 非構造メッシュ、三角形表面、四面体/六面体 cell
- 軸対称から 3D 粒子集団へ展開する方位角・統計重み
- 10,000–1,000,000 粒子の CPU/GPU 性能とメモリ
- moving mesh、場の不連続時刻、複数 file slice
- 外部熱流体場から内部 Poisson を解く field builder

これらについて「既存ベンチマークに通ったので対応済み」と主張してはならない。

## 7. 許可する用途と禁止する用途

### 7.1 許可する用途

条件を明記すれば、次に利用できる。

1. COMSOL 軌道の `r(t),z(t),v(t),Z(t),F_i(t)` を同じ保存時刻で追跡する時系列 V&V。
2. 初期状態を 1 粒子ずつ一致させる replay 入力。
3. 境界 ID、domain 3、混合三角形/四角形の geometry importer 検証。
4. 原始場の node/regular-grid sampler 検証。ただし正規化と支持判定を分離する。
5. 2 種類の ion-drag 定義が軌道へ与える感度の調査。
6. schema、単位、来歴、イベント出力の新しい canonical case 形式を設計する反例。
7. 既知の最終状態集計と時系列人口の回帰確認。

特に ion-drag variant の比較では、最終状態が同じでも途中軌道は大きく離れうる。[variant の時系列乖離](evidence/variant_divergence_time_series.csv) を使い、終点だけでなく最初に乖離する時刻とその直前の場・力・電荷を調べる必要がある。

### 7.2 禁止する用途

次の使い方は禁止する。

- 30/100 nm ケースで背景場の `particle_Knudsen_number`、`Epstein_response_time_s`、`Brownian_diffusivity_m2_per_s` を入力値として採用する。
- `inside_model_domain`、全列 NaN 判定、凸包、矩形包絡から particle domain を作る。
- `background_fields_mesh_points.csv` の行番号を node ID とみなす。
- 301×301 座標を浮動小数の完全一致だけで index 化する。
- 四角形を無視する、または補間規約を記録せず自動三角形分割する。
- 単一 Brownian realization の粒子単位一致を、確率モデル全体の正しさとみなす。
- 最終座標、最終状態、集団平均だけで軌道再現を合格とする。
- このケースだけで反射、Freeze、表面放出、3D、時間依存場を検証済みとする。
- 287 粒子の処理時間から 10^4–10^6 粒子や COMSOL 比の高速性を外挿する。
- どちらかの ion-drag variant を実験検証なしに「物理的真値」とする。

## 8. export 修理要件

### 8.1 原始量と派生量を分離する

canonical field package は、粒子属性に依存しない原始量だけを保存する。例はガス速度、温度、圧力、密度、粘度、平均自由行程、電位、電場、電子/イオン密度、電子/イオン温度、イオン速度である。

次は field package に固定値として保存せず、run ごとに粒子属性から計算する。

- Knudsen 数
- Epstein/Stokes 応答時間と補正係数
- Brownian 拡散係数・摩擦係数
- 粒子半径/質量
- 表面電位増分など粒径・電荷依存量
- ion-drag の断面積、Coulomb logarithm、screening cutoff

診断目的で派生量を export する場合は `derived=true`、依存パラメータ、式、評価時の数値を列 metadata に持たせる。同じ値を solver 入力として二重に権威化しない。

### 8.2 mesh field を node ID 付きで export する

理想形式は 1 node 1 行で、明示的な `node_id,r_m,z_m` を持つ。共有 node に domain ごとの不連続値が必要なら、無名の重複座標ではなく

```text
node_id, domain_id, side_id, quantity_id, value
```

のように差の意味を表す。量が node、element、integration point、boundary のどこに属するか、補間次数と component basis も metadata に含める。

変換段階では、座標照合、有限値衝突、全頂点被覆、余分座標、domain support を機械検査し、曖昧なら平均や nearest で修復せず失敗させる。

### 8.3 規則格子へ論理 index と支持を持たせる

規則格子は `i,j`、shape、原点、間隔、座標単位を持たせる。場の値支持は quantity ごとの support mask、幾何学領域は domain ID/cell locator として分ける。

格子外、domain 外、場の未定義、材料側を同じ NaN 1 種類へ畳み込まない。NaN は値欠損表現であって、境界ジオメトリーではない。

### 8.4 来歴を閉じる

各 export bundle に少なくとも次を必須化する。

- schema version、case ID、run ID、作成時刻
- source MPH の SHA-256
- COMSOL version/build、OS、export tool version/commit
- component、study、solution、dataset、parameter point の tag と表示名
- 座標系、長さ単位、vector basis、軸対称/no-swirl の別
- mesh ID/hash、element type/order、node ordering
- expression、単位、selection、evaluation location
- time/parameter index、averaging・smoothing・recovery・complex 値処理
- particle solver、内部 step、保存時刻、許容誤差、wall accuracy order
- RNG algorithm、seed、particle/component/step への割当規則
- 生成した全 artifact の SHA-256

export 後には「manifest の `d0`」ではなく、**各粒子依存列から逆算した `d0` が期待値と一致すること**を検査する。設定値と出力値の両端を結ぶ invariant が必要である。

### 8.5 sparse event ledger を追加する

現在の `particle_boundary_event_history_tidy.csv` は保存時刻と同じ 34,727 行を持つ。状態時系列としては利用できるが、イベントそのものを一意に表す sparse ledger を別に持つ方がよい。

推奨列は次である。

```text
particle_id, event_ordinal, event_time_s,
boundary_id, facet_id, hit_r_m, hit_z_m,
pre_vr, pre_vz, normal_r, normal_z,
post_vr, post_vz, outcome, random_u, state_before, state_after
```

これにより、保存時刻とは独立に first-hit と wall law を検証できる。

## 9. 追加すべきベンチマーク

現データはシステムレベルの複合ケースとして残し、足りない挙動を小さい独立ケースで補う。

| 目的 | 必要な追加ケース | 主な合格指標 |
|---|---|---|
| 表面放出 | 平面・曲面・軸対称面からの既知分布放出 | source facet、所有 cell、法線、面積/`2pi r` 重み、初期再衝突 0 |
| 境界 first hit | 1 本の平面、角、薄い障害物、同一 step 複数候補 | event 時刻・位置・boundary ID |
| 反射 | 鏡面、反発係数、拡散反射、付着確率 0/0.5/1 | pre/post 法線・接線速度、確率頻度 |
| Freeze/Disappear | 各境界へ解析的に到達する粒子 | 状態遷移と後続出力 |
| 軸 | `r=0` を横切る/接するケース | 非物理 wall hit 0、座標変換整合 |
| 補間 | P1 三角形、Q1 四角形の解析場 | node・内部点・辺での誤差 |
| 決定論的積分 | 一様加速度、線形抗力、既知電場、電荷緩和 | 解析解と収束次数 |
| Brownian | 自由拡散・Ornstein–Uhlenbeck、複数 seed | 平均、分散、MSD、信頼区間 |
| 時間依存場 | 2 slice 線形場、不連続時刻 | time interpolation と step split |
| 3D | 単純 tetra/hex と三角形壁 | cell walk、segment-triangle hit |
| 性能 | 10^4/10^5/10^6 粒子、出力量固定 | warm/cold 時間、峰値メモリ、各処理内訳 |

複合 COMSOL ケースで差が出た後に原因を推測するのではなく、この microcase 群で補間、力、積分、イベント、乱数を個別に固定する。

## 10. 再現手順

以下の値は監査時点のreview済みsnapshotである。旧root環境は`old_code/`へ退避済みであり、
それを再利用しない。P00より前の`analysis/`は監査方法を保存するsourceであって実行入口ではない。
P00で新solverのlocked uv環境を作る際、必要な監査dependencyと再生成commandを一箇所に定義してから、
同じlockで再生成する。plain `python`やarchive済み`.venv`からの再生成は禁止する。

同じ入力に対し、少なくとも次を確認する。

1. `dataset_audit.json` の `case_count` が 12。
2. 各ケースが 34,727 行、287 粒子、121 時刻、duplicate key 0。
3. `cross_case.unique_time_schedule_hashes` が 1 種類。
4. 各ジオメトリーファイルの `unique_hash_count` が 1。
5. mesh field が 33,449 行、正規化後 7,732 座標。
6. regular grid が 90,601 行、正規化後 301×301。
7. `field_redundancy_audit.json` の規則格子 size 比較が `different_columns: []`。

監査出力は入力データが変われば更新される snapshot である。将来の修正版 export では、既存 snapshot を上書きして問題を隠すのではなく、dataset version を上げて新旧の監査結果を併存させる。

## 11. 受入条件

このデータセットを「外部再計算用 canonical benchmark」と呼べるのは、少なくとも以下を満たした後である。

- 背景場を原始量へ正規化し、粒径依存派生列を削除または正しい parameter 文脈で再 export した。
- 10/30/100 nm の派生量 scaling invariant が全 node/grid 点で通る。
- mesh field が node ID と field association を持ち、無意味な重複行がない。
- regular grid に論理 index と独立した support/domain mask がある。
- source MPH から各 CSV までの version/hash/provenance が閉じている。
- Brownian 無効の決定論的参照ケースと複数 seed 統計ケースがある。
- 表面放出、反射、Freeze、Axis、各主要境界の正例が追加されている。
- event ledger が保存時刻から独立して first-hit 情報を保持する。
- 変換後の canonical bundle に対し、読み込み、単位、補間、境界、時系列の end-to-end 監査が自動実行される。

それまでは、このデータセットを **「構造的に良好な COMSOL 軌道参照と、重要な export 反例を含む研究用ベンチマーク」**として扱うのが妥当である。新基盤の設計では、この反例を場当たり的な例外処理で吸収せず、原始量の単一権威、明示的 topology、値と支持の分離、閉じた来歴という入力契約へ変換するべきである。
