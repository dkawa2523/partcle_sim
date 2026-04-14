# ICP RF Bias CF4/O2 2D Particle Benchmark Report

作成日: 2026-04-14
対象ケース: `icp_rf_bias_cf4_o2_si_etching_2d_source_part11_sio2_20nm_charged_tend10x`
出力先: `_out_icp_cf4_o2_v20_10k_source_part11_sio2_20nm_charged_tend10x`

本レポートは、COMSOL由来の2D r-z相当場を使い、ウェハーエッジ近傍の部品表面から発生した20 nm SiO2粒子の輸送・壁衝突・付着を評価するためのベンチマーク資料である。
PowerPointへ部分的に貼り付けやすいよう、各節は図・式・短い解釈をセットにしている。

---

## 1. ベンチマークの問題設定

### 1.1 対象現象

対象は、ICP RF bias CF4/O2 Si etching装置断面における、ウェハーエッジ横の部品表面から発生した微粒子の飛散・反射・付着挙動である。
粒子は20 nm SiO2を仮定し、COMSOLから抽出した流速、粘性、電場、電子温度、電子密度を用いて軌道を計算する。

### 1.2 計算領域と発生源

| 項目 | 値 |
|---|---:|
| 空間次元 | 2D |
| 座標 | `cartesian_xy`、COMSOL r-z断面を x-y として扱う |
| 発生源 part | `part_id = 11` |
| 発生源 x範囲 | 0.1535 m から 0.1790 m |
| 発生源 y位置 | 0.022 m付近、放出位置は y = 0.024 m |
| 粒子数 | 10,000 |
| 計算時間 | 5.0e-4 s |
| 時間刻み | 1.0e-7 s |
| 保存間隔 | 100 step |
| 保存フレーム数 | 51 |

### 1.3 粒子物性

| 項目 | 値 |
|---|---:|
| 粒子直径 | 20 nm |
| 半径 | 10 nm |
| 密度 | 2200 kg/m3 |
| 質量 | 9.215e-21 kg |
| 代表電荷 | -9.574e-18 C |
| 代表電荷数 | -59.76 e |
| 代表 q/m | -1.039e3 C/kg |
| 初速中央値 | 193.66 m/s |
| 初速 p90 | 291.17 m/s |
| 初速最大 | 461.98 m/s |

粒子電荷は、電子温度から簡易的な浮遊電位を与えるモデルで作成している。

$$
a_p = \frac{d_p}{2}
$$

$$
m_p = \rho_p \frac{4}{3}\pi a_p^3
$$

$$
\phi_p \approx -2.5 T_e
$$

$$
q_p = 4\pi \epsilon_0 a_p \phi_p
$$

ここで、$d_p$ は粒子直径、$a_p$ は粒子半径、$\rho_p$ は粒子密度、$m_p$ は粒子質量、$T_e$ は電子温度を eV 相当の電位スケールとして扱った値、$\phi_p$ は粒子浮遊電位、$q_p$ は粒子電荷である。
本ケースでは独立したイオン密度場が未出力のため、診断量として `ion_density_m3_assumed = ne` を用いている。

---

## 2. 課題とゴール設定

### 2.1 このベンチマークで確認したい課題

1. COMSOL由来の非一様場と複雑境界を使っても、粒子が数値的に破綻しないこと。
2. `invalid_mask_stopped`、`numerical_boundary_stopped`、`max_hits_reached` が発生しないこと。
3. 装置パーツ、solver medium、field supportの違いを可視化で解釈できること。
4. 粒子がどの壁に衝突し、付着・反射・消滅したかを part 単位で追えること。
5. 計算時間を10倍に伸ばしたとき、現象が時間不足で見えていなかったのか、物理条件そのものが支配的なのかを切り分けること。

### 2.2 合格条件

| 指標 | 合格条件 | 今回結果 |
|---|---:|---:|
| `invalid_mask_stopped_count` | 0 | 0 |
| `numerical_boundary_stopped_count` | 0 | 0 |
| `max_hits_reached_count` | 0 | 0 |
| `unresolved_crossing_count` | 0 | 0 |
| `boundary_event_contract_passed` | 1 | 1 |
| `active_outside_geometry_count` | 0 | 0 |
| 非有限位置・速度 | 0 | 0 |

数値計算の健全性という意味では、今回の10k、0.5 ms計算は合格である。

---

## 3. 手法

### 3.1 入力場

COMSOLから外部exportした場を `precomputed_npz` として読み込む。solver本体にはCOMSOL依存を入れない。

| 場 | 意味 |
|---|---|
| `ux`, `uy` | ガス流速成分 |
| `mu` | 動粘性ではなく動的粘度 Pa s |
| `E_x`, `E_y` | 電場成分 |
| `ax`, `ay` | 粒子電荷と質量を反映した外力加速度 |
| `T` | 温度 |
| `p` | 圧力診断量 |
| `rho_g` | ガス密度 |
| `phi` | 電位 |
| `ne` | 電子密度 |
| `Te` | 電子温度 |
| `particle_charge_C` | 場から作った粒子電荷 |
| `particle_q_over_m_Ckg` | 粒子の q/m |

今回の場は steady 扱いで、時間軸は `times = [0.0]` である。将来、時間変化場では同じ形式で `times` を複数持たせ、時間方向に線形補間する。

### 3.2 運動方程式

粒子位置を $\mathbf{x}$、速度を $\mathbf{v}$、ガス流速を $\mathbf{u}(\mathbf{x},t)$、粒子外力加速度を $\mathbf{a}_{ext}(\mathbf{x},t)$ とする。
本solverの基本形は次である。

$$
\frac{d\mathbf{x}}{dt} = \mathbf{v}
$$

$$
\frac{d\mathbf{v}}{dt}
= \frac{\mathbf{u}(\mathbf{x},t)-\mathbf{v}}{\tau_{eff}}
+ \mathbf{a}_{ext}(\mathbf{x},t)
$$

ここで、$\tau_{eff}$ は有限Reynolds数補正後の粒子応答時間である。

### 3.3 力場種ごとの力

#### 流体抗力

Stokes応答時間は次で与える。

$$
\tau_{Stokes}
= \frac{\rho_p d_p^2}{18\mu}
$$

粒子Reynolds数は次である。

$$
Re_p
= \frac{\rho_g d_p \|\mathbf{v}-\mathbf{u}\|}{\mu}
$$

Schiller-Naumann補正は次で与える。

$$
f_{SN}(Re_p)
=
\begin{cases}
1 + 0.15 Re_p^{0.687}, & 0 < Re_p < 1000 \\
0.01875 Re_p, & Re_p \ge 1000 \\
1, & Re_p \approx 0
\end{cases}
$$

有効応答時間は次である。

$$
\tau_{eff}
= \max\left(\tau_{min}, \frac{\tau_{Stokes}}{f_{SN}}\right)
$$

抗力加速度は次になる。

$$
\mathbf{a}_{drag}
= \frac{\mathbf{u}-\mathbf{v}}{\tau_{eff}}
$$

本ケースでは `min_tau_p_s = 3.0e-4 s` を使い、20 nm粒子に対する過度に小さい緩和時間で時間刻みが支配されすぎないようにしている。

#### 電場力

電場による力は次である。

$$
\mathbf{F}_E = q_p \mathbf{E}
$$

$$
\mathbf{a}_E
= \frac{q_p}{m_p}\mathbf{E}
$$

今回の入力では、export側で $\mathbf{a}_E$ を `ax`, `ay` として事前計算している。solver hot pathでは電場そのものではなく、加速度場として利用する。

#### その他の外力

solverの一般形では、重力やユーザー指定の体積力を足せる。

$$
\mathbf{a}_{ext}
= \mathbf{a}_E + \mathbf{a}_{body}
$$

今回の主外力は電場由来の `ax`, `ay` である。

### 3.4 場の補間

rectilinear grid上の場は、2Dでは双線形補間、時間変化場では時間方向の線形補間を行う。

$$
Q(x,y,t)
=
\sum_{i=0}^{1}
\sum_{j=0}^{1}
\sum_{k=0}^{1}
w_i(x) w_j(y) w_k(t) Q_{ijk}
$$

steady場では時間方向の重みは実質的に1点だけである。

### 3.5 境界条件と境界反応

境界はCOMSOL由来のパーツ境界を線分集合として扱う。
1 stepでの試行移動を次の線分で表す。

$$
\mathbf{x}(\lambda)
= \mathbf{x}_n + \lambda(\mathbf{x}_{trial}-\mathbf{x}_n),
\quad 0 \le \lambda \le 1
$$

この線分と境界線分の交差から衝突位置を求める。

$$
\mathbf{x}_{hit}
= \mathbf{x}_n + \lambda_{hit}(\mathbf{x}_{trial}-\mathbf{x}_n)
$$

$$
t_{hit}
= t_n + \lambda_{hit}\Delta t
$$

壁面法線を $\mathbf{n}$、入射速度を $\mathbf{v}$、法線速度を $v_n = \mathbf{v}\cdot\mathbf{n}$、接線速度を $\mathbf{v}_t = \mathbf{v}-v_n\mathbf{n}$ とする。

#### 付着

壁または粒子に設定された付着確率を $p_{stick}$ とし、乱数 $r$ に対して次を満たす場合、粒子は停止して堆積質量に加算される。

$$
r < p_{stick}
$$

$$
\mathbf{v}_{after} = \mathbf{0}
$$

本ケースでは wafer、chamber wall、sidewall、part 11 で `wall_stick_probability = 0.5` を設定している。

#### 鏡面反射

付着しなかった場合、鏡面反射を行う。反発係数を $e_w$ とする。

$$
\mathbf{v}_{after}
= \mathbf{v}_t - e_w v_n \mathbf{n}
$$

本ケースでは wafer、chamber wall、sidewall、part 11 で `wall_restitution = 0.95` である。

#### 消滅境界

field support の外側に出る境界は `disappear` として扱う。

$$
\mathbf{v}_{after} = \mathbf{0}
$$

これは物理的な付着ではなく、計算対象領域から外れた粒子を除外するための境界である。

### 3.6 数値積分

本ケースでは `etd2` を使用する。
各step内で場を一定とみなしたとき、速度方程式は厳密に積分できる。

$$
\frac{d\mathbf{v}}{dt}
= -\frac{1}{\tau}\mathbf{v}
+ \frac{1}{\tau}\mathbf{u}
+ \mathbf{a}
$$

$$
\mathbf{c}
= \mathbf{u} + \tau\mathbf{a}
$$

$$
\mathbf{v}_{n+1}
= \mathbf{c}
+ (\mathbf{v}_n-\mathbf{c})\exp\left(-\frac{\Delta t}{\tau}\right)
$$

$$
\mathbf{x}_{n+1}
= \mathbf{x}_n
+ \mathbf{c}\Delta t
+ (\mathbf{v}_n-\mathbf{c})\tau
\left(1-\exp\left(-\frac{\Delta t}{\tau}\right)\right)
$$

`etd2` では、まず半step予測で中点位置を作り、中点の場 $\mathbf{u}_{mid}$、$\mathbf{a}_{mid}$、$\tau_{mid}$ を再評価し、その値でfull stepを進める。
これにより、単純なEuler法よりも抗力緩和に対して安定で、非一様場に対しても中点評価の精度を得る。

### 3.7 valid mask と SDF

`valid_mask` は、fieldが定義されているsolver mediumを示す。fieldが無い装置固体部分は、粒子軌道計算の対象ではない。
SDFは診断用の符号付き距離で、概念的には次のように使う。

$$
\phi(\mathbf{x}) < 0 \quad \text{inside solver domain}
$$

$$
\phi(\mathbf{x}) = 0 \quad \text{on boundary}
$$

$$
\phi(\mathbf{x}) > 0 \quad \text{outside solver domain}
$$

本ケースでは、初期粒子は全てfield support内にあり、最終的にも active 粒子がgeometry外へ出ていない。

### 3.8 高速化手法

#### Numba hot loop

粒子ごとの時間発展は NumPy配列と Numba kernel で処理する。
計算量は概ね次である。

$$
O(N_t N_p)
$$

ここで、$N_t$ は時間step数、$N_p$ は粒子数である。今回の計算では $N_t = 5000$、$N_p = 10000$ なので、約5000万 particle-step である。

#### ETDによる安定化

陽的Euler法では、抗力緩和が強い場合に $\Delta t \ll \tau$ が要求されやすい。

$$
\mathbf{v}_{n+1}^{Euler}
= \mathbf{v}_n
+ \Delta t
\left(
\frac{\mathbf{u}-\mathbf{v}_n}{\tau}
+ \mathbf{a}
\right)
$$

一方、ETDは指数減衰を解析的に含むため、局所的に場が一定なら緩和項を厳密に扱える。

$$
\exp\left(-\frac{\Delta t}{\tau}\right)
$$

このため、抗力項のためだけに過度に細かいstepへ落とす必要を減らせる。

#### 保存フレーム間引き

全stepを保存すると、軌道配列メモリは次になる。

$$
M_{all}
= 8 d N_p (N_t+1)
$$

保存間隔を $s$ とすると、保存メモリは次になる。

$$
M_{save}
= 8 d N_p
\left(
\left\lfloor \frac{N_t}{s} \right\rfloor + 1
\right)
$$

今回の設定では $d=2$、$N_p=10000$、$N_t=5000$、$s=100$ なので、positions配列は約8.16 MBで済む。
もし全step保存なら約800 MBになり、可視化・保存I/Oが重くなる。

---

## 4. シミュレーションワークフロー

```mermaid
flowchart TD
  A["COMSOL .mph"] --> B["外部exporter"]
  B --> C["geometry npz"]
  B --> D["field npz"]
  B --> E["export manifest"]
  C --> F["provider contract check"]
  D --> F
  G["particles.csv"] --> H["input contract check"]
  I["materials.csv / part_walls.csv"] --> J["wall model builder"]
  F --> K{"strict checks pass?"}
  H --> K
  K -- "no" --> L["export / input を修正"]
  K -- "yes" --> M["prepared runtime"]
  J --> M
  M --> N["Numba ETD2 solver"]
  N --> O["wall events / final particles"]
  N --> P["runtime diagnostics"]
  O --> Q["graphs"]
  P --> Q
  Q --> R["report / interpretation"]
```

---

## 5. ベンチマーク結果

### 5.1 実行サマリ

| 指標 | 50 usケース | 500 usケース |
|---|---:|---:|
| 粒子数 | 10000 | 10000 |
| 保存フレーム数 | 51 | 51 |
| active | 9674 | 9672 |
| stuck | 325 | 325 |
| absorbed | 1 | 3 |
| wall events | 590 | 592 |
| solver core time | 874.1 s | 7611.7 s |
| estimated numpy bytes | 11.7 MB | 11.7 MB |

`t_end` を10倍にしても、付着数は変わらず、field support exitの消滅が2件増えただけである。
したがって、ウェハー付着が少ない主因は「計算時間不足」ではなく、現在の粒子電荷・初速・電場・発生方向の組み合わせにある可能性が高い。

### 5.2 最終状態

![Final state bar and pie](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/02_final_state_bar_and_pie.png)

最終状態は active free-flight が 9672、stuck が 325、absorbed が 3である。
数値異常停止は0であり、ほとんどの粒子は0.5 ms後もsolver medium内を飛行している。

```csv
state,count
active_free_flight,9672
contact_sliding,0
contact_endpoint_stopped,0
invalid_mask_stopped,0
numerical_boundary_stopped,0
stuck,325
absorbed,3
escaped,0
inactive,0
```

### 5.3 時間発展

![State counts time series](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/01_state_counts_time_series.png)

壁イベントは初期に集中し、その後は状態数がほぼ一定である。
これは、長時間化で付着が単調に増えるケースではなく、初期の放出方向・電場加速・初回壁相互作用が結果を強く決めていることを示す。

壁イベント累積数の最終値:

```csv
time_s,reflected_specular,stuck,absorbed
0.0005,264,325,3
```

### 5.4 空間上の最終状態

![Final state scatter geometry](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/03_final_state_scatter_geometry.png)

active粒子は主にfield support内に残り、stuck粒子は発生源part 11近傍に集中する。
ウェハーpart 3への付着は、この設定では確認されない。

### 5.5 軌道密度

![Trajectory density heatmap](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/04_trajectory_density_heatmap.png)

軌道密度は発生源part 11周辺からsolver medium内部へ広がる。
密度分布を見ると、発生直後に壁へ戻る粒子と、内部へ入って長時間飛行する粒子に分かれている。

### 5.6 速度分布

![Speed distribution by state](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/05_speed_distribution_by_state.png)

active粒子は最終時点で中央値約60 m/s程度まで減速している。
stuck/absorbed粒子は終端状態として速度0で記録されるため、速度分布上は停止側に出る。

### 5.7 サンプル軌道

![Sampled trajectories overlay](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/06_sampled_trajectories_overlay.png)

個別軌道を見ると、発生源近傍で反射・付着する粒子と、電場・流れに乗って計算領域内部へ進む粒子が分かれる。
「もっと飛散するはず」という直感に対しては、現状でも飛散はしているが、ウェハーへ戻る軌道ではなく、内部滞留側の軌道が支配的である。

### 5.8 壁 law 別のイベント

![Wall law counts](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/08_wall_law_counts.png)

今回の物理壁は `specular` lawで、付着確率0.5を持つ。field support外への境界は `disappear` である。
結果として、specular壁イベントが589、disappearが3である。

### 5.9 part別の壁相互作用

![Wall interactions by part outcome](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/09_wall_interactions_by_part_outcome.png)

```csv
part_id,outcome,wall_mode,count
11,stuck,specular,325
11,reflected_specular,specular,264
9001,absorbed,disappear,3
```

part 11でのイベントが支配的である。
part 9001はfield support exitであり、物理付着ではなく計算対象領域外への除外である。

### 5.10 stuck粒子のpart

![Stuck counts by boundary part](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/10_stuck_counts_by_boundary_part.png)

```csv
part_id,stuck_count
11,325
```

付着は発生源と同じpart 11にのみ発生している。
この結果は、現行設定ではウェハー再付着ベンチマークというより、発生源近傍での再衝突・反射ベンチマークになっていることを示す。

---

## 6. 幾何・field support の解釈

### 6.1 装置パーツ形状

![Device parts geometry](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/11_device_parts_geometry.png)

パーツ外形を色で塗らず、線として確認する図である。
solver mediumだけでなく、fieldを持たない装置部品も表示するため、計算領域と装置構造の関係を把握しやすい。

### 6.2 パーツID付き形状

![Device parts with IDs](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/12_device_parts_with_ids.png)

part IDの対応確認用である。
壁イベント集計の `part_id` と、装置上の位置関係を対応付けるために使う。

### 6.3 SDF

![Signed distance field](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/13_signed_distance_field_sdf.png)

SDFは、粒子が境界に対してどの程度近いか、内外判定が破綻していないかを見るための診断図である。
今回、初期粒子はSDFで約2 mm内側にあり、near-boundary初期化ではない。

### 6.4 geometry と field support

![Geometry field support mask](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/14_geometry_field_support_mask.png)

field supportは、力学場が定義され粒子軌道計算に使える領域を示す。
装置部品そのものは存在しても、そこは固体でありsolver mediumではないため、fieldが白または非supportとして表示されることがある。

### 6.5 medium support と装置部品

![Domain parts medium support](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/22_domain_parts_medium_support.png)

```csv
part_id,field_supported_element_count,support_fraction,medium_status
2,0,0.0,device_part_no_solver_field
3,15,0.123,device_part_touching_solver_field
4,4527,0.983,solver_medium_region
5,0,0.0,device_part_no_solver_field
6,0,0.0,device_part_no_solver_field
7,0,0.0,device_part_no_solver_field
8,0,0.0,device_part_no_solver_field
9,0,0.0,device_part_no_solver_field
10,0,0.0,device_part_no_solver_field
11,43,0.143,device_part_touching_solver_field
12,16,0.098,device_part_touching_solver_field
```

`device_part_touching_solver_field` は、固体パーツがsolver mediumに接している境界部品である。
これは「パーツが存在しない」という意味ではなく、「そのパーツ自体は固体で、隣接するsolver medium側にfieldがある」という意味で解釈する。

---

## 7. 力学場と物理量分布

### 7.1 力学場の合成量

![Mechanics field totals](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/15_mechanics_field_totals.png)

流速ノルム、電場ノルム、加速度ノルムなど、粒子運動へ直接影響する合成量を並べた図である。
粒子軌道の大域傾向は、局所的な発生方向だけでなく、電場加速度の向きと大きさに強く支配される。

### 7.2 流速成分

![Flow components](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/16_flow_components_ux_uy.png)

`ux`, `uy` の空間分布である。
流速の範囲は概ね `ux = -2.58 to 0.34 m/s`、`uy = -0.99 to 0.51 m/s` であり、初速100 m/s級の粒子に対しては、初期運動よりも長時間の緩和過程で効いてくる。

### 7.3 外力加速度成分

![Acceleration components](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/17_acceleration_components_ax_ay.png)

`ax`, `ay` は粒子電荷と電場から作った加速度場である。
有効support内での範囲は、`ax = -4.58e6 to 2.20e7 m/s2`、`ay = -9.88e6 to 5.80e7 m/s2` であり、粒子の長時間軌道に対して非常に強い影響を持つ。

### 7.4 電場成分

![Electric field components](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/18_electric_field_components_ex_ey.png)

`E_x`, `E_y` の分布である。
20 nm SiO2粒子を負帯電として扱っているため、電場方向と粒子加速度方向は符号を含めて解釈する必要がある。

### 7.5 スカラー物理量

![Scalar physics fields](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/19_scalar_physics_fields.png)

温度、圧力、密度、電位、電子密度、電子温度などの診断量である。
現時点では、電子温度は粒子電荷の作成に、電子密度はイオン密度代用の診断に使っている。

---

## 8. 軌道とイベントの可視化

### 8.1 壁イベント位置

![Wall event locations](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/20_wall_event_locations_by_outcome.png)

壁イベントの位置を outcome 別に示す。
stuck と reflected_specular はpart 11上に集中し、absorbedはfield support exit側で発生している。

### 8.2 最終状態別軌道

![Trajectories by final state](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/21_trajectories_by_final_state.png)

最終状態ごとの軌道を分けることで、発生源近傍で終端する粒子と、長時間activeに残る粒子の違いが見える。

### 8.3 COMSOL風の場と軌道

![COMSOL style field and trajectories](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/23_comsol_style_field_and_trajectories.png)

COMSOLで分布と装置構造を重ねて見るのに近い形式で、fieldと粒子軌道を重ねている。
物理場、装置パーツ、軌道を同時に見ることで、どの場の構造が軌道を曲げているかを解釈しやすい。

### 8.4 粒子密度と壁イベント

![COMSOL style particle density and events](report_assets/icp_cf4_o2_10k_tend10x/visualizations/graphs/24_comsol_style_particle_density_and_events.png)

粒子密度と壁イベント位置を重ねた図である。
イベントは発生源周辺に強く偏り、field support内部の粒子密度は広がる。

### 8.5 アニメーション

<img src="report_assets/icp_cf4_o2_10k_tend10x/visualizations/animations/trajectories_sampled_trails.gif" width="720">

サンプル軌道のアニメーションである。保存間隔を10倍にしたため、`t_end` を10倍にしてもアニメーションのフレーム数は約51に保たれている。

全粒子アニメーション:

<img src="report_assets/icp_cf4_o2_10k_tend10x/visualizations/animations/trajectories_all_particles.gif" width="720">

---

## 9. 境界診断

### 9.1 認識された境界

![Recognized boundary geometry](report_assets/icp_cf4_o2_10k_tend10x/visualizations/boundary_diagnostics/01_recognized_boundary_geometry.png)

solverが認識した境界線分である。境界数は468である。

### 9.2 認識されたdomain mask

![Recognized domain mask](report_assets/icp_cf4_o2_10k_tend10x/visualizations/boundary_diagnostics/02_recognized_domain_mask.png)

粒子計算で有効な領域を示す。
field supportと装置固体の違いを確認するための図である。

### 9.3 SDF診断

![Boundary SDF](report_assets/icp_cf4_o2_10k_tend10x/visualizations/boundary_diagnostics/03_signed_distance_field.png)

境界近傍の符号付き距離を確認する。
今回、`invalid_mask_stopped` は0であり、SDF上も終端粒子は壁近傍に正しく置かれている。

### 9.4 境界法線

![Boundary normals](report_assets/icp_cf4_o2_10k_tend10x/visualizations/boundary_diagnostics/04_boundary_normals_near_wall.png)

壁反射計算で使う法線方向の確認図である。
鏡面反射では法線方向の符号と向きが速度更新に効くため、境界法線の可視化は重要である。

### 9.5 流速ベクトルとgeometry

![Flow speed vectors](report_assets/icp_cf4_o2_10k_tend10x/visualizations/boundary_diagnostics/05_flow_speed_vectors_over_geometry.png)

白い部分はfieldが無い、またはsolver medium外の領域である。
粒子初期位置と軌道がstrict support内にある限り、固体パーツにfieldが無いこと自体は問題ではない。

### 9.6 mixed stencil

![Mixed stencil hotspots](report_assets/icp_cf4_o2_10k_tend10x/visualizations/boundary_diagnostics/06_mixed_stencil_hotspots.png)

mixed stencilは、補間ステンシルがvalid/invalidの境界をまたぐ場所である。
今回のmixed stencil violationは284件あるが、hard invalidは0で、粒子停止にはつながっていない。

### 9.7 hard invalid stop hotspots

![Hard invalid stop hotspots](report_assets/icp_cf4_o2_10k_tend10x/visualizations/boundary_diagnostics/07_hard_invalid_stop_hotspots.png)

hard invalid stopは0である。
これは、今回の計算がfield support外へ数値的に押し出されて停止したわけではないことを示す。

---

## 10. Mechanics可視化

### 10.1 最終状態 over geometry

![Final states over geometry](report_assets/icp_cf4_o2_10k_tend10x/visualizations/mechanics/final_states_over_geometry.png)

geometry上に最終状態を重ねた図である。
stuckはpart 11近傍、activeはfield support内部に分布している。

### 10.2 geometry layout with part IDs

![Geometry layout part IDs](report_assets/icp_cf4_o2_10k_tend10x/visualizations/mechanics/geometry_layout_part_ids.png)

part IDの位置確認用である。
part別イベント集計と対応させることで、どの装置部位が支配的かを説明できる。

### 10.3 mechanics maps

![Mechanics maps with geometry](report_assets/icp_cf4_o2_10k_tend10x/visualizations/mechanics/mechanics_maps_with_geometry.png)

力学場をgeometryに重ねた図である。
場が存在する領域と装置固体の関係を確認できる。

### 10.4 mechanics component maps

![Mechanics component maps](report_assets/icp_cf4_o2_10k_tend10x/visualizations/mechanics/mechanics_component_maps_with_geometry.png)

成分ごとの場を見ることで、粒子がどちら向きに曲げられるかを確認する。

### 10.5 trajectories with flow overlay

![Trajectories geometry flow overlay](report_assets/icp_cf4_o2_10k_tend10x/visualizations/mechanics/trajectories_geometry_flow_overlay.png)

粒子軌道と流れ場を重ねた図である。
初速が大きい短時間領域では粒子初期条件が支配的で、長時間では抗力と電場の寄与が支配的になる。

---

## 11. 考察

### 11.1 数値計算としての結論

今回の10k、0.5 ms計算では、次の数値問題は発生していない。

| 問題 | 結果 |
|---|---:|
| hard invalid field access | 0 |
| numerical boundary stopped | 0 |
| unresolved crossing | 0 |
| max hits reached | 0 |
| nonfinite position | 0 |
| nonfinite velocity | 0 |

したがって、現状の主要課題は「境界処理の数値破綻」ではなく、「物理条件が期待するウェハー再付着現象を出す設定になっているか」である。

### 11.2 物理現象としての結論

`t_end` を50 usから500 usへ10倍にしても、stuck数は325のまま変わらなかった。
増えたのはfield support exitのabsorbedが1から3になっただけである。

この結果から、次が示唆される。

1. part 11近傍で早期に付着・反射する粒子が結果を決めている。
2. 長く計算しても、ウェハーpart 3への付着は自然には増えない。
3. 現在の負帯電20 nm SiO2モデルと電場分布では、ウェハーへ戻る軌道よりも内部滞留・別方向移動が支配的である。
4. ウェハー再付着を検証したい場合、単に `t_end` を伸ばすのではなく、発生源の法線方向、初速分布、帯電モデル、RF時間変化場、シース近傍の電場モデルを見直す必要がある。

### 11.3 計算負荷

今回の実行時間は約7612 s、約127分である。
50 usケースは約874 sであり、概ね時間step数に比例して増えている。

一方、保存間隔を10倍にしたため、保存フレーム数とpositions配列サイズは維持できた。

| 項目 | 値 |
|---|---:|
| runtime steps | 5000 |
| positions array | 8.16 MB |
| estimated numpy bytes | 11.75 MB |
| solver core time | 7611.7 s |

メモリ面は良好だが、計算時間は長い。
今後の本番評価では、0.5 ms full 10kを日常ゲートにするのではなく、短時間の物理妥当性確認と、必要時の長時間確認を分けるべきである。

### 11.4 次に必要なベンチマーク

ウェハーへの再付着を本当に評価するには、次の比較が必要である。

1. 中性20 nm SiO2粒子: 電場力なしで、初速と抗力だけの基準を見る。
2. 負帯電20 nm SiO2粒子: 今回設定。電場による曲げを評価する。
3. 正帯電または時間変化帯電粒子: シース/RFの位相依存を評価する。
4. RF時間変化場: steady場では見えない周期的加速を評価する。
5. 発生方向の比較: 表面法線方向、斜め方向、cosine分布、エネルギー分布。
6. wall law感度: waferだけ付着確率を変える、source partの再付着を抑える、sidewall反射を変える。

---

## 12. まとめ

このベンチマークで、COMSOL由来のfield/boundary provider、strict input/provider contract、ETD2積分、壁衝突処理は、10k粒子・0.5 msでも数値的には安定に動くことを確認した。
一方、期待していた「part 11から飛散して一部がウェハーへ付着する」現象は、この物理設定では再現されていない。

重要な結論は次である。

1. **数値破綻は起きていない。**
2. **時間を10倍にしてもウェハー付着は増えない。**
3. **支配要因は境界バグではなく、粒子帯電、初速方向、電場、発生源近傍wall lawの物理設定である。**
4. **次は、RF時間変化・シース近傍帯電・発生方向分布を比較する物理ベンチマークに進むべきである。**
