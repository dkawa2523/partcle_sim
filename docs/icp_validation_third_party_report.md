# ICP Surface-Release Validation Report

このレポートは、ICPのCOMSOLエクスポート場を使った粒子軌道計算について、第三者が「初期条件」「境界処理」「力場」「drag」「wall law」が設定どおり妥当に働いているかを論理的に確認できるように整理したものです。

## 結論

- acceptance criteria: `True`
- 全ケースで表面上の粒子300個に `source.preprocess.boundary_release` が適用され、offset失敗は `0` です。
- C0からC5まで `hard_invalid=0`, `invalid_mask_stopped=0`, `unresolved_crossing=0` で、初期条件や境界判定による停止は見られません。
- C2とC4の比較で電場ON/OFFの差、C2とC3の比較で電荷符号の差、C6でwall lawの差を確認できます。
- 物理代表として主に見るべきケースはC3系、つまり `10 nm`, `negative charge`, `Epstein drag` です。正電荷ケースは力場符号と境界応答の検証用です。

## 本コードが解く問題

本コードは、既に与えられた粒子初期条件と場に基づいて粒子軌道を計算するsolverです。チャンバー部品破損やデポ剥離の発生確率そのものは推定しません。ユーザーが粒径、密度、電荷、初期位置、初期速度、壁条件、流体場、電場を与え、その条件のもとで粒子がどのように運動するかを解きます。

今回のICP検証では、COMSOL由来の2D幾何・流速・電場・ガス場を使います。粒子は部品表面上の座標から開始し、`boundary_release` によって流体側へ最小限だけ正規化されます。その後、drag、`qE/m`、wall lawを組み合わせて軌道と壁イベントを評価します。

## 数値経路

| 段階 | 入力/設定 | 確認する診断 |
|---|---|---|
| 初期条件 | `particles.csv`, source law, release位置 | `source_particle_diagnostics.csv`, `input_contract_report.json` |
| 境界正規化 | `source.preprocess.boundary_release=true` | `boundary_release_applied_count`, `boundary_release_failed_offset_count` |
| 力場 | `solver.forces`, `charge`, `E_x/E_y` | `force_contributions.csv`, `electric_q_over_m_particle_stats` |
| drag | `solver.drag_model` | `solver_report.json`, drag gas property figures |
| 壁相互作用 | `part_walls.csv` の wall law | `wall_events.csv`, `wall_summary.json`, final states |
| 軌道出力 | save frames, final particles | `positions_2d.npy`, graphs, GIF animations |

## 妥当性確認の考え方

一つの複雑なケースだけを見ると、初期条件、電場、drag、壁反射のどれが結果を支配しているか分かりません。そのため、C0からC6まで機能を分けて検証します。全ケースは同じ表面起源の粒子集合から始まり、ユーザー指定の物理offsetは `0 m` です。

## ケース一覧と主要結果

| Case | 目的 | Forces | Drag | Charge [e] | Source speed [m/s] | Net median [mm] | Path median [mm] | Wall events | Final state | Invalid/Hard |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|
| C0 | Boundary-only neutral release | drag | epstein | 0 | 2.0 | 0.0974 | 0.0974 | 0 | active_free_flight=300 | 0/0 |
| C1 | Initial-velocity neutral release | drag | epstein | 0 | 20.0 | 0.972 | 0.972 | 1 | active_free_flight=300 | 0/0 |
| C2 | Positive charge with electric field | drag,electric | epstein | 20 | 20.0 | 0.0816 | 0.507 | 4102 | active_free_flight=300 | 0/0 |
| C3 | Negative charge with electric field | drag,electric | epstein | -20 | 20.0 | 11.8 | 11.8 | 1 | active_free_flight=300 | 0/0 |
| C4 | Positive charge electric-off control | drag | epstein | 20 | 20.0 | 0.972 | 0.972 | 1 | active_free_flight=300 | 0/0 |
| C5 | Drag sensitivity: negative charge with Stokes-Cunningham | drag,electric | stokes_cunningham | -20 | 20.0 | 11.6 | 11.6 | 1 | active_free_flight=300 | 0/0 |
| C5b | Drag control: neutral Stokes-Cunningham | drag | stokes_cunningham | 0 | 20.0 | 0.955 | 0.955 | 1 | active_free_flight=300 | 0/0 |
| C6 | Wall-law sensitivity: stick | drag,electric | epstein | 20 | 20.0 | 0.0041 | 0.0299 | 267 | active_free_flight=33, stuck=267 | 0/0 |

元データ: [assets/icp_validation/icp_validation_suite_metrics.csv](assets/icp_validation/icp_validation_suite_metrics.csv) / [assets/icp_validation/icp_validation_suite_evaluation.json](assets/icp_validation/icp_validation_suite_evaluation.json)

## 図による全体評価

### 全ケースの軌道量と壁イベント

この図では、net displacement、path length、wall event数、final speedを横並びで比較します。C2はpath lengthは大きい一方、壁反射が多いためnet displacementは小さく見えます。C3は負電荷により大きく飛散し、壁イベントは少ないです。

![Suite motion and boundary summary](assets/icp_validation/graphs/01_suite_motion_boundary_summary.png)

### 電荷符号と電場ON/OFF

C2とC4は同じ正電荷・同じ初速ですが、C2だけ電場ONです。wall eventsが `1` から `4102` へ増えるため、正電荷ケースの壁反射増加は境界releaseではなく電場由来と判断できます。C2とC3では電荷符号だけで軌道スケールが大きく変わります。

![Force sign control summary](assets/icp_validation/graphs/02_force_sign_control_summary.png)

### dragモデル感度

C3とC5は電場が強く支配する条件なのでdragモデル差は小さめです。そのためC1とC5bの中立・電場OFF条件も併せて見ることで、dragだけの差を切り分けています。

![Drag model sensitivity](assets/icp_validation/graphs/03_drag_model_sensitivity.png)

### wall law感度

C2はspecular反射のため全粒子がactive free flightのまま残ります。C6は同じ力場でwall lawをstickに変え、267個がstuckになりました。壁到達を反射として扱うか終端として扱うかが結果へ反映されることを確認できます。

![Wall law final state summary](assets/icp_validation/graphs/04_wall_law_final_state_summary.png)

## ケース別の判断

### C0: Boundary-only neutral release

表面上の粒子が最小offsetで内部化され、cleanなfield supportに入るかを確認するケースです。300個すべてがactiveで、壁イベントもinvalid停止もありません。初期条件・境界正規化の基本経路は妥当です。

- setup: forces=`drag`, drag=`epstein`, charge=`0e`, source speed mean=`2.0 m/s`
- boundary: applied=`300/300`, failed offset=`0`, user source offset median=`0 um`, solver offset median=`2 um`
- result: net median=`0.0974 mm`, path median=`0.0974 mm`, wall events=`0`, final state=`active_free_flight=300`
- health: invalid=`0`, hard invalid=`0`, unresolved crossing=`0`

### C1: Initial-velocity neutral release

電場なし・中立で初速だけを20 m/sへ上げたケースです。約0.97 mmのnet/path displacementが出て、力場なしでも初速に応じて飛散できることを確認します。

- setup: forces=`drag`, drag=`epstein`, charge=`0e`, source speed mean=`20.0 m/s`
- boundary: applied=`300/300`, failed offset=`0`, user source offset median=`0 um`, solver offset median=`2 um`
- result: net median=`0.972 mm`, path median=`0.972 mm`, wall events=`1`, final state=`active_free_flight=300`
- health: invalid=`0`, hard invalid=`0`, unresolved crossing=`0`

### C2: Positive charge with electric field

正電荷で電場を有効にした符号検証ケースです。wall eventsが4102件と多く、正電荷がICP電場により近傍壁へ戻される挙動を示します。これは標準的な負帯電dust代表ではなく、力場符号と境界反射の確認です。

- setup: forces=`drag,electric`, drag=`epstein`, charge=`20e`, source speed mean=`20.0 m/s`
- boundary: applied=`300/300`, failed offset=`0`, user source offset median=`0 um`, solver offset median=`2 um`
- result: net median=`0.0816 mm`, path median=`0.507 mm`, wall events=`4102`, final state=`active_free_flight=300`
- health: invalid=`0`, hard invalid=`0`, unresolved crossing=`0`

### C3: Negative charge with electric field

負電荷で電場を有効にした物理代表寄りのケースです。net medianは11.8 mm、wall eventは1件で、C2と逆向きの飛散傾向が見えます。10 nm負帯電粒子のICP飛散評価ではこの系統を主に見るべきです。

- setup: forces=`drag,electric`, drag=`epstein`, charge=`-20e`, source speed mean=`20.0 m/s`
- boundary: applied=`300/300`, failed offset=`0`, user source offset median=`0 um`, solver offset median=`2 um`
- result: net median=`11.8 mm`, path median=`11.8 mm`, wall events=`1`, final state=`active_free_flight=300`
- health: invalid=`0`, hard invalid=`0`, unresolved crossing=`0`

### C4: Positive charge electric-off control

C2と同じ正電荷・初速で電場だけOFFにした対照ケースです。wall eventsは1件に戻るため、C2の多数反射が初速やboundary releaseではなく電場由来であることを示します。

- setup: forces=`drag`, drag=`epstein`, charge=`20e`, source speed mean=`20.0 m/s`
- boundary: applied=`300/300`, failed offset=`0`, user source offset median=`0 um`, solver offset median=`2 um`
- result: net median=`0.972 mm`, path median=`0.972 mm`, wall events=`1`, final state=`active_free_flight=300`
- health: invalid=`0`, hard invalid=`0`, unresolved crossing=`0`

### C5: Drag sensitivity: negative charge with Stokes-Cunningham

C3と同じ負電荷・電場ONでdragをStokes-Cunninghamに変えたケースです。この条件では電場支配が強く、Epsteinとの差は小さいですが、drag設定を変えても不正停止が出ないことを確認します。

- setup: forces=`drag,electric`, drag=`stokes_cunningham`, charge=`-20e`, source speed mean=`20.0 m/s`
- boundary: applied=`300/300`, failed offset=`0`, user source offset median=`0 um`, solver offset median=`2 um`
- result: net median=`11.6 mm`, path median=`11.6 mm`, wall events=`1`, final state=`active_free_flight=300`
- health: invalid=`0`, hard invalid=`0`, unresolved crossing=`0`

### C5b: Drag control: neutral Stokes-Cunningham

中立・電場OFFでStokes-Cunninghamを使うdrag単独対照です。C1と比較することで、電場に隠れないdragモデル差を確認できます。

- setup: forces=`drag`, drag=`stokes_cunningham`, charge=`0e`, source speed mean=`20.0 m/s`
- boundary: applied=`300/300`, failed offset=`0`, user source offset median=`0 um`, solver offset median=`2 um`
- result: net median=`0.955 mm`, path median=`0.955 mm`, wall events=`1`, final state=`active_free_flight=300`
- health: invalid=`0`, hard invalid=`0`, unresolved crossing=`0`

### C6: Wall-law sensitivity: stick

C2と同じ力場でwall lawをstickへ変更したケースです。267個がstuckになり、specular反射ではなく壁到達終端として扱えることを確認します。

- setup: forces=`drag,electric`, drag=`epstein`, charge=`20e`, source speed mean=`20.0 m/s`
- boundary: applied=`300/300`, failed offset=`0`, user source offset median=`0 um`, solver offset median=`2 um`
- result: net median=`0.0041 mm`, path median=`0.0299 mm`, wall events=`267`, final state=`active_free_flight=33, stuck=267`
- health: invalid=`0`, hard invalid=`0`, unresolved crossing=`0`

## 代表ケースの詳細図とアニメーション

### C2: Positive charge with electric field

![C2 COMSOL style trajectories](assets/icp_validation/cases/C2/23_comsol_style_field_and_trajectories.png)

- Animation: [C2 trajectories_all_particles.gif](assets/icp_validation/cases/C2/trajectories_all_particles.gif)
- Sampled trail animation: [C2 trajectories_sampled_trails.gif](assets/icp_validation/cases/C2/trajectories_sampled_trails.gif)

### C3: Negative charge with electric field

![C3 COMSOL style trajectories](assets/icp_validation/cases/C3/23_comsol_style_field_and_trajectories.png)

- Animation: [C3 trajectories_all_particles.gif](assets/icp_validation/cases/C3/trajectories_all_particles.gif)
- Sampled trail animation: [C3 trajectories_sampled_trails.gif](assets/icp_validation/cases/C3/trajectories_sampled_trails.gif)

### C6: Wall-law sensitivity: stick

![C6 COMSOL style trajectories](assets/icp_validation/cases/C6/23_comsol_style_field_and_trajectories.png)

- Animation: [C6 trajectories_all_particles.gif](assets/icp_validation/cases/C6/trajectories_all_particles.gif)
- Sampled trail animation: [C6 trajectories_sampled_trails.gif](assets/icp_validation/cases/C6/trajectories_sampled_trails.gif)

## Acceptance Criteria

- `all_boundary_release_applied`: `True`
- `all_boundary_release_failed_zero`: `True`
- `C0_C5_no_invalid_hard_unresolved`: `True`
- `C2_vs_C4_electric_changes_wall_events`: `True`
- `C2_vs_C3_charge_sign_changes_trajectory`: `True`
- `C3_vs_C5_drag_difference_measured`: `True`
- `C1_vs_C5b_drag_control_difference_measured`: `True`
- `C6_wall_law_terminal_state`: `True`
- `all_pass`: `True`

すべてのcriteriaがtrueであるため、この検証スイートの範囲では、初期条件、境界処理、力場、drag、wall lawが設定に応じて区別可能な結果を返していると判断できます。

## 限界と注意

- これは物理モデルの完全性証明ではありません。solverが、与えられた場・粒子・wall lawに対して一貫した軌道を返すことの検証です。
- sheath/ion drag、動的帯電、剥離発生確率、粗さ・付着力の校正モデルはこの検証ケースでは扱っていません。
- 正電荷ケースは標準的な低温ICP dust代表ではなく、電場符号と境界応答を検証するための意図的なケースです。
- wall lawのstickケースはsolver挙動の確認であり、実際の再堆積率を主張するものではありません。
- COMSOL faithful比較ではなく、COMSOL由来場を使った通常runの数値・物理経路の妥当性確認です。
