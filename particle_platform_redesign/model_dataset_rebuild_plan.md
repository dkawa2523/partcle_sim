# `model_dataset` 外部V&V再構築計画

## 1. 目的

本書は`model_dataset`を製品coreから独立した検証資産として再構築する計画である。現在の大規模
COMSOLケースを最初のgolden truthにせず、解析解を持つmicrocaseで基礎を固定した後、実チャンバー
モデルをsystem-level referenceとして使用する。

COMSOLモデル、exporter、comparison、reportは `tools/vv/comsol` に置く。製品coreはこのtoolを
importせず、外部toolだけがcoreの公開APIを呼ぶ。

---

## 2. 現在の資産

fresh inventoryは次のとおりである。

- 590 files、2,963,911,950 bytes
- COMSOLモデル2種：ion dragの`theory_consistent`と`image_minimal_corrected`
- 各モデルにCase P/A × 10/30/100 nm、合計12ケース
- 各ケース287粒子 × 121保存時刻
- 座標系は2D axisymmetric RZ
- 運動は固定10 µs RK4
- 電気、ion drag、Epstein drag、熱泳動、自由分子lift、DEP、重力・浮力、Brownian、動的電荷
- wallは主にStick、入口Freeze、pump Disappear。Reflectionは未設定
- 発生源は表面ではなく領域内のRelease from Grid

既存Java/PowerShell toolは、COMSOL tag、式、境界ID、保存solutionを調べる資料として再利用できる。
ただし絶対pathとcase固有tagが埋め込まれているため、新toolのarchitectureとしては継承しない。

---

## 3. fresh監査で確認した不備

### 3.1 粒子依存量が背景場へ混入

background bundleの`particle_Knudsen_number`、`Epstein_response_time_s`、
`Brownian_diffusivity_m2_per_s`は、10/30/100 nmで実質的に同じ粒径条件を保持している。
一方、粒子履歴内の径、質量、局所Kn、応答時間は各粒径で変化している。

対策：backgroundはprimitive fieldだけにし、粒子依存量を全てcore実行時に計算する。

### 3.2 mesh-point fieldはraw node/DOFではない

- field rows：33,449
- geometry vertices：7,732
- tolerant unique coordinates：7,732
- export設定：`smooth=material`, `recover=pprint`, `resolution=normal`

後処理評価の重複行であり、raw finite-element DOFやCOMSOL内部補間の完全な表現ではない。

対策：node/element ID、要素次数、局所node順、field basis、評価位置、smoothing/recoveryを明示する。

### 3.3 四辺形局所順が未記載

CSV列順`[n1,n2,n3,n4]`を周回順にすると多数の自己交差・ゼロ面積要素になる。実データの周回順は
`[n1,n2,n4,n3]`である。

対策：`local_node_ids`と`perimeter_node_ids`を別列でexportする。

### 3.4 regular-grid supportが無効

301×301格子の`inside_model_domain`は全行1だが、domain fieldが有限な点は一部だけである。

対策：geometry/domain topologyをsupportの権威とし、各fieldの有限maskを別に保存する。

### 3.5 event表に未来情報が混入

現在のevent historyは各保存時刻行にfinal statusと将来event timeを複製しており、一行一eventではない。

対策：正確なevent時刻ごとに独立行を出すsparse ledgerへ変更する。

### 3.6 RZ運動と方位診断が混在

速度方位成分は0だがBrownian方位力が非0で、3D合力magnitudeへ含まれている。

対策：積分対象r/z成分と、参考3D診断を別namespaceにする。3D Brownianを検証するcaseでは粒子を
Cartesian 3Dで積分する。

### 3.7 現行validationは構造検査中心

行数、列数、file存在は確認するが、粒径整合、field意味、quad順、support、event意味を検査しない。

対策：構造PASSと物理・数値PASSを分ける。巨大な総合validatorを作らず、export直後の少数の
意味検査を追加する。

---

## 4. 再利用、再出力、legacy隔離

### 4.1 再利用するもの

- 二つのMPHとSHA256
- geometryの頂点、三角形、境界edge、COMSOL entity ID
- Case P/Aの粒子非依存な原始場
- 実際に生成された初期粒子状態
- 動的電荷、各力、境界選択、sourceの式とparameter
- 保存軌道を過去のsystem regressionとして参照
- 二つのion drag式をmodel-form感度として参照

### 4.2 再出力するもの

- geometryを共通1セット
- Case P/Aのprimitive fieldを各1セット
- node、cell、facet、owner、局所node順、要素次数
- COMSOL field probe oracle
- Brownian-offの決定論run
- charge off/onを分けたrun
- charge、neutral drag、thermophoresis、二つのion drag、DEP、liftを一つずつ有効化したforce-ablation run
- DEP用`grad(mean_E_squared)`とlift用方位vorticityについて、solution・DC/RF平均・微分/回復法・単位・topology ID
- solver method、内部step、出力時刻、COMSOL build
- 少数sentinel粒子のfrozen RHS、各力、`dZ/dt`、可能ならRK stage
- 一行一event ledger
- 表面放出、鏡面反射、確率付着、boundary 37 Freeze、boundary 35 Disappearの新規microcase
- 12 packageごとのBrownian replicaと複数seed

### 4.3 legacy扱いにするもの

- 12ケースごとに複製されたgeometry/field package
- background内の粒子依存列
- `inside_model_domain`
- 現四辺形CSVをそのままpolygonへする処理
- 現`particle_boundary_event_history_tidy.csv`をevent logとして使うこと
- PNGを数値合否に用いること
- 方位成分を含むforce magnitudeをRZ加速度に使うこと
- 片方のion drag軌道を無条件に真値とすること

legacyは削除せず、manifestで`reference_only`とし、新しいrelease gateから外す。

---

## 5. 最小外部tool構成

```text
tools/vv/comsol/
├─ README.md
├─ cases/
│  ├─ V01_field_linear/
│  ├─ V02_deterministic_motion/
│  ├─ V03_surface_release_rz/
│  ├─ V04_wall_events/
│  ├─ V05_charge_coupling/
│  ├─ V06_brownian_ou/
│  └─ chamber_reference/
├─ comsol/
│  ├─ BuildMicrocases.java
│  ├─ ExportCase.java
│  └─ run_cases.ps1
├─ normalize.py
├─ compare.py
└─ reports/
```

行う処理は四つだけである。

1. COMSOL modelを生成または読込む。
2. 共通CSVへexportする。
3. canonical comparison tableへ正規化する。
4. core公開APIを実行し、解析解またはCOMSOL結果と比較する。

常設dashboard、plugin framework、caseごとの専用比較classは作らない。

---

## 6. 最初の六つのmicrocase

| ID | 検証対象 | 最小設定 | 主な正解 |
|---|---|---|---|
| V01 | field補間・support・要素検索 | triangle＋歪みquad、一次scalar/vector場 | 解析式 |
| V02 | 運動・積分 | ballistic、一定加速度、線形drag | 厳密軌道 |
| V03 | surface release | RZ直線境界、法線・初速・発生時刻 | 指定値、`2πr ds`分布 |
| V04 | wall event | stick、escape、specular、grazing、multiple hit | hit時刻・点・反射速度 |
| V05 | charge coupling | 一様E、線形charge relaxation | `q(t),v(t),x(t)`厳密解 |
| V06 | Brownian OU | 一様T、線形drag、多数replica | 平均、分散、MSD |

V01～V05はBrownian offで先に完成させる。V04の確率付着はCOMSOLと同じ粒子乱数を要求せず、
二項分布と反射後分布で比較する。V06もpathwise一致ではなく統計量で比較する。

その後だけ`chamber_reference`を使う。最初にCase P/A 30 nmのBrownian-off代表caseでexportと比較器を確認し、
合格後は二つのion-drag revision × Case P/A × 10/30/100 nmの全12 packageへ同じworkflowを展開する。

- 10 usを含む候補刻みのdrag/charge gateを先に評価し、両solverが受理する最大`h`から`h,h/2,h/4`を事前登録
- 代表5粒子は内部stepごとのfield、force、charge、state、可能ならRK stage
- 全287粒子は121保存時刻と一行一event
- ion-drag variant間はion drag以外の式・field・initial/boundary/solver hashを一致させる
- Brownian-onは全12 packageについて32独立seed、または事前登録した同等のconfidence budget
- 元のsingle-seed軌道は記述用に限り、pathwise PASSへ使わない

---

## 7. 最小export仕様

### 7.1 manifest

```text
format_version, case_id, coordinate_system,
comsol_version, comsol_build, model_sha256,
study_tag, solution_tag,
integrator, internal_dt_s, output_times_s,
random_seed, active_physics,
mesh_element_types, field_interpolation_settings
```

### 7.2 geometry

`mesh_nodes.csv`

```text
node_id, x_m, y_m, z_m
```

RZでは`r_m,z_m`とする。

`mesh_cells.csv`

```text
cell_id, cell_type, domain_id,
local_node_ids, perimeter_node_ids, geometry_order
```

`boundary_facets.csv`

```text
facet_id, boundary_id, boundary_name,
fluid_domain_id, node_ids,
normal_components, boundary_rule, material_id
```

### 7.3 fields

粒子非依存量だけを格納する。

- gas velocity、temperature、pressure、density、viscosity、composition
- electric potentialまたはelectric fieldとauthority
- electron/ion density、temperature、ion mass、ion velocity
- 必要なparticle-independent gradient

Kn、response time、Brownian diffusivity、粒子質量、電荷依存量は格納しない。

`field_probes.csv`は次を持つ。

```text
probe_id, time_s, position,
inside_support, domain_id, cell_id,
primitive field values
```

### 7.4 particlesとtrajectory

初期粒子：

```text
particle_id, release_time_s, source_boundary_id,
position, velocity, diameter_m, mass_kg,
charge_number, source_normal, statistical_weight
```

trajectory：

```text
particle_id, time_s, position, integrated_velocity,
charge_number, charge_rate, status, cell_id,
integrated force components, acceleration,
sampled primitive fields
```

RZの方位診断は別列群へ置き、積分合力と混ぜない。

### 7.5 events

一行一eventとする。

```text
particle_id, event_index, event_time_s,
boundary_id, event_kind, hit_position,
surface_normal, velocity_pre, velocity_post,
charge_number_pre, charge_number_post, ensemble_run_id
```

確率壁caseでは、必要なら使用した一様乱数または外部run IDを保存する。final statusを全trajectory行へ
未来情報として複製しない。

---

## 8. 再実行順序

初期案の「境界microcaseの直後にCase P/A 30 nmから全matrixへ展開する」という一括順序は、fresh auditより前の
計画であり、現在のnext-work authorityではない。監査後は、M3-Vで既存12 packageの適用域と比較可能性を調べ、M3-C0aで
MPH・式・parameter・field意味・seed cohortをoffline lockし、必要なP18-C/I/D/L/R modelを独立microcaseとともに実装した。
その後、M3-C0bをCase-A 100 nmで段階実行した。30 ms v3とpre-event v4は受理せず、v5の旧「全gate PASS」も位置relative
L2が絶対RZ原点に依存したため`INVALIDATED`とした。v5 raw artifactは`CHARACTERIZED`の履歴証拠として保持する。

逐次確認v6は、0.625/0.3125/0.15625 usの各runで全13,202 recordがactiveな287粒子×46 frame、0--450 usの
pre-event sliceだけを対象とする。原点不変な変位・速度・電荷のfine-pair relative L2は
`3.102727085428027e-5/3.92483251084038e-5/1.3511393490811483e-6`、観測次数は
`0.9041136/0.944312/1.123838`で、0.15625 usを運用上の固定stepとして選ぶgateだけをPASSした。これはsolver agreement、
普遍的な物理精度、30 ms/event収束、他case/variant/boundary、Brownianの認定ではない。

M3-C1のCase-A 100 nm pre-event frozen RHS、exported-P1/native-field workflow、原因分離用full-physics common-P1診断まで
完了した。cross-representation 6 gateは全FAILし、別statusのcommon-P1診断は事前登録9 gateを全PASSした。後者は
この限定sliceのsame-field agreementだけを認定し、native-field等価性や物理妥当性を認定しない。

続いて、力なしの解析的normal-impact boundary正例を隔離COMSOL copyで実行した。現行v2は2 scenario×3刻みのexact 6
configuration receiptをprocess logからfail-closedで照合する。boundary 37 Freezeと35 Disappearを10/5/2.5 us、
0--150 us、2.5 us保存で評価し、event 73 us、最初のterminal frame 75 usを確認した。全active frameは
`x=x0+v0*t` / `v=v0`に一致し、全run最大の位置/速度誤差は`4.726604209672303e-16 m` /
`1.7763568394002505e-15 m/s`（各上限`1e-12`）だった。Freezeはstatus 2でhit点R-Zと衝突前velocity payloadを保持し、
Disappearはstatus 4で位置・速度をNaNにした。step間event-time spreadは
`3.07371315899641e-17/2.71050543121376e-20 s`、56 PASS / 0 FAIL / velocity記述6件で、原本MPH hashは不変である。
これはCOMSOL側の隔離意味だけを閉じ、production parity、grazing/corner、full physicsを認定しない。v1は科学的に無効ではなく、
configuration receiptとactive-flight oracleの監査強度が不足した履歴成果物としてv2に置換した。

現在の到達点と残作業は次の順で固定する。

1. P18-Hはproducer非依存の最小`hold/held`として完了した。解析・公開回帰を先に閉じ、既存hash固定Freeze referenceとの
   candidate比較をCOMSOL再実行なしで15/15 PASSした。
2. B03のRZ projected Brownian＋fixed/continuous charge＋全決定論力は一つのstochastic proposalとして実装・検証を完了した。
   core受入にCOMSOL再実行は不要で、等方3-Dや一般SDEのstrong/weak 2次は主張しない。
3. M3-C2Aの最初のtheory-consistent Case-A 100 nm anchorはcommon-P1契約、独立pilot、32+32 seed finalまで完了した。
   保存native-field単一runは正式cohortへ昇格せず、out-of-plane無効のR-Z 2自由度というmodel意味も維持した。
4. 同じ契約のCase-P 100 nm一軸展開も、20 us、32+32独立seed、各287粒子、121 frame / 30 msで完了した。
   83区分R-Z/fate gateは最大empirical TV `0.010670731707317093`、同時上限`0.13119968456545308 < 0.15`でPASSした。
   終端gateはevent 0で非情報的である。元COMSOL `auxq`どおりの二電流same-form結果で、後続three-currentや
   species-resolved物理を認定しない。optional three-current productionはP21 priority 1で別revisionとして完了した。
5. accepted Case-P seed `319032/319047/319063`の287粒子owner discoveryは、受理済み科学payload・work・case identity・
   revisionを完全一致させて完了した。支配ownerは3 seedとも`integrators`（自己時間比42.58--42.86%）だったが、事前登録済み
   bounded ownerではないため最適化は未承認で本体変更はない。M3-C2A anchorは`CLOSED_ACCEPTED_WITH_LIMITATIONS`とする。
   10,000粒子以上の性能と外部process並列は利用SLAを先に定義した別work packageであり、benchmark完了条件にしない。
   後続の明示指示によるbounded chord follow-upはaccepted science/work identityを維持して12.29%短縮し、追加ownerへ進まず完了した。
   これはdataset/COMSOL比較の変更または製品scale認定ではない。
   M3-C0b/C1/C2の第2 ion-drag、10/30 nm、残るpackage展開も別の未認定軸として残し、
   各段階で必要なdynamic charge/force ablation、admissible step系列、derived-field/RK probeを閉じる。
6. 旧12ケースは`reference_only`の統合・感度資料に限定し、新しいPASSを上書きしない。

旧event v13 candidateの約`3e8` accepted piece/run見積りと、
3 macro stepがcertificateにより同じterminal pieceへ細分された結果は、artificial subdivisionを含む履歴値である。
event v14はこの過保守を解消し、Case-A 100 nmのsolver-only 3刻みも独立に評価済みである。保存済みM3-C1
evidence/evaluatorはv14履歴として固定し、旧見積りを現行計画の性能根拠には使わない。B03完了後の新matrixは現行event v22で
新candidate/evidenceを作り、高コスト・低情報な総当たりではなく上記順序で段階展開する。

V01～V06はこの順序と競合する別runtimeではなく、対応能力の解析・統計microcaseとして必要なstageで維持・追加する。
とくにV06/Brownianを決定論anchorなしに合否判定へ混ぜず、単一seed pathを正解にしない。

COMSOL licenseまたは再実行時間が一時的に不足しても、V01～V06の解析解側でcore開発を進められる。

---

## 9. 外部V&Vの合否

- V01：線形場を補間精度内で再現し、support・cell IDが一致
- V02：methodの理論次数で軌道誤差が収束
- V03：初期状態が指定値と一致し、RZ samplingが面積分布に一致
- V04：first-hit時刻・点・facet・反射速度が収束
- V05：chargeと軌道の連成解が解析解へ収束
- V06：平均、分散、MSD、平衡速度分布が信頼区間内
- chamber reference：field→force→charge→trajectory→eventのどの層で差が始まるか特定可能

最終座標だけの一致、PNGの見た目、`package_validation.csv`の構造PASSを合格条件にしない。

---

## 10. 完了条件

- 現12ケースが`reference_only`として隔離される。
- 六つのmicrocaseが解析解とCOMSOL exportを持つ。
- primitive fieldとtopologyが粒径variant間で重複しない。
- deterministic chamber referenceを同じtoolchainで再実行できる。
- external toolが製品coreの公開APIだけを使用する。
- 製品coreにCOMSOL名、tag、列名、version分岐が存在しない。
