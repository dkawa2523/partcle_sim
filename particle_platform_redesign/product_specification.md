# 半導体チャンバー粒子軌道計算基盤 実装前製品仕様書

## 0. 文書の位置づけ

本書は、新しい粒子計算基盤を実装する前に、製品目的、物理モデル、数値方式、入力、境界、
コード責務、拡張方法、運用方法を固定する主仕様である。既存コードを整理するための文書ではなく、
既存コードを実装母体にしないclean-room開発の出発点とする。

本書の優先順位は次のとおりである。

1. 半導体製造装置内のコンタミ粒子について、物理的に説明可能な軌道・壁到達・付着分布を得る。
2. 1万～100万粒子を、同じ物理条件の汎用有限要素解析より効率よく計算する。
3. 力、帯電、壁則、発生源を追加しても、既存の計算経路が分岐だらけにならないようにする。
4. 外部で求めた流れ・温度・プラズマ場を、COMSOLに限らず同じ標準入力へ変換できるようにする。
5. COMSOLとの比較は製品の検証手段の一つとし、solver coreの責務にしない。

文書間の仕様driftを防ぐため、権威を次のように分ける。

| 内容 | 権威 |
|---|---|
| 製品目的、範囲、非目標、stage、公開機能 | 本書 |
| module責務、依存方向、single-engine runtime | [architecture_proposal.md](architecture_proposal.md) |
| 方程式、相関、適用域、数値式 | [technical_research.md](technical_research.md) |
| 詳細な実装順、work package、stage出口作業 | [implementation_plan.md](implementation_plan.md) |
| Python環境、静的品質tool、quality gate | [quality_tooling_plan.md](quality_tooling_plan.md) |
| COMSOL・datasetを使う外部評価手順 | [vv_methodology.md](vv_methodology.md) |
| datasetの許可用途・禁止用途 | [data_quality_assessment.md](data_quality_assessment.md) |

[architecture_review.md](architecture_review.md) は、本書と詳細設計へ反映した指摘と判断理由を残す
レビュー記録であり、別の実装仕様を作るものではない。重複箇所が衝突した場合は上表の所有文書を
正とし、衝突を放置せず同じ変更で修正する。

---

## 1. 製品の目的

### 1.1 解く問題

外部計算または製品付属の場生成器から得た空間・時間分布と装置ジオメトリーを入力し、粒子ごとの
位置、速度、電荷などを時間積分する。一方向連成を標準とし、粒子は背景場を読み取るが、背景場を
変更しない。

標準状態を

\[
\boldsymbol{y}_p=(\boldsymbol{x}_p,\boldsymbol{v}_p,Z_p,\boldsymbol{\xi}_p)
\]

とする。ここで \(\boldsymbol{x}_p\) は位置、\(\boldsymbol{v}_p\) は速度、
\(Z_p=q_p/e\) は平均電荷数、\(\boldsymbol{\xi}_p\) は必要な場合だけ持つ追加状態である。
追加状態の例は粒子温度、離散電荷状態、表面滞在時間であり、使用しない状態を全粒子へ常設しない。

基本方程式は

\[
\frac{d\boldsymbol{x}_p}{dt}=\boldsymbol{v}_p,
\qquad
m_p\frac{d\boldsymbol{v}_p}{dt}=\sum_k\boldsymbol{F}_k,
\qquad
\frac{dZ_p}{dt}=R_Z(\boldsymbol{x}_p,\boldsymbol{v}_p,Z_p,t)
\]

である。球形粒子の標準質量は

\[
m_p=\rho_p\frac{\pi d_p^3}{6}
\]

だが、canonical particleでは次を独立した権威として持つ。

```text
mass_kg                  慣性
drag_diameter_m          drag、Kn、Re
electrostatic_radius_m   OML、電気・DEP model
displaced_volume_m3      浮力
model_weight             一つの計算粒子が代表する実粒子数
```

球形の簡便入力として`diameter_m + density_kg_m3`を受ける場合、case builderが上記へ一度だけ展開し、
導出式をprovenanceへ残す。入力で`mass_kg`が与えられた場合はそれを慣性の権威とし、runtimeで密度と
径から再構成しない。非球形・多孔質粒子の相当径を単一の`diameter`へ押し込まない。

### 1.2 主な利用場面

- チャンバーパーツ表面から剥離・再飛散した粒子の到達先推定
- 粒径、初速、発生位置、発生時刻、帯電状態の感度解析
- ウェハ、壁、電極、排気口ごとの付着確率・到達時間・入射条件の評価
- 流れ、熱泳動、静電力、イオンドラッグなどの寄与比較
- 装置条件または任意プラズマパラメータを変更した多数ケースの高速スクリーニング
- 定常2D軸対称場から始め、時間依存2D場、軸対称場中の3D粒子、完全3D場へ拡張する解析

### 1.3 製品の成功条件

成功を「COMSOLと同じ最終座標になること」だけでは定義しない。次を満たすことを製品要件とする。

- 選択した物理モデルと適用範囲が入力・結果に残る。
- 時間刻みとメッシュを細かくしたとき、決定論解が期待する次数で収束する。
- 壁衝突時刻、衝突点、境界ID、入射速度、壁応答を再現可能に記録する。
- 確率モデルは同一seedで再現でき、異なるseedでは正しい集団統計を与える。
- 主要な製品物理を有効にしてもPythonの粒子ごとのloopへ退化しない。
- 10^4、10^5、10^6粒子で計算、境界、出力を含む速度と最大メモリを測定できる。
- 新しい力または壁則を追加する変更範囲が、その所有moduleと少数の登録箇所に限定される。

### 1.4 非目標

初期製品では次を目的にしない。

- COMSOLのUI、study、変数名、内部solverを複製すること
- 流体、伝熱、詳細プラズマ化学を粒子solver内で再計算すること
- 粒子が背景場を変える二方向連成
- 粒子間衝突、凝集、破砕、表面成長を最初から同時実装すること
- 任意CADを自動修復する一般ジオメトリーシステム
- 全ての力相関を一つの自動選択式へまとめること

未対応の物理は別式へ暗黙に置き換えない。明示的に拒否するか、検証用reference backendでのみ
動くことを表示する。

---

## 2. 製品全体の責務境界

```text
外部解析 / 計測 / COMSOL / CFD / plasma code
                         │
                         ▼
                importer / field builder
                         │
          Canonical DataBundle（SI、明示topology）
                         │
                         ▼
┌──────────────── Particle Solver ────────────────┐
│ field sampling → force/charge → integration     │
│             → first-hit boundary → result       │
└─────────────────────────────────────────────────┘
                         │
             Canonical Result + Event Log
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
      analysis / visualization      external V&V
                                   └─ COMSOL compare
```

### 2.1 solver本体が所有するもの

1. 標準caseの読込みと一度だけの入力検査
2. 場の補間とsupport判定
3. 粒子状態、力、帯電、確率過程の時間更新
4. 移動区間上の最初の境界イベント
5. 壁応答と粒子lifecycle
6. CPUバッチ実行、将来のGPU実行
7. 軌道、イベント、集計のstreaming出力
8. 実行に必要な最小限の数値統計

### 2.2 本体の外へ置くもの

- COMSOL MPH操作、Java exporter、COMSOL列名変換
- COMSOL軌道との差分計算、原因診断、HTMLレポート
- `model_dataset` の再実行、修復、版管理
- CAD修復、mesh生成、一般的な再mesh
- 熱流体場とプラズマパラメータからPoisson方程式を解く静電場生成器
- 可視化GUI、ケース作成GUI、旧設定migration

静電場生成器は公式製品機能に含めるが、粒子solverの上流にある独立componentとする。同じ
Canonical Fieldを出力するため、外部プラズマ場と内部生成場は粒子solverから見分けられない。

### 2.3 COMSOLの位置づけ

COMSOLは次の三用途に限定する。

- 既存利用者の入力をCanonical DataBundleへ移すadapter
- 代表ケースの軌道、局所力、壁イベントを照合する外部V&V oracle
- 未確定モデルの感度を比較する研究用参照

COMSOL固有の固定刻み、乱数、状態名、境界挙動は `tools/importers/comsol/` と
`tools/vv/comsol/` に閉じ込める。必要なら
solverの一般的なRK4や壁則を選ぶが、`if comsol_case:` のような分岐をcoreへ入れない。

---

## 3. モデル選択の原則

### 3.1 原始場と粒子依存量を分ける

入力場には、粒径や電荷を変えても変わらない原始量を保存する。

- ガス速度、温度、圧力、密度、粘度、熱伝導率、組成
- 電位、電場、電場二乗またはその元となる複素電場
- 電子密度、正・負イオン密度、電子温度、イオン温度、イオン流速・質量
- 温度勾配、必要なら速度勾配または渦度
- 場の有効領域、時刻、mesh topology、単位、座標系

粒子Knudsen数、緩和時間、Brownian拡散係数、表面電位、drag係数のように粒径・速度・電荷へ
依存する値は、実行時に物理moduleが計算する。これを背景CSVへ保存しない。

### 3.2 無次元数で適用域を記録する

solverは少なくとも次を計算可能にする。

\[
Kn_p=\frac{\lambda_g}{d_p},
\qquad
Re_p=\frac{\rho_gd_p|\boldsymbol{u}_g-\boldsymbol{v}_p|}{\mu_g},
\qquad
St=\frac{\tau_p U}{L}
\]

さらに必要に応じて相対Mach数、Debye長比 \(a/\lambda_D\)、電荷緩和時間比、壁までの距離比を
評価する。無次元数は診断を増やすためではなく、選択した相関式が適用範囲にあるかを判断する
ために使う。

### 3.3 自動的な物理式切替を標準にしない

設定では `epstein`、`stokes_cunningham`、`talbot` のようにモデル名を明示する。実行中に
閾値を跨いだから別相関へ黙って切り替えない。広い適用域を持つ単一相関を選ぶか、利用者が
明示したblend modelだけが連続的に遷移する。

適用域外では次のいずれかを設定で選ぶ。

- `error`：実行を拒否する。正式計算の標準。
- `count`：感度解析用。継続するが、model別の範囲外件数・割合と未検証使用の印をresultへ必ず残す。

数値安定化のためのclampと、物理モデル適用域の逸脱を同じ処理にしない。

---

## 4. 力学モデル群

### 4.1 共通形式

各力は、粒子状態とその位置でsampleした原始場から加速度への寄与を加える純粋な計算とする。

```text
required fields + particle state + parameters → acceleration contribution
```

force自身はmesh検索、時間積分、境界処理、ファイルI/Oを行わない。production backendでは
粒子ごとのPython callbackを禁止し、粒子配列をまとめて処理する。

物理式を力[N]で説明しても、runtime境界では次へ統一する。

- 線形drag：`rate_s_inv` と `target_velocity_m_s`
- それ以外の決定論項：`acceleration_m_s2`
- 電荷などの連続内部状態：`rate_in_state_unit_per_s`
- 確率項：選択したSDE integratorが定める局所係数

診断上の力は指定probeについてだけ `force_N = mass_kg * acceleration_m_s2` から作る。modelごとに
力と加速度を混在させず、質量除算を複数箇所へ分散させない。

### 4.2 中性気体drag

#### Stokes–Cunningham

連続体またはslip領域の球形粒子では

\[
\boldsymbol{F}_D=
\frac{3\pi\mu_gd_p}{C_c}
(\boldsymbol{u}_g-\boldsymbol{v}_p)
\]

を用いる。\(C_c\) は選択したCunningham補正式であり、使用した係数とKnudsen数の定義を
caseへ保存する。有限Reynolds数を扱う場合は、別の明示モデルとしてdrag係数 \(C_D(Re_p)\)
を選ぶ。

#### Epstein

自由分子領域かつ相対速度が熱速度に対して小さい場合の基本形は

\[
\boldsymbol{F}_D=
\delta\frac{4\pi}{3}a^2\rho_g\bar c_g
(\boldsymbol{u}_g-\boldsymbol{v}_p)
\]

である。\(\delta\) は分子反射・熱適応を表す係数、\(\bar c_g\) は平均熱速度である。高い相対
Mach数を扱う場合は、明示的に非線形Epstein補正を選ぶ。

線形dragでは

\[
\frac{d\boldsymbol{v}}{dt}
=\frac{\boldsymbol{u}_g-\boldsymbol{v}}{\tau_p}+\boldsymbol{a}_{other}
\]

と書けるため、\(\tau_p=m_p/\beta\) を用いた指数更新をnative積分器の基本にする。これは
ナノ粒子で \(\tau_p\ll\Delta t\) となる場合の剛性を、極端に小さい刻みなしで扱うためである。

初期製品は `epstein` と `stokes_cunningham` を実装する。transition相関、非球形粒子、
高Re dragは後続とする。

P18-Rの`epstein_linear_effective_gas_sensitivity_v1`は上の線形Epstein式を再利用し、producerが混合気体を
一つの有効Maxwellian/pseudogasへ畳み込んだreference/sensitivity入力だけを受理する。caseは
`0 < maximum_speed_ratio <= 1`を明示し、全stage・連続pathで(lambda/age10)とともにfail-closedに検査する。
既存`epstein_linear_v1`の速度比上限`0.1`は維持し、このrevisionをspecies-resolved mixture truthやCOMSOL branchとは呼ばない。

### 4.3 重力と浮力

\[
\boldsymbol{F}_{g+b}=(m_p-\rho_gV_{disp})\boldsymbol{g}
\]

とする。慣性には`mass_kg`、浮力には`displaced_volume_m3`を使い、`m_p=\rho_pV_p`をruntimeで
仮定し直さない。低圧ナノ粒子では小さいことが多いが、式が単純で装置姿勢の感度に有用なため
Stage 1へ含める。

`gravity_buoyancy_standard_v1`は、axisを含むかどうかにかかわらず、すべての`axisymmetric_rz` domainで
`gravity_m_s2[0] = g_r = 0`を要求する。一定の非零`g_r`は空間内で向きが回転するradial body accelerationであり、
通常の一様重力として受理も再解釈もしない。これに対し`cartesian_xy`では第1成分`g_x`を非零にできる。
radial body accelerationが必要になった場合は、適用域を持つ別model revisionとして追加する。

### 4.4 電気力と磁気力

\[
\boldsymbol{F}_{EM}=q_p(\boldsymbol{E}+\boldsymbol{v}_p\times\boldsymbol{B})
\]

を一般形とする。初期製品は \(q\boldsymbol{E}\) を必須対応し、磁気項は要求ケースが得られた後に
追加する。電位しか入力されない場合、gradientはimporterまたはfield builderで一度生成し、
粒子loop内で不規則meshの数値微分を繰り返さない。

### 4.5 誘電泳動

球形粒子、準静電場の基本形を

\[
\boldsymbol{F}_{DEP}
=2\pi\epsilon_m a^3\operatorname{Re}(K_{CM})\nabla|\boldsymbol{E}_{rms}|^2
\]

とする。DCでは実数のClausius–Mossotti係数、RFでは複素誘電率と周波数を明示する。
`E` と `E_rms`、瞬時値と周期平均値を混同しない。gradient品質とmodel-form不確かさが大きいため、
Stage 1の初期製品へは入れない。Stage 3のP18-Dでは、producerがDCまたはcycle-meanの意味、solution、
回復法を固定したcanonical `grad(mean_E_squared)`を受け取る球形準静的revisionを、明示的なoptionとして追加する。
粒子loop内でEを微分せず、複素周波数依存やtravelling-wave DEPは別revisionとする。

### 4.6 熱泳動

最初のproduction revisionは
`waldmann_gallis_free_molecular_single_species_heat_flux_v1`である。coreは温度を微分せず、局所質量平均の
中性気体座標で定義された並進伝導熱流束 \(\boldsymbol q_{tr}\) をcanonical fieldとして受け取る。

\[
\bar c=\sqrt{\frac{8k_BT_g}{\pi m_g}},\qquad
\boldsymbol F_{th}=\frac{32}{15}\frac{a^2}{\bar c}\boldsymbol q_{tr}.
\]

単一気体のFourier域でproducerが
\(\boldsymbol q_{tr}=-\kappa_{tr}\nabla T\)を形成すれば、力は高温側から低温側を向く。solver内では
total heat flux、対流・放射・電子・イオン熱流束を並進伝導熱流束へ読み替えない。

v1は球形、単一中性気体、dilute one-way、自由分子・低相対driftに限定し、全pathで
\(\lambda/a\ge10\)かつ\(|\boldsymbol u_g-\boldsymbol v|/\bar c\le0.1\)をfail-closedに要求する。
この閾値はsharpな物理境界でなく保守的なrevision policyである。Talbot型transition/continuum、混合気体、
near-wall・accommodation補正、negative thermophoresis、粒子内温度偏りとphotophoresisは別revisionとする。

P18-Rの`waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1`は同じ力式を、producer認証済みの
one-effective-Maxwellian/pseudogasへ適用するoptional reference/sensitivity revisionである。入力`q_eff`はproducerが
所有する有効並進伝導熱流束であり、coreは温度gradient、species配列、mixture ruleを回復しない。
全stage・連続pathで(lambda/age10)かつcase明示の`0 < maximum_speed_ratio <= 1`を要求し、P16単一気体revisionの
上限`0.1`を変更しない。設定可能であることはspecies-resolved mixture truthまたはCOMSOL一致の認定ではない。

### 4.7 Brownian運動

慣性を保持するLangevin形式は

\[
m_pd\boldsymbol{v}
=-\beta(\boldsymbol{v}-\boldsymbol{u}_g)dt
+\sqrt{2k_BT_g\beta}\,d\boldsymbol{W}
+\boldsymbol{F}_{other}dt
\]

である。線形dragと局所係数をstep中固定できる範囲では、速度と位置を含む
Ornstein–Uhlenbeck過程の厳密更新を使う。単純な「ランダムな力をRK4へ入れる」方式をnative
solverの標準にしない。

緩和時間を消去したoverdamped形式は別integrator profileとし、同じstep内で慣性形式と混在
させない。壁イベントでstepを分割する場合、乱数増分を引き直さずBrownian bridgeまたは等価な
条件付き分割を使う。

Brownianは決定論core完成後の第2段階B01/B02で実装した。乱数identityは
`(seed, particle_id, macro_interval, root_stochastic_interval, tree_level, tree_index, component, stream)`
から決め、
accepted-step番号、粒子の並べ替え、thread数、chunkサイズによって結果が変わらないようにする。
B01でjoint OU更新、interval-tree Philox、conditional half-splitを固定し、B02でcase選択、材料first hit、
trajectory replay、checkpoint/resumeへ接続した。各leafはOU endpoint位置・速度から作るcubic Hermite numerical
pathであり、連続OU first-passageの厳密解ではない。B01/B02の平均更新は線形dragだけを厳密化したもので、
`F_other`を黙って落とさない。B02 production revisionはEpstein linear drag、fixed charge、Cartesian XY、terminal
stick/escapeに限定し、その意味とpayloadをB03で変更しない。
root covariance、conditional split、mean更新がfloat64で表現不能な粒子は最後のaccepted stateで
`nonfinite_physics`となり、同一batchの正常粒子は継続する。

Stage 3のB03では、現在のRZ比較用に2自由度meridional投影であることを明記したrevisionを追加して完了した。
root始点からのnoise-free midpoint predictorで線形Epstein/FDT係数、全additive force、continuous chargeの
`G=dZ/dt,J=dG/dZ<=0`を一度凍結し、`u_eff=u+a/gamma`のjoint exact OUとroot内affine exponential
chargeで進める。これはstateを二つの
half-stepでcommitする対称operator splittingではない。native/effective-gasの線形Epstein、fixed/continuous charge、
既存additive forceを扱い、terminal wallは`stick`/`escape`/`hold`に限る。axis hit後はaccepted prefixをfoldし、
残時間を新しいstochastic rootとして再開する。`macro_root_affine_exponential_v2`の全leaf intervalはprepared invariantへ照合し、
逸脱または数値的に証明不能ならfail-closedにする。これは等方3-D Brownianの代用ではない。製品標準の3-D確率運動は
P17後のCartesian 3-D revisionで扱い、単一seedの軌道一致を合否にしない。
B03 path arrayの静的な保守上限は一slab rowあたり`648 B`で、受入上限`2048 B`を満たす。正式characterizationは
2,000/20,000粒子×4構成×3反復を全粒子active・failure 0で完了した。計時とprocess RSSは
[`solver/evidence/b03/`](solver/evidence/b03/README.md)のmachine-local・non-gating証跡とし、製品共通の性能値へ
一般化しない。characterizationで検出した反復加算由来の終端tailはengine v34で修正し、macro timeを補償積和による
`start + n*dt`のindexed gridから構築する。endへのsnapはfloat64構築roundoff内だけで行い、科学的に意味のある残時間を消さない。

### 4.8 帯電粒子へのion drag

ion dragは、少なくともcollection成分と軌道散乱成分を区別する。

\[
\boldsymbol{F}_{id}=
\boldsymbol{F}_{collection}+\boldsymbol{F}_{orbital}
\]

遮蔽長、Coulomb logarithm、衝突性、イオンdrift、粒子表面電位の閉じ方によって結果が大きく
変わるため、普遍的なdefault式を置かない。`model_dataset` にある2式は有用な感度モデルだが、
どちらかを製品の物理的正解として固定しない。

P15-Fではこの原則に従い、単一・単価正イオン、非正粒子電位、linear two-species Debye screening、
Debye--Hückel表面電位、collisionless・非磁化backgroundに限定したBarnes型collection＋orbital modelを
production revisionとして追加した。力は相対流方向、Coulomb logとimpact parameterはfloor/clampなし、適用外は
fail-closedとする。collisional sheath、非線形screening、image force、電場方向化は別revisionである。

Stage 3のP18-Iでは、現在の比較対象で使われるrelative-flow screened式とelectric-field-directed image式を、
それぞれ独立したoptional sensitivity revisionとして追加する。既存Barnes revisionは変更せず、一つの`ion_drag`
categoryから排他的に選択する。二式のblend、自動fallback、Case P/Aによるcore分岐、軌道差を小さくするparameter fittingは行わない。

### 4.9 lift、圧力勾配、付加質量、履歴力

Saffman lift、圧力勾配力、付加質量、Basset履歴力は、連続体中の有限慣性粒子には意味がある。
一方、現在の主対象である高Knudsen数の10～100 nm粒子へそのまま適用できない。

- `saffman`：低Re・連続体・単純shearの適用条件を満たす場合だけ
- pressure-gradient / added-mass：流体密度が粒子慣性に対して無視できない場合
- Basset history：履歴積分用の状態と計算量が必要

これらは初期標準から外す。一方、現在の比較対象で式と入力が確認できる「自由分子lift」は、Stage 3のP18-Lで
`axisymmetric_rz`、no-swirl、高Knの明示適用域だけを持つoptional sensitivity revisionとして追加済みである。

\[
\boldsymbol F_L=K(\omega_\phi\boldsymbol e_\phi)\times(\boldsymbol u_g-\boldsymbol v),
\qquad K=C_L\pi\rho_g\lambda_g a^2,
\qquad a=\frac{\texttt{drag\_diameter\_m}}{2}.
\]

`C_L`は有限正値をcaseへ明示する。producerが同じgas-velocity solutionからsigned方位vorticity `[1/s]`を形成し、
coreは速度場を微分しない。gas velocity `[m/s]`、density `[kg/m^3]`、mean free path `[m]`とともに、
`lambda_g/a>=10`を全stage/pathでfail-closedに要求する。B02 Brownianとの同時利用は拒否する。これを一般的なliftや
Saffman liftとは呼ばず、物理的なdefaultにも自動選択にも使わない。

### 4.10 後続候補

- near-wall drag補正
- image charge、van der Waals、接触前の付着力
- diffusiophoresis、photophoresis、radiometric force
- 粒子温度と熱収支
- 粒子回転、非球形粒子
- 粒子間衝突、凝集、破砕
- 二方向連成

これらを予想してcoreへ空のinterfaceや設定を先に作らない。実ケース、式、必要状態、検証方法が
揃った時点で追加する。

---

## 5. 粒子電荷モデル

### 5.1 権威となる状態

内部状態は電荷数 \(Z\) を権威とし、電荷は \(q=Ze\) から求める。`charge_C` と `charge_number`
を独立した可変入力にしない。fixed chargeの場合も同じ状態配列を使用し、時間微分を0とする。

### 5.2 段階的に対応するモデル

#### Fixed charge

Stage 1で実装する。入力した \(Z_0\) を全時間で保持する。電場・境界だけを切り分ける決定論試験にも
使用する。`fixed`は`dZ/dt=0`だけを意味し、初期値 (Z_0) は各sourceが所有する。無帯電は
sourceの`charge_number: 0`と`charge: {model: fixed}`の組で明示する。

#### Continuous mean charge

電子と正イオンの収集率から

\[
\frac{dZ}{dt}=R_i-R_e+R_{emission}
\]

を積分する。Stage 2Aでは \(R_{emission}=0\) とし、適用域を分離したversion付きOML系平均収集modelを実装する。
粒子表面電位、有限Debye長、電子・イオン温度とdrift適用域に用いた式をmodel versionとして固定する。

P15の初回model IDは`oml_stationary_maxwellian_debye_huckel_v1`とする。stationary Maxwellian OMLの
電子・イオン収集率とDebye--Hückel capacitanceを組み合わせ、`a/lambda_D <= 0.1`を必須適用域とする。
ion driftは収集率の補正項にせず、resolved `M_i <= 0.1`をstationary近似のapplicability gateにだけ使う。
gate外や必要primitiveを有限に認証できないcaseを別model、clip、経験式へ自動切替しない。
`model_dataset`の帯電heuristicとCOMSOL結果は外部V&V・感度評価には使えるが、式、parameter、適用域の
core authorityではない。

現在の12 packageを同条件で比較するStage 3では、保存式を独立に再定義した
`aggregate_relative_drift_regularized_two_current_v1`をP18-Cのoptional reference revisionとして追加する。
正負表面電位branch、相対drift、正則化速度、ion-energy floor、指数範囲はrevision意味論として公開し、P15/P15-Dの
物理的defaultや適用域を変更しない。有限なZ invariant、rate/derivative boundを証明できない場合は実装を止め、
比較のために既存の数値安全条件を緩めない。
局所有効正イオン質量、電子・正イオンthermal voltage、背景screening長は正値canonical scalar fieldを唯一のauthorityとする。
caseは正則化前の相対ion speedの有限run-wide上限を必須指定し、stage/pathで超過した時はfail-closedとする。

有限相対driftが必要な単一・単価正イオンcaseには、P15-Dで
`oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1`を追加した。Maxwellian電子、
shifted-Maxwellian正イオン、球形完全吸収粒子、非正表面電位だけを扱い、caseが
`M_i=|u_i-v|/sqrt(8 k_B T_i/(pi m_i))`の有限正値上限を明示する。初期`Z<=0`と全宣言rangeでの
非正平衡をprepareで認証し、zero-driftではstationary OMLへ一致する。負イオン、複数イオン種、正電位、
emission、collisional/magnetized chargingへ自動拡張しない。

運動と電荷は同じstageまたは同じpredictor/corrector時刻で評価する。engineが電荷を先に更新するの
ではなく、integratorがstage evaluatorを介して連成する。stepの最初に電荷だけ更新し、その値を
全stepへ使う一次operator splitを標準にしない。

### 5.3 後続モデル

- 離散電子・イオン捕獲をbirth–death過程として扱うinteger charge
- 二次電子放出、光電子放出、熱電子放出
- 材料依存のsticking coefficient
- 衝突性sheath、非Maxwell電子、イオン種別電流
- 壁接触時の電荷移動

10 nm級で電荷個数が小さい場合、連続平均電荷と離散電荷は同じ物理ではない。両者を単なる
solver optionとして置換可能とせず、異なるモデルとして結果に記録する。

### 5.4 帯電モデルの数値要件

- 電流または捕獲率の単位を明記する。
- 指数関数の数値保護域と物理適用域を区別する。
- charge relaxation timeが時間刻みに対して短い場合を検出する。
- 非線形方程式で平衡電荷を求めるmodelは、bracketと残差を持つ独立solverとする。
- 電荷が非有限になった場合、0へ戻さずその粒子またはrunを失敗させる。
- 最初に受け入れたreference pathは`(x,v,Z)`を同じstageで評価するRK4とし、model固有のcharge-relaxation
  boundが解像可能と証明した非stiff範囲だけを受理する。有限なcharge invariant、rate上界、
  `L_Z >= |dR_Z/dZ|`を必須とし、`h L_Z <= 0.5`と全RK stageのinvariant包含を満たさなければfail-closedとする。
- 動的chargeはstage評価が必要な連続stateであり、forceの有無にかかわらずlinear/quadratic exact pathを使わない。
- native指数midpointとB03はmidpoint-frozen `G,J<=0`のaffine exponential root pathを使う。
  exponential側へRK4の`hL_Z<=0.5`をstability gateとして流用しない。
- accuracyは別runの`h,h/2,h/4`で選ぶ。clip、charge-only subcycle、第二engine、implicit/equilibrium fallbackは使わない。
- 離散電荷は連続`internal_rate`へ偽装しない。実装時にだけjump-process用strategyを追加し、
  捕獲時刻と種類をboundary eventとは別streamで記録する。

---

## 6. 粒子発生源

### 6.1 必須のsource

初期製品は次の二つだけを持つ。

1. `table`：粒子ID、位置、速度、径、質量、電荷、発生時刻を明示する。
2. `surface`：選択した境界面上から分布に従って発生させる。

volume、line、time series injectionは、実ケースが必要になった段階で追加する。
P05 production profileは粒子表と固定電荷の意味論を確定するためtable sourceだけを受理した。
P07 exact-path sliceはsurface sourceと境界上departureを同じengineへ導入した。

### 6.2 surface source

surface sourceは本製品の主要機能である。v0.1が直接設定できる範囲は、境界group/ID、連続particle ID、
count、一定particle property、`uniform | edge_fraction`位置、`fixed | normal`速度、fixed release timeである。
任意のrealized interior位置・速度・粒径・電荷・時刻はtable sourceで与えられる。ただし材料boundary上のtable
startはstrict interior規則で拒否されるため、任意分布を外部生成したboundary-start ensembleは現行v0.1では
直接表せない。v0.1の代表surface releaseは、単一facet上の`edge_fraction`、またはXYの`line_length`と
RZの`meridional_length | revolved_area`を明示した`uniform`で表せるため、`realized_surface_table`は追加しない。
これらで表せない代表用途が後続で確認された場合は、その時点で独立したversioned source形式を検討する。

次はsurface sourceの後続catalog候補であり、現行v0.1で対応済みとはみなさない。

- 物理的発生率からmodel particle count/weightを作る規則
- 面上の重み場、明示的な離散点分布
- 時刻表、Poisson発生などの発生時刻分布
- 離散粒径群、連続粒径分布
- 電荷分布
- cosine角度分布、外部表による速度
- 位置と速度の相関

2D軸対称境界edgeを3D表面としてsampleする場合、edge長 \(ds\) ではなく

\[
dA=2\pi r\,ds
\]

で重み付けする。\(r=0\) 付近を同じ粒子密度で過剰sampleしない。
`revolved_area`が3-D回転面上の一様fluxに対応し、`meridional_length`は2-D断面上の診断・manufactured case用である。
利用例で両者を同じ物理分布として扱わない。

source乱数は`seed, source_id, source_particle_ordinal, draw_kind`をkeyとする独立streamにする。vector成分が
必要なmodelは成分ごとに重複しないdraw kindを割り当てる。
thread、tile、他sourceの粒子数で位置・速度・径・release-time drawが変わらない。Poisson scheduleも
source内ordinalで識別し、global sequential RNGを使わない。

### 6.3 境界上の初期位置

粒子を固定距離だけ流体側へ移動して開始しない。realized scheduleは位置とcanonical source facet IDを
保持し、owner cellと表面法線はprepared geometryを唯一のauthorityとする。resident local-coordinateを
第二のauthorityとして保持しない。最初の速度が流体側ならdeparture、壁側なら即時impactとして扱う。必要な数値許容差は
ジオメトリーscaleとfloat64精度から一度だけ決める。
P05で材料boundaryを使うtable sourceはstrict interiorだけを受理し、境界上の初期点を
epsilon nudgeで修復しない。

P07の一定加速度surfaceでは、外向き法線`n`に対する`v·n`の明確な符号を常に優先する。roundoff budget内でも
非零なら曖昧として拒否し、厳密tangentだけを証明済み一定加速度の`a·n`で分類する。内向き加速度のtokenは別wall
hitで速度が変わるまで保持し、event ownerが各intervalでsource supporting lineの内側/境界band、tangentまたは
明確な内向き速度、内向き加速度を再証明した時だけsource facetを除外する。他facetは常に通常検索し、別wall応答後は
source facetも通常検索へ戻す。外向き加速度はzero-time impactであり、terminal stick/escapeは許すが、反射で
内向きdepartureを生成できなければfail-closedにする。macro partitionや微小`dt`でこの意味を変えない。

---

## 7. 境界理論とイベント

### 7.1 first-hit原則

境界判定はstep終端のinside/outside比較ではなく、trial path上で最初に交差する境界を求める。

```text
step開始状態
  → trial trajectoryを提案
  → 最初の交差時刻・位置・facetを求める
  → boundary lawを適用
  → 残り時間を新状態から進める
```

integratorは内部`StepProposal`を返す。

```text
endpoint_state
piecewise path / state_at(theta)
path_error_bound
stage_support_status
local_error_estimate
```

P05のballistic pathは解析的な直線segmentであり、そのsegment上のexact first hitを実装した。曲線eventは
同時に一般化しなかった。
P05で材料boundaryを有効にしたgeometryは、volume incidenceから導いた外周に対して
topology-completeで一意な`line2`を持つ。RZの両端`r=0`のedgeはaxis seamであって材料壁ではなく、
それを材料boundaryとして登録した入力は拒否する。engine v9以降のexact pathとengine v12以降の一般RK4は、
最初の材料hitより軸到達が先ならwall eventを作らずmeridional座標をfoldする。force-coupled RZはsigned radial
trial chartで積分し、各stageをcanonical RZ field basisへ写す。axis foldはboundary event、wall RNG ordinal、
interaction countを消費せず、fold後stateから同じtargetまで残時間を継続する。axis accessibilityはgeometryの
axis接触、またはboundaryless fully-supported regular boxの`r_min=0`からfieldsが一度だけ導き、axis nodeの
vector regularityに使う。`gravity_buoyancy_standard_v1`の`g_r=0`制約はaxis accessibilityに依存しない。
wall/axisの双方が局在済みで認証時間が重なる時だけwallを優先し、axis端点とmaterial
cornerの完全tieは一般RK4 cornerとしてfail-closedにする。

P06でRK4等のforce-coupled pathを追加する時は、始終点を一本のchordとみなさない。一方、full-step/
two-half-step差やcurve–chord偏差はaccuracy indicatorであって、一般の非線形場に対する軌道包含boundではない。
最初にphysics係数・field extremaから、versioned integratorの離散pathがrequired field support内かつmodel applicability内にあることを
証明する。revision 3aはboundaryless Cartesian XY、fixed charge、既存のEpstein/electric/gravity、全cell
supportedな`RegularLayout`だけを対象とする。外向き丸めenclosureはfull macro stepに加えて、出力用
`state_at()`が生成する任意の短縮RK4について、内部stage位置・速度とaccepted endpointをすべて覆わなければ
ならない。Epsteinでは`lambda/a`下限と`|u-v|/c_bar`上限も全区間で証明する。証明不能caseは出力scheduleに
依存せずfail-closedにし、revision 3aでhidden subdivisionを導入しない。

revision 3bで、各local始点から同じboundを再構築できるcaseだけcheap no-hitを確定し、候補が残る区間は
geometry-drivenなsequential dyadic RK4 pieceとして再積分する。このaccepted piece列を離散pathのauthorityとし、
同一macro proposalのparameter区間は流用しない。構成できないcaseはfail-closedな明示failureとする。hit状態は
同じintegratorでhit時刻まで再評価する。材料boundaryでは先行hitをtrial全体のvalidity判定より先に扱う。
放物線、near-grazing hit/no-hit、turning trajectoryのmanufactured caseとstep半減でevent収束を確認する。

event v14は、このglobal absolute safety enclosureをsupport、global-first
applicability、hit時刻までの短縮RK4再積分のauthorityとして維持しつつ、global supportを独立に証明済みのvalid `rk4_dense` rowのevent BVH queryにだけ
そのcurrent pieceのdense Bernstein位置・速度boundを渡す。dense boundが非有限または不正ならglobal boundを
保持してsplitし、required supportをglobalに証明できないcaseではevent queryもglobal boundへ戻す。したがって、
これはbroad-phase candidate生成の精密化であり、first-hit順序、wall law、endpoint、schemaを変更しなかった。
event v15はこのauthorityを維持し、物理position budgetとroundoff budgetを加算した。facet-local offset dotは
補償演算、Hermite評価・包絡はroot-relative TwoDiffを使う。現行event v16はvalidなRK4 dense rowに限り、position
Bernstein control enclosureの全4点がfacetの既存budget込みinside half-spaceにあることをinterval算術で証明できた候補を
convex-hull性からclearする。証明できない候補、非有限・不整合なcontrol、exponential・scalar pathは従来どおり
split/fail-closedとする。monotone-approach clearもcubic Hermiteのderivative-Bernstein enclosureだけが明示opt-inする。

support外のprovisional値はtrial生成にだけ使え、support外stageを含むstepをそのまま受理しない。v0.1の
required fieldはparticle domain全体を覆う。ただしdomain coverageだけでは有限個のRK評価点間にある連続曲線の
包含を証明しない。現行の一定加速度pathは、topology-completeな材料boundaryによる退出event、または全cell
supportedな`RegularLayout`のsupport boxに対する解析的な座標極値検査でこの証明を行う。将来部分supportを
許す場合はpath上のfirst support exitをwallと同様に局在し、eventがなければ`FieldError`とする。

revision 3aのglobal enclosureはregular support boxだけを包含判定のauthorityとする。P1/Q1のnode extremaは
field値のboundには使えても、非凸または複数cellからなるsupport全体のbox包含を証明しないため、unstructured
productionを同時に解禁しない。RZ basis、continuous charge、時間依存fieldも別のboundを必要とする。

一つのmacro stepに複数hitを許す。`max_interactions_per_step`はmacro step全体のhit総数ではなく、
一つのresidual-work intervalを二分するtriggerである。上限到達後、残区間に次hitが存在する場合だけ等分し、
event-freeな残区間は追加depthなしで受理する。splitする場合は
子intervalのinteraction countを0へ戻し、refinement depthを引き継いで前半から再試行する。
明示depth budgetを超えたときだけ`failed:numerical_event_budget`とする。位置を動かして回避せず、物理的な
stick/escape数にも加えない。v0.1の衝突はpoint-particle、すなわち粒子中心が境界へ到達したeventで
ある。有限半径offset surfaceや接触運動は後続modelとする。

### 7.2 初期boundary law

P05 production profileはparameterを持たない`stick`と`escape`だけを受理した。
P07 exact-path sliceはspecular、probabilistic stick、複数candidateのcorner応答を追加した。
現行`point_wall_laws_v5`は、parameterを持たない完全鏡面`specular`、係数付き`restitution`、
明示した反射fallbackを持つ`probabilistic_stick`、非deposition終端の`hold`を別の意味として受理する。

#### Stick / deposit

粒子をactive集合から外し、衝突位置、時刻、境界、入射速度、電荷をdeposition eventとして保存する。
P05ではhit位置に固定し、post-event速度を0、lifecycleを`stuck`とする。衝突直前速度は
boundary eventの`velocity_pre_m_s`に保存して失わない。

#### Escape

排気口などから粒子を計算領域外へ出し、escaped状態とする。消失した後の座標をNaN軌道として
大量保存せず、event時刻でtrajectoryを終了する。
P05では表面impulseを与えず、boundary eventのpost速度はpre速度と同じである。

#### Hold / nondeposition terminal

粒子をactive集合から外し、hit位置、時刻、入射速度、電荷を保持した`held`状態とする。`stick/stuck`の
depositionにも`escape/escaped`のlogical nullにも集計しない。post速度とpost電荷はhit時のpre値に等しく、
final/frame/probeのkinematicsは有効であるが、hit後の位置、速度、電荷、力を時間発展させない。
これは停止粒子を後から再開するpaused-particle modelや再飛散modelではない。

#### Specular / restitution reflection

移動壁速度を \(\boldsymbol{v}_w\)、相対入射速度を
\(\boldsymbol{c}=\boldsymbol{v}-\boldsymbol{v}_w\) とする。

\[
\boldsymbol{c}_n=(\boldsymbol{c}\cdot\boldsymbol{n})\boldsymbol{n},
\qquad
\boldsymbol{c}_t=\boldsymbol{c}-\boldsymbol{c}_n
\]

\[
\boldsymbol{v}'=\boldsymbol{v}_w-e_n\boldsymbol{c}_n+e_t\boldsymbol{c}_t
\]

`specular`はparameterを持たず、常に\(e_n=e_t=1\)の完全鏡面反射である。反発係数を入力すると拒否する。
`restitution`は`normal_restitution = e_n`を`(0,1]`、`tangential_restitution = e_t`を`[0,1]`で
ともに必須とする。係数をruntimeでclampせずload時に検査し、二つのlawを同じselectorへ畳み込まない。
法線は粒子domainから外向き、impactは`(v-v_w)·n > 0`と定義する。現行v0.1は静止壁`v_w=0`だけを扱い、
moving wallは対応modelまで拒否する。

#### Probabilistic stick

\[
U<p_{stick}, \qquad 0\le p_{stick}\le 1
\]

なら付着し、それ以外は必須の`otherwise`へ明示したparameterなし`specular`、または係数付き`restitution`を
適用する。fallbackをdefaultで補わない。現行v0.1が扱うのはboundary lawごとの定数付着率だけであり、
境界material tableは未実装である。material、入射energy/angle、壁温度、電荷等へ依存する付着率は、入力変数と
適用範囲を宣言したversioned modelとして後から追加する。

wall-law乱数は`seed, particle_id, physical_boundary_event_ordinal, law_stream`をkeyにする。数値的な
path refinement、thread、tileはphysical event ordinalを進めず、付着drawを変えない。source、wall、
Brownian、将来のcharge jumpはstream IDを分離する。

### 7.3 後続boundary law

- cosine lawによるdiffuse reflection
- 壁温度とaccommodationに基づくthermal re-emission
- specular/diffuse mixture
- 表面滞在時間を持つdesorption
- 入射エネルギー・角度・材料によるerosionまたは再飛散
- surface chargeとの相互作用

contact slidingを標準の粒子軌道へ入れない。必要なら壁上運動という別問題として導入する。

### 7.4 corner、同時hit、軸

- 複数facetのhit時刻が数値的に一致する場合は候補facet集合を保存する。標準eventの`normal`は
  law適用に実際に使ったeffective response normalとし、単一facetではその外向き法線、combined-normal
  反射では選択subsetの正規化合成法線を保存する。
- 安定したboundary priorityはcaseで明示し、ファイル順に依存しない。
- 初期policyは`priority_then_combined_normal_v1`とし、候補形成、同priorityの矛盾、combined normal、
  failure条件の詳細は`solver/docs/numerics.md`を権威とする。
- 許容差を広げて都合のよいfacet一つへ寄せない。
- P05は同時hit候補をfacet ID順で保持するが、productionは単一candidateに決まるhitだけを
  受理する。複数candidateのlaw・法線解決を便利なtie breakで先回りしない。
- RZの \(r=0\) は通常の材料壁ではなく座標軸である。
- meridional kernelではtrial pathを軸で分け、`r <- -r`, `v_r <- -v_r`に相当する基底変換を一箇所で
  行う。これはwall eventでもreflectionでもない。
- 同一位置の再hitはepsilon nudgeで避けず、相対法線速度とpath方向からdepartureを判定する。

### 7.5 event record

次はlogicalな標準event modelである。`normal`はraw candidate normalではなく上記effective response normalである。
法線と局在残差は結果の根拠でありprobe限定にしない。

```text
particle_id, event_ordinal, event_type, time_s,
primary_facet_id, candidate_offset, candidate_count,
boundary_id, material_id, law_id, outcome,
position_m, normal, velocity_pre_m_s, velocity_post_m_s,
charge_number_pre, charge_number_post, model_weight,
localization_residual, stochastic_draw_reference_if_used
```

`event_type`は少なくとも`release / boundary / failure`を持つ。boundary固有列は他typeではnullableとし、
欠損を物理NaNと混同しない。corner/edgeの同時hit候補は別のcolumnar
`event_candidates(event_row, facet_id)`へoffset/countで格納し、単一facetへ情報を潰さない。
入力の`boundaries[].law`はmodel selector、PreparedRunの`law_code`は非公開dense実行値で、いずれも
event列ではない。`law_id`は選択されたtop-level lawの安定semantic IDとする。`probabilistic_stick`等の
compound lawが選んだleaf結果は`outcome`とdraw referenceで表し、`law_name`という重複列を作らない。
P05はreleaseを粒子ごとのevent ordinal 0、最初のboundary eventをordinal 1とする。保存frameや
final payloadからhitを推定せず、このboundary eventを最後の有効なkinematicsと壁統計のauthorityにする。
物理配置v1はnullableな統合tableを作らず、`/events/release`と`/events/boundary`へ分け、group名が
`event_type`を表す。candidateはboundary groupのragged offsetで保持する。P05で常に確定済みの
`geometry_status`や未実装failure groupは保存せず、実装済み列の権威は`solver/docs/result_format_v2.md`とする。

---

## 8. 座標系

### 8.1 明示的に分ける四つのmotion mode

`2D`、`軸対称`、`3D再構成`を一つのflagで表さない。

canonical dataの座標表現とparticle motion modeも別の事実である。HDF5はgeometry/field dataが
`cartesian_xy / axisymmetric_rz / cartesian_xyz`のどれで格納されるかを所有し、SimulationSpecは下記の
motion modeを所有する。prepareが組合せを一度だけ検査し、result manifestへ両方を保存する。初期v1は
XY data/XY motionとRZ data/RZ-meridional motionだけを受理し、RZ data/Cartesian3D motionはStage 2Aの
schema revisionで追加する。

#### `cartesian_xy`

平面2D問題。位置・速度・力はx-yの2自由度で、面外方向を物理的に無視する。

#### `axisymmetric_rz_meridional`

背景場と粒子運動の両方をr-z子午面へ制限する高速kernel。状態は
\((r,z,v_r,v_z)\)。決定論的かつ方位成分が0であるケースに使う。

等方的Brownian運動は方位速度を生成するため、このkernelは物理的な3D Brownian運動と等価では
ない。r-zに投影した別モデルとして使う場合は、その意味を結果に明記する。

#### `axisymmetric_field_cartesian3d`

背景場は2D RZだが、粒子状態はCartesian 3D
\((x,y,z,v_x,v_y,v_z)\) で積分する。

\[
r=\sqrt{x^2+y^2}
\]

でRZ場をsampleし、radial vectorをCartesian基底へ回転する。等方Brownian、方位速度、軸通過、
3D wall impactを扱う標準的な軸対称場中3D粒子modeとする。geometryは2D RZ断面を権威とし、3D pathを
`(r(s),z(s))`へ写した一般には曲線のpathと断面境界の最初の交点をbracket/localizeする。RZ法線
`(n_r,n_z)`は衝突点の方位角で`(n_r cosθ,n_r sinθ,n_z)`へ戻す。3D voxelや回転三角面を生成する
必要はない。軸上ではregularity条件を満たさないradial/azimuthal fieldを拒否する。

#### `cartesian_xyz`

完全3D geometryと3D fieldを用いる。後続段階で追加する。

### 8.2 共通化するものと専用化するもの

共通化するのは粒子属性、力のscalar係数、時間、結果、RNGである。次は座標kernelが所有する。

- 位置・vectorの基底変換
- 軸の扱い
- geometry queryへ渡すpathの座標写像と法線の逆変換
- surface sampling weight
- 有効な運動自由度

実際のcontainment、facet candidate、first-hit queryは`geometry`が所有し、`events`がStepProposal上の
局在と残時間を調停する。座標分岐を各forceやboundary lawへ散らさない。

---

## 9. 場、mesh、補間

### 9.1 GeometryDomainとFieldLayout

衝突境界と場の離散化は別の事実である。同じmesh storageを共有できるが、同一であることを前提に
しない。

```text
GeometryDomain:
  particle domain, oriented boundary facets
  boundary/material ID, owner, adjacency, geometry index

FieldLayout:
  mesh_id, regular axes または vertices/connectivity
  interpolation basis, support/domain, point-location index

FieldQuantity:
  layout_id, nodal/cell location, components/basis
  time knots, values, field semantics
```

Stage 1のunstructured fieldはP1三角形とQ1四辺形、完全3Dの最初はtet4 volumeとoriented tri3
boundaryだけを扱う。hex8、高次、mixed elementは実データが得られてから追加する。node/element ID、
局所node順、Jacobian符号を明示し、四辺形をCSV列順のpolygonとして推測しない。geometry remeshは
外部case builder、field resampling/cacheは外部field preprocessorの責務でありruntimeはmeshを修復しない。

### 9.2 field sampling

各sampleは概念的に次を返す。

```text
values, support, cell_id
```

`values` と `support` は別の質問である。trial stepの数値計算を有限に保つため近傍値を返す場合でも、
support外という事実を隠さない。support外値はprovisional trial専用であり、そのstageを含むstepは
受理しない。ただしこの型を全層へ公開する大きなcontract frameworkは作らず、field moduleと
integratorの間だけで使う小さな内部結果とする。batch backendではobject配列でなくpreallocated
value/support/cell配列へ書く。

cell supportはsupported cellの閉包の和とする。共有面を含むcandidateにsupported cellが一つ以上あれば
insideで、最小supported cell IDを決定論的ownerにする。candidateがすべてmaskedならoutsideとし、
provisional値は物理距離が最小のsupported cellへlocal coordinateを射影した有限値に限る。この規則を
regular/P1/Q1で共通にし、`cell_hint`や探索順でsupport結果を変えない。有限なsampleを構成できない場合は
明示errorとし、NaN/infやclamp値を返さない。

v0.1で運動を駆動するrequired fieldはparticle domain全体を覆うcontinuous node-associated
regular/P1/Q1 fieldへ限定する。cell-associated quantityは保存できるが、共有面の側選択modelを定義するまで
force/chargeへ使わない。producerのNaNをsupport定義に使わず、adapterが明示supportを確定してから
masked-only DOFを有限placeholderへ正規化し、その方法と件数をprovenanceへ残す。placeholderは物理補間へ
使用しない。

### 9.3 point location

regular layoutはsupported containing-cell common pathだけ軸indexからO(1)個のcell候補を評価する。
outside/masked provisionalは物理最近傍意味論を守るためcompiled全cell走査を使う。P1/Q1はaccepted endpointの
previous-cell hintがstrict interiorを含む場合だけそのcellを即時採用する。P14でhint missと初回sampleの
supported containmentが支配的と確認されたため、field v3はfield-owned stackless BVHで候補を絞り、従来と同じ
exact predicateと最小supported cell IDで確定する。containing supported cellがないoutside/masked provisionalだけは
意味論を守る全cell走査へ戻る。neighbor adjacencyは実測上の必要性がないため追加しない。

包含許容差は参照座標の固定値だけで決めない。物理空間の後退誤差、局所element scale、座標ULP、
Jacobian conditioning、物理点再構成残差を一つのalgorithm revisionで解決する。悪条件cellを許容差で
insideへ広げず、そのrevisionが根拠とともに固定するmesh品質上限を超えるcellを明示的に拒否する。この
上限は利用者向けtuning knobにしない。large-offset/high-aspect P1/Q1の
平行移動不変性をverificationに含める。

### 9.4 mesh-nativeと高速cache

権威あるreference samplingは元meshのshape functionで行う。規則格子への再sampleは
`tools/field_preprocessor` が作る任意の高速cacheとする。

cacheには元field hash、格子仕様、connected domain、support mask、field/gradientの
誤差評価を保存する。v0.1ではcacheを別fieldまたは別DataBundleとして生成し、physics設定が参照するlayoutを
利用者が明示選択する。場所ごとにmesh-native/cacheを切り替えるhybridは、必要性とspeedupを実ケースで
確認するまで実装しない。将来追加する場合も一つの宣言済みsamplerであり、
障害時のsilent fallbackではない。cache生成や品質診断をsolver hot pathへ入れず、support誤分類、
壁近傍誤差、representative trajectory/event、実測speedupの受入条件を満たす時だけ採用する。

### 9.5 時間依存場

fieldは将来、\(F(\boldsymbol{x},t)\) を同じinterfaceで扱う。Stage 4Aはfixed spatial topologyの
`hold`または`linear`だけとし、時間範囲外をclampしない。field knotと明示discontinuityをstepの必須
分割点にし、各RK/ETD stageの実時刻で補間する。runtimeは前後2 snapshotだけをdouble bufferする。
moving meshや時刻ごとに異なるtopologyは初期非対応とする。

入力snapshot自体がunder-resolvedならsolver stepを細かくしても修復できない。外部preprocessorが
間引き検証、temporal second difference、gradient/interface情報で空間・時間解像度を評価し、
`warn/error`を記録する。不連続面を跨いで補間しない。

### 9.6 急峻な空間分布

sheath、狭い流路、材料interfaceなどの不連続・急勾配はmetadataで明示し、異なるdomainやinterfaceを
跨いだstencilで平滑化しない。Stage 1はmesh-native fieldと`h,h/2,h/4`軌道収束を権威とし、外部
preprocessorが要素差、gradient、必要ならcurvatureからmesh/cache adequacyを判定する。後続の
bounded-dyadic modeでは、1 stepの移動距離、cell crossing、drag/charge係数と加速度の相対変化を
step-level indicatorにする。入力mesh自体が不足する場合はstepを細かくして成功扱いせず、再export・
外部remeshを要求する。

---

## 10. 数値積分基盤

### 10.1 production integratorは二つから始める

#### `rk4_fixed`

- 一般的な決定論ODE
- 小規模reference計算
- 外部solverとの条件一致
- 電荷と運動を同じ4 stageで評価

COMSOL固有の名称は付けない。外部adapterが必要な刻みと設定へ変換する。
既知の線形緩和について`h/tau_min >= 2.5`となるcaseは、安全余裕を持ったRK4安定域外としてprepareで
拒否し、結果が有限だからという理由で継続しない。stiff caseはstepを明示的に小さくするか
`exponential_midpoint`を選ぶ。

#### `exponential_midpoint`

- 線形dragの指数更新
- 外力、場、continuous chargeの`G,J<=0`を同じmidpointで評価
- ナノ粒子の短い速度緩和時間を安定に扱うnative標準
- 位置も速度の指数解と整合する形で更新

局所的に `dv/dt = -(v-u)/tau + a` と置き、midpointで係数を固定する。現行integratorは
`charge_stable_exponential_midpoint_v3`、速度依存するP18-Lを含む現行path enclosureは
`exponential_midpoint_global_abs_enclosure_v3`である。`E=exp(-h/tau)`、
`A=1-E`として

\[
v_1=u+E(v_0-u)+\tau A a,
\qquad
x_1=x_0+uh+\tau A(v_0-u)+\tau\{h-\tau A\}a
\]

を使う。`A`は`-expm1(-h/tau)`で、`h/tau`が小さい領域の係数は級数で評価する。start係数による
半step predictorからmidpointのfield、charge、drag、加算加速度を評価する。chargeは
`A_Z=G_mid+J_mid(Z_0-Z_mid)`として`Z_1=Z_0+expm1(J_mid*h)/J_mid*A_Z`、`J_mid=0`では
`Z_0+h*A_Z`で進める。運動と同じcoupled proposalであり、一定係数では解析解と丸め誤差程度、可変場では
global second orderを要求する。非線形dragを暗黙にこの式へ入れない。

最初から多数のRunge–Kutta法、一般IMEX framework、任意ODE solver pluginを作らない。

### 10.2 step幅

step幅は少なくとも次の時間scaleと長さscaleに支配される。

- drag relaxation time
- charge relaxation time
- 局所mesh crossing time
- field snapshot間隔とfield変化時間
- wallまでの予測時間
- 非drag加速度による速度・位置変化

Stage 1は固定macro stepと別runの`h,h/2,h/4`収束を正式に支援する。field knot、global source
discontinuity、boundary eventでは必要な区間を正確に分割する。個別release時刻はその粒子だけの残時間work
として扱う。要求出力時刻はaccepted pathから評価し、保存scheduleのためにstepを分けない。精度目的の
hidden adaptivityを入れない。
現行charge-stable pathへsubcycleを追加しない。粒子ごとのPython adaptive solver objectや専用multirate
frameworkも持たない。

### 10.3 errorと物理状態を混ぜない

- 数値誤差budget超過をstickやescapeへ変換しない。
- 設定に従い、その粒子を`failed:numerical`にするかrun全体を失敗させる。
- field support外、geometry不定、非有限physicsは別reason codeにする。
- 安全側に止めたという理由で、物理的な壁到達数へ加算しない。

複雑なaccuracy state machineは作らない。粒子statusは `pending/active/stuck/escaped/failed/held` の6種、
failed reasonは小さな整数codeとする。

### 10.4 stochastic integrator

線形dragに対する位置・速度のjoint OU更新を第3のintegrator `ou_langevin`として実装する。
決定論integratorにrandom force callbackを差し込まない。一般曲面でのinertial OU first passageを
exactとは主張せず、geometry/outputから独立した固定depth dyadic nodeとcubic Hermite leaf pathを数値pathとする。
平面解析case、depth収束、弱収束、到達時刻統計で品質を規定する。最初のrevisionはCartesian XY、Epstein linear
drag-only、fixed-charge state、terminal stick/escapeだけを受理した。P18-Hは同じB02 terminal subsetへ
parameterなしのholdを追加済みであり、反射や任意forceを解禁しない。

RNGはcounter-basedとし、identityを`seed, particle_id, macro_interval, root_stochastic_interval,
tree_level, tree_index, component, stream`から作る。accepted-step番号やthread順へ依存させない。
eventでstepを二分した場合はjoint Gaussianの親incrementを条件付きで左右へ分け、出力時刻を増やしても
乱数pathを変えない。

### 10.5 reference solver

高精度DOP853、Radauなどは `tests/verification` または小さな`tools/verification`で少数粒子を解く
referenceとして使う。
production engineへSciPyの粒子別solverを埋め込まない。
Stage 1Aのscalar field/physics evaluatorもverification oracleとして維持するが、P10以降のproductionで
compiled tileが失敗した時のfallbackにはしない。

---

## 11. 一つの単純な実行経路

全production caseは次の同じloopを通る。

```text
load once
  ├─ parse all YAML/resources
  ├─ reject canonical numeric footprint from HDF5 metadata before payload materialization
  └─ validate canonical schema, SI metadata, local connectivity, static references

prepare once
  ├─ resolve model requirements and coordinate/integrator/backend capability
  ├─ build geometry index, field sampler, PhysicsPlan, memory plan
  └─ allocate SoA state and bounded output sink

for each macro interval:
  1. release pending particles
  2. group active particles into chunks/subcycle level
  3. integratorが各stageでfield・charge rate・physics寄与を評価
  4. endpointと誤差付きpathをpropose
  5. path上のearliest boundary hitを局在
  6. hitまで同じintegratorで再評価
  7. boundary lawを適用し、残時間を同じ経路へ戻す
  8. accepted segmentからframe/event/countをemit

close output and write run summary
```

力ごと、mesh形式ごと、比較対象ごとに別runtimeを作らない。`rk4_fixed`、
`exponential_midpoint`、`ou_langevin`は同じengine内のstep strategyである。scalar evaluatorは
verification oracleであり、production時に暗黙fallbackしない。保存scheduleはaccepted pathから
評価し、trajectoryの有無・間隔・probe数がfinal state、event、RNG pathを変えてはならない。

---

## 12. 最小ソフトウェア構成

### 12.1 公開API

利用者が覚える操作は三つにする。

```python
from chamber_particles import load_case, simulate, open_result

case = load_case("case.yaml")
summary = simulate(case, output="runs/case_001")
result = open_result("runs/case_001")
```

- `load_case`：既にSIへ正規化されたcanonical dataと設定を読み、schema、unit metadata、局所connectivity・
  owner参照、静的な参照整合を検査する。producer単位の変換はしない。
- `simulate`：開始時のprepareでmodel requirements、座標×integrator×backend capability、compiled
  evaluator、大域geometry topology、memory/resource planを検査し、その後に数値計算とstreaming出力を行う。
- `open_result`：最終粒子、event、保存したtrajectoryを遅延読込みする。

100万粒子の巨大なin-memory `Result` を標準で返さない。公開`PreparedRun`、公開preflight object、
多数のfacadeは作らない。CLIは同じ入口を使う。

```console
chamber-particles check case.yaml
chamber-particles run case.yaml -o runs/case_001
chamber-particles inspect runs/case_001
```

### 12.2 推奨ディレクトリ

```text
particle-platform/
├─ pyproject.toml
├─ src/chamber_particles/
│  ├─ __init__.py          # load_case / simulate / open_result
│  ├─ api.py               # 薄い公開facade
│  ├─ case.py              # SimulationSpec、SimulationCase
│  ├─ case_format.py       # canonical schema read/write/version
│  ├─ coordinates.py       # XY、RZ、RZ-field/3D、XYZ
│  ├─ geometry.py          # domain、BVH、point/facet query
│  ├─ fields.py            # FieldLayout、space/time sampling
│  ├─ sources.py           # release、weight、schedule
│  ├─ rng.py               # counter keyと乱数変換
│  ├─ physics/
│  │  ├─ catalog.py        # model ID/revision/requirements
│  │  ├─ forces.py
│  │  └─ charge.py
│  ├─ integrators.py       # step strategyとStepProposal
│  ├─ events.py            # earliest hitと残時間work
│  ├─ boundaries.py        # hit後の物理応答だけ
│  ├─ engine.py            # prepareと唯一のlifecycle loop
│  ├─ cpu.py               # SoA/tile/thread kernels
│  ├─ output.py            # sink/checkpoint/ResultView
│  └─ __main__.py          # 公開APIを呼ぶ薄いCLI
├─ tools/
│  ├─ importers/
│  │  ├─ comsol/
│  │  └─ tables/
│  ├─ case_builder/
│  ├─ field_preprocessor/
│  ├─ electrostatic_builder/
│  ├─ analysis/
│  ├─ visualization/
│  └─ vv/comsol/
├─ tests/
│  ├─ verification/
│  ├─ scenarios/
│  └─ performance/
└─ docs/
```

最初から `providers/contracts/preflight/diagnostics/facades` の多階層にしない。一般型置き場の
`models.py`は作らず、型は事実を所有するmoduleへ置く。上のfileが二つの独立した変更理由を持つほど
大きくなったときだけ分割する。詳細なowns/must-not-ownはarchitecture authorityである
[architecture_proposal.md](architecture_proposal.md)を正とする。
[architecture_review.md](architecture_review.md)は、その設計へ反映済みの指摘と判断理由を残すreview recordである。

### 12.3 安定させる拡張点

当面は次の四つだけを拡張点とする。

1. force model
2. charge/internal-state model
3. boundary law
4. field sampler

座標系、integrator、backendは自由pluginにせず、製品の組込み機能として追加する。全てをpluginに
すると対応組合せとテストが急増する。

各physics modelが宣言するのは次だけでよい。

```text
model_id + revision
category + contribution_kind + exclusive_group
required_fields
parameters
applicability summary
supported_coordinates + supported_integrators
compiled evaluator
```

抽象base class、dependency injection container、Python entry pointは、独立した第三者実装が実際に
二つ以上必要になるまで導入しない。

### 12.4 physics contributionの形

高速積分に必要な意味を失わないよう、physicsの出力を次の四種に整理する。

1. `linear_relaxation`：rateとtarget velocity
2. `explicit_acceleration`：その他の決定論加速度
3. `internal_rate`：電荷などの連続時間微分
4. `noise_coefficients`：選択SDE methodが要求する局所係数

全てを不透明な `force(x, v)` callbackにすると線形dragの指数更新ができず、GPU/CPUの最適化も
困難になる。一方、この四種より細かいgeneral contractは作らない。離散chargeは連続ODE/SDEでは
ないため、実装時にだけ第5のjump-process strategyを追加する。

---

## 13. 入力仕様

### 13.1 DataBundleとSimulationSpec

大容量のgeometry/fieldと、sweepごとに変わる物理・時間・出力を分ける。

```text
DataBundle: GeometryDomain、FieldLayout/FieldSet、boundary group、provenance
SimulationSpec: source、particle property、physics、boundary law、time、resource、output
PreparedRun: 上記を解決した非公開・immutableな実行計画
```

同じDataBundleを複数SimulationSpecから参照できる。`load_case`は両者を正規化した
`SimulationCase`を返し、`simulate`開始時にmodel revision、required fields、state layout、sampler、
memory planを一度だけ固定する。`load_case`は先にYAML/resourcesを確定し、HDF5 metadataから
canonical numeric bytesを求めて上限外をpayload展開前に拒否する。adapterがcanonical dataを安全に作るため、
利用者向け三APIとは別に
安定した`case_format.write()`を提供する。

### 13.2 ファイルを増やしすぎない

初期canonical caseは次の二つを標準とする。

```text
case.yaml    # 人が編集するcase spec、model選択、HDF5 path、expected content hash
case.h5      # mesh、field、particle/source table、boundary metadata、data座標表現、単位、provenance
```

CSV、COMSOL export、他solver形式はadapterがこの形式へ変換する。HDF5内のdataset名とshapeは
versioned schemaにするが、独立したschema frameworkや生成コードは作らない。
data座標表現、geometry、field layout、boundary ID、単位、producer provenanceはHDF5だけが所有する。
particle motion modeはYAMLだけが所有する。YAMLはSimulationSpec、HDF5 path、expected content hashを持ち、
data座標表現を二重記載しない。

### 13.3 YAMLの最小構成

```yaml
format_version: 2

case:
  name: chamber_screening_001
  data_path: case.h5
  expected_content_hash: "sha256:..."

time:
  start_s: 0.0
  end_s: 0.03
  dt_s: 1.0e-5

motion:
  mode: axisymmetric_rz_meridional

solver:
  integrator: exponential_midpoint
  backend: cpu
  seed: 1234
  event:
    geometry_rtol: 1.0e-12
    roundoff_ulps: 64
    max_refinements: 48
    max_interactions_per_step: 8
    corner_policy: priority_then_combined_normal_v1

resources:
  memory_limit_mb: 8192

physics:
  drag:
    model: epstein_linear
    revision: epstein_linear_v1
    gas_velocity_field: gas_velocity
    gas_density_field: gas_density
    gas_temperature_field: gas_temperature
    gas_mean_free_path_field: gas_mean_free_path
    gas_molecular_mass_kg: 1.1469e-25
    delta: 1.0
    applicability: error
  charge:
    model: fixed
  electric:
    model: coulomb
    revision: electric_coulomb_v1
    electric_field: electric_field
  gravity_buoyancy:
    model: standard
    revision: gravity_buoyancy_standard_v1
    gas_density_field: gas_density
    gravity_m_s2: [0.0, -9.80665]

sources:
  - name: source_parts_release
    type: surface
    boundary_group: source_parts
    count: 100000
    particle_id_start: 1000000
    particle:
      mass_kg: 3.1101767271e-20
      charge_number: -20
      drag_diameter_m: 3.0e-8
      electrostatic_radius_m: 1.5e-8
      displaced_volume_m3: 1.4137166941e-23
      model_weight: 1.0
      material_id: 0
    position:
      model: uniform
      measure: revolved_area
    velocity:
      model: normal
      direction: into_domain
      speed_m_s: 0.5
    release:
      model: fixed
      time_s: 0.0

boundaries:
  - boundary_group: wafer
    priority: 10
    law: probabilistic_stick
    probability: 0.8
    otherwise:
      law: specular
  - boundary_group: pump
    priority: 20
    law: escape
  - boundary_group: other_walls
    priority: 30
    law: stick

output:
  trajectories:
    selection: sample
    count: 1000
    schedule:
      interval_s: 2.5e-4
```

model固有parameterはそのmodel直下に置く。global parameter table、manifest default、backend defaultへ
同じ値を分散させない。categoryが無ければ無効とし、無効値の`null`は用いない。同じcategoryには
一modelだけとする。`fixed`は`dZ/dt=0`のみを表し、初期`charge_number`はtable/surface sourceが一意に
所有する。無帯電はsourceの値を0とし、blendは明示的なversioned composite modelである。
canonical particleは`mass_kg`、各相当径、`displaced_volume_m3`、`model_weight`を個別の権威値とし、
YAML loaderは密度と径から再構成しない。球形の簡便入力は外部case builderが一度だけ展開する。
`charge_number`はfinite、`mass_kg`、`drag_diameter_m`、`model_weight`は正値、`electrostatic_radius_m`と
`displaced_volume_m3`は、その物理を使わない計算粒子を表せるよう非負値とする。realized tableの
`particle_id`とsurface sourceが`particle_id_start/count`で予約する範囲は全sourceを通して一意な
非負signed-int64整数とし、thread数やtile分割に依存しないRNG・結果identityの権威にする。
tableのrelease timeはfiniteなSI時刻で、計算閉区間`[time.start_s,time.end_s]`内ならよい。時刻原点に
物理的な意味を強制せず、0以上という追加制約を置かない。
SIを標準とし、入力単位がSIでない場合はadapterで一度変換する。通常利用者へtile/chunkを公開せず、
memory limitから内部microtileを計画する。

trajectoryの保存時刻は`output.trajectories.schedule`だけが所有する。`time`は積分区間とstep policyを
所有し、保存間隔を重複して持たない。

### 13.4 入力検査の所有者

source-specific adapterは元形式の単位・列・ID・意味論を検査する。`case_format.read`と`load_case`は
canonical dataの局所schemaとcross-file参照を一度だけ次のように検査する。この二つは同じ条件の
重複検査ではなく、異なる境界の所有者である。

- schema version、dataset存在、shape、dtype、有限性、index範囲とsource identity
- 単位、data座標表現、局所node順、boundary rowと宣言owner edgeの整合
- 粒子の質量、径、発生時刻
- YAMLの期待content hashとHDF5論理content hash
- 全境界へのlaw割当
- field時間範囲とsimulation時間範囲

`simulate`のprepareだけが次を所有する。

- model categoryの重複、required field、parameter適用範囲
- data座標表現・motion mode・integrator・backend・compiled evaluatorの互換性
- resident field、particle state、thread数非依存tile slab、bounded outputを含むmemory/resource budget

`geometry.prepare`はcell incidenceから大域topologyを一度だけ検査し、non-manifold、重複boundary、
boundary vertex異常、自己交差・重なり、内部edgeのwall登録、RZ axis seam以外の外周欠落を拒否する。
boundary rowが0件でもvolume incidence監査を省略しない。CAD/mesh修復やfacet生成は外部builderの責務で、
coreは修復しない。この大域prepare不能は現在`simulate`の`SimulationError`であり、loaderの`CaseError`と
重複検査しない。

同じ条件をadapter、loader、preflight、runtimeで重複検査しない。runtimeでは内部不変条件をassert
するだけにする。入力に問題があればdefault、epsilon、別modelで修復しない。

---

## 14. 出力仕様

### 14.1 必須成果物

v0.1の永続形式はJSONとHDF5だけに絞る。final particle表とevent logは常に保存し、設定で無効化しない。
大規模runを一つのappend fileへ依存させず、閉じたepochをatomic commitする。

P05で導入した最小result pathは、`run.json`、一つのclosed segment、`final.h5`、`_SUCCESS`と最小lazy
`ResultView`を作った。P13はそのlogical event/frame/final意味を変えず、multi-segmentとdurable recoveryへ
physical layoutだけを拡張した。

finalはrun終了時刻のlifecycle snapshotであり、escaped粒子の位置・速度は
`kinematics_valid=0`をlogical nullとする。数値payloadはNaN sentinelではなく有限な最後hit状態を保持するが、
validityが0の時は非権威値であり科学値として使用しない。active/stuck/heldは`kinematics_valid=1`とする。
heldはhit時payloadを保持するだけで、run終端まで物理更新しない。

P13で同じlogical event/frame/final schemaを次のdurable layoutへ拡張した。

```text
run.partial/
  run.json                 # status、input/model/algorithm hash、実行統計
  segments/epoch-000000.h5 # event、frame、online countの確定segment
  checkpoints/A.h5
  checkpoints/B.h5        # 二世代交互
  final.h5                 # 粒子別最終状態
  LATEST                   # 最後にcommit済みのepoch
  _SUCCESS                 # 正常終了時だけ作成
```

segmentを一時名へ書き、close後にrenameし、最後に`LATEST`を更新する。`run.json.status=complete`と
`_SUCCESS`は全成果物の後にcommitし、最後に`OUT.partial`を要求された`OUT`へ同一volume内でrenameする。
通常の`open_result`は未完runを完成結果として開かず、明示した
recovery modeだけが`LATEST`までを読める。Parquet/CSVは外部analysisが必要な場合にexportする。

checkpointはengineがaccepted macro barrierでwork threshold到達、または最終macroを判定した時だけ作る。
`W=macro_step_count+accepted_particle_pieces+candidate_queries+refinements`、`T=max(2^20,128N)`であり、
cadence revision・resolved threshold・components・barrierはmanifestとresume identityへ記録する。このbarrierはtrajectoryの
output scheduleとslab幅から独立し、保存時刻のためにproduction stepを分割しない。
checkpointは時刻、SoA状態、active IDs、pending source cursor、frame/probe cursor、logical/physical event ordinal、
exact origin、surface-contact token、field-cell hint、event aggregateを含む。同一input hash、schema、physics revision、
algorithm/backend revision、座標・method、particle identityでだけresumeを許し、初期版にmigrationを作らない。

P13の一epochのdurable commit順を固定する。

1. segmentを一時fileへ書き、HDF5 flush/close後に確定名へatomic renameする。
2. inactive側checkpointを一時fileへ書き、close後に`A`または`B`へreplaceする。checkpointはsegment
   commit ID、event/frame count、content hashを参照する。
3. `LATEST.tmp`をcloseし、`LATEST`へreplaceする。ここだけをepochのcommit pointとする。
4. crash後は`LATEST`が指すsegment/checkpointだけを採用し、それより新しいorphan fileを無視する。
5. 最終epoch後に`final.h5`、complete `run.json`、`_SUCCESS`の順でatomic commitし、directoryをrenameする。

resumeはcheckpointのcount以降からevent/frameを発行し、`(particle_id,event_ordinal,event_type)`の重複も
欠落も許さない。engineはparallel roundをstable prefixしたevent/failure columnar batchをmain threadの
single-owner `ResultWriter`へ同期的に渡してbackpressureする。最初の`LATEST`前はepoch 0を初期stateから再実行し、final/run.json/`_SUCCESS`/
directory publicationの途中は次回runが検査して完了する。確率wall RNGのphysical ordinalも厳密に復元する。
engineが`W/T`とcommit判断を所有し、`output.py`はsegment→inactive A/B checkpoint→`LATEST`のatomic persistenceを所有する。
上記各境界へのfailure injectionをStage 1Bの受入試験として完了した。

P13の容量1 queueはbounded memoryのため同期ackを待ちcompute/I/Oを重ねなかったので、P14-Pでbackground threadと
queueを削除した。参照checkpointと最新segmentは
hash、全segmentは構造と累積countを検証するが、同shapeの過去segment値改変は検出契約外である。local filesystem上の
process failure回復を対象とし、power loss、remote filesystem、同じOUTへの複数process同時実行は保証外とする。

HDF5 segmentのlogical datasetは次とする。

```text
/events/*                     columnar event log
/frames/time_s
/frames/offset                ragged frame offsets
/frames/particle_id
/frames/position_m, velocity_m_s, charge_number, lifecycle
/series/*                     integer count、wall別weightなど小さな集計
/probes/*                     指定粒子だけのfield/force/stage情報
```

### 14.2 trajectory保存mode

選択と時刻scheduleを直交させる。

- `selection: none | sample | ids | all`
- `schedule: interval_s | explicit_times_s`

P05は最小subsetとして`selection: all`と`schedule.explicit_times_s`だけを受理する。時刻はfinite、
狭義単調増加、重複なし、計算閉区間内とする。release前の粒子はframeに含めず、release時刻と一致する
frameには初期状態を保存する。final行は`particle_id`順とする。その他のselection/scheduleは所有packageで
実装するまでprepareで拒否し、黙って無視しない。
frameはboundary eventに対し右連続とし、hit時刻ちょうどのstickはpost-event状態、holdはhit payloadを保持した
held状態、escapeは行なしとする。heldはinactiveのまま以後のframeへ残し、escaped粒子は以後のframeから除外する。

全粒子×全内部stepを既定で保持しない。付着率や人口履歴などはonline集計し、必要に応じて別の
analysis toolで再計算できる原始eventを残す。frameはaccepted pathから評価し、保存mode・間隔・probe
ID数を変えてもfinal state、event sequence、RNG pathが変わらない。現行writerはmain threadの同期single-ownerで
backpressureし、遅いdiskのためにeventを捨てない。P13の容量1 queueは履歴実装である。

### 14.3 最小run diagnostics

- pending/active/stuck/held/escaped/failed数
- accepted step、substep、boundary event数
- field sampling、physics、event、I/Oの経過時間
- peak resident memory
- applicability範囲外の粒子割合と対象model
- failure reason別件数

常時全粒子の各力、各RK stage、局在反復を保存しない。必要なparticle IDを指定したprobe出力だけが
詳細値を持つ。

### 14.4 後処理と可視化

`tools/analysis`は`open_result`のlazy scanだけを使い、fate統計、source-to-target matrix、重み付き
deposition map、到達時間、impact energy/angle、residence、ensemble信頼区間を計算する。
`tools/visualization`は`ResultView`またはanalysis tableだけを描画し、物理式を再計算しない。
COMSOL差分plotは一般可視化へ混ぜず`tools/vv/comsol`に置く。

---

## 15. 高速化方針

### 15.1 CPUを最初の製品backendにする

最初のbackendはNumPy＋Numbaによるfloat64 CPUとする。

- Structure of Arrays（位置、速度、電荷、状態を別の連続配列）
- 物理SoAのID対応は固定し、固定容量の`active_particle_index`だけをin-place stable compact
- v0.1は全粒子stateをresidentに置き、memory planからsingle-thread bounded tile slabを内部決定
- particle loopはNumba kernel内
- field、force、integratorで一時配列を再利用
- P1/Q1だけにaccepted endpointのprevious-cell IDを保持し、strict-interior fast pathで消費
- 境界はstackless BVH broad phase
- model dispatchは粒子loop外
- production computeはsingle-thread compiled passに限定
- field/geometryはread-only共有、eventと集計は固定columnar rowへ書いてstable prefix

force追加のたびに巨大な融合kernelを生成する仕組みから始めない。最初はfieldを一度sampleし、
選択した数個のcompiled evaluatorが同じacceleration bufferへ加算する。profileで配列passが支配的と
確認できた組合せだけを後から融合する。

specular反射や狭いgapではevent queueを小さいと仮定しない。hit後もactiveな粒子はrow target time付きflat SoA
work queueの次roundで残時間を処理し、event出力columnと分ける。共有listへatomic append
せず、stable prefix/compaction後のcolumnar batchをwriterへ渡す。boundary eventのidentityは
`(particle_id,event_ordinal)`、公開時のcanonical順は`(time_s,particle_id,event_ordinal)`とする。
deterministic CPUではfastmathを標準無効、force加算順と
reduction順を固定し、slab幅、出力schedule、checkpoint/resumeで粒子別event/RNGが変わらないようにする。

P09はNumPy reference engineのまま、`cpu.py`にresident layout、active index、bounded microtile、
solver-owned memory planを集約した。memory limitはload/prepare/runの予測solver-owned peakを制限するが、
OS hard RSS capではない。fresh/warm RSSは外部performance scriptが別に測る。現行regular locatorで使わない
per-layout hintは先行配列化せず、P10のcompiled samplerと同時に、実際に消費するP1/Q1
strict-interior hintだけを追加した。

P10は`coupled_rk4_engine_v16` / `compiled_cpu_tile_v1`として、field sampling、sample済みprimitiveの
physics、classical RK4算術をNumba 0.67のcompiled array passへ置換した。NumPyは`<2.6`とし、
`fastmath=False, parallel=False`を固定する。P1/Q1だけに実際に消費するresident hintを追加し、trialではなく
accepted endpointだけをcommitする。`state_at()`、wall hit時刻の再積分、residual piece、output scheduleは
同じproduction proposal/event経路を使う。CPU runtime layoutとmemory planはv2、physics runtimeはv2へ上げたが、
case/result schema、proposal、event、field semantics、physics model revisionは変更していない。

P11は`deterministic_particle_engine_v17` / `compiled_cpu_tile_v2` / proposal v4 / event v9 /
physics runtime v3として、同じcompiled physics passから線形drag rate、target velocity、加算加速度を返し、
start half-step predictorで得たmidpoint係数による`exponential_midpoint_v1`を一つの`StepProposal`へ追加した。
材料first hit、wall residual、RZ axis、`state_at()`、output scheduleはRK4と同じevent/commit経路を使う。
曲線/chord偏差は全短縮secantを含むvelocity enclosureから
`h * (v_upper - v_lower) + roundoff`で外向きに作り、position box全幅による過剰分割を避ける。
RK4の`dt/tau < 2.5`gateは`rk4_fixed`だけへ適用し、指数法へは適用しない。P11 closeout時点の指数法はfixed chargeに限定し、
非零charge rateを無視、後付けsubcycle、RK4 fallbackのいずれにも読み替えない。case/result schema、physics
catalog/model revision、field semantics、resident/output layoutは変更していない。

P12は`deterministic_particle_engine_v18` / `compiled_cpu_tile_v3` / `line_boundary_bvh_v3` / CPU runtime layout v3 /
memory plan v3として、一粒子一workerの非重複tile ownershipを同じengineへ追加した。field/geometryは
read-only共有、residual/event/failure/statisticsはworker-localとし、最大`W`個のin-flight tile waveを
main threadがtile順にstable mergeする。writerを呼ぶのもmain threadだけである。compiled BVH query、保守的な
RK4 clear/split事前認証、同時刻wall prefix batchは既存proposal/event意味論を変えないhot-path最適化である。
threads 1/2/4で科学出力はbitwise一致する。512粒子×4 macro stepのwarm実測では、
1/2/4 threadが0.440460/0.528229/0.632361 sで、編集前serial baselineに対する1 threadの改善は6.96倍だが、
正のthread scalingはまだ得られていない。

P13は`deterministic_particle_engine_v19` / `durable_segmented_result_v3` / checkpoint schema 1 /
memory plan v4として、固定64 macro-step epoch、worker-wave stream、容量1 writer queue、A/B checkpoint、`LATEST`、
auto-resume、明示recoveryを追加した。case/result schema 1、compiled tile v3、runtime layout v3、geometry v3、
proposal v4、event v9、physics catalog/runtimeは変更していない。通常実行とresumeの全公開payload/科学manifestを
raw identityで一致させ、segment/checkpoint/LATEST境界のfailure injection、orphan無視、参照artifact破損拒否を
検証した。最初の`LATEST`前、最終公開の全境界、確率wall RNG ordinalを含むverification/scenario 322件が合格した。
製品規模thread scalingとdurable I/O throughputはP14で判断した。現行milestoneとalgorithm revisionは
`implementation_plan.md`と実行manifestが所有する。本仕様はsingle-thread compiled engine、bounded memory、
YAML case schema v2、canonical HDF5 data schema v1、result/checkpoint schema v2という製品境界だけを所有する。

P14の結果ではouter ThreadPoolがregular 1M以外へ十分な効果を示さず、event-heavyでは遅化し、worker数に比例する
scratchも増えた。この方式は履歴としてP12/P14の記録に残すが、製品runtimeとして維持しない。P14-Pでは
`ThreadPoolExecutor`を廃止し、thread数非依存slab、stackless boundary BVH、row target time付きflat SoA event
wavefront、compiled boundary/RNG、stable columnar outputへ収束させた上でNumba内部parallelを評価した。
regular 1Mの4-thread speedupが0.923xで事前gateに届かなかったため、`resources.threads`、thread mask、並列専用
code/testを削除し、compiled single-thread engineを唯一のproduction経路とした。詳細な測定と採否条件は
[`solver/docs/parallel_execution_plan.md`](solver/docs/parallel_execution_plan.md)を権威とする。

現行v36はouter pool、future wave、worker別scratch、内部thread teamを持たず、thread非依存slab、
field/physics/integrator workspace、stackless boundary BVH、同期single-owner writerへ統一している。
memory plan v11はdeferred event depthを `event_work_bytes_per_particle = 24 * (max_refinements + 1)` として
`slab_event_work`へ独立計上し、depth依存容量を一般proposal scratchへ隠さない。
候補、event/failure staging、surface release、direct replayをnamed componentへ分離し、pack時だけのgatherは12.5%
safety marginが所有する。正確なbyte式は
[`solver/docs/parallel_execution_plan.md`](solver/docs/parallel_execution_plan.md)が所有する。
linear/quadratic exactと一般曲線eventのflat SoA wavefront、wall/axis locator、boundary/Philox、row numerical status、
batch surface release、direct replay、bounded event/failure stagingはengine接続済みである。直列smoke matrixは
regular/P1/Q1、cross-cell、20-hit event、cold/warm、none/all outputを検査する。代表用途の時間・mesh収束と
直列性能判断はP14-Uが完了し、配布可能性はP14-Rが所有する。
v20のmachine-readable exact payload digest snapshotは保存されていないため、P14の文書化値と当時の336件を移行根拠、
v27全公開payloadをP14-P integration anchorとする。証拠のないv20 bitwise同一は主張しない。

### 15.2 GPU/JAXの位置づけ

GPUは、次の条件がCPU版で確立した後に追加する。

- scalar/referenceとCPU batchの軌道・event意味論が一致
- regular/static fieldで十分な粒子数があり、転送費を上回る
- dynamicな多重壁hitをbounded loopで表現できる
- float64が必要なcaseとfloat32が許されるcaseを実測で分離

JAXのJIT、X64、乱数、制御flowは有用だが、最初のarchitectureをJAX制約へ合わせない。
GPUは別solverではなく、同じcanonical caseとresultを処理するbackendとする。

### 15.3 メモリ

100万粒子で、位置・速度6成分をfloat64で保持しても48 MBである。電荷、粒子属性、ID、status、
cell IDを含むresident state、field snapshot、thread数非依存tile slab、bounded output bufferをprepare時に
見積もる。一時field、RK stage、加速度を全N分確保せずtile分だけ保持する。一方、100万粒子×1000時刻の全軌道は
数十GB以上になるため、問題は状態配列よりtrajectory保存である。

性能評価ではkernel時間だけでなく、field sampling、壁、active compaction、serialization、disk I/Oを
分けて測る。10^4/10^5/10^6粒子、event-light/event-heavy、mesh-native/cache、output none/sample/all、
cold/warmについてwall time、processed step/event、peak RSS、bytes/particle、writer throughputを記録する。
独立caseのprocess並列はsolver外で別に測る。絶対目標値は実測baselineなしに捏造しない。

---

## 16. 持続的な運用と複雑化防止

### 16.1 一つの事実に一人の所有者

- HDF5 canonical schema、座標系、単位、provenance、算出content hash：`case_format`
- YAML case spec、HDF5 path、expected content hash：`case`
- 座標基底・軸規則：`coordinates`
- particle domain、topologyと交差：`geometry`
- field補間：`fields`
- 物理式：該当physics module
- step更新：`integrators`
- event局在と残時間：`events/engine`
- hit後の物理応答：`boundaries`
- lifecycle：`engine`
- file形式、checkpoint、lazy read：`output`

同じinvariantを複数層で契約化しない。

### 16.2 complexity budget

機械的な行数gateではなくreview原則とする。

- v0.1のcoreは約10～15実質module
- 公開操作は3つ
- 初期integratorは2つ
- 粒子statusは5つ
- 必須result logical familyは`final/events/series`の3つ（frames/probesは任意）
- 新抽象は二つ目の実例が生じるまで作らない
- production hot loopにPython callbackを入れない
- 新しい実行経路を追加する場合、置換される経路を同じreleaseで削除する
- 新しいforceのためにsolver、field reader、boundary、writerを変更するなら設計を見直す
- facade-only file、private helper配置test、文書存在testを作らない
- 診断は指定粒子と小さなrun summaryに限定する

Python環境と静的品質gateは [quality_tooling_plan.md](quality_tooling_plan.md) を権威とする。uvを唯一の
環境・lock・command基盤とし、Ruff、import-linter、Pyrefly、Radonをそれぞれformat/lint、依存方向、型、
関数complexityへ一対一で割り当てる。同じ指標のtoolや独自runner、clean-room codeに対するbaselineを
重ねない。

### 16.3 物理モデル追加手順

新モデルに必要なのは原則次だけである。

1. 文献、式、単位、適用域を物理カタログへ追記
2. required fieldとparameterを宣言
3. reference evaluatorを実装
4. CPU batch evaluatorを実装
5. registryへmodel IDを追加
6. 既知極限または解析解のverificationを追加
7. 一つの代表trajectoryと性能を確認

大量のmock、facade、schema class、モデル専用診断reportは要求しない。

### 16.4 versioning

- case schema、physics model、solver algorithmは別々のrevisionを持つ。
- modelの式を変えた場合は同じ名前の挙動を黙って変えずrevisionを上げる。
- run.jsonにcode commit、model revision、input hash、seedを保存する。
- 数値結果を変えないrefactorではphysics revisionを上げない。
- 旧case migrationはv1以降に、実際の利用caseが生じてから導入する。

---

## 17. 検証とテスト

### 17.1 core testは三層だけ

#### Verification

小さく高速な数式・数値試験。

Stage 0～1Aで固定するcore microcase packは次の`C01`～`C10`の10件とする。`C01`～`C05`は
解析軌道、残りは補間と境界eventを一因ずつ固定する。exact入力・期待値・有効化packageは
`solver/docs/numerics.md`を権威とする。

| ID | 内容 | 主に固定するもの |
|---|---|---|
| C01 | force無しballistic | `x=x0+v0t`、frame補間 |
| C02 | uniform linear drag | exponential velocity/position、RK4収束 |
| C03 | linear drag＋constant acceleration | exponential midpoint式 |
| C04 | uniform electric field＋fixed charge | charge sign、mass authority |
| C05 | gravity＋buoyancy | displaced volumeとmassの分離 |
| C06 | two-cell affine P1 / mapped Q1 field | locate、basis、shared-edge連続性、vector component |
| C07 | plane wallへのfirst hit | event time/point/normal |
| C08 | wall上release | zero-time departureと再衝突 |
| C09 | thin gapと複数反射 | residual time、boundary event ordinal、interaction budget failure |
| C10 | corner同時hit | candidate facet set、effective response normal、priority、曖昧policy failure |

現行`particle_engine_v36`ではC04/C05が証明済み`quadratic_exact`経路、C02/C03が一般
`rk4_reintegrated`経路の公開API scenarioとして有効である。後者はfully-supported common
`RegularLayout`、fixed charge、既存のEpstein/electric/gravity、連続support/applicability enclosureを満たす
XY/RZ caseを扱う。revision 3b/P06-RZでは同じsubsetをtopology-completeな材料boundaryと連成し、terminal
stick/escapeまで実行できる。P06-Uでは材料domainと完全一致するfully-supported P1/Q1も一般RK4へ追加したが、
boundaryless unstructuredは連続包含証明がないため拒否する。P15の動的電荷は同じXY/RZの
`rk4_reintegrated`と`exponential_midpoint_reintegrated`へ接続し、exact pathを無効化する。
P07 exact-path sliceではC08～C10も公開API scenarioとして有効であり、surface departure、静止壁の
specular/probabilistic stick、priority/combined-normal corner、複数hitとcap-driven residual split、ballistic RZ
axis foldを検査する。engine v10はCartesian XYの証明済み一定加速度surfaceも公開scenarioで検査する。
engine v11 / event v7で追加したCartesian XY一般RK4の厳密内向きsurface departure、single-facet active-boundary
residual、右連続state jump、次hit確認後のcap splitを公開scenarioで検査する。tangentまたはfacet端点/cornerからの
一般RK4 departureはfail-closedである。
engine v12 / event v8はEpstein RZ axis crossingの4次収束、frame schedule identity、away-axis Cartesian退化一致、
axis→wall順序、軸上不変状態、axis regularity failureを公開API/verificationで検査する。
P11はC03の`exponential_midpoint_reintegrated`を一定係数の全要求frameで丸め誤差精度、広い`h/tau`でfinite、
smooth可変係数で次数1.8以上として検査する。Stokes--Cunningham一定primitiveの閉形式、output schedule identity、
Cartesian材料event/residual、surface departure後の同面再衝突、RZ axis→wallも同じproposal/event経路の公開scenarioで検査する。

後続stageの3D、時間依存のverificationは、対応機能を実装するstageで追加する。chargeと限定OUは実装済みで、これらと
外部V&VのIDを`C01`～`C10`へ混在させない。

- ballistic、一定加速度
- linear dragの解析解
- uniform electric field
- fixed/continuous chargeの既知平衡
- OU過程の平均・分散
- P1/Q1補間とmanufactured field
- 平面壁のhit時刻、specular、restitution、確率付着と明示fallback
- axisymmetric surfaceの \(2\pi r\) sampling
- step・mesh半減による収束

#### Scenarios

公開APIを通る10～20個程度の小型case。

- surface release → stick/escape/specular/restitution
- 薄い障害物、複数hit、corner、軸通過
- field support外
- 複数径、複数発生時刻
- 電荷と電気力の連成
- axisymmetric field中3D Brownian

#### Performance

nightlyまたはreleaseで実施する。

- 10^4、10^5、10^6粒子
- eventが少ないcaseと多いcase
- mesh-nativeとregular cache。unstructuredはrealistic cell count、initial localization、cross-cell motionを分離
- trajectory off/sample/all
- cold JIT、warm run、最大memory、particles/s

P10の手動characterizationはfield-heavy/event-lightと小さいevent-heavyの二caseだけでcold/warm JIT、
semantic digest、RSS、revision、speedupを確認する。絶対thresholdは置かず、上記のsynthetic matrix判断は
P14が所有し、23行×3観測matrixで完了した。target-useの結合判断はP14-Uが所有し、18 raw観測＋6 medianと
別1M profileで完了した。read-only synthetic local profileではP1 stripの1000 warm sampleについて、hintなしfull searchが
100/500/1000/5000 cellで0.026/0.131/0.249/1.271 s、正しいstrict-interior hintが約0.0005 s
（51x～2576x）だった。これはP10のcompiled baselineとhint効果を示す非gating観測であり、large-mesh P1/Q1の
製品性能完了を意味しなかった。P14で支配的と確認し、field-owned BVHでsupported containmentだけをindex化した。
outside/masked provisionalはO(cell数) full scanを維持し、T04 cache/remeshは後続profile条件付きとする。
同じmatrixの3観測medianでregular 100k/1Mは20 workerが1 worker比1.8796x/4.7965xだったが、
event 10k×20 hitは0.8882xだった。P14-Pの内部parallel試行もregular 1Mで0.923xだったため、現行v36は
single-thread compiled runtimeだけを提供する。artifact byte rateは
public workflow全体のeffective rateで、writer単体帯域ではない。table startのvolume全走査もgeometry v4のBVHへ
置換し、局所的にfloat64で解像不能なcellはprepareで拒否する。memory plan v6は両index residentとfield
256 B/cell、geometry 1,024 B/cellのbuild transientを含む。絶対秒数とCOMSOL速度比較は受入条件にしない。

量ごとの誤差は原則

\[
E_q=\frac{|q-q_{ref}|}{atol_q+rtol_q\max(|q|,|q_{ref}|)}
\]

で評価し、位置、速度、電荷を一つの曖昧なscalarへ混ぜない。最小受入表を次へ固定する。

| 対象 | 最小受入基準 |
|---|---|
| RK4 | smooth coupled `x,v,Z`で観測次数3.5以上 |
| exponential midpoint | 可変場で1.8以上、一定係数で丸め誤差程度、広い`h/tau`で有限 |
| P1/Q1/tet4 | 解析fieldを再現し、隣接cellの共有境界で値・supportが一致し、owner選択が決定論的 |
| first-hit | 時刻bracket・facet距離がbudget内、step/mesh半減でtime/point/facet集合が収束 |
| specular | parameterなしで相対法線速度が厳密に反転し、接線速度が不変 |
| restitution | 相対法線速度が`-e_n`倍、接線速度が`e_t`倍 |
| surface source | RZ sample密度が`2πr ds`に一致 |
| continuous charge | 平衡、残差、coupled軌道が設定次数で収束 |
| axisymmetric 3D | 方位角回転に対して軌道・eventが共変 |
| OU | B01/B02のjoint平均・共分散・MSD・平面first-passageが統計区間内。B03は定数係数mean/full covariance exact、noise-off 2次、manufactured weak mean観測次数`>=0.9`。一般SDEのstrong/weak 2次は主張しない |
| time field | snapshot半減で所定次数、discontinuityを跨がない |
| cache | field/gradient budget内、代表caseのboundary outcome不変 |
| CPU parallel | thread/slab変更でID、event、RNG identityが不変 |
| output | 保存selection/scheduleを変えても計算結果が不変 |

### 17.2 テストしないもの

- private関数名
- helperの所属file
- directory treeそのもの
- 文書fileの存在数
- 現在の内部call sequence
- COMSOL固有列名とaliasをcore suiteで大量固定すること

### 17.3 外部V&V

COMSOL比較、`model_dataset` 回帰、可視レポートは `tools/vv/comsol` のsuiteとする。core releaseの
参考資料にはなるが、solver packageへCOMSOL dependencyを持ち込まない。
外部V&Vのmicrocaseは`V01`～`V06`とし、core microcaseの`C01`～`C10`と別namespaceにする。

| ID | 外部V&V対象 |
|---|---|
| V01 | field補間・support・要素検索 |
| V02 | 運動・時間積分 |
| V03 | surface release |
| V04 | wall event |
| V05 | charge coupling |
| V06 | Brownian OU統計 |

各caseの設定、比較手順、合否は[外部V&V方法論](vv_methodology.md)が所有し、本書では重複定義しない。

---

## 18. `model_dataset` の扱いと再作成方針

### 18.1 現在利用できるもの

- 12ケースの保存済み時間軌道、速度、電荷、力成分、状態
- Case P：詳細プラズマ場を外部入力する利用形態
- Case A：縮約Poisson問題で場を作る利用形態
- geometry、boundary ID、P1/Q1 mesh候補
- Epstein、電気、熱泳動、DEP、連続帯電、ion dragの式候補
- 装置・粒径・場の現実的なscale

### 18.2 製品仕様へ直接採用しないもの

- 30/100 nmでも別粒径のまま残った粒子依存派生場
- CSV列順を暗黙のquad周回順とみなす処理
- `inside_model_domain`を唯一のsupport authorityとすること
- no-swirl運動に使わない方位Brownian成分を運動合力へ含めること
- 単一seedのBrownian軌道をpathwiseな正解とすること
- 表面放出、反射、確率付着を含まないケースで、それらの製品品質を主張すること
- COMSOL内の数値floor、clip、固定RK4をnative製品仕様にすること

### 18.3 外部ツールとして再実行・再抽出するもの

現在のdatasetは修復禁止の固定資料ではない。次を新しいversionとして再作成する。

1. 粒子非依存のprimitive fieldだけをmesh-nativeに出力
2. node/element ID、要素type、次数、局所node順、owner domain/facet
3. 場の単位、式、solution、時刻、平滑化・recovery設定
4. Brownian-offの決定論case
5. 10 µs以下のmicro-traceと、必要ならRK4各stage
6. 正確なevent時刻、位置、boundary ID、前後状態
7. 真のsurface release
8. stick、escape、specular、確率stickを一つずつ分離したmicrocase
9. 同一点・同条件の多数Brownian replicaと複数seed
10. 電荷電流、各force、合力を指定query点で出すoracle table

既存datasetを上書きせず、version、MPH hash、exporter hash、COMSOL versionを変えた新bundleにする。
製品実装は解析解microcaseを主軸に進め、COMSOL再抽出の完了を待ってcore設計を曲げない。

---

## 19. 二つのプラズマ場利用mode

### 19.1 External field mode

外部のplasma/CFD/thermal計算から、電場、電子・イオン状態、流れ、温度をCanonical Fieldへ変換する。
solverは場の生成方法を知らない。

### 19.2 Reduced electrostatic builder mode

熱流体場と利用者指定のbulk plasma parameterから、独立した
`tools/electrostatic_builder` がPoisson問題を解く。

```text
thermal/fluid field + plasma parameters + electrostatic BC
                        ↓
        nonlinear Poisson / sheath approximation
                        ↓
      V, E, ne, ni, auxiliary plasma fields
                        ↓
               Canonical Field
```

Case AのSASS型モデルは最初の候補であるが、唯一の内部場モデルにはしない。builderごとに方程式、
境界条件、continuation、適用域、solver residualを保存する。粒子simulationとは別run IDを持つ。

field builderのテストはPoisson解析解、電荷保存、mesh収束で行い、粒子solverのtest suiteへ混ぜない。

---

## 20. 段階的な実装計画

### Stage 0：仕様と小型正解を固定

- 本書と物理モデルカタログをreview
- canonical YAML/HDF5の最小schema
- `C01`～`C10`のcore microcase packとanalytic expected data
- `model_dataset` 再抽出仕様
- v0.1に入れる物理の適用範囲表

この段階では既存solverを移植しない。

### Stage 1A：決定論RZ vertical slice

- `cartesian_xy`（解析microcaseと単純geometry）
- `axisymmetric_rz_meridional`
- P1 triangle/Q1 quadの静的場
- table sourceとsurface source
- fixed charge
- Epstein、Stokes–Cunningham、electric、gravity
- RK4 fixed
- stick、escape、specular、probabilistic stick
- scalar verification oracleと公開API scenario

合格基準は解析解、収束、first-hit、表面発生であり、COMSOL一致ではない。

### Stage 1B：高速決定論製品経路

- exponential midpoint fixed-step
- CPU/Numba SoA batch（P10完了）、event-heavy residual workとstable parallel merge（P12完了）
- segmented HDF5、checkpoint/resume、bounded writer queue、lazy ResultView（P13完了）
- memory planner（P09完了）、10^4～10^6粒子benchmark

合格基準は主要物理がcompiled経路を通り、slab/output変更で数値意味論が変わらず、
予測memory、実測RSS、compute/I/O scalingが説明できることとする。P14は直交synthetic matrixを完了したが、
主用途の結合性能を意味しない。

### Stage 1P：parallel runtime convergence

- outer ThreadPoolをNumba内部thread team一つへ置換
- thread数非依存tile slab、preallocated `into` pass、stackless boundary BVH
- row target time付きflat SoA event/residual wavefront、compiled boundary/RNG、stable prefix output
- regular 1Mの1/2/4 thread、single-thread回帰、microkernel/profileを先行gateで評価し、未達時の削除条件を適用

v26でworker wave、future merge、worker別scratch、object event/replayを削除し、内部thread team、slab、
stackless BVH、同期writer、exact/curved wavefront、row status、batch release、direct replay、bounded stagingまで
統合した。しかしregular 1Mでも4-thread speedupが0.923xだったため、製品gate未達としてcase schema v2から
thread設定と内部thread teamを削除した。現行v36は`fastmath=False, parallel=False`のsingle-thread compiled
production engine一つである。第二schedulerやexperimental flagは残さない。判断根拠は
[`solver/docs/parallel_execution_plan.md`](solver/docs/parallel_execution_plan.md)を権威とする。

### Stage 1U：代表用途gate（完了）

- surface release＋非一様場＋材料wall＋多数macro stepを一つのcaseで実行
- `h,h/2,h/4`とregular/P1/Q1 mesh系列の軌道・hit収束
- global enclosure、event refinement/failure、serial throughput、memory/outputの同時評価
- P14-P後の単一runtime上でlocal bound、限定的step controlの要否を判定。P14-Uの実測sourceはXY
  `line_length`の`edge_fraction`/`uniform`に限定し、RZ `revolved_area`の分布品質まで検証済みとしない。
  このgateだけを理由に`realized_surface_table`は追加しない

これはCOMSOL一致を正解にするgateではなく、現行数値coreが目的用途を精度・速度・入力表現の三面で解けるかを
判定するgateである。常設diagnostic frameworkや第二engineは作らない。正式releaseはXY時間/mesh収束、RZ収束/parity、
失敗0、出力utilityとidentityを満たした。profile費用はevents 28.7%、fields 26.8%ほかへ分散し、単一owner支配を
示さないため現行productionはengine v36のsingle-thread compiled経路一つを維持する。秒数はmachine-localな
非gating値であり、COMSOL比やportable性能ではない。T03とP14-Rのlocal gate/evidenceに加え、receipt固定の
remote Windows/Linux workflowも完了し、`0.1.0.dev0`開発baselineの配布可能性closureを満たした。これは正式版packageの
公開を意味しない。P15着手をそれまで禁止していた順序はユーザーの明示指示で解除した。

### Stage 2A：帯電・熱・決定論3D粒子

Stage 2Aは一括変更にせず、次の独立した縦切りで進める。

1. P15 continuous charge（完了）：`oml_stationary_maxwellian_debye_huckel_v1`、RK4-first reference、finite
   invariant/rate/derivative bound、charge-aware enclosure、event/output/checkpoint/compiled parityを受け入れ、
   その後native explicit midpoint chargeも第二sliceとして受け入れた。
2. P15-D relative-drift charge（完了）：単一正イオン種のshifted-Maxwellian OML、非正電位invariant、
   明示drift envelopeをP15と同じ両積分器・壁・XY/RZ・checkpoint経路へ接続した。
3. P15-E finite-speed Epstein drag（完了）：Maxwell鏡面/等温拡散混合を持つ一つのversioned
   free-molecular model。Kn・宣言速度比の連続適用域、低速級数、rate/Jacobian別bound、独立分子速度積分、
   両積分器の収束を受け入れた。linear modelへの自動切替や経験的blendはない。
4. P15-F collisionless Barnes ion drag（完了）：単一正イオン、linear two-species Debye screening、
   Debye--Hückel表面電位、collection＋orbital項を相対流方向へ加える。弱結合・collisionless適用域、独立
   impact-parameter積分、両積分器の収束を受け入れ、floor・image補正・電場方向化を移植しない。
5. P16 Waldmann--Gallis thermophoresis（完了）：単一気体の局所並進熱流束をprimitiveとする自由分子
   revisionをXY/RZの既存stage passへ接続した。Kn・相対driftの連続適用域、独立運動論moment、global bound、
   両積分器の収束、XY/RZ parityを受け入れ、温度gradient回復やTalbotへの自動切替は行わない。
6. P17 `axisymmetric_field_cartesian3d`：XYZ state、RZ field mapping、回転面first-hit、3-D normal、
   schema/output/memory更新。

全状態bounded dyadicはP15の剛性・誤差測定で必要性が確認された場合だけ追加する。
P15出口のM3-V外部tool評価は完了した。保存primitiveからreference charge rateとEpstein forceを式精度で再構成したが、
stationary OMLはion drift、Case-P species、scalar ion-mass前提、linear Epsteinは一部状態の速度適用域を満たさない。このため元の12 packageの全軌道一致は
`NOT_APPLICABLE`であり、production閾値は変更しない。後続の別成果物では、Case-A 100 nmの共通exact-P1場・共通3力・
固定電荷・pre-event範囲に限り、独立自己収束と事前登録幅で時間離散parityにPASSした。これをnative-field、boundary、
追加物理、Brownianへ拡張しない。`model_dataset`は優先順位とprovenanceの診断材料であり、coreの
golden truthにはしない。

### Stage 2B：確率過程

- Brownian数値基盤B01とproduction縦切りB02は完了した。P17のstate-dimension変更とは分離する。
- inertial Langevin/OU joint update、counter RNG、joint conditional splitを固定depth treeとしてproduction接続済み
- Cartesian XY・Epstein linear drag-only・fixed-charge state・terminal stick/escapeに限定し、cubic Hermite leafを
  既存event/replay/checkpointへ接続した
- 一般wallは有限depth numerical pathのfirst-passage分布収束で規定し、RMS距離をclear certificateにしない
- ensemble validationとrow-local float64 failureを受入済み。決定論exact-P1 trajectoryの外部M3-Vは完了したが、BrownianのCOMSOL照合はmulti-seed外部gateとして残す

### 独立field-production track

- imported plasma field adapterと同じcanonical field writerを使う
- F01でCase-A相当のversioned reduced electrostatic builderをfirst-party componentとして実装済み
- geometry semantic boundary group、plasma parameter、closure revision、nonlinear residualをprovenanceへ保存する
- 粒子engineはどちらのfield-production modeで生成したかによる分岐を持たない
- F01の受入範囲はstatic RZ、triangle-only P1、単一準中性bulk、C2 Boltzmann--Bohm closure、
  `2 pi r` FEM、damped Newton/matrix-free GMRES、canonical potential/E/density/temperature/ion-velocity出力
- F02でprovider adapterがmixed triangle/quadを品質基準付きでP1化し、代表規模のlinear solve、
  fixed-charge/electric統合、外部field V&Vまで完了した。builder/runtimeへmixed-element分岐は追加していない
- F02の外部比較は`2 pi r`軸対称lumped-volume重みを使う同一export node上の記述比較であり、独立mesh convergence、
  COMSOL trajectory、wall/Freeze parityは未検証のまま明示する

### Stage 3：追加プラズマ力と外部V&V

Stage 3はCOMSOL modeを追加する段階ではない。現在の12 packageで実際に選択された寄与を、producer非依存の
optional revisionとして設定可能にし、外部toolで寄与別・軌道別に評価できるところまでを対象とする。

1. M3-C0：MPH copyから式、parameter、primitive/derived field意味、Brownian-off刻み系列、正例Freeze/Disappear、
   multi-seed入力を再出力し、比較authorityを固定する。coreは変更しない。
2. P18-Cのaggregate two-current charge、P18-Iの二つのion-drag sensitivity、P18-Dの球形準静的DEP、
   P18-LのRZ rarefied-vorticity lift sensitivity、P18-Rの二つのeffective-gas sensitivity、P19-Lの局所continuous-path
   applicability certificate、P18-Hのgeneric terminal `hold/held`、B03のRZ projected Brownian compositionは完了した。P19-L完了時点ではfixed-step endpointとglobal support/event enclosureを
   変えなかった。その後のevent v14は、global enclosureのsupport/applicability/reintegration authorityを維持し、
   `rk4_dense`のevent BVH queryだけをcurrent dense Bernstein boundへ限定した。
3. P18-R audit：native linear Epstein replayは約`1.1e-15`でPASS、既存P15-E/P16のphysical applicabilityは
   12/12 `NOT_APPLICABLE`、PPR `q_eff`欠損によりthermophoresis pointwise replayは`NOT_TESTED`。保存frameは
   continuous pathを認証せず、COMSOL studyは再実行していない。
4. M3-C1の最初のslice：Case-A 100 nmのpre-event frozen saved-state producer-form replay 8/8を閉じた。P19-L後の
   exported-P1 candidateとnative-field referenceをprovenance固定して比較し、cross-representation 6 gateは全FAILした。
   別判定のfull-physics common-P1診断は事前登録9 gateを全PASSし、この限定sliceのsame-field agreementだけを閉じた。
   二つを同じPASSへまとめず、native-field等価性または物理妥当性へ昇格しない。
5. M3-C1のmaterial-event sliceも完了した。Case-A 100 nm、common canonical exact-connectivity P1、Brownian off、
   287粒子、first wafer stickまでに限定し、event evaluation v5はmaterial 20/20とpre-event prefix 9/9をPASSした。
   v14でsolver-onlyの`h, h/2, h/4`を再実行し、3量とも`ORDER_EVALUATED`で自己収束をPASSした。
6. P18-Hは解析的直線・曲線hit、右連続frame、resume、slab identity、OU/Brownianとzero-time surface、
   analysis非deposition分類を閉じた。既存hash固定Freeze referenceへの外部candidateはCOMSOL再実行なしで15/15 PASSした。
   B03 core、charge-stable coupling、work-scaled cadence、P20 performance closeoutも閉じた。meaning-matched external V&V/M3-C2は
   common-P1 Case-A/Case-P 100 nm finalまで完了した。Case-Pは20 us、32+32独立seed、各287粒子×121 frame / 30 msで、
   登録済みR-Z/fate gateをPASSした。これは元COMSOL `auxq`どおりの二電流same-form結果で、後続three-currentや
   species-resolved物理を認定しない。optional three-current productionはP21 priority 1で別revisionとして完了した。
   M3-C1 compact evidence/evaluatorはevent v14の履歴として固定する。M3-C2は単一seed pathを合否に使わない。

P18-R成果物自体はCOMSOL再実行なしの保存artifact auditである。その後、外部M3-C0bはCase-A 100 nmの
0--450 usを段階評価した。v5の旧「全gate PASS」は位置relative L2を絶対RZ座標で正規化して原点依存だったため
無効化し、`CHARACTERIZED`に留める。未見の0.15625 usを使う逐次確認v6は0.625/0.3125/0.15625 us、各run
13,202 active recordで、原点不変な変位・速度・電荷のfine-pair relative L2を
`3.102727085428027e-5/3.92483251084038e-5/1.3511393490811483e-6`、観測次数を
`0.9041136/0.944312/1.123838`と確認した。このPASSはpre-eventの運用上の刻み選択だけであり、solver一致または
普遍的な物理精度ではない。frozen saved-state producer-form replayは8/8完了し、P19-Lで時刻0の過保守blockerを解消した。
M3-C1 exported-P1/native-field比較の6 gateは全FAILしたが、その原因分離用common-P1診断は287粒子×46 frame、event 0で
事前登録9 gateを全PASSした。さらに同じ限定scopeでfirst wafer stickまで延長し、event v14 candidateはCOMSOL referenceに
対してmaterial 20/20とpre-event prefix 9/9をPASSした。event時刻、hit位置、terminal chargeの絶対差はそれぞれ
`2.157542807607049e-13 s`、`2.683964162031316e-14 m`、`4.7283812421028415e-09 e`である。v13の
query/refinement/accepted/depthは`16,427,517 / 7,792,306 / 8,635,211 / 16`で、450 usまでに
`7,623,460 / 7,792,306` refinement（`97.8331703092769%`）が既に発生していた。v14は
`842,927 / 11 / 842,916 / 11`、failure 0で、operator-observed shell wall-timeは同じlocal環境で約
`14m13s`から`36.5s`（約`23.4x`）へ短縮した。この時間値はmachine-local、概算、非gatingでありsolver報告値ではない。

v14のsolver-only 0.625/0.3125/0.15625 us再実行はrefinement 0で、位置・速度・電荷のRMS観測次数を
`2.029875353701904 / 2.0816971911764033 / 2.044084026475049`、fine-pair relative L2を
`6.099791486063973e-8 / 8.321356016032579e-8 / 1.3796067752988052e-8`として3量すべて
`ORDER_EVALUATED`、自己収束PASSとした。これはpiecewise P1場とmesh crossingを含むこの実caseの経験値であり、RK4の形式4次を
証明も否定もしない。旧v13の3刻み結果はprecision-stabilityの履歴として有効だが、artificial event subdivisionにより
3 runのaccepted-piece countが同一となったため、独立な時間収束の評価には使わない。

現行same-field comparison authorityはeval_v3の登録budgetと9/9 PASSであり、位置RMS/max/relative L2は
`4.06807283316903e-13 m / 1.2035778717837921e-12 m / 2.0316001855367802e-10`、速度は
`2.10847522277453e-9 m/s / 3.844306466969233e-9 m/s / 1.9630813983926987e-10`、電荷は
`1.1818881019499895e-7 e / 2.3758877887303242e-7 e / 4.670220207738041e-10`である。v14はsolver側の
event BVH broad phaseだけを変え、common-P1 COMSOL入力・reference・source MPHはhash-lock済みで不変なので、COMSOL studyを
再実行せず既存referenceへcandidateを再比較した。これらはCase-A 100 nm、common P1、Brownian off、first wafer stickまでの
限定結果であり、native-field parity、物理妥当性、Brownian、30 ms、他case/size/variantへ一般化しない。P18-Hも
解析・公開回帰と既存Freeze referenceへの外部15/15 gateを閉じた。B03 coreと二つのproduction sliceも完了したため、
P20 performance closeoutと意味を揃えたCase-A/Case-P 100 nm外部V&V/M3-C2は完了した。続く受理済みCase-P seed
`319032/319047/319063`の287粒子owner discoveryも、受理済み科学payload・work・case identity・revisionを完全一致させて完了した。
支配ownerは3 seedとも`integrators`（自己時間比42.58--42.86%）だったが、事前登録済みbounded ownerではないため
最適化は未承認でproduction変更はない。M3-C2A anchorは`CLOSED_ACCEPTED_WITH_LIMITATIONS`であり、
10,000粒子以上の性能、残るpackage、process並列は製品SLAを先に定義した独立work packageとする。COMSOL fittingは行わない。
このdiscovery判断後、明示指示によるbounded maintenanceとしてchord計算一箇所だけをcompiled batchへ統合した。
accepted 3 seedの科学payload/work/revisionを完全一致させ、end-to-end中央値を12.29%短縮したため変更を保持し、追加ownerへ進まず終了した。
これは製品scaleまたはCOMSOL速度比の認定ではない。authorityは
[`solver/evidence/m3c2/caseP_100nm_chord_optimization_v1/`](solver/evidence/m3c2/caseP_100nm_chord_optimization_v1/README.md)である。
この外部進捗は製品solverの目的・公開API・dependency方向を変更しない。
現行revisionはengine v36、compiled tile v18、proposal v10、event v16、boundary v5、result algorithm v5、
result/checkpoint schema v2、field location v4、memory plan v13、physics catalog v17、physics runtime v19、
RK4 enclosure v2、dense path v3、charge-stable exponential midpoint v3 / enclosure v3である。dense path v3は座標原点相対の
Bernstein enclosure、TwoDiff残差、world座標への方向付き外向き丸めを使う。dense位置評価も相対制御点と残差から
作り、始終点では保存済みendpointを厳密に戻す。公開chord-deviation boundだけはworld座標評価を覆う狭い
`8*eps`絶対座標termを含むが、物理的な曲率とenclosureは原点相対である。v3は、v2の広いpaddingにより
座標原点に依存して閉じなかったevent certificateを修正する。accepted endpoint、数学的なdense state path、
first-hit/event algorithm、engine、event、RK4 global enclosureのrevisionをdense path v3導入時には変更しなかった。

P17のCartesian 3-Dは物理的な3-D Brownianに必要な独立trackだが、RZ Brownian-off比較の開始条件にはしない。
common-fieldでのtime-discretization parityと、各producer固有fieldを含むworkflow agreementは別の判定として残す。

### Stage 4：時間依存と3D

同じreleaseへまとめず、次の二つを順番に独立milestoneとして実施する。

- Stage 4A：時間依存2D field、time-knotでのstep split、snapshot cache
- Stage 4B：tet4/tri3の3D geometryとfield、3D surface source、boundary BVH
- 各milestone後に大規模caseでCPU設計を再評価

### Stage 5：GPUと製品化

- GPUに適したcaseからbackend追加
- packaging、CLI、結果viewer
- 実測した条件に限りCOMSOLより高速という性能主張を行う

複数の大機能を同じreleaseで同時追加しない。一つの段階が解析解、scenario、performanceの三層を
通過してから次へ進む。

---

## 21. 現行`0.1.0.dev0`の採用状態

この表はP14-R baselineに、その後同じ単一engineへ受理した機能を加えた現行開発版の状態である。
初期baselineと後続stageを同一時点のrelease scopeとして扱わない。

| 項目 | 採用 | 後続・不採用理由 |
|---|---|---|
| clean-room package | 採用 | 既存runtimeは移植しない |
| COMSOL core dependency | 不採用 | adapter/V&Vだけ |
| RZ meridional deterministic | 採用 | 最初の高速vertical slice |
| RZ field + 3D particle | 独立P17 track | trajectory physicsのmodel追加と同じreleaseへ束ねない |
| full 3D | Stage 4 | geometry/eventの独立milestone |
| Epstein linear、finite-speed Epstein、Stokes–Cunningham | 条件付き採用 | 各revisionの明示適用域だけ。air相関をCF4/O2へ流用せず、finite-speed revisionも自由分子・等温Maxwell混合の範囲に限定 |
| electric、gravity | 採用 | 基本決定論力 |
| continuous charge | 採用（charge-stable slice完了） | RK4はexplicit `hL_Z<=0.5`を維持。native exponential/B03はmidpoint-frozen affine exponential `J<=0` root。clip・charge-only subcycle・第二engineなし |
| Brownian | B02/B03 production | B02の限定能力を維持し、B03でRZ projected force/charge連成を追加済み。単一seedを正解にせずM3-C2でmulti-seed分布を独立検証 |
| ion drag | P15-F production、P18-I optional | Barnesは維持し、比較対象二式は別sensitivity revision。blend・fallback・Case分岐なし |
| DEP | P18-D Stage 3（完了） | producer提供`grad(mean_E_squared)`を使い、gradient生成・回復不確かさをmodel不確かさと分離。Case-A 100 nm common-P1複合sliceはM3-C1でPASSしたが、DEP単独・native-field parityと物理妥当性は未認定 |
| free-molecular lift sensitivity | P18-L Stage 3（完了） | RZ no-swirl限定option。producer提供signed方位vorticity、正の明示係数、`lambda/a>=10`を要求し、一般lift/defaultとはしない。Case-A 100 nm common-P1複合sliceはM3-C1でPASSしたが、lift単独・native-field parityと物理妥当性は未認定 |
| benchmark-reference charge | P18-C Stage 3 | 保存式の全branchをversion化するが、production OML revisionを置換しない |
| neutral drag / mixture thermophoresis | P18-R Stage 3（完了） | producer認証済みone-effective-Maxwellian/pseudogas用の二つのsensitivity revision。`maximum_speed_ratio<=1`、`lambda/a>=10`、gate緩和なし。mixture truthやCOMSOL branchではない |
| terminal hold | P18-H Stage 3（完了） | generic `hold/held`の解析・resume・Brownian回帰と既存Freeze候補15/15を完了。COMSOL固有分岐、paused particle、再飛散は含まない |
| specular・restitution・確率stick | 採用 | `point_wall_laws_v5`の主用途壁挙動 |
| runtime plugin framework | 不採用 | 実例が揃ってから |
| automatic model fallback | 不採用 | 適用域を隠すため |
| JAX/GPU-first | 不採用 | dynamic eventとfloat64をCPUで確立 |
| trajectory全量保存default | 不採用 | 100万粒子で支配的になるため |

---

## 22. 実装開始の条件

P00～P02でcanonical schema、property authority、C01～C10を完了し、P03でXY/RZ coordinate規則とstatic
regular/P1/Q1 single-point field samplingを実装した。large-offset/high-aspect要素で再現したdefectは、
conditioning-awareな`field_location_v2`、最近傍supported provisional、非有限failureの回帰で閉じた。
P04ではtable source、release原点からの厳密ballistic、必須release/final、明示時刻frame、一つのclosed
segment、最小lazy `ResultView`を実装した。P05では大域topology監査、line BVH、ballistic exact first hit、
parameterなしstick/escape、boundary eventとterminal lifecycleを同じengineへ追加した。P06 revision 1では、
`cartesian_xy`のrequired field certificate、fixed charge、Epstein/electric/gravity、共通RK4 proposalと
kernel/physics verificationを実装した。post-reviewでは有限個のRK sampleが連続field supportを証明せず、frameの
有無がrun成否を変え得ることを確認した。前版engine v3は一般`rk4_reintegrated`を一律拒否し、その後の
revision 3aで`coupled_rk4_engine_v4`へ更新した。

revision 3b着手前hardeningでは、放物線support極値をdense stateと同じ演算順で評価するintegrator所有の
外向きintervalへ一本化し、受理意味論を`coupled_rk4_engine_v5`として記録した。proposalの位置・速度評価と
`coupled_rk4_proposal_v3`自体は変えていない。非一様`E_x ∝ -x`の公開API収束は位置・速度とも観測次数3.5以上で、
frame scheduleを変えてもfinal状態はbitwise同一である。

revision 2で有効なforce-coupled productionは、dragなし・厳密一様fieldから証明した一定加速度に限る。
topology-completeな材料boundaryでは解析放物線のfirst hitがdomain退出を捕捉し、boundaryなしでは全cell supportedな
`RegularLayout`に対して各proposalの座標極値を解析的にsupport検査する。C04/C05は公開API scenarioである。
revision 3aでは、boundaryless Cartesian XY、fixed charge、全cell supportedな`RegularLayout`について、global
field/model boundから全短縮RK4評価を含む外向きsupport enclosureとEpsteinの連続applicabilityを実装した。
証明不能caseはhidden subdivisionせずfail-closedにし、C02/C03の公開API時系列とoutput schedule不変性を検証した。
revision 3bは`coupled_rk4_engine_v6`として、材料boundary用のsequential accepted RK4 pieces、離散RK4 tube、
event-before-validity、証明不能時のfail-closedを実装した。同一macro proposalのparameter区間は流用せず、
accepted piece列をframe replayにも使う。非gating性能基準にはaccepted piece、candidate query、refinement、
最大深さを追加した。当該revisionで未対応だったRZ force couplingは後続P06-RZで完了した。
boundaryless unstructuredは別の連続包含gateまで未解禁である。continuous chargeは後続P15で完了した。Stokes--CunninghamはP06-Sで
明示air revision、Kn/Re適用域、XY解析oracle、away-axis RZ parityを揃えて追加済みである。
engine v7はこの数値意味を変えず、各粒子のleft-firstなoutstanding pieceを1件ずつwaveに出し、
完全に同じtarget timeのproposal行を256粒子chunkでbatch化する。event v5はintegrator所有のcomponentwise
chord deviationとroundoff幅から、events所有の単一facet外向き横断、normal time bracket、tangent/time-shiftを含む
position radius、endpoint clearanceを証明する。証明失敗時はsplit/full-tube fallbackとし、engine v7、proposal v3、
enclosure v1、schema/APIは不変である。64粒子の同一machine baselineはmaterial median 0.6450654 s、
boundaryless median 0.0471402 s、比13.6840である。event v4の0.9666176 sから約1.50倍、初期scalarの
2.3706326 sから約3.68倍で、accepted piece / candidate query / refinement / 最大深さは
1088 / 2496 / 1408 / 21である。engine v8は要求frameと重なるaccepted rowだけをreplay用に保持し、
64/256/1024粒子のtraced-allocation checkpointを完了した。この値はprocess RSSやP09 memory planではない。
同じrevisionでexact-mesh P1/Q1 material domainをP06-Uとして追加した。
engine v9はtable/surfaceを単一scheduleへ統合し、Philox4x32-10、fixed-time surface release、
静止壁のstick/escape/specular/probabilistic stick、priority/combined-normal corner、exact linear/quadratic pathの
reflection residual、C09のcap-driven split、ballistic RZ axis foldを追加した。source位置は
`edge_fraction`または明示measureのuniform、速度はfixed vectorまたはfixed-speed inward normalに限る。
manifestはseed、RNG/source/law revision、draw kind、resolved lawとwall/residual/axis集計を記録する。
engine v10 / event v6はCartesian XYの証明済み一定加速度surfaceへ拡張し、velocity優先のone-sided分類と
source-facet start-contact certificateを追加した。engine v11 / event v7はCartesian XY一般RK4の厳密内向き
surface departureとsingle-facet active-boundary residualを追加した。tangent、facet端点/corner、その他証明不能な
start-contactはfail-closedである。engine v12 / event v8でforce-coupled RZのsigned stage basis、axis event、
残時間継続を同じwork loopへ追加し、現行engine v36もその意味論を維持する。P11の指数法も
同じmaterial/RZ event loopを利用する。richer distribution、moving wallは
後続gateまで明示拒否する。
未使用機能の値を先にdefault化せず、次をowner packageの
着手条件とする。状態と詳細の権威は`solver/docs/decisions.md`である。

1. P03：conditioning-aware location、finite provisional、平行移動/high-aspect回帰（完了）
2. P04：data座標表現とmotion mode、typed frame schedule、必須final/release event、最小result schema（完了）
3. P05：大域topology audit、event要求値の単一resolver、ballistic exact first hit（完了）
4. P06 core：revision 1のphysics/RK4 verification、revision 2の証明済み一定加速度production、revision 3aの
   boundaryless XY regular support/applicability enclosure、revision 3bのsequential RK4材料boundary tubeは完了。
   engine v7のwavefront batching、event v5、accepted-path memory checkpointまで完了
5. P06-U：exact-mesh P1/Q1 material-domain一般RK4（完了）
6. P07：counter RNG、surface source、wall law、RZ measure/axis path split、C09 residual規則
   （engine v11 / event v7まで実装。分布拡張、moving wallは未完了）
7. P06-RZ：P07のaxis意味論を用いるforce-coupled RZ（engine v12 / event v8として完了）
8. P06-S：physics runtime集約後のStokes–Cunningham schema/applicability/oracle（完了）
9. P08：particle-local failure、series/probe、薄いCLIによるStage 1A closure（完了）
10. P09：HDF5 metadata preflight、固定ID対応のresident stateとresident-row active index、bounded microtile、phase memory plan、
    fresh/warm RSS/semantic harnessと代表規模characterization（engine v15で完了。synthetic全規模判定はP14で完了、
    unused hintはconsumerとともにP10へ移動）
11. P10：Numba field/physics/RK4 array pass、accepted endpoint hint、cold/warm semantic/RSS harness
    （engine v16 / compiled tile v1 / runtime layout v2 / memory plan v2 / physics runtime v2として完了）
12. P11：exponential midpoint、C03/極小・極大`h/tau`/可変係数収束、material/RZ/output parity
    （engine v17 / compiled tile v2 / proposal v4 / event v9 / physics runtime v3として完了）
13. P12：event-heavy residual work、compiled BVH/事前認証/wall prefix batch、stable parallel merge
    （engine v18 / compiled tile v3 / geometry v3 / runtime layout v3 / memory plan v3として完了）
14. P13：durable result、checkpoint/resume、bounded writer queue
    （engine v19 / result algorithm v3 / checkpoint schema 1 / memory plan v4として完了）
15. P14：10^4/10^5/10^6と直交thread/output条件のsynthetic performance matrix
    （engine v20 / compiled tile v4 / field v3 / geometry v4 / memory plan v6、23行×3観測matrixとして完了）
16. P14-P：outer ThreadPoolをthread非依存slab、stackless boundary BVH、flat SoA event wavefront、compiled
    boundary/RNG、stable prefix outputへ置換した後に並列gateを評価。未達のためthread APIと内部parallelを削除し、
    engine v27 / case schema v2のsingle-thread compiled runtimeへ収束して完了
17. P14-U：surface release＋非一様場＋材料wall＋多数stepを結合した時間/mesh収束、bound/event cost、
    serial end-to-end throughput・memory・utilityの代表用途gate。時間と2D meshのrefinementを分離し、
    10k/100k/1M×none/sample×3 fresh processと別1M profile、mode間core／mode内probe identity、RZ actual-field parityで完了
18. P14-R/T03：Windows/Linux配布smoke、再実行可能なperformance evidence、ResultViewだけを読む最小
    analysis/visualizationで製品v0.1を閉じる。T01/T02は実producer workflowの外部track、T04はprofile条件付き
19. P15（完了）：`oml_stationary_maxwellian_debye_huckel_v1`のRK4-first縦切りを受け入れ、その後
    explicit midpoint chargeも受け入れた。P15 closeout時点はfinite invariant/rate/derivative boundと
    `h L_Z <= 0.5`を両methodへ固定した履歴である。現行exponential pathはcharge-stable affine exponential
    updateへ置換され、RK4だけがexplicit gateを維持する。精度目的のhidden subdivisionは行わない。
20. M3-V（完了）：直接MPH inventory、12 packageの時間履歴、保存grid上のforce relevance、charge/Epstein
    formula parityとsampled applicabilityを外部toolで評価した。元の12 package全軌道一致は`NOT_APPLICABLE`。
    後続の別成果物exact-P1 pre-event companionは限定的な時間離散parityにPASSしたが、boundary、stochastic
    distribution、native-field空間収束、Cartesian 3-Dは`NOT_TESTED`とし、coreへ比較分岐を追加しない
21. F01（完了）：canonical thermal-flow RZ/P1から`boltzmann_bohm_sheath_c2_v1`を解く独立builder、
    semantic BC、Newton/GMRES、canonical plasma field/provenance、解析・収束・charge-balance testを実装。
    mixed-mesh adapter、代表case統合、COMSOL field V&VはF02として分離
22. F02（完了）：provider固有CSVをstrict adapterでcanonical P1へ変換し、1,987 node / 3,779 cellの代表meshで
    builderを解いた。既存RZ solverへの32粒子fixed-electric smokeと、同一export node上の記述的なfield比較を
    外部toolで完了した。独立mesh convergenceとCOMSOL trajectory／Freeze parityは未検証で、coreへ比較分岐を
    追加しない
23. P15-D（完了）：`oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1`を追加し、独立moment、
    zero-drift、微分、global bound、compiled parity、両積分器、材料wall、XY/RZ、checkpoint/resumeを検証した。
    catalog v6 / runtime v5 / compiled tile v8とし、engine v28とschemaを維持した
24. P15-E（完了）：`epstein_finite_speed_maxwell_mixed_equal_temperature_v1`を追加し、低速・高速極限、
    独立3-D分子速度積分、Jacobian/global bound、compiled parity、RK4/explicit-midpoint収束を検証した。
    catalog v7 / runtime v6 / compiled tile v9とし、engine v28、memory plan、schemaを維持した
25. P15-F（完了）：`barnes_collisionless_effective_speed_single_positive_ion_negative_debye_huckel_v1`を追加し、
    独立impact-parameter積分、zero limit、global bound、compiled parity、continuous-charge stage結合、
    RK4/explicit-midpoint収束、XY/RZ parityを検証した。catalog v8 / runtime v7 / compiled tile v10とし、
    engine v28、integrator、memory plan、schemaを維持した
26. P16（完了）：`waldmann_gallis_free_molecular_single_species_heat_flux_v1`を追加し、独立
    Chapman--Enskog moment、global bound、Kn/relative-drift連続適用域、compiled parity、両積分器収束、
    XY/RZ parityを検証した。catalog v9 / runtime v8 / compiled tile v11とし、engine v28、integrator、
    memory plan、schemaを維持した
27. P18-L（完了）：`rarefied_vorticity_sensitivity_rz_v1`をRZ/no-swirl限定で追加し、signed方位vorticity、
    `lambda/a>=10`、速度依存bound、pure/compiled parity、両積分器収束を検証した。catalog v14 / runtime v13 /
    compiled tile v15 / exponential enclosure v3とし、integrator v2、engine v30、proposal v7、schemaを維持した。
    B02 Brownianとの併用は拒否した。このP18-L完了時点ではCOMSOL軌道一致はM3-C1まで未検証だった
28. P18-R（完了）：`epstein_linear_effective_gas_sensitivity_v1`と
    `waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1`を既存式ownerへ追加した。producer認証済み
    pseudogas入力、`lambda/a>=10`、必須`0 < maximum_speed_ratio <= 1`を要求し、既存linear/P16の`0.1`を維持する。
    catalog v15 / runtime v14 / compiled tile v16とし、engine v30、proposal v7、integrator v2、schemaを維持した。
    保存auditはEpstein replay PASS、既存model applicability 12/12 `NOT_APPLICABLE`、thermophoresis replay
    `NOT_TESTED`で、COMSOL studyは再実行していない

未確定値をfallbackや空interfaceへせず、対応するsample caseと独立referenceを伴って確定する。

---

## 23. 主要な理論根拠

- 希薄気体dragの原典：
  [Epstein, On the Resistance Experienced by Spheres in their Motion through Gases](https://authors.library.caltech.edu/records/3dxz2-4rt66)
- 一般流体中の球形粒子運動：
  [Maxey and Riley, Equation of Motion for a Small Rigid Sphere in a Nonuniform Flow](https://doi.org/10.1063/1.864230)
- transition thermophoresis：
  [Talbot et al., Thermophoresis of particles in a heated boundary layer](https://doi.org/10.1017/S0022112080001905)
- free-molecular thermophoresis：
  [Waldmann, Über die Kraft eines inhomogenen Gases auf kleine suspendierte Kugeln](https://doi.org/10.1515/zna-1959-0701)
- 局所熱流束によるfree-molecular一般化：
  [Gallis, Rader, and Torczynski, Thermophoresis in Rarefied Gas Flows](https://doi.org/10.1080/02786820490490001)
- Saffman liftの適用条件：
  [Saffman, The lift on a small sphere in a slow shear flow](https://authors.library.caltech.edu/records/v662k-mg234)
- OU速度・位置の厳密更新：
  [Gillespie, Exact numerical simulation of the Ornstein–Uhlenbeck process and its integral](https://doi.org/10.1103/PhysRevE.54.2084)
- 微小粒子の離散帯電：
  [Draine and Sutin, Collisional Charging of Interstellar Grains](https://articles.adsabs.harvard.edu/pdf/1987ApJ...320..803D)
- ion dragのmodel-form不確かさ：
  [Khrapak, Basic Processes in Complex Plasmas](https://doi.org/10.1002/ctpp.200910018)
- COMSOLで利用可能な力・壁則を確認する外部参照：
  [Particle Tracing Module User's Guide](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/ParticleTracingModuleUsersGuide.pdf)
- COMSOL壁条件の意味：
  [Boundary Conditions](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_introduction.02.05.html)
- CPU JIT・並列化：
  [Numba performance tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html)
- GPU候補のfloat64、乱数、制御flow：
  [JAX default dtypes and X64](https://docs.jax.dev/en/latest/default_dtypes.html)、
  [JAX random numbers](https://docs.jax.dev/en/latest/random-numbers.html)

COMSOL資料は製品仕様の権威ではなく、対応可能な利用パターンと外部検証条件を確認するために使う。

---

## 24. 最終的な設計原則

> 本製品はCOMSOL比較器ではなく、外部場を利用する高速な粒子輸送solverである。

> coreは、場の補間、物理寄与、時間積分、最初の壁交差、壁応答、大規模batch実行だけを所有する。

> COMSOL変換、COMSOL比較、場生成、再mesh、可視化はcoreの外に置く。

> 最初から万能frameworkを作らず、表面発生から壁到達まで説明可能な一つの経路を完成させる。

> 新しい機能は、既存の意味論を複雑にする分岐ではなく、所有者が明確なmodelとして一つずつ追加する。
