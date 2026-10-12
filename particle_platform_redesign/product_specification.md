# 半導体チャンバー粒子軌道計算基盤 製品仕様書

## 0. 文書の位置づけ

本書はclean-roomで再構築した粒子計算基盤の製品目的、範囲、非目標、公開機能を所有する主仕様である。
現行packageは0.3.0、production engineはv62で、既存数値ownerを用いたserial/process実行を一つのengineで提供する。
追加JIT化は停止し、複数macro同期間隔は将来項目とする。旧solverはread-only archiveであり実装母体にしない。

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
| source別の性能・memory・品質・配布資格 | [parallel_execution_plan.md](solver/docs/parallel_execution_plan.md) |
| canonical入力/result・schema・完全なrevision一覧 | [case形式](solver/docs/case_format_v3.md)、[result形式](solver/docs/result_format_v3.md) |

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

現行0.3.0では次を目的にしない。

- COMSOLのUI、study、変数名、内部solverを複製すること
- 流体、伝熱、詳細プラズマ化学を粒子solver内で再計算すること
- 粒子が背景場を変える二方向連成
- 粒子間衝突、凝集、破砕、表面成長を最初から同時実装すること
- 任意CADを自動修復する一般ジオメトリーシステム
- 全ての力相関を一つの自動選択式へまとめること

未対応の物理は別式へ暗黙に置き換えず明示的に拒否する。検証referenceはcore外のoracleであり、
production fallbackや公開backend選択にはしない。

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
6. 既存CPUバッチのserial/process実行、同じ意味論を保つ将来拡張
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

現行production revisionの適用域policyは `error` だけとし、適用域外の実行を
fail-closedで拒否する。適用域外を数えながら計算を続けるmodeは、物理的に無効な
軌道を正式resultへ混在させるため実装しない。感度解析が必要な場合は、明示的に適用域を
持つ別model revisionか、solver外のparameter sweepとして扱う。

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

現行は明示的なlinear/finite-speed EpsteinとStokes–Cunninghamの適用域を持つ。transition相関、非球形粒子、
高Re dragは後続とする。

P18-Rの`epstein_linear_effective_gas_sensitivity_v1`は上の線形Epstein式を再利用し、producerが混合気体を
一つの有効Maxwellian/pseudogasへ畳み込んだreference/sensitivity入力だけを受理する。caseは
`0 < maximum_speed_ratio <= 1`を明示し、全stage・連続pathで(lambda/age10)とともにfail-closedに検査する。
既存`epstein_linear_v1`の速度比上限`0.1`は維持し、このrevisionをspecies-resolved mixture truthやCOMSOL branchとは呼ばない。

drag、thermophoresis、lift、gravity/buoyancyを合成する場合、各modelが重複して宣言する中性気体fieldと分子質量は
一つの背景authorityへ一致させる。linear EpsteinとWaldmann--Gallisはnative single-species同士または
effective-gas sensitivity同士だけを受理する。liftが無い場合もdensity照合を省略しない。

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

を一般形とする。現行は \(q\boldsymbol{E}\) に対応し、磁気項は要求ケースが得られた後に
追加する。電位しか入力されない場合、gradientはimporterまたはfield builderで一度生成し、
粒子loop内で不規則meshの数値微分を繰り返さない。

### 4.5 誘電泳動

球形粒子、準静電場の基本形を

\[
\boldsymbol{F}_{DEP}
=2\pi\epsilon_m a^3\operatorname{Re}(K_{CM})\nabla|\boldsymbol{E}_{rms}|^2
\]

とする。DCでは実数のClausius–Mossotti係数、RFでは複素誘電率と周波数を明示する。
`E` と `E_rms`、瞬時値と周期平均値を混同しない。現行の球形準静的DEPは、producerがDCまたはcycle-meanの
意味、solution、回復法を固定したcanonical `grad(mean_E_squared)`を受け取る明示optionである。
gradient品質とmodel-form不確かさを分離する。
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

P22では`talbot_cross_regime_radius_knudsen_v1`を別のoptional revisionとして使う。producer提供の`T`、
`grad(T)`、`rho`、`mu`、`k_g`、`lambda`と、case明示の`k_p,Cs,Cm,Ct`を使い、COMSOLと同じ
Talbot原著の半径基準`Kn_T=lambda/(drag_diameter_m/2)=2*Kn_p`でcross-regime Talbot式を評価する。
dragの`Kn_p=lambda/drag_diameter_m`とは式ごとに区別する。係数`Cs,Cm,Ct`は全て明示し、
直径基準の変数を使う場合は三係数とCunningham項を一括変換した代数同値式だけを認める。
v1は正の有限primitiveを要求するが
Knの数値cutoffを持たず、任意のKnの物理検証を主張しない。coreは温度を微分せず、Waldmannとの自動blend、
Knによるfallback、near-wall/negative thermophoresisを行わない。

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
`F_other`を黙って落とさない。B02 production revisionはEpstein linear drag、fixed charge、Cartesian XYに限定する。
B04ではこの数値pathを変えず、terminal `stick/escape/hold`に加えてactiveな`specular/restitution/maxwell_thermal`と
`probabilistic_stick` fallbackを同じfirst-hit経路へ接続した。active hitで使用済みrootの残余を流用せず、hit prefixを
確定してpost-wall stateから残時間を新しいstochastic rootとして開始する。既存B02 caseのpayloadはbitwise不変である。
root covariance、conditional split、mean更新がfloat64で表現不能な粒子は最後のaccepted stateで
`nonfinite_physics`となり、同一batchの正常粒子は継続する。

Stage 3のB03では、現在のRZ比較用に2自由度meridional投影であることを明記したrevisionを追加して完了した。
root始点からのnoise-free midpoint predictorで線形Epstein/FDT係数、全additive force、continuous chargeの
`G=dZ/dt,J=dG/dZ<=0`を一度凍結し、`u_eff=u+a/gamma`のjoint exact OUとroot内affine exponential
chargeで進める。これはstateを二つの
half-stepでcommitする対称operator splittingではない。native/effective-gasの線形Epstein、fixed/continuous charge、
既存additive forceを扱い、terminal wallとB04のactive wall lawを受理する。axis hitまたはactive wall hit後はaccepted prefixを
commitし、必要なaxis foldまたはwall応答の後に残時間を新しいstochastic rootとして再開する。
`macro_root_affine_exponential_v2`の全leaf intervalはprepared invariantへ照合し、
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

P22の`saffman_unbounded_creeping_shear_v1`はこれと別のoptional revisionである。XYまたはno-swirl RZの
signed面外vorticityをproducerから受け、壁から十分離れた球形・非回転・表面no-slip粒子にCOMSOL theoryと同じ
Saffman式を適用する。Stokes--Cunninghamまたはdragなしとだけ組み合わせ、半径基準で`lambda/a<=0.1`、
`Re_s<=0.1`、`Re_G<=0.1`、かつゼロshear以外は`Re_s<=0.1*sqrt(Re_G)`を全stage/pathでfail-closedに要求する。
最後の係数は`<<`を保守的に固定したrevision policyである。near-wall lift、粒子回転、finite-Re式へ自動切替しない。

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

### 5.2 対応するモデル

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

production sourceは次の二つを持つ。

1. `table`：粒子ID、位置、速度、径、質量、電荷、発生時刻を明示する。
2. `surface`：canonical facet ID、strict interiorのfacet内座標、速度、径、質量、電荷、発生時刻を
   粒子rowごとに明示する。

volume、line、time series injectionは、実ケースが必要になった段階で追加する。
P05 production profileは粒子表と固定電荷の意味論を確定するためtable sourceだけを受理した。
P07 exact-path sliceはsurface sourceと境界上departureを同じengineへ導入した。

### 6.2 surface source

surface sourceは本製品の主要機能である。現行`realized_internal_surface_contact_schedule_v5`は粒子ごとの
`particle_id`、canonical facet ID、strict interiorのfacet内座標、速度、release時刻、粒子物性、独立
`contact_radius_m`をHDF5へ保持する。有限半径中心はfacetから内向きへ半径だけoffsetする。
YAMLはsource名、`type: surface`、table名だけを結び、分布を生成しない。内部sourceは同じ共通列と
`position_m`を持つ。任意の分布や熱速度は外部入力作成側で一度realizeし、solver coreにdistribution engineや
source RNGを置かない。

次はfirst-party外部producerの後続catalog候補であり、現行でも生成機能を
対応済みとはみなさない。外部でrealize済みの結果をcanonical rowとして受け取る能力とは区別する。

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
回転面上の一様fluxを外部でrealizeする場合は`2πr ds`、2-D断面上の診断分布なら`ds`を使い、
両者を同じ物理分布として扱わない。生成結果はcanonical surface tableへ粒子ごとに固定する。
source乱数はsolver coreに持たず、thread、tile、output scheduleで初期条件が変わらない入力artifactとする。

### 6.3 境界上の初期位置

粒子を固定距離だけ流体側へ移動して開始しない。canonical surface rowはfacet IDとstrict interiorの
facet内座標を保持し、`sources.py`がprepared geometryからruntime位置を一度導出する。runtime scheduleは
その位置とcanonical source facet IDを保持し、owner cellと表面法線はprepared geometryを唯一のauthorityとする。
第二のresident local-coordinate authorityは持たない。最初の速度が流体側ならdeparture、壁側なら即時impactとして扱う。必要な数値許容差は
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

boundary groupごとの`contact_geometry`は未指定時`particle_surface`とし、粒子の独立した
`contact_radius_m`による材料接触を判定する。明示した`particle_center`は同じ粒子の中心通過を判定し、
仮想的な開口・検出面を材料壁と混在させられる。lawから方式を推測せず、escapeにもどちらの方式を指定できる。
物理radius、drag径、電荷半径、質量は保持する。一つのmesh/BVH上で候補ごとに有効接触半径、法線、残差を解決し、
材料端部のcapへ先に接触した場合は開口通過よりそのeventを優先する。表面sourceのoffsetとrelease/transferの
clearanceも同じgroup方式を使う。中心通過と材料接触の時刻順を局在budget内で証明できない同時候補は、
推測したpriorityや位置nudgeで応答せずfail-closedにする。XY/RZの既存2自由度と各既存integratorのscopeを維持する。

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

RK4等のforce-coupled pathでは、始終点を一本のchordとみなさない。一方、full-step/
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
補償演算、Hermite評価・包絡はroot-relative TwoDiffを使う。event v16はvalidなRK4 dense rowに限り、position
Bernstein control enclosureの全4点がfacetの既存budget込みinside half-spaceにあることをinterval算術で証明できた候補を
convex-hull性からclearする。証明できない候補、非有限・不整合なcontrol、exponential・scalar pathは従来どおり
split/fail-closedとする。monotone-approach clearもcubic Hermiteのderivative-Bernstein enclosureだけが明示opt-inする。
event ownerは最初のhit時刻を局在した後、そのbudget内で同時かつ共有nodeへincidentなfacetを集合として保持する。
exact、一般RK4、exponential midpoint、Brownianの全経路が同じpriority/combined-normal応答へその集合を渡す。
有限半径material contactとperiodic center crossingを同一のfirst-event順序で比較し、証明不能なcornerや
material/periodic混在は任意の一面へ丸めずsplit/fail-closedする。

support外のprovisional値はtrial生成にだけ使え、support外stageを含むstepをそのまま受理しない。現行の
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
stick/escape数にも加えない。現行の静的2-D XY/RZ境界は、粒子ごとに独立した
`contact_radius_m`を持つ。0は従来のpoint particle、正値はXY断面のdiskまたはRZ断面のsphere中心と
`line2` material facetのcapsule first contactである。drag径や静電半径から推定せず、接触時の中心を壁へ
snapしない。linear/quadratic exact、RK4、exponential midpoint、Brownianの全経路が同じfirst-event意味を
使い、有限半径で局在不能な区間をclearへ丸めずsplit/fail-closedする。

### 7.2 対応boundary law

P05 production profileはparameterを持たない`stick`と`escape`だけを受理した。
P07 exact-path sliceはspecular、probabilistic stick、複数candidateのcorner応答を追加した。
現行`contact_wall_laws_v7`は、parameterを持たない完全鏡面`specular`、係数付き`restitution`、
壁frameの鏡面／完全熱適応half-range Maxwell flux混合`maxwell_thermal`、明示した反射fallbackを持つ
`probabilistic_stick`、非deposition終端の`hold`を別の意味として受理する。

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
法線は粒子domainから外向き、impactは`(v-v_w)·n > 0`と定義する。`specular`と`restitution`は静止壁だけを
扱う。`maxwell_thermal`の`wall_velocity_m_s`はscattering frameを定めるが、静的geometryと整合するよう
group内の全facetへ接線方向でなければならない。法線方向へ移動する壁はmoving geometry対応まで拒否する。

#### Maxwell thermal reflection

`maxwell_thermal`は正の`wall_temperature_K=T_w`、`diffuse_reflection_fraction=alpha`を`[0,1]`、
finiteな2成分`wall_velocity_m_s=v_w`として必須にする。`alpha`は一般のenergy accommodation係数ではなく、
完全熱適応した拡散再放出を選ぶMaxwell混合確率である。残り`1-alpha`は壁frameの完全鏡面反射とする。
外向きunit normalを`n`、固定tangentを`t=(-n_y,n_x)`、`sigma^2=k_B T_w/m`とすると拡散branchは

\[
c_n=\sigma\sqrt{-2\log U_n}>0,\qquad c_t=\sigma Z_t,\qquad
\boldsymbol v'=\boldsymbol v_w-c_n\boldsymbol n+c_t\boldsymbol t,
\]

とする。`U_n`はopen uniform、`Z_t`は独立standard normalである。この法線分布
`p(c_n)=c_n/sigma^2 exp(-c_n^2/(2 sigma^2))`は入射数fluxで重み付けしたhalf-range Maxwell分布で、
固定zero速度を法線nudgeで通すものではない。RZ meridional modeでは既存2自由度の射影則であり、
3-D方位速度をサンプルした完全3-D cosine lawとは主張しない。

#### Probabilistic stick

\[
U<p_{stick}, \qquad 0\le p_{stick}\le 1
\]

なら付着し、それ以外は必須の`otherwise`へ明示したparameterなし`specular`、係数付き`restitution`、または
完全なparameterを持つ`maxwell_thermal`を適用する。fallbackをdefaultで補わない。現行が扱うのは
boundary lawごとの定数付着率だけであり、
境界material tableは未実装である。material、入射energy/angle、壁温度、電荷等へ依存する付着率は、入力変数と
適用範囲を宣言したversioned modelとして後から追加する。

wall-law乱数は`seed, particle_id, physical_boundary_event_ordinal, law_stream`をkeyにする。数値的な
path refinement、thread、tileはphysical event ordinalを進めず、付着・混合・熱速度drawを変えない。
付着、Maxwell branch、法線速度、接線速度とsource、Brownian、将来のcharge jumpはstream IDを分離する。

### 7.3 後続boundary law

`maxwell_thermal`を含む瞬時wall lawはpoint/finite-radiusで同じ応答catalogを使う。有限半径で変わるのは
first-contact geometryとcandidateごとの接触法線であり、law selectorを複製しない。次は同じpatchへplaceholderを
入れず、別revisionでstateとgeometryのownerを明示して実装する。

- finite-radius contactの現行範囲：静的2-D XY/RZのdisk/sphere対`line2` capsule first contact、surface sourceの
  inward center offset、table sourceのstrict clearance、candidate固有法線までを実装する。接触後のrolling/sliding、
  粒子間衝突、3-D surface offset、moving geometryは含めない。
- surface charging：wall material/electrical state、粒子charge transfer、保存event payloadと保存則を
  boundary/physics/output/checkpointの共同revisionで定義する。現行のstatic scalar law parameterに押し込まない。
- resuspension/desorption：wall-resident lifecycle、滞在時間、adhesion、接線stress、release ordinalと再開schedulerを
  source/lifecycle revisionとして定義する。terminal `stuck`/`held` rowを暗黙に再活性化しない。
- 入射energy・angle・materialに依存する付着／erosionは入力primitive、適用域、独立referenceを持つ別catalogとする。

入力作成側はwallと同じpolicy-free half-range Maxwell変換を使ってsurfaceの明示velocity rowを作れる。
solverはその速度を再標本化しない。Brownian surface releaseは外向き法線成分の絶対値がroundoff幅を越えることを
要求し、負ならdeparture、正ならzero-time impactとする。zero/tangent normal speedは位置nudgeせずprepareで拒否する。

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
`geometry_status`や未実装failure groupは保存せず、実装済み列の権威は`solver/docs/result_format_v3.md`とする。

---

## 8. 座標系

### 8.1 明示的に分ける四つのmotion mode

`2D`、`軸対称`、`3D再構成`を一つのflagで表さない。

canonical dataの座標表現とparticle motion modeも別の事実である。HDF5はgeometry/field dataが
`cartesian_xy / axisymmetric_rz / cartesian_xyz`のどれで格納されるかを所有し、SimulationSpecは下記の
motion modeを所有する。prepareが組合せを一度だけ検査し、result manifestへ両方を保存する。
現行はXY data/XY motionとRZ data/RZ-meridional motionを受理する。下記の二つの3D modeは
将来の独立利用case・geometry/field/event検証を伴う設計候補であり、0.3.0の公開能力ではない。

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

現行で運動を駆動するrequired fieldはparticle domain全体を覆うcontinuous node-associated
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
誤差評価を保存する。現行ではcacheを別fieldまたは別DataBundleとして生成し、physics設定が参照するlayoutを
利用者が明示選択する。場所ごとにmesh-native/cacheを切り替えるhybridは、必要性とspeedupを実ケースで
確認するまで実装しない。将来追加する場合も一つの宣言済みsamplerであり、
障害時のsilent fallbackではない。cache生成や品質診断をsolver hot pathへ入れず、support誤分類、
壁近傍誤差、representative trajectory/event、実測speedupの受入条件を満たす時だけ採用する。

現行preprocessor v4はnodal P1/regular/exact affine Q1からfull-support regularへのstatic/linear-time変換を認証する。
common partition上のvalue/stored-component gradient/support境界のrelative L2をXY/RZ物理測度で評価する。
全隣接snapshot区間の二次error/reference norm比をBernstein上界とroundoff allowanceで判定し、時間途中の相殺を
endpoint samplingで見逃さない。sample最大値やstationary root推定を連続保証に使わない。coverageの穴・重複、
解像不能patch、reference下界不足、資源上限超過を拒否する。warped Q1と部分targetは認証対象外で公開しない。
validationのmemory、workspace rows、patch work上限は必須入力とする。cache誤差の認証と利用用途のtrajectory/event
精度・speedupは別に判定し、このgateだけで場snapshot自体の時間解像度が十分だとは認定しない。

### 9.5 時間依存場

Stage 4Aの現行revisionはfixed spatial topologyの \(F(\boldsymbol{x},t)\) を同じfield interfaceで扱う。
canonical fieldは`time_s[T]`と`values[T,N,C]`を持ち、隣接snapshot間を各RK/指数法stageの実時刻で
線形補間する。全required fieldのtime knot和集合をmacro intervalの必須分割点とし、時間範囲外をclampしない。
現revisionは全snapshotをresident保持する。`hold`、periodic、明示discontinuity、streaming/double buffer、
moving mesh、時刻ごとに異なるtopologyは未対応であり、同じselectorへ未実装分岐を置かない。

入力snapshot自体がunder-resolvedならsolver stepを細かくしても修復できない。producerは
snapshot別の空間誤差や3 snapshot以上のleave-one-snapshot-out時間誤差で解像度を判断する。
現行preprocessor v4は指定された線形snapshot補間に対するcache誤差を全時間区間で認証する。明示指定のinterior knot omission診断はreport-onlyであり、元snapshotの時間adequacyを自動認証しない。診断の資源不足は適格cacheの公開を妨げない。
時間adequacyの採否はproducerと利用caseが所有し、不連続面を跨ぐ補間は現revisionの入力として受理しない。

### 9.6 急峻な空間分布

sheath、狭い流路、材料interfaceなどの不連続・急勾配はmetadataで明示し、異なるdomainやinterfaceを
跨いだstencilで平滑化しない。Stage 1はmesh-native fieldと`h,h/2,h/4`軌道収束を権威とし、外部
preprocessorが要素差、gradient、必要ならcurvatureからmesh/cache adequacyを判定する。後続の
bounded-dyadic modeでは、1 stepの移動距離、cell crossing、drag/charge係数と加速度の相対変化を
step-level indicatorにする。入力mesh自体が不足する場合はstepを細かくして成功扱いせず、再export・
外部remeshを要求する。

---

## 10. 数値積分基盤

### 10.1 決定論production integrator

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

局所的に `dv/dt = -(v-u)/tau + a` と置き、midpointで係数を固定する。integratorは同じ運動/charge状態を連成し、現行revisionの一覧はresult形式が所有する。
run-global enclosureを常に安全なfallbackとし、
start half-step predictorが通る局所field rangeから得たstage force boundとの共通部分だけを採用する。
連続applicabilityもglobal-firstで、未証明rowだけ同じ局所primitive rangeとprepared charge invariantで再認証する。
`E=exp(-h/tau)`、
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

現行は固定macro stepと別runの`h,h/2,h/4`収束を支援する。field knot、global source
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
drag-only、fixed-charge state、terminal stick/escapeだけを最初に受理した。P18-Hは同じB02 terminal subsetへ
parameterなしのholdを追加し、B04は既存first-hitとwall responseを使ってactive reflectionを追加した。B04は任意force、
非線形drag、3-D Brownianを解禁せず、各active hitでold stochastic tailを終了してpost-hit stateからfresh rootを開始する。

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

全production caseは同じengineを通る。load/prepareでcanonical、required field/model、座標、geometry、law、
memory/outputを一度解決する。workers=1は親、workers>1はrun内で永続するspawn workerが
同じ既存slab数値処理を使う。別physics/COMSOL/runtimeへdispatchしない。

各macroはreleaseを解決し、活動粒子を排他的global-row slabへ分け、各worker内でstage → proposal →
earliest hit → same-method hit prefix → law → residualを処理する。
親はconnection/sentinelを監視し、完了workerへ次slabを渡す。全task/output消費の完了を1 macroごとに待ち、
active整理・人口集計・要求出力・必要なdurable commitを完了して次macroへ進む。
RK stage、event refinement、OU leafごとの全worker barrierは置かない。

event/failureはworkerごと高々一batchをin-flightとし、親single writerが消費した後にACKする。
到着順storageを許すが、公開canonical event順は既存readerが一箇所で所有する。
workerはfileへ書かず、macro-wide sort buffer、第二scheduler、background writerを作らない。
ready barrier、ACK、終了join/cleanupはmacro barrierと別の待機である。

現行は1 macro同期に固定され、同期間隔の公開設定はない。複数macro連続実行は将来項目とする。
個別release時刻・出力要求でglobal数値stepを分割せず、accepted pathからframe/probeを評価する。
worker/slab/output/checkpoint変更で科学payloadとRNGを変えない。

## 12. 最小ソフトウェア構成

### 12.1 公開API

利用者が覚える操作は三つにする。

```python
from chamber_particles import load_case, simulate, open_result

case = load_case("case.yaml")
summary = simulate(case, output="runs/case_001", workers=1)
result = open_result("runs/case_001")
```

- `load_case`：既にSIへ正規化されたcanonical dataと設定を読み、schema、unit metadata、局所connectivity・
  owner参照、静的な参照整合を検査する。producer単位の変換はしない。
- `simulate`：開始時のprepareでmodel requirements、座標×integrator×backend capability、compiled
  evaluator、大域geometry topology、memory/resource planを検査し、その後に数値計算とstreaming出力を行う。
- `open_result`：最終粒子、event、保存したtrajectoryを遅延読込みする。

100万粒子の巨大なin-memory `Result` を標準で返さない。公開`PreparedRun`、公開preflight object、
多数のfacadeは作らない。workersはAPI/CLIの実行資源であり、case数値設定やRNG keyへ混ぜない。
CLIは同じ入口を使う。

```console
chamber-particles check case.yaml
chamber-particles run case.yaml -o runs/case_001
chamber-particles inspect runs/case_001
```

### 12.2 配置とmodule責務

new productionは`particle_platform_redesign/solver/`の独立uv projectだけへ置く。
旧root project/環境を復元しない。詳細なfile所有表は
[implementation_plan.md](implementation_plan.md)第4節、依存方向とruntimeはarchitectureが所有する。

coreはcase/coordinates/fields/geometry/topology/physics/integrators/stochastic/events/boundaries/
engine/cpu/outputに責務を分け、一般helpers/contracts、第二engine、空stage moduleを作らない。
adapter、builder、analysis/可視化、外部V&Vは`tools/`へ置き、coreからimportしない。

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

### 13.3 設定の所有

有効なschema・selector・列と設定例は[case形式](solver/docs/case_format_v3.md)とsolverのexamplesが所有する。
本書で古いYAML schemaやmodel IDの例を重複定義しない。

- `time`は積分区間とdt、trajectory scheduleは要求保存時刻を所有し、保存間隔を重複して持たない。
- sourceは位置・速度・release時刻・ID・独立物性をrealized canonical rowへ明示する。
- table release時刻はfinite SIで計算閉区間内とし、非負時刻の追加制約を置かない。
- SI以外の単位はadapterが一度変換する。coreでproducerごとの換算を行わない。
- workersはAPI/CLIへ置き、caseの数値設定に追加しない。
- 通常利用者へtile/chunkを要求せず、総memory予算で内部slabを解決する。
- 同期間隔Kの設定は未実装で、将来案をschema/APIへ先行追加しない。

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
- resident field/state、共有transport/replay、worker専用slab、bounded output/handoffを含む総memory/resource budget

`geometry.prepare`はcell incidenceから大域topologyを一度だけ検査し、non-manifold、重複boundary、
boundary vertex異常、自己交差・重なり、内部edgeのwall登録、RZ axis seam以外の外周欠落を拒否する。
boundary rowが0件でもvolume incidence監査を省略しない。CAD/mesh修復やfacet生成は外部builderの責務で、
coreは修復しない。この大域prepare不能は現在`simulate`の`SimulationError`であり、loaderの`CaseError`と
重複検査しない。

同じ静的条件をadapter、loader、preflight、runtimeで重複検査しない。動的finite/support/model適用域と
内部不変条件は既存numerical ownerで検査する。入力に問題があればdefault、epsilon、別modelで修復しない。

---

## 14. 出力仕様

### 14.1 必須成果物

JSON manifestとsegmented HDF5だけを永続形式とし、final/eventを設定で無効化しない。
現在のresult/schema/列は[result形式](solver/docs/result_format_v3.md)が所有する。
大規模runを一つのappend fileや全snapshot listへ依存させず、閉じたepochをatomic commitする。

finalはrun終了時刻lifecycle snapshotで、escapedは`kinematics_valid=0`のlogical null、
active/stuck/heldは1である。finiteな最後hit payloadはvalidity=0なら非権威値であり、科学値に使わない。
heldはhit位置・速度・電荷のinactive snapshotを保持し、通常更新を行わない。

partialにはmanifest、closed segments、交互A/B checkpoint、`LATEST`、finalを置く。
正常終了時だけcomplete manifestと`_SUCCESS`をcommitし、OUT.partialを同一volume内のOUTへrenameする。
通常open_resultは未完runを拒否し、明示recoveryだけがLATESTの確定prefixを読む。

checkpointはengineが全task/outputを消費したaccepted macro barrierでwork threshold到達、または最終macroを
判定した時だけ作る。既存cadenceの`W=macro_step_count+accepted_particle_pieces+candidate_queries+refinements`、
`T=max(2^20,128N)`、resolved threshold/components/barrierをmanifestとresume identityへ保持する。
output schedule/slab幅から独立で、保存要求のためにproduction stepを分割しない。
state、active/pending/cursors、logical/physical ordinal、accepted hint/replayに必要な状態を保存し、
strict input/schema/model/algorithm/座標/method/particle identityの一致でresumeする。migrationは作らない。

1. segmentを一時名へ書き、flush/close後に確定名へrenameする。
2. inactive checkpointを一時名へ書き、close後にA/Bへreplaceする。
3. LATESTをreplaceする。ここだけをepoch commit pointとする。
4. crash後はLATESTより新しいorphanを無視し、保存countからevent/frameを重複・欠落なく再開する。
5. final → complete manifest → _SUCCESS → directory renameの順で最終化する。

最初のLATEST前は初期stateから再実行し、final publication途中も次runが検査して完了させる。
physical RNG ordinalを厳密に復元する。child突然死で未完macroを公開せず、最後のdurable checkpointから回復する。
親single writerのbounded消費/ACKでbackpressureし、disk遅延でeventを捨てない。

参照checkpointと最新segmentはhash、全segmentは構造/countを検査する。
同shapeの過去segment値改変、power loss、remote filesystem、同一OUTへの複数runは保証外である。
CSV/Parquetなどは外部analysisが必要時にexportし、別production永続形式を追加しない。

### 14.2 trajectory保存mode

選択と時刻scheduleを直交させる。

- `selection: none | sample | ids | all`
- `schedule: interval_s | explicit_times_s`

時刻はfinite、狭義単調増加、重複なし、計算閉区間内とする。release前はframeに行を持たず、
release時刻ちょうどは初期状態を保存する。finalはparticle ID順とする。
受理するselection/scheduleの詳細はcase形式が所有し、未知設定を黙って無視しない。
frameはboundary eventに対し右連続とし、hit時刻ちょうどのstickはpost-event状態、holdはhit payloadを保持した
held状態、escapeは行なしとする。heldはinactiveのまま以後のframeへ残し、escaped粒子は以後のframeから除外する。

全粒子×全内部stepを既定で保持しない。付着率や人口履歴などはonline集計し、必要に応じて別の
analysis toolで再計算できる原始eventを残す。frameはaccepted pathから評価し、保存mode・間隔・probe
ID数を変えてもfinal state、event sequence、RNG pathが変わらない。親single writerでbackpressureし、
遅いdiskのためにeventを捨てない。

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

### 15.1 既存CPU数値ownerとprocess実行

既存NumPy/Numba float64 CPU数値実装を一つのengineで使う。追加JIT化は停止した。
既定workers=1、明示workers>1は同じslab処理をspawn workerで実行し、内部Numba parallelや旧ThreadPoolを併存させない。

- state/propertyは固定IDに対応するSoA、active/pending indexだけを整理する。
- accepted endpointだけがstate/cell hintをcommitし、trial/output sampleは変更しない。
- read-only field/geometry/model factsを共有し、state/replay更新は排他的row、scratchはworker専用とする。
- stage/proposal/event/Brownian workはbounded slabで確保・再利用し、全N temporaryを増やさない。
- 残時間workとevent/failure出力を別のflat数値arrayへ置き、各roundでrowごと高々一件をbounded stagingへ書く。
- event identityは`(particle_id,event_ordinal)`、公開canonical orderは`(time_s,particle_id,event_ordinal)`。
- fastmath無効、固定model加算/prefix順、counter RNGを維持する。
- ready後template解放、sealed transport、一shared replay rootと永続readerで不要な移送を減らす。
- all-task/output完了を1 macroごとに待ち、single writer/accepted checkpointを維持する。

粒子数や総時間だけでspeedupを保証しない。各macroの活動粒子仕事量とstartup/JIT/IPC/同期/outputを比較する。
現在の性能、memory、品質、配布、negative、source/input bindingは
[並列実行・性能文書](solver/docs/parallel_execution_plan.md)を唯一のownerとする。
旧P09～P14/Numba-thread試作の逐次履歴や性能表を主仕様へ重複記載しない。

### 15.2 将来のGPU等

GPU/別言語は具体的利用caseとprofileが必要性を示した時に検討する。
同じcase/model/RNG/event/result意味論を保ち、float64 accuracy、event outcome、統計、
compile/転送/出力込みend-to-end時間とmemoryを評価する。未指定backendのframeworkやstubを追加しない。

### 15.3 メモリ・評価

resident input/state/property、field snapshots、prepared index、共有transport/replay、worker専用scratch、
output/handoff/serialization reserve、page padding、safety marginの同時liveを一つの予算へ含める。
最小slabが収まらなければprepareで拒否し、要求worker数を黙って減らさない。
Brownian workは到達し得るadaptive max depthから保守的に計画する。
terminal stuck/heldは通常更新を止め、必要なsnapshotを保持する。

owned array planとPython/JIT/native allocatorを含むOS private commit/WSを区別する。
WS合計は共有pageを重複計上し得て、sampled値は瞬間peakを見逃し得る。
共有memoryからNUMA配置や総OS memory削減を保証しない。
全N×全時刻outputをresidentに持たず、out-of-coreは実測上必要な用途が定まるまで導入しない。

公開API時間にはworker起動/import/JIT/cache/IPC/writer/shutdownを含める。coldと複数warmを分け、
全科学identity、memory/cleanupを同じ入力で評価する。10^4/10^5/10^6の重いmatrixを毎PRへ重複させない。
COMSOL速度比には物理、field、刻み、粒子数、出力、hardwareを併記し、削減した仕事を同等速度比として示さない。

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

- 公開操作は3つ
- physical/lifecycle/support/accuracy/run integrityを一statusへ畳み込まない
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

基本core microcase packは次の`C01`～`C10`の10件とする。`C01`～`C05`は
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

現行engineではC04/C05が証明済み`quadratic_exact`経路、C02/C03が一般
`rk4_reintegrated`経路の公開API scenarioとして有効である。後者はfully-supported common
`RegularLayout`、fixed charge、既存のEpstein/electric/gravity、連続support/applicability enclosureを満たす
XY/RZ caseを扱う。revision 3b/P06-RZでは同じsubsetをtopology-completeな材料boundaryと連成し、terminal
stick/escapeまで実行できる。P06-Uでは材料domainと完全一致するfully-supported P1/Q1も一般RK4へ追加したが、
boundaryless unstructuredは連続包含証明がないため拒否する。P15の動的電荷は同じXY/RZの
`rk4_reintegrated`と`exponential_midpoint_reintegrated`へ接続し、exact pathを無効化する。
exact-path sliceではC08～C10も公開API scenarioとして有効であり、surface departure、静止壁の
specular/probabilistic stick、priority/combined-normal corner、複数hitとcap-driven residual split、ballistic RZ
axis foldを検査する。Cartesian XYの証明済み一定加速度surfaceも公開scenarioで検査する。
Cartesian XY一般RK4の厳密内向きsurface departure、single-facet active-boundary
residual、右連続state jump、次hit確認後のcap splitを公開scenarioで検査する。tangentまたはfacet端点/cornerからの
一般RK4 departureはfail-closedである。
Epstein RZ axis crossingの4次収束、frame schedule identity、away-axis Cartesian退化一致、
axis→wall順序、軸上不変状態、axis regularity failureを公開API/verificationで検査する。
C03の`exponential_midpoint_reintegrated`を一定係数の全要求frameで丸め誤差精度、広い`h/tau`でfinite、
smooth可変係数で次数1.8以上として検査する。Stokes--Cunningham一定primitiveの閉形式、output schedule identity、
Cartesian材料event/residual、surface departure後の同面再衝突、RZ axis→wallも同じproposal/event経路の公開scenarioで検査する。

時間依存fieldのverificationは、static退化、各stage実時刻、解析的piecewise-linear forcing、非grid knot分割、
grid一致knotの非重複、全layoutの補間、範囲外拒否まで追加済みである。3DはP17/Stage 4Bで別に追加する。
chargeと限定OUの既存検証および外部V&VのIDを`C01`～`C10`へ混在させない。

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
- joint OU/RZ projected Brownian、wall restart、checkpoint/worker変更resume（3Dは将来）

#### Performance

nightlyまたはreleaseで実施する。

- 10^4、10^5、10^6粒子
- eventが少ないcaseと多いcase
- mesh-nativeとregular cache。unstructuredはrealistic cell count、initial localization、cross-cell motionを分離
- trajectory off/sample/all
- cold JIT、warm run、最大memory、particles/s

現行source別の性能・memory・negativeは性能文書と固定raw evidenceが所有する。
cold/warm、startup/JIT/IPC/outputを含むpublic時間と科学identityを同じ入力で記録する。
旧thread matrixやmicrokernel速度を現在の製品能力へ付け替えない。

量ごとの誤差は原則

\[
E_q=\frac{|q-q_{ref}|}{atol_q+rtol_q\max(|q|,|q_{ref}|)}
\]

で評価し、位置、速度、電荷を一つの曖昧なscalarへ混ぜない。最小受入表を次へ固定する。

| 対象 | 最小受入基準 |
|---|---|
| RK4 | smooth coupled `x,v,Z`で観測次数3.5以上 |
| exponential midpoint | 可変場で1.8以上、一定係数で丸め誤差程度、広い`h/tau`で有限 |
| P1/Q1（tet4は将来） | 解析fieldを再現し、隣接cellの共有境界で値・supportが一致し、owner選択が決定論的 |
| first-hit | 時刻bracket・facet距離がbudget内、step/mesh半減でtime/point/facet集合が収束 |
| specular | parameterなしで相対法線速度が厳密に反転し、接線速度が不変 |
| restitution | 相対法線速度が`-e_n`倍、接線速度が`e_t`倍 |
| external surface producer | RZでrealizeしたsample密度が`2πr ds`に一致し、結果をcanonical rowへ固定 |
| continuous charge | 平衡、残差、coupled軌道が設定次数で収束 |
| axisymmetric 3D（将来） | 方位角回転に対して軌道・eventが共変 |
| OU | B01/B02のjoint平均・共分散・MSD・平面first-passageが統計区間内。B03は定数係数mean/full covariance exact、noise-off 2次、manufactured weak mean観測次数`>=0.9`。一般SDEのstrong/weak 2次は主張しない |
| time field | snapshot半減で所定次数、discontinuityを跨がない |
| cache | field/gradient budget内、代表caseのboundary outcome不変 |
| CPU process | worker/slab変更で全公開科学payload、event、RNG identityが不変、cleanup正常 |
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

## 20. 現行milestoneと将来項目

初期bootstrap、決定論vertical slice、既存CPU compiled化、durable I/O、charge/force/Brownian、
有限半径/periodic/time fieldは現行の単一engineへ統合済みである。
0.3.0は既存数値ownerによる同一runのprocess実行、共有transport、完了順slab配分、一batch ACK、
single writerと異常終了/worker変更resumeを提供する。
完了済みP00～P22の逐次手順、旧thread試作、休止native-team工程を現在の残件として列挙しない。

実装責務と現在のbacklogは[implementation_plan.md](implementation_plan.md)第16～21節が所有する。
完全3D、別物理、NUMA、別言語/GPU、out-of-coreは具体的case・独立reference・profileを用意した条件付き拡張とする。
利用者未指定のaccuracy/SLA、実験的製造予測保証、全CPU数比例効果を今回の必須条件へ追加しない。

将来の複数macro連続実行は、数値dtとは独立した最大連続macro数を検討する案である。
現在の同期間隔は1に固定で、0.3.0へ設定/API/stubを先行追加しない。
将来Kを増やしてもdt/field knot/release時計、stage・same-method hit、Brownian counter/条件付き分割、
毎macro人口、要求frame/probe、bounded output/ACK、single writerを維持する。
checkpointは全workerが同じaccepted時刻まで完了しoutputを消費した状態だけを公開する。
child途中死/resumeと将来の粒子相互作用/場feedbackの連成周期を含めて科学一致・時間・memoryを評価する。

## 21. 0.3.0の提供範囲と保証の境界

| 領域 | 現行能力・条件 |
|---|---|
| clean-room core | 独立uv project、三公開API、COMSOL非依存、single engine |
| 座標/場 | XY/RZ meridional、regular/P1/Q1、static/固定topology linear-time。完全3Dは将来 |
| 決定論 | exact退化形、RK4、exponential midpoint。actual stage/continuous support/applicability、same-method event |
| charge/force | 各明示revisionと適用域。RK4 explicit charge gate、指数/Brownianの連成更新を維持 |
| Brownian | joint OU、RZ投影、active wall fresh-root、base depthと条件付き追加refinement。連続OU exact first passageは非主張 |
| 境界 | terminal/active law、独立接触半径、group surface/center意味、静的XY periodic。moving geometry/再飛散は将来 |
| 実行 | 既定1/明示複数process、同じ数値owner、排他的row、共有input/state/replay、worker専用scratch |
| 同期/出力 | 1 macro固定、event/failure一batch ACK、親single writer、accepted durable checkpoint/resume |
| 外部tool | canonical adapter/builder/cache、V&V、最小analysis/可視化。core専用COMSOL分岐なし |

表は全座標×model×integrator×boundaryの組合せを保証しない。対応matrixとparameterをprepareで拒否/受理し、
モデル適用域は第3～10節と物理/数値文書、用途精度・速度・memoryは登録条件へ限定する。
完全なscientific revision一覧を主仕様へ重複転記せず、result形式と実run manifestを参照する。

source別の主用途・時間場補足・品質・配布資格は性能文書第9.27節が所有する。
Case A/PのBrownian full-step人口比較/parallel遅化は第9.28節、P14-U 1万粒子320stepは第9.29節へ分ける。
少粒子条件の遅化と大粒子条件の改善をともに保持し、粒子数/総時間だけで普遍的speedupを約束しない。

COMSOL人口gateのPASSはその登録済みcommon-field/cohortに限定し、native FE、全force/charge RHS、
詳細boundary、時間発展v/Z分布、物理的予測精度へ拡張しない。既存seed参照を新unseen familyと呼ばない。
DEP/lift等の複合slice数値一致を単独model validationと混同しない。
mixture/effective-gas sensitivityをspecies-resolved truthやCOMSOL専用branchへ読み替えない。

公開済み0.1.0/0.2.0のtag・資格・receiptはimmutableな履歴とする。
旧sourceの能力/性能を現行へ遡及付替えしない。
Git branch push、remote CI、tag、Release asset/registry公開はそれぞれ実操作のreceiptで状態を記録する。
local Windows/Linux資格を未実行remote CIや正式公開へ読み替えない。

## 22. 変更・拡張の着手条件

新model、座標、integrator、backendは、具体的利用caseと既存能力で解けない理由を先に閉じる。
必要field/state、所有module、適用組合せ、解析解/独立reference、時間/memory影響、削除旧経路を
既存decision recordへ短く記す。空interface/default/fallbackを先行しない。

変更は三API/single engine、stage連成、support/event-before-validity、RNG、output/checkpoint不変条件を維持する。
原因を表す最小verificationと公開scenarioを通し、hot pathなら同一条件before/after科学・時間・memoryを確認する。
置換したcode、設定、fixture、test、文書を同じ変更で削除する。

旧進捗・数値改良・否定的性能試作は性能記録、固定evidence、Git履歴へ一元化する。
保存archive/registration/receiptのbytesや採否を変更せず、現行主仕様を過去の進捗diaryにしない。
品質・配布commandはquality文書、実装順はimplementation plan、数式・numericsは各ownerが所有する。

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
