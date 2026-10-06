# COMSOL比較・検証・原因特定方法

## 1. この文書の目的

本書は、粒子追跡コードをCOMSOLと比較するときの検証（verification）、妥当性確認（validation）、差異原因の切り分け方法を定める。対象は現在の `model_dataset` と、今後作成する比較ケースである。

最終時刻の座標、最終状態の個数、平均値だけを一致判定に使ってはならない。それらは結果の要約であり、運動方程式、場補間、電荷連成、時間積分、境界イベントのどこが正しいかを示さない。比較の基本単位は、同一粒子についての「時間付き状態列」と「イベント列」である。

この方法論はCOMSOLを常に真値とみなすものではない。次の三つを区別する。

1. **互換性検証**：同じ入力、式、時間積分、乱数条件でCOMSOLの計算結果を再現する。
2. **数値検証**：解析解、製造解、高精度解、刻み幅収束に対して実装が期待次数で収束する。
3. **物理妥当性確認**：選択した物理モデルが実機や実験に適切かを評価する。

COMSOL一致だけで数値的・物理的に正しいとは判定しない。candidate solverの主要な数値gateは、解析解・manufactured
caseとcandidate自身の`h,h/2,h/4`自己収束である。candidateの刻みは自身のaccuracy・stability・applicability条件から
選び、保存COMSOLの固定刻みを模倣させない。COMSOLとの同一刻み1-step比較は、式、field表現、boundary、確率modelと
乱数割当まで同一と証明した専用互換性microcaseの診断にだけ使える。別の正当な数値解法がCOMSOLの各時刻値と完全一致
しないことを直ちに不具合とはせず、製品coreに「COMSOL互換モード」を設けない。

### 1.1 2D COMSOL比較の最小受入基準

通常の2D受入判断では、本書後半の全Gateと全診断成果物を毎回実行しない。必須判断は次の四つだけとし、後段の詳細Gateは
不一致をこの四責務のどこへ帰属するか分からない場合にだけ使う。性能比較は別work packageであり、この受入には含めない。

| 必須判断 | 比較するもの | 合格条件 | 不合格時に直す所有者 |
|---|---|---|---|
| 初期条件 | R-Z基底、単位、geometry/field hash、粒子ID、放出時刻・位置・速度、径、質量、初期電荷、選択model、boundary対応 | ID、件数、カテゴリ値は完全一致。数値はexport表現誤差内。表面sourceは明示粒子列の一致を先に使い、分布sourceは別の一回の分布試験で確認する | COMSOL exporterまたは外部adapter。coreへ補完・推測を追加しない |
| 場と運動右辺 | 同じ時刻・位置・速度・電荷におけるsupport、原始場、電荷変化率、electric、neutral drag、thermophoresis、選択ion drag、DEP、lift、gravity/buoyancyの各寄与 | 同じcanonical field表現ではsupport/領域を完全一致させ、各寄与の符号・方向・大きさと合力閉合を、事前に決めたexport/評価不確かさ内にする。Brownianは乱数値でなく係数をここで確認する | field差はadapter/field producer、式差は該当する一つの`physics` owner。合力だけを合わせない |
| 境界 | 表面からの離脱、最初のhit時刻・位置・boundary意味、法線、R-Z軸通過、lifecycle、反射時の衝突後速度と残時間 | 使用する各boundary lawに少なくとも一つの正例があり、event種類・boundary意味・terminal lifecycleが一致し、R-Z軸をwallにしない。時刻・位置・速度は両solverのevent収束不確かさ内。event 0件はPASSでなく`NOT_TESTED` | 交差は`geometry/events`、応答は`boundaries`。位置nudgeや許容値拡大で合わせない |
| 時間軌道 | Brownian-offの同一場・同一modelによる境界前の全時刻`(r,z,v_r,v_z,Z)`と最初のevent、Brownian-onの時刻別分布・fate | 両solverが独立に刻み収束し、決定論差がCOMSOL時間・field/export・candidate時間/event誤差の保守的な合成帯内。確率caseは事前登録したensemble帯内 | 最初に不一致となる上の三責務だけを直す。単一seedのpathや最終座標へ合わせない |

比較点は全保存行へ無条件に広げず、初期点、各力が支配的になる点、強いfield勾配、境界近傍、電荷の代表的な極値を含む小さい
固定集合を使う。native COMSOL有限要素場とexported P1場の差はfield representationの評価であり、trajectory coreの不具合に
しない。solver同士の互換性は両側へ同じcanonical fieldを与えて判定し、native-field精度はadapter/field producerの別statusとする。

現状の100 nm Case-A/Case-P common-P1 anchorは、初期意味、同一場での全決定論力・動的電荷を含む境界前軌道、Brownian
ensembleについて合格している。隔離したFreeze/Disappear意味とCase-A最初のwafer hitも確認済みである。さらにP21/M3-C3の
critical boundary microcaseは、一つの2D axisymmetric rectangleで表面接触からの離脱、specular reflectionと同step残時間、
R-Z軸通過をCOMSOL/public API/解析解で比較し、3刻み・132/132 gateをPASSした。最大solver間位置差は
`1.61339e-17 m`で、authorityは
[`solver/evidence/m3c0/critical_boundaries_v1/`](solver/evidence/m3c0/critical_boundaries_v1/README.md)である。

aggregate three-current production revisionはcatalog v17 / runtime v19 / tile v18で同じsingle engineへ統合され、標準
verification/scenario suiteと品質gateを通過した。Case-P派生companionのpriority 3はproducer-owned one-sided cacheにより
common-P1全1987節点のfiniteな負イオン5 primitive authorityを生成し、入力blockerを解消した。priority 4の共有three-current `Z0`を用いた
common-P1、Brownian-off、100 nm、287粒子、30 msの外部比較は、candidateと明示drag COMSOL referenceの各3刻み収束、
全frameのlifecycle/finite mask exact、共通の有限lifecycle stateの`r,z,Z`直接gate、共通active stateの`v`直接gate、
141件のevent/fate identityとevent時刻gateをすべて`PASS`した。元Case-P二電流anchorは不変である。この結果を非認定の
physical-model validationや普遍的COMSOL同等性へ拡張せず、P21と明示scopeの2D benchmarkは
`CLOSED_ACCEPTED_WITH_LIMITATIONS`、`2D_CRITICAL_VV_COMPLETE`とする。判定authorityは
[`solver/evidence/m3c3/caseP_three_current_companion_v1/`](solver/evidence/m3c3/caseP_three_current_companion_v1/README.md)の
`trajectory_evaluation.json`と明示dragの3件の`reference_dt_*_run_receipt.json`である。

元Case-PのCOMSOL `auxq` は負イオン密度を診断出力にだけ使い、帯電電流へ加えていない。したがって負イオン対応は元Case-Pの
不具合修正や既存anchorの再認定ではなく、明示的に選ぶ拡張modelとする。三電流revisionは`R_Z=Gamma_+-Gamma_e-Gamma_-`、
明示`n_-`,`V_-`,`u_-`,`m_-` fieldを使うaggregate singly-negative-ion collectionとし、screeningは既存の明示fieldだけを
authorityにする。species-resolved currentや負イオンprimitiveからのscreening再計算は範囲外である。
上記を閉じる改良は、(1) この一般charge revision、(2) 完了済みcritical boundary microcase、(3) charge revisionを使う
Case-P派生のmeaning-matched COMSOL companionとcandidateのrepresentative full-physics比較一つに限定する。3の入力は、
producer側でone-sided domain-3境界値を定義した同じcanonical節点順の`n_-`, total `u_{-,r}`, total `u_{-,z}`, `m_-`,
`V_{T,-}` exportで満たし、100 nm・287粒子・30 msの比較を一回だけ実行した。
既存のCase-A terminal evidenceと元Case-P anchorは再利用し、Case名によるcore分岐、新しい診断framework、全12 packageの
総当たりは作らない。

この2D比較は次をすべて満たした時点で、認定範囲を併記して`2D_CRITICAL_VV_COMPLETE`として終了する。このstatusは
すべての任意model、field表現、粒径、境界lawについての一般的なCOMSOL同等性を意味しない。

1. Case-AとCase-Pの代表anchorで初期条件と全有効右辺寄与が上表を満たす。
2. Brownian-offの同一場軌道が自己収束し、合成不確かさ内で一致する。
3. 表面放出、terminal response、specular reflection、R-Z軸通過の正例が境界条件を満たす。
4. Brownianを製品caseで使う場合だけ、複数seed ensembleが事前登録帯を満たす。
5. 認定scope内にfailure、support外受理、未説明の最初の分岐がない。scope外の未適用物理は`NOT_APPLICABLE`、入力authorityが
   ないconditional coverageは`BLOCKED / NOT_EVALUATED`として分離する。後者をPASSや物理的な不適用へ読み替えない。

終了後は追加粒径、第2 ion-drag、全12 package、native-field、3D、時間依存場をこの比較の追加Gateにしない。それらは新しいmodel
またはcoverageを要求された時だけ独立work packageで一軸ずつ評価し、既存の2D比較を再開しない。

## 2. 現在の `model_dataset` から分かることと限界

### 2.1 比較軸として使える事実

監査結果は次のファイルに固定した。

- [dataset_audit.json](evidence/dataset_audit.json)：ケース、表、列、幾何、状態整合性の詳細
- [case_audit_summary.csv](evidence/case_audit_summary.csv)：12ケースの行数、粒子数、時刻数、欠損数の要約
- [population_time_series.csv](evidence/population_time_series.csv)：時刻別の粒子状態集計
- [variant_divergence_time_series.csv](evidence/variant_divergence_time_series.csv)：二つのイオンドラッグ定義間の時刻別差異
- [variant_particle_divergence_summary.csv](evidence/variant_particle_divergence_summary.csv)：粒子別の差異発生時刻と最大差
- [field_redundancy_audit.json](evidence/field_redundancy_audit.json)：粒径別背景場の重複と粒子依存列の監査

各ケースは287粒子、121保存時刻、34,727行で、粒子・時刻キーの重複はない。初期行は放出表と一致し、最終要約は最後の保存行と一致する。欠損値は主に消失後の状態4に対応しており、活動中粒子の必須状態には欠損がない。したがって、保存済みの時間列は外部producerの挙動を記述する基礎データとして利用できる。ただし、同じ物理、field表現、boundary、確率意味論が確認されるまではsolverの数値合否を決めるreferenceにはしない。

一方、現在の比較表は内部時間ステップやRK4の各段階を保持していない。境界衝突の厳密なイベント時刻、衝突直前・直後の速度、法線、同一マクロステップ内の残り時間も保存していない。このため、現在のデータだけで「RK4実装が同一」「最初の壁交差が同一」と証明することはできない。

### 2.2 終点・最終状態だけでは不十分である実例

二つのイオンドラッグ定義を粒子IDと保存時刻で対応させると、Case Pの10 nmと30 nmでは最終状態が全粒子で一致するにもかかわらず、全粒子が途中で1 mmを超える位置差を経験する。1 ms時点の位置差中央値はP-10 nmで約14.8 mm、P-30 nmで約4.40 mmである。5 ms時点ではP-100 nmの中央値が約57.8 mmに達し、その時点ですでに状態一致率も約0.906へ低下する。

Case Aでは1 ms時点で位置差中央値が約0.61–1.16 mm、5 ms時点では約6.81–25.3 mmである。30 msでは消失済み粒子の位置が欠損になるため、有限座標を持つ粒子だけを比較すると選択バイアスが生じる。したがって「両者で最後まで座標が残った粒子だけ」の終点差は、系全体の一致度を表さない。

この実例はCOMSOLと新ソルバーの差そのものではないが、物理式の小さな違いが早期に軌道を分岐させ、最終状態集計だけではその分岐を検出できないことを示す。根拠となる値は [variant_divergence_time_series.csv](evidence/variant_divergence_time_series.csv) と [variant_particle_divergence_summary.csv](evidence/variant_particle_divergence_summary.csv) にある。

### 2.3 現在のデータを使う際の重要な制約

- 保存された背景場は粒径10、30、100 nmで実質的に同一である。しかし `particle_Knudsen_number`、`Epstein_response_time_s`、`Brownian_diffusivity_m2_per_s` は全粒径パッケージで10 nm相当になっている。これらを入力場として使わず、圧力、温度、密度、粘度、平均自由行程などの原始量と各粒子の径から実行時に再計算する。
- メッシュ点場の33,449行には同一座標の重複があり、丸め後の一意座標7,732点が幾何頂点数と一致する。CSVの行番号をメッシュ節点番号として扱わない。明示された節点ID・要素接続と正規化済み対応表が必要である。
- `inside_model_domain` を粒子領域の真偽マスクとして扱わない。粒子領域は領域ID、要素接続、境界ファセットで判定する。
- 現ケースの放出は内部格子からで、主要用途である表面放出を検証しない。
- 現ケースでは反射およびfreezeの正例がなく、壁相互作用全般を検証できない。
- 時間依存場、3D場、軸対称から3Dへの再構成、確率反射、複数衝突の網羅ケースではない。
- 保存済み100 nm・30 msのCase A/Pは手動の陽的fixed RK4 10 usで、Brownian featureが有効である。Brownian-offの
  candidate v3とは確率model/RNG意味が一致せず、COMSOL native finite-element fieldとcandidate exported-P1も同一表現ではない。
  したがって単一粒子path差は記述用であり、決定論的parityまたはcandidateのcore gateにしない。

現在の12ケースは「定常2D軸対称場における統合回帰ケース」として使い、機能網羅の根拠にはしない。

### 2.4 M3-V applicability/relevance closeout

M3-Vは`solver/tools/vv/comsol/`に置く外部toolであり、`chamber_particles`からimportしない。COMSOL 6.4で
二つのMPHを`loadCopy`、`-nosave`で読み、study/solution、Case P/A particle feature、Case-A nonlinear Poisson
closureとsemantic boundary selectionを直接inventoryした。実行前後のMPH SHA-256は一致する。

12 packageを一括評価した結果は`solver/evidence/m3v/`に保存する。重要な判定は次である。

- 保存primitiveからrelative-drift regularized two-current `dZ/dt`を再構成でき、current-scale residual最大は
  約`1.6e-12`である。元Case-P COMSOL `auxq`も電子＋正イオンの二電流を意図的に使い、負イオン密度は診断量に留める。
  従ってこれはsame-form provenanceであり、species-resolved物理または後続three-current拡張の妥当性ではない。保存履歴は
  exponent clampとion-energy floor branchも網羅しない。
- 記録された内部刻み10 usと解析的なfrozen-state `dR/dZ`から得る局所`h|dR/dZ|`は最大約`0.721`である。
  これはreference式の局所特性であり、continuous pathやCOMSOL integratorの安定性認証ではない。
- COMSOLのEpstein係数は`delta=1+sigma_R*pi/8`、`sigma_R=0.9`で再構成でき、force vector relative residual最大は
  約`1.1e-15`である。これはlinear Epsteinの速度・Kn適用域を拡張しない。
- P15 stationary OMLは全caseでion drift条件を満たさない。さらにCase P 6ケースは負イオンを含み、局所有効正イオン
  質量もP15が要求する単一scalar massへ一致しない。linear Epsteinも一部の粒径・保存状態で適用外となる。
  したがってproduction全軌道比較は`NOT_APPLICABLE`であり、閾値を緩めて一致させない。
- P15-Dの有限相対drift OMLを追加しても、この`NOT_APPLICABLE`判定は変わらない。Case Pは単一・単価正イオンという
  species契約を満たさず、Case Aの一部履歴はP15-Dが扱わない正表面電位を要求する。比較のために負イオンを無視したり、
  有効質量へ畳み込んだり、正電位branchを別式で補完しない。
- 二つのion-drag packageは全保存時刻でcensor-awareに比較する。Case Aは設定差がion dragだけだが単一Brownian
  realizationであり、Case Pにはlift式の構文差もある。初回分岐後のforce差を単純な原因寄与へ読み替えない。
- 決定論force積分は10 us / 100 us / 1 msの非一様保存grid上の台形近似であり、relevance用である。Brownianは
  sampled RZ RMSだけを報告し、impulseとして積分しない。

`comparison_manifest.json`の`execution_status=PASS`は監査処理が完了した意味だけを持つ。
`physics_certification_status=NOT_CERTIFIED`を別に保持し、`NOT_TESTED` / `NOT_APPLICABLE`を成功へ畳み込まない。

M3-Vは完了済みのhistorical closeoutとして書き換えない。Stage 3のM3-C0/C1/C2は、P15/P16のgateを緩めるのでなく、
現在選択されている帯電、二つのion drag、DEP、lift、必要ならdrag/mixture thermophoresisを明示的なoptional revisionとして
追加した後の新しい比較系列である。各caseは次の四statusを独立に持つ。

- `formula_parity`：凍結状態で保存式と各寄与が一致するか
- `numerical_trajectory`：同じmodel/fieldで刻み収束した時間軌道が一致するか
- `physical_applicability`：そのmodel revisionの仮定を全pathが満たすか
- `stochastic_ensemble`：複数seedの統計量が事前登録区間に入るか

一つのstatusを他のPASSで代用しない。

### 2.5 P18-R saved-artifact audit

P18-Rは既存保存成果物だけを読み、COMSOL studyを再実行せずにdrag/thermophoresis closureを層別監査した。
native linear Epstein式は最大相対残差約`1.1e-15`で`PASS`したが、これは式provenanceだけである。既存P15-E/P16の
`physical_applicability`はmixture/model authority不一致により12/12 caseで`NOT_APPLICABLE`、PPRにproducer-owned
有効並進伝導熱流束`q_eff`が無いためthermophoresis pointwise replayは`NOT_TESTED`である。保存frameの点判定は
accepted stageを含む連続path certificateにならない。

この結果を受けた二つのeffective-gas revisionは、producerがone-effective-Maxwellian/pseudogasを認証した設定で
reference/sensitivity runを可能にするだけで、species-resolved mixture truthやCOMSOL軌道一致を確立しない。
`lambda/a>=10`と明示`maximum_speed_ratio<=1`の連続path認証はproduction runtimeが所有する。外部auditはその代わりに
ならない。P18-R closeout時点では新しいCOMSOL studyが無く、M3-C1はcandidate preflightまでだった。これは
P18-R成果物の歴史的制約であり、現在のM3-C0b/M3-C1進捗は§9をauthorityとする。

後続M3-C1では不足していたPPR heat-flux primitiveだけを最小COMSOL補足としてexportした。既存v6と座標が完全一致する
13,202 active saved rowで、Waldmann producer-form replayのcomponent-scale normalized residual最大は
`4.4046499933294035e-16`、global relative L2は`1.0992667494449471e-16`となり、frozen saved-state producer-formは
8/8閉じた。この結果はsaved-row formula parityだけであり、P18-Rのphysical applicability、連続path、統合軌道をPASSにしない。

### 2.6 F02 field-production closeout

F02は軌道一致試験ではなく、外部入力から完成fieldを経て既存particle solverへ到達する経路の受入である。
`solver/tools/comsol_adapter/`がprovider固有CSVをcanonical P1へ変換し、F01 builderがpotential、electric field、
plasma primitiveを生成する。比較器と証跡は`solver/tools/vv/comsol/`と`solver/evidence/f02/`だけが所有し、
`chamber_particles`はCOMSOL、`model_dataset`、比較器をimportしない。

比較へ進む前に、source/reference bytesとhashの同一性、strict YAML/CSV、境界ID集合、axis facet、field metadata、
reference unitを機械検査する。新規exportはnode topology IDでmeshと値を結ぶ。F02の既存field CSVのようにnode IDが
ない場合だけ、明示tolerance内で各canonical nodeに厳密に1点、provider点も厳密に1回だけ使う全単射を証明し、
最大対応距離を保存する。補間、丸めbucket、最近傍埋めは比較前処理に使わない。

field normはP1 lumped weight

```text
w_i = sum_K integral_K 2 pi r N_i dA
```

で計算する。vectorは成分二乗和を同じweightで積分する。boundary potentialはsemantic groupごとに集計し、異なる
groupが共有するcorner nodeをexclusive nodeの指標と分離する。生成側のglobal charge balanceも併記する。

F02で検証したのは`same_exported_nodes=TESTED`だけである。単一reference meshなので
`independent_mesh_convergence=NOT_TESTED_SINGLE_REFERENCE_MESH`とし、COMSOL側のQ1/recovery/smoothing、F02のP1化、
closure、離散化の差をこの結果だけで分離しない。fixed-electric 32粒子runは生成fieldを既存RZ solverが消費できる
ことのsmokeであり、COMSOL trajectoryまたはwall/Freeze parityではない。比較差を小さくするためのcore/builder分岐、
tolerance緩和、hidden clampを追加してはならない。

## 3. 比較の不変条件

全比較で、次を満たさなければ数値差の評価へ進んではならない。

### 3.1 ケース同一性

比較対象ごとに次を機械照合する。

- COMSOLバージョン、モデルファイルのハッシュ、解データセット、study/solution番号
- 座標系、軸の向き、長さ・時間・温度・電荷・力の単位
- メッシュ節点、要素型、要素次数、接続順序、領域ID、境界ID、selection
- 物理式、係数、定数、粒子径・質量・電荷初期値、放出時刻・位置・速度
- 有効化した力、電荷モデル、壁条件、乱数条件
- 時間積分法、内部刻み、出力時刻、相対・絶対許容誤差、イベント設定
- 場の評価元、補間・外挿・smoothing/recovery設定、時間補間法

ハッシュまたは設定が異なる場合は「計算誤差」ではなく「入力差」として終了する。自動的な既定値補完、単位推測、欠損列からの代用は禁止する。

### 3.2 粒子対応

粒子は表の行順ではなく、不変な `particle_id` または `release_id` で対応させる。IDが再採番される場合は、放出面ID、面内座標、放出時刻、位置、速度、径、質量、初期電荷、乱数キーを含む一意キーをエクスポート時に生成する。

初期状態が一致しない粒子を軌道誤差の集計に混ぜない。まず初期条件不一致として件数と差を報告し、修正後に軌道比較を行う。

### 3.3 状態とイベントの分離

連続状態を

\[
y(t) = (\mathbf{x},\mathbf{v},Z,\ldots)
\]

とし、canonicalな離散状態を `pending / active / stuck / escaped / failed / held` とする。COMSOL固有の
`freeze/disappear` は外部adapterがraw stateとして保存する。boundary 37/35へ到達する正例microcaseでhit後の位置、
速度、電荷、statusを確認してから、外部manifestで`held/escaped`へ明示的に対応付ける。`held`はmaterial depositionと
別集計にし、COMSOL名やraw status codeをcore lifecycleへ入れない。壁衝突、領域退出、付着、反射、
消失はイベント列として別に保存し、座標欠損をイベント判定の代用にしない。

「数値精度の不足」と「安全上、領域内であることを証明できない」は別の状態である。前者を壁衝突や消失に読み替えない。

## 4. 段階型V&Vフロー

後段の比較は前段に合格した場合だけ実施する。これにより、最終軌道差から原因を推測するのではなく、最初に不一致になる層を特定する。

### Gate 0：エクスポートとインポート

**目的**：入力の意味を変えずに読み込めることを確認する。

**比較項目**：件数、ID、一意性、単位変換、座標基底、要素型・次数、接続、領域・境界selection、出力時刻、非有限値、粒子初期条件、列定義、ハッシュ。

**必須テスト**：

- 節点・要素・境界ファセットを往復し、IDと接続が完全一致する。
- COMSOL上で選んだ既知の領域内点・領域外点・境界上点を同じ分類にする。
- 10点以上のアンカー点で、座標、原始場、単位変換後の値を照合する。
- 初期粒子表は列ごとの絶対差をゼロまたはCSV表現誤差内にする。

**不合格時**：以降の軌道計算を実行しない。補間や許容差で隠さず、アダプターまたはエクスポートを修正する。

### Gate 1：場と幾何クエリ

**目的**：同じ \((\mathbf{x},t)\) で同じ原始場と領域分類を返すことを確認する。

評価点集合は次の和集合にする。

1. 全要素の代表点と、選択した要素の節点・辺/面近傍点
2. COMSOL参照軌道の各保存位置
3. 境界から法線方向の内側・外側に、局所要素寸法に比例して配置した点
4. 強い勾配、support端、軸 \(r=0\)、三角形/四角形接続部
5. 時間依存場では全時間ノットの直前・同時刻・直後

比較するのは速度、密度、温度、圧力、粘度、電位または電場、プラズマ密度・温度などの**原始量**である。Knudsen数、緩和時間、Brownian拡散係数など粒子属性に依存する量は、同じ原始量から両側で再計算し、Gate 2で比較する。

各スカラー場 \(f\) について絶対誤差、相対誤差、RMSE、最大誤差、p50/p90/p99を保存する。相対誤差の分母は場ごとに事前定義した物理スケール \(f_\mathrm{scale}\) を使い、

\[
e_f = \frac{|f_\mathrm{new}-f_\mathrm{ref}|}
{\max(|f_\mathrm{ref}|,f_\mathrm{scale})}
\]

とする。場ごとに任意の微小定数を後付けして合格させない。ベクトル場は成分誤差に加え、ノルム誤差と方向角誤差を記録する。ゼロベクトル近傍では方向角を評価対象外とし、その件数を報告する。

support、領域ID、セルID、最近傍境界IDは値誤差とは別のカテゴリ一致率で評価する。値が有限でもsupport外なら不合格であり、support内でも欠損なら不合格である。

### Gate 2：力・電荷・派生量の凍結状態比較

**目的**：時間積分の影響を除いて物理式を照合する。

COMSOL参照軌道から \((t,\mathbf{x},\mathbf{v},Z,d,m)\) を取り出し、両実装へ同じ状態と同じ原始場を与える。次を個別に比較する。

- 抗力係数、Knudsen数、Reynolds数、緩和時間
- 電気力、抗力、熱泳動力、揚力、DEP、重力・浮力、イオンドラッグなど各力成分
- 合力と \(\mathbf{a}=\sum\mathbf{F}/m\) の閉合
- イオン・電子電流、電荷数変化率 \(dZ/dt\)、飽和・整数化規則
- Brownian力または速度増分を決める係数（乱数値そのものはGate 6）

成分ごとに絶対値、ベクトル方向、合力に対する寄与率を比較する。合力だけが一致して個別力が相殺しているケースを合格にしない。粒子径、質量、温度、電荷のスケーリング試験を追加し、式の単位とべき乗を確認する。

COMSOLのParticle Tracingの力・壁・数値設定の意味は公式の [Particle Tracing Module User's Guide](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/ParticleTracingModuleUsersGuide.pdf) を基準にする。希薄気体抗力は [Drag Force in a Rarefied Flow](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.37.html)、誘電泳動は [Dielectrophoretic Force](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.33.html) のモデル定義と比較する。COMSOLのCharge Accumulation機能とプラズマ粒子のOML充電は同じモデルと仮定せず、式と適用範囲を明示した別ケースで検証する。

P18-Lのcore受入はRZ/no-swirlの
`F=K (omega_phi e_phi) x (u_g-v)`、`K=C_L*pi*rho_g*lambda_g*(drag_diameter/2)^2`に対する
独立oracle、適用域、数値収束までである。保存packageはtrajectory-localなsigned方位vorticityとsolver-step provenanceを
欠くため、exported-P1/native-field比較がFAILしてもCOMSOLのpointwise式parityとsame-field統合軌道一致を
P18-L完了から推測してPASSにしない。後続full-physics common-field診断は限定sliceのsame-field agreementだけを閉じ、
native-field pointwise parityまたはlift modelの物理妥当性を認定しない。

P18-Rでは`epstein_linear_effective_gas_sensitivity_v1`と
`waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1`を別々に判定する。前者のnative保存式replay PASSを
後者へ流用せず、`q_eff`欠損は`NOT_TESTED`のまま残す。両revisionのproducer-certified pseudogas authority、
`maximum_speed_ratio<=1`、`lambda/a>=10`を記録し、保存時刻の全行が閾値内でも連続path PASSとはしない。

### Gate 3：1ステップとRK段階

**目的**：正しい右辺を、正しい時刻・中間状態で評価しているかを確認する。

現在の `model_dataset` の保存済み100 nm・30 ms Case A/Pは固定10 µsの陽的RK4だが、Brownian-onの単一runであり、
内部RK段階も保存していない。Brownian-off candidate v3へ10 µsを強制しても同一step比較にはならないため、この保存runは
Gate 3を判定しない。次式による1-step/RK-stage比較は、両側へ同じ決定論的物理、同じfield表現と値、同じ初期状態を与えた
専用microcaseでだけ行う。同じ刻みを使うのはそこでstage wiringを切り分ける診断であり、candidateのproduction刻みまたは
数値合否を決める条件ではない。

\[
k_1=f(t_n,y_n),\quad
k_2=f(t_n+h/2,y_n+hk_1/2),
\]

\[
k_3=f(t_n+h/2,y_n+hk_2/2),\quad
k_4=f(t_n+h,y_n+hk_3),
\]

\[
y_{n+1}=y_n+\frac{h}{6}(k_1+2k_2+2k_3+k_4).
\]

位置、速度、電荷を連成させる場合、すべての段階で同じ中間状態から場、力、充電電流を再評価する。電荷だけを先に1ステップ進め、その値で軌道を進める逐次分離は、COMSOL側も同じ設定でない限り互換とはみなさない。

比較ケースは少なくとも、均一場、勾配場、強い抗力、強い充電、要素境界横断直前、壁衝突直前を含む。各段階の状態、場、力、電荷変化率と最終増分を記録する。

COMSOLから内部RK段階を直接出力できない場合、段階値を推測して合格判定してはならない。COMSOL Java API等で評価点を明示して右辺を取得するか、同じ凍結状態を評価する専用モデルを作る。どうしても取得できなければ、Gate 3は「未検証」とし、細かい刻みの1ステップ試験と自己収束試験で代替した事実を記録する。

### Gate 4：境界に達する前の決定論的連続軌道

**目的**：場・物理・積分が連続時間で正しいことを確認する。

決定論的parityを判定する時は両側でBrownian力、確率付着、確率反射を無効化し、同じ物理・field・boundary意味を固定して、
最初の境界イベントより前だけを比較する。保存済みBrownian-on COMSOLとBrownian-off candidateはこのgateを共有しない。
比較時刻と数値判定は次の系統を分離する。

- **参照時刻比較**：COMSOLの保存時刻に新ソルバーのdense outputまたは同じ出力刻みで評価した値
- **candidate自己収束**：candidateが自身の安全条件から選んだ`h,h/2,h/4`で、状態とeventを収束させる主要数値gate
- **同一刻み診断**：全意味が一致する専用microcaseに限り、同じ固定刻みの受理step/RK段階を原因切分けに使う任意診断

保存時刻が異なる場合、イベントをまたいで線形補間しない。各粒子のイベントで区切られた連続区間内だけで、高次dense outputまたはその区間の補間精度が確認された方法を使う。参照側の出力が粗く補間誤差を評価できない場合は、COMSOLを再出力する。

粒子 \(i\) の位置誤差は

\[
e_{x,i}(t)=\|\mathbf{x}_{i,\mathrm{new}}(t)-
\mathbf{x}_{i,\mathrm{ref}}(t)\|_2
\]

とし、速度、電荷、各力にも同様の時系列を作る。最低限、次を報告する。

- 各保存時刻のp50/p90/p99/最大位置・速度・電荷誤差
- 各粒子の時間RMSE、最大誤差、初めて閾値を超えた時刻
- 全粒子の誤差包絡線と、状態別・粒径別・初期位置別の分布
- 位置誤差を局所要素寸法、粒子移動距離、チャンバー代表長さでそれぞれ正規化した値
- 速度誤差を局所ガス速度または事前定義した代表速度で正規化した値

閾値超過時刻は原因探索用であり、その閾値単独を製品の合否にしない。candidateの許容値は解析解誤差とcandidate自己収束を
主要根拠として事前に構成する。COMSOL自己収束誤差と場エクスポート誤差は、同じ物理・field・boundary・stochastic意味を
持つ外部比較が成立する場合だけ、その比較用の合成不確かさへ加える。

### Gate 5：境界イベントとイベント後軌道

**目的**：最初の交差位置、壁条件、残り時間処理を比較する。

各粒子についてイベントを発生順に対応させる。イベントレコードには少なくとも次を含める。

- `particle_id`, `event_ordinal`, `event_type`
- 厳密または収束済みのイベント時刻と位置
- COMSOL boundary ID、意味上のboundary selection名、隣接領域ID
- 単位法線、衝突直前・直後の速度と電荷
- 付着確率、使用した一様乱数、判定結果
- 反射モデル、反発係数、接線モデル
- マクロステップの残り時間、その間に発生した追加イベント数

最初に、イベント種類・境界ID・順序の完全一致率と混同行列を評価する。一致したイベントについて時刻差、位置差、法線角度差、前後速度差を評価する。イベント種類または境界が異なった粒子では、それ以後の点ごとの差を「積分精度誤差」として集計しない。最初のイベント分岐を原因として記録し、その後は状態分布・生存率などの集団比較へ切り替える。

接線接触、角・稜線、同時に複数境界へ到達するケースでは、優先規則を仕様化する。軸対称モデルの \(r=0\) は座標の継ぎ目であり、物理壁として扱わない。COMSOLの壁条件と境界精度設定は公式ガイドおよび [Wall Accuracy Order](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.43.html) に照らしてケースごとに記録する。

### Gate 6：確率過程と確率境界

**目的**：単一軌道一致ではなく、正しい確率法則を確認する。

Brownian運動は時間刻みと乱数系列に依存する。COMSOLと同じ乱数生成器、粒子・成分・ステップへの割当て、内部刻みを再現できる場合だけpathwise比較を行う。それ以外は単一粒子軌道を一致判定に使わない。COMSOLのBrownian forceの定義と時間刻み依存性は [Brownian Force](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.08.html) と [Brownian Motion example](https://doc.comsol.com/6.4/doc/com.comsol.help.models.particle.brownian_motion/brownian_motion.html)、乱数引数と`UserDefined` modeは [Sampling Random Number Distributions](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_modeling.05.13.html) を基準に確認する。

現在の保存済み100 nm・30 ms Case A/PはBrownian-on、candidate v3はBrownian-offなのでpathwise gateは不成立である。
保存COMSOLの運動状態はout-of-plane無効のR-Z 2自由度であり、Brownian表の`r/phi/z`列や非零`Fbphi`は第3運動自由度を
意味しない。一方、native fieldを使う各case 1 runだけであり、interface乱数modeは`GenerateUnique`なので、保存parameter
値1/21を実効seedとみなせない。これはpreflight時点で意味一致cohortが0だった理由であり、保存済み履歴を正式cohortへ
昇格させない判断は維持する。後続M3-C2Aでは`UserDefined` seed authorityとcommon-P1 companionを新規実行し、保存履歴とは別の
Case-A/Case-P各32+32 seed cohortで確認的gateをPASSした。Case-Pは元COMSOL `auxq`が意図する電子＋正イオン二電流と
meaningを揃えたsame-form比較であり、species-resolved物理または後続three-current拡張の認定ではない。保存軌道との
時刻対応差は外部characterizationとしてだけ保持する。

統計比較では、複数の独立seedと十分な粒子数を用い、次を評価する。

- 自由拡散の各成分の平均変位、分散、共分散、MSDの時間勾配
- 位置・速度・電荷分布のWasserstein距離またはECDF差
- 境界別到達確率、付着率、消失率、生存関数
- 初回到達時間分布と信頼区間
- seed間分散、粒子数増加に対する信頼区間幅の縮小

chamber anchorではseedを独立cluster、同一seed内の287粒子をcluster内sampleとして扱う。粒子行bootstrapは使わず、
whole-seed resamplingで区間を作る。同じ整数seedを両solverへ渡してもRNG割当てが異なる限りpaired pathとは扱わない。
まず本比較と分離したpilot seedで両実装それぞれのstep/tolerance系列を評価し、同じ固定刻みを要求せず、共通出力時刻の
observableに対するfine-pair数値不確かさとMonte Carlo confidence budgetから
同値幅を結果を見る前に固定する。有意差が見つからないことだけを同値性の根拠にしない。

付着率 \(p\) の試験では、観測付着数を二項分布の信頼区間で評価する。期待値と数個のサンプルの差だけで不合格にしない。分布距離と信頼区間の判定条件は結果を見る前に固定する。

### Gate 7：刻み幅・メッシュ・出力収束

**目的**：一つの離散化設定への偶然の一致を排除する。

candidateは各決定論ケースで少なくとも \(h,h/2,h/4\) を計算し、境界前の連続状態について観測収束次数

\[
p_\mathrm{obs}=\log_2\frac{\|y_h-y_{h/2}\|}
{\|y_{h/2}-y_{h/4}\|}
\]

を求める。これがcandidateの主要な数値gateであり、刻み系列はCOMSOLの10 µsへ合わせずcandidateのstability・applicability
条件から選ぶ。イベントを含む場合はイベントidentity、時刻・位置・terminal stateも別に収束させる。衝突後の最終位置だけで
次数を求めない。場についてはcandidateのmesh/field表現の収束を先に評価し、COMSOL側のメッシュ系列、native補間、規則格子
cacheの系列は外部producer固有の別statusにする。

出力時刻を細かくしただけで内部解が変わっていないかも確認する。保存間隔、内部時間刻み、場メッシュ解像度を混同しない。

candidate合否はcandidateの自己収束と独立referenceで決める。COMSOLとの差を合否に使えるのは、式、field値と補間、boundary、
stochastic意味が同じで、両側の自己収束を別々に確認した場合だけである。その場合も最細解の推定離散化誤差区間または
事前設定した合成不確かさで判定し、参照解自体の未収束をcandidateの不具合としない。

### Gate 8：性能とスケーラビリティ

**目的**：物理的に同等な仕事量で速度と資源効率を比較する。

性能比較はGate 0–7を満たした設定だけで行う。COMSOLと新ソルバーで、粒子数、時間範囲、内部精度、力、壁条件、場、出力頻度、保存列、浮動小数点精度を揃える。出力を減らした実行と全軌道を書き出すCOMSOLを比較して「高速」と主張しない。

10^4、10^5、10^6粒子について、次を別々に測る。

- cold start（JITコンパイル・読み込み込み）とwarm run
- 入力変換、場サンプリング、力評価、積分、境界探索、圧縮、出力の時間内訳
- 総wall time、particle-step/s、ピークRSS、GPU使用量、出力量
- CPUコア数による強スケーリング、チャンクサイズ感度、衝突率感度
- p50だけでなく複数反復の最小/中央値/p95

測定環境、CPU/GPU、メモリ、OS、Python・Numba/JAX・COMSOLバージョン、スレッド設定を成果物に含める。速度改善は同じ精度ゲートを維持した比率で示す。

P14-Uの代表用途gateはCOMSOL runを必須にせず、同じcanonical field/geometryに対する自己収束と小粒子高精度referenceを
先に使って完了した。surface release、非一様場、材料wall、一般曲線event、多数macro stepを同一caseで有効にし、
固定空間mesh上の`h,h/2,h/4`と、`nx=ny`固定aspectのregular/P1/Q1 mesh系列を分離する。mesh系列はlayout別fine
reference、物理targetとnormalized facet clearanceを使い、raw facet IDを異なるmesh間のidentityにしない。
global extremaと、source・exact localized hitを含むdense path範囲の比、refinement深度、candidate数、failure reason、
solver plan/RSSも同じ成果物へ保存する。none/sampleは10k/100k/1Mを各3 fresh processで測り、共通科学payloadに加え
sample probe自体のdigestも反復間で一致させる。1M profileは時間測定と分離する。
solver内thread scalingはP14-Pで不採用と確定したため測らず、独立caseのprocess並列はcore外の運用として分離する。

P14履歴の20-worker値はsynthetic baselineである。regular 100k/1Mでは正のspeedupを示す一方、P1/Q1 10kでは
小さく、event-heavyでは遅化した。P14-Uは単一直列engineの代表用途gateであってparallel speedup証拠ではないため、
完了後も「並列化により主用途が高速」とは判定しない。

## 5. 許容値と合否ゲートの決め方

許容値を全量共通の絶対値一つにしない。また、比較結果を見た後で広げない。各ケースについて次の誤差源を先に推定する。

1. COMSOL時間離散化誤差
2. COMSOL空間離散化・場評価誤差
3. エクスポート桁数・補間誤差
4. 新ソルバー時間離散化誤差
5. 新ソルバー場補間・イベント位置決め誤差
6. 確率ケースでは標本誤差

決定論的な量 \(q\) の許容帯は、これらの独立性を正当化できる場合は二乗和平方根、できない場合は保守的な和で構成する。最低値を固定の「小さい数」で与えず、単位と代表スケールを伴う。許容帯、評価時刻、対象粒子、除外規則、必要な分位点をベースライン登録時に固定する。

各Gateの結果は `PASS / FAIL / NOT_TESTED / NOT_APPLICABLE` の四値とする。データ不足をPASSにしない。
PASSした座標系、物理モデル、境界条件、メッシュ型、時間積分法は、製品のsupport matrixではなく
**COMSOL外部V&V coverage table**として公開する。製品supportは解析解、scenario、性能試験から別に決める。

推奨する最低合格条件は次の構造である。数値はケース別の収束試験から埋める。

| Gate | 必須判定 |
|---|---|
| 0 | ID・接続・selection・単位・初期条件が完全一致 |
| 1 | support/領域分類が完全一致し、各場誤差が事前許容帯内 |
| 2 | 全力成分、派生量、電荷右辺が事前許容帯内で閉合も成立 |
| 3 | 各RK段階または代替1ステップ試験が許容帯内 |
| 4 | イベント前の時系列分位点、最大誤差、収束率が全て合格 |
| 5 | イベント型・境界・順序が一致し、時刻・位置・前後速度が許容帯内 |
| 6 | 規定seed数・粒子数で分布と到達統計が信頼区間基準内 |
| 7 | 期待収束次数または参照不確かさ包絡を満たす |
| 8 | Gate 0–7を維持した条件で性能目標を満たす |

## 6. 不一致原因の診断順序

最終軌道の形から原因を推測せず、「最初に差が現れた時刻」と「最初にFAILになったGate」を使う。

```text
ケースID・ハッシュ・件数・初期行が違うか
  ├─ はい → export/import、単位、座標、列対応、粒子対応
  └─ いいえ
      同じ (x,t) の領域分類・原始場が違うか
        ├─ はい → メッシュ接続、要素次数、basis、補間、support、時間slice
        └─ いいえ
            凍結状態の派生量・個別力・dZ/dtが違うか
              ├─ はい → 物理式、係数、径/質量、電荷、単位、符号
              └─ いいえ
                  RK段階または1ステップが違うか
                    ├─ k1から違う → 右辺評価または初期状態
                    ├─ k2以降で違う → 段階時刻、中間状態、連成順序、場再評価
                    ├─ 終端だけ違う → 重み、更新式、精度型
                    └─ 一致
                        最初の差は境界近傍か
                          ├─ はい → first-hit、法線、ID、角/接線規則、残り時間
                          ├─ いいえ、確率ON時のみ → RNG割当て、dt、統計検定
                          └─ いいえ → 累積丸め、非滑らかな場、未記録のCOMSOL設定
```

追加の兆候と確認点は次のとおりである。

| 兆候 | 最初に確認する項目 |
|---|---|
| \(t=0\) から一定倍率の差 | 単位、半径/直径、質量の再構成、2πr重み |
| 要素境界ごとに力が跳ぶ | 要素次数、節点順序、smoothing/recovery、誤ったDelaunay再メッシュ |
| 速度だけ先にずれる | 抗力係数、相対速度、緩和時間、段階内の速度評価 |
| 電荷が先にずれ、その後位置がずれる | 電流式、Z/q変換、電荷・電気力の段階連成 |
| 壁直前まで一致し、直後に分岐 | first-hit、法線方向、壁モデル、イベント後の残り時間 |
| 消失後だけ比較粒子数が減る | 欠損除外による生存者バイアス。イベント列と生存関数へ切替 |
| 刻みを半分にしても誤差が減らない | 場/物理/幾何差、参照誤差、イベント対応違い |
| CPU/GPUで確率結果だけ変わる | RNGキー、演算順序依存、fastmath、浮動小数点精度 |

差異レポートには、最初の不一致粒子、時刻、Gate、要素ID、境界ID、状態、場、各力、RK段階を一つの診断パケットとして保存する。大量の独立した診断値を並べるより、この因果鎖を一件ずつ追える形式を優先する。

## 7. COMSOLから追加で出力すべきもの

現在のデータセットを完全な参照系にするには、次を再出力する。

### 7.1 ケース・ソルバー・モデル設定

- COMSOL完全バージョン、MPHハッシュ、study/solution/dataset識別子
- 全パラメータの式、評価値、単位、依存関係
- 粒子追跡ソルバー、時間積分法、内部刻み、許容誤差、壁精度次数、ステップ当たり最大壁イベント数
- 各力・充電・壁ノードの有効/無効、selection、式、既定値を含む設定表
- 使用変数のCOMSOL式名、表示名、単位、座標basis
- 乱数seedと、COMSOLが公開できる範囲の乱数設定

COMSOLのデータセット評価ではsmoothingやrecovery等により節点値と任意点評価が異なることがあるため、評価設定もmanifestへ含める。[COMSOL Results API](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_ref_results.37.224.html) と [Mesh API](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_api_mesh.49.047.html) を参照し、エクスポート処理そのものをバージョン管理する。

### 7.2 幾何・メッシュ

- 節点IDと座標
- 要素ID、型、次数、全節点接続とCOMSOLの局所節点順序
- 要素の領域ID、境界ファセットID、隣接要素
- boundary ID、意味上のselection名、向き、隣接領域
- 粒子が許可された領域selection
- P2以上の要素では中間節点を含む完全な接続

規則格子場のNaN輪郭から壁を再構築しない。二次要素を頂点だけの一次要素として扱わない。四角形を黙って三角形へ分割しない。

### 7.3 場

- ガス速度、密度、圧力、温度、粘度、平均自由行程などの原始量
- 電位と電場成分、その電場がどのsolution/datasetから得られたか
- プラズマ密度・温度・イオン速度等、充電・イオンドラッグに必要な原始量
- 時間依存場では全ノット時刻、時間補間規則、範囲外方針
- 各量のsupport領域と非有限値の意味
- mesh-native参照値と、高速キャッシュ用再メッシュ値を別成果物として保存

粒子径に依存するKn、応答時間、拡散係数は背景場へ固定せず、派生式とともにテスト用期待値だけを別表で出力する。

### 7.4 軌道とイベント

- 全保存時刻の粒子ID、連続状態、離散状態、領域/要素ID
- 速度、電荷数、電荷、各力、合力、加速度、充電電流または \(dZ/dt\)
- 最初の壁イベントと以降の全イベントの厳密時刻・位置・境界・法線・前後状態・結果
- 消失後に座標を出さない場合も、終端イベントは必ず一行保存
- 外部比較マイクロケースでは各受理内部ステップと、可能ならRK段階値
- 確率ケースでは使用乱数または再現に必要なカウンタ情報

保存時刻だけから衝突時刻を逆推定しない。イベント出力が得られない場合は、衝突付近を十分細かく再計算し、時刻刻み系列でイベント位置の収束を示す。

## 8. 段階的なマイクロケース

大規模ICP/CCPモデルだけで原因を調べない。ただし、将来機能まで初期の必須suiteへ入れて
test基盤を肥大化させない。

### 8.1 外部V&V用の初期必須 V01～V06

| ID | 検証対象 | 最小設定 | 主な正解 |
|---|---|---|---|
| V01 | field補間・support・要素検索 | triangle＋歪みquad、一次scalar/vector場 | 解析式 |
| V02 | 運動・積分 | ballistic、一定加速度、線形drag | 厳密軌道 |
| V03 | surface release | RZ直線境界、法線・初速・発生時刻 | 指定値、`2πr ds`分布 |
| V04 | wall event | stick、escape、specular、grazing、multiple hit | hit時刻・点・反射速度 |
| V05 | charge coupling | 一様E、線形charge relaxation | `q(t),v(t),x(t)`厳密解 |
| V06 | Brownian OU | 一様T、線形drag、多数replica | 平均、分散、MSD |

V01～V05はBrownian offで先に完成させる。V04の確率付着は同一のCOMSOL乱数列を要求せず、
二項分布と反射後分布で評価する。V06もpathwise一致ではなく統計量で評価する。この六件が
初期外部V&Vの全必須caseである。

### 8.2 対応stageで追加するcatalog

| 追加時期 | case | 主な判定 |
|---|---|---|
| Stage 2A/2B | axisymmetric-field/Cartesian3D、確率壁 | 粒子自由度、分布、first-passage |
| P15-F/F01/F02（完了） | collisionless Barnes ion drag、reduced electrostatic builder/provider adapter | impact-parameter oracle、F01解析・nested-mesh収束、F02同一node記述比較。F02独立mesh収束は未検証 |
| M3-C0 / Stage 3 | comparison reference lock | 式・parameter・derived-field意味、ion-drag以外のvariant同一性、admissible刻み系列、seed cohort。力なしnormal-impactのFreeze/Disappear正例は完了 |
| P18-C/I/D/L/R完了 | reference charge、二つのion drag、DEP、RZ lift、effective-gas drag/thermophoresis closure | formula parity、独立oracle、model applicability、compiled parity、無効時同一性。P18-R closeout時の判断を維持し、後続M3-C1の最小PPR補足でfrozen saved-state producer-form replayだけを8/8閉じた |
| P19-L（完了） | localized continuous-path applicability certificate | integrator-owned dense RK4 pathとlocal cell rangeで元のfixed-step endpointを変えずに証明する。actual `model_applicability`と`indeterminate_applicability_certificate`を分離した。M3-C1 event v14はsupport/applicability/reintegration authorityをglobalに維持したまま`rk4_dense` event queryだけcurrent dense boundへ狭めた。event v15は同じauthorityと意味論を維持してroundoff認証を強化し、現行event v16はRK4位置Bernstein制御点のfacet half-space内包を証明できる候補だけをclearする |
| M3-C1（Case-A 100 nm anchor完了） | 12 Brownian-off deterministic companion | frozen saved-state 8/8、cross-representation 6/6 FAIL、common-P1 pre-event 9/9 PASS、field表現差局在化、first material-stick 20/20＋prefix 9/9 PASSを別statusで完了。v14 solver-only 3刻みは自己収束PASS、全3量`ORDER_EVALUATED`。限定anchorをnative-field parity・物理妥当性へ昇格しない |
| M3-C1（100 nm・30 ms candidate-first characterization完了） | Case A/P Brownian-off candidate v3 | candidate固有の`h,h/2,h/4`自己収束を主要gateとして両case PASS。保存COMSOLはfixed RK4 10 us・Brownian-on・native-fieldの単一runなので`CHARACTERIZED`だけとし、刻み模倣やpathwise合否に使わない |
| P18-H（完了） | producer-neutral terminal hold | 解析・resume・Brownianを含む公開回帰をPASS。既存hash固定Freeze referenceへのcandidateはCOMSOL再実行なしで15/15 PASS。力なしnormal-impact意味だけを認定し、full physicsやgrazing/cornerへ一般化しない |
| B03（完了） | charged/forced RZ Brownian composition | macro-root stochastic exponential-midpointによるRZ投影、fixed/continuous charge、線形Epstein、全決定論寄与の単一proposalを解析解・ensemble・identityで検証。一般SDEのstrong/weak 2次や等方3-Dは非主張 |
| M3-C2 | charged/forced RZ Brownian ensemble | common-P1 Case-A/Case-P 100 nm final完了。Case-Pは20 us、32+32独立seed、各287粒子×121 frame / 30 ms。83区分R-Z/fate TV `0.010670731707317093`、同時上限`0.13119968456545308 < 0.15`でPASS。終端gateはevent 0で非情報的。元Case-P `auxq`どおりの二電流same-form結果で、three-current、species-resolved物理、pathwise RNG、boundary parity、普遍的COMSOL同等性は非主張 |
| P21 / M3-C3（`CLOSED_ACCEPTED_WITH_LIMITATIONS`） | aggregate three-current / critical 2-D closure | priority 1 productionはcatalog v17 / runtime v19 / tile v18と標準品質gateで完了。boundary priorityも一つの3粒子microcaseで132/132 PASS、最大solver間位置差`1.61339e-17 m`。grazing/corner、multiple/probabilistic、force/native-fieldは非主張。priority 3はfull canonical負イオン5 primitive authorityを全1987節点へ生成して入力blockerを解消。priority 4は共有Z0とcommon-P1、Brownian-off、100 nm・287粒子・30 msの比較でcandidate/明示drag COMSOLの各3刻み収束、全frameのlifecycle/finite mask exact、共通有限stateの`r,z,Z`、共通active stateの`v`、141 event/fate identityをすべてPASS。元二電流anchorは不変で、普遍的COMSOL同等性と物理model validationは非主張 |
| Stage 4A | 時間線形場と不連続切替 | 時間補間、step split |
| Stage 4B | 単一四面体、3D surface、平面壁 | 3D補間、segment-triangle、面積分布 |
| 高次要素対応時 | P2以上の最小mesh | node/DOF順、形状関数、収束 |

各caseは一つの主因だけを変え、対応機能を実装するstageで初めて必須化する。解析解があるものは
COMSOL一致より解析解収束を優先し、複合chamber modelはmicrocase合格後の統合回帰に限定する。

## 9. `model_dataset` に対する実行順序

Stage 3の比較は次の順で固定する。

M3-C0bではCase-A 100 nmのCOMSOL再実行を段階化した。30 msの10/5/2.5 us v3は非漸近で`CHARACTERIZED`に留め、
全粒子がactiveな0--450 usだけを細分した。v4は電荷次数の事前gateを満たさなかった。v5は当時「全gate PASS」と
記録したが、位置relative L2を絶対RZ座標で正規化して原点依存だったため、そのPASS/admissionを`INVALIDATED`とし、
v5は履歴上の`CHARACTERIZED`に留める。未見の0.15625 usを追加した逐次確認v6は0.625/0.3125/0.15625 us系列で
pre-eventの運用上の刻み選択だけをPASSした。M3-C1 frozen saved-state producer-form replayは最小PPR補足後8/8を閉じた。
このM3-C0b v3はBrownian-offにしたdeterministic companionであり、保存済みBrownian-on・fixed RK4 10 usの
100 nm Case A/P原軌道やBrownian-off candidate v3とは別成果物である。
P19-Lは初期sampleが適用域内でもglobal continuous-applicability enclosureだけでは証明できなかった時刻0 blockerを、
fixed-step endpointを変えない局所certificateで解消した。exported-P1/native-field比較は完了し、cross-representation
6 gateがすべてFAILした。事前登録したfull-physics exact-connectivity common-field COMSOL診断はその後完了し、
絶対値とrelative L2の9 gateをすべてPASSした。これは同じcanonical P1場を使うCase-A 100 nm、Brownian-off、0--450 us、
event前のsame-field agreementだけを認定する。続く力なしboundary semantics probeでFreeze/DisappearのCOMSOL側意味を
分離した。その後、field表現差局在化とcommon-P1 first wafer-stick eventも閉じ、event v14でsolver-only 3刻みを再実行した。
P18-HとB03 coreは完了した。後続の100 nm・30 ms candidate v3はCase A/Pとも自己収束をPASSした。現在順序は
charge-stable couplingと長時間run向けdurable I/O cadenceを先に閉じ、その後に必要性と意味一致を確認した外部COMSOL比較、
stochastic caseでは独立seed ensembleへ進む。
保存済みM3-C1 compact evidence/evaluatorはevent v14の履歴として変更しない。以下のGate 0入力監査、field/geometry確認は
各該当gateの前提である。

B03 coreの受入はCOMSOL再実行に依存せず、定数係数OUのmean/full covariance、noise-offの2次収束、
manufactured charge-electricのweak mean観測次数`>=0.9`、RZ axis restart、tree-depth first-passage、
slab/output/checkpoint identityで閉じる。一般state-dependent SDEのstrong orderやweak 2次を主張しない。
これらを完了してB03をcloseした。将来COMSOLのstochastic比較を再実行する場合は、charge-stable couplingとdurable I/O
cadenceを先に閉じ、同じ物理・field・確率意味を持つ独立seed cohortまたは事前登録した同等confidence budgetを作る。
単一seed軌道をB03 coreの正解にしない。

現行100 nm・30 ms characterizationでは、candidate v3のCase A/PはいずれもBrownian-offの独立`h,h/2,h/4`系列で
state自己収束をPASSし、Case Aはevent自己収束もPASSした。保存COMSOL Case A/Pはmanual explicit fixed RK4 10 us、
Brownian-on、native finite-element fieldの単一runで、candidateと確率・field意味が一致しない。そのため保存軌道は
`CHARACTERIZED`な外部記述であり、candidateへ10 usを強制する理由にも、particlewise parityの合否にもならない。

1. M3-C0で監査済みMPH copy、COMSOL/exporter hash、12 packageのmanifest、式、parameter、粒子ID、121時刻を
   Gate 0で固定する。二つのion-drag variantは、ion drag以外の式、field、initial/boundary/solver設定が同一であることを
   machine gateにし、既知のlift構文差を残したrunへion-drag差を帰属しない。
2. 幾何の7,732一意頂点と接続を正規化し、CSV重複行を節点IDへ全単射で対応付ける。推測対応は不合格とする。
3. 粒子領域3を要素・境界から構築し、既知の放出287点を領域内として確認する。DEPの
   `grad(mean_E_squared)`とliftの方位vorticityはsolution、DC/RF平均、回復法、topology ID、単位を伴って再出力する。
4. 10 usを含む候補刻みをdrag/charge gateで先にcharacterizeし、比較に使う最大`h`から`h,h/2,h/4`を
   結果を見る前に決める。比較のために`h/tau`、`h L_Z`、model applicabilityを緩めない。
   v5は原点依存metricのため採用せず、v6の0.625/0.3125/0.15625 usをCase-A 100 nm pre-eventの運用上の
   step-selection系列としてだけ受理する。観測次数は約1であり、RK4の形式4次または普遍的な物理精度を主張しない。
5. M3-C1の最初のsliceとして、frozen saved-state producer-form replayは最小PPR熱泳動補足export後に8/8を閉じた。
   13,202 active saved rowは既存v6座標と完全一致し、Waldmann replay residualはcomponent-scale normalized最大
   `4.4046499933294035e-16`、global relative L2 `1.0992667494449471e-16`である。saved-row式一致を
   continuous-path applicabilityまたはtrajectory agreementへ読み替えない。
6. integrated candidateはCOMSOL native finite-element fieldではなく、export済みnode値とexact connectivityを使うP1である。
   完了したP19-Lは元のfixed-step RK4 endpointを変えないintegrator-owned dense pathとcertificate-only部分区間制限を実装し、
   actual `model_applicability`と`indeterminate_applicability_certificate`を分ける。gateを緩めたrunを比較へ採用しない。
7. P19-L後、candidate prepare設定、canonical input、run manifest、result、source exportのhashを一つのchainへ固定して
   `exported-P1 candidate vs COMSOL native-field reference`を比較した。candidateを`native-field`と呼ばない。3 candidate runは
   287粒子×46 frame、event/failureなしで完了し、COMSOL reference自己収束はPASSした。cross-representationのfine差は位置
   RMS/max `3.437485727086595e-4/2.660729292906687e-3 m`、速度
   `1.919910730173562/11.668961060303342 m/s`、電荷`17.58471490428266/157.15163693423065 e`で、6 gateすべてFAILした。
8. 当時のevent v13 candidateは指定した0.625/0.3125/0.15625 usのmacro stepを使ったが、certificate refinement後のaccepted pieceは全runで
   `8,450,307`、最深nominal leaf幅も`2.44140625 ns`だった。fine-pair差は設定済みfloat64 representation-scale floor未満の
   production-output stabilityである。この結果は履歴上有効だが、artificial event subdivisionが3 runを同じaccepted-piece countへ
   駆動したため、RK4次数、floor未満の誤差、三つの独立な実効grid収束を意味しない。
9. 6 gate FAILで事前登録した条件が成立したため、full-physics exact-connectivity common-field COMSOL診断を別成果物・別statusで
   実行した。軌道差を読む前に、candidate/reference自己収束、t=0初期状態の4096-ULP基準、位置・速度・電荷それぞれの
   RMS/max/relative L2上限を固定した。t=0は287粒子×5成分の1,435値を全PASSした。
10. 初回common-field fine軌道差は当時の登録9 gateを全PASSした。現行eval_v3 authorityでは、位置RMS/max/relative L2が
   `4.06807283316903e-13/1.2035778717837921e-12 m/2.0316001855367802e-10`、速度が
   `2.10847522277453e-9/3.844306466969233e-9 m/s/1.9630813983926987e-10`、電荷が
   `1.1818881019499895e-7/2.3758877887303242e-7 e/4.670220207738041e-10`で、事前登録9 gateを全PASSした。
   登録budget SHA-256は`ab3713fb54aba02f7a208920e377b1fb90daaba3e129a04e57da8b4b3048b7b5`、comparison resultは
   `9fa50b0481ea662c07a64ba258977d0004831a50636c10d9ba5c96f1d94f90bf`である。
   旧M3-V common-field runnerは固定電荷・共通3力のreduced sliceなので、この動的電荷と全決定論力の結果へ読み替えない。
   cross-representation authorityは
   [`solver/evidence/m3c1/case_a_100nm_pre_event_v6/`](solver/evidence/m3c1/case_a_100nm_pre_event_v6/)、common-field authorityは
   [`solver/evidence/m3c1/case_a_100nm_common_p1_v1/`](solver/evidence/m3c1/case_a_100nm_common_p1_v1/)である。
11. common-field PASSはsame-field integration/wiringの限定証拠であり、native-field等価性、物理modelの妥当性、境界、Brownian、
   30 ms、他case・粒径・variant、普遍的COMSOL同精度を認定しない。
12. 力なしの解析的normal-impact正例を先行実行した。現行v2は2 scenario×3刻みのexact 6 configuration receiptを
    COMSOL process logから照合し、欠落・重複・形式不正・設定差を科学評価前にfail-closedで拒否する。boundary 37 Freezeと
    35 Disappearを10/5/2.5 usで評価し、eventは73 us、最初のterminal保存時刻は75 usだった。全active frameを
    `x=x0+v0*t` / `v=v0`でgateし、全run最大の位置/速度誤差は`4.726604209672303e-16 m` /
    `1.7763568394002505e-15 m/s`で、各`1e-12`上限を満たした。Freezeはstatus 2でhit点R-Zと衝突前velocityを保持、
    Disappearはstatus 4で位置・速度をNaNにした。step間event-time spreadは
    `3.07371315899641e-17/2.71050543121376e-20 s`、56 gate PASS、FAIL 0、velocity記述6件である。原本MPH hashは
    不変でcore変更はない。これは隔離したCOMSOL意味だけを認定し、production solver parity、grazing/corner、full physicsを
    認定しない。v1は科学的に無効ではなく、configuration receiptとactive-flight oracleの監査強度が不足した履歴成果物として
    v2にsupersedeされた。current compact authorityは
    [`solver/evidence/m3c0/boundary_semantics_v2/`](solver/evidence/m3c0/boundary_semantics_v2/)である。
13. common-P1 Case-A 100 nmを最初の自然なwafer stickまで延長した。event v5 evaluationはmaterial 20/20と
    pre-event prefix 9/9をPASSした。event時刻、hit位置、terminal chargeの絶対差は
    `2.157542807607049e-13 s / 2.683964162031316e-14 m / 4.7283812421028415e-09 e`である。terminal velocityは
    COMSOL Freezeとsolver stickの保存意味が異なるためcross-solver gateにしない。scopeはCase-A 100 nm、common canonical
    exact-connectivity P1、Brownian off、287粒子、first wafer stickまでに限る。
14. v13のglobal absolute enclosureをevent BVH boundにも流用したことがartificial subdivisionの原因だった。
    450 us時点で`7,623,460 / 7,792,306` refinement（`97.8331703092769%`）が発生済みで、最終
    query/refinement/accepted/depthは`16,427,517 / 7,792,306 / 8,635,211 / 16`だった。event v14はglobal supportを独立に
    証明済みのvalid `rk4_dense` rowのqueryだけcurrent dense Bernstein boundへ限定し、global boundはsupport、global-first applicability、shortened-RK4
    reintegrationのauthorityとして維持する。invalid dense boundはglobal boundを保持してsplitし、global supportを証明できない
    caseはevent queryもglobal fallbackとする。v14は`842,927 / 11 / 842,916 / 11`、failure 0だった。
15. operator-observed shell wall-timeは同じlocal環境で約`14m13s`（約853 s）から`36.5s`（約`23.4x`）へ短縮した。
    これはmachine-local、概算、非gatingで、solver-reported runtimeやmanifest authorityではない。v14はsolverのevent BVH
    broad phaseだけを変更し、common-P1 COMSOL input/reference/source MPHはhash-lock済みで不変なのでCOMSOLを再実行せず、
    既存referenceに対してcandidate comparisonを再計算した。
16. v14 solver-only 0.625/0.3125/0.15625 us再実行はquery/refinement/acceptedが順に
    `206,927/0/206,927`、`413,567/0/413,567`、`826,847/0/826,847`だった。自己収束はPASSし、位置・速度・電荷の
    RMS観測次数は`2.029875353701904 / 2.0816971911764033 / 2.044084026475049`、fine-pair relative L2は
    `6.099791486063973e-8 / 8.321356016032579e-8 / 1.3796067752988052e-8`、全量`ORDER_EVALUATED`である。
    `candidate_self_convergence.json` SHA-256は`ca0ef84cda1f1ddfc2b5eecc357eced4cb3648b48d582de10195c825bbbf38c7`。
    観測次数約2はpiecewise P1場とmesh crossingを含むこの実caseの経験値で、RK4の形式4次を証明も否定もしない。
17. field表現差の局在化も別authorityで閉じた。material-eventとfield-localizationのPASSをnative-field parity、物理妥当性、
    Brownian、30 ms、他case/size/variant、普遍的COMSOL精度へ昇格しない。現行compact authorityは
    [`solver/evidence/m3c1/case_a_100nm_material_event_v1/`](solver/evidence/m3c1/case_a_100nm_material_event_v1/)である。
18. P18-Hは解析・resume・Brownianを含む公開回帰と既存Freeze candidate 15/15を閉じた。B03 coreも解析・manufactured・
    identity gateで完了した。どちらもCOMSOLを再実行していない。P18-H compact authorityは
    [`solver/evidence/p18h/hold_freeze_v1/`](solver/evidence/p18h/hold_freeze_v1/)とする。
19. 100 nm・30 ms candidate v3はCase A/Pについてcandidate固有の刻み系列で自己収束を先に判定し、両caseをPASSした。
    保存COMSOLとのaligned history差は、Brownian-on/offとnative-field/exported-P1差を含むため外部characterizationに限定する。
    同じ固定刻みをcandidateへ強制せず、保存COMSOLをcoreのgolden gateにしない。
20. scalar continuous-charge stable updateとdeterministic work-scaled cadenceを完了し、保存package/seed preflightから
    common-P1 Case-A 100 nmの実行契約、runner validation、独立pilotへ進めた。V2 pilotの4 seedを95%推論へ使う解釈は無効化し、
    V3ではconfiguration screeningだけに限定した。
21. pilotと非重複のCOMSOL/candidate各32 seed、各287粒子×121時刻を実行した。Brownian-onのGate 6はsingle-seed pathではなく、
    seed×固定source populationに対する4終端人口曲線の同時Hoeffding同等性とした。最大差0.5989%、95%同時上限3.877%で
    事前登録5%幅をPASSした。連続moment/quantile/occupancyとR-Z overlayは説明用の非判定証拠である。
22. 同じ契約をCase-P 100 nmへ一軸展開し、20 us、32+32独立seed、各287粒子×121 frame / 30 msを完了した。
    登録済み83区分R-Z/fate gateは最大empirical TV `0.010670731707317093`、同時上限`0.13119968456545308 < 0.15`でPASSした。
    終端gateはevent 0で非情報的である。これは元COMSOL `auxq`どおりの二電流same-form結果で、後続three-currentまたは
    species-resolvedな帯電物理を認定しない。
23. accepted candidate seed `319032/319047/319063`の287粒子owner discoveryは、受理済み科学payload・work・case identity・
    revisionを完全一致させて完了した。3 seedとも支配ownerは`integrators`（自己時間比42.58--42.86%）だったが、
    事前登録済みbounded ownerではないため`optimization_authorized=false`で本体変更はない。この時点でM3-C2A anchorは
    `CLOSED_ACCEPTED_WITH_LIMITATIONS`とする。追加COMSOL、追加seed、10,000粒子以上のscale、残るpackage、process並列は
    この科学benchmarkの完了条件ではない。性能は利用SLAを先に定義した独立work packageでのみ評価する。
24. 後続の明示指示によるbounded performance follow-upでは、外部比較を再開せず、candidateのchord batch一箇所だけを
    compiled ownerへ統合した。同じaccepted 3 seedの科学payload/work/revisionは完全一致し、end-to-end中央値は12.29%短縮した。
    これはCOMSOL速度比または10,000粒子scaleの認定ではなく、追加ownerへ進まず終了する。
25. P21/M3-C3 priority 2では、一つの2D axisymmetric rectangleと3粒子だけを使い、surface contact departure、
    specular reflection＋同step残時間、R-Z axis passageをCOMSOL/public API/解析解で比較した。3刻みの132 gateは全PASS、
    最大solver間位置差は`1.61339e-17 m`である。grazing/corner、multiple/probabilistic、力・native fieldは非主張とし、
    aggregate three-current production受入は別途621 tests＋品質gateで完了した。
26. priority 3はproducer側のone-sided domain-3 cacheからfull canonical負イオン5 primitive authorityを全1987節点へ生成し、
    入力blockerを解消した。source MPHはhash不変で、座標nudge、欠損補完、別domain fallbackはない。
27. priority 4は共有three-current `Z0`を両solverへ渡したcommon-P1、Brownian-off、100 nm・287粒子・30 ms比較を
    独立coverageとして一回実行し、
    candidate/明示drag COMSOLの3刻み収束、全frameのlifecycle/finite mask exact、共通有限stateの`r,z,Z`、
    共通active stateの`v`、141 event/fate identityをすべてPASSした。元Case-P二電流anchorと
    完了済みP21/2D benchmarkは再開せず、普遍的COMSOL同等性や物理model validationは主張しない。

M3-C0a offline lockは上記1のうち、MPH identity、現12 package、式・parameter、field名/単位、variant差分、
候補刻み、32 seed cohortを固定済みである。ただし現履歴はBrownian-onで、Case Pのvariantにはlift式差があり、
derived-field provenance、accepted-step/RK stageは不足する。Freeze/Disappear正例は後続probeで閉じたが、
このためGate 0全体は未完了であり、
`PARTIAL_RERUN_REQUIRED`をP18のformula source確認とCOMSOL再実行準備にだけ使う。軌道一致や物理認定へ昇格しない。

M3-C0b v5のraw artifactは上記履歴を再現するが、旧position relative L2は
\(\lVert x_h-x_{h/2}\rVert_2/\lVert x_{h/2}\rVert_2\)で絶対RZ原点に依存していた。そのため旧compact
PASS/admissionは`INVALIDATED`、v5の科学的statusは`CHARACTERIZED`であり、current step authorityではない。

M3-C0b v6はCase-A 100 nm theoryのBrownian-off、287粒子×46 frame、0--450 usだけを対象とする。
位置は粒子ごとの初期位置を引いた変位で正規化し、0.625/0.3125/0.15625 usの各runで全13,202 recordがactiveだった。
0.3125→0.15625 usの変位・速度・電荷relative L2は
`3.102727085428027e-5/3.92483251084038e-5/1.3511393490811483e-6`、観測次数は
`0.9041136/0.944312/1.123838`である。位置上限`1e-4`はv5 overlapを開示して設定し、未見fine pairで逐次確認した
運用上のcalibrationであって普遍的なaccuracy toleranceではない。このPASSは0.15625 usをpre-eventの固定stepに
選ぶだけで、RK4形式次数、絶対物理精度、M3-C1 solver agreement、30 ms/event収束、他size/case/ion-drag variant、
accepted RK stage、Freeze/Disappear、Brownianを認定しない。

二つのイオンドラッグvariantは、片方を真値、もう片方を誤りと自動判定するためのものではない。それぞれ別の物理仕様としてGate 2を満たすか確認し、軌道分岐の感度試験に使う。

## 10. 比較成果物の最小構成

通常の一比較で残す成果物は次の四つに限定する。

- `comparison_manifest.json`：入力hash、model revision、field表現、許容帯、四つの必須判断と適用範囲
- `critical_probes.parquet`：固定した少数点の初期条件、原始場、各力、電荷右辺
- `trajectory_events.parquet`：共通出力時刻の状態、最初のboundary event、自己収束結果
- `comparison_summary.json`：合否、最初の不一致、認定範囲、`NOT_TESTED/NOT_APPLICABLE`項目

Brownianを比較する時だけ`ensemble_metrics.parquet`を追加する。最初の不一致をこの情報だけで所有moduleへ局在化できない場合だけ、
該当粒子のstage probeまたは詳細field probeを一時生成する。性能成果物、全粒子full trace、巨大診断reportはこの比較成果物へ
含めない。R-Z軌道図は説明用に生成できるが、図の重なりをPASS authorityにしない。

## 11. 完了条件

2D比較の終了判断は§1.1の四つの必須判断と五つの終了条件だけで行う。現在の100 nm common-P1
Case-A/Case-P anchorは、その限定scopeについて`CLOSED_ACCEPTED_WITH_LIMITATIONS`である。aggregate three-current productionと
critical boundary microcaseも完了した。Case-P派生三電流比較はcommon-P1、Brownian-off、100 nm・287粒子・30 msの
独立coverageとして完了し、candidate/明示drag COMSOLの3刻み収束、全frameのlifecycle/finite mask exact、
共通有限stateの`r,z,Z`、共通active stateの`v`、141 event/fate identityをすべてPASSした。
元Case-P二電流anchorとは異なる任意物理なので終了条件から分離し、普遍的COMSOL同等性や物理model validationへ拡張しない。
したがってP21と明示scopeの2D benchmarkは`CLOSED_ACCEPTED_WITH_LIMITATIONS`、`2D_CRITICAL_VV_COMPLETE`である。

性能、全12 package、追加粒径、第2 ion-drag、native-field、3D、時間依存場はこの終了条件に含めない。詳細Gate、内部step probe、
多数の統計表は、四つの必須判断のどこで差が始まるか不明な場合だけ使用し、通常比較の常設要件やsolver coreの依存物にしない。
