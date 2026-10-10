# v0.2.0 support, qualification, limitations, and error guide

この文書は、`0.2.0`の固定topology 2-D版を利用する時の入口です。公開済み`0.1.0`は
static 2-D baselineとしてtag上で固定します。対応する設定、明示的な非対応範囲、失敗時の最初の確認先を
一か所にまとめます。式、適用域、数値許容差、永続形式の詳細を再定義する文書ではありません。厳密な正本は次です。

- 入力：[`case_format_v3.md`](case_format_v3.md)
- 物理モデル：[`physics_models.md`](physics_models.md)
- 数値計算：[`numerics.md`](numerics.md)
- 結果：[`result_format_v3.md`](result_format_v3.md)
- 並列化・性能判断：[`parallel_execution_plan.md`](parallel_execution_plan.md)

COMSOLは入力producerまたは外部V&Vの一つです。solver coreの実行には不要であり、本表の「対応」は
任意のCOMSOL設定との一般同等性を意味しません。

## 0.2.0の用途別検証保証

保証するのは、登録した入力・方程式・解像度について、下表の判定基準を満たすことです。
対応selectorの全組合せや、任意のチャンバーの予測精度を一括して認定しません。個別profileの入力、SI誤差budget、
粒子数・seed・刻み・depth、独立reference、適用域は
[用途別受入条件](../../reviews/v0_2_release_use_case_acceptance_2026-10-10.json)、結果と証拠のidentityは
[0.2.0検証記録](../../reviews/v0_2_release_qualification_2026-10-10.json)を参照してください。

| 用途 | 保証の範囲と判定 |
|---|---|
| 決定論運動・反復鏡面壁 | A01/A02：指定した調和運動の位置・速度、20回平面鏡面反射のevent時刻・最終状態を独立解析解の絶対SI budgetで評価 |
| 可変係数・連成Brownian | B01/B02/B03：時間線形drag/flow、位置affine flow、一様stationary OML＋Coulombの指定ensembleで平均・共分散・電荷を独立Green/scalar referenceと比較。境界の一般first-passageや一般SDE次数へ拡張しない |
| Brownian吸収・反復反射・RZ到達 | C01–C04：一定係数・point particleの登録h/depthごとにCDF誤差、MC幅、経験的reference幅を分離して判定。RZの元N16k D2判定未達は保持し、別登録N64k追試のPASSと区別 |
| 物理式とmodel適用域 | D01/D02：Talbot/Saffman原著式、drag/charge/ion drag等の独立極限・解析運動・pure/compiled・適用外拒否。model-form感度の数値確認であり、species混合・付着・発生源の実験的validationではない |
| 限定COMSOL比較 | 共通P1・100nmの登録Case A/Pと決定論三電流caseを現行v46で再評価。保存native参照をhash固定して使うversion再資格であり、新しい独立cohortとは呼ばない。Disappearの原因/IDと、衝突のないCase-Pのwall応答は未認証 |
| 規模・性能・再現性 | 正式P14、B03、P14-Uの既存release条件で反復測定し、科学payload identity、failure、memory plan、収束を判定。時間とprocess RSSは測定機に限定し、利用者未指定のSLAやOS hard RSS capを保証しない |
| 配布 | 同一tagのWindows/Linux CIで品質gate、Quick Start、wheel、runtime-only installを検査。source/wheel/install全Python bytesとREADME由来METADATA、三API・CLI・time-field・接触modeを照合 |

背景場のmesh/time解像度、回復gradient/curl、source・wall parameter、実験的製造KPI、追加modelの精度は、
実際の用途入力と独立referenceを伴って別途評価します。有限depth Hermite pathの連続OU exact first-passageや
zero-miss probabilityは0.2.0の保証に含めません。

## 公開する実行境界

利用者向けPython APIは`load_case(path)`、`simulate(case, output)`、`open_result(path)`の三つです。CLIの
`check`、`run`、`inspect`も同じ経路を使います。最初の実行方法は[Quick Start](../README.md#quick-start)を参照してください。

`check`はYAML、canonical HDF5、hash、source/boundary参照と早期memory gateを検査します。物理モデルの組合せ、
field support、geometry topology、積分器安定性、完全なmemory planは`run`のprepareで解決します。このため
`check`の成功だけで計算可能とは判定しません。

## 対応範囲

| 項目 | `0.2.0`の対応範囲 |
|---|---|
| 座標 | `cartesian_xy`の2自由度運動、または`axisymmetric_rz`場中の`axisymmetric_rz_meridional` 2自由度運動 |
| geometry | 静的2-D volume meshと`line2`材料境界。RZの`r=0`は壁でなくaxis event |
| field | 静的または固定topologyの線形時間snapshotを持つregular、P1 triangle、Q1 quadrilateral。全required fieldは同じlayout、SI単位、明示basisを使用 |
| source | 粒子ごとのrealized table、またはfacet・facet内座標・速度・release時刻・物性を粒子ごとに保持するrealized surface table |
| motion | 無力場ballistic、一定加速度の厳密経路、固定刻み`rk4_fixed`、`exponential_midpoint`、対応Brownian用`ou_langevin` |
| backend | CPU上の一つのsingle-thread compiled engine。粒子stateはresident、scratchはbounded slab |
| output | release/boundary/failure event、lifecycle series、final state、任意の全粒子frame、明示particle probe、checkpoint/resume |

stage評価する一般曲線運動では、材料boundaryがvolume外周を完全に覆うか、boundarylessなら全cell supportedな
regular field boxが必要です。材料domainとP1/Q1 fieldを組み合わせる場合はnodes/connectivityとsupportが完全一致する
必要があります。詳細なcertificateは入力・数値正本を参照してください。

時間依存fieldはcanonical data schema v3の`time_s[T]`と`values[T,N,C]`で与えます。実際のstage時刻で
snapshot間を線形補間し、run内のsnapshot knotではmacro intervalを分割します。required fieldの時刻範囲はrun全体を
覆う必要があり、端点外をhold、clamp、extrapolationしません。時刻軸のない`values[N,C]`は従来どおり静的場です。

### 物理モデル

次の表はselectorの索引です。方程式、必須field、parameter範囲、連続pathでの適用域は
[`physics_models.md`](physics_models.md)だけを正本とします。無効なcategoryはYAMLから省略し、`charge`だけは必ず明示します。

| category | 対応するmodel revision |
|---|---|
| charge | fixed、`oml_stationary_maxwellian_debye_huckel_v1`、`oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1`、`aggregate_relative_drift_regularized_two_current_v1`、`aggregate_relative_drift_regularized_three_current_v1` |
| drag | `epstein_linear_v1`、`epstein_linear_effective_gas_sensitivity_v1`、`epstein_finite_speed_maxwell_mixed_equal_temperature_v1`、`stokes_cunningham_allen_raabe_air_v1` |
| ion drag | `barnes_collisionless_effective_speed_single_positive_ion_negative_debye_huckel_v1`、`relative_flow_screened_collection_orbital_aggregate_ion_v1`、`electric_field_directed_image_orbital_sensitivity_v1` |
| thermophoresis | `waldmann_gallis_free_molecular_single_species_heat_flux_v1`、`waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1`、`talbot_cross_regime_radius_knudsen_v1` |
| electric / gravity | `electric_coulomb_v1`、`gravity_buoyancy_standard_v1` |
| DEP | `quasistatic_spherical_gradient_e2_v1` |
| lift | `rarefied_vorticity_sensitivity_rz_v1`（RZのみ）、`saffman_unbounded_creeping_shear_v1`（XY/RZ） |
| Brownian | `inertial_langevin_fdt_epstein_linear_midpoint_2d_v2`（XY / RZ meridional、fixed/continuous charge、線形Epsteinと適用域が交わる既存additive force） |

solverはcategoryごとに一つのrevisionを選び、指定された寄与を同じstage stateで合成します。適用域外で別モデルへ
自動切替、値のclamp、producer名に基づく分岐は行いません。

### 境界則

静的2-D geometryのwallについて次を受理します。粒子ごとの`contact_radius_m=0`はpoint particle、正値は
XY disk / RZ sphereの中心とmaterial `line2`のfinite-radius first contactです。半径はdrag径や静電半径から
推定しません。接触時の保存位置は粒子中心で、反射はcandidate segment最近点から得る接触法線を使います。

| law | 意味 |
|---|---|
| `stick` | hit位置で速度0の`stuck`終端 |
| `escape` | `escaped`終端 |
| `hold` | hit時の位置・速度・電荷を保持する`held`終端。再飛散待ちではない |
| `specular` | 完全鏡面反射。係数指定不可 |
| `restitution` | 明示した法線・接線反発係数による反射 |
| `maxwell_thermal` | 壁frame鏡面と壁温度のhalf-range Maxwell flux再放出を明示率で混合。wall velocityは静的facetへ接線方向だけ |
| `probabilistic_stick` | 定数付着確率。非付着時はspecular、restitution、またはmaxwell_thermal |

同時hitは`priority_then_combined_normal_v1`だけを使用します。最小priorityの候補で応答が一意にならない場合は、
任意の法線を選ばず粒子単位failureにします。B04以降のBrownian runもterminal lawに加えて`specular`、
`restitution`、`maxwell_thermal`、`probabilistic_stick`を受理します。active hitでは使用済みOU rootのtailを
継続せず、hit prefixとwall応答をcommitして残時間をfresh stochastic rootから進めます。
surface sourceのvelocityはcanonical HDF5の粒子rowに明示します。Brownianはsurface contactを係数midpoint評価より
先に解決し、zero-time active response後はpost-wall stateからfresh rootを開始します。熱的な初速度や任意分布は外部入力作成側で
realizeし、solverは再標本化しません。Brownian surface releaseは外向き法線成分の絶対値がroundoff幅を越える必要があり、
負ならdeparture、正ならzero-time impactです。zero/tangent velocityはnudgeせず拒否します。

## 明示的に非対応のもの

- Cartesian 3-D粒子、完全3-D geometry/field、axisymmetric fieldを回転展開した3-D軌道
- zero-order hold／不連続snapshot、時間で変わるfield layout/topology、moving geometry、粒子によるfieldへのself-consistent feedback
- GPU、solver内部thread/multiprocess、out-of-core particle state。独立caseのprocess並列はsolver外で行う
- adaptive time stepping、汎用ODE/DAE solver、適用域外での自動subcycleや別物理へのfallback
- boundarylessなstage-evaluated P1/Q1一般曲線運動
- moving geometry／法線wall motion、速度・温度依存付着率、再飛散、surface charging、rolling/sliding、粒子間衝突、3-D finite-radius surface
- solver core内のsurface distribution catalog、sampling、source RNG。外部でrealize済みの任意角度・速度・時刻は
  canonical particle rowとして入力できる
- 自動的なspecies混合則、transition-regime blend、未登録model revision
- isotropic 3-D Brownian、連続OU trajectoryのexact first-passage、一般state-dependent SDEの高次精度主張
- 旧schemaの互換reader、checkpoint migration、既存result directoryの上書き

非対応機能を未知key、別model名、過去の`resources.threads`で指定しても無視せず拒否します。必要な機能は
別revisionと独立verificationを追加してから利用範囲へ入れます。

## エラーと対処

### 公開例外

| 例外 | 意味 | 最初の対処 |
|---|---|---|
| `CaseError` | YAML/HDF5を読めない、schema・hash・参照・静的値が不正 | `chamber-particles check case.yaml`を実行し、最初の具体的なmessageを修正する |
| `SimulationError` | prepareで非対応、または実行・I/Oを安全に完了できない | messageが示すfield、model、geometry、dt、memory、出力先を修正し、同じcaseを再実行する |
| `IncompleteResultError` | 完成していないresultを通常modeで開いた | 同じcaseと出力名で`simulate`して安全にresumeする。調査だけなら`open_result(path, recovery=True)`でcommit済みprefixを読む |

CLIはこれらをJSONでstderrへ出し、exit code 2を返します。内部例外やprivate helperを利用側の分岐条件にしません。

### よくあるmessage

| messageの要点 | 原因 | 対処 |
|---|---|---|
| `unsupported ... version` / `unknown key` | 現行schemaでない、または古い／独自key | 現行schemaへ明示的に変換する。aliasや未知key削除で意味を推測しない |
| `content hash does not match` | `case.h5`とYAMLの論理hashが不一致 | 意図したHDF5をcanonical writerで確定し、その返却hashでYAMLを更新する |
| `boundary laws do not match data groups` | HDF5の境界groupとYAMLが一対一でない | 全groupへちょうど一つのlawを割り当て、未知groupを除く |
| `not a supported pair` | data座標とmotion modeが不整合 | XY/XY、またはRZ/RZ-meridionalの組合せにする |
| required field／basis／support error | field名、unit、components、basis、layout、supportのいずれかがmodel要求と不一致 | model正本のrequired fieldをproducer側でcanonical化する。solver内で補間・欠損埋めしない |
| field time range／`time_s` error | snapshot時刻が非有限・非単調、shape不一致、またはrequired fieldがrun全体を覆わない | canonical v3 writerで狭義単調なsnapshot軸と全run coverageを作る。endpoint clampや外挿で補わない |
| topology／boundary completeness error | non-manifold、重複edge、外周欠落、RZ axisをwall登録 | geometryを修復し、材料外周だけを一度ずつ`line2`へ登録する |
| `boundaryless ... requires a fully supported regular field box` | unstructured fieldに材料境界がなく連続pathのsupportを証明不能 | topology-completeな材料boundaryを与えるか、fully-supported regular layoutを使う |
| `rk4_fixed requires dt ...` / continuous-charge dt error | 固定刻みが明示安定性gate外 | `dt_s`を下げて収束確認する。線形緩和が支配する場合だけ`exponential_midpoint`を検討する |
| `predicted solver memory exceeds ...` | solver-owned planが`memory_limit_mb`を超える | 正当な上限へ増やすか粒子数・入力規模・要求出力を減らす。これはOS RSS hard capではない |
| result／partial identity mismatch | 出力先が存在、またはpartialが別case・revision | 完成resultは上書きせず新しい出力名を使う。partialは一致する元caseだけでresumeする |

### run全体の成功と粒子単位failure

局在できる数値不能はrunを中断せず、その粒子を`failed`にしてfailure eventへ保存します。したがって
`simulate`成功後も、manifestの`failure_reason_counts`とfinal lifecycleを必ず確認してください。reason codeの数値対応は
各resultのmanifestが権威です。

- `field_support`／`model_applicability`／`nonfinite_physics`：場のcoverage、物理revisionの適用域、入力primitiveを修正する。
- `integrator_accuracy`／`indeterminate_applicability_certificate`：`dt_s`を下げて収束を確認し、場の急変が入力解像度で表現されているか確認する。
- event／boundary／surface departure系：`dt_s`、geometry品質、corner law、releaseの位置と内向き速度を確認する。

failureをepsilon、tolerance拡大、clampで消して成功扱いにしません。解析でfailure粒子を除外する場合も、除外数と理由を
結果とともに報告します。

## schemaとversionの方針

正式公開版`0.1.0`はYAML case schema 2、canonical HDF5 data schema 1、result schema 2、checkpoint schema 2です。
`0.2.0`のpackage metadataはYAML case schema 3、canonical HDF5 data schema 3、
result schema 3、checkpoint schema 2です。現行readerは旧data/result schemaを互換読込しません。

- readerは現行versionだけを受け、未知version・未知key・未知datasetをfail-closedで拒否します。
- YAMLはraw file hash、HDF5はcanonical logical content hashを使います。編集後に古いhashを流用しません。
- 完成resultの`run.json`が、実際に解決したschema、algorithm、physics model revisionのauthorityです。
- 科学的意味、RNG、event順、永続列が変わる変更は対応revisionまたはschemaを更新します。
- 旧reader、migration shim、compatibility aliasをproductionへ併存させません。旧resultを保持する場合は、それを作成した
  package、lock、入力、manifestを一緒に保存します。変換が必要ならcore外の明示的な一回変換にします。

現行algorithm IDの一覧をこの文書へ複製しません。実行ごとの正確な値は`run.json`、開発上の現行rosterは
[`implementation_plan.md`](../../implementation_plan.md)をauthorityとします。
