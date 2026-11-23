# プロジェクト最終レビューと採点

評価基点: `78c8c412f0b78d8cd2d6a9e96f5f4b312bf97747`、2026-11-23。
対象は現在提供している独立solver、通常client、公開操作、外部V&V、検証・配布・説明。
今回の作業はレビューであり、productionの修正ではない。現在のXY/RZ等の範囲を評価し、未提供の3D/GPU等を欠陥として採点しない。

## 判定と採点

**総合77/100（等重み平均77.25）。技術者向けalphaとして条件付き合格。一般利用者向け正式版化には、下記の公開入力・出力4項目の修正が必要。**

| 観点 | 得点 | 評価した強み | 主な減点理由 |
|---|---:|---|---|
| 利用性・入力出力の信頼性 | 78 | 一つのCLI、編集可能な入力、check/run、元case不要の結果利用 | exact ID誤結合、false-ready、nested入力例外、単独JSONの部分公開 |
| 責務分離・保守性 | 78 | core/client分離、単一engine、明確な数値owner、依存境界 | accepted状態更新3箇所、production式の複数body、親専用値のworker転送 |
| 拡張・削減の容易さ | 73 | typed plan、モデルrevision、明示field、現在のcompatibility整理 | 他カテゴリに依存するbinding、proof/facts/説明への変更波及、検証側inventory重複 |
| 科学・数値の説明／検証 | 80 | 独立解析解・OU・連成・event検証、誤差とclaimの区別 | 結果のscalar/role説明不足、内容identityの保証範囲、現行全体の数値資格不足 |

点数は現在の基盤の工学的評価であり、正しい物理計算の確率や物理精度の百分率ではない。
前回の等重み78.25（丸め78）から、利用性を82→78へ見直した。追加確認したR15/R16の影響を反映したものである。
保守性・拡張性・科学説明の改良案は未実装であり、設計を具体化しただけで実装点を加算しない。

## 優先指摘：現在再現できる公開入力・出力の不具合

P1は正式提供前に優先修正する項目、P2は通常改修が必要な項目とする。確率や装置上の危険度を表す分類ではない。

1. **P1 / R15：場CSVの整数IDをfloat化した後に検査するため、異なるIDの結合が成功する。**
   [CSV reader](../solver/src/chamber_particles_tools/csv_import/workflow.py:1063)で情報を失い、[join](../solver/src/chamber_particles_tools/csv_import/workflow.py:977)では元tokenを判別できない。
   mesh ID `9007199254740992`とraw field ID `9007199254740993`を同じ座標に置いたprobeがexit0・公開まで進んだ。
   そのprobeで異なる場値による誤軌道まで実証したという意味ではない。identityの誤受理自体が問題である。
   readerでscope選択とexact integer解決を行い、物理float値と分けて同じrow permutationを渡す必要がある。

2. **P1 / R16：static builderが時系列温度をcheckでreadyとし、build途中で失敗する。**
   [metadata受入](../solver/src/chamber_particles_tools/electrostatic_builder/workflow.py:487)に時間意味の検査がなく、consumerはstatic配列として扱う。
   今回も公開checkのfalse-readyを再現した。同じ4節点入力の小型buildがsolve後にIndexErrorとなる証拠は保持され、sourceも一致する。
   static thermal recordを共通prepareで確定し、check/buildの両入口でsolve前に拒否すべきである。
   誤った完成電場の成功を実証したものではない。

3. **P2 / R06：source設定・CSV行構造の解決が値の使用より遅い。**
   [source設定](../solver/src/chamber_particles_tools/source_builder/producer.py:250)の`columns:null`が今回もTypeError・exit1となった。
   [CSV行選択](../solver/src/chamber_particles_tools/source_builder/producer.py:203)では余剰cellのsilent acceptanceも残る。
   型付き設定・全row幅・選択後のexact IDをownerで解決する。CLIのcatch-allで内部例外を隠す修正にしない。

4. **P2 / R07：analyzeの単独JSONが完成前に公開される。**
   [公開先への直接write](../solver/src/chamber_particles_tools/cli.py:228)により、中断probeでは28 bytesの不完全JSONが残った。
   現行sourceは同一で、問題は未修正。完成temporaryから既存fileを上書きしない原子的公開へ置換する必要がある。

これらは、正常系54 testsの成功と両立する。既存の正常確認がこの入力境界・中断条件を捕まえていないためである。

## 構造・変更容易性の指摘

以下は確認した構造上の不足であり、現行の合法入力で新しく誤った軌道を示したという主張ではない。

| 課題 | 現行の根拠 | 根本改良 |
|---|---|---|
| 他カテゴリへ依存するprimitive binding | [runtime.py:1242](../solver/src/chamber_particles/physics/runtime.py:1242)、1297、1343、1402 | 各modelのroleを既存field offset表へ直接bindし、必要な物理背景の互換検査を明示する |
| accepted material更新を3箇所が所有 | [transport.py:7163](../solver/src/chamber_particles/transport.py:7163)、7636、9725 | accepted prefix後の共通更新を一ownerへ。method固有continuation、axis、periodicは固有の意味を保つ |
| production算術の複数body | [forces.py:1125](../solver/src/chamber_particles/physics/forces.py:1125)、[compiled.py:2156](../solver/src/chamber_particles/physics/compiled.py:2156) | familyごとのscalar lawを共有。独立oracle、boundの証明、丸め規則は役割別に保持する |
| 親schedule/説明がworkerへ残る | [PreparedTransport](../solver/src/chamber_particles/transport.py:469)、[serialization](../solver/src/chamber_particles/execution.py:214) | zero-readの親専用値を先に移し、実際のworker参照を具体的recordへ移してからfull planを除く |
| 実行事実の説明が不完全・重複 | [catalog.py:971](../solver/src/chamber_particles/physics/catalog.py:971)、[例題受入](../solver/tests/release_workflow.py:383)、[式の加算順説明](../solver/docs/physics_models.md:22) | resolved scalar/roleを一度投影、実inventoryを受入へ渡し、文書を実演算順へ合わせる |

新規model用のproof追加漏れは将来の危険として残るが、現在のclosed catalog全てに誤った証明が使われているとは判定しない。
worker payloadの増加は測定済みだが、現在のRSS cap超過を実証したわけではない。
fileの行数、古い命名、未参照検索だけを減点や削除根拠にしていない。

producer・外部比較・reportにも残件がある。宣言した派生係数groupの入力変更と全出力の再生成を閉じるR09、
保存population比較のdense容量と用途の制限R11、全区間診断から局所窓の数値資格を推定できないR12を、個別の能力として扱う。
reportの[projection受入](../analysis/build_report_snapshot.py:386)はrow形の検査が不足し、parseとhashが別読込み、
[snapshot公開](../analysis/build_report_snapshot.py:494)もin-placeであるR20が残る。
新規合成入力の制御probeではrows=A/hash=Bを観測したが、過去reportの実データが誤っていた証明ではない。
完成snapshotと公式buildのhashがあることは、元projectionとのone-read bindingを代替しない。

## 科学・数値の到達点と限界

- 現行にも単一engine、同stageのx/v/Z連成、accepted-only commit、first hitと残時間、joint OUとconditional split、安定ID/RNG、親single writerがある。
- 独立な解析式・covariance・charge/eventの確認を保持し、今回もOU、非線形連成収束、exact hitの3 testsを追加実行して成功した。
- `resolved_models`に全scalarとrole→fieldが残らず、結果だけから解く式の全入力を説明する能力に不足がある。元入力hashの存在はこの説明を補わない。
- [結果identity](../solver/src/chamber_particles/output.py:1325)と[final値の読込み](../solver/src/chamber_particles/output.py:1347)は保証範囲が異なる。有限値の内容改変まで同じidentityで区別できる仕組みではない。過去結果が実際に改変されていたという証拠ではない。
- 元の10 nm・各287 computational particlesというA/P 2 case・各40実現の保存評価は、大局的な輸送の実用的対応を支持する。面33の停止割合差は全287粒子を分母としてA −4.72/P −4.07 pp、100 µs時点survivor速度差は +13.47/+17.55%という局所留保が残る。個別Brownian乱数実現の一致は要求しない。43,220粒子の別Vigus結果へこの判定を一般化しない。
- A/P比較は元sourceと入力に束縛された証拠である。元replayの50 source中7件は現行と異なる。今回の現行HEADによる新規A/P・COMSOL再計算や全数値資格へ読み替えない。
- 現行の縮約電場builderと履歴S3-Mの全材料・浮遊導体は別の能力。履歴の電場差約4%を現行builderの精度として使わない。
- dt、場mesh/回復、Brownian path解像、ensemble、reference精度は異なる評価軸。停止数やmedianの近さだけから全方程式・装置予測の認定をしない。

## 実行した確認と再利用した証拠

| 確認 | 今回の結果・範囲 |
|---|---|
| lock、Ruff format/lint | 成功。format182 files |
| Pyrefly | 0 errors。172 warningsは省略表示されており、warning0とはしない |
| import-linter | 49 modules・126 dependencies、6契約すべて維持 |
| complexity | 全function/method CC<16 |
| input/result/CLI/documentation/report tests | **54 passed /29.96 s** |
| 独立数値tests | **3 passed /1.50 s**。全suiteとは区別 |
| 公開の不正source／static check probe | 2件実行。内部例外とfalse-readyの残存を確認 |
| 過去の全suite | 実JUnitとhashを再確認：core/scenario1240、外部156。今回は再実行していない |
| 現行source binding | 67 sourceは実行済み記録と一致。A/Pの元実行sourceとの一致とは別 |
| 公開通常flow・benchmark | 撤去後の8 CLI、解析hit/保存位置、P06/P10/P14/P19/B03の実行記録を検査。今回の新規simulationとは数えない |
| 撤去と保全 | 分類済み77,774件残存0、追加一時probe4本撤去、native479 SHA・既知11環境保持の独立確認を維持 |

直近のremote CIは観測したsource/runの範囲だけを記録し、進行中runやこのレビュー文書を加えた後のcommitを成功と推定しない。
全model組合せ、新規COMSOL、全ensemble、scale/release matrix、browser/buildを今回実行したとはしない。

## 改良順序と最終判断

1. exact ID、static input、source解決、JSON公開の4項目を、公開入口の最小回帰とともに修正する。
2. model事実・直接binding・proofの構築、親/workerの受渡しを整理する。
3. scalar算術とaccepted material更新をfamily/責務単位で集約し、科学payloadと影響性能を確認する。
4. 結果の内容binding、producer依存係数、比較の容量/claim、reportと例題の事実投影を閉じる。

具体的な変更範囲・撤去・caller・検証・依存は[全体改良設計](foundation_root_cause_review_2026-11-23.md)の20課題・13改良単位を参照する。
今回の[レビュー証拠](../solver/evidence/final_project_review_2026_11_23/review.json)は、採点、各独立reviewer、実行と未実行を束縛する。
撤去工程は完了している。現在の基盤は通常の技術者向け利用と原因調査が可能なalphaであり、上記の公開不具合と責務上の残件を明示した条件付き評価とする。
