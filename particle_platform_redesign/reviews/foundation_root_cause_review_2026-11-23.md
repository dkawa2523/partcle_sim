# 基盤全体の根因調査と改良案 — 2026-11-23

対象は `cleanup/current-only-platform` の作業ツリー。基点は `75100e74f1243b527e065c9e6a68121221408b39`。
前回の責務再構成と今回の撤去を反映したコードを調べた。提案を現行仕様と混ぜず、実装済みの撤去、
新たに確認した不具合、構造上の危険、目的次第で必要になる機能を区別する。本書は改良案であり、
[architecture](../architecture_proposal.md)や現行の物理・数値仕様を置き換えない。

## 調査からの判断

主要な根因は、**同じ科学的事実の解決・使用・保存が、まだ一つの所有者からの受渡しになっていないこと**である。
単一engine、四つのcore API、installed client、実行と共通のprepareは成立している。
しかし、モデルの設定は結果説明で一部失われ、場の共有は他モデルの型による分岐へ変わり、
同じ式やaccepted wall transitionは複数のproduction経路で更新されている。
そのため、現在の計算が通っていても、追加・削減時に離れた箇所を同時修正する必要が残る。

抜本的な改良は、ファイル数・行数の削減ではなく、次の所有を揃えることにある。

1. 解決済みモデルのidentity・scalar設定・field role・適用域条件。
2. sample済みprimitiveへのbindingと、科学的に必要なモデル間の一致条件。
3. productionの局所式と、別に導出するbound・適用域証明。
4. accepted boundary hitによる状態遷移と、積分法固有の継続。
5. 入力・派生係数・結果・説明に渡す意味と、完成物の公開。

汎用plugin、任意ODE framework、第二engine、巨大な診断台帳を導入する理由は見つからなかった。

## 範囲と実行の区別

| 領域・独立経路 | 調査対象 | 確認・実行 | 限界 |
|---|---|---|---|
| 機能選択・CLI/API | `api.py`、`cli.py`、catalog、installed examples | 四API、console/moduleの同一経路、Windows/Linux wheel一巡 | 全modelの全直積を試した意味ではない |
| 入力とsource | canonical/YAML、CSV importer、recipe、source producer、publisher | 自分のCSV、internal/surface、編集例題、独立物性とexact ID | malformed nested設定の新規不具合はR06 |
| 場変換・生成 | field cache、P1転送、縮約electrostatic builder | installedのcache/electric例、selected native CSV import | 全材料・浮遊導体は現行機能外 |
| 物理 | catalog、runtime、forces、charge、compiled | 独立式、coupled charge、noise/OU、適用域・bound | 全装置・全regimeの物理認定ではない |
| 積分・event・壁 | integrators、stochastic、geometry、events、boundaries、transport | 解析hit、corner、残時間、Brownian/複数反射/軸oracle | 連続OUのexact first-passageは未認定 |
| 実行・状態・保存 | engine、cpu、execution、output、time/lifecycle | process、slab、resume、immutable context、公開結果 | payload全体の改変検出にはR08の制限 |
| 解析・export・表示 | selection、analysis、visualization、CLI | result-only利用、cohort/coverage、完成物公開 | 単独JSON公開はR07 |
| COMSOL/V&V | native意味、明示CSV式評価、保存集団、matched比較 | 最終外部suite156件、保存nativeに対するcurrent candidate小型再実行 | 新規native solve・Vigus全再計算ではない |
| 性能・検証 | 8 executable benchmark modules、verification/scenario、品質gate | 影響5moduleを実行。継続時core全体1240件成功 | smokeを10⁶粒子実測・全scalingへ外挿しない |
| 文書・report・資産 | 正本、README、report snapshot、旧結果/native区分 | 文書33・skills4、snapshot tests。手動削除後も既知native479件の全SHA一致 | 記録済み撤去対象は残存0。未知資産・全環境の網羅保証ではない |

実行の詳細は[今回の記録](../solver/evidence/current_only_2026_11_23/README.md)を参照する。
初回の旧regex失敗と修正後のclean runを区別し、8883e3aのcleanは1228 passed / 679.713 s、
その後のR13整理を含む全体は1240 passed / 473.52 s。外部は156 passed / 39.10 sのsource-bound記録を保持する。
手動削除後は同じ実装の小型ballistic公開操作を新たに実行した。以前の全体suiteの再実行や、下記の新規提案の実装・認定とはしない。

## 既に改良した点と、今回残っている点

前回・今回の実装で、通常clientのwheel収録、単一CLI/publisher、実行と共通のcheck、result4 context、
state/workspace所有、モデルmetadata/field制約の集約、共通macro clock、不要な旧campaign・P11専用runner等を整理した。
CSV診断のfloat経由ID丸め、非有限診断、重複headerも確認して修正した。
これらを未着手の課題として再計上しない。

以下は、調査中に新たに確認した未解消点または明示すべき制限である。
「構造」は現在の合法なモデルで誤計算したという実証ではなく、変更時の波及・漏れの危険を意味する。

### R01 — 結果の同定はできても、設定の説明が完結しない

**確認済み・優先A・説明可能性。**
[catalog.py:971](../solver/src/chamber_particles/physics/catalog.py:971)の`resolved_models()`はmodel/revisionを保存し、
scalar値を追加するのはTalbot・一部lift・electric cap等に限られる。
[engine.py:497](../solver/src/chamber_particles/engine.py:497)でrun factsへ渡し、manifestとresumeにもこの部分集合を使う。
[ResultContext](../solver/src/chamber_particles/output.py:116)は形状・source・境界の解釈を持つが、完全な物理設定を補わない。

実行probeでは合法なEpsteinの`delta=1.0`と`1.2`が異なるplanへ解決される一方、
`resolved_models()`は等しい辞書になった。入力hashは異なるため、誤ったresumeを許す指摘ではない。
結果だけを受け取った人が「どの係数・field roleでこの式を使ったか」を説明しにくい問題である。

根因は、モデルidentityの投影を科学設定の説明にも使い、一部モデルだけ手で情報を追加したこと。
typed planから正規化済みscalar・role→field名・明示適用域条件を一度だけ作り、checkとresultへ渡す。
resumeに必要なidentityはそこから必要部分を消費する。field全配列や入力全量をcontextへ複製しない。
受入では、係数一つを変えた二caseの説明が区別されること、default解決値とroleが残ること、
元caseなしでも係数・field role・適用域の説明ができることを公開APIで確かめる。
result-only解析自体は既に提供されており、結果単独での再計算可能性は別要件とする。

### R02 — primitive共有が他カテゴリの型への依存になっている

**構造確認済み・優先A・責務/拡張。**
[runtime.py:1297](../solver/src/chamber_particles/physics/runtime.py:1297)のTalbot、
[:1343](../solver/src/chamber_particles/physics/runtime.py:1343)のlift、
[:1402](../solver/src/chamber_particles/physics/runtime.py:1402)のSaffman bindingは、
drag/thermophoresisの型によって再利用offsetを選ぶ。一方、required fieldは既に名前で重複除去され、
[:4014](../solver/src/chamber_particles/physics/runtime.py:4014)にはfield名→sample offsetの表がある。

根因は、必要なphysical background一致と、同じ配列を読む最適化を同じ分岐で扱ったこと。
各modelが宣言したroleを既存のoffset表へ直接bindすれば、同名fieldは自然に共有される。
同じgas backgroundを科学的に要求するルールはcatalogの一箇所へ残す。
新dragを足すために無関係なTalbot/lift binderを変更しないことを目標とする。
同名/別名field、複数モデル併用、意図した不一致拒否を確認し、同stageの力・G/Jを維持する。

### R03 — modelの式と安全証明の追加漏れを、既存familyへ流し得る

**構造確認済み・優先A・科学/数値拡張。**
compiled stageは未知codeを拒否するが、continuous applicabilityの
[drag](../solver/src/chamber_particles/physics/runtime.py:2863)、
[charge](../solver/src/chamber_particles/physics/runtime.py:2924)、
[thermophoresis](../solver/src/chamber_particles/physics/runtime.py:2980)、
[ion](../solver/src/chamber_particles/physics/runtime.py:3003)、
[lift](../solver/src/chamber_particles/physics/runtime.py:3036)は、残りのcodeをそれぞれ既存の証明familyとして扱う。
現行code集合では対応しており、未知codeが利用者入力から現在通ると実証したものではない。

根因は「今は残りが一種類」という閉じた前提が、将来のmodel追加時にも暗黙に引き継がれること。
既知familyを明示し、それ以外は拒否する。model→stage family/proof familyの対応をprepareで閉じる。
既存の全force bound coverageは維持する。stage評価、global bound、local enclosure、適用域証明は
異なる数学なので一つのcallbackへ統一しない。新modelの証明不足をゼロ寄与・既存証明で代用しない。
受入は現行modelの境界内外、unsupported family拒否、constant specializationの成立/不成立、独立bound検証。

### R04 — production式が配列評価とcompiled評価へ二重化している

**構造確認済み・優先B・保守/科学authority。**
Epsteinの局所式とgateは[forces.py:1071](../solver/src/chamber_particles/physics/forces.py:1071)、
[:1125](../solver/src/chamber_particles/physics/forces.py:1125)と
[compiled.py:2156](../solver/src/chamber_particles/physics/compiled.py:2156)にある。
NumPy入口とhot rowの違いは必要だが、係数・符号・適用域を別々に維持する必要はない。

カテゴリ所有の小さなNumba対応scalar式をproduction authorityとし、配列・compiled入口が使う。
electric/gravity/DEPの既存共通factorを足場とし、force全体の抽象frameworkは作らない。
boundは同じ点評価の単なる呼出しではなく別の証明として残す。testsのDecimal/解析式・独立oracleも残す。
検証は独立値・極限・適用域と同一入力の科学payload、影響するhot benchmarkの時間/memoryを組み合わせる。
丸め順が変わる場合は単なる移動と称さず、数値差を記録して許容とrevisionを判断する。

### R05 — accepted wall transitionを積分法別の継続処理が重複所有する

**構造確認済み・優先B・状態/保守。**
[transport.py:7163](../solver/src/chamber_particles/transport.py:7163)と
[:9725](../solver/src/chamber_particles/transport.py:9725)は、hit正規化、resident x/v/Z、
ordinal、lifecycle、event、統計、replay jumpをそれぞれ更新する。
exactのorigin/residualと曲線stack・OU restartの違いは本来必要である。

根因は共通のaccepted状態遷移が、その後のmethod固有継続と一緒に実装されたこと。
`_canonical_hit_position`、`_set_boundary_lifecycle`、`_append_boundary_event`は既に共有されている。
残るpost-responseの状態・ordinal・ledger更新を一つにし、exact/curved/OUの継続adapterを残す。
curvedのfinite-radius endpoint/residual補正、exact_origin、OU restartは固有の所有者へ残す。
hit生成・交差探索・numerical pathまで統一しない。第一hit、corner、多重hit、残時間、軸、terminal右連続性、
RNG/ordinal、worker/slab/output/resumeの科学identityを確認する。compiled inlineと容量を維持して測る。

### R06 — nested入力設定が解決前のまま処理へ入り、公開エラー境界を漏れる

**公開CLIで再現済み・優先A・使いやすさ/責務。**
`init_example('xy-ballistic')`のrecipe2で`source.configuration.columns: null`だけを変更すると、
`chamber-particles --json create ... -o ...`はstructured errorではなくexit1のTypeError/tracebackになる。
出力は公開されなかった。
[producer.py:226](../solver/src/chamber_particles_tools/source_builder/producer.py:226)で
[:250](../solver/src/chamber_particles_tools/source_builder/producer.py:250)の`set(columns)`前にmappingを確認していない。
filterも[source読込み](../solver/src/chamber_particles_tools/source_builder/producer.py:319)で設定解決前に使う。

根因はtop-level key検査とnested値の正規化が分かれ、`Any`がconsumerまで残ること。
source所有の入口で必要なmapping/column/units/filterを小さく正規化してからCSVを読む。
CLIにTypeErrorのcatch-allを追加して内部不具合まで入力エラーへ隠さない。
不正なmapping・column shape・filterと有効なinternal/surfaceを公開CLIで確認し、exit2・有用なfield名・未公開出力を確かめる。

### R07 — 単独analysis JSONだけ完成前に公開される

**書込み中断probeで再現済み・優先A・結果再利用。**
[cli.py:219](../solver/src/chamber_particles_tools/cli.py:219)の`_write_derived_json`は公開先を直接`open('x')`する。
一度の書込み中断を注入するとexit2になるが27 bytesの部分JSONが公開先に残った。
case/bundle/export/viewは完成物をpublishするため、ここだけ同じ利用上の保証から漏れる。

根因は一ファイル出力を軽い例外として扱い、「完成物だけ公開」の責務を使っていないこと。
既存publicationの考え方を小さな単一ファイルにも適用し、同じfilesystemで完成後に公開する。
汎用transaction frameworkは不要。既存先の非上書き・中断時の公開物なし・成功JSONの一致を確認する。

### R08 — result identityの保証範囲が、数値payloadの完全性と異なる

**改変copyで確認済み・優先B・意味/整合。**
有効resultのcopyで`final.h5`のfiniteな`model_weight`だけ2倍にしても、`open_result`は受理し、
analysisのweightは1→2になる一方、source identityは同じだった。
[output.pyの読込み検査](../solver/src/chamber_particles/output.py:1295)と
[identityの投影](../solver/src/chamber_particles/output.py:1925)は構造・manifest/context等を検査し、
全数値payloadのcontent bindingではない。通常のsolver出力が自然に壊れるという実証ではない。

根因は「同じ入力/manifest/contextからの結果」と「数値内容まで同じ結果」を同じidentityと読み得ること。
まず保証範囲を明示する。再利用で内容同一性が必要なfinal/event等は、既存completionでhashを束縛する案を比較する。
全frameをopenごとに再走査する変更や新診断subsystemは先行導入しない。
正常読込みの費用、copy/repackの意味、有限値改変の検出、partial/resumeの整合を測って採否を決める。

### R09 — primitiveと派生係数の整合はproducerの手順に依存する

**コード上の制限・優先B・科学入力。**
[Vigus producer](../solver/tools/vv/comsol/prepare_vigus_transport.py:325)はprimitiveから係数群を生成し、
上流hash・parameters・係数名を保存する。既存係数がある入力は拒否している。
一方、[電場builderの出力](../solver/src/chamber_particles_tools/electrostatic_builder/workflow.py:194)は
新E/plasmaを作り、粒子のprescribed charge/ion/DEP再生成は自動では行わない。
[説明](../solver/src/chamber_particles_tools/electrostatic_builder/README.md)にも明記されている。
小型4節点probeでは、既存の派生名`previous_grad_e2`をゼロで与えて新しい電場をbuildすると、
新Eの最大値1750 V/mに対して旧gradient表はゼロのまま保持された。
独立prescribed fieldも許す設計であり、誤った粒子輸送caseが実行されたと実証したものではない。
このprobeは旧値の保持を確認したもので、正しいgradientが非ゼロであることを示す検証ではない。

根因はbundleの「値・単位・hash」と、producerが意図する派生関係の保証が別であること。
独立prescribed fieldに勝手な関係を推定してはならない。
`_full_force_fields(data, parameters, properties)`は既にI/Oを持たない係数生成であり、再分割だけでは改良にならない。
この既存処理を利用し、新primitive→依存出力群というproducer間のhandoffを明示して、
入力primitive/parameter identityと出力係数群を一つの生成操作へ束ねる。
派生関係を明示したcaseだけ、stale係数を検出する小さな記録を既存provenanceへ残す。
Vigus固有の同次粒子条件をgeneric coreへ持ち込まず、heterogeneous物性をどう扱うかは別に解決する。
節点再構成、off-node差、ゼロEの意味、単位・基底、依存量の同時更新を検証する。

### R10 — 外部CSVの意味解決前に曖昧さを失っていた

**確認・修正済み。再発防止の根因。**
局所式診断のIDをfloat64へ通すと`2^53`超で丸まり、DictReaderで重複列を許すと最後の値に置換された。
shared frozen-state readerと保存集団・mapped trajectory readerを修正し、既に重複拒否していた描画にも回帰確認を加えた。
installed source importerは元々exact integer/unique headerを扱い、selected field importerはexact schemaを確認している。

形式ごとのreaderで、header一意性→exact identity→物理float→意味/coverageの順に情報を失わず解決する。
異なるCSVを巨大な共通validatorへ押し込む必要はない。存在検査だけでtyped/意味の検査を済ませない。
新規常設基盤ではなく、各現行入口の小さな検査と公開操作の回帰例へ閉じた。

### R11 — 集団比較は単一case用のdense処理で、大規模・ensemble再利用が局所化されていない

**コード確認済みの能力限界・優先C・用途次第。**
[evaluate_saved_population.py:173](../solver/tools/vv/comsol/evaluate_saved_population.py:173)は全framesをlist化し、
時間×全ID×5状態のdense配列も作る。nativeも全行listとdense配列を持つ。
比較は同じID集合を要求し、equal computational counts、共通保存時刻で記述する。
これは保存RZ比較の正当な限定で、unpaired ensemble/physical weight一般評価器ではない。

10⁶粒子・126時刻ではstate配列一つだけでも約5.04 GBになり、frame/参照等は別に必要となる。
この規模の実測ではなくshapeからの計算である。coreのbounded scratchから外部評価までboundedと推定しない。
必要になったらpublic ResultViewから時刻/batch単位で集約し、unpaired集団比較とpaired deterministic診断を別入口の意味として保つ。
既存summaryで足りる目的に新たなensemble frameworkは不要。
再利用する場合はsource/停止先/時間/occupancy、seed間変動、打切り/coverage、独立標本数を明示する小さな評価へ移す。

### R12 — 数値資格を説明できる情報は、macro設定と全run集計に偏る

**コード確認済みの保守的制限・優先C・資格が必要な時。**
[matched evaluator](../solver/tools/vv/comsol/evaluate_matched_trajectory.py:758)は実macro分割を確認し、
[:801](../solver/tools/vv/comsol/evaluate_matched_trajectory.py:801)以降は既知policyと全runのrefinement/axis/residual/event等がゼロの十分条件で
accepted rootと結び付ける。別cohort/時間窓にeventがあっても、その情報を対象窓へ局在できず資格は保守的に未確認となる。
この制限を消すためにglobal countから局所精度を推定してはいけない。

根因はmacro、accepted root/path、安全証明細分、保存clockが別の事実である一方、保存する実行情報の粒度が異なること。
実用比較は現行のraw/population評価で継続できる。eventを含む資格が必要になった場合だけ、
明示ID/窓への既存probe拡張か、それに適した独立収束実験を選ぶ。全粒子の巨大path ledgerは先行追加しない。
数値証拠を増やせば必ず同等性になるわけではなく、model/field/referenceの対応も別に必要である。

### R13 — 科学scenarioのprivate容量定数を所有検証へ整理

**確認・整理済み。`8883e3a`後の追加撤去・検証の保守。**
当初のforce/Brownian scenarioには544/472/171/851 bytesやtree容量式が残り、
横断確認でwall/source scenarioにも272/64/624/52 bytes、event/queryの内部式が見つかった。
これらの固定値を三つのscenarioから撤去し、予算内・phase会計・slab差・科学identity・failure局所性は保持した。
ゼロ容量の既知specialization確認は、不要な非ゼロbankを持たないという役割で保持している。

[既存workspace所有検証](../solver/tests/verification/test_workspace_ownership.py)は実allocation rootを数える。
不足していたsurface releaseとexponentialのcurved wall/certificate/replayを、既存fixtureで追加した。
run中に確保されるBrownianのbase/adaptive treeと二つのpending bankはprepareだけでは検証できないため、
実bankの同時容量をpublic reserveへ照合する小型検証を同じownerへ加えた。
N=1/13、base/max=(0,0)/(3,3)/(0,8)/(3,8)/(10,10)を確認し、private byte式のcopyは持たない。

B03 runnerからもtreeのprivate式と専用の重複fieldを撤去し、completed planとslab reserveのreportを保持した。
B03の独立transient stage上界（raw646、丸め後648 bytes/row）は変更していない。
actual bank確認は全transient/RSSの証明ではなく、両者を混同しない。
変更後の関連30検査が成功（22.44 s）。B03の32粒子・4行も再実行し、各science hashが変更前と一致した。
これは同じ標本の実装identity確認で、新しい精度認定や独立ensembleではない。

### R14 — 現行能力・履歴の科学証拠・旧software生成物を一括して扱えない

**能力/運用上の制限・優先B。**
提示レポートS3-Mは全材料・浮遊V/Qの問題だが、現行installed builderはgas-only RZ/P1縮約である。
S3-Mのfield差4%等を現行実装の精度や配布機能の証拠として転用できない。
全材料生成が目的なら、材料/topology/V/Q/interface charge、個別Q/global balance、gas側回復、
依存係数再生成まで別の拡張として定義する。縮約solverの責務整理だけで全材料対応済みとはしない。

また旧campaignにはcandidate結果、nativeが実際に使ったsource、nativeの刻み選択記録、環境が混在する。
静的未参照だけで消すとnativeの根拠を失う。今回も196 native authorityのhashを保持し、旧candidate層だけを整理した。
形式の新旧だけでも撤去対象は決められない。実際に、現行形式の旧P10/P11 worker debug生成物も残っていた。
producerの役割、現行生成処理とのcanonical値・設定の同一性、実consumerと保護authorityを照合して整理し、
現行benchmark・独立oracle・dated maintenance記録・実行環境は保持した。単なる形式cutoffや名前の一致を廃止判定にしない。
以後は既存の出力先・manifestの役割区分を使い、参照入力、実験記録、生成candidate結果、配布説明を分ける。
benchmarkの実行code/input/独立reference/評価/生成結果を同一の正解データとして扱わない。
新しいcleanup frameworkは不要。当時の物理削除blockは設計課題や資産の必要性とは別の実行条件だった。
利用者の削除後、記録済み対象の残存0と既知native479件の全SHA一致を独立確認した。

## 改良を進める単位と順序

| 順序 | 改良単位 | 触る所有者・受渡し | 同時に除去するもの | 完了を判断する確認 |
|---|---|---|---|---|
| 1 | 入力正規化と完成物公開 R06/R07 | source producer、既存publisher、CLI | raw Anyの先行使用、analysisだけの直接公開 | 不正入力/中断時の公開挙動、有効例題、wheel外一巡 |
| 2 | 解決済みモデル説明 R01 | catalog typed plan→run facts→check/result | 各consumerのモデル別補足、説明できないIDだけの投影 | scalar/role/単位/適用域が元入力なしでも区別可能 |
| 3 | binding・証明family R02/R03 | catalog→既存offset表→typed numeric/proof record | 他カテゴリ型によるoffset再利用、暗黙の残余family | 既存組合せ、追加モデルの局所性、未対応証明拒否 |
| 4 | 科学式authority R04 | forces/charge内scalar→batch/compiled | production同式の二重管理 | 独立式・極限・適用域、科学payload、hot benchmark |
| 5 | accepted transition R05 | transportの共通commit→method別継続 | exact/curvedに重なるstate/ordinal/event更新 | 壁law/corner/残時間/OU/axis、resume/process identity |
| 6 | 派生係数とresult内容 R08/R09 | producer操作/provenance、completion/解釈 | Eだけ交換する手順、identityの保証範囲の曖昧さ | stale検出の対象限定、値再構成、改変copyと費用 |
| 7 | 検証/評価の簡潔化 R11–R14 | 既存tests/bench/V&V/report | private容量固定・用途を超えた一般化・旧候補出力 | 独立oracleを保持、必要範囲を再実行し説明と一致 |

同じchangeで全段階を一度に数値変更しない。受渡しの変更と方程式・補間・RNG・event規則の変更を分ける。
各段階で必要な独立referenceを実行し、影響する例題・CLI・説明・不要経路を同時更新する。
過去全suite成功を新sourceの証拠にしない。

## 追加・拡張・削減での変更範囲

| 変更例 | 現状の波及 | 改良後も必要な科学的変更 | 不要にする波及 |
|---|---|---|---|
| 同区分にdrag追加 | descriptor/plan、式、stage、bound、proof、noise整合、Talbot/lift binding | drag入力/式/適用域、必要なnoise連成、独立極限 | 他カテゴリのoffset選択・engine変更 |
| 新しい加速度区分 | PhysicsPlan固定列、StageBindings、合成、bounds、constant証明、説明 | 加速度とそのbound/適用域、role/説明、既存状態で閉じるか | 既存x/v/Zで閉じる場合のintegrator/event/RNG変更 |
| 粒子温度など新state | source、resident、stage、dense、event、checkpoint、result、memory、解析 | 熱方程式と運動の連成・単位・hit時の意味、保存則 | 任意状態用のODE frameworkを先に作ること |
| 新wall law | boundary応答、RNG、説明、保存則・残時間試験 | 新law固有の分布/エネルギー/charge条件 | first-hit探索の複製、別engine |
| 一modelを削減 | selector/resolver、plan/code、proof、専用入力/例題/検査/文書 | 残るmodelが使うshared primitive・独立oracleの保全 | 互換alias、旧設定reader、役目を失った生成結果 |

出力channelは加算加速度、線形緩和、charge G/J、noiseで違う。新categoryを全て「force callback」へ統一すると、
drag/noise平衡や帯電feedbackを失う。固定のtyped recordを必要なchannelへprepareする範囲に留める。
実際の新カテゴリが定まる前にchannelの汎用登録APIを追加しない。

## 科学・数値を説明する受入方針

輸送の差には、問題同定、初期source、geometry/field転送、model-form、粒子積分、確率path、
event、参照精度、有限標本が関わる。相互作用するため、総差から各項の寄与率を自動で分解できない。

| 確認 | 独立した根拠・実験 | 判断しないこと |
|---|---|---|
| 作用と連成 | 同じx/v/Z/tの個別force/G/J、独立式と極限 | final countだけによる式の正しさ |
| 時間積分 | deterministic解析/製造解、実際のresolution | 保存clockをdtと扱うこと |
| Brownian | joint x/v covariance、条件付きsplit、path解像度、独立ensemble | 同じIDの軌道重なりを乱数非結合の誤差基準にすること |
| geometry/event | 解析hit、壁近似・corner・残時間の独立検証 | 保存点chordだけによる未保存bounceの断定 |
| 場 | 同一点、同じmeasure、off-node転送/回復、mesh | Eの平均4%を全軌道4%精度とすること |
| 用途の対応 | source別空間分布、停止先/時刻、active曲線、速度/charge、coverageと不確かさ | 独立pathのRMSだけ・全停止数だけの判定 |
| 実機予測 | 源rate/weight、capture則、測定との独立照合 | 数値stickやCOMSOL一致だけの物理認定 |

乱数の厳密対応は要求しない。目的に対して十分な集団対応があれば、その範囲を肯定的に判断する。
差・局所留保・不確かさを示し、許容値がなければ結果に合わせた確認的PASSを作らない。
既存A/P40実現等の保存根拠と、新規実行・新規独立標本を区別する。
Vigus43,220粒子や全材料caseは、必要な入力/native artifactが揃った別の実行として扱う。

## 実装前の結論と確認状況

先に直すのは、再現済みの公開入力/出力不具合と、モデル事実・binding・proofの所有である。
その後に式とaccepted transitionを整理すれば、拡張時に修正する箇所を減らしつつ、科学的根拠を保持できる。
full-material、3D、GPU等は必要な利用目的がある場合の能力拡張であり、今回の責務改良と同義ではない。

今回の新規調査は上記の提案を整理したもの。R13の追加撤去・確認を除き、R01–R09/R11/R12/R14を実装したとは主張しない。
旧生成物の物理削除は利用者が実行し、記録済み77,774件の残存0と保護対象の保持を確認した。
現在の撤去結果は[削除後の確認](../solver/evidence/current_only_2026_11_23/execution/post_manual_cleanup.json)、
当時の停止理由は[削除前の記録](../solver/evidence/current_only_2026_11_23/blocked_cleanup.json)へ分ける。
撤去完了は対象不在の確認によるものであり、提案の実装完了や新しい科学的精度認定とはしない。
