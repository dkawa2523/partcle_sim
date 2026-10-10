# 粒子輸送ソルバーの具体的改良設計 2026年10月9日

最優先は、**確認済みの不具合を小さい変更で修正し、利用する物理モデル・場・境界について必要な精度を示すこと**である。cache生成、COMSOL比較、Brownian到達精度、性能最適化は、それぞれの利用条件に応じて進める。全機能の認証が終わるまで、無関係な修正や用途を止める計画にはしない。

本書は[専門レビュー](expert_review_2026-10-09.md)と[物理・数値レビュー](theory_numerics_validity_review_2026-10-09.md)を、変更範囲と合格条件を選べる設計へ整理し、同日の実装結果を追記したものである。G01〜G12は根拠を追うための課題IDであり、全項目を必須機能として追加する一覧ではない。調査基準はHEAD fa51c2e10f32153afbb08d54012ec676f70508c4とレビュー時の変更を含む作業ツリーで、package 0.2.0候補、engine v44、canonical case/data/result schema 3、checkpoint schema 2である。下記の実施範囲を超えたCOMSOL・100万粒子の認証は示さない。

この文書は既存仕様の正本を置換しない。採用する変更は、各作業で既存の仕様ownerへ反映し、旧処理・旧設定・旧説明を同じ変更で除去する。old_codeの環境・packageは利用せず、model_datasetをsolver依存にしない。

### 第1段の実装反映（engine v44の履歴）

以下の課題記述は修正前の調査根拠を残したものである。現行動作は各ownerの仕様・codeとこの実施範囲で読む。

| 作業 | 反映した内容・証拠の範囲 |
|---|---|
| A1/A2 | CI gateを独立stepへ分離。ローカルPowerShellの意図的exit 17を失敗として伝播し、clean-installed wheelの三APIとQuick Startを確認。final容量とmanifest N、非負・狭義昇順IDをpublish/open/readで一つのfile validatorが検査。remote Windows/Linux CIは未実行 |
| A3a/A3b | 巨大数値を既存domain例外へ翻訳。core・adapter・builder・preprocessor・C1 templateが同じ狭いYAML文法を使用。nested duplicate/merge拒否、通常aliasとraw-byte hash保持。producer入力formatはv2へ更新 |
| A4 | Talbotを半径Knの明示Cs,Cm,Ctへ置換し、原著・径変換・漸近・bound・pure/compiled・公開scenarioを検証。catalog v23、runtime v22、compiled tile v21。engine/proposal/event/RNG/schemaは据置き |
| B1 | static P1→full regularをcommon partitionとXY/RZ weighted value/gradient/support-boundary L2で判定。coverage穴/重複、未解像patch、資源超過、未対応time/Q1を無publishで拒否。固定点gateは除去。独立レビューで追加発見した、大原点でのbilinear gradient丸め上界不足も混合二階差のallowanceで修正し、解析sqrt(1/3)のtight-budget false-publishを拒否 |
| C1 | current writer fixture、recipe executor記録、prepare/runとactual resultの照合を実装。履歴監査・既存result再投影は現在source identityと区別。可搬な外部CI試験とdatasetを要するローカル監査を分離 |
| C2 | schema-2 preflightは期待・観測artifactのhash、JSON pointer、型、値を再検査。C2 assembler/evaluatorとC3 evaluatorはsource model、canonical field、対象実receipt集合との対応も要求。C3はprepared configとsummary/run receipt両方の実行config hashを照合。status文字列、別caseのinventory、未知revision、改変summary/実行configでは合格しない。終端statusからinletを推測する旧辞書を除去し、不完全なboundary mapと未観測原因をAMBIGUOUS/NOT_TESTEDとして保持 |
| C2の実機観測 | COMSOL 6.4のfrom-scratch RZ境界caseは[132/132 gateを合格](../solver/evidence/m3c0/critical_axis_native_2026_10_09_v1/README.md)。[Case-P限定probe](../solver/evidence/m3c2/actual_binding_probe_2026_10_09_v1/scoped_probe_receipt.json)はhash固定の履歴P1表上でsource/companion設定と共通βの数値形成を観測し、source MPHのhash不変を確認。drag/Brownian βは最大1 ULP差。CSV単位headerの不整合とbatch wrapperのError statusが残り、単位認証・wrapper正常完了・現行canonical field parityはNOT_TESTED。実assembly RHS・terminal boundary ID/原因、新ensembleの認証へ拡張しない |
| H1/M/D1 | method別境界budgetとaccepted prefix/residualの意味を整理。Barnes弱drift、aggregate引力増分、mixtureの独立極限を検証。OU期待共分散を独立oracleへ統一し、時間線形係数・位置依存流速の実noise ensembleとharmonic dense local/globalを追加。用途全体の物理validationや一般SDE次数は認定しない |
| F0/E | [現行10k warm/cold baseline](improvement_baseline_2026-10-09.json)と[profile](improvement_profile_2026-10-09.json)を保存。測定した1macro/1slabではbuffer allocationは支配費用でなく、staging再利用は保留。未測定の長時間restartへ一般化しない |

C2のsoftware gateと限定実機観測を反映した。第1段ではC3の新しい登録ensemble認証、native export単位の不整合解消、F1の用途accuracy/SLAは別の完了判定とし、B2/H2/D2は条件付き能力として残した。後続の実施結果は次節を権威とし、旧campaignの合格を現行修正の再認証へ流用しない。

第1段の検証は、core verification/scenario **811件合格**、adapter/builder/preprocessor/COMSOL外部tool **463件合格**。後者は現行fixtureと履歴artifact監査の試験であり、463件のnative COMSOL solveを意味しない。可搬な外部CI subsetは92件合格した。`uv lock --check`、Ruff format/lint（examplesを含む）、Pyrefly、三つのimport契約、CC<16、diff whitespace検査も合格。OSのapplication-controlでCLI launcherが拒否されるimport-linterは同じPython entryを使用した。

第1段のcoreの27 Python fileとwheelのbytes一致、runtime-only clean installの三API、Quick Startの生成/check/run/inspect/analysis/visualization、P14の8-row・64粒子smokeを確認した。Windows/Linuxのremote CI、現行100万粒子の用途認証、新COMSOL ensembleは未実行。性能は上記のlocal baselineとprofileの範囲に限定する。

### 2026-10-09の残計画の実装反映（engine v45・履歴）

後続の依頼によりB2/H2/D2を実施した。このcloseout時点はengine v45、event v21、geometry v7、source v5、memory plan v16、
gradient v2で、physics catalog v23/runtime v22/compiled tile v21/proposal v10と各schemaは維持する。
第1段の観測・測定artifactは履歴として固定し、新しい入力と現行executorを使う外部比較を別に記録する。

| 作業 | 現行実装と検証範囲 |
|---|---|
| B2 | P1/regular/exact affine Q1→full regularのstatic/linear-time cache。共通partitionのXY/RZ normと時間Gram/Bernstein比上界で全区間を判定。warped Q1/partial target/未解決reference下界/資源超過は無公開。同じ値/gradient samplerでsnapshot static viewを検証し、時間Gramを外部で構成。第二locatorなし。関連71試験合格 |
| H2 | groupのparticle_surface/particle_centerを明示。粒子radiusを保持し、一mesh/BVHでcandidateの法線/残差/source offset/clearanceを解決。平面・開口端cap・混在順序・XY/RZ・曲線・半径・slab/output/resumeの16試験合格。端cap離脱と反射originのmacro保持の反例を修正し、曖昧な同時modeをfail-closedに維持 |
| D1/D2 | 非dyadic Hermite covariance、charge/Coulomb mean、folded RZ OU、中心判定の実noise identityを追加。独立FV Kramers referenceのgrid/dt/dv/domain/mass確認と16k ensemble first-arrival CDFを別評価。[D2証拠](d2_kramers_first_arrival_2026-10-09.json)はMC幅と経験的reference indicatorを分離。有限Hermiteの連続OU first-passage exactness/一般次数を認定しない |
| C2：実使用条件 | COMSOL 6.4.0.429のcurrent field/controlでactual設定とSI単位を観測。Interpolation.importData後に失われるargunitを共有ownerで再設定。Freeze/Stickのboundary IDを小controlで観測し、DisappearのID未観測は保持。CaseA/Pのsource→companion、共通β、axis-normalized Ωφ、RNGの意味とhashを新登録へ固定 |
| C3：決定論 | COMSOL 5/2.5/1.25µs・candidate 2.5/1.25/0.625µsの6実行完了。状態の自己収束は約2次、全lifecycleと141件のboundary ID/group一致。古いcandidate CSVがactual IDを固定labelへ落とす不具合を修正し、元結果から再projection。元CSV/receipt/初回BLOCKEDを保持し、計算条件・seed・閾値を変更しない。[再判定](../solver/evidence/comsol_binding_recert_2026_10_09_v1/c3_axis_normalized_after_projection_evaluation.json)は登録済み経験的判定でPASS |
| C3：Brownian人口比較（M3-C2） | Case-A/PそれぞれCOMSOL32＋candidate32の未使用seed、固定20µs/depth3で全128実行完了。[Case-A](../solver/evidence/comsol_binding_recert_2026_10_09_v1/caseA_axis_normalized_final_evaluation/evaluation_manifest.json)・[Case-P](../solver/evidence/comsol_binding_recert_2026_10_09_v1/caseP_axis_normalized_final_evaluation/evaluation_manifest.json)とも登録終端/RZ分布gateはPASS。独立集計が128 CSV・4,445,056行、3,904意味binding、64 native seed/dt/RNG readbackを照合。pathwise一致や連続SDEの精度を認定しない |
| F0/F1 | [現行性能受入記録](product_performance_acceptance_2026-10-09.json)に、解析小case、10kの同条件before/after、100k/1M単回、10k混在contactの測定を保存。全対象のfailure 0・計画memory、default 10kのbefore/after payload一致、小解析caseと混在case全rowの解析一致を確認。利用者の用途accuracy/SLAは未指定で、工学的測定scopeの完了と製品SLA認証を区別 |
| E | 第1段のprofileでstaging allocationは支配費用でないことを確認し、再利用への置換を採用しない判断を維持。必要性のない第二workspaceや並列backendを追加しない |

現行core verification/scenarioは**838件合格**、最終修正後の外部tool全試験は**486件合格**した。uv lock、Ruff format/lint、
Pyrefly、三つのimport契約、CC<16を確認。現行wheelの27 Python fileはsourceとbytes一致し、runtime-only
clean installの三API、Quick Startの生成/check/run/inspect/analysis/visualization、
[8-row・64粒子の現行smoke](final_current_smoke_2026-10-09.json)を確認した。486件はnative solveの件数ではない。
最終外部試験で見つかった旧診断optionとartifact集合の2件の期待値は、現行の出力・境界identity契約に合わせて修正した。
productionの計算条件は変更せず、全486件を再実行して合格した。

性能は同一machine・private cache・fresh childの測定である。10kのwarm中央値はregular 0.422 s、20-hit event
4.586 sで、第1段baselineに対して約4.1%/8.7%遅く、速度改善を主張しない。両caseのscientific payloadは完全一致し、
solver-owned memory planの増分は0。regularの100k/1Mはsimulate 4.521/46.172 s、peak process RSSは約542/874 MB、
solver-owned planは約315/706 MB。混在contactの10kは0.358 sで解析的fate/time/position/radiusを全rowで確認した。
これらは選択した決定論presetの費用であり、全複合物理、Brownian到達、COMSOL速度比、他machineのSLAではない。

決定論のdirect差は位置RMS 1.0982e-7 m、速度RMS 3.5141e-4 m/s、電荷RMS 0.0015065 e、event timeはRMS
4.7024 ns/max 27.5271 ns。PASSは事前登録の4×（双方のfine-pair差）という経験的許容量を含む判定で、
絶対floor単独では位置・電荷がFAILである。厳密な誤差上界や任意の実用途の絶対精度を認定しない。
同一native fine solveの出力時刻で共通active 25,503行を独立検査し、native合力変数Ftr/Ftzと7力の再構成和は
rel L2 7.83448e-17、最大成分差4.03897e-28 Nで一致した。各内部stageのassembly、auxq actual RHS、各native寄与は未観測。
[同一形状上の軌道図](../solver/evidence/comsol_binding_recert_2026_10_09_v1/derived_c3/c3_geometry_overlay.svg)は、
保存済みの287粒子から固定12 IDを選び、同じR-Z座標・縮尺でgeometryと境界eventを重ねた。
入力・結果hashの一致と描画receiptを保存し、見た目の重なりを数値認証へ流用しない。

Brownianの新pilot→held-out final評価を、独立pre-outcome監査を通過した入力・seed・統計条件で完了した。
各系は32×287=9,184粒子で、121時刻はunion bound対象としsample数へ乗算しない。
overall α=.05を両caseと二gateへ配分した登録条件で、終端曲線の差の同時信頼上限は
Case-A **0.040128262937**、Case-P **0.035010667118**（margin .05）、
固定80 R-Z bin＋3終端区分のTV上限は **0.130735400519 / 0.129537665328**（margin .15）となり、両caseとも合格した。
これは固定source、相互作用なし、独立particle/seed streamというsampling条件下の登録人口observableの比較である。
初期係数/FDTとSI単位、実seed/dt/RNG getter、正常完了guard、source MPH hash不変を数値比較と分けて確認した。
source既定のd0=10nmと登録100nm overrideを区別し、read-only controlでρpと登録massの一致を観測した。
後からの補強観測を事前登録や各replicaの新しいnumeric mass getterへ書き換えない。

[独立監査](comsol_recert_independent_review_2026-10-09.json)は結果と登録radiusを別実装で再集計して一致を確認した。
Case-Aのnative 6,904/candidate 6,910件のescaped/disappearedは観測status/timeとして比較したが、native境界ID/原因は未観測。
Case-Pは両系全9,184粒子activeであり、この比較からCase-Pのwall応答を認定しない。
連続noise time-law・連続SDE bias・個別native寄与/auxq RHS・全内部stage・native FE/curl・実験的な物理妥当性は未認証のまま保持する。
未観測scopeを第1段の合格で代用しない。今回の実装、品質gate、release smoke、登録比較の完了状態とsource hashは
[最終検証記録](current_revision_verification_2026-10-09.json)にまとめた。
warped Q1 cache認証、一般3D、利用者の用途accuracy/SLAは今回の完了scopeに含めない。

### 現在の残件と着手順（2026年10月10日）

2026-10-09のengine v45の改良・比較は[履歴検証記録](current_revision_verification_2026-10-09.json)、
engine v46 / event v22の追加改良とlocal Windows/Linux検証は
[候補検証記録](v0_2_candidate_verification_2026-10-10.json)として固定する。
0.2.0の用途別保証は[固定受入条件](v0_2_release_use_case_acceptance_2026-10-10.json)に列挙した
入力・数値設定・指標・budgetに対する工学的検証である。利用者が承認したSLAや実験的な製造予測へ読み替えない。
科学的な結果は[0.2.0統合検証記録](v0_2_release_qualification_2026-10-10.json)、
実際の公開結果は[公開後の確認記録](v0_2_release_publication_2026-10-10.json)が所有する。
0.2.0は宣言した用途別保証と正式公開を完了した。以下の後続は保証範囲の追加が必要な用途だけを対象とする。

| 対象 | 0.2.0の状態と完了条件 | 追加主張が必要な用途だけの後続 |
|---|---|---|
| F1：用途条件 | 完了。決定論、線形OU、限定first-arrival、model式・適用域のprofileと数値budgetを固定。性能は登録機での測定とcompletion・memory plan・identityの検証とし、絶対時間SLAを設けない | 使用装置・parameter・粒子数・出力・hardware・必要KPIとSLAを定め、対象caseで適格性を確認する |
| D1/D2・C3：数値精度 | 完了。既存独立参照に加え、固定した決定論解析解2件と独立OU/chargeの3 profileを公開LF sourceで再実行し合格。元N16k RZ D2のNOT_METと別seed/N64kの独立PASSは両方保持 | 一般time/state-dependent到達精度、使用field・wallでの必要KPIを独立参照とh/depth系列で検証する。有限Hermiteから連続OU exact first-passageやzero-missを主張しない |
| H1/H2・C2/C3：境界とCOMSOL | 登録scopeで完了。現行v46のC3三刻み・Case-A/Case-P各32 runをhash固定した保存native referenceと再比較し、登録gateに合格。独立監査と公開用620 authorityのSHA照合も完了。四判断は初期条件PASS、完全RHS NOT_TESTED、完全境界挙動NOT_TESTED、時系列PASSとして保持 | Case-AのDisappear原因/境界ID、Case-Pの今回未発生wall応答、専用pre/post速度・残substepを保証する場合は実観測を追加する。C3 fine-pair allowanceは絶対誤差boundではない |
| M/G11：物理式と場 | 完了。独立式・極限・適用域・感度の既存検証を採用し、選択した既存Talbot/Saffman・Barnes・aggregate・effective-gasの44試験を公開LF sourceで再確認して合格 | 実装式の検証と実物理のvalidationを分ける。実験的予測、個別native force/auxq RHS/全内部stage、native FE/curl、壁近傍局所場の精度を必要な用途で確認する |
| F1/E：性能 | 完了。登録P14全19条件×3反復（57観測）、B03全8条件×3反復（24観測）、P14-Uの18観測・別profile・収束/RZ確認は全て合格。計99観測で、source/recipe/lockの事前登録との一致とLF公開sourceへの対応を保存 | 別hardwareや未測定model/scaleのSLAは対象条件で測る。全用途へのspeedupを主張せず、未達hotspotが実測された場合に同精度・同科学payloadで改良する |
| A1/P14-R：正式公開 | 完了。release/chamber-particles-0.2.0とv0.2.0をpushし、同一資格commit ba70a2cのbranch/tag CIは両OSで合格。tag CIは各OSで847 core＋120 portable試験、静的gate、Quick Start、性能smoke、wheelとclean installを完了。GitHub Releaseのwheel・SHA256SUMS・Release Notesを公開し、実download bytes・source 27件・UTF8 METADATAを照合した | 別OS/Python版、PyPI、sdist等は今回の配布scopeに含めない |

用途別保証は固定profile内の検証であり、各実行の自動的な誤差上限や全物理条件を保証するものではない。
安全failureと精度不足、数値・MC・reference・field表現の不確かさを区別する。
Windowsのローカルconsole launcher拒否4551は当該hostのAppControl制約であり、tag CIの両OS console合格と、
公開wheelのローカルAPI/module CLI合格とは分けて記録する。policyやsourceは変更していない。
完了したfailure injection回帰やlocal両OS gateを実装残件へ戻さない。

**条件付き後続**は、warped Q1/partial-support cache認証、一般3D/等方3D Brownian、GPU等である。
warped Q1のmesh-native評価自体は対応済みで、未対応なのはcache認証の範囲である。
実利用で必要になるまで能力拡張を採用しない。Eのstaging再利用・並列backendは不採用判断で閉じており、
具体的な変更理由・state寿命・計測根拠のないmodule分割や第二engineを追加しない。

本節は着手・完了状態を整理したもので、固定済み科学証拠のcriteria・source hash・NOT_MET/NOT_TESTEDを変更しない。

## 1. 現行処理を踏まえた改良の配置

三つの公開入口と、一つのproduction engineを維持する。

~~~text
load_case
  → case: YAML/spec、参照・構造検査
  → case_format: canonical HDF5、logical hash、footprint

simulate
  → engine._prepare: geometry、fields、models、schedule、memoryを解決
  → 一つのmacro-step / slab / accepted-piece loop
      → fields: 実stage位置・時刻のprimitive評価
      → physics.runtime / compiled: 力・帯電・摩擦・適用域
      → integrators: coupled proposal、dense state
      → events: first-hit、support、prefix、残時間
      → boundaries: 解決済みwall law
      → accepted state、field hint、物理RNG ordinalだけをcommit
  → output: segment/checkpoint commit、final、manifest、_SUCCESS

open_result
  → output: committed segmentとfinalの整合確認
  → lazy ResultView / bounded event iterator

外部tools
  raw export → canonical producer → field cache品質 → 公開三API
  COMSOL設定・実評価 → binding receipt → meaning preflight → V&V
~~~

根拠は[API](../solver/src/chamber_particles/api.py:36)、[prepare](../solver/src/chamber_particles/engine.py:3626)、[run loop](../solver/src/chamber_particles/engine.py:1031)、[field owner](../solver/src/chamber_particles/fields.py:193)、[durable writer](../solver/src/chamber_particles/output.py:605)である。

改良後も、積分器がCOMSOLを知る、physicsがgeometryやHDF5を読む、outputが別の物理検証を行う、といった責務の混在は作らない。汎用性はケース名による分岐をなくし、補間方式・物理量・状態寿命・比較意味を明示することで得る。任意pluginや第二engineは必要ない。

## 2. 課題の状態と適用する作業

### 対象用途から必要な作業を選ぶ

着手前に、既存caseと製品要件から、使用model revision、XY/RZ、field layoutと時間依存、cache利用、boundary lawと接触方式、出力mode、必要KPI・粒子数・計算時間を確認する。この選択はcase/manifestと既存製品要件へ記録し、新しい設定fileや汎用acceptance frameworkは作らない。

| 作業の種類 | 対象 | 着手・完了の判断 |
|---|---|---|
| 確認済み不具合の修正 | G01/G04/G05、Talbotを所有するG10、cache品質gateのG02 | 原因を表す反例が拒否され、正常入力・結果・式を維持できる。G02/G10は該当機能の修正であり、無関係な軌道の欠陥ではない |
| 外部比較の整合 | G03/G06、G12のCOMSOL側設定と観測 | 対象producerの実使用設定・変換・観測範囲を照合してから比較する。coreの常時COMSOL gateにはしない |
| 数値・モデル精度の追加証拠 | G07/G11、G12のboundary accuracy | 使用method/modelと必要KPIの誤差を独立referenceで測る。検証不足をそのままアルゴリズムの不具合と呼ばない |
| 計測に基づく最適化 | G08/G09 | 現行baselineで時間・memory・allocationの支配箇所を確認し、同じ精度・科学的payloadを維持して改善する |
| 条件付き能力拡張 | G12の混在contact geometry、warped Q1/time cacheの追加認証、将来3D | 実利用で既存能力では足りない場合だけ採用。非対象用途の完了条件へ入れない |

### 優先順位と根拠

| ID | 課題 | 確認状況と影響 | 改良の主owner | 優先度 |
|---|---|---|---|---|
| G01 | PowerShell CIが途中失敗を見逃す | 非zero終了後の成功でstepが成功する反例を確認 | solver-release.yml | 最優先・短期 |
| G02 | field cacheの固定点品質gate | 確認済み：局所featureを100%消失しても誤差0でpublish。mesh-native samplerの誤評価ではない | field_preprocessor、必要な補間微分はfields/cpu | 最優先・cache生成 |
| G10 | Talbotの原著とKn・係数の結び付けが不一致 | 確認済み：半径Knの未変換係数を直径Knへ使用。高Knで力比2へ向かう | physics forces/catalog/runtime/compiled | 最優先・Talbot model |
| G03 | COMSOLの実使用物理・境界を認証していない | Case-P source selector、両caseの軸Freeze継承、補間後FDT係数差最大約0.461%を確認 | 外部runner、normalizer、preflight | 最優先・外部比較 |
| G04 | finalとmanifestの粒子数・ID整合 | 全列同時短縮、重複ID、逆順IDの受理を確認 | output | 高 |
| G05 | 巨大数値の例外漏れ、YAML方針の不一致 | 公開例外からOverflowErrorが漏れる。builder/preprocessorは重複keyを受理 | case、physics.catalog、各producer | 高 |
| G06 | 歴史evidenceと現行executorの混在 | 実測35 failed / 14 errorsは外部tools全体の当時の結果。個別原因を分類してcurrent fixture/recipeを修正 | 外部prepare/runner/tests | 高・現行外部再実行 |
| G11 | 近似モデルの誤差と入力producerの保証が混在 | model-form差：Barnes弱drift collectionの25%差、aggregate引力増分π/4。mixture等は追加証拠が必要 | physics仕様、既存verification、外部producer/V&V | 使用modelの用途精度を決める前 |
| G07 | endpoint・dense・確率経路・event精度の証拠が異なる | 説明とaccuracy scopeの課題。RK4 denseは別cubic、限定OU moment検証は一般SDEを認定しない | integrators/stochastic verification、numerics、AGENTS | 使用methodの精度認証前 |
| G12 | 境界判定optionと外部比較の意味が未整理 | method別budgetの説明差と比較意味の差。混在contact geometryは新機能であり確定bugではない | events/geometry/engineの説明、外部boundary V&V。条件付きcase/source拡張 | 説明修正は先行。外部認証・能力拡張は用途別 |
| G08 | event/restart scratchの反復allocation | allocationは確認済み。実用speedup・必要性は未計測 | engine、cpu memory plan | 対象hotspotと寿命を確認後 |
| G09 | 現行複合物理の規模別・等精度性能実証 | 現行64粒子smokeと過去1M実測が別revision。現行性能不良を確定した結果ではない | performance harness、製品要件 | baselineは先行可。性能認証は使用機能の精度確認後 |

G02の追加反例は前回レビューの指摘を強める。通常のproduction engineが局所場を誤評価するという反例ではなく、**不正確なcacheを正確と認定して選ばせる外部gateの欠陥**である。G03も、過去の統計計算全体を無効とする指摘ではない。登録済みpopulation指標の合格と、same canonical RHSの認証を分けて扱う。

### 汎用化する対象と、所有者を増やさない方針

| 汎用化する事実 | 一つのowner | 具体的な扱い |
|---|---|---|
| 物理式・規約・係数 | physicsのmodel revisionと既存計画型 | 原著式へ同値変換し、明示係数をstageへ渡す |
| どの場を評価するか | canonical field＋既存sampler | 同じ位置・時刻・basisでprimitiveを補間してから非線形式を評価する |
| producer固有のfeature・境界 | external adapter/runner | source、変換後companion、実read-backを別々に保存し、一般的な意味へ正規化する |
| 比較を始めてよいか | 既存meaning_preflight | hash付き実証拠と期待値を照合する。COMSOL APIは呼ばない |
| 数値曲線と精度 | integrator/proposal、既存verification | endpoint、dense、表現されたfirst-hit、連続物理過程を区別する |
| 完成と再現性 | output、既存contract/hash lock、追加するexecutor identity | 既知N・ID・actual revisionを検査してpublish/replayする |

Case-A/Pをcoreの条件分岐にしない。新しいplugin registry、一般validation framework、別engineは設けない。改善に伴う新しい設定は、必要な精度・資源・source変換だけを既存ownerへ追加する。

### 設定追加と実装詳細の決め方

| 判断 | 本計画での扱い |
|---|---|
| 現行設定で表せる | dt、geometry budget、Brownian base/adaptive depth、modelの明示係数を利用し、同義の精度presetや別budget keyを増やさない |
| 複数の実consumerがある | 四つのYAML入口や二つのterminal normalizerだけを狭いleafへ統合。domain検査・I/O・物理式まで共通frameworkへ移さない |
| producer固有の設定 | COMSOL order/storage/cap、native field selector、source→companion変換はadapter/runnerが所有。coreは物理・数値の意味を受ける |
| 用途に必要な新しい意味 | cacheの有界検証資源、混在contact geometryだけを該当ownerへ追加。不要な用途では要求しない |
| 内部配列のshape・再利用方式 | 下記は候補設計。実装時のcall graph・同時生存・profileで決め、私有helper名やallocation回数をtest契約にしない |

各変更は、解く症状、適用条件、一つのowner、独立期待値、置換・削除する旧経路、完了の証拠を持つ。合格後に新しい条件や全組合せtestを無制限に加えず、残る未認証範囲をその用途のscopeとして記録する。

## 3. G01：CIを各品質gateで確実に停止させる

### 現行の原因

[solver-release.yml](../../.github/workflows/solver-release.yml:45)は、複数のnative commandを一つのpwsh stepへ記載する。GitHubのpwsh処理は最後の終了コードを確認するため、途中のuv/Ruff/Pyrefly等の失敗を後続成功が覆える。ErrorActionPreferenceだけではnative commandの失敗を一律に停止できない。[GitHubのshell仕様](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#exit-codes-and-error-action-preference)

### 変更案

品質gateは一つのnative commandにつき一つのstepへ分ける。

~~~yaml
- name: Check lock
  run: uv lock --check
- name: Check formatting
  run: uv run --locked ruff format --check src tests tools
- name: Check lint
  run: uv run --locked ruff check src tests tools
- name: Check types
  run: uv run --locked pyrefly check --summarize-errors
- name: Check imports
  run: uv run --locked lint-imports
- name: Check complexity
  run: uv run --locked python scripts/check_complexity.py
- name: Verify solver behavior
  run: uv run --locked python -m pytest tests/verification tests/scenarios -q
~~~

Quick Startも生成、check、run、inspect、analysis、visualizationを独立stepにする。step間でPowerShellのローカル変数は保持されないため、runner.temp配下の同じ明示pathを各stepで構築する。

wheel作成、clean runtime準備、wheel install、release smokeも分ける。runtime path、Python executable、wheel pathは必要な値だけGITHUB_ENV/GITHUB_OUTPUTへ渡す。UV_PROJECT_ENVIRONMENTの設定を一つ前のstepのローカル状態へ依存させない。path選択などの理由でnative commandを同一stepへ残す場合は、その直後に次を置く。

~~~powershell
uv sync --locked --no-dev --no-install-project
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
~~~

publishのgh release viewは、既存releaseの有無を調べる意図した分岐である。この終了コードを通常gateと一括して扱わず、upload/createの結果を確実に確認する。新しいCI runner frameworkは作らない。

**最小確認：** 一時的なCI branchで最初・中間gateを意図的に失敗させ、後続stepとartifact uploadへ進まないことをWindows/Linux両方で確認する。その後、正常pipelineでclean-installed wheelの三APIとCLIを確認する。workflow変更に物理revision更新は不要である。

## 4. G02：場cacheをsource/target共通領域で認証する

この修正はcacheを生成・選択する用途に適用する。mesh-native fieldを直接使うcore計算は、cache認証完了を待つ必要がない。先行変更では、実利用する静的layoutの組を一つ選び、正しい指標でpublish可否を閉じる。未対応layoutは明示拒否し、同じ共通partition処理で多項式layout、warped Q1、時間区間へ順に広げる。

### 原因を固定する回帰

[resample_field](../solver/tools/field_preprocessor/numerics.py:190)はproduction samplerを使ってtarget node値を生成する。この方式自体は維持する。欠陥は、その後の[_validation_metrics](../solver/tools/field_preprocessor/numerics.py:257)がtarget triangleの3点、quad/regularの4点と外周midpointだけを使うことである。

追加確認した最小caseは、sourceの両axisが [0, 0.09, 0.10, 0.11, 1] のregular場、25nodeのうち (0.10, 0.10) だけ1、他は0、targetが単位正方形1cellである。

~~~text
cacheの4node                       すべて0
source中心                         1
現行value/gradient/boundary誤差     すべて0
全acceptance limitが0               cache.h5をpublish

独立tensor-hat積分
value L2 error                     0.02/3 = 0.0066666667
value relative L2 error            1
gradient L2 error                  sqrt(8/3) = 1.63299316
gradient relative L2 error         1
~~~

source nodeを追加するだけならこの例を検出できる。しかし、cell面積の重み、実勾配、境界交差、Q1内部変化まで認証できない。source nodeは補助witnessとし、合否の主証拠を次の積分へ置換する。

### 共通partitionとsupport

現行[ensure_target_covered](../solver/tools/field_preprocessor/numerics.py:145)が使う凸cell交差を、交差面積だけでなく交差polygonを返す処理へ変更する。

~~~text
target supported cell
  → source AABB candidate
  → source_cell_id / target_cell_id付き交差polygon
  → triangle patch
  → 値・勾配・reference normを重み付きで逐次積分
~~~

sourceの小cellを先にpartitionへ取り込むため、局所featureを初期標本が見ず、adaptive分割も起きない失敗を防げる。全cell pairを保持せず、targetごとの候補とpatchをstreamする。現行の全cell検査点配列を除去する。

coverageでは、gapとpositive-area overlapを別々に検査する。個々のcanonical cellの向き・Jacobian検査は、全cellの非重複の証明ではない。交差面積の総和だけではgapとoverlapの相殺を排除できない。source coverage multiplicity=1を確認し、target自体の重複も拒否する。float64で判定できないsliverは固定epsilonで消さず、未認証としてpublishを止める。

これはexternal cache工具の責務である。solverのgeometryへcache専用処理を入れない。

### 指標の意味を固定する

Ωをtarget supported cell union、u_sを元の補間場、u_cをcacheとする。

\[
E_v^2=\frac{\int_\Omega\|u_c-u_s\|^2\,d\mu}
                 {\int_\Omega\|u_s\|^2\,d\mu},\qquad
E_g^2=\frac{\int_\Omega\|\nabla u_c-\nabla u_s\|_F^2\,d\mu}
                 {\int_\Omega\|\nabla u_s\|_F^2\,d\mu}.
\]

XYは dμ=dx dy、RZは dμ=2πr dr dzとし、measureをreportに記録する。ここでRZの勾配は、stored componentをcanonical r,z座標で微分したものである。3D vectorの全covariant gradientを認証する意味には広げない。

境界値誤差はΓ=∂Ωについて、XYではds、RZでは2πr dsで積分する。source/target cell edgeとの交点で境界segmentを分割する。boundary_valueはsupport境界の指標であり、実チャンバーmaterial wallの指標ではない。必要なら既存geometryのfacetを同じ積分処理へ渡し、material_boundary_valueとして別名・別limitを定義する。RZ axis seamの回転面積は0である。

reference normが0なら、errorも0と認証できた場合だけrelative error=0とする。errorが非zero、または判定不能ならrejectする。分母へepsilonを足さない。global L2合格は狭い領域の最大誤差を保証しない。局所精度を要求する用途に限り、単位を持つabsolute L∞上界と明示limitを追加する。

### 補間勾配は既存field ownerへ置く

[_affine_gradients](../solver/tools/field_preprocessor/numerics.py:365)のleast-squares affine fitを合否判定から除去する。

[PreparedFieldSet](../solver/src/chamber_particles/fields.py:193)と[cpuの補間leaf](../solver/src/chamber_particles/cpu.py:1241)で、既存locator、cell ID、basisを共有した必要最小限の解析微分を所有する。値の再samplingやQ1 inverse mappingを外部toolへ複製しない。先行layoutのgradientだけを実装し、全layoutの認証APIを一度に作らない。

- P1は定数reference gradientをJacobian逆転置で変換する。
- regularは既存bilinear basisの解析微分を使う。
- Q1は ∇x u=J^{-T}∇ξη u を使い、既存Q1 Jacobian処理を共用する。
- coreのoptional gradient v2はstatic nodal fieldに限定する。time cacheの検証は外部toolが隣接snapshotをstatic viewとして既存samplerへ渡し、空間momentから時間Gramを構成する。time-dependent gradient APIや第二locatorは追加しない。

point value/gradientの誤差囲い、location residual、Hessian boundは、warped Q1の保証を実装する段階で必要性を判断する。先行する多項式積分の必須APIにしない。採用する場合もfield ownerの狭いopt-in出力と固定row workspaceへ限定し、非利用のproduction engineへ計算・allocationを追加しない。配列shapeは実装時の消費箇所で決める。これはcache検査用の補間微分であり、thermophoresisやDEPのproducer-owned primitiveをcoreで再生成する変更ではない。

### P1/regularとwarped Q1を段階的に閉じる

現行producer v4はP1/regular/exact affine Q1からfull regular targetへのstatic・fixed-topology linear-time cacheを認証する。[公開cache例](../solver/tools/field_preprocessor/README.md:15)と[workflow試験](../solver/tools/field_preprocessor/tests/test_preprocessor.py:65)を同じ経路で実行する。sourceが全target supported cellを覆うことを共通partitionで確認し、非矩形・穴付きsupportを外接矩形で埋めて同等化しない。affine Q1はstored geometryの厳密な条件で判定し、1 ULPのwarpもaffineへ丸めない。共通patch内のP1/regular/affine Q1は多項式である。値二乗はP1同士でdegree 2、bilinearを含めると最大total degree 4、RZのr重みを含めるとdegree 5である。次数を覆う正weightの積分則とroundoff marginを、独立manufactured polynomialで確認した。

一般warped Q1 cacheは現行認証scopeから外し、validation_unavailable_for_warped_q1で無publishにする。direct runtime samplingの対応とは別である。将来この認証を必要とする場合も、高次quadratureのcoarse/fine差を保証上界と呼ばず、次の保守的boundと計算roundoffを独立に確認する。

1. Jacobian determinantの正の下界、J^{-1}の上界。
2. nodal値とbasis微分による勾配上界。
3. mapping微分を含むHessian上界。

\[
H_xu=J^{-T}\left(H_{\xi\eta}u-\sum_k
  \frac{\partial u}{\partial x_k}H_{\xi\eta}x_k\right)J^{-1}.
\]

patchの代表点qと半径ρから、値を u(q)±Lρ、勾配を ∇u(q)±Hρで囲い、error/reference二乗積分の上下界を作る。Q1逆写像の残差、point評価、Jacobian/微分、polygon面積、moment累積のfloat誤差も囲いに含める。解析式を実装しただけでは計算された上下界の保証にならない。外向きroundingと確認済みroundoff boundで認証できなければunresolved_validationとする。

~~~text
error_upper <= limit² × reference_lower    合格
error_lower >  limit² × reference_upper    不合格
それ以外                                   patch追加分割
depth/work/memory budget終了                unresolved_validation、無publish
~~~

embedded quadrature差は分割順の判断と推定値reportに使う。合否の保証を置換しない。zero limitや閾値直近では有限budgetで認証できないことがある。その場合の正しい結果は未認証である。

完成したpairだけを認証scopeとして記録する。warped Q1、partial target、coverage不足、未解決reference下界、資源超過はpublishしない。tensor-hatのregular pairは独立weighted normで誤cacheを拒否する。旧固定点の合否gateは除去し、fallbackを残さない。warped Q1の未対応と、static/linear-time対応pairの修正完了を分ける。

### 時間依存場と資源budget

time knotsは保持する。snapshotだけのrelative error判定では、中間時刻のreference相殺を見逃す。各time interval内は双方がlinearなので、二乗norm・cross momentは時間のquadraticになる。現行はerror/reference Gramの丸め囲いをBernstein係数へ変換し、分母の正の下界と比の保守的上界で全区間を判定する。未解決区間は有界な再分割を行い、52段の内部上限または資源budgetで解決しなければ無publish。stationary pointの計算はreport-onlyであり、合否を所有しない。時間平均reportでは不均一Δtを重み付けする。

空間積分が区間で認証される場合は、moment係数の区間、分母の非負性/零点、stationary pointの位置の不確かさまで伝播させる。Q1の推定momentから一つの極値を計算するだけでは合否の保証にならない。時間区間を追加分割し、budget内で上限を決められなければ未認証としてrejectする。

producer v4は明示指定のinterior knot omission感度をreport-onlyで生成する。原場の時間解像度が物理的に十分かというproducer側の判断と、与えられた線形snapshot場に対するcache誤差の認証を分ける。診断の未指定・資源不足を適格cacheの拒否へ流用せず、未観測pulseや連続原場の精度は認証しない。

現行PreprocessorSpecificationのvalidationはmemory_limit_mb/workspace_rows/max_patch_workの三つであり、field acceptance limitとは別の資源budgetを一度だけ解決する。52段の時間比上界の内部深さを第二の公開精度設定にしない。source読込みは既存read_with_infoのnumeric_array_limit_bytesを使い、取得footprintを元にsource/candidate resident配列、polygon/index、scratch、writerに必要な配列をresample前に見積もる。時間区間対応のvalidation scratchは隣接2snapshotと有界なdepth-first処理へ限定する。現行workflowではsource全値とtarget全snapshot配列がresidentなので、このscratch制限を総memoryが2snapshotだけになる保証とはしない。資源preflightはpublish前に成立させる。

config format v2、preprocessor/validation/support/field-semantics revisionを更新し、gradient/boundのrevisionをprovenanceへ記録する。productionの値評価を変えなければcase/data/result schema、engine、既存value samplerの数値revisionは据置きにできる。

**先行段階の最小回帰：** 対応pairのaffine/多項式norm、非均一cell面積、supportのgap/overlap、boundary/RZ measure、budget不足時の無publish。tensor-hatと未対応pairの無publishで元の原因を閉じ、原field・geometry・sourceを保持する。regular bilinear、warped Q1、時間相殺はその能力を採用する変更で追加し、初回の全組合せgateにしない。

**完了条件：** 対応pairのshared derivative verification、公開preprocess workflow、標準品質gate、代表source/targetのmemoryと処理時間。これはcache品質gateの修正完了である。製品でcacheを採用する段階では、使用する物理のsource/cacheを同じ時間刻みで比較し、必要な到達・電荷・fateへの影響を別に評価する。場normの合格だけで全物理のtrajectory精度を保証したとはしない。

文書ownerは[field_preprocessor README](../solver/tools/field_preprocessor/README.md)、[T04計画](../implementation_plan.md:2323)、[cache製品要件](../product_specification.md:996)、[cache責務](../architecture_proposal.md:758)である。

## 5. G03：COMSOLの実使用物理と境界を比較前に確認する

比較の共通設計は、producerが実際に使ったprimitive・model・座標・source・boundary・数値条件をrequestと照合することである。以下のCase-A/Pは確認済みの適用例であり、coreや共通checkerへcase名分岐を導入する理由にはしない。receiptのidentityは常に確認し、force/FDT/event probeは有効modelと比較質問に必要な範囲へ限定する。

### 修正箇所と物理の維持

[RunM3C2StochasticCampaign.configurePhysics](../solver/tools/vv/comsol/comsol/RunM3C2StochasticCampaign.java:301)はdf1の値欄をP1式へ変えるが、u、temperature、pressureのsource selectorを設定・検査していない。保存Case-Pの設定にはnative fieldを選ぶsourceが残る。COMSOLのDrag Forceは、式欄と他physicsからのsource選択を個別に持つ。[COMSOL公式仕様](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.06.html)

推奨は、修正済み[M3-C3のexplicit Epstein drag](../solver/tools/vv/comsol/comsol/RunM3C3CasePThreeCurrent.java:280)と同じ物理表現をC2へ適用することである。native df1を無効化し、canonicalの速度・密度・温度とresolved parameterから明示drag forceを生成する。df1 inactive、custom force active、domain selection、StudyStep、affected pp1をread-backする。

分子質量、粒径、Epstein係数δをJavaの別magic constantへ増やさない。[generated requestのowner](../solver/tools/vv/comsol/run_m3c2_comsol_campaign.ps1:475)へresolved parameterを渡し、request hashへ含める。Maxwell混合表面則でσ_Rを用いる場合は、δ=1+πσ_R/8の対応をここで一度解決し、noise側で再解釈しない。

補間済みの同じprimitiveから、次を一つの明示式として構成する。

\[
\beta(x,t)=\frac{\pi d^2}{3}\rho_{P1}(x,t)
\sqrt{\frac{8k_BT_{P1}(x,t)}{\pi m_g}}\delta,
\quad F_{drag}=\beta(u_{P1}-v),\quad\mu_B=\beta/(3\pi d).
\]

Brownianの温度もこの`T_P1`へbindする。random-force white-noise強度は`2β k_B T`、accelerationでは`2β k_B T/m_p²`、candidateの`γ=β/m_p, Θ=k_B T/m_p`では`2γΘ`となる。有限stepのjoint OU covarianceとは区別し、force/acceleration/velocity incrementの係数を混同しない。

現行[C2 binding](../solver/tools/vv/comsol/comsol/RunM3C2StochasticCampaign.java:214)は`μ_P1/λ_P1`からμ_Bを作る。節点でconstitutive relationが成立しても、μ、λ、ρ、Tの別々のP1補間はその非線形関係をcell内部で保存しない。[独立係数監査](theory_comsol_coefficient_checks_2026-10-09.json)では、Case-A/Pの3779重心sampleでβ比の最大差約0.461%を確認した。**両caseで、補間の後に同じβを形成する変更が必要**である。物理精度の根拠なくμ・λ・ρを一つの新fieldへ上書きしてはいけない。

二電流Case-P anchorを三電流へ変更する提案ではない。物理revision、分布、初期charge、seed規則、統計toleranceを比較結果へ合わせて変更しない。

### 軸境界と終端eventの対応

[source Case-P設定](../../model_dataset/cf4_o2_etch_caseA_nonlinear_sass/cases/formal_iondrag_theory_consistent/caseP_100nm/external_reproduction/config/particle_physics_feature_settings.csv:6)とCase-Aは`axi1=Freeze`を保存する。[C2 runner](../solver/tools/vv/comsol/comsol/RunM3C2StochasticCampaign.java:357)とC3はmaterial/inlet/pumpを設定するが、軸ではactiveの確認しか行わない。[比較contract](../solver/tools/vv/comsol/cases/m3c2_caseP_100nm_stochastic_pilot_v1.json:62)は座標軸通過を宣言している。この不一致を、座標系名だけでSUPPORTEDにしない。

sourceのmodelを直接改変せず、独立companionへ次を明示する。

1. 2DOF no-swirl、out-of-plane off、candidate側axisはcoordinate seamという比較目的を固定する。
2. 原sourceの軸lawとselectionをread-backする。
3. companionでは、既存[critical microcase](../solver/tools/vv/comsol/comsol/RunM3CCriticalBoundaries.java:155)と同じ軸Bounce設定を明示し、no-swirlでの座標通過への対応を実microcaseで確認する。candidate coreのaxisをmaterial reflectionへ変更しない。
4. source→companionの変更前後とhash、理由をreceiptへ記録する。この比較を、元のFreeze模型の完全再現と呼ばない。
5. source Freezeそのものの再現が比較目的なら、現行candidateでは未対応としてその質問をNOT_APPLICABLEにする。意味が未決ならAMBIGUOUS、具体的な同値外部変換が残る場合だけADAPTER_REQUIREDとする。

[C2 normalizer](../solver/tools/vv/comsol/normalize_m3c2_comsol_pilot.py:1318)と[C3 normalizer](../solver/tools/vv/comsol/normalize_m3c3_caseP_three_current.py:337)の`held→gas_inlet_hold`という固定辞書を両方から除去する。COMSOL statusはlifecycleを表すだけで、接触したboundary identityを一意には与えない。receiptのactual boundary response/selectionと実event boundary IDを一次対応表にする。

二つのconsumerがあるため、新規の狭い`tools/vv/comsol/boundary_response_mapping.py`を一つのrule ownerとする。操作は`resolve_terminal_boundary(outcome, observed_boundary_ids, actual_response_by_id, other_terminal_causes_excluded)`のみで、I/O、COMSOL API、case ID、独自geometry solverを持たせない。既存[normalize_m3c_boundary_semantics](../solver/tools/vv/comsol/normalize_m3c_boundary_semantics.py:70)はforce-free一粒子の既知scenarioとDisappear解析再構成に結合しており、一般event同定の共通moduleへ転用しない。

- 実IDがあればactual selectionへ照合する。
- IDがなく、同じresponseを持つ全active selectionが一つのsemanticで、他の終端原因も除外済みなら、そのgroupだけを同定する。facetは未観測として記録する。
- axis Freezeとinlet Freeze、異なるsemanticのcorner、unknown ID、他原因未解決は同定不能とする。位置からIDを捏造せず、statusだけからinletへfallbackしない。

stop位置を追加観測に使う場合は、既存geometry authorityと位置budgetによる一意な同定を別証拠として記録する。単一group同定をexact facet/event認証へ昇格しない。hit identity、position、pre/post stateが未観測なら該当評価をNOT_TESTEDとする。

Freeze→holdはterminal lifetimeとして許容するが、その境界がinletであるという装置名はcoreへ持ち込まない。人口の`held`件数と、inlet/axis/material別のboundary parityを分ける。既存Case-P C2のterminal event 0は後者の証拠にならない。

### 積分設定と保存時刻を別々にread-backする

`NewtonianFirstOrder`、out-of-plane、manual classical RKのmethod/order/`rktimestep`、`tout/tlist`、step update方針、`WallAccuracyOrder`、`StoreExtra`、status保存、wall interaction上限を実read-backへ含める。[COMSOL Time API](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_api_solver.51.51.html)。上限の実property名・発火原因の取得方法はlive modelと小caseで確認してから採用する。境界geometry・上限由来のDisappear・event観測の具体化はG12を一つの設計authorityとする。

現在のC2/C3のOrder 1はsource継承値へのassertである。Order 1はwall/release区間でforward Eulerを使うため、bulk RK4四次と別のscopeになる。[公式wall accuracy](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.02.html)。Brownianの比較でこの設定を無条件にOrder 2へ変えず、期待値と実値をreceiptへ記録する。決定論event収束ではこの設定を含めてreference refinementを行う。

内部step履歴を取得しない場合は`internal_step_history=NOT_TESTED`とする。`tlist`の121点を内部step履歴へ流用しない。manual RKの`rtol`は設定値として保存し、trajectory accuracyの認定値にしない。

read-backには公式[PropFeature API](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/api/com/comsol/model/PropFeature.html)のproperty存在・型・許容値と型別getterを使用する。step式は[parameter評価](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_api_general.47.50.html)で秒単位の数値へ解決し、式と実数値を保存する。未対応propertyでexpected値を観測値として代理出力しない。

### 小さいexternal binding receipt

既存RunnerValidationと本campaignが同じconfigurePhysics経路を通る構造を維持する。C2のRequest.javaはPowerShellがcontractから生成し、RunnerValidationは既存runnerへの薄いentryである。変更後は`loadCopy`直後にsource snapshotを取得し、`configurePhysics → auto sequence生成 → manual solver設定適用 → companion snapshot → solve`の順で対象bindingを確認する。override前のdefault設定を実使用設定として保存しない。次のreceiptを使い、FDTはBrownian有効時、component RHSは有効force/charge、境界観測は要求するevent scopeを対象とする。非対象項目の未実施を別用途の比較失敗へ拡張しない。

| receipt項目 | 必須内容 |
|---|---|
| input/execution identity | source MPH前後hash、COMSOL version、component/physics/study/solution/dataset/domain/side、canonical hash、P1 table hash。source/設定snapshot/実companionのidentityを区別 |
| feature binding | tag/type、active、selection、StudyStep、affected particle、quantity/unit、宣言式、read-back式、source selector、field owner |
| source/companion boundary | 原sourceとcompanion双方のaxis/material/inlet/outlet lawとselection、point/finite contact、内部面/pairの意味、変換理由、boundary identityとterminal causeの取得方法 |
| resolved model | model revision、mass/diameter/gas mass/accommodation、β、Brownian temperature/effective viscosity、2D/RZ/out-of-plane |
| fixed probes | 事前固定した同一状態キー、primitive実評価値、charge rate、各force、摩擦・noise係数、support/finite判定 |
| integration/output | formulation、RK method/order/fixed-step設定、tout/tlist/update、WallAccuracyOrder、StoreExtra、status保存、wall上限、内部履歴/event観測の取得status |
| per-check status | selector/read-back/primitive/component RHS/FDTそれぞれの結果、tolerance、失敗箇所 |
| campaign identity | request hash、各replicaのreceipt hash、study自動生成後の再確認 |

現C2/C3 campaignのrelease probeとreceipt生成には[common-P1 preparer](../solver/tools/vv/comsol/prepare_m3c1_common_p1_tables.py:351)を利用する。一般的なbinding照合を、このpreparerの固定particle集合や全field bundleへ依存させない。各adapterは比較に必要なcanonical primitiveとstate keyを供給する。probeはreleaseだけでなく、field hashから再現可能に選ぶcell内点、source inputだけで事前算出したdeclared closure差が最大のcell、support/axisの片側点、複数の粒子速度・chargeから必要なものを選ぶ。nodeだけの一致で合格させない。time fieldを比較する場合はknotと区間中間も含める。

parameterとprobe planを結果を見る前に固定する。closureの数値誤差とfield location/interpolation誤差に応じて、quantityの単位を持つabsolute budgetとrelative budgetを定める。ゼロforce・ゼロdriftはabsolute条件で判定する。許容trajectory差を局所RHS toleranceへそのまま転用しない。Brownianの乱数力はpathwise一致を求めず、摩擦・温度・noise係数とensembleを別に検証する。

重要な区別として、[M3-C3 total診断](../solver/tools/vv/comsol/comsol/RunM3C3CasePThreeCurrent.java:517)は取得式を足した再構成値である。COMSOLの組み立て済みODE RHSを直接読んだ証拠ではない。receiptにはtotal_kind=reconstructed_component_sumと記録する。実assemblyを取得できる確認済みAPI/変数がない場合、その項目を認証済みにしない。featureのactive/selection/StudyStep、component評価、決定論trajectory収束は別々の証拠として示す。

### preflightとevaluatorへ接続する

[meaning_preflight](../solver/tools/vv/comsol/meaning_preflight.py:159)は現在、`evidence`の非空文字列と`mapping=direct`の宣言からSUPPORTEDを作る。宣言を実証拠の認証と同一視できない。既存ownerへ、必要な範囲だけhash付きartifact照合を追加する。

~~~json
{
  "observed": {
    "path": "actual_binding_receipt.json",
    "sha256": "<receipt digest>",
    "pointer": "/companion/coordinate/axis_meaning"
  },
  "expected": {
    "path": "comparison_contract.json",
    "sha256": "<contract digest>",
    "pointer": "/coordinate/axis_meaning"
  }
}
~~~

この形式は提案であり、現行inputの対応済みschemaではない。adapterがsource・companion・実read-backを一般的な意味へ正規化し、raw観測への参照を残す。preflightは二artifactの存在/hash、JSONの有限値・pointer・型・実値と登録expected値の一致だけを確認する。COMSOL tag/selector解釈、式parser、solver呼出しは追加しない。自己申告のPASS文字列へのpointerだけでは通過させない。

数値probeのreference式とtoleranceは既存の科学評価ownerが持ち、generic checkerへ式言語・callbackを追加しない。meaning_preflightをbinding照合の唯一ownerとし、各normalizerの固定期待辞書を第二checkerとして残さない。actual receiptの成立は実runner read-back/probeとexecution hashで示す。hashが正しいことだけではread-backの物理内容を証明しない。

必須証拠欠損・hash不一致・pending変換は比較前に停止する。未認証mappingはADAPTER_REQUIRED、意味やboundary identityが不明ならAMBIGUOUS、必要な検査が通ったscopeだけSUPPORTEDとする。adapter変換を実施しただけで原sourceと物理が同じとは記録しない。sourceとcompanionの意味を分ける。inventory/receiptのformatとtool revisionを更新し、current executionでは旧文字列evidenceを認証として受理しない。

[C2 normalizer._receipts](../solver/tools/vv/comsol/normalize_m3c2_comsol_pilot.py:879)と[C3 configuration検査](../solver/tools/vv/comsol/normalize_m3c3_caseP_three_current.py:113)はraw形式・request identityを保持し、literal期待辞書の意味authorityをcomparison contractへ移す。[C2 evaluator](../solver/tools/vv/comsol/evaluate_m3c2_stochastic_ensemble.py:3606)と[C3 evaluator](../solver/tools/vv/comsol/evaluate_m3c3_casep_three_current.py:659)は、全対象receiptのhash・preflight scopeをmetric計算前に要求する。source-token検査はhash provenanceの補助へ限定する。[anchor preflight](../solver/tools/vv/comsol/m3c2_anchor_preflight.py:327)は歴史anchorとseed reservationのownerとして維持し、live binding checkerへ転用しない。

**最小回帰：** active native featureの値欄がcanonicalでもselectorがnativeなら停止、P1のnode closureが真でもcell内で差が出るcase、axis Freeze＋seam宣言の拒否、両normalizerでheldのaxis/inlet不明・異semantic corner・unknown ID、probeのu/T改変、receipt欠損/hash/pointer/型違い・一replicaだけ未認証で評価前停止。inactive df1のselectorを比較authorityとして検査しない。correct bindingではprimitive・drag・β/FDT、axis比較scope、integration/output設定を確認する。実設定と登録条件の値照合を抜いた自己申告PASSは拒否する。

**C2 stochastic campaignの実COMSOL完了条件：** 両caseのRunnerValidation、auto sequence後read-back、固定probe、axis crossing microcase、事前登録pilot、未使用seedのfinal ensemble。新しい数値影響とensemble合否は、実行するまで未確定である。過去のfinal seedを調整に利用した場合、それを新しい未使用final cohortとは呼ばない。

再実行対象はdrag selector・FDT・axis mappingが変わるCase-P M3-C2と、FDT・axis mappingが変わるCase-A M3-C2である。これらのstochastic campaignを認定する時に登録pilot→未使用seed finalへ進む。generic binding checkerの実装やC3決定論比較の完了条件へensemble finalを要求しない。C3はBrownian offだがaxis source継承の影響を持つ。軸設定を変えたC3 companionはread-back、axis microcaseと対応trajectory/event比較を更新する。変更に無関係な独立microcaseを全面再実行する必要はない。TalbotのCOMSOL runtime parityは別の未検証model mappingであり、今回のC2/C3合格を転用しない。

過去evidenceを上書きせず、「登録population gateの計算上PASS」と「same canonical RHS bindingの未認証」を併記するaddendumを作る。runner/normalizer/receipt/evaluator/比較contractのrevisionを更新し、coreのmodel/engine/schemaは変更しない。

## 6. G04：完成結果の粒子数とIDを同じownerで検証する

[ResultWriter](../solver/src/chamber_particles/output.py:608)はscheduleの粒子総数をparticle_capacityとして受け取る。[finalize](../solver/src/chamber_particles/output.py:727)は現在、final.particle_id.sizeから新しい粒子数を作る。[completed reader](../solver/src/chamber_particles/output.py:1464)と[_validate_final_file](../solver/src/chamber_particles/output.py:2279)は全final列相互のshapeを確認するが、manifestの粒子数と照合せず、IDの単調性も確認しない。

変更はoutputへ限定する。

1. finalizeのNをself._particle_capacityと照合する。finalの短縮がmanifestの新しい正当なNにならないようにする。
2. _open_completed_pathで既存_summary_from_manifestを使い、particles/macro_stepsの非negative integerを早期に検査する。新しいJSON validatorは作らない。
3. _validate_final_fileへexpected_particle_countを渡し、dtype/rankと全列shapeをそのNで検査する。
4. particle_idを固定row blockで走査し、非negative・厳密昇順を確認する。block末尾と次block先頭も比較する。np.diffではなく直接大小比較し、整数overflowを避ける。IDが連番である必要はない。
5. 既存contact radius、validity、lifecycle/reasonの走査も同じbounded blockで処理する。
6. final.tmp.h5を書いた後、publish前に同じfile validatorで検査する。ID検査をarray用・file用の別規則に増やさない。

[read_final](../solver/src/chamber_particles/output.py:877)はファイルを後から開くので、そのhandleも同じvalidatorへ通す。open時点の検査結果を後から開く別handleの保証に使わない。結果ディレクトリの利用者による同時変更はsupported運用に追加しないが、open後の短縮/重複を静かに返す経路は閉じる。

この変更の保証は容量、列構造、非negative・一意・昇順IDの整合である。sourceへのexact ID membershipや全event/frame参照整合は別の検査範囲であり、manifestと全データを整合するように書き換えた任意改変の検出を保証しない。final checksumやcheckpoint読込みを完成readerの必須条件へ追加する必要はない。

**最小回帰：** 正常なdurable scenario結果を複製し、全列同時短縮、重複、逆順、block境界の重複をparameterizeしてopen_resultの拒否を確認する。疎な昇順IDは受理する。writerの容量違反では_SUCCESSをpublishしない。既存resumeとrecovery scenarioを確認する。

formatの表現は変わらないためschemaを増やさない。読み書きvalidationだけのpatchならengine/RNG revisionは不要である。resume/result algorithm revisionは、実装がcommit/identityの意味まで変わった場合のみそのownerで更新する。

## 7. G05：入力文法と数値例外を狭い責務で統一する

### 数値変換

[case._number](../solver/src/chamber_particles/case.py:702)のfloat(value)は、巨大なYAML integerでOverflowErrorを出し得る。[公開API](../solver/src/chamber_particles/api.py:36)はこれをCaseErrorへ翻訳しない。

修正は変換を所有する箇所で行う。caseではOverflowErrorをlocation付きValueErrorに、[physics.catalogのnumeric parser](../solver/src/chamber_particles/physics/catalog.py:1916)ではPhysicsConfigurationErrorに翻訳する。Clausius–Mossotti factor、unit interval、pairなどの直接float変換も同じ原則で確認する。外部adapter/builder/preprocessorのYAML numeric parserも各producerの既存例外へ翻訳する。

bool拒否、finite検査、正値・範囲検査は既存のdomain ownerに残す。API全体でExceptionをcatchしてprogramming errorを隠さない。数値変換のための広いvalidation frameworkを新設しない。

### 共通YAML leaf

caseとadapterは別々のduplicate-key loaderを持ち、builder/preprocessorはsafe_loadを使用する。四つの主要入口の実需要があるため、solver内の新しい小module yaml_input.pyで、UTF-8 decodeと安全な文書構築・duplicate key拒否だけを一意に所有させる案を採る。

~~~text
yaml_input.parse_document(raw_bytes) → object / ValueError
  fileを開かない
  hashを決めない
  case/physics/producer schemaを知らない
  COMSOL/toolsをimportしない
~~~

case._read_yamlと各parse_configurationがこのleafを使う。raw bytesのSHA256は各現在のownerで維持し、再dumpしたYAMLを入力hashにしない。四主要入口の旧loader/safe_loadの設定読込みを同じ変更で置換する。G06を変更する際は、[candidate template読込み](../solver/tools/vv/comsol/run_m3c2_candidate_pilot.py:1322)にもこのleafを使う。safe_load→safe_dumpで重複keyを消してからcoreへ渡す経路や、template専用の第二strict loaderを残さない。rootの三APIは増やさない。

方針は、全階層の重複keyを拒否し、current caseが受理していないYAML merge keyを明示拒否する。これはcaseの従来動作を維持する一方、[adapterのflatten_mapping](../solver/tools/comsol_adapter/workflow.py:55)、builder/preprocessorのsafe_loadが受理していたmerge構文を制限する入力言語の変更である。duplicate拒否の不具合修正と区別し、producerのtool/config revisionと移行説明を同じ変更へ含める。通常aliasの扱いを無条件に拡張せず、現在の有効入力を確認する。各producerの未知key・必須key・unit・model検査はそのproducerに残す。mergeを黙って展開する互換parserは残さない。

**最小回帰：** 公開load_caseとCLIで巨大time値をCaseError/所定exitへ変換すること、simulateで巨大physics parameterを所定例外へ変換すること、四入口でnested duplicateを拒否すること、通常のcanonical case/producer configを受理すること。parserのクラス名や内部配置をtestしない。

canonicalの構造を変えなければcase/data HDF5 schema変更は不要である。producerの受理文法が変わる部分は、そのtool/config identityで記録する。yaml_inputのownerと低水準moduleからの禁止を、既存owner表・既存import contractへ反映する。別のcontract suiteは追加しない。

## 8. G06：歴史監査と現行再実行の入口を分ける

外部toolsの35 failed / 14 errorsを、pytest skipや旧schema reader追加で消さない。旧evidenceのhash監査と現行solverでの再計算は、要求するidentityが違う。

### 歴史監査

過去のnormalized CSV/JSON、receipt、registrationをhash lockして、既存external evaluatorで監査する。元のengine/model/schemaを歴史identityとして表示する。旧resultを現行open_resultへ通すreaderや、solver内のschema migration engineは作らない。

過去版の再実行を別途必要とする場合は、元のimmutable release/sourceを独立した環境で指定する。current solverが過去版として動いたと記録しない。old_codeの環境を再活性化する提案ではない。

### 現行再実行

[現行prepare](../solver/tools/vv/comsol/run_m3c2_candidate_pilot.py:1324)はtemplateを検査した後にoutputを作り、生成caseをload_caseへ渡す。[現行template検査](../solver/tools/vv/comsol/run_m3c2_candidate_pilot.py:1193)はphysics/boundary行列が中心で、現行formatとexecutor provenanceの完全な事前照合は未実装である。変更後はinput、template format、recipe、期待executorをoutput作成前に検査する。_case_documentで古いformat_versionを数字だけ更新して互換扱いにしない。

[_load_prepared](../solver/tools/vv/comsol/run_m3c2_candidate_pilot.py:1759)のhistorical revision受理を、新しいsimulateの実行許可へ使わない。prepare/runではcurrent prepare revisionと期待executorを照合する。recover/renormalizeは保存resultの読込み・再投影なので、元のmanifest actual identityとlocked記録の一致、current reader/projectionの対応、元receipt/hashの保持を確認する。再集計する環境のsolver sourceが元executorと完全同一であることまでは要求しない。新しいtoolは実際のrunner/normalizer revisionを記録し、歴史producerを名乗らない。

期待executorは新しい追加設計であり、まず既存recipeのexecution記録へ、immutable solver sourceまたはwheelの一つのidentity、uv.lock hash、Python/distribution version、必要なresolved revisionsを加える。両配布方式の一般frameworkは作らず、採用した実行形態を一つ閉じる。editable installでpackage versionやdirty変更を含まないHEADだけをidentityにしない。実行は三APIを通し、open_result後のmanifestからactual engine/model/RNG/boundary/result等を照合する。private engineを直接importする別経路は作らない。

現行inputは保存済みraw mesh/field/source exportから、既存[adapter](../solver/tools/comsol_adapter/workflow.py:177)や[theory preparer](../solver/tools/vv/comsol/prepare_m3c1_theory_100nm_matrix.py:411)を通して再生成する。旧HDF5のschema markerだけを書き換えない。

現行test fixtureは、小さいinputをcurrent canonical writerで作り、current template/recipeを使うものへ置換する。歴史hashを監査するtestは実行fixtureと分けて残す。例として[current_recipe_copy](../solver/tools/vv/comsol/tests/test_m3c2_candidate_pilot.py:79)のnoise revisionだけを変更する方式を廃止する。

**最小回帰：** 旧schema/template/executorがoutput作成前に停止すること、current inputのprepare→load→simulate→openとactual revision一致、historical normalized evidenceのhash監査。real COMSOLが不要なcurrent tool testsは独立CI stepで実行する。licenseがない状態のreal COMSOL coverageをPASSへ置換しない。

外部runner/recipe/evaluatorのrevisionを更新する。coreの旧schema拒否とBrownianの現行revisionは維持する。

## 9. G07：endpoint・dense・実noise・first-arrivalの精度を分けて閉じる

### 既存testを正しく位置付ける

[stochastic weak-mean test](../solver/tests/verification/test_stochastic.py:113)はJointOuIncrementを0にする。線形問題のmean計算の検証としては有効だが、noise振幅やnoiseで動いた位置のfield評価を独立に検証する証拠にはならない。決定論極限を表すtest名へ改めて維持する。[researchの説明](../technical_research.md:634)、[実装計画](../implementation_plan.md:1815)、[V&V methodology](../vv_methodology.md:683)も同時に修正し、以下を追加する。

最初の縦切りは、説明の修正と、[既存の独立OU/conditional Gaussian検証](../solver/tests/verification/test_stochastic.py:333)を[公開OU scenarioの期待値](../solver/tests/scenarios/test_brownian_run.py:63)へ適用することである。定係数Qやhalf-split oracleを別にもう一組作らない。その後、使用する可変係数のA/B、非dyadic内部時刻、対象のRZ/chargeへ必要な範囲だけ追加する。endpoint・dense・Brownian・到達を一つの巨大な検証完了条件にしない。

### Endpoint・dense・field界面を別々に測る

[既存RK4 endpoint](../solver/tests/verification/test_integrators.py:1542)と[charge連成](../solver/tests/verification/test_integrators.py:1622)の解析収束caseは維持する。h系列を固定終時刻で評価し、位置・速度・電荷それぞれを事前固定した物理scaleで無次元化する。smooth caseでは既存の四次/二次期待に加えて最細絶対誤差を判定する。次数だけの合格にしない。

[dense charge test](../solver/tests/verification/test_integrators.py:3127)は、exact startから内部一stepの誤差がO(h⁴)であることを明示する名前へ変更する。RK4 endpointの局所O(h⁵)と同じ保証ではない。調和振動子の解析解でθ=0.37,0.61等の位置・velocityを測り、一step内部誤差と固定終時刻のdense出力誤差を分ける。`dense velocity = d(dense position)/dt`をRK4の受入条件にしない。現在は別cubicであり、差を含むenclosureが必要である。

[integrator stability gate](../solver/src/chamber_particles/engine.py:3990)のdrag・charge成分別制限を連成系全体のaccuracy certificateと呼ばない。例えばx''=-ω²xのhω=3ではdrag/charge制限だけでRK4の安定性を保証できず、|R(3i)|>1となる。独立解析解とh/h2/h4で用途KPIの誤差を判定する。今回、新しいgeneral Jacobian validatorや第二adaptive engineを追加しない。

fieldについては、[既存time-knot分割](../solver/src/chamber_particles/engine.py:1305)を維持する。全cell同一のaffine lawと、連続だが界面で導関数が変わる二つのaffine lawを[既存force-coupled scenario](../solver/tests/scenarios/test_force_coupled_run.py:968)へ小caseとして追加する。後者は区間ごとの解析運動と界面時刻からreferenceを作り、release/interfaceのphaseを事前集合でずらす。fieldは全h系列で同一にし、mesh系列を別にする。kink/grazingへ一律四次を要求せず、最細誤差・phase集合の包絡・到達時刻を判定する。

### Oracle A：時間変動する摩擦・noise振幅

\[
dx=v\,dt,\quad
dv=-\gamma(t)(v-u(t))dt+\sqrt{2\Theta\gamma(t)}\,dW,\quad
\gamma(t)=\gamma_0(1+a t),\quad\Theta=k_BT/m.
\]

Tを一定、u(t)をaffineとする。canonicalの時間linear gas densityからγ(t)を構成し、productionの実stage時刻評価とFDTを検証する。

Γ(t)=∫γ(s)ds、R(t,s)=exp[-Γ(t)+Γ(s)]を使い、meanとcovarianceを独立Green kernel積分で求める。以下のcovariance式は全粒子が同じ決定論初期状態、P0=0の場合である。ランダム初期状態を使う場合は初期共分散の伝播項を加える。

\[
G(t,s)=\int_s^t R(q,s)dq,\quad
\operatorname{Var}(v_t)=\int_0^t2\Theta\gamma(s)R(t,s)^2ds,
\]
\[
\operatorname{Cov}(x_t,v_t)=\int_0^t2\Theta\gamma(s)G(t,s)R(t,s)ds,\quad
\operatorname{Var}(x_t)=\int_0^t2\Theta\gamma(s)G(t,s)^2ds.
\]

NumPy Gauss–Legendreの次数を倍化してoracle誤差を確認する。production integrator、production OU update、runtime evaluatorをreference計算へ呼ばない。既存locked dependencyだけで作れる。

### Oracle B：noiseで移動した位置へのfield再評価

γ、Tを一定、u_x(x)=-Kxとする。regular/P1上で正確なlinear fieldを使い、u_y=0を独立OU対照とする。

\[
dY=AYdt+B\,dW,\quad
Y=(x,v)^T,\quad
A=\begin{pmatrix}0&1\\-\gamma K&-\gamma\end{pmatrix},\quad
B=(0,\sqrt{2\gamma\Theta})^T.
\]

2×2のclosed-form exp(At)と∫exp(As)BB^T exp(A^Ts)dsでGaussian mean/covarianceを計算する。現行methodは前rootのnoiseを反映した次root始状態から決定論midpointを予測し、そのfieldを再評価する。この位置依存更新とh収束を検証するのであり、現在のroot内でnoiseを先にdrawして確率midpointを評価する別方式へ変える提案ではない。これは粒子方程式を検証するmanufactured flowであり、流体の連続式・運動量式を満たすbackground solutionのvalidationではない。独自model callbackをproduction catalogへ追加する必要はない。

### 判定とcoverage

両caseはcurrent supported Cartesian XY、境界・適用域に余裕を持つ設定で実行する。物理parameterからγを独立に導出し、runtimeの値をoracleの入力にして自己一致を作らない。unexpected failureが一件でもあれば統計gateをinvalidとし、その粒子を除いて合格させない。

h、h/2、h/4とnoise path depth Dを別軸で調べ、h×depth×seedの全組合せを必須にしない。no-event endpointのh次数は独立moment伝播で測り、公開実drawは登録した少数hで確認する。[一rootで凍結される係数](../solver/src/chamber_particles/engine.py:1799)を使うno-event終端のdepth系列はendpoint再構成identityを主に確認し、depth accuracy系列は非dyadic内部時刻とfirst-arrivalへ適用する。sample数、seed群、評価時刻、同時信頼区間、acceptance toleranceを先に固定する。mean、variance、x-v covariance、1D CDFを評価し、Monte Carlo誤差以下の差から観測次数を主張しない。ensemble追加を許す場合も最大sample数、追加判定時点、各判定へのα配分を事前登録する。登録上限でも解像できなければその項目は未確定とする。

非零covarianceの決定的伝播により離散化次数を測るverificationと、公開APIの実drawによるmean/full covariance検証を分ける。[既存公開OU test](../solver/tests/scenarios/test_brownian_run.py:36)は期待covarianceをproduction helperから得る部分を独立Green積分/解析式へ置換する。固定Nの実ensembleだけでweak二次を認定しない。統計的bias差が事前計画したconfidence幅を十分上回る場合だけ次数を報告し、そうでなければaccuracy判定と`order=NOT_RESOLVED`を別に示す。

Gaussian meanのconfidence band、sample varianceのχ² concentration、正規化したX+V/X−Vのvarianceから作るcovariance区間を使える。例えばn=N−1、観測量数M、family error αに対して、t=log(2M/α)、β=2sqrt(t/n)+2t/nとする。不偏sample variance s²からtrue varianceの保守的区間`[s²/(1+β), s²/(1−β)]`を作り、β<1を条件とする。s²を中心に単純な±β幅を置かない。合否は推定biasのconfidence上限が登録budget以下かで判定する。gamma・温度・modelの適用域だけでなく、この統計budgetも実行前に固定する。

h変更でmacro ordinalも変わるため、同じseedの軌道を同じWiener pathとしてstrong errorへ利用しない。同じhのdepth比較では現在のconditional treeの共通階層を利用できる。

no-event終端のdepth不変性と、非dyadic中間frameのdepth収束を区別する。一定係数OUのleaf端点(x0,v0,x1,v1)のjoint Gaussian covariance Cを独立Green積分から求め、Hermite shapeと導関数の線形写像Lによる`L C Lᵀ`を有限pathのoracleにする。まずその有限表現と実装の一致、次に真のOU内部時刻分布との差を測る。

fixed macrogridでglobal queryを0.37h,0.61h等に固定し、Dbase=Dmaxのdepth系列で比較する。queryのleaf内phaseはDで変わるため、厳密単調減少を要求せず、登録query集合の最細誤差と包絡を確認する。frame/probe頻度、slab size、resumeを変えても同じphysics/RNG設定の最終scientific payloadが変わらないことは、既存公開scenarioを再利用して確認する。途中frameの追加によりproduction pathやnoise drawを変えない。

これは時間・位置依存の実noise検証を増やす。任意nonlinear SDEのstrong/weak 2次、連続OUのfirst-passage、3D Brownianを認定するものではない。

### First-arrivalは独立Kramers referenceへ接続する

[現行depth test](../solver/tests/scenarios/test_brownian_run.py:814)は有限pathの登録depth間での安定性を表す名前へ改め、小smokeとして維持する。Bernstein certificateは表現されたHermite曲線の幾何保証であり、continuous OUのzero-miss証明ではない。

到達確率・到達時刻が製品KPIの場合、最初の外部referenceを一定γの1D integrated OU、平面terminal boundaryに限定する。位置domain x<bで、

\[
\partial_t p=-v\partial_xp+\gamma\partial_v(vp)+\gamma\Theta\partial_v^2p,
\qquad p(b,v<0,t)=0,
\]
\[
f_\tau(t)=\int_{v>0}v\,p(b,v,t)\,dv
\]

をsolver外の小さいreference calculationで解く。吸収boundaryは入口velocityのzero-inflowであり、全velocityへp=0を課さない。半範囲boundaryの根拠は[Hwang–Jang–Velázquez式1.3–1.4](https://arxiv.org/pdf/1311.4635)、friction項と到達fluxはここで選んだOUからの導出である。

reference側にもΔx/Δv/Δt系列、velocity domain・遠方position domain拡張、初期delta近似の系列を持たせる。positivity、残存mass＋吸収mass＋人工外側流出massの整合を確認する。candidateのproduction update/conditional tree/boundary routineをreferenceへ呼ばない。新しいproduction engineとして実装しない。

candidateはh固定のDbase=Dmax系列と、十分深いDを固定したh系列を分ける。Dmaxの増加だけを一様精度の改善と呼ばない。登録到達CDF/fateに同時binomial/CDF confidence bandを適用し、bias区間、reference不確かさ、MC幅を各budgetに照合する。coarse/fine差だけを誤差の保証上界と呼ばず、referenceが経験的収束ならその結論へ限定する。reference不確かさをbudget内に決められなければfirst-passage accuracyはNOT_RESOLVEDとする。

### RZと連成chargeを混ぜずに追加する

RZは現行meridional 2DOFへscopeを固定する。[既存RZ OU test](../solver/tests/scenarios/test_brownian_run.py:204)を独立oracleへ置換し、軸横断はradial flow/forceが零の一定係数系でsigned OUの`r=|X|`からfolded Gaussian CDFをreferenceにする。axis lifecycle、failure、depth、slab/output/resumeを確認する。一般3D等方Brownianの二つの横断noiseによる半径分布を期待値へ使用しない。

noise-onのcontinuous chargeには、[現行stationary OML row](../solver/src/chamber_particles/physics/compiled.py:1240)の`oml_stationary_maxwellian_debye_huckel_v1`を選べる。このrevisionはcharge rate自体がv非依存で、relative speedはapplicability gateに使う。一様plasma、一定γ/E、同一初期Z、維持できる負電荷branchに限定し、独立scalar ODEまたは単調branchの`∫dZ/F(Z)=t`反転からZ(t)を求める。production charge updaterをreferenceへ呼ばない。

κ=eE/mとして、mean velocityへの追加は`κ∫exp[−γ(t−s)]Z(s)ds`、mean positionへの追加は`κ∫(1−exp[−γ(t−s)])/γ Z(s)ds`である。covarianceは一定係数OUで分離できる。drift gateとfield supportには十分な余裕を設け、unexpected failureを除外して合格させない。

空間一様plasmaだけでは、すべてのrevisionでZ(t)が決定論になるとは限らない。shifted OMLとaggregate等のrelative drift依存を残すrevisionへこのscalar referenceを転用しない。それらには非Gaussianなcoupled referenceと適切な統計条件が必要である。stationary charge追加もA/Bの最小Gaussian検証に続く別caseとし、一般位置依存charge/dragの次数保証へ昇格しない。

### Denseとproducerの説明を揃える

[AGENTSのstate_at説明](../AGENTS.md:142)と[計画の包括表現](../implementation_plan.md:763)は曲線を一律再積分とするが、現在のRK4 dense pathはposition/velocity/chargeのpolynomial extensionを使う。[現行numericsのdense説明](../solver/docs/numerics.md:1370)へ合わせ、methodごとに次を一度だけ定義する。

| 処理 | methodごとの意味 |
|---|---|
| exact proposalのframe | exact path評価 |
| RK4 denseのframe | accepted rootのimmutable polynomial評価 |
| exponential midpointのframe | proposal始点から対象時刻まで同じmethodで再評価 |
| Brownianのframe | accepted conditional-tree leafのHermite numerical path評価 |
| deterministic hit prefix | 局在時刻まで同じintegratorで再積分し、support/適用域/残差budgetを確認 |
| Brownian hit prefix | 同じsampled Hermite pathを制限。乱数を再drawしない |
| wall/axis後の残時間 | 応答後状態から新proposalを作る |
| eventと同時刻のframe | event応答後の右連続state |

exact-start一stepのdense interior誤差O(h⁴)とendpoint局所誤差O(h⁵)を区別し、smoothな固定終時刻でのglobal四次は別試験で確認する。現在のdenseがglobal四次を達成しないと断定する反例はなく、未試験範囲へ次数を拡張しないことが課題である。文書driftを直すために現在の数値方式を古い再積分の説明へ戻さない。数値式を変えなければ文書・test追加だけでengine revisionを増やさない。

producer metadataは、one-effective-Maxwellian、gradient/vorticity recovery、RZ source面積重みなどの**producer宣言**と、reader/runtimeが**数値的に検査した項目**を分けて示す。説明文字列の存在を物理validationと呼ばない。source/charge/wallの実機妥当性は、数値oracleとは別の証拠として追跡する。

## 10. G10：Talbotを原著の半径Knへ一本化する

### 規約の修正と係数の選択を分ける

[現行pure式](../solver/src/chamber_particles/physics/forces.py:254)と[compiled式](../solver/src/chamber_particles/physics/compiled.py:659)は、Kn=λ/d_dragに、半径Knに対応する係数を未変換で与える。[Talbot著者preprint](https://escholarship.org/content/qt22f5r6cz/qt22f5r6cz.pdf)に合わせ、**新revisionではKn_R=λ/a=2λ/d_dragへ一本化する**。λは同じprimitiveを使用し、HDF5の値を二倍に加工しない。drag径とmassを独立入力として扱う現行契約も維持する。

採用候補名は`talbot_cross_regime_radius_knudsen_v1`とする。次の一式を仕様のauthorityに置き、既存pure・bound・compiled経路へ反映する。

\[
\Lambda=k_g/k_p,\quad K_R=2\lambda/d_{drag},\quad
C_T=\frac{C_s(\Lambda+C_tK_R)}{(1+3C_mK_R)(1+2\Lambda+2C_tK_R)},
\]
\[
\mathbf F_{th}=-\frac{6\pi d_{drag}\mu_g^2}{\rho_gT_g}C_T\nabla T_g,
\qquad\mathbf a_{th}=\mathbf F_{th}/m_p.
\]

ここでC_Tは本書の無次元force factorであり、caseの`thermal_exchange_coefficient`であるC_tと区別する。質量、μ、ρ、T、gradientのSI単位、cold側へ向かう符号を明記する。原著の`12πa`と現行の`6πd_drag`は同値である。

三係数は現行catalogと同じくcase必須とする。原著の推奨組は`C_s=1.17, C_m=1.14, C_t=2.18`。現行で使う`1.17,1.146,2.2`を選ぶ場合は出典付きcoefficient variantとし、原著の数値と同一とは呼ばない。係数preset、Knからの係数自動推定、旧revisionの自動変換は作らない。

径Knを残してC_mとC_tを二倍にする案も数学的には同値だが、文献値の転記時に規約を再び混ぜやすい。径基準との同値変換は独立testに限定し、productionには半径基準の一経路だけを残す。COMSOLの説明図の符号・Kn表記だけから実softwareのRHSを断定せず、将来COMSOL同等性を主張する場合はG03の実point probeで確かめる。

### 変更を一つのmodel単位で閉じる

| owner | 具体的な変更 | 置換・削除するもの |
|---|---|---|
| [forces](../solver/src/chamber_particles/physics/forces.py:197) | pure evaluatorのKnを2λ/dへ変更。evaluation列名を`knudsen_radius`へ変更 | `knudsen_diameter`、diameter-based説明 |
| [global bound](../solver/src/chamber_particles/physics/forces.py:357) | lower/upper双方を半径Knで構成。現在のindependent primitive enclosureを同じ規約へ変更 | 旧Knによるbound |
| [catalog](../solver/src/chamber_particles/physics/catalog.py:1406) | 一つのnew model revisionを解決し、明示三係数をTalbotPlanへ渡す | 旧径revisionの受理・alias・active fixture |
| [runtime](../solver/src/chamber_particles/physics/runtime.py:1738) | 既存stage bindingを利用する。追加の係数変換はしない | 新しいparallel evaluatorは不要 |
| [compiled](../solver/src/chamber_particles/physics/compiled.py:642) | 既存Talbot rowのKnだけを同じ半径規約へ置換 | 第二dispatch・旧rowの並存 |
| [resolved manifest](../solver/src/chamber_particles/physics/catalog.py:398) | 新revisionとcaseの明示係数を既存resolved_modelsから記録 | 古いmodel名を新式の結果へ付ける経路 |
| 既存spec・verification・scenario | 原著式、Kn、係数policy、実期待force、非default係数を更新 | 旧式の自己再生だけによる正当化 |

伝播は`engine._prepare → resolve_physics_plan → TalbotPlan → prepare_physics_runtime → bound`と、`PhysicsRuntime.evaluate → evaluate_physics_tile_into → compiled Talbot row`である。評価だけを直しboundを放置すると、support/event enclosureが実forceを含む根拠を失う。bound・manifestまで同じ変更で閉じる。

pure evaluatorとboundの`TALBOT_DEFAULT_*`省略可能引数も削除し、直呼びtestへ三係数を明示する。係数のauthorityはcase physics blockに集約し、runtimeやtest helperへ別defaultを残さない。[technical_research](../technical_research.md:277)、[physics_models](../solver/docs/physics_models.md)、[既存decision](../solver/docs/decisions.md)、[P22計画](../implementation_plan.md:2121)を同じ作業で更新する。dated review/evidenceは当時の結果として保持する。

### 最小検証と完了条件

1. 原著の半径式を独立に書き、複数のKn_R・Λ・非default三係数でforceとaccelerationを検証する。production correction helperをoracleへ呼ばない。
2. 検証内の径式へ`C_m,d=2C_m,R, C_t,d=2C_t,R`を与え、同値性を確認する。旧径式・未変換係数なら失敗する値を含める。
3. `Kn_R→0`で`C_T→C_sΛ/(1+2Λ)`、`Kn_R→∞`で`Kn_RC_T→C_s/(6C_m)`、zero gradient、符号を確認する。漸近formulaの一致を、全粒子条件での実験accuracyと混同しない。
4. 既存[scalar/bound test](../solver/tests/verification/test_talbot_saffman.py)、[runtime parity](../solver/tests/verification/test_talbot_saffman_runtime.py)、[公開composition scenario](../solver/tests/scenarios/test_talbot_saffman_run.py)を更新し、global bound包含、pure/compiled一致、manifestの係数記録を閉じる。公開scenarioの有限値確認だけでforceを認証しない。
5. 旧model revisionと旧checkpointを既存identity比較で拒否する。schema markerの書換えや旧YAMLの黙示再解釈は行わない。

更新対象はTalbot model revision、catalog、physics runtime、compiled tileである。engine loop、integrator、proposal、RNG、case/data/result/checkpoint schemaの意味・構造を変えなければ維持できる。resolved_modelsは[resume identity](../solver/src/chamber_particles/engine.py:1420)と[final manifest](../solver/src/chamber_particles/engine.py:10083)へ既存経路で伝わる。global revision lockによってTalbot以外の旧checkpointも再開を拒否され得るが、今回だけlockを迂回しない。修正前後ではforceが変わるため、scientific payloadのbitwise一致を合格条件にしない。

## 11. G11：近似モデルの誤差とproducerの保証を用途別に閉じる

### 数式を再現できることと、物理的に十分な精度を分ける

Barnes、aggregate、effective gas、rarefied liftを一度に別modelへ置換しない。最初に、現行revisionの式再現、適用域、独立referenceとの差、用途のaccuracy budgetを同じ表で判断できるようにする。

| model | 現行証拠・限界 | 最小の改良 | 利用判断 |
|---|---|---|---|
| Barnes collection/orbital | impact-parameter oracleは有効。無電荷・弱drift collectionはMaxwellian平均に対して係数が3/4 | 既存ion-drag verificationへ独立速度分布平均を一件追加。collection/orbital/totalを分離 | 弱drift上限を狭めても25%差は消えない。要求accuracyを満たす範囲だけ採用 |
| aggregate charge | Decimal oracleは保存式の独立算術再生。静止OMLの引力増分とπ/4の係数差 | floor/regularizerが支配しない零driftでstationary OMLと比較 | 全式へ4/πを掛ける修正は行わない。引力項・total current・平衡Zの差を分離 |
| aggregate ion drag | relative-flowとE方向imageは異なるmodel | 外部V&Vでdrift・電位・screeningの小matrixを比較 | 特殊なcoflowやu_i⊥Eの一致を一般momentum transferへ拡張しない |
| effective gas | 選択したpseudobath内部の式・FDT整合は検証可能 | species-resolved referenceと縮約を比較する小verificationを追加 | mixtureの平均force、noise、heat-fluxの精度を別々に示す |
| rarefied lift | cross-product・Kn gate・boundは検証可能。C_Lの一般的実験accuracyは未認証 | producer指定範囲でoff/nominal/上下限の感度比較 | 根拠が未確定ならsensitivityとして表示。任意C_L=1を補完しない |

### Barnesとaggregateの独立reference

Barnes collectionは、同じcollection cross-sectionをshifted Maxwellianで速度平均したmomentum referenceと比較する。無電荷球の例では、wはionとparticleの相対速度として、

\[
\mathbf F_{coll}=\pi a^2n_im_i\mathbb E[|\mathbf w|\mathbf w],\qquad
\mathbf F_{coll}=\frac43\pi a^2n_im_i\bar c_i\mathbf U+O(U^3)
\]

となる。現行effective-speed式の弱drift係数は`πa²n_im_i c̄_i`で、referenceへの比が3/4へ向かう。[理論レビューの独立check](theory_independent_checks_2026-10-09.json)もこの差を確認した。これは現行のapplicability gateを満たす極限に残るmodel-form差である。[現行impact-parameter test](../solver/tests/verification/test_ion_drag.py:29)を消さず、同fileに独立3D Gaussian速度求積と次数倍化によるreference収束確認を追加する。

charged/orbitalへ広げる場合も、同じbinary-scattering/screening closureを速度平均したreferenceと表記する。self-consistent plasmaの真値ではない。collision、nonlinear shielding、negative ion compositionを別のmodel-form要因として扱う。[Khrapakらのscreening議論](https://doi.org/10.1103/PhysRevE.66.046414)に照らした適用条件を、既存physics仕様へ反映する。

aggregate chargeは[現行charge owner](../solver/src/chamber_particles/physics/charge.py:154)と独立stationary OML rateを比較する。零drift・引力側で、現行の`V_i,eff=4V_i/π`による引力増分の比π/4と、全rateの比を区別する。1 m/s floorや0.01 V等のregularizerが支配しない尺度を選ぶ。有限driftのreferenceはshifted velocity integralから作り、electron/ion frame、単価ion、温度・massの規約を揃える。[Thomas–Coppins](https://arxiv.org/abs/1305.5763)等の解析と比較する場合も、その粒径・drift・charging仮定を併記する。

実用KPIには、frozen-state current差だけでなく、同じbackgroundでのZ(t)、平衡Z、electric/ion-dragへの伝播を使う。独立referenceがない領域のtrajectory差はmodel sensitivityとする。現在のDecimal/frame/rotation/bound testを、別のphysics accuracy証拠と読み替えない。

### Effective gasはspecies縮約を明示する

共通bulk velocity・低driftの場合、各speciesのEpstein運動量交換係数をδ_sとするとdrag係数のreferenceは、

\[
\beta_{species}=\frac{4\pi a^2}{3}\sum_s\delta_s\rho_s\bar c_s
\]

である。producerのβ_effと比較し、単一種への厳密還元と混合種の縮約誤差を分ける。speciesごとのbulk velocityが異なる場合は単一u_effの成立を仮定せず、force sumをreferenceにする。

同じfree-molecular/heat-flux closureのthermophoresis referenceは、

\[
\mathbf F_{species}=\frac{32a^2}{15}\sum_s\frac{\mathbf q_{tr,s}}{\bar c_s}
\]

とする。単にΣq_tr,sをc̄_effで割る縮約との差を[existing sensitivity verification](../solver/tests/verification/test_effective_gas_sensitivity.py)へ追加する。相殺でreferenceが零へ近づく場合は`ε_abs+ε_rel‖F_reference‖`を使い、relative errorだけで合否を判定しない。q_trの符号・reference frame・並進成分と、原著の熱流束定義を揃える。[Gallisらのfree-molecular heat-flux model](https://doi.org/10.1080/02786820490490001)

Brownian併用では、選んだpseudobath内のrandom-force white-noise強度`〈ξ_i(t)ξ_j(t')〉=2β_eff k_BT_eff δ_ij δ(t−t')`の一致と、物理mixtureのfluctuationを再現することを分ける。速度SDEの瞬時拡散係数なら`2β_eff k_BT_eff/m_p²`であり、有限stepのjoint OU state covariance Q(h)とは別である。異なるspecies温度・速度分布を一つのT_effで表せると自動認定しない。productionへspecies arrayや第二engineを追加する前に、外部referenceと用途budgetで必要性を判断する。

### Producerとcoreの境界

既存[producer_metadata](../solver/src/chamber_particles/case_format.py:416)と外部receiptを使い、次を記録する。

- 対象model revisionと使用scope、組成・分布・表面条件、λの定義、q_trのframe。
- effective-gas縮約、gradient/vorticity回復、RZ source重み、source強度・wall responseの根拠。
- referenceのURI/digest、検証parameter範囲、absolute/relative budget、未検証の条件。

数値係数はcaseのphysics blockを一ownerとし、metadataへ別defaultや重複値を置かない。自由metadataに新たなcore parserや`certified: true`による合格経路を加えない。証拠不足は外部V&VのNOT_TESTEDとして残す。coreが行うのは現行stage/continuous-path numeric・applicability gateであり、source記述だけからphysical validationを生成しない。metadataは既存content hashに含まれ、data/result/resume identityで追跡できる。

### 用途budgetと式置換の条件

用途ごとに、到達時刻、到達位置、terminal/fate確率、Z(t)、付着量から必要なKPIを選び、accuracy budgetを先に固定する。次の比較で誤差要因を切り分ける。

| 比較 | 固定するもの | 調べるもの |
|---|---|---|
| model/reference | 同じprimitive、source、boundary、十分収束したh | model-form差 |
| field representation | 同じmodel、source、boundary、h条件 | native/canonical/cacheによるfield差 |
| numerical h/depth | 同じmodel/field/source/boundary | endpoint/dense/event/statistical誤差 |
| producer/source/wall sensitivity | 他の条件を固定した登録variation | 未確定input/physicsの影響 |

差を自動的に足して厳密なtotal errorと呼ばない。coupled charge/force/eventではinteractionがあるため、重要KPIについて必要な交互作用caseを確認する。Monte Carlo幅も別に表示する。

budgetを超え、用途に必要な領域が現行revisionで表せない場合だけ、新しいphysics作業で式置換を設計する。その際はpure式、derivative/invariant、global/local bound、compiled row、catalog/model revision、manifest、適用域、既存specを一つの変更で置換する。広いgateや速度上限の調整だけで、独立referenceとの差を隠さない。

G11の最初の作業は、使用modelのevidence・verification・producer説明である。Barnesは無電荷弱drift一例、aggregateはregularizer非支配の零drift一例、effective gasはsingle-species還元と低drift二species一例から始める。既存の独立checkを既存verificationへ取り込み、charged orbitalの広いmatrixやtrajectory referenceを最初から全modelへ要求しない。式を変えない限りphysics/runtime/compiled scientific revisionは維持する。source強度、adhesion/deposition、mixture分布の実験validationが未完なら、数値V&Vを終えてもそのscopeは未完として示す。

## 12. G12：境界判定とCOMSOL比較optionを物理・数値・観測に分ける

### 採用する改良と、増やさない検出経路

**現行の連続path first-hitを維持し、既存精度controlsの意味、COMSOL側の比較profile、取得できたevent証拠を明確にする。有限実壁と仮想出口を同じrunで扱う用途に限り、境界群ごとの接触geometry optionを追加する。** COMSOLの内部処理を推測した互換detectorや、endpoint/chordへのfallbackは追加しない。

境界には、接触した時刻・面を決める幾何判定、衝突前後の数値積分、stick/reflection等の物理応答、保存・描画の四つの意味がある。設定名が似ていても、この四つを同一と扱わない。

### 現行の処理と保証範囲

| 段階 | 現行owner・処理 | 維持する保証／制約 |
|---|---|---|
| geometry準備 | [geometry.prepare](../solver/src/chamber_particles/geometry.py:117)：静的XY/RZ、外周inventory、法線、flat BVH | geometryとfield meshは独立。RZ axisをmaterial wallへ入れない |
| broad phase | [exact batch](../solver/src/chamber_particles/events.py:748)とcurved batch：path enclosureのAABB。finite radiusでは膨張 | 候補を保守的に列挙する。AABB hitを物理collisionにしない |
| narrow phase | exact直線/放物線、RK4 dense曲線、exponential短縮path、Brownian Hermite leaf | endpointがinsideでもpath上の最初のhitを探す。証明不能ならsplit/failure |
| point/finite contact | [contact query](../solver/src/chamber_particles/events.py:398)、segment/capsuleと端cap | pointは中心交差、finiteは中心からsegmentまで半径a。drag径やtoleranceを半径にしない |
| hit state | [canonical hit](../solver/src/chamber_particles/engine.py:7320)、deterministic prefix再評価、Brownian prefix制限 | deterministic再評価の残差を検査。Brownianのhitを独立drawで作り直さない |
| 同時candidateと応答 | [boundary rules](../solver/src/chamber_particles/boundaries.py:21)、priority/signature/combined normal | 同時集合を保持し、互換responseを一回適用。曖昧なcornerを一面へ丸めない |
| 残時間 | deterministic再proposal、Brownian wall/axis/periodic後fresh root | hit前のaccepted prefixを保持し、応答後からtargetまで継続。axisでwall RNGを進めない |
| 保存 | eventのfacet集合・normal・pre/post state・budget、右連続frame | frame追加で検出path/RNGを変えない。描画線分をcollision証拠にしない |

例えば独立解析path `x(t)=1−6t+6t², 0≤t≤1` と壁x=0を考えると、両endpointはx=1で、chordも壁へ届かないが、最初のcrossingは`(3−sqrt(3))/6`である。これはendpoint/chord detectorを汎用optionとして採用しない理由になる。既存exact/curved検証へこの原因を表すcaseを再利用または追加し、速度やdtだけを基準に「壁を飛び越えない」と認定しない。

### COMSOLと揃えるべき設定

| 項目 | COMSOL 6.4の公開仕様 | 比較での扱い |
|---|---|---|
| Wall accuracy order 1 | 衝突前後の運動をforward Eulerで計算 | historical/Brownian比較条件として記録。bulk RK4四次とは別scope |
| Wall accuracy order 2 | 衝突前はTaylor二次、衝突後はRK2。mid-step releaseにも作用 | 決定論のreference refinement候補。単なる「root detectorの二次」と呼ばない |
| Store extra time steps | wall近傍時刻の保存・表示を改善 | 計算精度optionにしない。plotが全衝突を表示するとは限らない |
| standard particle-wall contact | 粒径を設定しても境界検出はpoint mass | native COMSOL parityではcandidateのpoint接触を明示。finite contactは別scope |
| maximum wall interactions | solverの一step内の境界interactionまたはvelocity reinitializationが上限超過するとDisappear | 物理outlet escapeと区別。数値guard由来の消失を付着・排気へ集計しない |
| union/assembly内部面 | 無BCのUnion内部面は通過、Assembly pairはParticle Continuityを必要とする | field cell界面をwallへ変換しない。identity continuityをtranslation periodicと同一視しない |

order/release/上限の根拠は[Particle Tracing interface](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.02.html)、保存と表示は[Improving Plot Quality](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_modeling.05.21.html)、point contactは[Fluid Flow modeling guide](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_modeling.05.05.html)、pairは[Particle Continuity](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.22.html)に対応する。

COMSOLはBrownianとOrder 2の組合せについて、release/wall後のRK2外挿が極端な加速度を生み得るためOrder 1を検討するよう記載する。[Brownian Force設定](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.43.html)。したがってC2を機械的にOrder 2へ上げない。Brownianのreferenceは、同じFDT・DOF・wall lawの独立ensembleとh収束で判断する。

公開資料だけではnative wallの専用root solver、内部tolerance、grazing/corner tie-breakingを確定できない。結果plotのtoleranceやfield locatorの設定をwall toleranceと推測して転記しない。実property/read-backとmicrocaseで確認するまで、その項目は未観測とする。

### Optionの配置と必要性

| option／control | 判断 | ownerと具体化 |
|---|---|---|
| integration dt | 既存を利用 | h/h2/h4でhit time、pre/post v/Z、残時間後stateを収束させる |
| geometry_rtol/roundoff_ulps | 既存を利用 | event局在budgetの単一resolver。局所facet長、global bbox/位置norm、座標・時刻ULPを反映。真のODE/SDE軌道誤差やgeometry近似誤差の保証と分ける。全source適用判定を制御する設定とはしない |
| max_refinements | 既存を利用 | 決定不能区間の有界work。増加でarithmetic floorやBrownian base不足を解決した扱いにしない |
| max_interactions_per_step | method別説明を必須修正 | deterministicはresidual intervalの二分trigger。Brownianはmacro内fresh-root event restartの上限 |
| Brownian base/max depth | 既存を利用 | uniform base収束とcandidateだけのadaptive精細化を別軸で調べる。G07/D2へ接続 |
| corner_policy/priority | 現行policyを維持 | candidate集合とresponse signatureを比較。COMSOLの不明なcorner選択へfitしない |
| COMSOL WallAccuracyOrder | external比較optionを追加 | reference runnerのrequest/contractで1または2を明示し、actual read-backを照合。coreにCOMSOL keyを追加しない |
| COMSOL event storage/cap | external比較optionを追加 | StoreExtra/status/実上限と取得可能なevent情報を登録。表示制限とsolver guardを区別 |
| boundary groupのcontact_geometry | 混在用途に限り追加 | `particle_surface`または`particle_center`。lawとは別の幾何意味としてcase/prepareへ解決 |
| endpoint/chord/Euler互換detector、位置nudge、曖昧時のstick fallback | 追加しない | 曲線の通過見落としや数値failureの物理化を防ぐ。Euler診断が必要なら外部verificationで行う |

`max_interactions_per_step`の現行二つの意味は[deterministic説明](../solver/docs/numerics.md:505)と[Brownian説明](../solver/docs/numerics.md:1231)に分散する。deterministicでは次のhitが存在しcounterが上限に達した時に残intervalを二分し、子counterを0へ戻す。event-free tailは受理できる。Brownianでは[macro入口](../solver/src/chamber_particles/engine.py:2234)でcounterをresetし、[active surface開始応答](../solver/src/chamber_particles/engine.py:2274)と[restart queue](../solver/src/chamber_particles/engine.py:3071)がwall/axis/periodic等のevent restartを累積する。応答をcommitした後、次root開始前の[判定](../solver/src/chamber_particles/engine.py:2188)が`count > limit`であり、`count == limit`は許容する。超過後にevent-free tailが残っていてもfresh rootは開始できない。axisはwall eventではないがfresh-root workを消費する点を明記する。guardによる短縮残rootは別counterで、`count > max_refinements`により有界化する。

Brownian surface-sourceの適用gateは、現行[方向判定](../solver/src/chamber_particles/engine.py:3959)で固定`128ε‖v‖`を使う。この事実も説明し、roundoff_ulpsが全ての境界判定を統一制御するとは記載しない。共通direction classifierへ整理する場合は既存event ownerへ集約し、surface sourceの許容域が変わるかを独立に検証する。今回このgateによる軌道誤判定や許容source拒否は再現しておらず、確定bugとして数えない。

現行は一case一integratorであり、二上限を独立に設定する実利用理由はまだ弱い。まず説明とlimit直前/直後の公開回帰を閉じる。将来両予算の独立調整が必要なら、意味別keyへ置換し旧keyを除去する。今回だけalias、methodごとの隠しdefault、一般budget frameworkを追加しない。

### 外部referenceの二つの比較profile

既存Java/PowerShell runnerのrequestに、wall order、wall cap、event storageの期待値を明示する。設定後のactual値とsource/companion差はG03のreceipt・preflightへ一度だけ接続する。別boundary runnerやcore engineを作らない。

1. **歴史条件の再現profile**：保存sourceのOrder 1、point contact、対応wall law、DOF、release/step条件を明示する。現行C2/C3はOrder 1の継承assert、[critical runner](../solver/tools/vv/comsol/comsol/RunM3CCriticalBoundaries.java:150)は明示setである。sourceの軸FreezeとG03のcompanion軸Bounceへの変換を別記し、companion比較を原source模型の完全再現とはしない。力が零のmicrocaseの一致を、加速度を持つ衝突区間のaccuracy証拠へ拡張しない。
2. **決定論accuracy profile**：nonzero acceleration/drag/chargeを一軸ずつ有効にし、referenceのOrder 1/2と各h系列を独立に登録する。candidateも自分のintegrator/hで収束させる。境界衝突とreleaseのmacro内phaseを複数値でずらし、first-hitとpost-event residualを比較する。target差を見てからorder/tolerance/phaseを選ばない。

これらは外部比較の設定profileであり、同じ物理caseに異なる数値条件を記録するもの。candidateをOrder 1へ劣化させてcoarse COMSOLと合わせる必要はない。相手の数値誤差、候補の誤差、geometry表現差を別々に示す。

COMSOLのDiffuse scatteringは運動エネルギーを保存する角度散乱であり、壁温度から速度を再標本化するmaxwell_thermalと別である。[Wall設定](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.04.html)。pure diffuse thermal branchは[Thermal Reemission](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.38.html)へ対応させ、DOFと分布を確認する。mixed thermal/specularをnative Mixed diffuse/specularへ名前だけで写像しない。具体的な同値外部変換が未完ならADAPTER_REQUIRED、core未対応ならNOT_APPLICABLE、意味未決ならAMBIGUOUSとしてG03の分類へ揃える。Bounce/specular、静的Stick、Freeze/hold、boundary由来Disappear/escapeはそれぞれ意味を確認する。

### Eventを観測した範囲だけ比較する

candidateのevent列とCOMSOLのterminal status/stop time、反射event、plotの折れ曲がりは異なる証拠である。StoreExtraを有効にしても、実solutionから何が取得できたかを確認し、次を個別に記録する。

- event timeの実値または観測bracket、boundary ID/selection、position、normal、pre/post stateの取得status。
- 同一時刻のnative stateがpre側かpost側か。右連続のcandidate frameと取り違えない。
- referenceのwall action count/causeが取得できたか。fs=4だけではcap由来、outlet由来、その他の未確認原因を区別できない。

正常なboundary由来のDisappearだけをescapeへ写像する。cause未確認の消失は外部比較で未同定として残し、G03の共通boundary mappingへ`other_terminal_causes_excluded`の根拠を渡す。上限を大きく設定したことだけを、cap未発火の証明にしない。

最小microcaseは一回のhitと残時間を持たせ、取得できるpre/post stateと解析解で閉じる。多重反射の全event sequenceを主張するには対応する実event dataが必要であり、隣接outputを線形補間して全hit logを作らない。boundary identityやpre/postが欠ける項目はNOT_TESTEDとする。eventと同時刻のframeは必要ならpre/postを別比較し、単一rowの連続性違いでtrajectory全体をFAILにしない。

### Geometry表現誤差をhit局在誤差から分ける

同じP1 fieldを与えても、COMSOLのgeometry表現とcandidateのline2境界が同一とは限らない。sourceのentity/curve、exportのline2、対応boundary group、法線のauthorityをreceiptへ記録する。

既存[円壁line2収束tool](../solver/tools/vv/comsol/evaluate_curved_wall_line2_convergence.py:1)は、16/32/64/128 facetでhit time/point、normal、reflected vを独立円解と比較する。このworkflowを再利用して、h固定のgeometry系列とgeometry固定のh系列を分離する。normal誤差がreflection後の軌道へ増幅するため、hit位置の合格だけで反射を認証しない。比較profileが同じpolygonを要求する場合は外部companionで同じsegment geometryを作る。native curveを保持する場合はgeometry表現差を含むend-to-end scopeへ限定する。

COMSOL側のfacet/corner識別がcandidateのcanonical rowと一対一でない場合は、semantic groupとhit位置/法線を先に対応させる。未知のnative corner primary choiceをcanonical facet IDのgolden answerにしない。

### 条件付き新option：実壁surface接触と仮想面center交差

現行[geometryのcontact mask](../solver/src/chamber_particles/geometry.py:155)は全boundaryを有効とし、[periodic準備](../solver/src/chamber_particles/engine.py:7377)だけが材料surface contactとcenter transferを分ける。したがってpositive contact_radiusでは、escape/hold群もcapsule contactになり、virtual outletのcenter crossingを同じrunで明示選択できない。これは現行仕様内の動作であり、確定bugとはしない。

用途が混在を必要とする場合、boundary groupへ`contact_geometry: particle_surface | particle_center`を明示する。particle_surfaceの実効半径は粒子a、particle_centerは0とするが、粒子のcontact_radius_mそのものは改変しない。有限実壁とlogical出口の違いをescape lawやgroup名から自動推測しない。axisはcoordinate seam、periodicは現行center transferの別topologyとして維持する。

| owner | 同時に必要な変更 |
|---|---|
| case/prepare | wall lawのparametersとは別のtyped設定としてgroup modeを一度解決し、static facet mode配列へ変換。未指定は現在と同じparticle_surfaceへ解決して記録し、半径0ではpointへ退化。旧keyのaliasを増やさない |
| geometry/events | 既存BVHを共有。surface capsuleとcenter segmentを別に局在し、uncertainty intervalで最初のeventを比較。既存periodic arbitrationの根拠を一般化し、第二detectorを作らない |
| contact normal | [現行normal](../solver/src/chamber_particles/geometry.py:290)をcandidateごとのmodeへ対応。center面はoriented facet normal、surface面は中心から最近点へ向くmaterial側unit normal |
| prefix/canonical hit | [finite residual](../solver/src/chamber_particles/engine.py:7249)と[point projection](../solver/src/chamber_particles/engine.py:7284)を一resolverへ整理し、全candidateのmode別残差を検査。primary modeだけを切り替えない |
| sources/initial check | [surface offset](../solver/src/chamber_particles/sources.py:163)、[初期半径clearance](../solver/src/chamber_particles/engine.py:7453)をsurface facetだけへ適用し、center containmentは維持。hit後/restart前の半径clearanceにも同じmodeを使用。center sourceはboundary上のcenterと既存inward departureを使う |
| boundary response/output | law/RNGは現行ownerを利用。manifestにresolved group modeを記録し、event normal/center位置を正しい意味で保存。memory/resume identityを更新 |

mixed同時candidateは、point cornerのshared-node条件も、全facetへ同じaを当てるfinite residual条件も満たさない場合がある。表面候補には`|distance(center,segment)−a|`、center候補にはpoint-on-segment残差を使用する。accepted prefixを保った一つのresolverで全candidateを確認し、順序・法線・応答が不明ならsplit/failureにする。単にmaskを変える、surface候補の中心を壁面へsnapする、混在候補をprimary面へ一律snapする、modeに応じて粒子radiusを0へ書き換える方法は使わない。center候補の許容projectionも全candidate残差と位置budgetの範囲内に限定する。

COMSOL標準のpoint比較だけなら、現行のcontact_radius_m=0で足りる。新optionはその比較のためだけに実装しない。finite contactをnative COMSOLへ比べるには、radius別のoffset geometry等を外部で独立に作り、その曲線・端cap・source/support条件まで検証する必要がある。多粒径を一つのoffset壁で同等と認定しない。

### 最小検証と完了条件

| 検証 | 独立期待値・判定 |
|---|---|
| exact通過・再入、加速hit | 直線/放物線の解析root。endpoint inside/chord clearでも最初のhitを認識 |
| post-hit residual／mid-step release | 一様加速度・linear dragのpiecewise解析解。macro内phaseをずらし、位置・pre/post v/Z・残時間後stateを判定 |
| tangent／near-grazing／corner | hit/miss/未解決を区別。incident集合、law互換性、normal、並べ替え不変性。曖昧をhit/missへfitしない |
| budget/cap | deterministic二分とBrownian macro restartを別に検証。上限直前/直後、event-free残時間、accepted prefix、失敗reason。physical escapeへ変換しない |
| axis/material/periodic | first-event順序、fold/transfer、wall/RNG ordinal、residual。同時不確定は明示failure |
| Brownian | 同じbaseでadaptive費用、base系列で到達accuracy、wall後fresh-root。G07/D2の独立first-arrivalへ接続し、連続zero-missを主張しない。terminal平面のKramers referenceをactive wall/repeated reflectionの分布認証へ拡張しない |
| curved geometry | h系列とfacet系列を分離。hit point/timeとnormal/post-vを別budgetで評価 |
| 新contact modeを採用した場合 | 定速・明確な入射・初期distance>aの平面へのsurface hit `t=(distance−a)/abs(v_n)` とcenter hit `t=distance/abs(v_n)`、端cap、複数半径、混在同時hit、surface source、point退化、各対応integrator、output/slab/resume。幅wの開口はw>2aの通過とw<2aの材料端cap先行接触を分ける |

全case×全lawの総当たりは不要である。既存verification/scenarioを再利用し、不足する原因の小caseだけ追加する。COMSOLの実比較はnonzero acceleration/dragを持つOrder 1/2、mid-step release、single reflectionのresidualから始める。critical force-free、保存stateのみ、event 0の比較を新しいboundary accuracy認証へ転用しない。

境界節を追加した時の確認では、events/geometry/boundary scenarioからfinite radius、corner、near-grazing、axis、budget等の19 testと、Brownian reflection/output/adaptive resumeの3 testが通過した。これは現行回帰の確認であり、提案optionやCOMSOLの新profileを実装・認証した結果ではない。

説明と外部profileは[technical_researchの境界節](../technical_research.md:723)、[numerics](../solver/docs/numerics.md:499)、[V&V Gate 5](../vv_methodology.md:398)、[外部workflow](../solver/tools/vv/comsol/README.md:1142)へ反映した。V&V Gate 5のWall Accuracy Orderリンクも正しいinterface資料へ修正した。test/docsとadapter設定だけならcore数値revisionは維持する。contact geometryを実装する場合はcaseの入力契約、geometry/event/source/engineの該当科学identity、manifest/resume/memoryを一括で更新し、構造不変のschemaまで一律に増やさない。

## 13. G08：scratch再利用は配列寿命を条件に進める

大きいmoduleを機械的に分割しても、allocationとaliasの問題は解決しない。現行baselineで対象allocationの時間・peak影響を測り、効果が見込めるbufferだけrun lifetimeへ持ち上げる。下記は寿命に沿った候補順であり、全段階の実装を必須にしない。数値式を変えないevent/failure再利用は、その既存回帰とpayload不変性で進められ、G07の全accuracy認証を待たない。

### 現行の寿命

| 配列・処理 | 実際の所有と寿命 | 再利用の条件 |
|---|---|---|
| field/physics stage workspace | prepareで各4個を確保。runtime evaluationはfilled prefix viewを返す | stage rateをsnapshotした後。viewをproposalに残さない |
| boundary staging | slab内で生成。payload作成時にadvanced indexing/ragged再配列でowned copy | 同期write成功後にrow/candidate countをreset |
| failure staging | payloadはowned copy。現行はfresh allocation前提 | 同期write成功後にcount=0を追加 |
| curved stochastic restart time | curved resultから配列viewを直接返す | next root bankへのコピーが終わるまでreset禁止 |
| endpoint/dense controls/enclosure | root proposalとevent prefixが同時にliveになり得る | owned snapshotを維持。単一scratchへの共用は禁止 |
| frame/probe replay | proposal stateをscatter copy。macro全slab、finalize、全出力までlive | 全frame/probe writeが戻った後だけreset |

根拠は[stage workspace確保](../solver/src/chamber_particles/engine.py:3733)、[stage snapshot](../solver/src/chamber_particles/integrators.py:3658)、[curved restart返却](../solver/src/chamber_particles/engine.py:5321)、[rootへのコピー](../solver/src/chamber_particles/engine.py:3089)、[event payload作成](../solver/src/chamber_particles/engine.py:9397)、[replay copy](../solver/src/chamber_particles/engine.py:9495)、[同期HDF5書込み](../solver/src/chamber_particles/output.py:593)である。

_SlabResult自体は統計だけを返すが、これを根拠にcurved state全体をreturn直後から再利用してよいとは言えない。stochastic_restart_time_sのconsumerまで追う必要がある。

### 段階1：event/failure staging

現在のengine内buffer型を維持し、memory plan解決後に一回確保してserial slab loopへ渡す。新しい一般SlabWorkspace frameworkは作らない。S=slab_particles、E=min(S,event_staging_capacity)、C=event_candidate_capacityとする。

| 対象 | raw numeric arrays | 注意 |
|---|---:|---|
| boundary staging | 159E + 8 + 8C bytes | strings、owned writer payloadとの同時生存は別 |
| failure staging | 22S bytes | owned payload作成時のpeakは別 |
| 現行planのevent allowance | 624S + 16C + 16 bytes | raw容量へ無条件縮小しない |
| 現行planのfailure allowance | 52S bytes | 同時生存を含む |

boundary write成功後のrow_count=0、candidate_count=0、offset[0]=0は[現行flush](../solver/src/chamber_particles/engine.py:9329)の規則を維持する。failure bufferの再利用時はwrite成功後にcount=0を追加する。全配列のzero-fillは不要で、filled prefix以外を読まない。write失敗時にbufferをresetして計算を継続しない。最初の変更はこのstagingだけで閉じ、root/RK/replayも同時に書き換えない。

### 段階2：Brownian pending rootを二bankへする

[root wave生成](../solver/src/chamber_particles/engine.py:2285)のcurrent/next allocationを、engine所有の二bankへ置換する。一bankはparticle index、start time、root ordinal、event/guard restart countの5 scalar列、start position/velocityの2 vector列、countを持つ。容量は72S+8 bytes、二bankで144S+16 bytesである。

~~~text
A=current、B=next
  → B.count=0
  → Aを消費し、restartをBへappend
  → root waveのgenerator/tree/proposal/runtimeがunwind
  → A/B swap
  → 新しいnext.count=0
~~~

curvedのrestart time viewは[_queue_langevin_event_rows](../solver/src/chamber_particles/engine.py:3089)でnext bankへコピーされるまでliveである。そのviewの解放とcurrent bankのresetを区別する。current bankのreset/swapは[whole root wave](../solver/src/chamber_particles/engine.py:2289)が戻り、generator/tree/proposal/runtimeの消費が終了してから行う。_commit_brownian_tree_proposalのreturnだけでbank全体を再利用可能とはしない。

particle ID、macro/root ordinal、tree address、physical wall ordinal、event/guard restart countを変更しない。queue countのresetと、物理ordinalのresetを混同しない。既存stochastic memory allowanceには二wave分が含まれるため、二重計上と根拠のないmemory削減を避ける。

### 段階3：profileが根拠を示した場合だけRK transient

型とstage意味はintegrators、byte accountingはcpu、run lifetime確保はengineが所有する。既存_compiled_rk4_stage_state_intoを使用し、第二kernelを作らない。

最小候補はacceleration [4,S,2]、charge rate [4,S]、v2/v3/v4 [3,S,2]、transient position [S,2]、transient charge [S]で、float64部分は168S bytesである。status/support/applicabilityの3S bytes、必要ならhalf-stepの8S bytesを追加する。gather、candidate、enclosureは別に計上する。

transient position/chargeはそのstage評価とrate snapshot後に再利用できる。v2/v3/v4はendpoint combineまで保持する。failure後に旧slabの値を読まないよう、実際に読むprefixのstatus/flags/ratesとfallback stateを初期化する。endpoint/dense controls/enclosureはowned snapshotを維持する。

replay再利用はさらに後順位とする。Kmax出力時刻、Rmax選択粒子で、現在の列構成のraw容量は42KmaxRmax + 8Kmax + 8Rmax bytesである。macroごとの[K,R] viewを初期化し、presence、lifecycle、NaN state、release-at-output-timeを現在と同じ意味で埋める。frameありmacroとprobe-only macroでrow選択が違うこと、常時保持へ変えることでphase peakが重なることをmemory planへ反映する。

### 確認と採用条件

公開scenarioで、大小slabと最後の短いslab、success→failure→success、corner candidate、Brownian複数restart、疎密output、checkpoint/resumeを原因別に確認する。private bufferの名前やallocation回数を固定するtestは作らない。

同一科学設定でfinal/frames/probes/events/failures、RNG identity、ordinalの一致を確認し、変更したruntime/memory identityは別途確認する。timing metadataはbitwise scientific payloadに含めない。異なるhの比較ではpayload一致を要求せず、G07の精度条件で評価する。

memory planは実際の同時生存を更新し、必要なmemory-plan/engine runtime revisionをownerで判断する。[CpuMemoryPlan](../solver/src/chamber_particles/cpu.py:1542)はsolver-owned arraysの予測であり、process RSSのhard limitではない。raw容量式をそのままRSS削減量と呼ばない。数値演算順・draw addressを変えるなら対応するscientific revisionも更新する。allocation減少だけでspeedupと呼ばず、hotspot改善と隣接caseのregressionを測定してから採用する。

## 14. G09：性能を「同じ精度」と「現行revision」で実証する

前回確認した現行smokeは8行×64粒子である。過去の代表100万粒子実測はengine v27であり、現行v44のSLAではない。[performance README](../solver/tests/performance/README.md:346)と既存P14/P20 harnessを拡張し、別benchmark frameworkを作らない。

### 最初の測定matrix

最初は変更対象のhotspotと隣接する低event caseの現行baselineを測る。これは精度の認証結果ではなく、改善対象を選ぶための計測なので、C/D/Mの全作業を待たずに実行できる。その後、製品の使用機能について下記の代表matrixへ拡大する。baseline、最適化の前後比較、製品性能認証を一つの完了条件へ束ねない。

| family | 狙い | scale/output |
|---|---|---|
| regular・低event | field/stage/proposalの基本費用 | 10k → 100k → 1M、none/sample |
| P1/Q1 material | locator、support、cross-cell、boundary費用 | 同じ段階。Q1 cache保証と区別 |
| event-heavy | hit/residual/corner stagingの費用 | 段階scale、複数hitを維持 |
| Brownian | midpoint/tree/restart/replayの費用 | 10kから安全に拡大、none/sample |
| coupled charge/full force | 本来の用途でのcostとaccuracy | 代表supported model組合せ |
| time field/finite contact | 現行対応機能の費用。混在contactを採用した場合はその追加費用 | 使用する軸だけ独立に測定 |

family全組合せの総当たりは不要である。代表用途、変更hotspot、製品KPIに関わる組合せを先に固定する。memory gateを通らないscaleは無理に実行せず、不適合理由を記録する。

各行は三APIをfresh childから実行し、cold起動、warm複数回median、load/simulate/open、peak RSS、solver-owned plan、artifact bytes、accepted work/event数、case/algorithm identityを記録する。私有の空Numba cache、単一thread、固定machine/environmentを明示する。既存harnessのscientific digestを使う。

### 製品合格条件の決め方

「1Mを解ける」だけでは運用要件にならない。対象physics、物理時間、event密度、出力cadence、memory上限、許容wall time、許容到達/charge/fate誤差を一行の受入条件にする。値が未確定のSLAは、測定結果とは別の未決要件として表示する。任意の速度目標や許容誤差を今回の提案で承認済みにしない。

最適化前後は同じhでscientific payload一致を確認する。hを広げる改善は、独立oracleか登録referenceの誤差budgetを満たすhを両版で選び、費用を比べる。異なるhでの速さだけをalgorithm改善と呼ばない。

COMSOL速度比はG03のmeaning/binding、対象DOF、fate/event、出力条件、精度が揃ったcaseで測る。同じP1 transport比較とnative FE end-to-end比較を分け、COMSOL field solve込み/除外を両者で揃える。compile/startupとsteady solveを分ける。Brownianは独立seed ensembleの精度・信頼区間で揃え、単一seed軌道一致を精度条件にしない。

current SLAを満たすこと、または具体的な未達hotspotが残ることを判断してから、さらに最適化する。profileなしで並列backend、GPU、DI、特殊case engineへ拡張しない。

## 15. 実装単位、依存関係、完了判定

一つの巨大変更にせず、次のreview可能な単位で進める。これは新しい恒久milestone台帳ではなく、既存implementation_planへ採用するための作業順である。

| 作業単位 | 具体的deliverable | 依存 | 完了の証拠 |
|---|---|---|---|
| A1：release gate | G01のCI step分離 | なし | 意図的CI失敗、正常clean-installed wheel |
| A2：result境界 | G04の容量/ID検査、共通file validator | なし | 公開durable/resume/recovery回帰、標準品質gate |
| A3a：数値入力の例外 | G05のOverflowを既存domain例外へ翻訳 | なし。YAML共通化と独立 | 公開API/CLIの巨大数値と正常入力 |
| A3b：YAML文法の一意化 | G05の狭いleaf、四主要入口の旧loader除去、merge方針のproducer移行。C1のtemplateも再利用 | 共通文法の確定。A3aと並行可 | nested duplicate拒否、各正常workflow、merge契約変更の明示 |
| A4：Talbot規約 | G10の半径Kn、pure/bound/compiled/catalog/spec/fixtureを一括置換 | なし | 原著oracle、径変換同値性、漸近・bound・parity・公開scenario、旧revision拒否 |
| B1：実用pairのcache修正 | static P1→regularを基準にcommon partition、weighted norm、必要gradient、bounded workspace、資源preflightを閉じ、旧合否gate除去 | 対応pairのbasis/gradient。parserを触る部分だけA3b | 解析norm、support/境界/RZ、hat/未対応pairの無publish。製品採用評価は別 |
| B2：必要なcache能力の追加 | regular/affine Q1、warped Q1 bound/adaptive、time-interval ratioを必要な順に追加 | 使用するpairのB1基盤 | 追加能力ごとの独立norm・時間相殺・budget回帰、品質/費用。未使用能力は保留可 |
| C1：現行external再実行基盤 | current writerによるfixture/template→狭いrecipe executor記録とprepare/run照合→歴史監査・再投影の区別 | 対象current入力/identity。使うYAML入口だけA3b。A2の検査方針を再利用 | license不要の対象tools tests、三API replay、旧format/executorの早期停止。postprocessへ実行lockを過剰適用しない |
| H1：core境界の意味と回帰 | G12のmethod別budget説明、既存controls、first-hit/応答/residualのscope。不足する解析小caseだけ追加 | 独立。使用methodに対応する既存verification | limit前後、同時候補、axis/periodic、accepted prefixとidentity。COMSOL profileはC2へ、連続first-arrivalはD2へ |
| C2：producer bindingとCOMSOL適用 | hash付きexpected/observed照合、狭いboundary rule、対象modelのactual probe。C2/C3では共通β・axis変換・G12外部profile | C1、H1の意味定義。全boundary accuracy系列は待たない | fixture gate、実RunnerValidation、対象axis/FDT/設定のstatus。BrownianはOrder 1を基準 |
| C3：要求scopeの外部認証 | 決定論は影響したtrajectory/加速hit、Brownian campaignは登録pilot→未使用seed final。歴史evidenceは追補 | C2。cacheは使用pair/timeに必要なB1/B2だけ | 要求scopeのsame-field、ensemble/event証拠。Order 1/2は決定論、geometry系列は曲面比較。未観測はNOT_TESTED |
| H2：混在contact geometry（条件付き） | 実利用が要求する時だけgroup mode、同一BVH、候補別normal/residual、source/clearance、identityを一括変更 | H1の意味設計。YAML文法変更と独立 | 平面・開口端cap・mixed hit・複数半径、対応methodと同一設定のoutput/slab/resume。law/RNG規則を維持 |
| M：使用modelの精度・producer scope | G11の小さいcollection/current/species reference、感度、適用範囲と用途budget | なし。使用modelだけ選択 | 式再生とmodel-form差の分離。必要な式置換は独立作業 |
| D1：使用methodの数値証拠 | 説明と独立定係数oracleの再利用→必要なendpoint/dense/kink・A/B→用途別の非dyadic/RZ/charge | 各小caseの入力能力。A3全体は不要 | 対象methodごとの独立oracle、登録confidence、output/resume identityと限定次数 |
| D2：first-arrival accuracy（用途条件） | 独立Kramers reference、h/depth別軸、到達CDF/fate | 定係数OU/conditional splitの独立検証と公開identity、到達KPI budget。D1全体は不要 | reference精度・MC幅・candidate biasを分離。未解像ならNOT_RESOLVED |
| F0：現行baseline | 変更hotspotと隣接caseを既存P14/P20で測り、費用・payload・work・memoryを記録 | なし。認証未完scopeは計測結果へ明記 | current revisionとcold/warm条件、time/plan/RSS/output。製品accuracy合格とはしない |
| E：計測で選ぶ最適化 | staging再利用を最初の候補とし、root二bank/RK/replayはprofileで必要時だけ採用 | F0、変更処理の既存回帰。数値意味変更時だけ対応D1 | 同一case/hのpayload・ordinal一致、memory lifetime、hotspot改善と隣接regression |
| F1：用途別の製品性能認証 | 使用physicsと必要scaleでaccuracy/SLAを認証。COMSOL速度比較は独立した比較claim | 使用機能のB/M/D/Hだけ。Talbot利用はA4、Brownian到達はD2、COMSOL速度claimはC2/C3。Eは必要時だけ | 用途ごとのaccuracy/cost/identity。全機能・全1M組合せを通常gateにしない |

A1/A2/A3a/A3b/A4は独立したreview単位にする。B/C/M/D/HとF0は対象owner内で並行できる。C1は実行時のactual global revisionsを記録し、A4前後のrecipeを混在させない。科学比較に未認証cacheを使わず、統計評価後にbinding/toleranceを調整しない。H1の意味定義をC2が利用し、実際の外部境界証拠をC3で閉じる。H1へC3完了を要求する循環は作らない。H2は混在接触用途がなければ実装しない。Eは変更処理の既存回帰とF0があれば着手でき、D1全体を待たない。

最初の共通修正はA1/A2/A3aで、A3b・H1の説明整理は並行できる。model所有者はA4、cache所有者はB1を高優先で閉じ、外部比較を実行する担当はC1/C2へ進む。F0で改善費用を先に測り、必要なM/Dの証拠と使用機能を揃えてF1を判断する。H2・warped/time cache・D2は利用条件に従って選ぶ。現時点で未確定の用途accuracy/SLAを任意値で承認済み要件にしない。

D1は、一つのPRで全caseを追加せず、説明・定係数公開oracle、使用する可変係数case、対象の内部時刻/RZ/chargeの順に閉じる。Mも使用modelの一つの独立極限から始める。最適化Eは、計測で効果が示せない段階では採用しなくてよい。F1の合格は選択した製品scopeについて示し、未使用能力の保留を隠さない。

### 各変更で共通に必要なgate

現行AGENTSで指定するuv lock、Ruff format/lint、Pyrefly、三つのimport contract、complexity、tests/verificationとtests/scenariosを実行する。外部toolの変更ではそのownerのcurrent workflow testsも追加して実行する。historical hash監査とreal COMSOL coverageは別に表示する。

確認する内容は、原因に対応する公開動作・独立期待値・数値不変条件である。private helper配置、source tokenの有無、全関数のcontract suiteを合格条件にしない。既存testが同じ原因を捕まえる場合は再利用し、同じvalidatorを二つ作らない。

### Revisionの扱い

| 変更 | 更新するidentity | 原則維持するidentity |
|---|---|---|
| CI、数値例外、説明修正 | package patch/関連文書。必要なinput tool identity | physics、RNG、engine numerical revision |
| resultの整合検査強化 | validationの変更記録。commit意味変更時のみresult algorithm | bytes構造が同じならschema |
| cache指標・budget | config/tool/validation/support/field-semantics、gradient/bound | production value評価不変ならengineとcanonical schema |
| Talbot規約・式 | Talbot model、catalog、physics runtime、compiled tile。関連fixture/execution lock | 演算意味が同じengine loop/integrator/proposal/RNG、各schema |
| COMSOL binding/replay | external runner/normalizer/evaluator/receipt/contract/registration | core model、engine、RNG |
| 混在contact geometryを実装 | caseの入力契約とresolved identity、geometry/event/source/engineの変更owner、memory plan、manifest/resume。変更したbytes構造だけschema更新 | 式が同じwall-law、物理RNG規則、無変更のforce model。既存2D/RZ能力のscope |
| 独立oracle/test追加 | V&V scopeと証拠 | production numerical revision |
| producer scope/近似精度の整理 | producer receipt/provenance、reference/検証scope。入力内容変更時は既存data hash | 式を変えなければmodel/runtime/compiled、既存schema |
| scratch再利用 | memory plan/runtime identity。実演算やRNG変更時は対応algorithm | 不変なmodel式・schema |
| 将来の3D能力 | state/coordinate/integrator/event/physics/V&Vの専用revision | 既存2Dを黙って別意味へしない |

revision番号を先に一律v45等へ決めず、実際に変更するownerで確定する。schema、algorithm、model、packageのversionを同じものとして扱わない。

## 16. 今回進めない改良と、その判断条件

engine/eventsの約1万行、compiled forceの大きい引数列は保守性の課題である。ただし、行数だけで分割するとstateの寿命とnumerical authorityが見えなくなる。G08を通して、accepted commit、root scheduling、record packing、replayの独立した変更理由が明確になった部分だけを既存ownerの範囲で分離する。新しいplugin registryやgeneric context bagで引数を隠さない。

新しい力を追加しても、source/壁/fieldの物理妥当性は埋まらない。当面は現行modelの適用域、effective-gas形成、OML電子reference frame、primitive recovery、source強度、adhesion/depositionの実測根拠を整理する。新しいelectron-drift gateや式を導入する場合は、独立速度積分と新model revisionを伴う別のphysics作業にする。

P17の3Dは重要だが、G01–G07の修正の代用にならない。実用途がtransverse diffusion、swirl、回転面到達の3D分布を必要とする時点で、XYZ state、RZ field mapping、回転面event、3D isotropic noiseを一体で設計する。現行2DOF RZ Brownianへθ/r項を付け足して一般3Dとして扱わない。

改良完了の判断は点数の上昇ではなく、**確認済み反例が拒否されること、数値oracleと比較scopeが独立に成立すること、現行revisionで製品accuracy/SLAが示されること**で行う。これらが揃った時点で専門レビューを再採点する。
