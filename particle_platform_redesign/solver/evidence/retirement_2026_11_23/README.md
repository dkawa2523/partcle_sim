# 現行経路への整理

対象は現在の入力作成、計算、保存結果の利用、外部V&V、性能測定、配布・説明の全経路。
作業前sourceは`e9529fc024d2591d018b5809cd8c86a8f9b89f0f`、作業branchは`refactor/retire-obsolete-workflows`。
旧仕様・旧成果物との互換は提供しない。現在必要な科学的確認は現行経路へ移す。

## 調査と責務

| 利用経路 | 現行の所有者 | 確認対象 |
|---|---|---|
| モデル選択・設定・登録 | physics/catalog、case、唯一のCLI | selector、required fields、revision、対応組合せ |
| 自分の入力を準備 | tools側のinputs、CSV/source/cache/electrostatic | recipe→canonical→check、物性authorityとprovenance |
| 計算と再開 | engine、transport、execution、数値owner | accepted state、stage、event、RNG、容量、single writer |
| 結果を解釈・再利用 | output/ResultView、analysis/export/view | context、cohort、分母、terminal、欠損、独立した出力 |
| native参照との比較 | tools/vv/comsol | 同定・正規化・実用比較と数値認定の区別、解析/native参照 |
| collision参照 | tools/vv/gas_collision | productionと独立した衝突積分・jump、適用範囲 |
| 性能・配布 | tests/performance、release smoke/workflow | 入力・実行・科学参照・評価・生成結果を分離 |
| 説明・report・技能 | README/各owner文書、snapshot、4技能 | 現行入口、実行証拠、欠落と0、出典・identity |

公開入口、model登録、設定、field/source/stateの受渡し、数値kernel、出力、通常client、
外部V&V、性能、wheel/CI、report/技能を調査した。実際に行った確認は以下のreceiptへ区別して保存した。

## 撤去と保持の基準

旧版だけを実行・投影・認定するキャンペーン、互換入口、重複した数値経路とそれら専用の確認を撤去する。
ソフトウェア試験の旧候補出力・source snapshot・版別release生成物・大量の重複図も対象とする。
撤去はGitの既存履歴に保持し、実行可能な互換層へ戻さない。

元MPH、native観測、科学的な独立参照・導出・実験記録、`model_dataset/`、`old_code/`のread-only archive、
環境と学習済み資産は撤去の推定対象にしない。reportの共通runtimeは製品所有の現行依存であり、solverの旧互換層ではない。
入力case/data3、result4、checkpoint2は現在必要な形式であり、数字の大小で旧仕様とは判定しない。

[A/Pの保存科学参照](../platform_restructure_2026_11_23/science/README.md)は、特定の入力・source・観測量の証拠として保持する。
過去の計算・図の一致を、新sourceの独立精度認定や新しい標本数として数えない。

## 実装・撤去の結果

- sourceの旧`prepare_case`を撤去し、recipeから一つのpublisherへ接続した。完全な設定・assumption provenanceを保持する。
- 旧scalar event/axis/preclassifier 52定義と未使用helper7件を撤去した。必要な検査は現在のinto kernelと解析解へ移した。
  残した数値定義のASTは独立に照合して変更なし。allocating入口はbounded storageを準備して同じkernelへ渡すため保持した。
- 固定native campaignの実行・投影・seed lock・旧source freezeと専用設定/tests等160ファイルを撤去した。
  generic descriptor、population、係数producer、独立local式、field比較、小型native/解析benchmarkの役割を残した。
- 旧accepted Case-P hash guardと固定64 cadenceをmonkeypatchする性能経路を撤去した。
  現行の9 workloadへ整理し、公開APIのworker比較を追加した。P19のsource物性展開と現行容量検査も修正した。
- 旧release/candidate出力、重複図、source snapshot、pytest中間結果、一度限りの旧campaign scriptを撤去した。
  履歴はGitへ保持し、current runtimeから実行するalias・reader・変換dispatchは残していない。
  追跡外でも明示した旧software出力を1,794ファイル、約2.38 GB整理した。
  源入力に属する12 assetは別管理の入力資料として保持した。最後の中間trial2件も具体的な生成fileの検査後に撤去した。

履歴判断文書は当時の理由と科学的記録と明示し、現在の仕様・責務・残件の所有文書へ案内した。
AGENTSへ個別の撤去履歴を積まず、現行の責務と互換を提供しない方針を記載した。

## 実行した確認

| 対象 | 実行と結果 | 証拠 |
|---|---|---|
| 品質 | lock、Ruff format/lint、configured Pyrefly0、import6、CC<16 | [verification](verification.json) |
| core/scenario | 全体1,190 passed /561.21 s。レビュー修正後のevent62 passed /5.59 s。現行collection1,192件の2追加条件を別確認した | 同上。件数を足して1,252件とはしない |
| 実配布 | clean runtime wheelのconsole/module/Python、実process・cleanup、13操作・5例題、自分のCSV/internal/surface、元caseなし結果利用 | [wheel](clean_wheel.json)、[installed flow](installed_workflow.json) |
| 配布漏れ | 予定commitだけの隔離src/cwdで外部134 passed /164.56 s。必要nativeログを含み、元workspaceの未追跡資料へ依存しない | [verification](verification.json)、[independent review](independent_review.json) |
| native/解析 | XY保存21観測、fresh RZ132/132、曲線壁の収束/解析参照が成功 | [V&Vの条件と結果](vv/README.md) |
| 衝突参照 | 2条件、20万jump、1万interval、独立積分と解析極限。関連7 passed | [衝突](gas_collision/README.md) |
| 性能 | 現行9経路を実行。1万粒子320 macroはW1 29.278202 s /W4 18.353255 s、全science一致、既知target10,000、failure/残留worker0 | [実測](performance/README.md) |
| 保存A/P | 各40実現を現在のreaderで再集計し各15群の値を再現。新solve/新標本ではない | [population recheck](population_recheck.json) |
| 説明 | 33文書・4技能の参照/接続、snapshot/helper7 passed、公式report build、desktop/390pxとsource overview/data previewを確認 | [report](../../../report_app/README.md)、[表示記録](report_verification.json) |

性能はwarmup後各worker1観測、無frame/probeの条件である。約1.5953倍は局所観測で、一般scaling/SLAや
全10万/100万粒子release matrixを認定しない。実用A/P対応と境界33・100µs survivor速度の留保を維持し、
乱数path一致、全native方程式、実機のadhesion/fluxの認定へ広げない。

## 独立レビューと要件監査

[利用者・実装・残存/誤削除レビュー](independent_review.json)と[独立科学レビュー](scientific_review.json)を実施した。
抜けていた区間外交差・遠方巨大facetの2回帰を現行APIで復元し再実行した。
native参照ログの配布漏れを修正し、隔離した予定commitで再実行した。
判断履歴を現在の仕様と誤読する冒頭を修正した。追加の必須指摘は残っていない。

| 要件 | 完了の根拠 |
|---|---|
| 全主要領域・独立経路を対象とする | 上表のcatalog→入力→単一engine→保存→client、外部2経路、性能/配布/report/技能を対応付けた |
| 名前・静的未参照だけで削除しない | 登録・設定・import・呼出側・CLI・CI・実filesystemと実行を照合し、必要な役割を現行ownerへ移した |
| 互換/代替経路を残さない | 旧source facade、scalar探索、固定campaign、旧cadence/guardを専用設定/testsと撤去。未知/旧形式はstrictに拒否 |
| 科学確認を簡潔に維持 | current kernel＋解析解/native観測/独立OU・衝突、容量/保存/失敗確認を保持。private呼出回数・旧版固定だけの検査を撤去 |
| benchmarkを再実行し結果を更新 | 9現在経路、別精度確認、native/解析3経路、衝突を実行し表/reportを再生成した |
| 実データ・環境を推定削除しない | MPH/native観測・科学研究記録・入力asset・model_dataset/old_code/envを保持 |
| 横断残存/誤削除確認 | 実filesystem旧module/設定、呼出側、CLI、技能、CIを再検索。独立レビューと隔離checkoutで確認 |
| 使い方・責務・互換方針を更新 | README、AGENTS、現行所有文書、4技能への影響先、reportを整合した。撤去履歴は本記録 |

この依頼の実行可能な必須作業に未完はない。全材料V/Q、物理3D、装置予測など製品拡張の条件付き項目は
現行の[実装計画](../../../implementation_plan.md)へ保持しており、今回の撤去完了や科学認定へ混ぜていない。
branch/remote CIの最終状態はGitと実際のCI記録へ限定し、local passから推定しない。
