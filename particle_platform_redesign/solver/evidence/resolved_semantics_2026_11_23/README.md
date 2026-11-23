# 責務・受渡し・数値説明の根因調査と再構成

開始source: `824a7ac8df1e4d69f9fc17f8fa3f7b27598af290`。対象は現行基盤の全主要領域であり、
歴史的Vigusレポートの旧実装を現在sourceと同一視しない。この記録は変更理由と実行証拠を集める。
現行仕様の所有者は[architecture](../../../architecture_proposal.md)、[物理](../../docs/physics_models.md)、
[数値](../../docs/numerics.md)、[比較方法](../../../vv_methodology.md)である。

## 根本原因と変更

問題はfileの大きさだけではなく、同じ事実の複数管理、入力から結果までの受渡しの不明確さ、
異なる証拠を同じ判定に使うことにあった。全域prepare・行別stage失敗・局所証明は固有の責務として残した。

| 領域と利用への影響 | 根本原因 | 行った改良・撤去 | 主な検証 |
|---|---|---|---|
| 入力追加・削減後の実sourceが不明確 | bundle内table、別sources引数、recipeの暗黙overlayが同じ集合を決めていた | `create_case(bundle, settings, output_directory)`へ一本化。recipe 2で全table集合を明記し、未記載tableを除去。実release選択はsettingsへ分離 | 独自2 CSV source追加、片方削減、再生成、解析hit時刻・重みを公開入口で確認 |
| producer成功後に説明や編集元が欠ける | bundle/case/receipt/編集元を別々に公開していた | case＋編集元、bundle＋receipt、export/viewをstagingで完成後公開。通常3producerと外部Vigus/F02/P1を更新。任意report_path/--reportを撤去 | 失敗注入で未完成directoryなし、入力不変、同じ出力先へ再試行成功。各producer公開入口を実行 |
| model追加がmetadata・binding・式・compiled ABI全体へ波及 | 制約が正値boolとmodel別特例へ分散。選択identityをruntimeから逆推定し、約100引数のABIが意味を隠していた | catalogがfinite/positive/nonnegativeを宣言。descriptor/bindingをplanが保持。family別固定numeric recordを受け取る7引数へ。重複production scalar式を共有 | 独立Decimal/積分/解析oracle維持。3family各4096行のstage payloadは前後同一。exact zero拒否も修正 |
| 計算可能なstageをpathの証明と誤認する | primitiveとproof roleが分かれず、未対応寄与を暗黙に0とする余地があった | 選択modelのglobal/local/constant証明coverageをprepareで確認。欠落を0にせず、constant specializationに全作用・連成・座標・noiseの肯定的証拠を要求 | 適用域外、局所証明、連成charge、constant specialization反例を含む科学検査 |
| 親・worker・checkpointの変更が広く波及 | 完全なSimulationCase、解決済み事実、worker settingsが混在。永続列も複数定義だった | 親の_RunFactsとtyped PreparedTransportを分離。workerは必要settings/配列だけを受ける。12永続列をPersistentParticleStateが所有しsnapshot/allocation/bytesへ投影 | resume/rollback、worker/slab/output identity、memory、異常終了cleanup。checkpoint schema・更新規則は維持 |
| dt変更の説明が実計算精度に対応しない | nominal dt、field knot後macro、受理root、証明内部細分、保存clockを混同 | MacroGridが固定macro/field knot/丸めを所有。engineと比較器が同じ時計を利用。窓の実macro refinementと、既知の再積分なし条件を資格の前提として分離 | macro64/64/64反例、macro2/4/8でも受理root16/16/16のexp midpoint/RK4反例、実root16/32/64正対照。raw差・実用比較は維持 |
| 安全budgetや全域上界を精度と誤読する | 異なるbudgetの意味が曖昧で、準備上界の公開keyがmaximumだった | NumericalWorkPolicyがproposal/certificate/係数guard/残時間budgetを区別。prepared_stability_upper_boundsはrequested base dt×準備全域上界と明示 | budget・数値・比較演算を変えず公開manifestと全callerを更新 |
| clientごとにID・identity・分母を再解決する | nested manifestは可変。ID照合が粒子数×要求数の探索だった | 再帰的に不変化し、JSONは独立documentへ。束縛ResultCohort/common identityをanalysis/export/viewで再利用。整列ID索引を一度解決。基本lifecycle意味を一ownerへ | 1M行/30k ID選択中央値5.845 s→0.01563 s。同session選択処理の計測でsimulation速度ではない。reuse/改変拒否/JSON/context差替え検査 |
| 利用例・CLI・編集入力が個別に陳腐化 | 例題定義が分散し、一部source/rowだけを編集元へ戻していた | 静的例題定義からCLI choicesを導出。全source全rowから編集入力を作成。新registryは作らない | 5例題、独自internal/surface、CSV/cache/electric、table削減をWindows/Linuxのwheelから実行 |
| 検査やreportのPASSが現行sourceを意味しない | 内部dict型/revision固定、古いcaller、過去PASSと現在sourceが混在 | JSON意味/公開動作へ検査を更新。性能clientも既存Pyrefly対象へ。source/入力/範囲へ実行証拠を束縛。reportは結論のconsumerに留める | 全体core/scenario、外部suite、9性能系列、clean wheel、3観点review。過去失敗は履歴と区別 |

## 追加・拡張・削減の変更単位

| 変更 | 変更するownerと受渡し | 維持する境界 |
|---|---|---|
| 同force familyの別model | catalog入力/適用域/proof role→式/bounds→stage binding→固定compiled record→独立oracle/例 | engine/writer/COMSOL専用dispatchを増やさない |
| 新物理category | 上記に合成順・全寄与coverage・constant証明・workspace/memory・結果説明を追加 | 汎用plugin/DIや第二engineを作らない |
| 新動的state | 初期authority→stage/dense law→accepted prefix/event→persistent列/checkpoint/resultの縦切り | 後処理だけの更新で連成を分断しない |
| 壁law | 応答・RNG ordinal・manifest・保存則・残時間scenario | first-hit/corner集合をlaw別に複製しない |
| field producer | 外部primitive/単位/基底→canonical bundle＋receipt→共通load/check | coreはproducer名/native tagを知らない。依存charge/ion/DEP再生成はproducerの明示責務 |
| 解析/表示 | 同じResultView/cohort/contextを消費しcoverageと量を説明 | state/ID/lifecycle codeを再定義しない |
| 機能削減 | selector/宣言→production→caller/config→test/docsを同時撤去 | 旧新二重経路・互換aliasを残さない |

exact/curved/OU、global/local bound、support/誤差認証、terminal/first arrival/occupancyは必要な違いを維持する。
production数値規則は一owner、独立oracleはverification/V&Vへ残す。

## 実行と独立確認

最終集約は[実行確認](verification.json)、[独立review](reviews.json)、[性能](performance.json)、
[配布identity](distribution_identity.json)に記録した。巨大raw resultはGitへ含めずhashとローカルlocatorを残す。

- 最終core/scenario **1227 passed /572.89 s**。最初の変更中実行13 failed/1207 passedとは区別する。
- 最新portable COMSOL/gas-collision外部suiteは**146 passed /32.83 s**。lock、Ruff、Pyrefly、6 import契約、CC<16も成功。
- 9性能系列を現行入口から再実行。同一YAML/HDF5・同条件の1万粒子では1 worker 28.771 s、4 workers 17.400 s、failure/残留childは0。
  変更前の同一入力とowned memoryも比較した。単発・別sessionの時間差を一般的speedupと認定しない。
- A/P各287 ID・40 seed・同windowの**80再計算完了、failure 0、全保存科学payloadが変更前と一致**。
  旧出力一致だけで正しさとせず、保存native再集計、解析/製造解、OU/charge/壁/場の独立検査を併用する。
- [population比較](population_comparison.json)、[全80入力・結果identity](population_identity.json)は現行再実行に基づく。
  A/Pを80独立replicaと数えない。大局的輸送・停止先・時間は実用的に対応し、面33と100 µs survivor速度には局所留保が残る。
- [Windows](clean_wheel_windows.json)と[Linux](clean_wheel_linux.json)の独立installから4 core API、worker、time field、contactを実行。
  [Windows通常workflow](installed_workflow_windows.json)と[Linux通常workflow](installed_workflow_linux.json)で全入口・5例題・独自入力・結果再利用を実行した。
- 独立利用者reviewはCLI/CSV/2source/削減・再生成/重み/cohort/3producerを実行した。
  構造reviewはexact zero拒否とJSON境界caller漏れ、科学reviewは受理root不変の資格gate不足を検出した。
  必須指摘は修正後の元witnessで再確認し、未解消0。独立した最終構造reviewの45検査も成功した。
  採点は構造85/100、保守80/100、科学・数値の説明88/100であり、native精度や実機予測の認定値ではない。

## 結論の範囲

実用比較、数値的native parity、装置の物理予測は別の判断である。乱数pathを一致させる必要はないが、
母集団・clock・source/weight・停止/欠損の意味・複数実現の不確かさ・用途上の許容を明示する。
各力の寄与率、native未保存反射、disappearance原因は今回新たに確定していない。

今回の実行対象は、現行workspaceで入力と保存native evidenceを同定できるA/P各287 IDである。
歴史的レポートの43,220粒子S2/S3-Mを今回再計算したとは称さない。そのportable caseと別管理artifactは現行cloneに含まれない。
同じ歴史的計算の再調査には、対応case/export/result/source-mapをhash付きで引き渡すことが再開条件となる。

現行の独立electric builderはgas-onlyである。歴史的レポートの全材料・浮遊V/Q版を実装済みとは称さない。
全材料V/Q、物理3D、装置adhesion/flux等は入力/物理検証を要する条件付き拡張であり、
今回の保守・拡張性改善の完了へ混ぜない。
