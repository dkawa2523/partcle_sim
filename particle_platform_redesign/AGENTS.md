# Project change rules

この規約は `particle_platform_redesign/` 以下に適用します。visible reportには専用の
[AGENTS.md](report_app/AGENTS.md)も適用します。旧archiveの規約を現行projectへ持ち込みません。

## 判断と参照

利用者が機能を選び、自分の入力を準備し、実行して結果を解釈・再利用できることを目的に変更します。
優先順位は、物理・数値の正しさ、一つの事実の所有者、規模に対する容量と効率、読みやすさ、変更・削減の容易さです。

[文書案内](README.md)から対象のownerを読みます。責務・依存と変更の受渡しは
[architecture](architecture_proposal.md)、現行能力は[製品仕様](product_specification.md)、
未完の実装と条件は[implementation plan](implementation_plan.md)、作業手順と確認は
[quality tooling](quality_tooling_plan.md)が所有します。式・適用域と数値規則はsolver/docsの各ownerを参照します。
この規約へmodule所有表、実行コマンド、schema/revision一覧、作業日誌を重ねません。
矛盾はその事実のownerで解決し、呼出側と説明を同じ変更で合わせます。

## 実装境界

- productionは独立uv project `solver/` だけへ置きます。root環境・旧package・旧test/configは復活させません。
- 旧実装は失敗例・式候補・provenanceの調査対象です。採用する内容は独立に導出・検証し、copyやimportで移植しません。
- core/APIから通常client、native adapter、V&V、analysis、visualization、reportへ逆依存しません。
  import違反をfacade、re-export、`TYPE_CHECKING`で隠しません。
- engine、canonical形式、設定parser、数値規則のownerを重複させません。通常CLIとPythonは同じ処理を呼びます。
- 公開計算APIは四操作を保ち、mutableなprepare/workspace、writer、backendを公開しません。
  `check_case`は実行同一prepareを使い、worker・writer・粒子更新を開始しません。
- 公開例外は既存三分類を使い、catch-allや成功形defaultで不正入力・内部不変条件の破損を隠しません。
- 新しい一般plugin/DI/ODE framework、別scheduler、model専用engine、重複validator、広い契約/診断基盤を作りません。
- `models/helpers/utils/contracts`等の一般置き場を増やしません。行数で分割せず、独立した変更理由で責務を決めます。
- field生成の宣言済み依存と全output群を出版前に照合します。独立prescribed fieldへ依存を推測せず、検査で物理modelを選択・再生成しません。

## 数値計算の不変条件

以下を変更時に必ず維持する。

- canonical値とruntime値はSIを使い、量名へ単位を含める。
- `mass_kg`を慣性のauthorityとし、実行中に粒径から再構成しない。
- drag径、electrostatic半径、排除体積、model weightを混同しない。
- fieldの値とsupport判定を分ける。
- trial値を有限化しても、support外proposalをeventなしで受理しない。
- safety failureとaccuracy不足を同じstatusにしない。
- 欠損、非有限、適用外modelをdefault、epsilon clamp、別modelで隠さない。
- 固定長の位置押戻しや散在するmagic toleranceを追加しない。
- toleranceはgeometry scale、時間、速度、float64 roundoffから一箇所で解決する。
- field、force、chargeは実際のintegrator stage位置と時刻で評価する。
- continuous chargeと運動を同じ状態として連成する。
- chargeだけをsubcycleして後から運動へ渡さない。
- accepted stepだけがparticle stateとcell hintをcommitする。
- first hitをendpointだけでなく`StepProposal`の経路上で求める。
- hit状態は同じintegratorでhit時刻まで再評価し、残時間を再実行する。
- 要求出力時刻のためにproduction stepを分割しない。
- 個別release時刻のために全粒子stepを分割しない。
- RZ meridionalのaxis crossingをwall reflectionとして扱わない。
- point particleと有限半径接触をtoleranceで混同しない。
- 数値式、補間、event順、tolerance、RNG keyの変更をrefactorと呼ばない。

## 状態・event・RNGの不変条件

- physical state、lifecycle、support、accuracy、run integrityを一つのstatusへ畳み込まない。
- 永続particle列はresident stateと同じ宣言からprojectionし、checkpointのcursor/index/hintは各ownerで検査する。
- 粒子IDは固定し、compact配列のindexをidentityにしない。
- output frameはaccepted `StepProposal.state_at()`から評価し、resident stateを出力時刻へcommitしたり
  production stepを分割したりしない。曲線pathの`state_at()`はproposal始点から同じintegrator規則で再評価する。
- wall eventは最初のhit時刻だけを応答へ渡し、その局在budget内で同時なincident facet集合を保持する。
  exact、一般RK4、exponential、Brownianで同じ候補集合、corner policy、combined normalを使う。
- zero-time surface departureを位置nudgeで解決しない。
- RNGはparticle ID、model stream、物理的draw ordinalから決める。
- tile幅、出力頻度、checkpoint/resumeでdrawとeventを変えない。
- stochastic stepを分割する場合、独立乱数を引き直さず条件付き分割を使う。
- Brownianの`interval_tree_depth`は全rootへ一様に適用する数値path精度のauthorityとする。
  `adaptive_max_depth`はそのpathがwall、RZ axis、または証明不能区間の候補になった時だけ条件付き分割する上限であり、
  base depthを置き換えない。両者が同じなら従来の固定depth pathへ退化する。
- 有限depthのcubic Hermite pathはsampleしていない連続OU軌道のexact first-passage、zero miss probability、
  またはbase depth不足の自動補償を主張しない。

## 容量・並列・出力の制約

- resident stateとbounded scratchを区別します。outer batch/out-of-coreや追加JITは、必要性と費用が示されるまで導入しません。
- hot loopに粒子ごとのPython callback、thread-ID依存private API、全量proposal/event bufferを追加しません。
- worker数は実行資源です。model、dt、物理RNGの調整ノブやcase設定へ混ぜません。
- 同一runの並列化は既存process ownerへ接続します。ThreadPoolExecutor、Numba内部parallel mask、model別backendや自動tunerを足しません。
- 親時計・single writer・毎macroのaccepted commitを維持します。worker側writer、macro全量sort buffer、第二数値loopを作りません。
- 共有入力・state/replay、worker専用scratch、IPC、移送時の同時live容量を省略せず会計します。
  stochastic scratchは到達可能な最大depthから計画し、実際のrefine数が少ないことを理由に過小評価しません。
- storage到着順を物理順のauthorityにしません。公開canonical順、RNG、科学payloadはslab/worker/output/resumeで変えません。
- 常設診断はlifecycle/event/failure件数、集約timing、memory、短いfailure reason、実revisionに留めます。
  full trace/stage probeは明示IDだけに限定し、診断用の別state・engine・結果形式を作りません。
- caseと編集元、bundleとreceipt、export/viewは完成物として公開します。結果identityと実際に検査した範囲は別の事実です。

## 検証と主張

解析解、manufactured case、独立式・OU・collision referenceを保持します。既存出力との一致だけを正しさの根拠にしません。
private helper名・呼出順・file配置を固定するtest、重複assert、失敗を隠すignoreやbaselineを増やしません。
原因を表す最小回帰caseを選び、必要な科学的性質を維持したまま旧確認を更新・統合します。

適用する確認と変更の完了条件はquality toolingで一度定義します。文書だけの変更で数値・性能・native資格を取り直したとは言いません。
過去のPASS・採点・benchmarkは実行sourceに限定し、実行したことと記述・作成したことを分けます。
COMSOLの実用比較、数値的native parity、実験的validationを混同せず、数値stickを物理付着やfluxへ換算しません。

## 置換・撤去

置換時は旧入口、selector、設定、呼出側、専用test、説明、不要な生成物を同じ変更で除去し、互換reader/aliasを残しません。
名前や静的未参照だけで削除せず、公開入口、登録、動的利用、実入力と保存物の役割を確認します。
元MPH・native観測・独立oracle・実験記録・外部入力・実行環境と、廃止softwareの生成物を区別します。
履歴やhash固定の証拠を現在の仕様へ書き換えず、現行文書からその時点と限界を明示して参照します。
