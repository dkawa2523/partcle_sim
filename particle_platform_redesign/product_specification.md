# 粒子輸送基盤の製品仕様

## 目的と成功条件

外部の場、静止形状、実現粒子源から、チャンバー内の粒子位置・速度・電荷と境界eventを計算します。
利用者が機能を選び、自分の入力を準備し、検査・実行・結果の解釈と再利用までCLIとPythonで進められることが製品要件です。
0.5.0はalphaです。通常の入力作成・解析・表示は計算coreと同じwheelに収録します。

優先順位は、物理・数値の説明可能性、一つの事実の所有者、10⁴〜10⁶粒子に対する容量と効率、利用・変更の容易さです。
COMSOLとの対応は外部検証の一つであり、物理的正しさそのものや通常の実行前提にはしません。

成功には、入力・model・単位・数値条件・identity・failure/coverageを追跡できることが必要です。
独立referenceと収束確認で計算を説明し、同じ科学条件のend-to-end時間とmemoryを測ります。
出力頻度、slab、worker数、resumeで科学identityを変えません。

## 現在提供する範囲

対応の詳細と拒否条件は[support and errors](solver/docs/support_and_errors.md)、選択modelは[physics models](solver/docs/physics_models.md)を参照します。
下表は提供範囲であり、selectorの全直積や任意の用途精度を保証する一覧ではありません。

| 領域 | 現行の能力 |
|---|---|
| 利用経路 | 同じCLI/Pythonで入力作成、load/check/run、結果のinspect/analyze/export/view |
| 座標・形状 | Cartesian XYとRZ meridionalの2自由度、静止2D volume、line2材料境界、RZ axis fold |
| 背景場 | regular/P1/Q1、staticまたは固定topologyの線形時間snapshot、stage位置・時刻での評価 |
| 粒子源 | realized internal/surface、固定ID/release、位置・速度・電荷・独立物性・weightを結合した入力 |
| 物理 | catalogにある固定/連続帯電、drag、電気力、重力・浮力、熱泳動、ion drag、DEP、lift、Brownian、明示reference係数 |
| 積分とevent | 厳密退化形、固定RK4、指数midpoint、対応joint OU、first hit、同時facet、axis、壁応答と残時間 |
| 接触とtopology | 独立接触半径、center/surfaceの接触意味、静的XY translation periodic |
| 実行 | resident particle state、bounded scratch、既定1/明示複数process、単一engine・single writer・毎macro同期 |
| 保存と再利用 | accepted prefix、release/boundary/failure、final、frame/probe、checkpoint/resume、immutable解釈context |
| 入力client | CSV変換、source/case出版、誤差条件付きcache、gas-only RZ/P1縮約電場 |

`check`は`run`と同じ準備を行い、対応条件・support・源・境界・資源を調べます。workerやwriter、粒子更新は開始しません。
検査成功と用途精度の認定は別です。値のclamp、別式、最近傍、欠損ゼロで未対応や失敗を隠しません。
公開操作・入力の作成・結果の再利用は[利用ガイド](solver/README.md)が所有します。

## 量と結果の意味

慣性mass、drag径、電気半径、接触半径、排除体積、weightは独立したauthorityです。
任意の源分布は外部でrealizeし、coreは再samplingしません。固定IDとcompact indexを混同しません。
drag緩和、加算加速度、電荷rate/Jacobian、noiseは異なる役割を持ち、実stageで連成します。

first hit、terminal capture、全interaction、保存時刻のoccupancy、計算粒子数、weight合計、物理flux/質量を区別します。
数値的なstick/holdは物理付着の認定ではありません。確率比較は目的に合う母集団・分布・不確かさで判断します。
prescribed数学referenceの再現を、物理modelの適用域認証や実験validationへ広げません。

保存結果のcontextはgeometryと解釈情報を保持し、元caseなしの解析・表示に使います。
背景fieldや完全な初期sourceを複製せず、contextだけで再計算可能とはしません。
現行の入力・結果・checkpoint形式は[case format](solver/docs/case_format_v3.md)と[result format](solver/docs/result_format_v5.md)が所有し、旧reader/migrationは併存させません。

## 採用した拡張方向と未実装の境界

3D粒子運動とgas interactionの拡張方針は[implementation plan](implementation_plan.md)へ引き継ぎます。
RZ場を参照するXYZ粒子と完全3D meshは異なる実装です。有限速度の平均dragと独立collision oracleがあることは、
非線形noise/SDEや粒子熱状態の輸送接続が完了したことを意味しません。
既存の線形時間場、連成帯電、joint OU、有限半径接触、process実行は実装済み能力として維持します。

coreの責務に、COMSOL UI、流体・伝熱・詳細plasmaの再計算、任意CAD修復、二方向連成、粒子間衝突を含めません。
3Dや全材料電場等の未実装項目、用途ごとの数値資格、源rate/weight・surface physics・実験比較の不足は、それぞれ別の残件です。
既存reportは説明clientであり、新しい計算GUIではありません。

## 文書と証拠

責務・依存は[architecture](architecture_proposal.md)、変更と確認は[quality tooling](quality_tooling_plan.md)、
式・methodはsolver/docs、外部比較は[V&V methodology](vv_methodology.md)が所有します。
過去の採否は[判断履歴](solver/docs/decisions.md)、実行sourceと観測値は各receiptに保持します。
以前のPASSを現行sourceや別modelへ付け替えず、文書の変更だけで実装・数値・物理の資格を更新しません。
