# 粒子輸送基盤の製品仕様

## 目的と成功条件

外部の場・静止形状と実現粒子源から、半導体製造チャンバーの粒子位置・速度・電荷・境界eventを計算する。
利用者が機能選択、自分の入力作成、検査、実行、結果の解釈・再利用までCLIとPythonで完結することを製品要件とする。
0.5.0はalphaで、通常の入力作成・解析・表示を一つのwheelへ含める。
優先順位は物理・数値の説明可能性、一つの事実の所有者、10⁴〜10⁶粒子への容量と効率、利用・変更の容易さである。

COMSOLとの対応は外部検証の一つで、物理的正しさそのものや実行前提ではない。
入力・model・単位・数値条件・hash・failure/coverageが結果に残り、独立reference/収束で説明できることを要する。
同じ科学条件のend-to-end時間とmemoryを測り、出力頻度・slab・workers・resumeで科学identityを変えない。

## 所有文書

| 内容 | 権威 |
|---|---|
| 利用目的・提供範囲・非目標 | 本書 |
| 責務と依存・受渡し | [architecture](architecture_proposal.md) |
| 現行の式・適用域・積分・schema | solver/docsのphysics_models、numerics、case/result形式 |
| 実装・現在の残件 | [implementation plan](implementation_plan.md) |
| 導出と文献根拠 | [technical research](technical_research.md) |
| 品質・配布 | [quality tooling](quality_tooling_plan.md) |
| 外部比較の判断 | [V&V](vv_methodology.md) |
| source別の既存性能資格 | [parallel execution](solver/docs/parallel_execution_plan.md) |

矛盾は所有者で解決し、例外やfallbackで隠さない。式・schema・revision一覧を本書へ複製しない。
以前の詳細仕様と採否はGit履歴へ保持する。native観測・実験記録と現行の実行証拠は、旧software生成物と区別して保存する。

## 利用経路と責務

`models`で必要入力・単位・対応条件を確認し、`init`または`inputs`と`create`でcaseを作る。
`check`は実行と同じprepare、`run`は実行とdurable resume、`inspect`はfailureとcoverage、
`analyze/export/view`は分母・weights・statusを保つ結果利用を担当する。
core APIは`load_case/check_case/simulate/open_result`。通常clientは同じdistributionの外層が公開する。
consoleとmoduleは同じdispatcherを呼び、外側で数値kernelやvalidatorを再実装しない。

| 領域 | 現行能力と境界 |
|---|---|
| 座標・形状 | Cartesian XY/RZ meridionalの2自由度、静止2D volumeとline2境界。axisは材料壁でなく座標fold |
| 場 | regular/P1/Q1、static/固定topology linear-time。stage時刻で補間、knotでmacro分割、時間範囲外拒否 |
| support | 必須fieldは同一layout。一般曲線unstructured場はparticle geometryとの同一mesh等の証明可能な組合せを要求 |
| 源 | realized internal/surface、固定ID/release、位置/表面位置・速度・電荷・独立物性・weight |
| 物理 | catalogの固定/連続帯電、drag、Coulomb、gravity/buoyancy、thermophoresis、ion drag、DEP、lift、Brownian、明示reference係数 |
| 積分 | ballistic/一定加速度の厳密退化形、固定RK4、指数midpoint、対応joint OU。実stageのx/v/Z連成 |
| event/壁 | first hit、同時facet、priority、axis、反射後残時間。stick/escape/hold/specular/isotropic/restitution/thermal/確率stick |
| 接触/topology | 独立接触半径とgroupのcenter/surface意味、静的XY translation periodic |
| 実行 | resident state、bounded scratch、既定1/明示複数process、共通slab、親時計/single writer/毎macro barrier |
| 結果 | accepted prefix、release/boundary/failure ledger、final、frame/probe、checkpoint/resume、immutable context |
| 上下流 | CSV変換、最初のsource/case作成、誤差認証cache、gas-only縮約電場、JSON/CSV解析・形状付き表示・説明report |

selectorの全直積は保証しない。model/field/積分/壁の対応と適用域はprepareが受理/拒否する。
checkは実行準備検査であり、用途精度認定ではない。clamp/別式/最近傍/欠損ゼロで失敗を隠さない。

## 科学的な意味

慣性mass、drag径、電気半径、接触半径、排除体積、weightを独立authorityとして扱い、runtimeで相互に再構成しない。
任意放出式・分布は外部でrealizeし、coreは再samplingしない。固定IDをcompact indexで置換しない。
drag緩和、加算加速度、G/J、noiseには異なる役割があり、同stageで連成しても任意callbackへ統一しない。
prescribed数学referenceの対応を物理modelの適用域認証や実験的validationへ拡張しない。
first hit、terminal capture、全interaction、frame occupancy、computational count、weight合計、物理flux/質量を区別する。
数値stickは実付着の認証ではない。確率軌道は同じIDの重なりでなく目的に合う分布と不確かさで判断する。

result5のcontextは形状・解釈metadataを一度保存し、背景場・初期全配列を複製しない。
contextだけで再計算可能とはしない。旧result reader/checkpoint migrationは併存させない。

## 非目標と引き継ぐ残件

coreにCOMSOL UI、流体/伝熱/詳細plasma再計算、任意CAD修復、二方向連成、粒子間衝突を持ち込まない。
既存reportは説明clientで、新しい計算GUIではない。旧solver archiveは実装母体にしない。
3D、非線形noise/粒子熱状態、全材料/浮遊VQ、不連続/周期/streaming場、moving geometry、surface charging/再飛散、
同期改善、NUMA/GPU/out-of-coreは[backlog](implementation_plan.md#条件付きbacklog)の着手条件とともに保持する。
実装済みlinear-timeやcollision oracleを未実装へ戻さず、空API/flagを先行追加しない。
