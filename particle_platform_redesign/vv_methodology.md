# COMSOL比較・検証・原因特定方法

## 1. 目的と判断の分離

本書は外部参照との比較の意味、必要な証拠、原因の調べ方を定める。方程式と適用域は
`solver/docs/physics_models.md`、積分・event・確率pathは`solver/docs/numerics.md`、
実行入口と入力契約は[外部tool README](solver/tools/vv/comsol/README.md)が所有する。
特定campaignのstep、seed、boundary ID、粒径、許容値を別modelのdefaultにしない。

COMSOLも数値解である。次を独立に判断する。

| 判断 | 根拠と限界 |
|---|---|
| 空間的な輸送傾向 | 同じ母集団・時刻・geometry・軸で示した空間分布。見た目だけで到達時刻や停止先を認定しない |
| 用途に対する実用的対応 | 停止先・時間・occupancy・速度・電荷等の用途指標、coverage、差と不確かさ。全方程式の同等性を含まない |
| 数値的に資格付けされたsolver同等性 | 実際の問題の意味、独立した数値精度、使用する境界と全比較範囲を確認した結果 |
| 物理妥当性 | 源のrate/weight、装置・表面則、実験等の独立証拠。COMSOL一致から推定しない |

保存結果の実用評価は、内部stepや乱数実現の対応が不明でも実施できる。独立Brownian pathの
同一ID位置RMSを数値誤差とは扱わない。利用目的に適切に対応していれば、その範囲を肯定的に
説明し、局所差と未認定事項を併記する。資格未確認の主張だけを`NOT_TESTED`にする。

`PASS/FAIL`は記載した判定に限定する。`NOT_APPLICABLE`は作用が本当に非適用の場合、
`NOT_TESTED`は証拠・実行・精度が不足する場合に使う。入力準備や実行の成功は数値的合格ではない。
すでに観測した結果から選んだ許容値は探索的であり、事前登録した確認試験へ読み替えない。

## 2. 必要な範囲を選ぶ

| 依頼 | 完了する作業 |
|---|---|
| 比較計画 | 用途指標、比較可能な意味、独立精度計画と不足する証拠を整理 |
| 保存結果の評価 | 実際の保存観測、全母集団とcoverage、差・条件付け・制限を記述。新規native solveを前提にしない |
| 原因調査 | 最初に違う層を小さい識別実験で特定。支持された原因か、次の必要観測まで進める |
| 資格付き比較 | 同じ問題、事前budget、両側の独立数値精度、必要な全指標を実行・評価 |

一つの保存例、最終個数、一部の成功、別caseの過去PASSで全体を認定しない。
新規科学計算を依頼された場合は、可能な必要実行まで進め、計画だけで終えない。

## 3. 比較する問題の同定

### 3.1 入力とauthority

一つの比較記録に次をbindし、既存の抽出・準備記録があれば再利用する。

- 原MPHの前後hash、COMSOL版、component/geometry、solution/dataset/saved member、保存時刻と単位。
  現在のparameter値と過去の保存tupleを混同しない。元MPHは再build/solve/saveしない。
- 候補のcase/data/content hash、実行sourceと環境、command、completed resultの実revision。
  ファイル名やHEADだけではdirty実行版を同定しない。
- coordinate/DOF、RZ基底、geometry/field selection、domainとboundaryのindex規則、field support。
- stable粒子ID、release、joint realized x/v/Z、独立したmass・drag径・電気半径・排除体積・contact。
  同じ分布から再samplingしたものを同一realized初期条件とはしない。
- 各作用の実式・primitive・gradient・material、charge feedback、wall/axis、時間依存の意味。
  feature名だけで同じ式とは判定しない。

coreはCOMSOLをimportしない。通常入力はcanonical writer/installed client、計算は
`load_case -> check_case -> simulate -> open_result`を通す。`check_case`は実行と同じprepare
であるが粒子を進めない。COMSOL固有の抽出、式の係数生成、参照正規化は外部ownerが担当する。

### 3.2 比較intent

- **Matched full case**：方程式、primitive、source、contact/wallを対応付けた同じ問題。
- **Matched slice**：両側で同じ作用を実際に使う限定粒子/domain/time/physics。
  有効forceを保存軌道から引いても、除外後の参照軌道にはならない。
- **Common-model companion**：元MPHを保全した別copyで共通modelを選び、readbackしてsolveする。
  元MPHとの差を残し、companion一致を元問題の認定へ拡張しない。
- **Original versus analogous**：近似・DOF・field/noise等の差と用途を明示する。
  記述的差やmodel感度を評価できるが、未対応の式へsolver-parity PASS/FAILを付けない。

新規の資格付き比較は`meaning_preflight.py`とsemantic inventory format 2を使う。
`same_canonical_field_solver_parity`は同じcanonical fieldを実際に使う問題、
`native_fe_end_to_end_reproduction`はnative FEからのfield/geometry転送差を含む問題である。
SUPPORTEDは準備の意味が揃ったという条件で、数値精度の合格ではない。
保存データの実用評価をこの資格gateで不要に停止させない。

## 4. 初期値から観測量まで確認する

| 層 | 必要な確認 | 差が出た場合のowner |
|---|---|---|
| 初期state | 全ID/release/x/v/Z/mのjoin、containment、surface facet ownership | 抽出/入力producer |
| 場 | 同じ点・時刻のsupport、SI、基底、primitive/gradient。native FE→P1 off-node転送差も測る | adapter/field producer |
| 個別右辺 | 同じx/v/Z/tで各force、charge G/J、合力閉合。Brownianは係数を比べる | physicsまたは係数producer |
| 積分 | 解析解、manufactured case、独立resolution。chargeと運動の同stage連成 | integrator/charge |
| event/壁 | earliest path hit、同時facet、hit x/v/Z、law、残時間、lifecycle。軸はmaterial壁でない | geometry/events/boundaries |
| 保存・集計 | accepted path上の観測、時刻別status、terminalと欠損の区別、重複停止の除外 | output/外部評価 |

1-step同刻み比較は同じ式・field・RNG割当が確認された専用局所診断でのみ行う。
候補のproduction stepをCOMSOL保存間隔や内部刻みに合わせない。候補に必要な刻みは自身の
accuracy/stability/applicabilityから選ぶ。出力時刻を得るために計算stepを切らない。

native中間bounceやfirst-hit ledgerが未取得なら、保存点間chordの壁横断から壁bugを断定しない。
Disappearや欠損suffixをoutlet escapeと同義にせず、最後の有限座標を実hit位置としない。
Freezeは数値terminal outcomeであり、実機adhesionの証拠ではない。

## 5. 指標と不確かさ

### 5.1 決定論的軌道

初期条件を対応付けた同一粒子について、要求した観測時刻のx/v/Z、lifecycle、supportと
必要なeventを比較する。Euclidean vector差と成分差、RMS/max、ID/time別coverageを区別する。
欠損・非finite・terminal行を分母から黙って除かない。各自のsurvivor中央値は条件付けが異なる。

`evaluate_matched_trajectory.py`はXY/RZの2成分、対応ID、同じ明示clock、
一つのrelease sliceのpre-event x/v/Zを扱う。CSV/public resultの実descriptorを使う。
位置だけ・異clock・unpaired ensemble・general staggered releaseをこの入口へ押し込まない。
`compare-pair`は実在する二つの保存結果で記述的に比較でき、架空の第三resolutionを要求しない。
資格付き比較はgenuine coarse/medium/fine、別々のself-sensitivity budgetとcross budgetを使う。
細かい二解の差だけで絶対誤差を保証しない。
親macro解像度はknown `macro_grid_revision`、actual releaseと対象時間窓をbindし、
`time_grid.MacroGrid`が復元する実分割で確認する。全区間のstep件数増加やnominal dt比だけでは確認しない。
field knot等で非一様になる分割では、nominal dt比をobserved orderの根拠にしない。
macro分割だけではevent/safety再積分後の粒子区間を認定しない。現行比較器はreview済みengine/path policyと、
既存の全run refinement/depth・axis・residual split・wall/periodic event・failure件数が全0という十分条件を
別の`fixed_root_execution`前提へ用いる。全runの正の件数を対象cohort/windowへ帰属させず、前提を未確認にする。
同じ受理区間へ飽和した複数macro設定の微小差を独立resolutionと読まない。自己差の資格と次数解釈には両前提が
必要であり、不足時もraw差・empirical target verdict・実用比較は保持する。一般のaccepted pathや確率strong収束の
認定、絶対誤差保証を追加するものではない。

### 5.2 確率輸送と停止

乱数レベルの一致を要求せず、目的に合う複数seedの集団を比較する。

- 同じ時刻・領域・sourceの空間分布とoccupancy。
- 停止先別count/proportion、停止・到達時間の分位/曲線、active-time積分。
- active speed/chargeの分布。各自のsurvivor条件付けと小集団の不確かさ。
- escape/failed/held/不明消失、coverageと打切り。

初到達、全wall interactions、terminal capture、保存height-window occupancyは別量。
停止座標が各frameへ再掲されても一粒子をterminal一回として数える。最終statusを過去frameへ流用しない。
computational particle重み1のcountを実flux、濃度、deposit massへ換算しない。
source rate/weightがあっても測定対象とsurface capture modelを確認する。

独立粒子の不確かさだけでなくseed間変動、同じsource/fieldに条件付けた差、native replica間の独立性を
明示する。複数の指標・時刻を確認的に判定するならfamilyとmultiplicityを事前に決める。
observed min/max帯内や非棄却は同等性の証明ではない。用途の許容値がない場合は差を数値化して
暫定的な実用判断を述べ、結果に合わせた合格基準を作らない。

### 5.3 数値精度と場

particle dt、field mesh/回復、Brownian base/adaptive depth、native reference精度を別々に評価する。
Brownian有限depth Hermite pathを連続OUのexact first-passageとは主張しない。
同じ親乱数の条件付き分割が必要な比較と、独立ensemble比較を区別する。
dtを縮めてもfield転送・closure・geometryの誤差床は消えない。

field比較は同じsample点とmeasureを使い、RZ体積なら`2*pi*r*dA`を明示する。
電場vector相対L2、成分RMS、potential RMS、局所最大差を混ぜない。全領域平均が近くても
境界付近の局所差や粒子感度が小さいとは限らない。電場変更時はcharge/ion/DEP等の依存量も
同じ新fieldから再生成し、古い係数を残した混合caseを避ける。

## 6. 原因調査

1. source/時刻/単位/member/DOFを確認する。
2. 初期state→同じ点のfield/support→個別force/G/J→1 step→first hit/応答/残時間→集団の順で、
   最初に説明できない差を探す。
3. 最大三つ程度の識別仮説を立て、小さい独立probeを実行する。複数作用を同時に変えた改善を
   一作用の寄与率とは呼ばない。保存full-force参照をmatched ablationの代用にしない。
4. 支持された原因、除外した原因、残る不確かさ、次に必要な一つの観測を既存比較記録へ残す。
5. 入力/場の差はそのproducer、式はphysics、交差はgeometry/events、応答はboundaries、
   集計は外部evaluatorで直す。見た目を合わせるためcoreへcase分岐やhidden係数を足さない。

native内部step/RNG/未保存eventが不明なら、その範囲を明記して用途指標の結論を返す。
すべての残差をBrownian乱数差へ帰属させない。closure、field transfer、event、
reference不確かさやmodel-form差も独立に残る。

## 7. 実行と記録

比較ごとに既存の一つのmanifest/summaryへ、目的・intent、固定/変更条件、
入力とreferenceのidentity、実command/環境/完了status、指標と分母、許容値の根拠、
不確かさ・coverage・未認定範囲を記録する。raw tableや既存inventoryを重複複製しない。

benchmarkの実行code、固定input、独立native/解析参照、評価方法、生成結果を分ける。
old result形式をrelabellingして読まず、必要な科学データは当時の環境から明示exportするか現行で再実行する。
古い実装のhashを固定するtestを現行の数値検証とはしない。
現行public APIを実際に動かし、独立参照と比較した新しい結果だけを現行の説明へ使う。

現行の確認記録は[撤去・再実行記録](solver/evidence/retirement_2026_11_23/vv/README.md)、
以前の実用比較と因果の証拠は
[保存比較の科学記録](solver/evidence/platform_restructure_2026_11_23/science/README.md)を参照する。
過去の実験記録はそのsourceと範囲の証拠であり、現行sourceの自動的な再認定ではない。
