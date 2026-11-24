# 採用した拡張と未完の作業

本書は現在残る実装・評価を、その着手条件と完了証拠へ対応付けます。
完了済み工程の再実装手順や、過去レビューの不具合一覧ではありません。
現在の提供範囲は[製品仕様](product_specification.md)、変更が通るmoduleと受渡しは[architecture](architecture_proposal.md)、
作業手順と実行確認は[quality tooling](quality_tooling_plan.md)を参照します。

## 採用済みの拡張方針・未実装

次の方針を保持します。現行の対応API/設定が存在するという意味ではなく、詳細modelと検証条件を閉じてから実装します。

| 拡張 | 現在あるもの／不足 | 実装時に閉じる範囲と証拠 |
|---|---|---|
| G5/P17：RZ場を参照するXYZ粒子 | 現行はRZ meridionalの2自由度。方位運動・3D Brownianは未接続 | 3成分state、RZ→XYZ基底、axis、回転面へのfirst hit、3D壁law、source、dense/event/checkpoint/result、容量。回転共変、解析軌道・面hit、独立確率分布 |
| G5/P17：完全3D mesh/場 | 現行geometryは2D volume/line2。RZ場のXYZ参照とは別 | 体積cellと面境界、3D locate/support、vector basis、球と面/辺/頂点の接触、3D源と保存。独立要素補間・surface hit・収束・資源確認 |
| G3：非線形gas interaction輸送 | 有限速度の平均dragとcollision quadrature/jump/frozen oracleは実装済み。非線形noise/SDEを含む高drift確率輸送は未接続 | 抵抗と拡散tensorの仮定・保存則・漸近、実stage/state依存、壁後の確率path、対応積分。独立collision referenceと時間/path収束 |
| G3：粒子熱状態 | 背景温度fieldと既存drag/noise/熱泳動は実装済み。粒子温度の動的stateは未接続 | 熱収支・熱適応・drag/noiseとの温度authority、初期値、rate/bounds、運動との連成、保存/再開。熱的極限と独立reference |

3Dと時間変化の[設計参考](reviews/3d_transient_trajectory_extensions/README.md)は、2D定常を出発点とした差分説明です。
固定topologyの線形時間補間、実stageの場評価、連成charge、midpoint OU、durable出力/processは再実装対象に戻しません。
物理仮定・導出の採否は[technical research](technical_research.md)、採用済み式の適用域は[physics models](solver/docs/physics_models.md)が所有します。

## 条件付きbacklog

| 項目 | 着手条件 | 必要な変更と完了証拠 |
|---|---|---|
| G1/G2：追加の外部入力・field表現 | 現行のselected nodal域、side、static/linear-time、layoutでは必要な入力を表せない | realized相関、物性、frame、identity、supportを保持。高次・不連続・多領域ごとの独立転送検証 |
| 周期・不連続・逐次時間場 | 周期位相、切替、全snapshot常駐を超える入力が必要 | 入力時計・片側値・knot/phase・保持区間・resume。補間誤差と粒子dtを別に検証。moving mesh/feedbackは暗黙追加しない |
| 全材料・浮遊導体の独立電場 | gas-only縮約で対象問題を表せず、全材料のV/Q解が必要 | 材料epsilon、電源V/浮遊Q、界面、個別Q/global balance、gas側回復とcharge/ion/DEP依存再生成。独立材料/容量解とmesh評価 |
| surface charging・再飛散 | 表面stateや捕集後の再放出を用途が要求する | 壁の保存則、lifecycle、release authority、checkpoint。terminalを暗黙再活性化しない |
| G6：複数macroをまとめる同期 | 同一用途のprofileで毎macro IPCが支配的 | dt/knots/release/RNG、毎macro集計、観測、bounded ACK、accepted checkpoint/resumeを維持。途中死、科学identity、時間/memoryを比較 |
| NUMA・GPU・out-of-core | 実host、粒子/場規模、accuracy/SLAが必要性を示す | 単一経路と同じmodel/RNG/event/result意味。転送・serializationを含む容量、end-to-end費用、独立正しさ確認 |

必要性が未確定な設定名、stub、別engine、自動tuner、互換aliasを先行追加しません。
既存の独立collision oracle、解析解、native観測を、productionへ接続されていないことだけを理由に削除しません。

## 実装不足と区別する評価上の残件

- **用途精度**：deterministic、field/time、event、Brownian path、reference精度を別々に確認します。
  nominal dtや全区間件数だけで選択window・accepted pathの収束を認定しません。controller追加は必要性と費用を測定して決めます。
- **native parity**：同じ問題・入力・観測意味と独立resolution/ensembleが必要です。保存A/Pの実用的対応を全方程式の数値資格へ広げません。
- **装置予測**：物理的なsource rate/weight、capture/reemission則、実験比較が必要です。計算粒子数や数値stickだけでは付着fluxを認定しません。

これらは全機能の利用を止める一括gateではありません。要求する用途と観測量を選び、不足する証拠を明示します。

## 現行実装と過去工程の追跡

0.5.0の全領域改良・実行・独立レビューは[foundation記録](solver/evidence/foundation_2026_11_23/README.md)が所有します。
[初期撤去](solver/evidence/retirement_2026_11_23/README.md)、[意味・境界の再編](solver/evidence/resolved_semantics_2026_11_23/README.md)、
[追加撤去](solver/evidence/current_only_2026_11_23/README.md)は各source段階の履歴です。
過去P00〜P22/G1〜G4/0.1〜0.3工程を現在の未実装一覧へ戻さず、修正済み指摘と有効な残件を区別します。
