# 参照COMSOLと粒子輸送結果の比較と差の原因

作成日 2026年11月22日

**今回のA/P各287粒子の比較では、大局的な粒子輸送、総停止率、停止までの代表時間、運動粒子数の時間変化は適切に合っている。** Brownian運動の同一ID軌道や乱数値を一致させる必要はない。候補を各40回実行して比較すると、参照COMSOLの結果は、これらの集団観測量について候補の実現間変動から見て異常な外れ方をしていない。

一方、**境界33への停止割合は候補平均が約4〜5ポイント低く、100 µsで運動中の粒子の速度中央値は約13〜18%高い。** この面への精密な捕集量予測や、全時刻の速度分布まで一致したとの判断は留保する。大きな物理式・帯電・境界の取り違えを支持する証拠はなく、場転送と離散確率過程の差は確認できたが、局所的な差への寄与率は未分離である。

## 比較した問題と実行条件

対象は `caseA_10nm_vigus_equations_unboosted.mph` と `caseP_10nm_vigus_equations_unboosted.mph` の保存粒子解 `partVigusLaw -> sol37`。各287粒子、粒径10 nm、RZの2自由度、初期電荷−1 e、全粒子の放出時刻0 sであり、初期位置・速度・電荷・ID・質量等を候補へそのまま対応付けた。初速度の増幅は行っていない。提示された過去のVigusレポートにある30 nm・43,220粒子・1.25 msとは別の問題である。

| 項目 | 今回の条件 |
|---|---|
| 参照 | 各caseの保存COMSOL 1実現、0〜30 ms、121時刻 |
| 候補 | 各caseのseed 21〜60、40実現。初期粒子と場・物理・壁条件を固定し、乱数seedのみ変更 |
| 候補の場 | native場を細分化したP1入力。7,752節点、15,116三角形 |
| 候補の物理 | 8種類の力と動的帯電。Coulombだけ電荷を±17 eで制限し、帯電式とion dragには実電荷を使用 |
| Brownian対応 | `fixed_rk4_white_limit`。native固定RK4の乱数共有から導出した白色弱極限を明示して使用 |
| 候補の数値条件 | macro幅1 µs、Brownian base depth 2、adaptive max depth 5 |
| 比較期間 | 共通の0〜5 ms・96保存時刻。既存4実現の30 ms結果を再利用し、新規36実現は5 msまで計算 |
| 完走とcoverage | 全80実行で粒子failure 0。全粒子が5 msより前にterminalとなり、時間平均に右打切りはない |

停止は数値的な保持停止を意味し、実機での接着を認定するものではない。全て重み1のcomputational particleであり、物理的なfluxや付着質量ではない。COMSOLの消失A36個/P33個は原因不明のまま扱い、候補の明示的escapeと物理的に同一視していない。

入力同定とsourceの根拠は[前段のnative調査](../../.artifacts/case_ap_vigus_repair_20261122/native_report_section.md)、実行集合は[Case A](../../.artifacts/vigus_population_assessment_20261122/ensemble/caseA/results.json)と[Case P](../../.artifacts/vigus_population_assessment_20261122/ensemble/caseP/results.json)にある。今回の比較作業でsolver coreの式や係数を変更していない。

## 総停止率と時間変化の対応

候補平均は40実現の平均。時間の±は実現間の標準偏差であり、平均の信頼区間ではない。

| 観測量 | A COMSOL | A 候補 | P COMSOL | P 候補 |
|---|---:|---:|---:|---:|
| 最終保持停止数 | 251 | 平均249.675 | 254 | 平均251.775 |
| 最終保持停止率 | 87.456% | 86.995% | 88.502% | 87.726% |
| 停止率差 候補−参照 | — | −0.462ポイント | — | −0.775ポイント |
| 全粒子の平均terminal時間 [µs] | 200.397 | 203.489 ± 10.751 | 187.383 | 203.055 ± 10.832 |
| 平均terminal時間の相対差 | — | +1.54% | — | +8.36% |
| 保持停止した粒子のterminal時間中央値 [µs] | 153.670 | 平均150.717 | 142.311 | 平均147.672 |
| active割合の時間曲線 参照対候補平均の最大差 | — | 4.52ポイント | — | 4.23ポイント |
| 累積保持停止割合の時間曲線 最大差 | — | 2.99ポイント | — | 5.10ポイント |

総停止率の差は両caseとも1ポイント未満で、停止・輸送の時間スケールも近い。Pの平均terminal時間には約8.4%の差があるが、候補40実現の範囲178.605〜220.079 µsに対して参照は187.383 µsであり、短い側の通常変動と両立する。参照1件と候補40件を合わせた順位はAが17番目、Pが5番目だった。以前の4実現だけで見えたPの時間差を、系統的不具合と断定する根拠は弱まった。

時間曲線を一つの関数として評価した順位比較でも、active曲線、累積保持停止曲線、停止先全体の構成に特異な乖離は見られなかった。停止先構成は、保持停止面・非保持terminal・activeという粗い観測カテゴリであり、保持停止した粒子だけの条件付き分布ではない。探索的なglobal rank確率はactive曲線でA 0.488/P 0.610、停止先全体でA 0.707/P 0.415。停止先構成のtotal variation距離はA 6.56%/P 7.16%だった。これは単一粒子の位置差や最終個数だけに依存した判断ではない。[全数値と統計手順](../../.artifacts/vigus_population_assessment_20261122/population_predictive40/README.md)

![COMSOLと40実現の運動粒子数とterminal時間と境界33停止数](../../.artifacts/vigus_population_assessment_20261122/population_summary.png)

図の青い帯は40実現の経験的な最小〜最大であり、同時信頼帯ではない。時間曲線の表示範囲は0〜1.5 ms、terminal平均の集計期間は5 ms。各ヒストグラムの一つの標本は287粒子全体の1実現である。

## 空間分布と局所的に残る差

全active粒子を同じgeometry・時刻・軸尺度で比較すると、チャンバー内への広がりと主要な輸送傾向は対応する。既存4実現の補助比較では、10 µsの位置・速度・電荷の分布距離は候補実現同士の距離と同程度だった。100 µsの20 mm格子によるactive位置とnonactive割合の距離も、参照対候補と候補同士で近い。ただしこの粗い格子は、狭いgapやmm規模の着地点精度を保証しない。[空間分布の定義と数値](../../.artifacts/vigus_population_assessment_20261122/spatial_findings.md)

![同時刻の全active粒子群](../../.artifacts/vigus_population_assessment_20261122/spatial_clouds.png)

上図の候補はseed 21の例示。集団判断は上記40実現の結果と区別している。

| 留保する観測量 | A | P | 判断 |
|---|---|---|---|
| 境界33停止数 参照／候補平均 | 74／60.450 | 73／61.325 | 全287粒子に対して−4.72／−4.07ポイント |
| 境界33停止数 候補40実現の範囲 | 45〜72 | 46〜74 | Aの参照は候補の最大より2個多い。局所捕集量には留保が必要 |
| 100 µs active速度中央値 参照／候補の中央値平均 [m/s] | 759.08／861.32 | 737.51／866.92 | 候補が約13.5%／17.5%高い |
| 100 µs active電荷中央値 参照／候補の中央値平均 [e] | −7.015／−7.071 | −9.354／−8.479 | Pは候補が負帯電の小さい側 |

速度・電荷は各々の時点で残っている別々の運動粒子群に条件付けた量であり、同じ粒子の誤差ではない。100 µsの速度中央値は参照が候補40実現を下回ったため、その差は明記する。一方、10/20/50 µsの加速スケールは近く、200/500 µsは実現間変動と両立する。

境界33や100 µsは以前の結果を見て注目した探索的な観測である。A/Pも共通に近いnative乱数実現を持つため、両方の同方向差を二つの独立な不具合証拠として数えない。これらの差は、全体輸送の肯定を取り消す理由にはならないが、全表面・全時刻の精密一致を主張することもできない。

## 差の原因について確認したこと

原因調査は、初期状態、同じ点の場と物理右辺、境界の意味、確率過程、時間分解能の順で行った。初期点だけでなく、COMSOL粒子が実際に到達した10/50/100 µsのactive状態も評価した。A 701行/P 670行の同じ位置・速度・電荷を使ったため、異なるBrownian軌道の差を物理式の差と混同していない。

| 原因候補 | 確認結果 | 現在の判断 |
|---|---|---|
| 初期状態や粒子物性の取り違え | 全287 IDの実現位置・速度・電荷・release・物性を保持 | 今回の差を説明する証拠はない |
| 現在のP設定と保存解の不一致 | 元MPHのcopyで粒子解を30 ms再実行し、19変数×34,727行がfloat64のbitで一致 | Pでは保存解を取り違えた説明を支持しない |
| 力・帯電式の欠落や大きな結線誤り | native primitiveから独立に計算したnon-Brownian合成加速度は、実訪問stateのnative右辺へ相対L2約6e−15〜3.2e−14、帯電速度へ約8e−16〜2e−15で一致 | 初期だけでなく後時刻でも、大きな式・符号・質量除算の差は支持されない |
| samplerやruntimeの誤実装 | 独立P1補間との相対差は約1e−16、sample後の加速度・帯電速度・Jacobianは約9e−16以下 | 同じcanonical場を読む経路の誤りは支持されない |
| native FEからP1への場転送 | 細分化後も実訪問stateの合成加速度差はA約0.56〜1.25%、P約1.51〜4.83%。100 µsではA 1.25%/P 1.51% | **実在する近似差**。電場・ion係数等に局所差が残るが、停止配分差の寄与率は未分離 |
| Brownianの離散確率過程 | native固定RK4は同じ時刻の乱数をstage間・隣接step間で共有。定数係数20,000粒子の独立microcaseでも連続OUとの共分散差を確認 | **実在するモデル対応差**。今回の白色弱極限でもfinite-step・壁到達則までは同一でない |
| 境界33の形状・法線・selection・接触・反射則 | nativeの水平面とcanonical facetsが対応。point contact、確率0.5、速さ保存の内向き半円一様角の分布が対応 | この面の取り違えを支持しない。途中の全hitや有限確率pathとの相互作用まで同等とは認定しない |
| macro幅の粗さだけが原因 | 保存済み1/0.5/0.25 µsの各4実現で、100 µs速度差や平均terminal差は単調には減らない | 刻みを縮めれば全て解消する説明は支持されない。確率pathが変わる4実現なので収束認定でもない |
| 有限の粒子数と乱数実現 | 各40実現では総停止率・時間曲線・停止先全体が参照と両立 | 大局的な差は通常の実現変動と整合。局所差の全てを乱数だけに帰属しない |

場転送の個別項では、100 µsのP ion dragに約18.4%の相対差があるが、これはion項だけの値である。同じ点の合成加速度差は1.51%であり、全加速度が18%異なるとは読めない。raw primitiveを使えばnative右辺へ丸め誤差水準で戻るため、ここでは場表現の近似とcoreの式を分けて考える。[実訪問stateの全項・帯電成分・単位付き診断](../../.artifacts/vigus_population_assessment_20261122/native_visited/visited_transfer_analysis.md)

Brownianの白色弱極限は停止数へ合わせて倍率を選んだものではない。nativeの固定step幅とnoise正規化幅がともに1 µsであることを確認し、RK4重みとendpoint乱数共有から長時間の有効force拡散を `D_eff = 5D/9` と導いた。元の連続white-force係数DをそのままOUへ入れる問題とは区別した。この導出は遠方壁・定数係数の共分散を説明するが、位置依存係数や壁first-passageの完全一致までは保証しない。[独立Brownian検証](../../.artifacts/case_ap_vigus_repair_20261122/native/microcase_covariance_analysis.json)

## 適切に合っていると判断する範囲

今回の要求である乱数一致を求めない比較では、**大局的な空間輸送、総停止率、代表terminal時間、時間曲線についてA/Pとも適切な対応と判断する。** 同じIDの曲線をさらに重ねるためのcore修正や、停止結果への係数fitを進める根拠はない。

判断には、全粒子と共通clockのcoverage、failure、分母・statusの意味、複数実現の変動、曲線全体と停止先全体の距離、局所差の大きさを併用した。一つの287粒子cohortを一つの実現とし、96時刻を独立標本に数えたり、A/Pを80独立実現としてpoolしたりしていない。順位比較は、参照と候補が同じ結果分布なら交換可能という仮説の下での探索である。物理的な許容差を結果から作ったり、統計的非棄却を厳密な同等性の証明としたりしない。

境界33の局所停止量を設計判断へ使う場合は、今回の4〜5ポイント差を扱える用途許容差を先に定める必要がある。さらに原因を分離する必要が生じた場合の次の小型比較は、同じ定数係数と平面壁によるnative固定RK4対OUのfirst-passage・反射後残時間であり、full caseのnoiseや壁確率の総当たり調整ではない。数値的な全方程式同等性、AとPの応答差の精度、実機の物理的付着予測は、この大局的対応とは別の認定範囲である。

## 証拠への入口

以下の詳細artifactはGit追跡外のローカル実行物であり、cloneだけでの取得を保証しない。

- [40実現の数値と順位と入力173件のSHA](../../.artifacts/vigus_population_assessment_20261122/population_predictive40/assessment.json)
- [40実現の比較詳細](../../.artifacts/vigus_population_assessment_20261122/population_predictive40/README.md)
- [原因候補の層別auditとA/Pの依存](../../.artifacts/vigus_population_assessment_20261122/cause_audit.md)
- [実訪問stateの局所診断JSON](../../.artifacts/vigus_population_assessment_20261122/native_visited/visited_transfer_analysis.json)
- [native read-only exportのartifact manifest](../../.artifacts/vigus_population_assessment_20261122/native_visited/artifact_manifest.json)
- [100 µsの保存刻み感度](../../.artifacts/vigus_population_assessment_20261122/100us_time_sensitivity.json)
- [実行完了とsource不変の記録 A](../../.artifacts/vigus_population_assessment_20261122/ensemble/caseA/completion.json)と[P](../../.artifacts/vigus_population_assessment_20261122/ensemble/caseP/completion.json)
- [図の入力と表示条件とSHA](../../.artifacts/vigus_population_assessment_20261122/population_summary_receipt.json)
- [前段の改良と独立検証](../vigus_transport_repair_2026_11_22/README.md)

今回のCOMSOL原MPHのSHA-256は、A `347061d82453944a64076644c8c83d8e88165b066f3c2470e49a408fd40a4b8c`、P `fc16c9eeee0a4b244d2e01d631cea7df928e51d33b29e77a6f511254f930491d`。実訪問state抽出でも元MPHを保存・改変せず、前後のhashを照合した。
