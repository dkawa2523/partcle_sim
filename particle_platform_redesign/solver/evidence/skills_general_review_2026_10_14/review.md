# 汎用COMSOL比較skillsの最終レビュー

対象はリポジトリの4 skillsと直接参照される文書です。別チャンバー・放出条件・物理モデルでも、保存結果から可能な判断を行い、必要な情報だけを追加取得できるよう改訂しました。本レビューではsolver実装、比較器のコード、MPH、保存計算結果は変更していません。

## 責務と利用方法

| Skill | 所有する判断・成果物 |
|---|---|
| [inspect-comsol-model](../../../../.agents/skills/inspect-comsol-model/SKILL.md) | nativeの有効式・選択・solution・外部依存・初期状態・出力意味を証拠に結び付けて取得／再利用する |
| [prepare-comsol-field-case](../../../../.agents/skills/prepare-comsol-field-case/SKILL.md) | 式と適用域を対応付け、geometry・field・joint source・境界をchecked caseへ変換し、独立した数値計画で必要範囲を実行する |
| [validate-comsol-trajectories](../../../../.agents/skills/validate-comsol-trajectories/SKILL.md) | 軌道と目的別の成果指標を評価し、確認済み範囲・不確かさ・原因所有者・妥当な改良を判断する |
| [visualize-particle-trajectories](../../../../.agents/skills/visualize-particle-trajectories/SKILL.md) | 保存証拠から全粒子左右図・詳細軌道・population図を作り、表示上の限界を明示する |

共通の新しい[用途別評価ガイド](../../../../.agents/skills/validate-comsol-trajectories/references/agreement-and-observables.md)は、空間的な見た目、用途上の観測量一致、数値的同等性、物理妥当性を分けます。厳密parityが未確認でも、保存された位置・配送率等について可能な限定結論を出せます。ケースA/Pの粒子数、時刻、境界ID、寸法、許容差を新モデルのdefaultへ移していません。

## 指摘と対応

| 指摘 | 最終対応 |
|---|---|
| 「ほぼ合う」の用途別判断がなく、NOT_TESTEDが評価全体の否定に見える | 4種類のclaimと、目的・SI尺度・母集団・coverage・許容差に基づく判断を分離 |
| 全粒子の比較依頼でもoverlay／個粒子galleryを優先する | 全軌道比較は左solver・右referenceを主図にし、同じ軸範囲・aspect・期間・線設定を使用 |
| 旧V&V文書が全比較へ完全一致・前段全合格・固定アンカー数を要求する | qualifiedな同一問題認定と、保存資料／類似モデルの実用評価・局所probeを区別して権威文書を整合 |
| ファイルhashや異なるmeshのcell/facet IDの差を入力不具合とする | 各artifact自身のreceiptを照合し、別meshは物理位置・semantic region/surfaceと検証済mapで評価 |
| 式の同等性と数値収束が同じstatusに混ざる | semantic correspondenceと独立数値分解能を分離。解析参照／exact scopeの精度確認を人工的な三runに置き換えない |
| 同じ数値IDだけで別MPHからmass等を継承できる | 同一のbound cohortと全継承authorityの確認を要求し、別populationは必要物性を明示取得 |
| moving wall全般を未対応と書いている | 静的geometry上の対応済み接線壁速度と、法線運動・moving geometry等の未対応を区別 |
| 非一様時刻の点数、survivor平均、原因不明消失を誤集計し得る | row-RMSとtime-RMS、計算数と物理重み、到達と捕獲、first-hitと反復hit、active residenceとholdを区別。unknownの率は上下限を示す |
| 不確かさを許容差に足すほど合格しやすくなる | 目的許容差と比較不確かさを分離。確認的同等性は妥当な差の区間が固定許容域へ含まれるか評価 |
| position-only／weighted／unpaired等を現行CLIへ押し込める | 実際のtool要件と対象外を明記。未知v/Zを捏造せず、必要量に絞った外部集計を使用 |
| 省略CSV行や保存点間の壁横切りを連続軌道・core不具合と誤認する | expected clockへの明示missing正規化／gap分割、保存点polylineの限界、event確認を明記 |
| 描画READMEが旧RZ-only・CSV-only能力のまま | 現行v5のXY/RZ・optional canonical geometry・ID alignment PASSの限界へ更新 |

## 独立した適用確認

初期レビューは取得／準備担当と評価／可視化担当の2エージェント、適用確認は前の結論を渡していない別エージェントで実施しました。[入力](forward_inputs.json)は実COMSOL計算の認証資料ではなく、技能の判断を試す6種類の合成依頼です。

| 合成依頼 | 独立エージェントの判断 |
|---|---|
| 位置だけの資料、非一様時刻、最大差5 mm以内の用途 | 全6行の最大差3 mm、等保存行RMS 2.082 mm。保存位置の用途基準を満たす。未保存の連続軌道・v/Zを認定せず、mandatory v/Zの比較CLIを使わない |
| 空間軌道は同じ、配送時間差0.1 s以内の用途 | 到達率は両側12/12だが時間差1 sで用途基準外。初期速度とforce設定を先に確認し、core原因を断定しない |
| 独立Brownian標本、重み付き配送差、固定margin ±0.05 | 点差+0.04でも差の区間[-0.02,+0.09]が許容域外へ跨ぐため同等性は不十分。unpaired pathや等重み二項式を使わない |
| 別MPHに同じID、energy-direction初期条件 | 新mass等を取得するまで数値速度を作らず、template物性を継承しない |
| 省略観測行の線が壁を横切る、接線壁速度あり | gap／保存点／geometry近似を先に確認。静的壁の対応済み接線応答を無効化せず、wall core欠陥を断定しない |
| 100粒子中referenceの20粒子が原因不明Disappear | 既知捕獲60でreference率[0.60,0.80]、candidate 0.60。差は最大0.20まで不明で、margin0.05の一致を立証できない。未知20を除外しない |

この適用確認は推論・手順の妥当性を検査しており、別モデルの実計算精度や現行toolの未対応機能を認定するものではありません。

2担当による改訂後の最終レビューで、対象範囲に未解決のblocking findingがないことを確認しました。最後の表記指摘も、Gate一覧のraw ID／接続一致を検証済変換とsemantic対応へ限定して反映しています。

## 検査と根拠

- skill-creatorの`quick_validate.py`：4 skills合格。Windowsの既定cp932を避け、`uv run --locked python -X utf8`で検査。
- 全skillsのローカル文書参照41件とUI metadata：合格。[検査記録](link_and_metadata_check.json)
- 編集対象の`git diff --check`：合格。構造検査と意味レビューは別に行いました。
- 改訂前[hash記録](before.json)と改訂後receiptで、利用するskillsのbytesを確認可能。
- 数値刻みと出力時刻は[COMSOL公式説明](https://www.comsol.com/support/knowledgebase/1254)、壁状態は[COMSOL境界理論](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.38.html)、出力集計は[COMSOL accumulator](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_math.06.15.html)と照合。
- intended useと認定の区別は[NASAのM&S標準](https://standards.nasa.gov/standard/NASA/NASA-STD-7009)、収束／漸近範囲は[NASAの収束評価](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html)、打ち切りは[NISTの説明](https://itl.nist.gov/div898/handbook/apr/section1/apr131.htm)を参照。これらの機関固有の承認手順を本skillsの要件にはしていません。

改良の判断は、失敗した用途基準→最初の未説明差→証拠に基づく所有者→独立verification→固定基準でのbefore/after比較、の順に具体化します。COMSOLとの図上の重なりだけを目的とした物性調整や本体変更を促す指示はありません。
