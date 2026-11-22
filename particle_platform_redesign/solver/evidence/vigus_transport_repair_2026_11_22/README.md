# Vigus 輸送の入力解釈と計算経路の修復

対象はユーザー指定のA/P 10 nmモデルである。提示された11月8日のレポートを変更仕様の候補として読み、現在のコードとnativeの保存内容へ対応付けた。30 nm・43,220粒子の過去評価を今回の正解値として移植していない。

元のHEADは`a4c71f104a3c9870cc72166beafded4af428eb3f`。改修は独立uv projectの単一engine、既存のstage評価・OU・event・writerへ接続した。`old_code`の環境・packageを使用せず、COMSOLは外部の抽出・比較に限定した。

## 調査で確定した課題と変更単位

| 優先度・層 | 確認した課題 | 修復・整理した責務 | 再発を検出する確認 |
|---|---|---|---|
| P0 保存問題の同定 | 古いlabelにinactive、Epstein、specular、初速10000倍等が残る。今回の実値と一致しない | active feature、Equation View、saved solution/member、実現粒子を同定。原MPHのSHAを前後照合 | 全287 ID・初期x/v/Z/release/物性、121時刻、全境界coverage |
| P0 物理modelの対応 | 標準Epstein/FDT/OML等を別のCMD・独立noise・帯電式へ黙って読み替えられない | 明示したreference revisionをcatalog→runtime→compiled stageへ接続。標準modelの適用域を保持 | 独立局所代数、G/J、有限charge invariant、OU full covariance、公開run |
| P0 粒子物性 | 一つの粒径から質量・電気半径・接触を再構成すると別問題になる | mass、drag径、電気半径、排除体積を別authorityとして全sourceへbind | 異なる物性の誤結線拒否、同じ物性subsetの正常利用 |
| P0 動的帯電 | 初期ZやCoulomb用capを全stateへ使うとcharge/ion feedbackが変わる | x/v/Zを同stage評価。今回の±17はCoulombだけに適用 | capを横切る連続charge＋二次ion forceの区分解析解 |
| P0 Brownianの意味 | 式のDを一致させても、native RK4の時刻key共有forceと連続OUは同じ過程でない | 原Dと固定RK4白色弱極限をproducerの必須選択として分離。後者は独立導出5D/9、別case | 20,000粒子native定数係数microcase、全x/v共分散、独立RK4重み合成 |
| P0 壁応答 | 鏡面・cosine・熱再放出は内向き半円一様角の散乱を代用しない | 速さ保存の`isotropic`を既存boundary ownerへ追加。stick drawとは独立 | 角度moment/分位、速さ、cornerの内向き性、p=0.2、残時間、workers/出力不変性 |
| P0 周辺呼出側 | inlet boundary37も無条件stickであり、wall1だけの設定では不足 | axis・wall・outlet・inletを別semantic groupへ対応 | 未割当境界のadapter拒否を維持し、全perimeterを一回ずつ対応 |
| P0 primitive取得 | 保存背景に新しいaliasがなく、式・unit・selectionを取り違えやすい | nativeで確認した式を展開し、node/元cell中心/source点をSI・ID付きで抽出 | 全41列finite、全点domain3、全座標・source ID・保存memberを検査 |
| P0 場の転送 | native高次tri/quadからnodal P1への変換には実測差がある | 原場を保全。全edge中点を新たにnative評価し、1回uniform subdivision | 同じsource287点・元cell中心2953点でbefore/after。ゼロ/最近傍補完なし |
| P0 field builder | 未指定外周や独立した無anchor成分があっても残差0の非一意解を成功にできた | 公開入力は全非axis外周を明示、各連結成分のanchorを必須化。数値APIの明示natural条件は保持 | 非一意な実再現例、欠落facet、独立成分、正常な複数anchor/axis case |
| P0 保存結果の意味 | escape等の正当な欠損suffixを全行存在要求で拒否すると比較できない | event ledgerの時刻以後だけlifecycleを復元し、x/v/Zは欠損のまま保持 | 通常/zero-time escape、genuine failure、未説明のpre-terminal gap拒否 |
| P0 集計 | 最終statusを過去へ流用、native消失をescape、停止をfirst-hitと扱うと結論が変わる | 時刻別status・terminal stop・欠損cause・母集団を分離する小型外部評価 | 全34,727 lifecycle slot、停止先別count/時間、active曲線、速度/電荷分布 |
| P1 可視化 | IDごとの全行再走査と全ID個別SVG生成が大規模例を支配 | grouped alignment、共通表示範囲、全粒子overview＋明示IDのみdetail | 同入力・同detailで数値/図同一、時間/RSS、保存順を反転したevent sample |
| P1 通常用途への影響 | 新reference入力の追加が標準modelの結果・実行費用へ影響し得る | 同じuv環境・同じ公開caseでHEAD baselineと変更後を測定 | 通常Epstein/FDT/Stokesの全科学payload bitwise比較とwarm時間/RSS |

追跡した主要な所有箇所は次のとおり。既存の積分・event経路を再利用した箇所を含む。

| 処理 | 実装と検査の所在 |
|---|---|
| model選択・required fields・stage | [catalog](../../src/chamber_particles/physics/catalog.py)、[runtime](../../src/chamber_particles/physics/runtime.py)、[compiled](../../src/chamber_particles/physics/compiled.py)、[charge bounds](../../src/chamber_particles/physics/charge.py) |
| x/v/Z・Brownian・hit・残時間 | [engine](../../src/chamber_particles/engine.py)、[integrators](../../src/chamber_particles/integrators.py)、[stochastic](../../src/chamber_particles/stochastic.py)、[boundaries](../../src/chamber_particles/boundaries.py) |
| 係数生成と入力対応 | [transport producer](../../tools/vv/comsol/prepare_vigus_transport.py)、[producer tests](../../tools/vv/comsol/tests/test_vigus_transport.py) |
| 独立fieldの問題設定 | [builder workflow](../../tools/electrostatic_builder/workflow.py)、[numerics](../../tools/electrostatic_builder/numerics.py)、[builder tests](../../tools/electrostatic_builder/tests/test_builder.py) |
| 保存・公開・時刻別集計 | [output](../../src/chamber_particles/output.py)、[publication tests](../../tests/verification/test_output_publication.py)、[population evaluator](../../tools/vv/comsol/evaluate_saved_population.py) |
| 表示と大規模データ処理 | [matched trajectory viewer](../../tools/vv/comsol/plot_matched_trajectories.py)、[軽量visualization](../../tools/visualization/__init__.py) |

## 今回のnativeと過去レポートの境界

今回の各caseは287粒子、10 nm、初期Z=-1、RZ 2自由度、0–30 ms・121保存時刻。実現初速は約0.0090–0.2478 m/sであり、過去のboostを引き継いでいない。現行設定とEquation Viewで8 force categoriesと動的帯電を確認した。電気力のcap±17は保存active行で実際に作用している。

30 nmレポートの初期Z=3、cap非active、1.25 ms、43,220粒子、全材料独立電場の4.009%等は今回の値ではない。またレポート10.2Aの`ion+lift`というラベルにはdragが混入している。提示定数でion+lift単独の定数項は`(6.5,10)`、dragを加えた合計が`(-2.5,16)`である。テストをその誤ラベルへ合わせなかった。

現在のcoreに追加したのは数学的reference modelであり、CMD/Brownianの物理的整合、高Kn Saffman、イオン電流の反発branch、実付着の妥当性を認定するものではない。

## Brownianの不一致が繰り返した理由

nativeは`F=sqrt(D/timestep)*randomnormal(...)`を古典RK4で評価する。`timestep`は内部刻み変数でなく固定parameter 1 µsだった。同じ物理時刻では同じ乱数となり、midpointの2段と隣接intervalのendpointが共有される。

摩擦ゼロの極限では、一stepの速度重みは`(1/6,2/3,1/6)`、速度分散は連続OUの1/2、位置分散は5/12。N stepの速度分散は`D*h/m²*(5*N/9-1/18)`で、hとnoise正規化刻みが等しい長時間の白色弱極限は5D/9となる。native定数係数実験の一step速度分散比は約0.502、100stepでは約0.562で、この機構を独立に支持した。

したがって、初期位置・Dの式・保存間隔だけを合わせても実効確率過程は合わない。nativeのhだけを変更して固定`timestep`分母を維持する実験も、noise強度を保った収束試験にはならない。今回のproducer v2は解釈を必須入力にし、5/9を輸送結果へfitする自由倍率にはしない。有限stepのjoint covariance、可変係数、壁first passage、native random pathまでの同値は主張しない。

## 場の転送で測定した改善

元1987節点・3779 P1三角形から、追加5765点のnative保存場評価により7752節点・15116三角形へ細分化した。native FE meshの再計算ではなく、同じ保存場の候補側representationの改善である。実現source・壁semantic ID・幾何体積は保持した。

| 同じ点での相対L2差 | A 元→細分化 | P 元→細分化 |
|---|---:|---:|
| 初期287点のE | 1.816%→1.049% | 8.918%→7.176% |
| 初期287点のcharge G | 9.056%→2.310% | 2.337%→0.552% |
| 初期287点のcharge J | 8.650%→2.207% | 2.506%→0.589% |
| 初期287点の合成決定論加速度 | 1.830%→1.058% | 9.022%→7.261% |
| 元2953 cell中心のE、RZ体積重み | 7.292%→1.545% | 10.658%→3.980% |

元cell中心1点ずつの最後の指標は収束済みの体積積分ではない。Pの初期点とgrad(E²)には大きな局所差が残る。DEPの百分率差が大きくても初期加速度の絶対差は小さく、百分率だけで主要原因にはしない。独立P1補間と既存samplerは約1e-16、独立式とruntimeは約1e-15の相対差で対応し、計算結線の欠陥と転送近似を分離した。

## 保存全粒子計算での到達点

原Dの連続white-force case、明示したRK4白色弱極限case、場を細分化したcaseを別に保存した。以下のfull-window集計は各287粒子・30 ms・121時刻を計算し、failureを集計から落としていない。

| 観測量 | native A | 原D A | native P | 原D P |
|---|---:|---:|---:|---:|
| 最終停止数 | 251 | 254 | 254 | 253 |
| その他terminal | 36消失、原因不明 | 33 escape | 33消失、原因不明 | 34 escape |
| capture粒子のterminal時刻中央値 [µs] | 153.670 | 118.746 | 142.311 | 117.433 |
| release→terminal平均時間 [µs] | 200.397 | 159.117 | 187.383 | 156.197 |

capture中央値の母集団は各参加者で数値的に停止した粒子。平均terminal時間は全287粒子で、nativeの原因不明消失も含む。個数・割合・ppはいずれも重み1のcomputational particle数に基づき、実sourceの物理weight/fluxを推定していない。

RK4弱極限caseのseed21–24では、Aの平均terminal時間は202.415–215.566 µs（seed平均208.547）、Pは206.779–213.278 µs（211.083）。nativeはそれぞれ200.397/187.383 µs。停止数だけの近さを成功判定にせず、Pに全4seedで同方向の時間差が残ることを記録した。4seedの範囲やSDは信頼区間でも同等性判定でもない。

同じ4seedで場だけを1回細分化した平均terminal時間はA208.790 µs、P211.679 µs。粗い場からの変化はそれぞれ+0.243/+0.596 µsで、今回の細分化だけではPの時間差を解消しなかった。これを「場の誤差は無関係」と一般化せず、短時間noise-off比較とnative保存設定の再実行で残差の層を調べた。

独立したnoise-off companionからnativeの合成F/mとcharge rateを直接取得した。初期287点でnative原始場を使った独立式との差は、Aが加速度2.57e-15・charge rate1.71e-15、Pが4.02e-15・2.41e-15の相対L2だった。この点ではnativeの実RHSと式の対応を確認できた。P1転送を含む差や保存stochastic解の過去RHSまで同一という証明とは分ける。

残差の大きいPでは、元のBrownian・clock・method・背景参照を復元したloadCopy上でparticle-only sol37を全30 ms再実行した。19量・全34,727行がNaN maskを含めて元保存解とbitwise一致した。今回Pの残差を「current設定とsaved解の取り違え」に帰属する証拠は得られなかった。これはnative再実行の再現性の証拠であり、candidateの数値的同等性の認定ではない。背景studyの再計算と原MPHのsaveはしていない。

さらにBrownianだけを外したnative 0–10 µs・11時刻と、候補の独立刻み0.25/0.125/0.0625 µsを比較した。candidate全runでfailure/境界eventは0。nativeは全粒子active、保存点の壁距離は少なくとも1 mmで、壁から十分離れた短時間の診断として使った。

| 最細刻みの全particle-time RMS差 | A 元→細分化 | P 元→細分化 |
|---|---:|---:|
| 位置 [µm] | 0.66850→0.40314 | 2.35845→1.88583 |
| 速度 [m/s] | 0.166786→0.101710 | 0.582446→0.465293 |
| 電荷数 [e] | 0.0209620→0.00533622 | 0.00104074→0.000237594 |

細分化した場での最後の候補刻み変更による自己差は、位置RMSがA5.86e-11/P2.80e-11 m、速度RMSがA2.61e-6/P9.90e-7 m/s。今回の短時間範囲では刻み差よりnativeとの転送差が大きく、dtだけを縮め続ける修復は支持されなかった。native側の自己刻み収束や壁後経路の認定はこの診断に含めない。

確率輸送も同じ細分化場・seed21–24でh=1/0.5/0.25 µsを点検した。seed平均terminal時間はA208.790/208.043/213.931 µs、P211.679/208.470/209.506 µsで、単調な収束やP残差の解消は観測しなかった。小さい2刻みは宣言した0–2 ms prefixで実行し、全16runの全287粒子が2 ms前にterminalとなったため、このterminal平均に打切りはない。全run failure0。変更したmacro root間の同seedを同じBrownian pathとは呼ばず、4seedから確率的同等性や収束次数を認定しない。

停止先差が大きいboundary33も追加点検した。nativeの20 edgesとcanonical parent20 facetsは同じz=22 mmの水平面で、細分化40 facetsは正確な二分割。向き・選択・point contact・同一phiによる速さ保存・反射0.5/otherwise Stickの対応に欠陥は見つからなかった。薄い曲線形状という仮説はこの境界の記録から支持されない。残る差を乱数だけに帰属したり、noise倍率を再fitして解消したことにはしない。

## 公開操作と再現入力

以下は`particle_platform_redesign/solver`から実行する。作成済みcaseの再実行にCOMSOLは不要。出力先は未使用directoryを選ぶ。

```powershell
uv sync --locked
uv run --locked python -m chamber_particles check .artifacts/case_ap_vigus_repair_20261122/caseA/refined1/transport_rk4_white_limit/case.yaml
uv run --locked python -m chamber_particles run .artifacts/case_ap_vigus_repair_20261122/caseA/refined1/transport_rk4_white_limit/case.yaml -o .artifacts/vigus_replay_A_new --workers 1
uv run --locked python -m chamber_particles inspect .artifacts/vigus_replay_A_new
uv run --locked python -m tools.vv.comsol.evaluate_saved_population --reference .artifacts/case_ap_vigus_repair_20261122/native/caseA_particles_long.csv --candidate .artifacts/vigus_replay_A_new --reference-terminal-times --output .artifacts/vigus_replay_A_new_population.json
```

Pは同じ対応の`caseP`へ置き換える。`refined1/transport/case.yaml`は原Dの連続white-force解釈、`refined1/transport_rk4_white_limit/case.yaml`はnative離散化に対応した近似入力である。全係数・source・hashは各caseとbuild manifestに残る。

係数の再生成は公開外部producerを使う。

```powershell
uv run --locked python -m tools.vv.comsol.prepare_vigus_transport PRIMITIVE_CASE NEW_TRANSPORT --parameters PARAMETERS_V2 --interval-tree-depth BASE --adaptive-max-depth MAX
```

入力field・parameter v2・モデルbinding・Brownian選択・限界は[外部tool説明](../../tools/vv/comsol/README.md)が所有する。raw MPH、native export、今回のcase/resultはGit追跡外のローカルartifactであり、通常cloneに含まれる配布物とは扱わない。

## 次の改良を実施する順序と完了の証拠

| 条件・順序 | 実施内容 | owner | 必要な完了証拠 |
|---|---|---|---|
| 今回の原因識別で完了 | 同じcurrent equationのBrownianなし短時間native companionと独立刻みcandidateを比較、Pの元設定全区間を再実行 | 外部native/V&V、既存physics/integrator | 初期join、壁から離れた範囲、実RHSとx/v/Z、候補自己刻み差、P保存値のbitwise再現。残差を単一作用へ推測で帰属しない |
| 場の誤差が判断を妨げる場合 | 追加native点評価とrepresentation細分化、source近傍・実通過域の力/G/J感度を測定 | field producer/adapter | 同じ点・norm・物理clockの差。粒子dtを縮めるだけで場の差を消したことにしない |
| 数値的同等性を認定する場合 | nativeとcandidateでnoiseの意味を同定し、両側の時間/path/field収束とensembleを分離 | prepare/validate skills、外部V&V | 事前の物理判断budget、実resolution・coverage・不確かさ。後から観測差へ許容値を合わせない |
| 独立電場も必要な場合 | 全電気topology、誘電体、電源V・浮遊Qを解き、ne/ni/charge/ion/DEP依存係数まで再生成 | electrostatic builderと外部case producer | 材料解析解、個別Q/全charge、gas側回復、同一点field比較。今回の取込み場計算では未達 |
| 実装置の付着量を予測する場合 | source rate/weight、capture/reemission則と実験を整備 | 外部物理model・実験評価 | 物理単位のflux/付着massと測定照合。数値stickだけをadhesion認定しない |

全材料独立場や3D/nonlinear collisionを今回の輸送修復の前提に加えていない。着手時もone engineと一つのauthorityを維持し、旧production経路、曖昧な設定、誤った期待値を同じ変更で置換する。

## 品質と独立点検

### Windowsでの完成結果の公開

計算終了後の`OUT.partial → OUT`というdirectory renameが、WinError 5で一時的に失敗する問題も実際に再現した。
HDF5のopen-file IDはsyscall直前に0、失敗直後の自processの対象file handleも0だった。既知のfile/HDF5 handleを
保持する対照では拒否とownerを検出し、解放後に成功した。一方、実失敗時の外部processは捕捉できず、AV等の原因は未確認である。
実失敗3件は、追加のfile openやDELETE probeを介さず25 ms待っただけで同じrenameが成功した。
以前の個別再実行が通過していたことは、この公開問題を修復した証拠ではなかった。

`output.py`の一つのdirectory公開処理を、finalizeと完成partialの再公開で共有した。WindowsのWinError 5/32/33だけを
25 ms間隔・最大1 sのI/O期限で再試行する。実renameの成功だけで公開を確定し、出力先の出現、partialの不正、他OSや
他errorは成功へ読み替えない。恒常的な拒否は元errorで失敗し、完成partialを保持する。物理計算、時間刻み、許容値は変えない。

新規15件を含むoutput・populationの42 testsが成功した。実Win32 read handleの短時間保持・解放で公開成功、保持継続で
期限切れ・partial保持、解放後の再公開を確認した。修復後の公開API 200回では、209回のrename中の実WinError 5を9回回復し、
API失敗0だった。全200結果のfinal/event/series/frame/probe科学arrayは修復前の同じ検証caseとbitwise一致した。
これは一sessionの観測であり、恒常的なfilesystem拒否の解消や原因processの特定ではない。
[公開処理の根拠と実行記録](../../.artifacts/review_vigus_20261122/publication_repair_receipt.json)に対照・失敗・source hashを残した。

### 数値の保全と検査

最終固定sourceの正式verification/scenario全体は**1016 passed、445.68 s**。既存のoverflow負例にwarningが1件あり、失敗ではない。開始・終了の254 Python/config入力hashは全件同一だった。追加のproducer v2＋populationは21 passed、cap-crossingの公開解析解、壁・builder・viewerの影響確認も成功した。lock、全262ファイルのRuff format/lint、configured Pyrefly、既存3 import契約、CC<16 gateを確認した。[最終品質receipt](../../.artifacts/review_vigus_20261122/final_quality_gate.json)へ固定した範囲の結果であり、全native一致・実機妥当性へ読み替えない。

builderのbefore証拠は、HEADのworkflow/numericsを隔離した[今回の再現記録](../../.artifacts/review_vigus_20261122/unanchored_builder_reproduction_20261122_complete/receipt.json)に保全した。無anchor成分のV=0とV=1でfree residualは厳密0、V=10でも4.04e-28 Cと既存absolute budget1e-18 C以内だった。修復後の公開経路は未指定perimeterを、低位数値経路は無anchor成分を拒否し、両成分へ電位を与えた正常な公開buildは所定の0/10 Vとなった。見つからなかった過去のprobeを実測証拠として再利用せず、新たに再現した日時とsourceを記録した。

通常モデル3用途のbefore/afterでは、final・release・boundary/failure・時系列・全frame/probeの科学payloadがbitwise一致した。viewerは同じ1024 ID×126時刻・同じdetail出力で123.329→4.072 s。全粒子overviewは維持し、個別SVGは明示したIDだけ生成する。

新reference実装の初版では通常Stokes/RK4に約12%の遅延を再現した。profileで各rowへ渡す未選択9配列の束の費用を分離し、prepare済み整数offsetと既存packed slab一つへ統合した。追加engine/JIT/通常model専用dispatchは作っていない。同条件の逆順測定はbaseline3.41288 s、初版3.76683 s、修正版3.41942 s（baseline比+0.19%）、RSS195.42/197.18/196.51 MB。全24 complete Stokes runで科学payloadが一致した。一つのWindows環境の観測であり、一般的な無負担の保証にはしない。

packing修正後に今回のA/P preferred caseも全30 ms再実行し、元resultの全科学arrayとbitwise一致した。新source hash・case/data hash・新旧run hashは`case{A,P}/packing_recheck/comparison.json`へ保全した。

既存の期待値も点検した。旧catalog/runtimeの固定文字列は所有constantへ統合し、物理の数値許容は変更していない。builderの欠落facetテストは別の設定欠落が失敗理由を隠さないよう分離した。比較器の「全frameに全IDがある」という誤要求はevent時刻に基づくcoverageへ置換し、未説明の欠損は引き続き拒否する。開発途中の非公開型を含む旧Numba生成cacheは再生成し、互換shimを残していない。

プロジェクトskillの参照にも残っていた「noiseはlinear Epsteinのみ」という記述は標準FDTに限定した。明示係数revisionとisotropicは現行ownerへ接続し、Brownianの正規化parameter・実step・時刻key共有をinspectionから渡す対象に加えた。inspectがnative証拠、prepareがmodel対応と数値計画、validateが比較判定、visualizeが保存証拠の表示を所有する区分は維持した。新しいcase値をdefault化せず、`AGENTS.md`は変更していない。
独立レビューで数値計画に残っていたFDT限定の記述も修正し、4技能・12 Markdown・45相対リンク/見出し参照とmetadataを再確認した。[技能参照の点検記録](../../.artifacts/review_vigus_20261122/skill_reference_review.json)に対象hashを保存した。

native・物理数値・V&Vの別担当が、モデル意味、global/local bounds、cap、wall RNG、残時間、入力依存、結果coverageと修復の影響を点検した。原因として未再現の仮説を確定扱いせず、残る近似と未認定事項を各比較receiptへ保持する。

詳細raw根拠は[今回の実行artifact](../../.artifacts/case_ap_vigus_repair_20261122/)、[品質・性能・独立レビュー](../../.artifacts/review_vigus_20261122/)にある。大量の調査用Java/Python probeをproductionへ移していない。

native同定・式・係数導出・microcase・companionの詳細は[日本語native記録](../../.artifacts/case_ap_vigus_repair_20261122/native_report_section.md)を参照。
全287粒子を同じ軸・equal aspect・全121保存時刻で描いた[Case A](../../.artifacts/case_ap_vigus_repair_20261122/caseA/figures_refined1_rk4_white_limit_s21/comparison.png)と[Case P](../../.artifacts/case_ap_vigus_repair_20261122/caseP/figures_refined1_rk4_white_limit_s21/comparison.png)は表示を点検済み。線は保存点間の直線で、未保存の反射や欠損位置は再構成していない。個別path一致の判定図ではない。
