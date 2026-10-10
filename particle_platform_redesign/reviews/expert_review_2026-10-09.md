# 粒子輸送プロジェクトの専門レビュー 2026年10月9日

現行プロジェクトの総合評価は **79／100点**である。明示した2D問題を再現可能に解く数値基盤はよく成立している。実チャンバーの汚染粒子到達・付着を広く定量予測し、同等精度でCOMSOLより効率よく計算するという当初目的は、限定条件の実証まで進んだ段階である。次の重点は機能の追加より、場の意味、キャッシュ品質、外部比較の再実行性、release gateの信頼性を閉じることである。

## 対象と判断の範囲

評価対象は現在の作業ツリーであり、HEADだけではない。HEADは fa51c2e10f32153afbb08d54012ec676f70508c4、レビュー開始時には変更・未追跡を含む117のstatus行があった。現行packageは未公開の0.2.0候補、engine v44、case／data／result schema 3である。公開済み0.1.0、過去のV&V成果物、現在の候補を区別した。

主対象は独立uv projectのsolver、数値verification、公開API scenario、COMSOL adapterと外部V&V、electrostatic builder、field preprocessorである。設計資料とreport_appの説明も照合した。ソルバーの全機能組合せを形式的に証明した評価ではなく、専門分野別のソース調査と、重要な反例の実行に基づくレビューである。COMSOLの新規solve、実機測定との照合、現行版の100万粒子計測、report_appのブラウザ操作検証は行っていない。

旧コードの環境・packageは使用せず、production code、既存仕様、入力datasetを変更していない。このレビューは設計の新しい権威文書ではなく、改善判断のための記録である。

### 採点方法

点数は完成率や不具合発生確率ではない。物理・数値の説明可能性、実装品質、検証の強さ、再現性、当初目的への実証を評価した審査点である。未実装と明記された機能を隠れた不具合としては数えていない。

| 評価軸 | 評価点 | 重み | 主な評価理由 |
|---|---:|---:|---|
| シミュレーションの物理と数値 | 88 | 30% | 独立oracle、連成帯電、積分収束、eventとsupportの分離が強い。可変係数SDEと3D解釈に制約 |
| コード設計と実装品質 | 79 | 25% | 三API、一engine、責務分離が成立。CI、result整合、例外境界に確認済み欠陥 |
| COMSOL入力と外部V&V | 76 | 20% | 比較方法論と限定scopeの証拠が丁寧。Case-P dragの場所有に重大な再確認事項 |
| 検証運用と再実行性 | 70 | 15% | core 739件合格。外部toolsは旧schema／revisionと現行版が不整合 |
| 性能実証と製品目的への到達 | 70 | 10% | 旧revisionの大規模実証はある。現行複合物理の大規模SLAと同等精度COMSOL速度比は未成立 |
| 総合 | **79** | 100% | 加重平均78.85を整数へ丸めた値 |

## 当初目的への到達

製品目的の正本は[product_specification.md](../product_specification.md:35)、現行機能の入口は[support_and_errors.md](../solver/docs/support_and_errors.md:27)である。

| 当初の目的 | 到達状況 | 残る条件 |
|---|---|---|
| 外部場とgeometryから粒子輸送を解く | 2D XY／RZ、SI canonical入力、P1／Q1／regularで成立 | 任意producerの科学的意味はproducer側で認証が必要 |
| 表面発生から壁到達を記録する | realized surface schedule、first-hit、残時間、各wall law、hold、有限接触の候補実装まで成立 | 剥離・再飛散・adhesionを実機から予測するモデルは別課題 |
| 粒子電荷と運動を連成する | fixed／continuous charge、OML、aggregate二電流／三電流を明示選択 | species-resolved物理や実機電荷の妥当性は未認定 |
| Brownianを再現可能に計算する | 2自由度OU、counter RNG、conditional tree、active wall、resumeが成立 | 可変場の実noise精度、真の3D等方拡散との違いを定量化する必要 |
| 時間依存場を使う | 固定topologyの線形snapshot、実stage時刻評価、knot分割を現行候補で実装 | 不連続場、moving topology、moving geometryは非対応 |
| COMSOLと意味を揃えて比較する | 同じP1場の限定決定論・境界・集団比較を多数完了 | Case-P Brownianのdrag source、native FE再現、現行候補の新機能coverage |
| 1万～100万粒子を効率よく計算する | 旧revisionで代表用途の規模・memory実測を保存 | 現行full-physics／Brownian／finite contactの規模別SLAと等精度比較 |
| 3Dへ拡張する | 責務とwork package P17を定義 | XYZ state、RZ mapping、回転面event、3D Brownianは未実装 |
| 第三者が修正できる | owner表、少数の依存契約、uv品質gateを整備 | 約1万行のengine／events、大きいkernel境界、文書driftの整理 |

当初の広い装置予測目的への専門家評価は70点程度である。数値方式の正しさ、選んだ近似モデルの妥当性、実機の源・壁・背景場の妥当性は、異なる証拠を必要とする。

## 各処理の詳細評価

### 入力と設定

case.pyはYAMLの重複key、未知key、参照、基本値を検査し、設定を再帰的にfreezeする。case_format.pyがcanonical read／write／logical hashを一意に所有する。HDF5のdtype、rank、SI情報、source identity、危険なlink構造などを早期に検査し、numeric footprintをmemory gateへ渡す構造は適切である。

一方、load_caseの成功はgeometry、physics組合せ、完全なmemory planまで含む実行可能性の認定ではない。これらはsimulateのprepareで解決する。現行support文書はその差を説明しており、CLIに別preflight engineを追加する必要はない。巨大数値の公開例外漏れと、外部builder／preprocessorの重複key受理は修正対象である。

### 粒子物性と発生源

mass、drag径、electrostatic半径、displaced volume、contact半径、model weightを別authorityとして持つ設計は正しい。粒径から慣性を実行中に再構成しないため、多孔質粒子や相当径を追加するときの誤用を防げる。

sourceは外部でrealizeした位置・速度・release時刻を入力し、固定particle IDとrelease順を別々に管理する。source RNGや任意分布catalogをcoreへ入れない判断は責務を明確にする。ただし、入力scheduleを再現できることは、実機でどこから何個剥離するかを予測できることとは別である。RZ回転面の分布を作るproducerは面積重み2πr dsを保持する必要がある。

### 座標とgeometry

XYとRZ meridionalを明示し、RZ軸を材料wallから分離してsigned chartと基底変換で扱う点はよい。gravityの一定radial成分を標準重力として受理しない判断も適切である。

geometryはcontainment、BVH、first-hitを所有し、wall lawやfield値を持たない。point particleと有限半径接触を独立contact radiusで分ける現行候補は、接触位置をtoleranceで代用する方法より物理的に明確である。有限接触、periodic seam、同時corner、曲線pathの組合せは複雑であり、個別microcaseを越えた任意形状の認定には至っていない。

### 場の探索と補間

fieldの値とsupportを分け、stageの実位置・実時刻で評価する。受理時だけcell hintをcommitする構造は、trialでdomain外の値を使っても正常stateへ紛れ込ませない。P1／Q1／regularの検証と固定topologyの時間補間もよく構成されている。

geometryとfieldは概念上独立だが、現行一般曲線運動では材料domainとP1／Q1 fieldのnodes・connectivity・supportに同一性の制約がある。これは文書化された能力制限である。任意の異なるmesh同士を既に扱えると説明してはいけない。

上流のfield cacheには局所特徴を見落とす誤差gateの反例がある。core samplerの正しさだけでは、粗いcacheへ置き換えた物理場の品質を保証できない。

### 力と帯電

physicsはsample済みprimitiveからrateとaccelerationを計算し、mesh探索、COMSOL、I/Oを持たない。線形dragと加速度を分け、連続電荷と運動を同じstage／midpointで連成し、charge-only subcycleを導入していない点を高く評価する。

有限速度Epsteinは分子速度積分、Waldmannは運動論moment、Barnesはimpact-parameter積分などの独立oracleを持つ。pureとcompiledの相互一致だけで検証を終えていない。Talbot、Saffman、DEP、aggregate charging、ion-drag sensitivityを明示revisionで分け、適用域外で別式へ切り替えない方針も適切である。

ただし、effective-gasやaggregate modelの数値一致は、混合気体・多イオン種の物理的真値を認定しない。OMLのcollisionless、unmagnetized、Maxwellianなどにはproducerの物理前提が残る。runtimeのapplicable判定は実装した数値条件の検査であり、装置物理の全面的認証ではない。

### 時間積分

smooth manufactured caseのRK4、可変係数exponential midpoint、continuous chargeの連成収束がある。midpoint-frozen affine exponential chargeを使い、stiffnessとaccuracyを別に扱う設計は実用的である。固定macro stepを選ぶために別runのh、h/2、h/4を使う方針も説明可能である。

出力時刻や個別releaseのために全粒子stepを分割しない点は、再現性と効率の両方に有効である。一方、RK4のaccepted endpointとdense interiorは同じ精度主張ではない。現行のdense pathとAGENTSの再積分要求の衝突は解消すべきである。

### 境界と残時間

endpointだけで壁を判断せず、path上のfirst-hitを局在し、hit prefixと応答後の残時間を扱う構造は本製品の強みである。同時incident facet集合、combined normal、zero-time departure、axis、support、accuracyを別に扱っている。

stick、escape、holdを区別し、holdを堆積数へ加えないことはCOMSOL Freezeの意味にも合う。反射や熱壁の規則は再現可能なモデルとして実装されているが、実際のチャンバー表面の付着確率、roughness、再飛散の校正は別に必要である。

### Brownianと乱数

joint OUの位置・速度共分散、conditional half-split、固定particle IDとphysical ordinalに基づくcounter RNG、slab／output／resume identityの設計は強い。active wall hit後の使用済みtailを破棄し、post-wall stateからfresh rootを開始する意味も明示されている。

現行のfinite-depth Hermite pathは離散的な数値pathであり、連続OUのexact first-passageやzero missを主張していない。base depthとcandidateだけを細分するmax depthを分けたこともよい。

残る重要課題は、実noiseを伴う可変係数SDEの独立精度検証と、2自由度RZを3D等方粒子輸送として解釈しないことである。軸foldを行う1成分radial GaussianではE[r²]=σ²だが、3D運動の2成分transverse radiusではE[r²]=2σ²となる。P17は見た目の3D化だけではなく、到達人口や径方向密度の物理解釈を変える。

### CPUとmemory

一つのserial compiled engine、resident SoA、bounded slab、read-only field／geometryを維持している。失敗したparallel経路を削除し、kernel単体speedupを製品のend-to-end性能へ読み替えなかった判断は適切である。

memory planはsolver-owned配列の予測であり、OS RSS hard capではないと説明している。Brownian max depthを基準に計画する点も保守的である。ただしscratchの一度確保と再利用は全経路で徹底されておらず、OU／eventのwaveごとの配列生成には計測に基づく改善余地がある。

### 出力と再開

同期single-owner writerがsegment、inactive checkpoint、LATESTの順でcommitする設計は追跡しやすい。final、manifest、success marker、directory publicationの順序、失敗注入、resume identityも丁寧である。ResultViewから後処理する構造は計算と可視化を分離している。

今回確認した欠陥は、完成finalの粒子行数をmanifestに照合していないことである。正常writerの計算誤答を確認したわけではないが、欠落したartifactを完成扱いで開けるため、結果の人口や堆積率を利用する運用では修正が必要である。

## 優先して修正する指摘

P1は次のreleaseや主要な科学的認定に使う前に扱う問題、P2は再現性・入力・保守の具体的な修正課題とする。P0相当の主要force式の明白な誤実装は、今回の調査範囲では確認していない。

### P1 CIが途中の品質gate失敗を見逃す

[solver-release.yml](../../.github/workflows/solver-release.yml:45)はpwshの一つのstepで複数のnative commandを続け、各終了コードを確認していない。GitHubは末尾のLASTEXITCODEをstepへ反映するため、前のRuff、型、依存、complexityの失敗が最後のpytest成功で隠れ得る。[GitHub公式の終了コード仕様](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#exit-codes-and-error-action-preference)

GitHub相当のPowerShell設定で、最初のcommandがexit 7、次がexit 0のとき、後続が実行されscript全体がexit 0となることを再現した。個別に実行した今回の品質gate合格は有効だが、workflowが常に全gateを拒否条件にしている保証はない。

改善は各gateを別stepへ分けるか、各command直後に非zeroをexit／throwすること。Quick Startとwheel検証の複数commandにも同じ原則を適用する。意図的な前段失敗が確実にredになる小さい確認が完了条件である。

### P1 Case-P Brownian比較の同一場という主張を再確認する

[RunM3C2StochasticCampaign.java](../solver/tools/vv/comsol/comsol/RunM3C2StochasticCampaign.java:327)はdragの速度・温度・圧力の値欄をP1へ変更するが、対応source selectorを変更・確認していない。保存Case-P設定ではselectorがnativeのroot.comp1.u、root.comp1.T、root.comp1.spf.pAを指す。受理済campaignのJava hashは現在のfileと一致している。

同じ設定方法を用いた後続M3-C3の[COMSOL診断log](../solver/_out_m3c3_casep_three_current_v4/diagnostic_dt_1p25us/comsol_process.log:3)には、P1の値欄とnative sourceの混在が記録されている。保存RHS診断のdrag相対L2差は約1.289e-4、合力は約4.588e-6でBLOCKEDとなった。[M3-C3のREADME](../solver/evidence/m3c3/caseP_three_current_companion_v1/README.md:50)はその問題と明示dragへの修正を記録する。[COMSOL公式Drag Force説明](https://doc.comsol.com/6.4/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.06.html)も他physicsからのsource選択とUser definedを別設定として扱う。

確定した問題は、M3-C2 runnerが同じfield authorityを保証しておらず、後続Case-Pで同じ方法が混在を生んだこと。M3-C2の32 replicaに対する修正後の誤差と統計gateへの影響は今回未測定である。既存population gateの算出結果は保持できるが、純粋なsame-field solver parityの認定は再確認が必要である。

外部runnerでselectorを明示し、実使用primitiveと同一状態RHSを認証する。過去receiptを保持したうえで訂正scopeを記録し、必要な比較revisionだけを再実行する。solver coreの係数fitやgate緩和で対応すべき問題ではない。

### P2 field cacheの誤差gateが局所特徴を見落とす

[field_preprocessor/numerics.py](../solver/tools/field_preprocessor/numerics.py:257)はtarget cellの固定quadratureとboundary midpointだけで誤差を計算する。source側の細かい特徴が検査点の間にあると見落とす。

公開resample処理で、1 m正方形・12個のP1 triangleに300+100x Kの背景と中心±0.01 mの局所tent +300 Kを与え、2×2 regularへ変換した反例を確認した。

| 再現の指標 | 結果 |
|---|---:|
| 報告value relative L2 | 1.81e-16 |
| 報告gradient relative L2 | 1.50e-15 |
| 報告boundary relative L2 | 0 |
| 全limitを0.001としたgate | PASS |
| 元場とcacheの中心温度 | 650 K／350 K |
| P1領域積分による実際のvalue relative L2 | 0.0069739 |

指定0.1%に対して実誤差は約0.697%となる。supportの完全被覆と物理量が正値であることは満たしているため、zero-referenceだけの特殊問題ではない。sheathや局所温度・電場のピークを粗いcacheへ移す用途では影響が大きい。

source／target cellの共通分割で積分するか、source節点と小さいcellを含む検査を導入する。現在のmetricは固定stencil上の推定誤差であり、場全体の誤差上限として扱えない。最小の局所tent回帰caseを一つ追加するのが適切である。

### P2 外部toolsが現行checkoutで再検証できない

公式のmodule invocationでtools suiteは35 failed、324 passed、14 errorsとなった。主要因は過去schema 1／2の入力を現行schema 3 readerへ渡すことと、固定した過去algorithm revisionを現行runtimeへ要求することである。例は[case_format.py](../solver/src/chamber_particles/case_format.py:1154)、[run_p18h_hold_candidate.py](../solver/tools/vv/comsol/run_p18h_hold_candidate.py:295)である。

過去の科学的PASSを一括して否定する結果ではない。一方、現行版で記載commandから再検証できる状態は未達である。

historical evidenceには当時のpackage、lock、commandを対応させ、current用fixture／recipeは外部producerで明示的に再生成する。既存raw referenceを上書きしたり、productionへ旧readerを復活させたり、gateを緩めたりしない。履歴の再現と現行版の比較を、実行versionの所有で分けるべきである。

### P2 完成finalの粒子行欠落を受理する

[output.py](../solver/src/chamber_particles/output.py:1484)はfinal内部の列shapeを検査するが、manifestの粒子数へ照合しない。正常2粒子run C04の一時copyから全final列を1行減らすと、complete=True、manifestの粒子数2、final IDは[401]として開けた。同数の重複ID[401,401]、逆順[402,401]も受理した。

[result_format_v3.md](../solver/docs/result_format_v3.md:220)は全resident粒子をparticle ID順で格納する契約である。final行数とmanifest粒子数の比較、bounded scanでのID一意性・順序確認が必要である。正常writerから欠落することは今回確認していない。

### P2 外部設定の重複keyで物理条件とgateが変わる

[electrostatic_builder/workflow.py](../solver/tools/electrostatic_builder/workflow.py:141)と[field_preprocessor/workflow.py](../solver/tools/field_preprocessor/workflow.py:116)はyaml.safe_loadを使い、同一mappingの重複keyを後値で上書きする。

electron_temperature_Vを4.0と8.0で重複させると8.0を採用して成功し、value_relative_l2を0.001と0.9で重複させると0.9を採用して成功した。入力条件や誤差の受入幅が黙って変わるため、各configuration ownerで重複keyを拒否すべきである。core case parserとCOMSOL adapterには既に拒否する実装がある。一般validator frameworkを追加する必要はない。

### P2 巨大入力値が公開例外を抜ける

[case.py](../solver/src/chamber_particles/case.py:705)のfloat変換にtime.end_s=10の400乗を与えるとOverflowErrorとなる。[api.py](../solver/src/chamber_particles/api.py:36)はこれをCaseErrorへ変換せず、CLIもtracebackとexit 1になる。通常のJSON errorとexit 2の契約に合わない。

数値変換ownerでOverflowErrorを説明付きValueErrorへ変換し、公開例外分類へ統一する。巨大整数を一つ与える回帰caseで確認できる。

## 数値仕様と検証の改善課題

### RK4 dense pathの仕様を一意にする

[AGENTS.md](../AGENTS.md:142)は曲線state_atを始点から同じintegratorで再評価すると要求する。一方、[integrators.py](../solver/src/chamber_particles/integrators.py:326)は保存したdense extensionを評価し、[numerics.md](../solver/docs/numerics.md:1377)はその意味を明示している。

x''=-x、x0=1、v0=0、h=1の再現で、state_at(0.5)の位置は0.875、始点から半幅RK4を実行した位置は0.8776041667となった。新dense実装を誤りと断定する結果ではなく、規約と実装が異なることの証拠である。

出力dense path、event探索path、hit prefix再積分の役割と誤差を権威文書に揃える。endpointの形式4次をdense interiorへそのまま広げない。旧方式の復元だけを目的に変更する必要はない。

### 実noiseを伴う可変係数Brownianの精度を検証する

[test_stochastic.py](../solver/tests/verification/test_stochastic.py:113)のweak mean試験はOU incrementを0として進める。これは決定論meanの検証として有用だが、係数が確率的な位置・速度に依存する問題の弱収束の証拠にはならない。定数係数OUの平均・共分散、depth安定化、containment、slab identityは既に強い証拠がある。

一つの小さい可変係数manufactured SDEを選び、実noiseの独立seed ensembleと高精度referenceで平均、分散、電荷、first-arrivalを検証する。macro h、h/2、h/4とbase depthを別軸で評価し、Monte Carlo不確かさと離散化誤差を分ける。これでgeneral SDEの高次精度を主張する必要はない。

### producerの科学的意味の認証責任を明示する

[physics_models.md](../solver/docs/physics_models.md:111)はeffective-gasをproducer認証済みpseudogas入力に限定するが、runtimeは汎用provenanceとprimitive条件を検査し、その集約近似の認証内容を必須にしていない。最小metadataだけのcaseでも公開simulateは成功した。

producerの利用責任として要求するのか、version付きの最小宣言をprepareで必須にするのかを決める。型や単位検査で物理的closureを認証したと説明しない。

OMLでは電子Maxwellianの静止frameも明確にする余地がある。粒子とイオンを同じ高速で動かすとion drift gateを満たすが、laboratory frameで静止した電子に対しては電子driftが大きい場合がある。通常dust速度を代表する反例ではなく、現時点ではframe前提の設計課題とする。

## COMSOL比較の到達度と制限

[validate-comsol-trajectoriesの基準](../../.agents/skills/validate-comsol-trajectories/SKILL.md)に沿って、初期条件、primitive／RHS、境界、時間軌道の意味を分けた。

| 比較成果 | 認められる範囲 |
|---|---|
| Case-A common-P1決定論 | 100 nm、287粒子、pre-event 450 µsの位置・速度・電荷gate合格 |
| Case-A初回wafer hit | material 20/20、pre-event 9/9、event時刻・位置・電荷比較を完了 |
| Freeze／Disappearとhold | 正例probe、hold／Freeze candidate 15/15を完了 |
| critical boundary | 表面離脱、specular残時間、RZ軸の132/132 gateを完了 |
| Case-A Brownian | 32＋32独立seed、30 ms、終端人口曲線の5%同等性幅でPASS |
| Case-P Brownian | 83区分RZ／fate、15%幅のpopulation gateはPASS。ただしdrag ownershipを再確認 |
| Case-P三電流companion | 明示dragへ修正後、100 nm、287粒子、30 ms、3刻み、141 terminal eventまでPASS |
| Case-A追加粒径／ion drag | 10／30 nm relative-flow、100 nm imageのevent-free決定論sliceを確認 |
| native FEとexported P1 | 厳しいcross-representation軌道gateはFAIL。同じP1場のPASSへ混ぜていない |

Case-P Brownianのevent 0という結果は、その終端gateに境界parityの情報がないことを意味する。Case-Pの15%幅とCase-Aの5%幅は検定対象・metricが違い、同じaccuracy等級として比較できない。検定PASSは登録した区分・統計量と幅の範囲での結果である。

same-field comparisonは場を固定した粒子solverのagreementを調べ、native-field reproductionは場生成・輸出・補間を含む誤差を調べる。さらに実機の物理妥当性には測定とのvalidationが必要である。この三つを分ける設計は適切であり、保存された過去のNOT_TESTEDを後続で閉じた成果まで未達に戻すべきではない。

受理済evidenceは記載されたschema、model、algorithm revisionのsnapshotである。現行0.2候補の時間場、有限半径、periodic、新Brownianの任意組合せへ認定を自動拡張できない。

## 保守性と性能の評価

現行coreは26 Python file、51,203行、1,260関数・methodを持つ。engine.pyは10,165行／188関数、events.pyは10,128行／189関数である。physicsのcompiled入口には91引数、curved event kernelには46引数がある。Numbaのflat-array境界には理由があるが、CCだけでは引数順序、array owner、変更時の影響を測れない。

行数だけの機械分割は適切ではない。prepare／resumeの調停、OU root継続、accepted output replayなど、独立した変更理由が現れた箇所でownerを整理する。kernel境界の変更はpayload identityと性能の両方で確認する。一般plugin、DI、第二schedulerを追加する必要はない。

event stagingのslabごとの確保、OU次waveのwhile内確保、curved wavefrontのcopyはboundedであり、これだけでunbounded memoryや速度劣化とは断定しない。allocator／copyの寄与がprofileで確認された場合に、必要なbufferだけを再利用workspaceへ集約する。

[P14-U release証跡](../solver/evidence/v0.1/p14u_release_v1.json)には旧engine v27で10k／100k／1M、2 output mode、各3 fresh processの実測がある。1M noneのpublic end-to-end中央値は約553.8秒、process peak RSSは約799.7 MBだった。これは代表用途の局所machine観測であり、現行v44のfull physics、Brownian、有限接触の性能保証でも、COMSOLとの同等精度速度比較でもない。

当初の速度目的には、粒子数、step数、要求accuracy、geometry規模、場表現、境界event頻度、出力量を固定したSLAが必要である。その条件で10k／100k／1Mを測り、COMSOL比を述べる場合は同じ場・物理・精度・出力を揃える。GPUや内部並列の追加を性能課題の自動的な解決策にはしない。

report_appは2026年9月23日時点の製品仕様・実装前計画を要約するsnapshotで、現在の到達度dashboardではない。既存snapshotを歴史資料として残すのは妥当だが、現行状況を示す場合はdateとrevision、現在の完了範囲を別に明示する必要がある。UIの実行検証は今回の採点へ含めていない。

## 今回の実行検証

solverの独立uv環境を使用し、lockの変更・dependency更新は行っていない。

| 検査 | 結果 |
|---|---|
| uv lock --check | 合格 |
| Ruff format src tests tools | 225 fileで合格 |
| Ruff lint src tests tools | 合格 |
| Pyrefly | 0 errors。tool出力は61 suppressed、246 warnings非表示も報告 |
| import-linter | 3契約合格、0違反 |
| complexity gate | 全関数・methodがCC 16未満 |
| core verification／scenarios | 739 passed、94.07秒 |
| 外部tools | 324 passed、35 failed、14 errors、27.56秒 |
| 現行performance smoke | 8条件完了。64粒子、regular／P1／Q1、初期探索／cell移動、時間snapshot、境界、cold／warmと出力を確認 |

Windowsの実行ポリシーがlint-importsのlauncherを拒否したため、同じlock内のPython entry pointを直接呼んで3契約を確認した。外部toolsはdocumented commandのpython -m pytestで実行した。これらの環境・入口の問題と、実際のschema／revision不整合による失敗を区別した。

小型反例は一時directoryで実行した。確認対象はCI終了コード、完成finalの行欠落、巨大数値、重複YAML、cacheの局所特徴、dense path、producer意味の受理である。COMSOL source問題は保存receipt、hash、feature設定、後続診断を照合した結果であり、新しいCOMSOL runによる修正効果の測定ではない。

performance smokeは科学payloadのidentity、各条件の完了、出力shape、memory plan、compiled dispatchを確認した。exact_global_algorithm_revisions、exact_repeats、exact_within_science_keyはいずれもtrueだった。8条件は各1観測で、warm条件には別warmupがある。現行版の10k／100k／1M scalingやCOMSOL速度比を示す測定ではない。

再実行する場合はsolver directoryで次のcommandを使う。新しいperformance JSONは一時directoryなどの別出力先へ保存する。

    uv lock --check
    uv run --locked ruff format --check src tests tools
    uv run --locked ruff check src tests tools
    uv run --locked pyrefly check --summarize-errors
    uv run --locked lint-imports
    uv run --locked python scripts/check_complexity.py
    uv run --locked python -m pytest tests/verification tests/scenarios -q
    uv run --locked python -m pytest tools -q
    uv run --locked python -m tests.performance.p14_matrix --suite smoke --json OUTPUT.json

今回のWindowsでlauncherを使えなかったimport-linterだけは、同じentry pointを次のように実行した。

    uv run --locked python -c "from importlinter.cli import lint_imports_command; lint_imports_command()"

## 次に閉じる作業

| 優先 | 作業 | 完了を示す証拠 |
|---|---|---|
| 1 | CIのcommand別失敗検出 | 前段gateを意図的に失敗させ、後段成功でもworkflowがred |
| 2 | cache誤差検査をsource特徴へ対応 | 局所tent反例を拒否し、値・gradientの誤差metricの意味を明示 |
| 3 | Case-P COMSOL dragのauthority確認 | selector、実primitive、同一状態RHSのreceipt。影響する比較だけ訂正 |
| 4 | 外部toolsのversion所有を整理 | historical再現とcurrent fixture生成がそれぞれ明示commandで動作 |
| 5 | final整合と入力例外を修正 | 行欠落・重複／逆順ID、巨大整数、重複YAMLの最小回帰 |
| 6 | 数値規約と実装の意味を一致 | dense出力とhit prefixの役割を一意に記載 |
| 7 | 可変係数Brownianの精度を分離 | 実noise ensemble、独立reference、h系列とdepth系列、統計区間 |
| 8 | 装置予測と規模SLAを定義 | 2D近似の限界、必要ならP17、実測源／壁校正、現行版の規模計測 |

新しい力modelや一般化よりも、まず入力と比較条件を信頼できる状態へ揃えることが、既存の数値基盤を製品価値へ結び付ける。
