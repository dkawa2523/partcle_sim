# 粒子輸送基盤の課題・根因・具体改良設計

再調査基点: 9015de024aabc145b6615ccc18bab486055bf894。調査日: 2026-11-23。
対象は現行の独立project全体。これは実装に移せる粒度の提案であり、現行仕様を変更した記録ではない。
現行の権威はproduct_specification、architecture_proposal、solver/docs。採用時はそのowner文書と呼出側を同じ変更で更新する。
以前の14項目を現行sourceで照合し、新しい公開入口の不具合と接続上の課題を追加した。R10/R13の完了部分と、未実装の提案を区別する。

## 1. 結論と、先に変える単位

最大の問題は、大きなfileそのものではない。**入力の意味を確定する責任と、確定した事実を次へ渡す責任が、一部の境界で閉じていないこと**である。
source設定は未解決のまま使用され、外部IDはfloatへ変換してからexact性を検査し、static builderは汎用FieldDataの時間意味を解決していない。
物理側では一度選んだmodelをbinding・proof・説明で再解釈し、event側では同じaccepted material responseを三入口で更新している。

このため、正常例を動かしても、不正入力のsilent acceptance、遅い失敗、変更箇所の非局所化、説明の欠落が残る。
根本改良は次の順にする。

1. 誤ったidentity結合とstatic入力誤受理を止める。source設定/CSVと単独JSON公開も同じ境界責務で閉じる。
2. 型付きmodel recordからrole・scalar・stage・proof・check/result説明を一方向に構築する。
3. 親の時計/説明/資源planとworkerの数値recordを分離し、resident stateを既存ownerのまま入口へ渡す。
4. accepted prefixの後、method固有continuationの前にあるmaterial状態更新を一ownerへ集約する。
5. 結果の内容identity、宣言した派生係数、科学比較、reportの入力/実行/buildの根拠を別々の意味として結線する。
6. 置換した経路・重複定義・専用設定/tests/docsを同時撤去し、独立oracleと有効な観測記録を保持する。

単一engine、core四API、通常clientのwheel収録、共通prepare、SI、独立粒子物性、accepted-only commit、conditional OU、
親single writerとbounded ACKは成立している。これらを再構築する提案にはしない。
現行の全材料/浮遊導体非対応と、既存機能の不具合も混同しない。

## 2. 利用目的と調査範囲

基盤の目的は、利用者が自分の外部場・geometry・realized sourceを準備し、機能を選び、検査・計算し、
結果を解釈・再利用できること。科学的意味、実行資源、結果の説明、拡張/削減の変更範囲を追えることを成功条件とする。

| 主要領域・独立経路 | 現行ownerと確認範囲 | 現行で成立している点／残る課題 |
|---|---|---|
| 機能選択・CLI/API | api、cli、catalog、pyproject entrypoint | console/moduleは一dispatcher、core四API。selector全直積の保証ではない |
| YAML/canonical/recipe | case、case_format、yaml_input、inputs、bundle_publication | load/check/公開の役割は分離。recipe2の全source集合を保持。nested source解決にR06 |
| source | sources、source_builder、coordinates | joint realized ID/release/x/v/Z/独立物性/weightを保持。CSV構造/lexical errorの扱いにR06 |
| CSV場変換 | csv_import | provider/selection/単位/基底/geometryを明示。ID丸め誤結合を新規確認、R15 |
| 場補間・cache | fields、field_numerics、field_preprocessor | 値/support、時間knot、axis、認証cacheを区別。新規の具体的不具合は本調査でなし |
| 縮約電場・係数producer | electrostatic_builder、prepare_vigus_transport | static gas-only RZ/P1。時系列thermal入力のfalse-readyはR16、派生係数handoffはR09 |
| 物理全9カテゴリ | catalog/runtime/compiled/forces/charge | 27 descriptor、22 family群を確認。R01–R04、R19 |
| 積分・確率 | integrators/stochastic/rng | 同stage x/v/Z、joint OUとconditional splitを保持。一般SDE strong orderの認定ではない |
| geometry/event/壁 | geometry/events/boundaries/transport/topology | first hit/同時facet/axis/残時間は分離。material commit三入口の重複はR05 |
| 実行・容量・process | engine/cpu/execution/transport/time_grid/lifecycle | 一slab経路、ready/ACK/replay寿命は明瞭。親専用payloadが残るR17 |
| 結果・resume | outputとcpuのpersistent projection | macro barrier、context/checkpoint binding。内容identityの保証範囲はR08 |
| 集計/export/表示 | selection/analysis/visualization | cohort/weight/coverage、完成directory公開。単独JSONだけR07 |
| 外部V&V | tools/vv/comsol、gas_collision | raw/practical/qualified/physicalを分離。dense比較R11、局所資格R12 |
| 配布・examples・検証 | release_workflow、verification/scenarios/performance、quality gates | 例題受入inventoryの重複R18。R13の整理は既に完了 |
| report・根拠・説明 | analysis/build_report_snapshot、report_app | app identity/公式runtimeを保持。projection shape/read/hash/publicationにR20 |
| 文書・資産 | 正本、作業記録、native manifest | 旧成果物撤去は完了済み。native/環境を改良対象の廃止codeと同一視しない |

installed Pythonは49 files（core31、通常client18）、外部V&V非test Pythonは18 files。
inventoryとowner/functionの詳細読解を区別し、全function・全組合せを実行したとはしない。

## 3. 今回の実行証拠と限界

[今回の調査manifest](../solver/evidence/foundation_design_2026_11_23/investigation.json)、
[client probe](../solver/evidence/foundation_design_2026_11_23/client_probe_evidence.json)、
[worker payload probe](../solver/evidence/foundation_design_2026_11_23/payload_probe.json)、
[report read/hash probe](../solver/evidence/foundation_design_2026_11_23/report_read_hash_evidence.json)が根拠である。

| 実施した確認 | 観測結果 | 主張の範囲 |
|---|---|---|
| source構造/lexical不正の公開create | columns:null、filter:null、未存在filter列、不正IDはtraceback/exit1。余剰data cellは成功・公開 | R06の入力境界bug。誤った力計算を実証したものではない |
| installed field CSVのexact ID | mesh 9007199254740992 とraw field 9007199254740993を同一IDと結合してexit0/公開 | R15。座標は意図して同じにした。identityの誤受理を実証 |
| static builderのtime-dependent thermal入力 | 速度time-dependentはcheckでIndexError。温度だけの場合check=ready、buildで小型solve後IndexError | R16。4節点/2三角形。誤った完成電場が成功したとはしない |
| analyze書込み中断 | structured exit2だが28 bytesの部分JSONが公開先に残る | R07。Windows改行変換込みの新しい測定。以前の27 bytesという別probeと区別 |
| report projection入力 | rows=[17]、非存在参照名を受理 | row形状の受入欠落。任意の外部参照を禁止すべきという意味ではない |
| report one-read bindingの制御probe | rowsはA、記録hashは後のBとなる | 既存関数間で新規合成入力だけを書換え。過去のreportが実際に誤っていた証明ではない |
| worker serialization | frame/probe時刻0→4096で21,905→58,771 bytes、配列共有容量は同じ | 親専用metadataの複製。child起動/advanceなし、RSS cap超過の実測ではない |
| 既存client/report関連tests | 51 passed /33.95 s | 現行正常経路の限定実行。上の境界bugを既存testsが捕まえていないことも示す |

今回の新規native COMSOL solve、全Vigus、A/P ensemble、全suite、clean wheel、性能scale試験はない。
1240件のcore/scenario、156件の外部tests、削除後public flowとP10六観測は、
[前段階の完了記録](../solver/evidence/current_only_2026_11_23/completion.json)のsource-bound証拠として保持し、今回の新実行へ数え直さない。

## 4. 課題台帳：現行不具合と構造上の危険を分ける

優先Aは利用者の誤受理/不完全公開を先に止める項目、Bは基盤の変更容易性・説明の責務、Cは用途が必要になった時の能力整備。
「構造」は現在の合法な計算が誤っているという判定ではない。

| ID | 分類・優先 | 根因と実際の影響 | 改良するowner |
|---|---|---|---|
| R01 | 説明欠落・B | resolved_modelsが一部scalarしか保存せず、role→fieldも消える。input hashとは別の不足 | catalog typed record→check/_RunFacts→result |
| R02 | 変更結合・B | 自分のprimitive offsetを他categoryの型/bindingから流用。drag追加/削減がTalbot/lift/ionへ波及 | runtime binder、既存field offset表 |
| R03 | 将来変更の安全性・B | 残余stage codeを既存proof familyとみなす。現行closed catalogでは対応、将来の証明追加漏れが危険 | concrete prepareとproof construction |
| R04 | 数式authority重複・B | array/compiledのEpstein・OML等が同じ物理式を別bodyで維持 | force/charge scalar law、compiled selection |
| R05 | accepted状態の重複所有・B | exact/curved/surface三入口でresident/ordinals/lifecycle/event/statを更新 | transportの共通material commit |
| R06 | 確認済みbug・A | nested設定/row形/ID lexical解決が後段。tracebackだけでなく余剰cellのsilent acceptanceあり | source_builderの設定/CSV owner |
| R07 | 確認済みbug・A | analysis JSON公開先へ直接書き、中断で部分fileとretry障害が残る | 単独JSON publication |
| R08 | identity保証範囲・B | manifest/input/context同定と、final/events等の数値内容の完全性が別。有限weight改変を同一source_identityにできる | outputのcompletion/read/identity |
| R09 | producer handoff・B | primitiveと明示派生係数の関係が生成手順に依存。独立prescribed fieldは合法 | 宣言した派生groupのproducer操作 |
| R10 | 外部reader修正済み | frozen/trajectory CSVのID丸めと重複headerは既に修正。別のinstalled field入口に残るR15を未修正へ取り違えない | 既存修正を保持、R15のownerへ |
| R11 | 外部評価の容量/用途・C | 全frames＋time×ID×state dense。paired-IDの記述比較をunpaired/weighted一般評価に使えない | saved-population evaluator |
| R12 | 資格付け能力・C | global countsから選択ID/時間窓のaccepted path解像を同定できない | 数値計画、既存probe、外部評価 |
| R13 | 整理済み | scenarioのprivate容量定数を実allocation所有確認へ移行済み。独立transient上界は保持 | 現行testを維持、再構築しない |
| R14 | 能力境界 | 現行gas-only縮約電場と履歴S3-Mの全材料/浮遊VQは別問題。過去field差4%を現行精度へ転用不可 | 製品範囲と条件付き全電気領域producer |
| R15 | 新規確認済みbug・A | CSV identityをfloat化した後の2^53検査では丸めを検出できず、異なるnode IDのjoinを公開する | csv_importのtyped field-table read/join |
| R16 | 新規確認済みbug・A | static builderがtime_sを検査せず、汎用FieldDataをstatic arrayとして使用。check=readyでもbuildで失敗 | builder共通PreparedBuilderInput |
| R17 | 資源/責務結合・B | 親専用schedule/説明/planがPreparedTransportに残り全workerへserializeされる | engine._PreparedRun、transport、ProcessTeam |
| R18 | 検証側の事実重複・B | 例題定義は一ownerだがinstalled受入が四名とfive passedを別保持 | release_workflowの実inventory消費 |
| R19 | 説明不整合・B | physics_modelsの加算順説明と実compiled順が異なる。演算順変更で解決すべきでない | 物理文書と実行順の明示 |
| R20 | 根拠handoff・B | projection row形、同一bytesのparse/hash、snapshot公開、実行source/current design/build identityの意味が未閉鎖 | 既存report snapshot producerと検査呼出側 |

R06/R07/R15/R16は公開操作で再現した不具合。R01/R02/R04/R05/R17/R18/R19は確認した構造/説明上の不足。
R03/R08/R09/R11/R12は保証範囲や将来変更・用途の条件を含む。これらを同じ「バグ数」へ合算しない。
R10/R13、撤去済み旧処理、正常なprocess ACK等を未着手へ戻さない。

## 5. 根因を基盤の設計条件へ変換する

| 表面に現れる症状 | 本質的な原因 | 改良後の責務境界 |
|---|---|---|
| malformed入力のtraceback/余剰cell無視/誤join | lexical identity・表構造・意味・物理floatを区別する前に情報を失う | roleのownerがraw bytesを一度解決し、consumerへ確定形だけ渡す |
| static fieldをcheckしたのにbuildで落ちる | canonicalとして合法であることを、producer固有のstatic前提と同一視 | producerのprepareがtemporal/layout/units/axisを解決してcheck/buildへ渡す |
| model追加で他categoryと説明を修正 | model選択から解決済み事実を渡さず、各層が再解釈 | model record→直接role binding→stage/proof→同じfactsの投影 |
| 壁応答の修正が入口ごとに必要 | 共通のaccepted material更新とmethod固有continuationが一体 | prefix受理と固有継続を保持し、その間の共通更新だけ集約 |
| 観測が増えるとworker準備payloadが増える | 親の制御/説明とworkerの数値入力を一つのprepared型へ入れる | 親run plan、worker transport、macro-local replayを分離 |
| hashがあるのに説明/内容同一性が不足 | 入力identity・実行設定・数値内容・引用根拠・build成果物を一つの同定と読む | 各identityの対象/生成時点/検証範囲を明示し、一ownerで結線 |
| 同じ検査を別tests/docsが持つ | 事実ownerと、その事実を確かめる独立oracleの区別が曖昧 | inventory/規則は一ownerを消費。独立数学/解析referenceは別に保持 |

型を増やせば解決するわけではない。新recordは、既存callerが必要とする情報を過不足なく確定し、
その後の再解釈・他category参照・raw Any・二重read・同じstate更新を実際に除去できる場合に限る。
field support/適用域/accuracy、point evaluation/global bound/local証明、material/axis/periodicは異なる事実なので無理に統一しない。

## 6. 改良後の全体構成と受渡し

~~~mermaid
flowchart TD
    A[外部CSV・場・実現source・設定] --> B[各producerの型付き解決とone-read binding]
    B --> C[canonical SI bundleとcase]
    C --> D[load_case / 共通prepare]
    D --> E[親run plan: 時計・観測・説明・資源]
    D --> F[worker transport: 数値record・共有配列・capacity]
    F --> G[実stage sampleと連成rate]
    G --> H[method固有proposalとevent prefix]
    H --> I[accepted prefix]
    I --> J[共通material commit / axis・periodicは別]
    J --> K[method固有continuation]
    K --> G
    I --> L[親single writer / macro checkpoint]
    J --> L
    L --> M[結果context・内容binding・ResultView]
    M --> N[cohort共通のanalysis / export / view]
    M --> O[外部V&V: 用途比較と資格を分離]
    N --> P[reviewed projection]
    O --> P
    P --> Q[one-read snapshot / 公式build / 出典付きreport]
~~~

core公開APIはload_case、check_case、simulate、open_resultの四つを維持する。
通常clientも同じdistribution/唯一CLIを使う。公開API追加や新しいbackend/plugin registryはこの再構成の前提ではない。
以下の新しい型名・signatureは設計例であり、現在の公開APIや実装済み関数ではない。
目標は小さなfixed recordと既存ownerへの集約である。互換不要の置換対象には旧新二経路を残さない。
frozen dataclassだけでは内部のMapping/listは不変にならない。record確定時にtuple・深い不変値・canonical JSON bytesへ正規化し、raw mutable入力へのaliasを残さない。説明/JSON用に返すものは独立projectionとし、immutable runtime/contextを表示編集で変更できる形へ戻さない。

## 7. 入力と完成物公開をownerで閉じる

### 7.1 — sourceの設定とCSV構造を、値のconsumerへ渡す前に解決する（R06の拡張）

**確認済みbug。優先A。** 所有位置は`source_builder/producer.py`の`read_csv_source:309`、`_select_rows:203`、`_matches:214`、`_configuration:226`、`_integer_column:187`、`_csv_values:197`。`read_csv_source`は設定解決前に`config.get('filter_equals', {})`を使い、`_source_from_rows:322`が後から`_configuration`を呼ぶ。その検査も`columns`/`units`をmappingだと仮定し、返り値が`None`なので解決された事実が残らない。`Any`がvelocity/column/property consumerまで届く。

`_select_rows`はunique headerを検査するが`DictReader`のrow幅を検査しない。余剰cellが`None` keyへ入っても、後段が必要列だけ読むため成功する。canonical writerは既に落としたcellを検出できない。逆に欠損列・不正integer tokenはKeyError/TypeError/decimal.InvalidOperationとしてCLI境界を漏れる。CLIの許容例外一覧が短いことが根因ではない。

**提案する責務と型（例示signatureであり現行APIの記述ではない）。** source owner内にsmall frozen recordを置き、raw configurationを一度解決する。汎用CSV frameworkや全producer共通schemaは作らない。

```python
@dataclass(frozen=True, slots=True)
class SourceCSVSpecification:
    table: str
    kind: Literal['table', 'surface']
    columns: SourceColumnMapping   # scalar列名／position pairを解決済み
    units: SourceUnits             # 使用するSI scaleを解決済み
    velocity: VelocityInput        # modeに必要な列数、direction、frame/basisを確定
    filter_equals: tuple[ColumnPredicate, ...]
    inherit_from: SourceInheritance | None
    metadata: SourcePreparationClaim

def parse_source_configuration(raw: Mapping[str, object]) -> SourceCSVSpecification: ...
def source_from_csv_bytes(raw: bytes, data: DataBundle,
                          spec: SourceCSVSpecification) -> RealizedSource: ...
def read_csv_source(path, data, config: Mapping[str, object]) -> RealizedSource:
    # 既存公開shapeを保ち、ownerのparseとbytes入口へ進む
    ...
```

`columns`は各canonical量のrequired scalar/pairを解決し、unitは対応する量のscaleへ変換する。modeごとのvelocity列・direction列・component順を解決する。物理finite/shape/frame/basisの数値検査は既存convert_velocityの所有を保つ。source parserが同じ基底検査を複製しない。table/surfaceのrequired位置、dimensionless ID、metadata/inheritanceの意味をここで確定する。

CSV readerはUTF8/unique header→必要なmapped/filter列の存在→全rowのheaderと同じ幅→filter→選択rowのexact identity/物理floatの順で処理する。無関係な**正しくheaderに宣言された列**やfilterで除外される行の科学値へ不要なcanonical要求を課さない。ただしrow幅やquotingは表構造の問題なので選択前に拒否する。DecimalのConversionSyntax/InvalidOperationはinteger lexical conversionの行・列ownerでValueErrorへ変える。`except Exception`をCLIへ加えない。

`inputs._recipe_source:132`はCSVを今のようにsnapshot→公開readerの再読込→再比較せず、一度読んだbytesを同じsource ownerのbytes入口へ渡す。これによりCSV解釈とCSV hashが同じbytesにbindする。config pathも一度読んだrawからparseとhashを得る。公開Pythonの`read_csv_source`はpath wrapperとして一つの実装を消費する。全bundle設定parserを統合しない。

**変更先と撤去。** `_configuration()->None`と`Mapping[str, Any]`consumer、二重read/after-the-fact比較を置換する。`inputs._recipe_source`、examplesのeditable設定generator、direct source users/release_workflow、source_builder README、case recipe説明、source verification/input/CLI scenariosを同じchangeで更新する。dict→recordのためだけに新しいclient dialectは作らない。format2の科学意味は同じで、合法入力のsource列・release時刻・weight・mass・速度・hash意味を維持する。malformed入力を拒否する範囲の変更は明記する。

**失敗・identity。** parse/CSV失敗はValueErrorのfield/row/columnを含み、CLI exit2、case destinationなし。canonical writerとcheck_caseの拒否は既存publisherが担う。recipe receiptのCSV/config hashと解釈が同じsnapshotにbindする。継承は明示table×exact IDだけ。許されている「継承するproperty columnの不在」を一般欠損エラーに変えない。

**公開受入。** 上記five subcases＋short row/duplicate header/invalid column shape/invalid filter値を`create --json`でexit2/有用なfield名/tracebackなし/未公開と確認する。一つのparameterized原因回帰群にまとめる。valid internal/surface、新IDで全物性明示、既存IDで明示inherit、filterで初期rowを選ぶflow、`2^53+1`のexact particle IDを保持する既存能力を残す。private helper名を固定するtestにしない。

### 7.2 — selected場CSVのidentityをfloat64へ落とす前に束縛する（R15：外部reader修正とは別の現行bug）

**新規確認済みbug。優先A。** `csv_import/workflow.py:_read_fields:884`は`_numeric_csv_matrix:1037`で全cellをfloatへ変換し、`_match_field_rows:968`でnode IDを検査する。`identifiers`が整数・finiteか、`abs > 2**53`かを検査するのは既に丸めた後である。`9007199254740993`は`9007199254740992.0`となり、`>`条件を通る。exact mesh ID側は`_csv_int:1318`のPython intを保っているため、異なる外部IDを「正しいscoped_node_id bijection」と呼んで公開できる。

今回の4節点/2三角形probeは座標を同一にして、mesh第1vertex ID=`9007199254740992`、そのfield rowのraw node IDだけ`9007199254740993`とした。公開CLIはexit0、maximum coordinate mismatch0、`field_join_method: scoped_node_id`、canonical geometryに前者のIDを保存した。異なるfield値やmeshを誤計算した実証ではないが、外部同定の誤結合そのものは成功経路で実証した。raw source hashが正しいことは意味の正しい結合を補わない。

**型と所有提案。** 同じfield-table readerがidentity列をexact integerとして解決し、physical valuesだけfloat64へ変える。CSV source readerと同じ巨大validatorを作らない。

```python
@dataclass(frozen=True, slots=True)
class ProviderFieldTable:
    values: FloatArray                  # physical/scoping数値列、node ID列を含めない
    value_column_index: Mapping[str, int]
    node_external_id: Int64Array | None  # scope選択後のIDをlexicalにexact解決
    provider_row_count: int             # 選択前の全provider row数

def read_provider_field_table(raw: bytes, columns: ColumnDictionary,
                              specification: FieldSpecification,
                              domain_external_id: int) -> ProviderFieldTable: ...
def match_field_rows(mesh: MeshResult, table: ProviderFieldTable,
                     coordinates_m: FloatArray, ...) -> RowPermutation: ...
```

readerは選択domain IDと宣言scopeから先にrow maskを解決し、選択rowのIDをexact変換して、既にscope選択済みのProviderFieldTableを返す。選択外の許可されたNaN IDをInt64Arrayへ詰めることはしない。physical valuesとIDに同じmask、join時に同じpermutationを適用し、identityを値matrixへ重複保存しない。**本設計ではfield外部IDを既存mesh/canonicalのint64 authorityへ揃える。** 物理的制約ではないfloat都合の2^53上限と、floatからexact性を推定する検査を撤去する。正確に一致するint64 IDを受理し、小数/非有限/int64範囲外tokenはownerでValueErrorにする。外部node IDの符号をparticle IDと混同せず、既存mesh/canonicalの規則へ合わせる。

選択scope外の欠損を許している契約を保ち、scope maskをraw ID tokenと物理値へ同じように適用してからjoin対象IDをexactに解決する。node_id_columnなしのcoordinate-bijection入口も維持する。domain_scopeでidentity型を宣言する箇所は同じ原則で確認するが、未実証の別column丸めを既知bugとはしない。int64への受理範囲拡張もadapter revision/README/testsへ明記し、誤join拒否だけの変更とは称さない。

**変更先と撤去。** `_numeric_csv_matrix`のnode ID float経由と`_match_field_rows`の「floatからexact性を推定」する検査を撤去。`_read_fields/_select_field_rows/_field_column_indices`はtyped tableから必要列を受ける。provider CSV/COMSOL双方、selected nodal transfer tests、CSV importer READMEのID範囲・結合意味、source reportに波及する。geometry reader/Q1 split/axis回復の数学やcanonical schemaは変更しない。正しく受理していたケースの数値・ID・単位・基底を保持する。producer revision/provenanceを更新すればcanonical content hashは意図して変わり得るため、hash全体の不変は要求しない。adapter revisionは意味変更の程度に応じてownerで決定し、旧parser/互換aliasを残さない。

**失敗・公開受入。** 今回のraw ID不一致はexit2でbundle/report未公開。`2**53`境界の合法ID、正しいlexical小整数、duplicate/非integer/out-of-range ID、row orderの変更、node-IDなし座標bijection、selected domain外のNaNの意図した扱いを公開importで確認する。source hashが原bytesへbindする既存snapshotは維持する。既に修正済みの外部frozen-state/trajectory CSV読込みを未修正へ戻す必要はない。

### 7.3 — static電場builderへ渡すthermal入力をstaticなrecordとして解決する

**新規確認済みbug。優先A。** `electrostatic_builder/workflow.py:_require_field:487`はlayout/association/components/basis/unitだけ検査し、`time_s`を見ない。`_select_thermal_fields:456`は`temperature.values[:,0]`と`velocity.values[axis,0]`をstatic arrayと仮定する。canonicalは有効なlinear-time fieldを認めるため、canonical validationの成功はbuilderのstatic前提を証明しない。

`_read_builder_input:181`はcheck/build共通だが、未解決FieldDataをそのまま返す。速度がtime-dependentならcheck自体がIndexError。温度だけtime-dependentなら`check_from_configuration:157`は使用されないthermal値のshapeを無視してreadyを返せる。`_build:194`は小型solve後に`_generated_fields:286`へ`gas_temperature.values[:,0]`を渡し、`_ion_velocity:391`でbroadcastが壊れ、axis indexing:412がIndexErrorになる。probeはこのfalse-ready→build tracebackを同じ入力で確認した。完成bundleが誤った温度で成功したという主張はしない。

**提案signature。** builder owner内でstatic supportを明示して解決する。coreのgeneral RequiredFieldMetadataや新validator frameworkへbuilder政策を押し込まない。

```python
@dataclass(frozen=True, slots=True)
class StaticThermalInput:
    velocity_m_s: FloatArray     # [node,2], RZ
    temperature_K: FloatArray    # [node], finite/positive
    velocity_field_name: str
    temperature_field_name: str

@dataclass(frozen=True, slots=True)
class PreparedBuilderInput:
    data: DataBundle
    input_info: CaseFileInfo
    layout: P1TriLayout
    thermal: StaticThermalInput
    fixed_node_ids: Int64Array
    target_potential_V: FloatArray

def prepare_builder_input(spec: BuilderSpecification) -> PreparedBuilderInput: ...
```

既存`_read_builder_input`を置換し、selected fieldに`time_s is None`を要求してからstatic slicing/axis正則性を調べる。snapshotの暗黙選択、time平均、first knot fallback、static扱いのclampを追加しない。明示時刻でstatic fieldを外部作成したい利用は別の入力作成操作であり、今回のbuilderへ曖昧なtime selectorを追加しない。

checkはこの同じrecordから既存topology/anchor/numerical-control checkを呼び、solveせず小さいreportを返す。buildは同recordの確定形だけsolve/generated-fieldsへ渡す。現在のgas-only model、数学式、closure、GMRES、全材料非対応は維持する。

**変更・撤去・受入。** tuple7要素とconsumer側の`values[:,0]`再解釈を撤去する。`_read_builder_input/check_from_configuration/_build/_generated_fields`、builder READMEのstatic入力、electrostatic verification、`inputs electrostatic check/build` scenario、reduced-electrostatic editable exampleを更新する。velocity/temperatureそれぞれ、また両方のlinear-time入力をcheck/buildともexit2かつfield名付きで拒否し、solverを呼ぶ前/出力なしとする。既存static 4節点例はcheck→build→case create→particle runで既存数値payloadを維持する。shape bug修正を理由に新たな全材料/非定常電場機能を混ぜない。

### 7.4 — 単独analysis JSONも完成したfileだけ公開する（R07）

**確認済みbug。優先A。** `cli.py:_result_operation:140`→`_write_derived_json:223`。source result内への書込みと既存file上書きを拒否する方針は正しいが、公開targetを`open('x')`してからserialize/writeする。中断後も部分fileが残り、次の同じ操作もFileExistsErrorになる。分析summaryは同じで、publication責務だけ欠ける。

**提案signatureと操作単位。** source/result path制限はCLI/client ownerに残す。JSON encode完了→同一filesystem sibling temporaryへwrite/flush/fsync→atomic create-if-absentで公開する。既存`case_format.write:309`が既に`os.link(temporary,destination)`と非対応filesystemの明示失敗を採用している。これを手本に小さいJSON file writerを置く。coreにclient JSONをimportさせず、汎用transaction contextや第二publisher体系は作らない。

```python
def write_analysis_json(result_path: Path, destination: Path,
                         document: Mapping[str, object]) -> None: ...
```

所有はanalysis/derived-outputに置くか、単一利用のCLI helperをそのまま小さく修正する。wrapper層追加自体は改善ではない。POSIXの`rename/replace`で既存fileを上書きしてしまわないよう、完成fileのpublication primitiveはcreate-if-absentを保持する。hard link非対応ならOSErrorとして未公開で失敗し、direct-write fallbackを作らない。atomic visibilityとpower-loss後のdirectory durabilityは別保証であり、今回の既存契約へ後者を勝手に追加しない。

**変更先と受入。** 直接target `open('x')`を撤去し、CLI analyze output、support/errors/README、CLI scenarioだけを同じchangeで更新する。既存bundle/case/export/view stagingを汎用frameworkへ統一しない。JSON内容・analysis revision・source_identityは変えない。公開`main/analyze`でencode failure、1 write中断、publication conflict、success、retryを確認する。元result内拒否、既存target bytes保持、中断時targetなし、temporary掃除、成功JSONがstdoutと同じ科学summaryであることが受入。parse不能JSONを後から削除するrepair手順は追加しない。

## 8. 物理の選択・結線・証明・説明を一方向に構築する

### 8.1 model record：identity・role・scalar の一つの所有

`ModelDescriptor` は revision の静的な仕様を所有し続ける。実行用 offset や関数を catalog に入れない。
選択済み record は descriptor、caller の field binding、正規化済み科学 parameter を一つに保持する。
現行の category ごとの typed plan を廃止して dict に戻すのではなく、**typed plan 自体を選択済み record にする**。
`PhysicsPlan.selected_models` は stored な第二リストにせず、category slot の record を既定順に辿る read-only projection とする。

以下は concrete shape の例であり、公開 API の追加宣言ではない。

```python
@dataclass(frozen=True, slots=True)
class BoundFieldRole:
    specification: ModelFieldDescriptor  # role、SI、basis、constraint の唯一の宣言
    field_name: str                     # caller が選んだ canonical name

@dataclass(frozen=True, slots=True)
class EpsteinLinearModel:
    descriptor: ModelDescriptor         # 選択した revision identity
    gas_velocity: BoundFieldRole
    gas_density: BoundFieldRole
    gas_temperature: BoundFieldRole
    gas_mean_free_path: BoundFieldRole
    gas_molecular_mass_kg: float
    delta: float
    maximum_speed_ratio: float          # v1 の 0.1 も解決済みの値として持つ
    applicability: Literal["error"]

@dataclass(frozen=True, slots=True)
class PrescribedNoiseModel:
    descriptor: ModelDescriptor
    force_spectral_density: BoundFieldRole  # D [N^2*s], positive
    interval_tree_depth: int
    adaptive_max_depth: int                 # 省略時の解決値も保持
    particle_properties: ReferenceParticleProperties
```

他 model も今の有限な typed union の concrete record として表す。fixed charge の選択 identity は明示 record で保持するが、
stage の `G=J=0` への退化は同じ経路とする。typed plan から model/revision を推定し直さない。
全 role を同一 constructor で `BoundFieldRole` へ結び、`RequiredField` は field 名ごとに得る派生 projection とする。
同名で異なる unit/basis/constraint の要求は現行どおり拒否する。別名を同じ物理量と推測してまとめない。

正規化 scalar の説明は catalog の concrete record から一度だけ immutable document へ投影する。
parameter 名を `endswith('_field')` だけで分類する新しい汎用 serializer は作らない。
現行 `_descriptor()` は `_field` suffix から field parameter を求めているため、将来 `field` 以外の配列入力や
別の field key を追加する時は **明示 role 宣言が基点**になるよう整理する。
`describe_models()`、selector key 集合、role metadata、説明 parameter 集合を同じ revision 宣言から作り、
説明だけ後から `isinstance` で少数例外を足す形を終える。
型付き scalar record の全科学 member が説明へ投影されたかは、その model の実入力→check/result の意味テストで確認する。
モデル全種類の大量 snapshot や private dataclass field 名だけを固定するテストは作らない。

### 8.2 stage binding：共有配列と科学的一致を分離

現行 `runtime.py:4014` の field 名→offset は再利用する。field sampler の column layout と physics の必要 role が
一致する一つの immutable layout を prepare 内で構築し、各 model は自分の field role をそこへ直接 bind する。
同名 field は自然に同じ列を指す。Talbot binder に drag 型、lift binder に thermophoresis 型を渡さない。
同様に Barnes/aggregate-ion binder も自分の role を直接 bind する。

```python
def bind_talbot(model: TalbotModel, columns: PrimitiveColumns) -> TalbotStageBindings:
    # constructor は各 role の metadata・width に一致する列だけを返す
    return TalbotStageBindings(
        temperature_K=columns.scalar(model.gas_temperature),
        gradient_K_m=columns.vector2(model.gas_temperature_gradient),
        density_kg_m3=columns.scalar(model.gas_density),
        viscosity_Pa_s=columns.scalar(model.gas_dynamic_viscosity),
        conductivity_W_m_K=columns.scalar(model.gas_thermal_conductivity),
        mean_free_path_m=columns.scalar(model.gas_mean_free_path),
        particle_conductivity_W_m_K=model.particle_thermal_conductivity_W_m_K,
        thermal_slip=model.thermal_slip_coefficient,
        momentum_exchange=model.momentum_exchange_coefficient,
        thermal_exchange=model.thermal_exchange_coefficient,
    )
```

`PrimitiveColumns` は現在の小さな offset/width/constraint 表の型付き表現であり、locator、sampler、registry を増やす提案ではない。
native 境界では今と同じ offset integer／数値 scalar の `NamedTuple` へ落とす。
inactive family に `-1` を置くこと自体を廃止する必要はない。ただし active record の required offset が負でないことを
prepare の constructor で証明し、inactive offset をモデル欠落の代用にしない。

科学的一致の owner は catalog の composition に残す。neutral gas の role 名＋field 名＋molecular mass、
native/effective-gas 仮定、ion species/background、image ion drag と Coulomb の E、reference cohort の物性一致は
互いに異なる意味の規則である。同じ column を読む最適化の成功は、この科学的整合の証拠にしない。
逆に、科学的一致が確認済みなら offset 再利用のために相手 model の内部 binding を参照しない。

### 8.3 四つの数値 channel と noise の具体的な結線

現行 `PhysicsNumericOutputs` を維持し、その意味を次のとおり固定する。

| channel | 値／SI／意味 | 消費先と禁止する読み替え |
|---|---|---|
| linear relaxation | `gamma [1/s]`、`u [m/s]`、`a_drag=gamma*(u-v)` | exponential と OU の凍結係数。finite-speed では gamma は stage の v に依存し、物理的な大域線形性を意味しない |
| additive acceleration | `a_add [m/s^2]` | thermophoresis、ion、DEP、lift、electric、gravity の現行順。v/Z 依存でも additive channel であり、constant の証明ではない |
| continuous charge | `G=dZ/dt [1/s]`、`J=dG/dZ [1/s]`、finite invariant | 同じ stage の x/t/v/Z で運動と連成。charge-only subcycle、Z の平衡置換、後付けをしない |
| noise | `theta [m^2/s^2]`＋選択した noise law | joint OU の平均・共分散。摩擦 rate、additive acceleration、charge G/J と混同しない |

`a_total` と `a_add` は別々の現行加算を保存する。`a_total=gamma*(u-v)+a_add` と最後に作り直すだけでも丸めが変わり得る。
lift の速度依存・回転結合を scalar gamma へ移すことは数式変更なので、この整理に含めない。

現行 noise の結線は `compiled.PrescribedPhysicsBindings:93`、`runtime.py:4155,4269,4289`、
`transport.py:1179–1223` に分散している。`force_diffusion>=0` が prescribed/FDT の selector として使われる。
現行の closed prepare では合法だが、意味を明示するため次の固定 numeric subrecord を `compiled.py` に所有させる。

```python
class NoiseStageBindings(NamedTuple):
    code: int               # NONE / FDT_EPSTEIN_LINEAR / PRESCRIBED_FORCE_DIFFUSION
    temperature_K: int      # FDT だけ。drag.gas_temperature role と同じ column
    friction_kg_s: int      # prescribed だけ。drag.friction_coefficient role
    diffusion_N2_s: int     # prescribed だけ。noise.force_spectral_density role
```

FDT では catalog が native/effective-gas 線形 Epstein のみを受理する現行対応を保つ。
同じ stage の `T_g>0` と resident `mass_kg>0` から `theta=k_B*T_g/mass_kg`、drag から `gamma,u` を得る。
prescribed では `D [N^2*s]>0`、`beta [kg/s]>0`、resident m から、現在と同じ算術
`theta=(D/beta)/(2*m)`、`gamma=beta/m` を得る。D を温度と解釈しない。
noise の record は depth や RNG draw を所有しない。base/max depth は typed noise model→transport settings、
joint OU と conditional tree は stochastic、RNG addressing は rng、root の係数凍結と restart は transport/integrator が所有する。
NONE で theta を勝手に positive default にしない。新 noise code は必須 coefficient role と positivity を constructor で閉じる。

OU は現在の root midpoint で `gamma,u,theta,a_add,G,J` を確定してから draw する。
`J<=0`、fixed の `G=J=0`、同じ prepared charge invariant、axis/wall 後の fresh root を保つ。
parameter/binding 整理で一般 multiplicative-noise SDE の次数や連続 OU の exact first-passage を主張しない。

#### 現行の全 family を通す対応表

同じ row を持つ複数 revision も identity を併合しない。下表の「scalar」は説明へ残す正規化 parameter であり、
revision 固有の物理定数まで実行設定へ重複追加するという意味ではない。
role の厳密な unit/constraint/basis は catalog の既存 `ModelFieldDescriptor` が唯一の owner である。

| category／typed plan family | 自分で bind する field role | 正規化 scalar／独立物性 | stage と prepare/proof の対応 |
|---|---|---|---|
| fixed charge（現行 `None`＋selected descriptor） | なし | Z0 は source のみ | `CHARGE_FIXED`、G=J=0。charge invariant追加なし。constant family可 |
| stationary/shifted OML `PlasmaContinuousChargePlan` | ne,ni,Te[K],Ti[K],ui | mi、実際の drift limit、error policy、radius authority | 各 `CHARGE_OML_*`。OML invariant/G/J bound、Debye＋drift proof。shifted の negative-charge 制限は維持 |
| aggregate two/three current `AggregateRelativeDriftChargePlan` | ne,ni,Te[V],Ti[V],ui,mi-field,screening。three は n-/T-/u-/m- も | maximum relative speed、error policy、radius authority | 各 aggregate code。独立 charge invariant、G/J bounds。正負 population の speed proofを明示 |
| prescribed charge `PrescribedExponentialLinearChargePlan` | electron offset c、slope alpha、ion rate Ai、slope betaZ | Ae、reference properties | `CHARGE_PRESCRIBED_*`。G/J と forward invariant。追加速度regimeなし |
| linear Epstein native/effective `EpsteinDragPlan` | ug,rho,T,lambda | mg,delta,maximum speed ratio,error | `DRAG_EPSTEIN`。rate boundとEpstein norm proof。native/effectiveは科学仮定が異なる |
| finite Epstein `FiniteSpeedEpsteinDragPlan` | ug,rho,T,lambda | mg,diffuse fraction,maximum speed ratio,error | `DRAG_EPSTEIN_FINITE_SPEED`。G-rate と radial-Jacobian K の上界を分離。Epstein norm proof |
| Stokes–Cunningham `StokesCunninghamDragPlan` | ug,rho,mu,lambda | error policy。air相関定数はrevisionで固定 | `DRAG_STOKES_*`。rate上界、Kn/Re proof。FDT/OUへ自動拡張しない |
| prescribed drag `PrescribedLinearFrictionPlan` | ug,beta | reference properties | `DRAG_PRESCRIBED_LINEAR`。beta/mとtarget bound。追加regimeなし、finite velocityは要する |
| FDT noise `InertialLangevinNoisePlan` | selected dragのT role（追加fieldなし） | base/max depth | theta=kBT/m。linear Epsteinだけ。OU/tree/restartは数値owner |
| prescribed noise `PrescribedForceDiffusionNoisePlan` | D。selected prescribed dragのbeta | base/max depth,reference properties | theta=(D/beta)/(2m)。frictionとdiffusionの各正値を確認 |
| Waldmann native/effective `WaldmannGallisThermophoresisPlan` | ug,T,q_translational,lambda | mg,maximum speed ratio,error | `THERMOPHORESIS_WALDMANN_*`。q/T加速度上界、Kn＋drift proof。native/effective identityは維持 |
| Talbot `TalbotThermophoresisPlan` | T,gradT,rho,mu,k_g,lambda | k_particle,Cs,Cm,Ct,error | `THERMOPHORESIS_TALBOT`。別導出のcross-regime bound/static＋local proof |
| Barnes `BarnesCollisionlessIonDragPlan` | ne,ni,Te[K],Ti[K],ui,lambda_in | mi,maximum drift,error,radius authority | Barnes stage、charge invariantを使うglobal force bound、scale/Debye＋collisionality＋drift proof |
| relative-flow screened `RelativeFlowScreenedIonDragPlan` | ni,Ti[V],ui,mi-field,screening,lambda_in | maximum relative speed,error,radius authority | relative-flow stage、global＋local additive bound、relative-speed proof |
| image orbital `ElectricFieldDirectedImageIonDragPlan` | ni,Te[V],Ti[V],ui,mi-field,screening,E | error,radius authority | image stage、global force bound。追加速度regimeなし。CoulombとのE一致はcomposition |
| prescribed ion polynomial `PrescribedChargePolynomialIonDragPlan` | F0,F1,F2 | reference properties | polynomial stage。共通Z invariantでglobal bound、追加regimeなし |
| DEP `QuasistaticSphericalDielectrophoresisPlan` | grad(mean E²) | epsilon_r,real K_CM,max point-dipole radius | DEP stage、radius applicability/global bound。constant family可、Eそのものを代用しない |
| rarefied lift `RarefiedVorticityLiftPlan` | ug,rho,lambda,azimuthal omega | lift coefficient,error | RZ lift stage、velocity-coupling bound、Kn proof。scalar frictionへ移さない |
| Saffman `SaffmanLiftPlan` | ug,rho,mu,lambda,out-of-plane omega | error、選択座標 | XY/RZ code、coupling/shear bounds、Kn/slip/shear proof。orientationを保持 |
| prescribed lift `PrescribedRotationalLiftPlan` | ug,c_rotation | reference properties | rotation stage、coupling bound、追加regimeなし。RZ axis zero条件を保持 |
| Coulomb plain/cap `ElectricPlan` | E | cap revisionのみmaximum abs Z | electric stage、Z invariantを使うglobal＋local additive bound。capはこのforceだけ。constant family可 |
| gravity `GravityBuoyancyPlan` | rho | gravity vector。mとdisplaced volumeは別authority | gravity stage、density intervalのglobal bound。RZ radial gravity=0。constant family可 |

ここにある global proof の数式を単一 point evaluator へ押し込まない。
local additive tightening は現行では主に relative-flow ion と electric が対象で、他寄与は証明済みglobal baseに残る。
「local role が空」を、その力がゼロ／local proof済み／常に一定と読み替えない。
pointで追加regimeなしの family でも、fieldのfinite/sign、derived finite、共通state invariant、supportは引き続き必要である。

### 8.4 proof family を肯定的に構築する

現行は型付き bounds と `_prepare_*_applicability_context()` が不一致を拒否している。
`_complete_force_bound_coverage():2366` も全選択 force category に実際の bound があることを要求する。
これは有効な防壁で、削除しない。one model/category の現行制約下で category 集合による coverage は十分である。

残る危険は stage の family code と proof の family code を同一 integer として流し、
「残りはこの family」という row の default 分岐を使うこと。
現在の context constructor は plan/bounds の型を照合するが、非 NONE branch の code がその型に対応する既知 code かを
全て検査するわけではない。将来、stage code を追加して proof を足し忘れると、この仮定が破れる。

各 concrete model の prepare は **stage binding と、実際に構築できた proof の typed 組**を返す。
例えば `prepare_epstein(model, particles, ranges, columns)` は linear/finite の stage binding と
Epstein rate/Jacobian bound、Epstein norm applicability、local role を一組で返す。
stage で計算できたから proof も available とする constructor は作らない。

```python
@dataclass(frozen=True, slots=True)
class PreparedDrag:
    stage: DragStageBindings
    acceleration_bound: DragRateBound
    continuous: EpsteinSpeedProof | StokesReKnProof | NoExtraRegimeProof
    local: EpsteinLocalProof | StokesLocalProof | NoExtraRegimeProof

def seal_physics(plan, prepared_models) -> PreparedPhysics:
    # 全選択 force の実 bound、charge invariant、noise/drag の対応、
    # stage family と proof family、active offset、座標、finite を確認
    # 親の explanation と worker numeric records を同じ結果から投影
    ...
```

これは arbitrary model callback や proof plugin interface ではない。今の有限な union と数個の constructor で閉じる。
最終 native proof record は stage code と別の、closed な `PROOF_NONE/PROOF_EPSTEIN_SPEED/PROOF_STOKES_RE_KN/...`
code を持ち、既知 family を列挙する。未知値は prepare で拒否し、native の防御でも numerical/configuration failure にする。
未知 code を既存 proof へ落とさない。証明に必要な array を先に添字参照してから未知 code を拒否する順序も避ける。

同じ proof family の共有が妥当な例は明示して残す。

- linear と finite-speed Epstein は thermal speed 定義と限界値を prepare で解決すれば、同じ相対速度 norm gate を使える。
  finite-speed の加速度 rate 上界 G と安定性用 radial Jacobian 上界 `G+S*G'` は別の値である。
- stationary/shifted OML は半径/Debye と drift gate の family を共有できるが、charge law、invariant、drift limit の科学的意味は別である。
- two/three-current aggregate は正・負 ion の gate の有無を明示する。負イオン密度 zero の厳密退化は現行検証を維持する。
- prescribed や image-ion に追加の速度 regime gate がないことは明示的な `NoExtraRegimeProof` である。
  これは zero force でも、primitive/state の finite 検査不要でもない。

global、local、constant も次を区別する。

| 証拠 | 意味／owner | 不成立時 |
|---|---|---|
| global bound | 全 snapshot の補間 extrema と property/invariant から run 全域の係数・加速度を外向きに包絡 | 必須 bound がない選択は prepare failure |
| local applicability | proposal tube×実時間区間に含まれる primitive box と velocity/charge box 上の regime 証明 | 未証明と違反を区別。既存 certificate budget と fail-closed を維持 |
| local additive bound | local Z、ion-flow 等で global bound を狭める独立の interval 算術 | 任意の補助tighteningが未対応/非有限なら、その改善だけを捨てて既に有効な有限global boundを保持。実stageの非有限値・support外・適用域違反を隠さない |
| constant acceleration | constant canonical DOF、fixed state、全 contribution、座標条件の肯定的証明 | `None`＝利用不可。ゼロ寄与とは扱わず一般経路 |

`runtime._constant_acceleration():3878` の全 model coverage、fixed charge、XY、field constant 検査はそのまま必要。
全 node/snapshot の厳密 constant を、sample 一点の一致や tolerance へ緩めない。
local bound を使う前の predictor support 証明も numerical owner の要件で、point sample が有限なことでは代用しない。

### 8.5 production の scalar math authority と独立性

純粋な scalar law を family の owner に一つ持ち、NumPy入口と compiled stage はその law を呼ぶ。
`forces.py:1242` の finite-speed factor、`:2947,2955,2962` の electric/gravity/DEP factor は既存の足場である。
NumPy の shape/domain/例外と native の行別 status はそれぞれの入口に残せる。
law の定義・branch・係数・局所 regime metric を二重所有しない。

```python
@njit(inline="always", fastmath=False, error_model="numpy")
def epstein_linear_row(mass_kg, drag_diameter_m, vx, vy, ux, uy,
                       density_kg_m3, temperature_K, mean_free_path_m,
                       molecular_mass_kg, delta, maximum_speed_ratio):
    # 現行 compiled の算術順・finite判定・error意味を authority にする
    # return ax, ay, gamma, applicability, error
    ...
```

この body は `forces.py` に所有し、compiled の `_drag_row` は selection/offset/status 合成だけにする。
array evaluation は row body を NumPy buffer へ書く入口とする。prepare で body を呼ぶ必要がある場合は
既存の `CPUDispatcher.py_func` と同じ方法で Python body を使い、check に追加の巨大 JIT warmup を要求しない。
array evaluator の性能が必要なら同じ row body の compiled `*_into` loop を使う。粒子別 Python callback は入れない。
shape/共有 domain 検査を消して no-check native body を利用者へ露出しない。

`charge.py:802` と `compiled.py:1576` 等の重複も同じ方向で移すが、一回の変更で全数式をまとめて書き換えない。
Epstein→Stokes→熱泳動/lift→ion→charge のように、独立 oracle と runtime coupling の範囲が閉じる family 単位で行う。
body の移動と数式の改善・norm の変更・安定化を混ぜない。
現在 `np.linalg.norm`、sqrt(sum squares)、hypot の入口差や NumPy/math の丸め差があるため、
「共有化だから bitwise 不変」と無検証で主張しない。現行 hot row の算術を維持し、array側に差が出る場合は独立期待値、
最大差、適用域 boundary、under/overflow、revision の要否を記録する。

bound は point law の単純な呼出しにしない。monotonicity、interval enclosure、外向き丸め、state invariant は別の数学である。
scalar body を共有しても、`Decimal`、Maxwell velocity quadrature、impact-parameter quadrature、解析極限、
Green-kernel OU 積分、manufactured trajectory は独立の科学 oracle として残す。
同じ body 同士の parity は binding/型変換の検査であって科学的な独立検証ではない。

### 8.6 deterministic construction・cache・compile の責任

1. case load は syntax/hash/source reference を検査する。model selector の意味は catalog が一度解決する。
2. category 順、role 順、field name sort を固定し、同じ入力から同じ plan/column layout/説明順を作る。
3. source の `mass_kg`、drag diameter、electrostatic radius、displaced volume、Z0 をそのまま権威とする。
   複数 reference model の同次 cohort binding をそこで照合する。
4. fields が required metadata、axis regularity、time coverage、support、global/constant 証拠を準備する。
5. 各 concrete model prepare が stage/proof を構築し、composition が noise/drag、charge-force の対応を seal する。
6. 目標として、親は explanation/active revisions を `_RunFacts` へ保持し、worker は必要な numeric records/配列/settings と
   小さな packing metadata を受ける。descriptor、元 YAML、provenance を worker callback の実行条件にしない。
   これは現行で完了した境界ではなく、以下の caller 移行を要する。
7. transport は existing `@njit(cache=True, fastmath=False)` owner を呼ぶ。モデル組合せごとの関数生成、
   descriptor を closure へ capture する compiler、任意 callback registry は作らない。

Numba disk cache は実装/環境に対する性能上の cache であり、model identity や proof 成立の authority にしない。
固定 shape record を使い、数値 parameter の変更の度に新しい Python callback 型や overload を作らない。
spawn worker の process-local overload と disk cache の費用は現行性能測定で含める。
check が physics prepare を共有する事実は維持するが、worker/writer/粒子更新や別 model compilation campaign を始めない。

#### 親／worker 間の具体的な移行順序

現行 `PreparedTransport.physics:484` は full `PhysicsPlan` を保持し、同時に
`PreparedTransport.dynamics.runtime.plan` も同じ plan を保持する。
`execution.ProcessTeam._start():215` は prepared 全体を payload にするため、片方の field を消すだけでは descriptor は worker から除かれない。
したがって、親専用 clock/schedule 値を `_PreparedRun` へ移す第一段階と、physics seal 後の worker projection は別工程にする。
前者だけで「numeric-only worker が完成」とは呼ばない。

後者では新しい同等 runtime を並存させず、既存 `PhysicsRuntime` と `PreparedTransport` の field と caller を次の順で置き換える。

| 現在の caller／必要情報 | 最小の移行先 | 維持する意味 |
|---|---|---|
| `transport.py:4422,4682–4683,7941,7963` の `requires_stage_evaluation` | 既存 `prepared.capabilities.stage_evaluation` | exact origin、release/start時刻、stage evaluator有無。第二のhas-stage flagは増やさない |
| `transport.py:2096` と `:2176,2434,2599,2829,3262` の noise有無 | 既存 `prepared.capabilities.stochastic` | stochastic経路の存在と誤callerの拒否。model object全体は不要 |
| `transport.py:2179` の noise revision検査 | parentのcatalog/physics sealで受理revisionとnumeric noise codeの対応を一度確定 | 不支持revisionを通さない。workerでdescriptorを再解釈しない |
| `transport.py:2655,2923,3321` の base/max depth | `PreparedTransport.brownian_interval_tree_depth: int|None` と `brownian_adaptive_max_depth: int|None` の2つの具体scalar | base path精度、追加条件付きrefinement上限、capacityを別の値として保持。stochastic=true時はsealが有効integerを保証 |
| `_StageDynamics.__post_init__():1608` の `runtime.plan.noise` | `runtime.native_context.bindings.noise.code`（第6節の固定record） | Langevin workspaceの有無。field/stage borrowed bankの容量を維持 |
| `PhysicsRuntime.evaluate_batch():679`、`_pack_local_ranges():1605` の `plan.required_fields` | sealした同一column表由来の `primitive_columns: tuple[(name,offset,width), ...]` | dict入力のshape、順序、packed列、local proof必要列。別samplerや別required-field authorityを作らない |
| variance convenience API `runtime.py:597–625` の noise/drag型とfield名 | 同じ `NoiseStageBindings.code` とT/beta/D offset、上記packing表のoffset→name | parentで証明済みnoise/drag対応を使用し、必要fieldのshape/finite/signと同じ既存variance kernelを維持 |

packing表は第5節の一つの column layout を縮約した immutable tuple であり、descriptor や全 parameter record を複製しない。
現行 `PreparedFieldSet.numeric_context.component_offsets`／`FieldWorkspace.field_names` と同じ順序から作り、
prepare が一致を確認する。host convenience API のための field 名・width は必要な小 metadata として明示する。
noise variance convenience API は、noise の offset に該当するfield名だけをこの小表から引き、
FDTならT、prescribedならbeta/Dを現行と同じshapeで検査する。無関係な全primitive入力を新たに要求しない。
未知noise codeは拒否し、同じ `brownian_thermal_variance_into`／`prescribed_force_variance_into` を呼ぶ。

`_CurvedWavefront` 等のprepare時容量算定（例 `transport.py:1324`）も同じ解決済みmax depthを使う。
ここでは数値settings frameworkや汎用projection registryを追加せず、2つのdepth scalar、既存capabilities、
固定noise binding、既存runtimeのpacking表で閉じる。
全callerを移して公開/API convenienceの意味とworkspace bytesを確認してから、
`PreparedTransport.physics` と `PhysicsRuntime.plan` を同じ変更で削除する。
親の `_RunFacts` はplanが残るprepare中に一度作り、その後にworker graphをsealする。
stage/root/current時刻、snapshot値と時刻、noise policy、EventSpec、seed、integrator、必要capacityはworkerに残す。
この段階の受入は既存の科学payload、noise/OU係数、dict入力、local証明、slab/workers/resumeの意味とmemoryを含む。

### 8.7 source/result 形式への具体的な効果

この責務整理では canonical field/source 配列と入力 YAML の物理的意味を変えない。
source の物性は増やさず、異なる意味の粒径・質量を一つへまとめない。
説明は **現行 `resolved.physics_models[category]` の flat mapping を完成させる案**を採る。
nested な新 facts 形式や result schema 5 をこの変更の前提にしない。
現行の `dict[str, str|float]`／`_RunFacts.models` はこの案にほぼそのまま使え、
integer の depth を型へ明示するなら `str|int|float` とする。別の科学設定 serializer を output に作らない。

例えば、catalog の正規化済み record から次の説明を一度だけ作る。

```json
{
  "drag": {
    "model": "epstein_linear",
    "revision": "epstein_linear_v1",
    "gas_molecular_mass_kg": 6.6335209e-26,
    "delta": 1.2,
    "maximum_speed_ratio": 0.1,
    "applicability": "error",
    "gas_velocity_field": "neutral_flow",
    "gas_density_field": "neutral_density",
    "gas_temperature_field": "neutral_temperature",
    "gas_mean_free_path_field": "neutral_lambda"
  }
}
```

これは形式例の独立値であり、新規 case の default ではない。
元 role parameter 名を key にするため、caller に新しい role 命名体系を覚えさせない。
noise は `interval_tree_depth`／解決済み `adaptive_max_depth`、prescribed charge は解決済み電子 reference rate、
各 force は全科学 scalar と applicability parameter を含める。
reference model の物性は `particle_properties.mass_kg` 等の4個の規定 key に flatten する。
gravity vector だけは座標に合わせて `gravity_x_m_s2,gravity_y_m_s2` または `gravity_r_m_s2,gravity_z_m_s2` に投影する。
unit/components/basis/constraint は同じ descriptor と required-field projection から説明し、元 field 配列を保存しない。
通常 client の model discovery と result 説明はこの role metadata を消費し、別の unit/basis 定義を持たない。

check は `CheckedModel` に immutable `parameters: tuple[tuple[str, str|int|float], ...]` を追加し、
既存 category/model/revision の identity 表示を維持する。CLI の check JSON もその projection を使う。
JSON 化と再帰不変性は既存 output owner を使い、ResultView に mutable nested object を戻さない。
description と resume は同じ正規化 facts から必要部分を投影する。説明だけの表示と科学的 parameter identity は区別する。
全 field array、初期全 particle array、run 全局所 certificate を context へ重ねない。

input hash が既に resume を保護していることを保ち、結果の説明欠落を口実に第二の case parser を result reader に作らない。
この flat 拡充案は case schema、result schema 4、checkpoint配列 schema を維持し、catalog/runtime の説明 revision と
result algorithm revision を更新する。resume identity へ増えた科学 facts が入るため旧 checkpoint との無音互換を主張しない。
API check document の追加 field、normal client、reader、resume、documentation を同じ変更で更新し、互換 alias を作らない。
model 式が不変なら scientific model revision を無理由に変えない。
数式・gate・加算順・丸めが変わる場合は、その科学的意味に応じた revision と独立検証を別途要する。

R09 の派生関係は core model の推測にしない。declared derived group だけについて producer provenance に
`producer revision / upstream field+layout identity / normalized parameters / particle cohort authority / output field group+hash`
を小さく保存し、当該 producer の生成操作で閉じる。現在の `_full_force_fields():169` は I/Oなしの既存係数生成なので再分割だけは改良でない。
builder が変更した upstream と declared dependency が交わる場合に、明示的な再生成または未整合としての拒否を client handoff で行う。
依存を宣言していない prescribed field に E 由来と推測した invalidation を適用しない。
generic core に Vigus 固有の同次粒子条件・係数生成・依存 DAG 実行器を入れない。

## 9. accepted transitionと親/workerの責務を閉じる

### 現行のphaseとmethod: 現行の縦切りと、変えてはいけない境界

| phase | 現行ownerと入口 | commitしてよい事実 | 維持する境界 |
|---|---|---|---|
| prepare | `engine.py:_prepare:740`、`transport.py:resolve_transport_capabilities:382` | capability、field/support、constant specialization、work budget、容量 | checkとrunは同じprepare。worker/writerをcheckから起動しない |
| 親の時計 | `engine.py:run_simulation:248`、`time_grid.py:MacroGrid:16` | macro index/time、release/frame/probe cursor | field knotはmacroへ併合。観測・個別releaseは全粒子stepを分割しない |
| resident | `cpu.py:PersistentParticleState:515`、`ParticleState:567` | accepted x/v/Z、lifecycle、terminal time、両ordinal、exact origin、contact state、failure | 固定粒子IDとresident indexの対応を維持。active indexはwork選択でidentityではない |
| slab | `transport.py:_advance_particle_slab:2051` | 所有slabのaccepted数値状態だけ | serial/processは同じ関数。親の時計、writer、process管理を持たない |
| proposal | `integrators.py:StepProposal:487`、`state_at_times_into:568` | candidateと借用sample、独立のsupport/applicability/numerical verdict | evaluatorとscratchはaccepted stateではない。sample後にresidentを更新しない |
| event | `events.py:locate_exact_first_event_into:955`、`locate_curved_first_event_into:1515` | 最初のhit/axis/split/clearと局在budget、incident set | geometryは探索、boundaryはlaw、transportは受理と継続を担当 |
| prefix受理 | `transport.py:_evaluate_exact_targets:8634`、`_commit_curved_event_prefixes:6489` | hitまでの受理済みpiece、cell hint、replay、accepted count | 後続law失敗でもvalid prefixは消さない。tailのsupport失敗で先行hitを隠さない |
| response | `boundaries.py:resolve_boundary_responses_into:486`、`_resolve_boundary_response_row:1268` | post速度、outcome、primary facet、combined normal | cornerの候補集合とpriorityを積分法側で再実装しない |
| accepted impact | `transport.py:_commit_exact_wall_response:9725`、`_commit_curved_wall_row:7163`、`_commit_surface_release_response_row:7636` | resident/ordinal/lifecycle/material event | 現在ここが重複。method継続と分ける対象 |
| observation | `transport.py:_record_replay_rows:10275`、`_record_replay_jump:10388`、`engine.py:_finalize_replay_buffer:1416` | accepted pathからのframe/probe。event時刻は右連続 | output scheduleでdraw・production partitionを変えない |
| IPC | `execution.py:_ProcessEventSink._handoff:58`、`ProcessTeam._receive_handoff:374` | 同期消費完了後のACK | 1 workerにつき高々一handoff。worker側HDF5 writerを作らない |
| durable | `engine.py:_checkpoint_state:708`、`output.py:ResultWriter.commit_epoch:1067` | 全slab/output完了後のaccepted macro snapshot | intra-macro continuation/treeをcheckpointへ追加しない。LATESTは最後 |

#### methodごとの意味は同じではない

| 経路 | prefix/state評価と必要な継続 | 共通化してはいけない点 |
|---|---|---|
| linear/quadratic exact | `_build_step_proposal:4660`のexact specialization、`_evaluate_exact_targets:8634`でhitまで評価。反射後は`_set_exact_origin:9880`から残時間 | analytical originを丸め済みmacro endpointへ置換しない。exact stackはintervalごとのinteractionを持つ |
| RK4 | proposalの`rk4_dense`は不変Bernstein rootを持ち、観測とdense applicability certificateで使う。event-prefix自体は`_build_curved_event_prefix_batch:6650`から同methodで再計算 | dense制限、same-integrator短縮stage enclosure、accepted sequential pieceを同一物と呼ばない。RK4 prefixを単純なdense restrictionに変える案は数値method変更 |
| exponential midpoint | 実際のstageでdrag/force/G/Jを評価し、`exponential_midpoint_reintegrated`を短縮再計算。predictor/enclosureは独立 | frozen-start predictor、midpointのlocal適用域、連成chargeのaffine/J安定化をRK4の式へ統一しない |
| OU/Brownian | guarded coefficient root、joint x/v increment、base/adaptive conditional tree、有限depth Hermite leafを経て同じ`_advance_macro_proposal:4318`を使う | hit/axis後は既存root tailをそのまま継ぐのでなく、committed post-stateから新rootへ。候補細分は親incrementへ条件付け、独立noiseを引き直さない |

`_is_reintegrated_path:4890`は現状`rk4_dense/cubic_hermite/exponential_midpoint_reintegrated`をまとめている。
これは「全てのdense sampleが再積分」という意味ではない。`StepProposal.state_at_times_into:568`はRK4/Hermiteを
元の不変rootからsampleし、他のpathはprefix evaluationを使う。`_proposal_event_bounds:6131`はRK4のglobal short-step
support enclosureが成立した行に限って、より狭いdense Bernstein boundsをeventへ使用する。ここを名前だけで置換しない。

### 9.1 N01: accepted material impactの状態更新が三箇所に分散

**構造確認済み。R05の具体化。現在の合法入力で誤計算したとは未実証。**

- `transport.py:_commit_curved_wall_row:7163`はcanonical hit、resident x/v/Z、logical/physical ordinal、lifecycle、event、
  contact departure、wall count、wavefront、replay jump、OU restartを続けて更新する。
- `transport.py:_commit_exact_wall_response:9725`は同じmaterial更新の後にexact stack/current/originとreplayを更新する。
- `transport.py:_commit_surface_release_response_row:7636`もcanonical hit、resident、両ordinal、lifecycle、event、wall countを更新し、
  その後にsurface departureが確定しなければ別failureを記録する。

既存の共有点は`_canonical_hit_position:7873`、`_set_boundary_lifecycle:9850`、`_append_boundary_event:9958`、
`_record_replay_jump:10388`。したがって共通化の対象はこれらを繋ぐmaterial状態遷移であり、
first-hit locatorとboundary lawを合併することではない。

#### 現在の差の分類

| 差 | 判定 | 理由・保存する動作 |
|---|---|---|
| curvedのfinite-radiusはreintegrated endpointをraw centerに使い、state residualをbudgetへ含める | 必要 | `7163`のendpointとlocator位置の差を認証範囲へ入れる。exactは直接のevent位置を使える |
| exact originとexact stack、curved stack/current、OU restart | 必要 | 同じ物理応答の後でも数値再開のauthorityが異なる |
| contact departure: curvedはactiveならDEPARTURE、exact/surfaceは`_post_response_surface_state:7620` | 現在は対応するが保守リスク | helperはcurved capabilityかfinite material contactかを見て同じ規則を表す。shared commitから一度だけ呼べる |
| exact/curvedに両ordinal上限検査、surface-responseに同検査がない | 到達済みbugとはしない | initial surface responseは一度だけで初期releaseから来る。上限に近い反復通常hitの経路では既に検査する。共通ownerなら将来の新入口でも漏れない |
| curvedはterminalでもcurrent x/v/Zとinteractionを更新、exactはterminalならcurrent timeとstack終了のみ | 必要なscratch差 | terminal residentは同じ。終了したcontinuation scratchの定義を同一化する必要はない |
| accepted prefixのreplay/cell hint/accepted countがlawより先 | 必要 | policy/topology失敗時も既受理pieceは実在。wall応答成功まで延期するとfailureと観測の意味が変わる |
| surface post-departure失敗はmaterial eventの後にfailureを追加 | 必要 | 応答した物理eventと、その後の数値的継続不能を区別。wall eventを巻き戻したりwall失敗へ変換しない |

#### 提案する最小のsignatureとfield所有

`cpu.ParticleState`を既存のresident ownerとして借りる。新しいmutable state型やimpact ledgerは作らない。
slab入口から同じstateを渡せるようにした上で、transportに次の小さなhost関数を置く。

```python
def _commit_material_impact(
    prepared: PreparedTransport,
    particle: int,
    raw_hit: BoundaryHit,
    responses: BoundaryResponseBatch,
    response_row: int,
    velocity_pre_m_s: np.ndarray,
    charge_pre_number: float,
    resident: ParticleState,
    pending: _BoundaryEventBuffer,
    statistics: _EventStatistics,
) -> BoundaryHit:
    ...
```

`raw_hit`は既存型を使う。callerがmethod固有のaccepted prefix位置、finite radiusのresidual、incident set、局在budgetを
既に確定したものとして渡す。`responses`とrowは既存のbounded batchを借り、primary facet、effective normal、outcome、
post velocityを読み出す。新たに粒子ごとのresponse objectや配列copyを足さない。

共通関数は次だけを所有する。

1. 変更前に両ordinalの上限、必要なpending capacity、応答の内部整合を確認する。既存writer/geometry異常はrun-fatalのまま。
2. primary facetで既存canonical projectionを行い、局在residualを`max`で結合した`BoundaryHit`を作る。
3. resident x/v/Zをpost状態へ書く。material wallでZはprefixのZをそのまま使う。wall専用chargeモデルを暗黙追加しない。
4. logical ordinalとphysical wall ordinalを各一度進め、既存`_set_boundary_lifecycle`でactive/terminalを設定する。
5. 新logical ordinalでmaterial eventを一度appendし、contact stateを`_post_response_surface_state`で設定し、wall countを一度進める。
6. resolved `BoundaryHit`を返す。cell hint、accepted piece count、stack、origin、OU tree、親clockは変更しない。

これは一般transaction APIではない。allocation/IO例外時に全stateをrollbackするframeworkを作らず、
未commit epochは既存ResultWriter recoveryが捨てる。数値上のmaterial response成立とdurable commitを混同しない。

**replayの処理順は意図的にmethod継続側へ残す。** `_record_replay_jump`は既に一つの実装ownerであり、
「同じhelperを複数箇所で呼ぶこと」自体は重複authorityではない。surfaceではmaterial event後のdeparture failureを先に
決める必要があり、共通commitから無条件にactive replayを書けば中間active状態を作る。callを消すためのboolean方針引数や
callbackは足さない。最終的なfailure/terminal上書きでたまたま隠れることにも依存させない。

#### method固有continuationの具体形

現在のexact/curved関数はraw-hit作成、共通commit呼出し、次の固有再開だけにする。既存NamedTupleのworkspaceをそのまま使う。

```python
def _continue_exact_material_impact(
    wave: _ExactWavefront,
    row: int,
    particle: int,
    hit: BoundaryHit,
    resident: ParticleState,
    replay: _ReplayBuffer,
) -> None:
    ...

def _continue_curved_material_impact(
    wave: _CurvedWavefront,
    row: int,
    particle: int,
    hit: BoundaryHit,
    resident: ParticleState,
    replay: _ReplayBuffer,
) -> None:
    ...
```

- exact: current timeをhitへ。terminalならstack終了。activeならcurrent x/v/Z、現在stackのinteraction、exact originを更新し、
  同時刻replay jumpを書き、元のtargetへのresidualを既存loopで実行する。`departing_facet_id`やfinite contactの既存寿命は維持する。
- curved deterministic: current time/x/v/Z、interaction、departure certificateを更新。terminalならstack終了。
  activeならreplay jump、元targetまでの新しいsame-integrator proposalへ戻す。
- curved stochastic: 上記post-stateは共通だが`_schedule_stochastic_restarts:5845`でroot終端とrestart timeを登録し、
  `_queue_langevin_event_rows:3633`と`_queue_langevin_roots:3545`に返す。古いHermite tailやconditioned右子を反射させて再利用しない。
- surface: zero-time wall responseを共通commitしてから、現在のpost-release classifierの結果を検査する。
  継続不能なら既存failureを追加して終わる。継続可かつactiveならreplay jumpと既存のsurface→method初期化を使う。
  「surfaceを第四のintegrator」とする新dispatchは作らない。

関数を増やすこと自体が目標ではない。最初のchangeではexact/curved固有部分をcallerに残してもよい。
上のcontinuation署名は独立した再開規則を明記するもので、薄いwrapperだけになる場合は作らない。

#### accepted-prefix boundaryをまたぐ順序

```text
trial proposal
  → first-hit/axis/clear判定（lawなし）
  → method固有hit prefixの構成
  → numerical / support / applicability / localization residual判定
  → accepted prefixのcell hint・観測replay・accepted piece countを一度受理
  → periodic/material分類（periodicは別transition）
  → material incident setからcontact normal、physical wall ordinalでRNG、boundary response
  → 応答不能ならaccepted prefixを保持したparticle failure
  → 応答成功なら共通material commit
  → surface固有post-departure確認（必要な入口だけ）
  → terminal終了、またはmethod固有post-state/replay/残時間・OU restart
  → bounded event batch同期flush
```

clearでは`_commit_proposal_rows:5435`がresidentとcell hintをcommitし、event prefixでは`6489/8634`がprefixのhintとreplayを先に
受理する。新しい共通impactからfield cellを再locateしない。canonical wall projectionとprefix endpointの微差は既存budgetで扱い、
新しいnudgeやtoleranceを足さない。hitまでのchargeは同じcoupled prefixのZであり、wall探索後にchargeだけ進めない。

#### 統合しないtransition

- RZ axisは`_commit_curved_axis_wave:6715`、`_commit_exact_axis_rows:8795`。basis foldでありmaterial law、physical ordinal、wall RNGは不要。
- periodicは`_commit_curved_periodic_row:7044`、`_commit_exact_periodic_response:9567`。logical eventだけを進め、wall counter/physical ordinalを進めない。
- numerical failureは`_commit_failure_rows:9071`と既存failure owner。terminal wallのSTUCK/ESCAPED/HELDとFAILEDを共通statusにしない。
- source activationは`engine.py:_activate_releases:1386`。release eventはmaterial responseではない。
- geometryのfirst-hit、eventsのcorner同時性、boundariesのlaw priority/combined normalは各ownerを維持する。

### 9.2 N02: 親専用scheduleと説明値がworker payloadに残る

**構造確認済み。誤計算・現在のmemory limit超過は未実証。**

`transport.py:PreparedTransport:469`は`frame_times_s/probe_times_s/field_time_split_s/maximum_dt_over_tau/
maximum_dt_charge_lipschitz/boundary_rules/memory_plan`を持つ。
`execution.py:ProcessTeam._start:203`は`store.dumps((prepared, _process_state(...)))`でその全体を送る。
上記scheduleとstability説明値はengineのrun/facts/resume/manifestだけが読み、transportの粒子更新は読まない。
boundary ruleもtransportは`compiled_boundary_rules`を使い、Python rulesはengineの説明で使う。

`ProcessArrayStore`の共有化はndarrayだけである（`_ProcessArrayPickler:509`、`dumps:663`）。
時刻のPython tupleはworkerごとにpickle/unpickleされ、frame/probe総数に比例するmetadataが各workerへ複製される。
これは同じ時刻が実行中のreplay viewにも必要という意味ではない。各macroの観測時刻は既にreplay payloadとして渡る。
現行memory planは明示どおりsolver-owned numeric arraysの予測であり、Python/Numba全RSS上限とは主張していないため、
この観察を「planがRSS保証へ違反した」と言い換えない。

実際のspawn argumentの経路は`ProcessTeam._start:203`→residentを`_share_process_state:83`でshared viewへ置換→
hintをsharedへ置換→`store.dumps((prepared, _process_state(state)), private_arrays=...)`でscratch rootをprivate指定→
actual page reserveを反映したpreparedの最終payload→`context.Process(target=_process_particle_worker, args=(child, payload))`。
子の`_process_particle_worker:94`は`reader.loads(payload)`で同じprepared/stateを得る。
`_ProcessArrayPickler.persistent_id:518`はndarrayだけdescriptorへ変換し、tupleは通常のpickle payloadに残る。
readerの`_buffer:764`はprivate rootをcopy、shared rootをmappingとして解決する。共有化が観測tupleを削除する段階はない。

| field/事実 | transport/execution数値側でのread | 移動/保持 |
|---|---|---|
| `frame_times_s`、`probe_times_s` | prepared属性の数値readは0。親engineがcursor/slice/manifest検査に使用 | 親へ。workerはmacro replayに含まれる時刻だけ読む |
| `field_time_split_s` | prepared属性の数値readは0。親grid/resume/manifestで使用 | 親MacroGridへ |
| `maximum_dt_over_tau`、`maximum_dt_charge_lipschitz` | prepared属性の数値readは0。admissionはprepareのlocal値で既に実施 | 親factsの説明値へ。admission自体を削らない |
| Python `boundary_rules` | transport/execution数値readは0。worker lawはcompiled版 | 親facts/説明へ。`compiled_boundary_rules`は保持 |
| full `memory_plan` | **0ではない**。`_advance_particle_slab:2051`はstaging capacity、`_flush_boundary_event_wave:10046`はslab capacity。serial slab loopもcapacityを使用 | full planは親へ、必要capacityはtransportへ一度投影 |
| `time.dt_s` | `_initialize_surface_releases:7387`と`_commit_surface_release_responses:7494`がclassification intervalとして使用 | base dtをworkerに保持。親clockだけを理由に全time値を削らない |
| `solver.seed/integrator/event`、`work_policy` | root/wall RNG、method、tolerance、split/restart/interactions | workerに保持 |
| `physics.noise`のbase/adaptive depth、coefficient policy等 | `_advance_brownian_slab:2570`、tree/guard/restartで使用 | workerに保持 |
| `schedule.release_time_s`、proposal/current/root/stageの実時刻 | particle-local release短縮、field sample、coefficient guard、dense state | workerに保持 |
| `fields.numeric_context.time_s`、各snapshot値 | `fields.py:_prepare_numeric_field_context:624`のsnapshot列（`686`）、stage interpolation（`sample_prepared_fields_into:717`）、local time-range proof（`_local_field_snapshot_bounds_into:1126`）で使用 | **workerに必須**。親macro split tupleと異なる。全snapshot field knotsを親に隔離しない |

bounded serialization witnessは`payload_probe.py`と`payload_probe.json`に保存した。
公開C01をload/prepareし、実`ProcessArrayStore`とresident shared conversion/private-array指定を使用した。
同じpreparedのhost tupleだけを`dataclasses.replace`し、他の配列・workspace・caseは同じである。
frame/probe tupleを変えたcandidateはserialization測定専用であり、変更後scheduleの合法runを認定するものではない。
子processの起動、粒子advance、writer作成は行っていない。

| 変更したtuple | 時刻数 | payload bytes | shared-array bytes | private-template logical bytes |
|---|---:|---:|---:|---:|
| 両方空 | 0 | 21,905 | 950,272 | 315 |
| frameのみ | 64 | 22,483 | 950,272 | 315 |
| frameのみ | 4096 | 58,771 | 950,272 | 315 |
| probeのみ | 64 | 22,483 | 950,272 | 315 |
| probeのみ | 4096 | 58,771 | 950,272 | 315 |

4096時刻で+36,866 bytesという差はserialized metadataの増分であり、実child RSSやmemory cap超過の測定値ではない。
initial probeは`tests`のimport path不足で終了し、script入口へsolver rootを明示後に上記測定が成功した。

**この段階でworker payload全体がnumeric-onlyになったとは呼ばない。** `PreparedTransport.physics`だけでなく
`PreparedTransport.dynamics.runtime.plan`も現在の`PhysicsPlan`を参照する。
transportは`requires_stage_evaluation`とnoiseのmodel/depth/coefficient/restart設定を使い、
`transport.py:_StageDynamics.__post_init__`内の`1608`もworkspace準備に`runtime.plan.noise`を読む。
さらに現行runtimeのmapping入口は`physics/runtime.py:597/609/610/623/625`でnoise/dragを、
`:679/1605`でrequired_fieldsを読む。これらの中にはprepare/API向け入口も含まれるため、
workerのhot loopで使わないという理由だけでruntimeの公開・参照用途まで削除しない。
N02の第一changeは上表でzero numerical readを確認した親専用値だけを移す。
その後に物理側のseal projectionを行うなら、prepared.physicsとruntime.planの両経路を追い、
必要なnoise/method control、sample binding、proof arrays、field role/constraintをそれぞれのownerから渡し、
host APIのmapping入口には必要なplanを残すか同じ意味の明示projectionへ置換する。
この残存確認とcall-site置換は別の実装段階であり、full planを空のstubにして成立させない。

#### 親とworkerの具体的な配置案

新しい設定frameworkを作らず、既存`engine._PreparedRun:240`を拡張する。

```python
@dataclass(frozen=True, slots=True)
class _PreparedRun:
    case: SimulationCase
    transport: PreparedTransport
    facts: _RunFacts
    macro_grid: MacroGrid
    frame_times_s: tuple[float, ...]
    probe_times_s: tuple[float, ...]
    memory_plan: CpuMemoryPlan
```

- field splitは`macro_grid.snapshot_splits_s`からresume/manifestへ投影し、別tupleの正本を置かない。
- stability boundsとresolved boundary説明は`_RunFacts`へ（物理metadata担当のR01と同じprojectionに合わせる）。
  Python rulesの保持がprepare後に不要なら正規化description作成後に解放し、compiled rulesだけtransportに残す。
- `PreparedTransport.memory_plan`を除き、実際に数値loopが必要な`slab_capacity: int`と
  `event_staging_capacity: int`だけを置く。既存`event_candidate_capacity`はそのまま。
  capacityはplanから一度決め、run中に二つの値を変更し合うAPIを作らない。
- transportに残す`time`は実際にsurface classificationで使う`base_dt_s`だけへ縮められるが、
  このscalar整理のために新settings classを作る必要はない。seed/integrator/EventSpecはtypedな数値入力として必要。
- `_ProposalReplayWorkspace.allocate:618`は全scheduleでなく`observations: bool`を受ける。
  現在も全tupleから使っているのは空か否かだけであり、scratch layoutは変えない。

`ProcessTeam`には親側資源planを明示する。

```python
ProcessTeam(
    prepared: PreparedTransport,
    state: ParticleState,
    active_index: ActiveIndex,
    event_statistics: _EventStatistics,
    memory_plan: CpuMemoryPlan,
)
```

`team.memory_plan`が実mapping page paddingを反映したplanを返す。workerへ送る`prepared`にはfull planを戻さない。
現在`_start`がpage reserve後にpreparedをreplaceして再serializeしている部分は、親plan更新と一回のworker payload確定へ整理できる。
ただし最初のserializationでしか分からないroot/page数の確認は残す。共有root登録を二重にしないためのmetadata計測と最終payloadの
生成は区別し、単に`dumps`回数を減らすことを性能目標にしない。

変更callersは`engine._prepare/run_simulation/_validate_checkpoint_cursors/_validate_completed_run_state/_resume_identity/_manifest`、
`api.py:check_case`内のcheck report投影（現行`96/118`）、`execution.ProcessTeam`のstart/advance/replay capacity、
`transport._advance_active_slabs:1991/_advance_particle_slab:2051/_flush_boundary_event_wave:10046`。
全workerが使わない親入力は親に寄せるが、field support/proofやnoise planなど実際のtransport必須値まで削らない。

### 9.3 N03: resident宣言を入口まで通す

**保守上の接続課題。既存state再統一のやり直しではない。**

`cpu.py:515/567`でpersistent/resident列は既に単一宣言となり、checkpoint projectionもそこから得る。
一方、`execution.py:_process_worker_slab:122`、`ProcessTeam.advance:280`、
`transport.py:_advance_active_slabs:1991/_advance_particle_slab:2051`は同じ13本の列を再列挙している。
新しいstate追加時に「型に追加したがworker callerで渡さなかった」を作りやすい。現行で漏れている列は見つけていない。

N01の足場として、次の入口へ縮める。

```python
def _advance_particle_slab(
    prepared: PreparedTransport,
    particle_index: np.ndarray,
    macro_interval: int,
    macro_start_s: float,
    macro_end_s: float,
    resident: ParticleState,
    replay: _ReplayBuffer,
    writer: _EventSink,
) -> _SlabResult:
    ...
```

これはresident SoAのownerを通す変更であり、slabごとのstate copyや`ParticleState`再allocationを意味しない。
親の`_RunState`をworkerへ渡さず、現行`_process_state:90`が作るborrowed`ParticleState`を使う。
低位Numba kernelには必要な配列だけを渡す現行方法を維持し、jitclassや一般state-access callbackは導入しない。
全10k行の引数を一度に変えず、入口と共通impactへ至るcallerを小さいchangeで通す。callerが一度arrayへ展開する段階を
長く残し、二つのstate APIとして常設することはしない。

### 9.4 N04: pathの名前と証明phaseを混同しない

**誤計算の指摘ではない。将来変更時の解釈リスク。**

`transport.py:_is_reintegrated_path:4890`と`_advance_curved_events:5549`の説明は、RK4 dense・Hermite・
exponentialを単にreintegratedと総称する。実際には以下は独立した事実である。

1. field/force/G/Jを評価する実stage。
2. candidate endpointのmethod。
3. 不変dense path上の観測・局所applicability certificate。
4. eventのgeometry enclosureとfirst-hit候補。
5. hitまでのsame-method prefix再構成、またはOU/Hermite restriction。
6. post-eventから残時間を再実行するcontinuation。

R05実装時はhelper名とdocstringを実際の担当に合わせ、例えば`_is_curved_event_path(proposal)`のような内部名へ置き換える。
新path enumや公開設定を足さず、wireの`path_kind`、既存algorithm revisionを無意味に変えない。
pathの数理自体を変える場合は別changeで、独立reference、収束、model/proposal/event revisionを評価する。

## 10. 結果の内容identityと再利用契約を具体化する（R08）

### 10.1 現在の保証と、変更する理由

output.ResultView.source_identity()はmanifestの正規化hash、case/data identity、context bindingを返す。
contextとcheckpointには内容hashがあり、読込みはschema/shape/値域/件数/時刻を検査する。
一方finalの有限なmodel_weight改変や、schemaを満たすsegment数値変更まで同一性へ束縛していない。
「通常計算が自然に壊れる」「resume保護がない」という指摘ではない。
問題は、派生集計が同じsource identityを掲げても、同じ数値内容を使ったとまでは説明できないことである。

ResultWriter.commit_epochはsegmentをhashしてcheckpointへ保存するが、checkpointはA/B二世代である。
これを全過去segmentの完全性保証と読んではならない。completed openも全segmentのstructure/countを検査するが、
全payloadのdigest照合ではない。この差をoutputの一ownerで解決し、analysis/export独自の別result identityは作らない。

### 10.2 提案する保証の単位

| 対象 | 意味 | 生成・検証owner |
|---|---|---|
| input identity | 同じcanonical問題と設定を使ったか | case/case_format、既存case/data hash |
| run facts | どのmodel/revision/数値規則と解決値を使ったか | prepare→_RunFacts、既存manifest/resume projection |
| payload binding | どの完成context/final/segment bytesを指すか | outputのcompletion |
| verified payload scope | 今回実際に読んだfileがそのbindingと一致するか | output reader。未読fileまで検証済みとしない |
| scientific equivalence | 同じ方程式・精度・統計目的に対応するか | 独立verification/V&V。hashだけで認定しない |

まずidentityの対象を明示し、そのうえでcompletionへschema既知のpayload rosterをbindする案を採る。
概念的にはrelative path、artifact kind、byte size、SHA-256をcontext/final/各committed segmentについて持つ。
roster自体をcanonical JSONとしてmanifestにbindする。raw file hashはrepackで変わる。
これは厳密な入力成果物の同定であり、同じ科学値のHDF5 re-layoutを不正な物理と呼ぶものではない。
新たなlogical numerical hash方式を同時発明しない。科学payload equalityは既存の列・順序を使う別検証として保つ。

### 10.3 書込み・読込み・resume

1. segment commitとLATESTの順、macro barrier、A/B checkpointを維持する。
2. finalizeでfinalをvalidate/syncしてから、closed context/final/segmentsのbindingを確定する。
3. 現在は全過去digestをdurableに保持していないので、初期実装はfinalize時に全payloadをblock読みしrosterを作る。
   O(output bytes)の追加I/OとO(segment count)の小metadataを測る。未計上の大配列は作らない。
4. complete manifestと_SUCCESSはその後に公開する。hash作成失敗をcompleteへ丸めない。
5. open_resultはrosterのshape/relative path/coverageとmanifestを検査する。巨大な全payload再hashを毎回暗黙実行しない。
6. read_final、iter_frames、iter_boundary_event_batches等が消費するfileは、そのfileのdigestを検証してから値を渡す。
   hashしたpathを閉じて無条件に別fileを再openしてはならない。検証とHDF5読込みを同じ開いたfile instanceへ結ぶか、再open時にinstanceと期待digestを再確認する。path名だけを検証cacheのkeyにしない。
   一ResultView内の検証結果は期待digestと検証したfile instanceへ束縛する。file差替えを注入した受入も行う。外部が同じfileを並行in-place変更する場合の安全性・真正性まで保証する新locking frameworkは作らない。
7. source_identityは期待されるpayload bindingを含む。検証した範囲と全体の期待identityを混同しない。
   通常analysisはfinal、events、seriesの必要fileを既存経路で読み、その同定を派生receiptへ渡す。
8. partial recoveryはcompleted rosterを装わない。LATESTのcommitted prefix、checkpoint/最新segmentの既存検証を維持する。
   完成時にprefix全体をroster化する。未commit tailは引き続き捨てる。

これは設計上の採用候補で、追加I/Oの許容は実測で決める。もし実用途がfinalのみならfinal binding先行の小変更は可能だが、
その段階を「全event/frameの内容完全性も完了」としない。全targetを満たすまでB09の残作業として保持する。
append-only digest indexによる高速化は、finalize scanが実測で支配的な場合だけ検討する。
新しい汎用Merkle frameworkやreaderの大量検証flagは先行導入しない。

### 10.4 影響・撤去・受入

outputのwriter.finalize/open/read/source_identity、result形式文書、normal clientのreceipt、V&V/result reader tests、
resume/publication testsを同じ変更で更新する。永続必須bindingとreader受理規則が変わるので、
result/schema又はalgorithm revisionをownerで決定し、旧readerやmigration aliasを併存させない。
model式が変わらなければ物理revisionまで変更しない。

**公開例外の受渡しも同じchangeで閉じる。** 現行api.open_resultの例外翻訳はopen中だけで、ResultViewの後のread/iterator失敗まで包まない。遅延digest検証が私有ResultOpenErrorを漏らす設計にしない。三つの公開例外の定義だけを小さい低位ownerであるsrc/chamber_particles/errors.pyへ移し、api/package rootはその同じclassを使う。outputはapiを逆importせず、ResultViewのpublic read/iter境界で内部のI/O/schema/digest errorをSimulationErrorへ翻訳する。generatorは作成時だけでなくiteration中の例外を包む。第二ResultView/facadeやRuntimeError catch-allは作らない。
analyze/export/viewの破損payloadはpublic SimulationError、CLI exit2、tracebackなし、部分成果物なしとし、元々の不正なmethod引数は適切なValueErrorとして区別する。core四操作と公開例外三分類は増やさない。

受入は正常final/events/frameの値と科学identity維持、有限値だけの改変、過去segment改変、file差替え/欠損、
roster重複/path逸脱、hash中断、publication failure、partial/resumeで行う。
completionとreadのI/O時間、peak memory、dataが多い時の費用を同じcaseで測る。
structure検査をhashへ全部置換しない。hash一致でもそのfileが有効なschema/意味を持つかは別に必要である。

## 11. producerの派生関係と更新を具体化する

### 11.1 — 宣言した派生関係のfreshnessをproducer操作に限って所有する（R09）

**現行の明示制限／構造。優先B、利用次第。** `electrostatic_builder/workflow.py:_build:194`は新しいE/plasma群を加え、元field/sourceを保持する。`_GENERATED_FIELD_NAMES:44`の衝突は拒否するが、任意名の古いprescribed係数を検出・更新しない。READMEの「dependent coefficientsを再生成する」は現行能力の正直な制限である。coreは独立prescribed fieldも許すため、任意の名前から依存を推測する修正は誤りになる。Vigus係数generatorは既にI/O-free `_full_force_fields`であり、分割するだけでは改善にならない。

**提案。** 新しいgeneration frameworkを作らず、既存producer provenanceに「明示的にこのprimitive/parameters/particle-property authorityから生成した出力群」という小さなrecordを持たせる。既存場cacheの`numerics.field_sha256:769`は値/time/metadataとlayout**名**をbindし、`layout_summary:789`が別にlayout内容hashをbindしている。field hashだけでlayout内容も同定できるとは扱わず、再利用可能な同定ownerを調べて両者を明示的に組み合わせる。hash規則をproducerごとに発明しない。layout・basis・timeとprimitive identity、parameters、heterogeneous/same-property条件、出力field集合を一つのgeneration recordへbindする。

```python
@dataclass(frozen=True, slots=True)
class DerivedFieldGeneration:
    producer_revision: str
    primitive_bindings: tuple[FieldBinding, ...]
    parameter_document: Mapping[str, object]
    particle_property_binding: ParticlePropertyBinding
    output_bindings: tuple[FieldBinding, ...]

def regenerate_declared_coefficients(data: DataBundle, request: CoefficientRequest
                                     ) -> tuple[tuple[FieldData, ...], DerivedFieldGeneration]: ...
```

このsignatureはVigus等の**そのproducer固有operation**に留める。coreにnative/Vigus依存を入れない。background builderがprimitiveを置換するoperationでは、明示dependency recordのあるoutputだけ同時再生成又はstaleと拒否する。独立prescribed fieldは保持する。現在のelectrostatic builderは同名primitiveとの衝突を拒否するため、存在しない更新機能を前提に自動再生成を追加しない。まずこの記録を出すproducer/consumer一組を選び、実際のhandoffで検証する。

**失敗・受入。** declared derived groupのpartial再生成を公開せず、bundle+receipt一操作を既存publicationで完成させる。geometry/source/物性heterogeneityの前提が違えば拒否し、平均値やsingle-particle係数へ黙って縮約しない。節点再構成だけでなくoff-node差、ゼロE、unit/basis、全部のdependent outputs、primitive変更時のstale拒否、独立prescribed group保持、particle properties変更時の拒否を、そのproducer公開入口で確認する。旧係数生成route・手動名配線・重複hash規則・その専用設定/tests/docsは同時置換。新たなphysic modelや全材料電場解法はこの変更の目的でない。

## 12. reportの根拠・snapshot・buildを結線する

### 12.1 — reviewed projection→snapshot→buildの内容・source・claimを同じbytesへ束縛する

**入力受入のholeは確認済み。lineage/atomicityは構造上の限界。優先B。** `analysis/build_report_snapshot.py:_reviewed_input:386`はrowsがlist、sourceがdict、source.filesがtruthyであることしか要求しない。`rows=[17]`、`files=['missing-evidence.json']`が受理され、snapshotを返した。`check_documentation._reviewed_row_errors:98`は後からrow objectを検査するため、snapshot生成とauthoritativeな受入検査が離れている。これを架空の実測が現行reportへ混入した実証と混同しない。現行reviewed projectionsは過去source/current tiny run/未実施のscopeをかなり丁寧に明記している。

`build_snapshot:408`はoptional projectionを`_reviewed_input`で読んだ後に`_record(path)`で再読込hashする。producer、authority、catalog、optional inputのhashを残すが、one-snapshot bytesの所有がないので、readとhash間に編集されたprojectionはrowsとhashが別版になり得る。今回rootは新規合成projectionだけを関数境界で変更する制御probeを行い、rows=Aのままoptional hash=Bとなることを確認した。過去reportの誤りを実証したものではない。`main:475`は`args.output.write_text`でin-place更新し、途中writeで旧data.jsonを壊し得る。report buildが別操作なのは正当で、公式build manifestはsnapshot/HTML/runtime identityをbindするが、snapshot writerの中断や元projectionとの対応をそれだけで救わない。

**scopeを誤らない規則。** `source.files`にはexternal/native/archived/.artifactsの引用が意図的に含まれる。全参照fileが今のcheckoutに存在すること、現在sourceと同一であることを一律に要求しない。実行済みPASSは元sourceと入力/runの範囲に限定する。current authority hashesは「現在のdesign参照」、imported receipt hashesは「その証拠のbytes」、build hashesは「届けたartifact」の異なる事実である。現在のauthority hashを各queryへ追記しただけでhistorical科学結果をcurrent qualificationへ昇格させない。

**ownerと型の提案。** human reviewed projectionの最小document受入、projection snapshot、source bindingは`analysis/build_report_snapshot.py`が所有する。codeのowner変更だけで済む範囲は同fileに置く。protected runtimeにforkやprojection registryを追加しない。

```python
@dataclass(frozen=True, slots=True)
class ReviewedProjection:
    document_sha256: str
    query_documents: Mapping[str, ReviewedQuery]
    location: str  # root-relative local file又は外部reference

@dataclass(frozen=True, slots=True)
class EvidenceReference:
    location: str
    availability: Literal['bound_local', 'external_reference', 'historical_unavailable']
    sha256: str | None
    execution_source: str | None
    scope: str

def read_projection(path: Path, allowed_query_ids: frozenset[str]) -> ReviewedProjection:
    # one read: raw bytesをparseとhashの両方へ使う
    ...
def build_snapshot(existing: ExistingReport, projections: tuple[ReviewedProjection, ...],
                   design: DesignReferences, build_status: AuthoringStatus) -> SnapshotDocument: ...
def publish_snapshot(snapshot: SnapshotDocument, destination: Path) -> SnapshotBinding: ...
```

型は巨大な全row analytics schemaにしない。各queryは現在UIが使うrequired row fields/型だけ確認し、unknown optional row metadataを無意味に禁止しない。`unavailable`のvalueはnullとし、row objectとsource.files list-of-referenceをproducer受入で確認する。metricDefinitionsのcomponentIds/queryIds整合はreport consumerとの既存接続で確認する。同じrow validityを`check_documentation`にコピーして二重所有しない。そちらはproducerの最小公開検査を消費するか、snapshotに含むvalidated shapeを検査する既存責務へ狭める。単にtestを削って検査を弱めない。

**既存runtimeへのwire投影を明示する。** EvidenceReferenceはproducer内部の型であり、そのobjectをsource.filesへ直接入れない。現行source-provenance.jsのreviewedFilesはstring又はname/label/fileを読むため、locationだけのobjectは消える。source.filesには既存対応のlocation文字列又はlabel/hrefを投影する。構造化したavailability/hash/execution_source/scopeはsnapshot.reproducibility.reviewed_evidenceへ保持し、同じrecordを既存source.evidenceFlowのtitle/detailへ表示・copy可能な形で投影する。現行rendererはfilesをlabel/hrefへ正規化するので、files objectにhashを足しただけで可視になったと主張しない。この方法なら保護runtimeを変更せず、既存の参照/出典欄を保持できる。source previewとcopyの両方で値・hash・scopeが一致することを受入にする。

`bound_local`と宣言したreferenceだけexact bytes hashとlocationを検証する。missingなら失敗か、作成者が明示した`historical_unavailable`としてscopeを保持する。自動でPASS→unavailableへ書換えたり、外部引用へ架空hashを作ったりしない。projection自体のhashはone-read bytesに必須。current designで利用する`EXAMPLE_NAMES`等のfactはsource file identity又は計算値を含む既存snapshot catalog記録にbindする。例題source全量やsolver全ソースを各queryへ重複埋込みしない。

**snapshot→buildの操作単位。** 既存app identity/presentationを一度読み、projection bytes/current authority bytes/catalog projectionを確定し、query/claimを検査し、new snapshotを生成・serializeし、同directoryのtemporaryからatomic replaceして完成fileだけ公開する。snapshot更新は既存fileを意図して置換するため、B04のcreate-if-absentとは違うpolicyである。buildは公式toolをそのsnapshotへ実行し、build manifestのsnapshot digestと生成時bindingが一致することを確認する。snapshot書込み成功とbuild未実施を独立記録し、後者を前者へ巻き戻す疑似transactionは作らない。build失敗時のsource snapshotは完成物で保持できるが、dist/current presentationを新build成功と呼ばない。公開・sharingは別の明示authorization境界のまま。

**同時変更・撤去。** `_reviewed_input`と`_record(path)`の二重read、in-place `data.json` write、string内JSONだけに依存した重要bindingを置換。現在のten query IDs・app ID/title/presentation/filters/authoring revisionを維持する。snapshot tests、check_documentation、report README、reviewed projectionsのsource形、公式build/QAへ渡すbindingを同じchangeで更新する。protected file変更が必要かを具体的diffで判定し、ordinary authored/data更新だけで済むならprotected権限を要求しない。既存runtime verifier/build wiringを緩和しない。

**受入。** malformed row/nonlist source.files、unavailable nonnull、未知query、宣言したlocal hash不一致はsnapshot公開前拒否。external/historical-unavailable参照はscopeが保たれ、current qualificationへ昇格しない。projectionの読込み後に元fileが編集されても、完成snapshotのrowsとprojection hashは同じone-read bytesに対応する。write中断は以前のdata.json bytesを保持する。成功後のbuild manifestが同じsnapshot hashで、source inspectionの値・条件・referenceと図表が一致することを公式build/可視QAで確認する。新しいCI runner/frameworkを作らず、既存のsnapshot tests/docs gateと必要な公式buildを使う。

### 12.2 read/hashの新しい制御証拠

rootのprobeでは有効なquality projection Aを既存_reviewed_input経路で読んだ後、_recordがhashする直前に同じ新規fileだけをBへ変更した。
生成rowsはA_read_rows、記録されたoptional-input SHAはBのものだった。対策はretry回数や後付け照合の追加ではなく、parseとhashへ同じraw bytesを渡すことである。
既存codeをmemory内で差し替えた境界probeであり、production sourceや実report入力を変更していない。
[実測のbinding差](../solver/evidence/foundation_design_2026_11_23/report_read_hash_evidence.json)を保持し、過去のreportが競合で壊れたとは主張しない。

## 13. 検証側のinventoryも一つの事実を消費する

### 13.1 — installed例題inventoryを受入でも同じownerから消費する

**構造上の変更漏れ。優先B。** 現在五例はすべて対象に入っており、現行機能の欠落を実証したものではない。`examples.py:_EXAMPLES:43/EXAMPLE_NAMES:50`が唯一の実定義、CLI `choices`、`test_input_clients.py:20`、reportの`len(EXAMPLE_NAMES)`はこれを使う。一方`tests/release_workflow.py:383`は四名をliteral loopし、別途`_first_source_case:141`がxy-ballisticを使い、receipt:399はliteral `'five passed'`。六例目追加/一例削除でclean wheel受入が気づかず、receiptが古い宣言になる。

**設計。** test側はinstalled `EXAMPLE_NAMES`をimportし、固有の独立oracleを持つxy-ballistic flowと、その他の`name != 'xy-ballistic'`を動的inventoryから実行する。あるいは全inventory loopを一つにして、独立own-CSV oracleと重なる小例の二重実行が必要か費用で決める。run結果から`examples: {requested_names, executed_names, outcomes}`を作り、字句`'five passed'`を撤去する。全例の意味やoracleをproductionの「expected output」へ登録しない。例題inventoryはcodeを一つに、独立数学oracleはtest側に置く。

```python
def verify_installed_examples(root: Path, names: tuple[str, ...]) -> tuple[ExampleRun, ...]: ...
```

これはtest-only小関数でよい。production acceptance/plugin registryを導入しない。CLI一覧とinstalled workflow要求集合の一致は公開name集合で確認し、private `_EXAMPLES`構造を固定しない。readonly inventory消費にatomicityやschema migrationは不要。release receiptは終了したrunから構成し、失敗途中にcompleteを出さない。例題追加/削減とCLI/README/reportの数・clean wheel受入を同じchangeで更新する。Windows/Linux空cwd・checkout fixtureなしという既存配布gateを維持する。例題数のdynamicな一致が、物理oracleの十分性を証明したことにはしない。

## 14. 科学比較・数値説明・大規模集計を用途から分離する（R11/R12）

### 14.1 比較する量と主張を先に確定する

現行saved-population evaluatorの同ID・同window・RZ・computational countという限定は正当である。
その入口を全purposeへ強制したり、unpaired比較のために人工的なIDや欠けたv/Zを作らない。
将来の改良では、既存の記述比較を保ち、追加用途が必要な場合だけ次を分離する。

| 利用 | 必要な対応 | 有効な判断 |
|---|---|---|
| paired deterministic診断 | 同じjoint source、同じ作用/場、対応IDと意味、必要な独立resolution | 同一点のforce/G/J→一step→first hit→輸送差、数値資格は条件付き |
| same-cohort stochastic集団 | 同じsource集団・clock/window・壁/重みの意味。乱数path結合は不要 | source別空間分布、停止先/時刻、active曲線、速度/電荷、coverageとばらつき |
| unpaired ensemble | source分布/weight/rateの対応、独立実現単位、事前指標 | population observableとseed間不確かさ。個別path RMSを数値誤差にしない |
| native FE end-to-end | geometry/field transfer/回復を含む実際の双方 | 場生成/転送誤差を含む再現性。same-canonical-field比較と区別 |
| 実機予測 | 源rate/weight、表面capture則、装置条件と測定 | 独立実験に対するvalidation。数値stickやCOMSOL一致だけでは認定しない |

許容が指定されなければ差・coverage・限定された実用判断を報告する。
見た後の結果に合う閾値を作って確認的PASSにしない。
目的に適切に合う集団観測は、その範囲で肯定的に説明する。qualified parityがNOT_TESTEDでも全成果を無価値扱いしない。

### 14.2 streamingする範囲と、残る必要容量

_candidateは全framesをlist化し、states[T,N,5]、status[T,N]、observed[T,N]を作る。
N=10^6、T=126のstates一配列は5.04 GBというshape計算になる。これは実測scale結果ではない。
改良ではResultViewのframe iteratorとboundary event batchを消費し、frameごとのcounts/quantilesとterminal情報を集約する。

- 同じID集団の索引、release/terminal/coverageはO(N)。一frameとそのquantile scratchもO(N)は必要になり得る。
- 全historyを同時保持しないことでO(TN)を避ける。exact quantileまで定数memoryになるとは主張しない。
- boundary batchはO(B)。初回/terminal/repeated interactionを区別し、集計に必要なevent順の前提を明示する。
- native CSVは正規化済みtime/ID順を前提にstreamするか、既存変換ownerで順序を確定する。
  arbitrary unsorted native tableを無条件memory内へlist化する新fallbackは作らない。
- exact common stored clockの比較を保持する。表示のための補間を数値的なtrajectory対応へ昇格しない。
- physical weightがないnativeはcomputational countのまま。比較できるweightだけを両側で同じ意味にする。
- own-survivorの速度/電荷分布と共通active subsetのpaired差を別量とする。
- missing、disappeared_unknown、failed、pending、heldを分母とcoverageへ残し、escapeへの推定変換をしない。

小型caseでは旧dense計算をproduction fallbackとして残さず、verification外の一時独立集計又は手計算へ照合する。
現行summaryで十分な用途にunpaired ensemble frameworkを追加しない。
必要になった入口だけ既存external toolへ明示引数/別評価関数として実装し、coreへCOMSOL依存を入れない。

### 14.3 数値的な資格を説明できる条件

macro dt、field knot、particle-local root、certificate細分、Brownian base/adaptive depth、保存clockは別の事実である。
現行matched evaluatorがglobal nonzero event/refinementを理由に保守的にNOT_TESTEDを返すことは、誤った認定ではない。
局所資格が本当に必要なら、指定ID/時間窓の既存probeへ必要事実を足すか、その窓を独立に切り出してresolution実験を行う。
窓始点の粗いtrajectory stateをコピーしてdtだけを縮めても、それ以前の誤差は消えない。元releaseから再実行するか、必要なstate・cursor・RNG/root履歴を保持した正当なcheckpoint/replayから始める。供給した始状態に条件付けた局所伝播実験を選ぶ場合は、その状態と履歴条件に限定した比較と明記し、元run全体の資格へ昇格しない。Brownianのroot係数・conditional tree・draw履歴を捨てたwindow restartをpathwise refinementと呼ばない。

保存する最小事実の候補は対象window内のaccepted interval数/最小最大幅、method/root policy、
使用したevent/axis/reintegration、field/time/geometry resolutionである。
巨大な全粒子path ledgerや常時stage traceを追加しない。countだけでstrong orderや連続OU exact first-passageを推定しない。

誤差原因は入力同定→joint source→同一点field/support→個別force/G/J→一step→最初のhit/response/remainder→populationの順に調べる。
model/field/refの誤差床がある時、dtだけを縮め続けない。
ablationでは参照側も同じslice/作用へ変更し、保存full-force軌道からforceだけ引いたものをmatched参照にしない。
乱数・closure・field transfer・event・有限標本の寄与率を総差から自動的に配分することもできない。

### 14.4 検証・撤去の条件

R11の改良ではdense production配列/listとその前提専用の設定・testsを撤去し、
count/quantile/censoring/terminal bracket/clock/coverageの意味確認を既存external testsへ移す。
既知小型reference、欠損/不正ID、非一様clock、未放出・failure・unknown disappearance、own-survivorの違いを確認する。
streamingと旧結果が等しいだけで物理的正しさの確認にしない。

R12では必要なclaimを限定したprobe/独立resolutionだけ追加する。
参照の内部step/乱数が不足する場合は資格を留保し、実用観測の比較は継続する。
COMSOL access・原MPH・不足export・実験weightsなど真の外部条件は、必要なclaimと再開条件に結び付けて残す。

## 15. 全材料・浮遊導体を含む独立電場：必要になった場合の別拡張（R14）

これは現行static gas-only builderの責務整理だけでは実現しない。Vigus等で独立電場の全電気領域対応が目的なら、
B13として入力topology、物理式、拘束、出力、依存係数まで一つの追加能力として実装する。
過去S3-Mのfield差4%や保存電圧を、現行機能の資格・初期guess・interface境界値へ流用しない。

### 15.1 ownerと入力

通常clientのelectrostatic_builderが電気領域producerを所有する。particle engineは変えない。
particle gas meshとfull electrical meshは別の役割なので、無理に一GeometryDataへ統一しない。
producer入力として次を明示する。以下は概念的なfixed recordで、汎用CAD/material pluginではない。

| record | 必要な内容 | 検査 |
|---|---|---|
| ElectricalMesh | SI RZ nodes、tri3、part/cell/surface identity、gas出力への対応 | conforming、正面積、単一part/cell、facet所有、外面/界面 |
| Material | part→正の等方epsilon_r、Poisson volume source、plasma closureの各指定 | 名前から導体/真空を推定しない。sourceなしとclosureなしを混同しない |
| ConductorGroup | 接続part/node group、既知V又は未知V＋既知総Q | 同group equipotential、固定/浮遊競合・異group接触・無anchor拒否 |
| ElectricalBoundary | 既知V、outward D、interface free sheet charge、axis symmetry | 全外面に意味を付ける。surface native traceによる未知界面の代用をしない |
| SolveControls | nonlinear/linear budget、残差/個別Q/global balance予算 | convergenceとiteration上限を分離。失敗時に別closureへ切替えない |
| GasOutput | gas part集合、P1 layout/support、gas-side recovery規則 | solid側Eを混ぜない。particle source/clock/物性をfitしない |

この拡張で既存のgas-only問題が同じ一般化弱形式のsubsetとして表せるなら、旧gas-only専用assembly経路を置換する。
二つの同等Poisson実装やcompatibility modeを残すことを目的にしない。
ただし現行reduced modelと新modelのclosure/boundary問題が異なる場合は、異なる科学revisionとして明記する。

### 15.2 数学と数値処理

対象は -div(epsilon grad V)=rho、E=-grad V、measure=2*pi*r*dA。
P1 cellのstiffnessはcell一定epsilonと2*pi*r_c*Aを用いる。非線形sourceとJacobianは同じquadratureで評価する。
free sheet loadとprescribed outward-D loadの符号を区別し、interface V連続とD jumpを弱形式へ反映する。

法線は外面でn_out、D_n=D・n_outと定義する。荷重は
~~~text
b_i = integral_Omega( N_i*rho*2*pi*r dA )
    + integral_interface( N_i*sigma_free*2*pi*r ds )
    - integral_GammaN( N_i*D_n*2*pi*r ds )
~~~
内部界面はminus側からplus側への法線nを固定し、(D_plus-D_minus)・n=sigma_freeとする。非零sheetと非零outward-Dを同じ符号で足さない。volume source/plasma closureを評価するのは明示した非導体cellだけで、equipotential導体内部には置かない。導体表面の指定総Qと、非導体界面free sheetを二重計上しないよう、input topologyで対象を分離する。

conductorは物理epsilon=0と呼ばず、equipotential制約で表す。
V=P*y+V_fixed とし、普通のfree nodeと浮遊group voltageをreduced DOFへ写す。
R=P.T*(K*V-b(V))-Q_prescribed、J=P.T*(K-db/dV)*P を解く。
Pは数学上の記号で、巨大dense matrixを作る指定ではない。node→reduced DOFとgroup reductionを既存sparse/matrix-free方針に沿って実装する。
Q=0はV=0ではない。initial guessとDirichlet条件も別である。

nonlinear closureのgate/floor・温度K/V・ion mass・sourceのactive branchを明示し、解析Jacobianを対応させる。
solver successだけで受理せず、free residual、各浮遊groupのcharge残差、global charge balanceを全て確認する。
正負groupの残差が相殺していても個別誤差を隠さない。
必要なlinear solverと依存追加は実規模/condition/memoryの測定後に選び、外部package導入を先に目的化しない。

### 15.3 gas出力と粒子への接続

full V/group reactionは監査用output、particleへ渡すV/E/plasmaはgas側layoutへ切り出す。
free sheet charge=0ならnormal D連続からnormal Eにepsilon比のjumpがあり、非零sheetでは宣言したD jump則に従う。このため全材料のelement gradientをnode平均しない。
gas-side recovery、axis radial zero、grad(E^2)、転送/回復誤差を別に記録する。

新V/Eでne/ni/uiを変えたら、宣言済みのcharge collection/ion polynomial/DEP等もB10のproducer操作で再生成する。
electron-temperature出力KとVigus係数入力V、density/ion-speedのfield名や意味は明示mappingで変換し、名前だけ一致させない。
neutral flow/T/rho/mu、粒子源・壁・数値clockを無関係に変更してagreementを調整しない。

### 15.4 独立受入

二層capacitorのV連続/D連続とE jump、浮遊slabの
Vf=(Q+C1*V1+C2*V2)/(C1+C2)、Q=0/正/負、異initial guess、個別Q非相殺、
無anchor/競合拒否、gas-side E、source/Jacobian微分、zero-space-charge極限をCOMSOLなしで確認する。
その後にmesh/quadrature/recovery収束と、同じgas点・同じRZ体積measureで外部referenceを比較する。
非零free sheet、非零outward-Dの符号と2*pi*r*dsの面measure、conductor-source除外も独立microcaseで検査する。zero loadのcaseだけでは符号誤りを検出できない。
電場RMS改善と粒子停止/時間/分布の改善を別observableで判定する。局所最大差と軌道感度も隠さない。
原MPH・必要export・fullmaterial対応入力がない場合は、その比較claimに必要な外部条件として残す。

## 16. 実装できる単位・依存・撤去・完了条件

全体の到達点を残したまま、変更理由が独立する単位で実装する。Aの不具合修正を先に行い、
数式・path・RNGを同時に変更しない。下表は実装案で、今回実行した作業一覧ではない。

| 単位 | 対象・主な変更/受渡し | 同時撤去・更新 | 完了を判定する実行 |
|---|---|---|---|
| B01 exact field identity | R15。field tableは物理floatとexact node IDを分離、同じmask/permutationを適用 | float-ID authorityと事後exact検査。_read_fields/_match_field_rows、mapping説明、provider tests | 2^53/2^53+1の誤join拒否、合法境界、重複・不正・並替、coordinate-bijection、正常canonical値 |
| B02 static builder input | R16。共通prepareからStaticThermalInput/PreparedBuilderInputへ | 汎用FieldData tupleの再解釈、consumerのstatic slicing。check/build/例題/docs | 時系列velocity/Tを両入口でsolve前拒否、正常static小型solve/輸送維持 |
| B03 source input resolution | R06。raw config/CSV bytes→typed source spec→exact realized rows | validation-only helper、Any consumer、余剰/不足cell無視、二重read。recipe/Python/CLI/tests | malformed subcasesをexit2・未公開、valid internal/surface/filter/inherit/units/速度変換 |
| B04 JSON publication | R07。完成temporary→create-if-absent公開 | 公開先open('x')後write。analysis CLI/docs | encode/write/publish各failure、旧file保持、部分targetなし、retryと成功summary |
| B05 model facts/binding/proof | R01/R02/R03。typed model record、直接role offset、concrete stage+proof prepare、seal | selected identityとtyped parameterの再推測、他category offset借用、残余proof分岐、説明の少数例外 | 全27 descriptorの必要な代表/域境界、背景一致、未対応proof拒否、全force coverage、check/result説明 |
| B06 production scalar laws | R04。force/charge family単位にhot算術を一ownerへ | array/compiled二重body。binding/例外入口は必要な役割を残す | 独立式/極限/微分/域/極値、同入力科学payload、hot time/memory/rounding差 |
| B07 stateとparent/worker | R17。既存residentをslab入口へ、親clock/説明/full planを_PreparedRunへ | 13列のcaller再列挙、zero-read schedule/説明のworker複製。必要capacityへ投影 | worker/slab/output/resume同一、payloadが親観測数へ比例しない、shared/private/replay/accounting |
| B08 accepted material commit | R05。exact/curved/surfaceの成功response更新を共通ownerへ | 重複resident/ordinal/lifecycle/event/contact/stat更新。method継続は残す | first hit/同時facet/corner/finite radius、surface failure、右連続、残時間、axis/OU再開、identity/性能 |
| B09 result content binding | R08。completion payload roster、消費fileの検証とidentity scope | client独自のhash推測、manifest同定＝全内容検証という説明 | 有限改変/旧segment/差替え/欠損/中断/recovery、正常再利用、I/O/memory測定 |
| B10 producer dependency | R09。primitive+layout/clock+parameter/property→宣言した全出力group | 一組の実handoffで旧手動配線/重複fingerprintを置換 | dependency変化時の再生成/拒否、独立prescribed保持、全群公開、zero E/off-node/単位 |
| B11 comparison capacity/claim | R11/R12。必要ならframe/batch集約、pairedとunpaired目的を分離、局所資格は限定probe/実験 | dense production history、用途外の無理なID対応、global数から局所資格推定 | 小型独立集計、clock/censoring/coverage、seed不確かさ、memory規模、必要claimだけの独立resolution |
| B12 evidence/example/doc closure | R18/R19/R20。実inventory受入、実加算順文書、one-read projection/atomic snapshot/claim scope | literal例題名/count、重複row validator、parse/hash別read、in-place snapshot write | installed空cwd Windows/Linux、row/hash/中断/履歴参照、公式buildと出典・値・表示一致 |
| B13 full electrical domain（条件付き） | R14。材料/全mesh/VQ/interface/source/gas recoveryと係数handoff | 採用時に同じ弱形式のsubsetになる旧gas専用実装を統合。異なるmodel意味は明示 | 材料/浮遊Q解析解、Jacobian/残差/charge balance、mesh/回復収束、同じgas measureと粒子observable |

### 16.1 依存関係とreview境界

~~~mermaid
flowchart LR
    A[B01–B04 入力・公開bug] --> B[B05a 解決済みrecord/facts/直接binding]
    B --> C[B05b proof constructionとseal]
    B --> D[B07 state/親worker]
    C --> E[B06 scalar lawをfamily単位]
    D --> F[B08 material commit三入口]
    C --> F
    A --> G[B12 evidence/例題/説明]
    D --> H[B09 result内容binding]
    B --> I[B10 派生係数handoff]
    H --> J[B11 用途比較・容量]
    I --> K[B13 全電気領域が必要な場合]
~~~

B01–B04は独立ownerで並行可能だが、同じ入力の意味を変える変更を一括して効果分離不能にしない。
B05aはstage算術を変えないrecord/facts/binding、B05bは証明familyの明示構築である。
B06はさらにfamily単位の数学移動とし、一般branch追加と同時に全式をrewriteしない。
B07はまず確認済みzero-readの親専用値とresident入口を移し、既存runtimeの必要readを壊さない。

**worker physicsの段階境界に注意する。**
現在PreparedTransport.physicsはPhysicsPlanで、transportはnoiseとrequires_stage_evaluationを読む。
StageDynamics.runtime.planもnoise容量判定に使い、PhysicsRuntimeのmapping評価入口はrequired_fields、Brownian入口はdrag/noiseを使う。
したがって「descriptorは親だけ」と書いてplan属性を削除するだけではcallerを壊す。
B05/B07で具体的なnoise controls、既存TransportCapabilitiesのstage_evaluation/stochastic、primitive packing layoutとnative/proof recordを一度解決し、
同じPhysicsRuntime/transportの必要callerへ渡してから、workerに不要なselection説明を外す。
別の同等runtimeや長期併存するstate APIを作らない。数値kernelへは既存の配列を渡す。

B09/B10/B11は科学的に別の保証である。内容hashができても派生係数の物理的freshnessは自動で証明されず、
streaming評価になっても数値資格やnative真正性は増えない。
B13はB02のstatic入力を正しくしてから、別の科学問題として進める。

### 16.2 changeごとの完了票

新しい常設管理systemは作らず、既存change/PR記録に次の六点を残せばよい。

1. 解く利用case・失敗基準と変更するowner。
2. 固定する入力/source/model/clock/seed/field/referenceと、変更する意味。
3. 実装した受渡しと、同時に削除した旧経路・設定・呼出側・専用検査・説明。
4. 独立reference/公開scenario/影響benchmarkの実行command・source/input identityと結果。
5. 科学payload、rounding、failure/coverage、時間/memoryの差と適用範囲。
6. 未確認事項と、それを判定できる次の実験又は真の外部条件。

既存testsで足りる性質に新しいcontract suiteを足さない。private実装の移動だけをmockで固定しない。
一つの成功例だけ、担当別の完了宣言だけ、古いCIの緑だけで基盤全体の整合としない。

## 17. 検証・性能・形式・撤去を同時に閉じる

### 17.1 検査の単位

| 改良の性質 | 必要な確認 | 保持/更新する既存owner |
|---|---|---|
| 入力解決/lexical bug | 公開CLI/Pythonの誤入力拒否、出力不在、合法source/fieldの値とidentity | input/source/csv/builder/CLI scenario |
| facts/binding/proof構築 | 正規化default/role、同名/別名field、science mismatch拒否、既知familyの包含 | model descriptors、required fields、local applicability、独立force/charge tests |
| scalar math共有 | 独立値/極限/微分/域境界、overflow/underflow、array/native結線、同stage連成 | Decimal/解析/積分oracleを保持。共有body同士の比較だけにしない |
| resident/worker整理 | serial/process、slab/output/resume identity、実shared/private allocation、payload増分 | process_arrays、workspace ownership、process/durable scenarios |
| material commit整理 | first hit、candidate CSR、ordinal/RNG、surface departure failure、右連続、残時間/axis/OU | wall/source/contact/events/Brownian/projected-midpoint tests |
| output内容binding | 正常値保持、有限改変/欠損/差替え検出、commit中断/recovery、read費用 | output/publication/durable/result tools |
| producer派生関係 | primitive/layout/time/parameter/property変更、全出力群、独立prescribed保持 | 該当producerとcache/canonical readback |
| report/installed inventory | one-read hash、row shape、中断時旧snapshot保持、全例の実行集合、出典/表示一致 | 現行snapshot tests/docs gate/installed acceptance/公式build |

当面の改善で全COMSOL、全benchmark、10^6粒子を毎回走らせる必要はない。
一方、変更後のコードで必要な独立性質を実行せず、「testを書いた」「過去にPASS」で完了にしない。
既存の簡潔な確認が同じ性質を検査している場合は追加しない。

### 17.2 不要な検査を残さない

置換するprivate helper名・dataclass field順・固定byte定数・診断文言のsnapshotを新設しない。
現在のallocation ownership確認、独立transient上界、解析event/OU/charge極限、failure局所性は必要である。
検証用referenceはproductionから独立に残し、同じ数値式の別production入口だけを撤去する。
一つの作用を削除しても他作用が使うprimitiveや独立oracleを静的未参照だけで削除しない。

### 17.3 数値・永続形式の変更を隠さない

| 変更 | 判断 |
|---|---|
| typed parse/直接binding/親専用metadata移動 | 合法入力の物理意味・演算・RNGを維持。誤受理入力の拒否はbug fix |
| model説明flat mapping拡充 | arrays/schema4を維持できる。check/result/resume説明revisionと全consumerを更新。物理revisionを自動変更しない |
| 共通material commit | 演算順・正規化位置・event順・ordinal・prefix/再開規則が同じなら構造変更として検証 |
| scalar算術、regime、force加算順、event/geometry/RNG規則の変更 | 数値/科学変更として独立referenceと該当revision。refactorという名で差を隠さない |
| 必須payload roster/新state/新material問題 | 意味と受理規則に応じschema/algorithm revisionを更新。現行reader/fixtures/docs/callersを一度に移し旧readerを撤去 |
| report snapshot/authoring copy | artifactのidentity/presentation/共有を保持。current design、historical execution、build freshnessを別に記録 |

互換不要は、旧経路を別名wrapperにして残す理由をなくす。
物理referenceやraw/native入力の保存まで不要という意味にはしない。
破壊的変更が必要な場合は、変更前のsource/input/run identityを履歴に保全し、productionは現行一経路にする。

### 17.4 性能受入

hotなstage/commit/worker/outputを変える時だけ、同じcase、dt、tree、field、wall、output、workers、
slab、warm/cold条件で時間とmemoryを測る。
既存P10のfield/relaxation、P19 local certificate、B03のBrownian、process/output-heavy等から影響する代表を選ぶ。
JIT償却を改訂間speedup、serialization減少を10^6粒子core改善、RSSをowned-array budget保証と呼ばない。
科学payloadはfinal/release/canonical boundary/failure/lifecycle/共通frame/probeの列と順序で比べる。
HDF5 container bytes、到着順、timing、worker metadataを物理identityと混同しない。
新しいcontent hashのI/O費用もload/simulate/open/readの実際の利用経路へ含める。facts/schema/provenanceの意図した更新はmetadata差として記録し、旧canonical/result identity hash全体の不変を受入にしない。数値列の保存と、変更された説明/同定の正しさを別に照合する。

## 18. 追加・拡張・削減で何が変わるか

| 変更 | 必ず変える層 | 改良後に不要にする波及 | 独立確認 |
|---|---|---|---|
| dragの別model/revision | descriptor/typed model、scalar law、stage/proof constructor、compatibility | Talbot/lift/ionのoffset再利用分岐、engine、model別result serializer | rate/Jacobian/域/緩和極限、noise対応、trajectory |
| 新しいadditive区分 | PhysicsPlan slot/合成位置/global bound/constant coverage、worker record、説明 | Python callback登録、第二engine | 力とbound、状態依存、全作用coverage |
| 粒子温度等の新state | source→resident→stage/dense→hit/継続→checkpoint/result/memory→解析 | 任意ODE frameworkの先行導入、裏subcycle/default | 熱収支と連成、invariant、hit/再開、保存/再利用 |
| 新wall law | boundariesのlaw/RNG/説明、保存則・分布・残時間 | first-hit探索の複製、method別law実装 | angle/energy/charge則、corner、複数hit |
| 一model/例題の削減 | selector/union/code/proof/関連設定・例題・tests/bench/docs | 互換alias、消したmodelのbinderを参照する他category | 残るmodelの単独/併用とinstalled inventory |
| 出力項目追加 | その値のowner、永続projection、reader/context/consumer | 同じ列のcheckpoint/result/worker別手書き定義 | 単位/欠損/identity、resume/selection/export |

新カテゴリと新stateは変更の大きさが異なる。既存x/v/Zで閉じるadditive forceのためにevent/RNGを変更しない。
逆に新stateを一field足しただけで対応済みにしない。初期値からdense hit状態・保存まで縦切りで閉じる。
新modelの式が未検証なら、型とbindingが通ることを科学的な受入へ代用しない。

## 19. 明示して残す条件と、今回の完成範囲

今回具体化したのは調査と設計であり、B01–B13のproduction実装ではない。
新規不具合の修正、parent/worker整理、内容hash、fullmaterialの能力を実装済みとはしない。
前段階の旧成果物撤去は完了済みで、今回の未実装提案へ戻さない。

3D、moving geometry、非線形noise、二方向連成、粒子間衝突、surface charging/再飛散、
GPU/out-of-coreは、現在の能力の欠陥と同一視しない。
必要な利用目的、既存方法で解けない理由、物理モデル、独立reference、容量/性能の実測が着手条件である。
今回のroot cause修正と同じchangeに空API/flagや第二runtimeを先行追加しない。

実装開始時は、まずB01–B04で誤受理/不完全公開を止め、その後B05/B07で受渡しを整理する。
その足場からB06/B08を実施し、B09/B10/B11/B12の再利用と説明を閉じる。
全材料電場が必要ならB13を別の科学問題として進める。
各段階でcaller/設定/保存/例題/説明/検証/撤去を一緒に完了し、できた一部を基盤全体の完成へ読み替えない。

## 20. コード参照と、採用時に更新する正本

| 主題 | 現行source |
|---|---|
| source設定/CSV | [producer.py](../solver/src/chamber_particles_tools/source_builder/producer.py) |
| field ID/selection/transfer | [csv_import/workflow.py](../solver/src/chamber_particles_tools/csv_import/workflow.py) |
| static field builder | [electrostatic_builder/workflow.py](../solver/src/chamber_particles_tools/electrostatic_builder/workflow.py) |
| case/producer publication | [inputs.py](../solver/src/chamber_particles_tools/inputs.py)、[bundle_publication.py](../solver/src/chamber_particles_tools/bundle_publication.py) |
| model record/role | [catalog.py](../solver/src/chamber_particles/physics/catalog.py) |
| binding/global/local/proof | [runtime.py](../solver/src/chamber_particles/physics/runtime.py) |
| stage order/fixed numeric records | [compiled.py](../solver/src/chamber_particles/physics/compiled.py) |
| scalar law/oracle boundary | [forces.py](../solver/src/chamber_particles/physics/forces.py)、[charge.py](../solver/src/chamber_particles/physics/charge.py) |
| parent facts/clock | [engine.py](../solver/src/chamber_particles/engine.py)、[time_grid.py](../solver/src/chamber_particles/time_grid.py) |
| resident/worker/commit | [cpu.py](../solver/src/chamber_particles/cpu.py)、[execution.py](../solver/src/chamber_particles/execution.py)、[transport.py](../solver/src/chamber_particles/transport.py) |
| proposal/conditional OU | [integrators.py](../solver/src/chamber_particles/integrators.py)、[stochastic.py](../solver/src/chamber_particles/stochastic.py) |
| geometry/event/law | [geometry.py](../solver/src/chamber_particles/geometry.py)、[events.py](../solver/src/chamber_particles/events.py)、[boundaries.py](../solver/src/chamber_particles/boundaries.py) |
| result identity | [output.py](../solver/src/chamber_particles/output.py) |
| public result利用 | [selection.py](../solver/src/chamber_particles_tools/selection.py)、[analysis](../solver/src/chamber_particles_tools/analysis/__init__.py)、[visualization](../solver/src/chamber_particles_tools/visualization/__init__.py) |
| 比較/資格 | [saved population](../solver/tools/vv/comsol/evaluate_saved_population.py)、[matched evaluator](../solver/tools/vv/comsol/evaluate_matched_trajectory.py) |
| 派生係数 | [prepare_vigus_transport.py](../solver/tools/vv/comsol/prepare_vigus_transport.py)、[field cache numerics](../solver/src/chamber_particles_tools/field_preprocessor/numerics.py) |
| 例題/配布 | [examples.py](../solver/src/chamber_particles_tools/examples.py)、[release_workflow.py](../solver/tests/release_workflow.py) |
| report根拠 | [build_report_snapshot.py](../analysis/build_report_snapshot.py)、[snapshot tests](../report_app/tests/test_reviewed_snapshot.py) |

採用時は[architecture](../architecture_proposal.md)、[implementation plan](../implementation_plan.md)、
[物理](../solver/docs/physics_models.md)、[数値](../solver/docs/numerics.md)、
[入力形式](../solver/docs/case_format_v3.md)、[結果形式](../solver/docs/result_format_v4.md)、
[V&V](../vv_methodology.md)、該当client READMEと利用例を変更の意味に合わせて更新する。
AGENTS.mdへ今回の撤去履歴や禁止事項を積み上げない。現行owner/判断基準が変わる場合だけ、その規約を置き換える。
この文書にある具体的な型・関数案はcurrent code APIとしてtestしない。実装後の受入は利用振る舞いと科学的性質を基準にする。
