import React from "react";

import {
  DataComponent, DataTable, MetricCard, ReportSection, RichNarrative, useDataApp,
} from "../../data-app-public.jsx";

function Narrative({ id, title, queryIds, value, className }) {
  const { reviewedRows, visible } = useDataApp();
  if (!visible(id)) return null;
  const ids = queryIds.filter(Boolean);
  return <ReportSection id={id} title={title} queryId={ids[0]} queryIds={ids}
    sourceRowsByQuery={Object.fromEntries(ids.map((queryId) => [queryId, reviewedRows(queryId)]))}
    showHeading={false} className={className}>
    <RichNarrative id={`${id}:body`} value={value} label={`${title}を編集`} />
  </ReportSection>;
}

export function ReportContent() {
  const { appTitle, setAppTitle, canEdit, mode, reviewedRows, visible } = useDataApp();
  const decisions = reviewedRows("design_decisions");
  const physics = reviewedRows("physics_scope");
  const modules = reviewedRows("module_responsibilities");
  const numerics = reviewedRows("numerical_contracts");
  const results = reviewedRows("result_layers");
  const workPackages = reviewedRows("implementation_work_packages");
  const quality = reviewedRows("quality_findings");
  const roadmap = reviewedRows("roadmap");
  const metric = (area) => decisions.find((row) => row.area === area)?.count ?? "未取得";
  const exampleCount = roadmap.find((row) => row.phase === "current")?.example_count ?? "未取得";

  return <article className="report-content particle-redesign-report"
    aria-label="半導体チャンバー粒子輸送基盤の現行仕様と利用手順">
    <header className="report-hero">
      <p className="report-kicker">CHAMBER PARTICLES 0.5.0 · CLI AND PYTHON WORKFLOW</p>
      <h1 data-data-app-title contentEditable={canEdit && mode === "edit"}
        suppressContentEditableWarning
        aria-label={canEdit && mode === "edit" ? "レポート名を編集" : undefined}
        onBlur={canEdit && mode === "edit" ? (event) => setAppTitle(
          event.currentTarget.textContent.trim() || appTitle,
        ) : undefined}
        onKeyDown={canEdit && mode === "edit" ? (event) => {
          if (event.key === "Enter") { event.preventDefault(); event.currentTarget.blur(); }
        } : undefined}>{appTitle}</h1>
      <RichNarrative id="report:introduction" className="report-deck" label="導入を編集"
        value="利用者が機能を選び、自分の入力で計算し、結果を解釈・再利用するまでの案内。現行0.5.0 alphaの使い方・責務・数値の意味を、実行時の条件に束縛された品質・性能・COMSOL比較の証拠と区別して示す。" />
    </header>

    <Narrative id="executive-decision" title="この基盤で行うことと読み方"
      queryIds={["design_decisions", "physics_scope"]}
      value={`## この基盤で行うことと読み方

**外部の形状・場・粒子源を、自分の入力として検査・計算し、結果を再利用する。** 現行0.5.0 alphaはCLIとPythonを同じwheelから提供する。計算coreは運動・帯電・衝突を解き、通常clientは入力変換、場の生成、解析・export・表示を担当する。

**使い始めるときは利用経路、改良するときは責務表を読む。** 以下は「機能を選ぶ → 自分の入力を作る → check/run → 結果を解釈する」の順に並ぶ。モデルや数値条件の詳細は物理・数値節で確認する。

**現在の仕様と、過去に実行した証拠を区別する。** authorityとcatalogのhashはこの設計snapshotが参照する文書・実装を示す。品質・性能・COMSOL比較の数値は各receiptのsource、実行時条件、母集団に束縛される。文書の再生成はそれらの再計算ではない。

**実行可能、実用的な対応、数値認定、装置予測は別の判断である。** checkの成功は準備完了を示す。収束や物理妥当性は用途に応じた独立の証拠を必要とし、未提供の測定はunavailableとして扱う。`} />

    <div className="report-facts" aria-label="現行仕様の要点">
      <MetricCard id="metric-api" title="core公開操作" queryId="design_decisions" sourceRows={decisions}
        value={metric("core_api")} description="load_case · check_case · simulate · open_result" />
      <MetricCard id="metric-extension" title="通常client" queryId="design_decisions" sourceRows={decisions}
        value={metric("normal_clients")} description="入力4機能 · analysis · visualization" />
      <MetricCard id="metric-tests" title="testの層" queryId="design_decisions" sourceRows={decisions}
        value={metric("test_layers")} description="verification · scenarios · performance。実行件数ではない" />
      <MetricCard id="metric-stages" title="編集可能な例題" queryId="roadmap" sourceRows={roadmap}
        value={exampleCount} description="wheelのinitから生成。別装置のdefaultではない" />
    </div>

    <Narrative id="single-loop" title="機能を選び、公開入口から実行する"
      queryIds={["design_decisions"]}
      value={`## 機能を選び、公開入口から実行する

\`models → init / create → check → run → inspect → analyze / export / view\`

最初にmodelsで候補modelの要求と適用条件を調べ、initで編集可能な例題を作る。自分のmesh・field・sourceを使う場合は、通常clientのCSV import、source builder、field preprocessor、gas-only electrostatic builderから必要な機能を選ぶ。入口と入力例の正本はsolver READMEと各client READMEにある。

Pythonでは\`inputs.create_case(bundle, settings, output_directory)\`または\`create_from_configuration\`でcaseを公開し、coreの\`load_case → check_case → simulate → open_result\`を使う。CLIも同じ処理を呼ぶ。checkは実行と同じprepareを使い、粒子更新・worker起動・結果出力は行わない。

COMSOLの発見・抽出・比較はcheckout内の外部V&V経路で行う。通常wheelの入力作成や結果利用に、MPHやmodel_datasetの存在を要求しない。`} />

    <section className="report-section">
      <Narrative id="implementation-sequence" title="自分の入力を作り、確認して公開する"
        queryIds={["implementation_work_packages", "roadmap"]}
        value={`## 自分の入力を作り、確認して公開する

入力のunit・basis・ID・field supportと独立物性を確定し、CSV mapping、source設定、case recipeを作る。例題の寸法、境界ID、seed、dtは新しい対象のdefaultではない。

recipe format 2はcanonical input_bundle、完成bundleに含めるsource table全量、case settingsを持つ。CSVから作るtableと\`existing_table\`で明示したtableだけを残す。\`settings.sources\`はその中から放出対象を選び、bundleのtable構成を暗黙には変更しない。

source format 2はinternal tableまたはsurface rowを受け取り、位置またはfacetとstrict-interior parameter、速度、ID/release、物性を明示する。\`inherit_from\`を選んだときだけ物性をexact-ID joinで継承する。CSV mapping format 4はcaller-owned csvとcomsol providerを区別し、csvに架空のnative証拠を付けない。

通常bundle producerはbundle.h5とreport.jsonを同じ新規directoryへ公開する。case publisherはdata pathとlogical hashを埋め、load/checkと編集元の受渡しまで完成してから公開する。作成後はcheckの診断を解決してrunし、inspectで実際の完了状態とfailure coverageを読む。`} />
      {visible("work-package-table") && <DataComponent id="work-package-table" title="利用者フローごとの完成条件"
        queryId="implementation_work_packages" kind="table" sourceRows={workPackages}>
        <DataTable rows={workPackages} searchable={false} compactNumbers={false} label="利用者フロー"
          columns={[{ field: "package", label: "単位" }, { field: "capability", label: "機能" }, { field: "deliverable", label: "受渡し" }, { field: "gate", label: "確認" }]} />
      </DataComponent>}
    </section>

    <Narrative id="result-store" title="結果だけを渡し、同じ母集団で再利用する"
      queryIds={["result_layers"]}
      value={`## 結果だけを渡し、同じ母集団で再利用する

case/dataはschema 3、結果はschema 5、checkpointはlayout 2である。immutable context.h5は元case/dataのhash、geometry、境界/sourceの意味と計算粒子数、basis/unitを保持する。元caseを移動しても結果の意味を読める。

ResultViewのmanifestは入れ子まで不変である。独立したJSON copyは\`manifest_document()\`、派生成果物に記録する結果identityは\`source_identity()\`を使う。消費fileは同じopen handleでhash照合し、全payloadは\`inspect --verify\`で検査する。receiptでは結果identityと実際に検査した範囲を区別する。

開いたResultViewまたは\`select_result\`で束縛したResultCohortを、解析・export・表示へ渡す。final、反復interaction、first hit、terminal、保存active曲線は別量である。exact selectionと分母を保ち、表示用sampleを全体統計へ置き換えない。stuck/heldも数値的な結果として解釈する。

export/viewはreceiptを含む全成果物を完成してからdirectory単位で公開する。現行readerへ旧結果の互換経路は足さず、旧環境でexportするか、条件を明示して再実行する。`} />

    {visible("result-table") && <DataComponent id="result-table" title="結果から説明までの受渡し"
      queryId="result_layers" kind="table" sourceRows={results}>
      <DataTable rows={results} searchable={false} compactNumbers={false} label="result layers"
        columns={[{ field: "layer", label: "層" }, { field: "contents", label: "内容" }, { field: "rule", label: "規則" }]} />
    </DataComponent>}

    {visible("decision-table") && <DataComponent id="decision-table" title="現行の製品判断"
      queryId="design_decisions" kind="table" sourceRows={decisions}>
      <DataTable rows={decisions} searchable={false} compactNumbers={false} label="製品判断"
        columns={[{ field: "area", label: "領域" }, { field: "decision", label: "判断" }, { field: "reason", label: "理由" }]} />
    </DataComponent>}

    {visible("module-table") && <DataComponent id="module-table" title="責務の所有者"
      queryId="module_responsibilities" kind="table" sourceRows={modules}>
      <DataTable rows={modules} searchable={false} compactNumbers={false} label="module ownership"
        columns={[{ field: "module", label: "module" }, { field: "owns", label: "所有する" }, { field: "must_not", label: "所有しない" }]} />
    </DataComponent>}

    <Narrative id="complexity" title="追加・変更・削減を責務の境界で行う"
      queryIds={["design_decisions", "numerical_contracts"]}
      value={`## 追加・変更・削減を責務の境界で行う

同じ区分へmodelを追加するときは、catalogの入力・適用条件・proof roleと、その区分の数式実装、独立検証、利用例を対応付ける。producer固有の分岐を共通engineへ足さない。新しい区分が必要な場合はstate・stage payload・積分・event・保存のうち何が変わるかを先に定め、固有の連成を既存区分へ無理に押し込まない。

機能を削減するときは、登録、設定、呼出側、例題、文書、確認も同時に整理する。置き換えた経路や旧新の二重validatorを残さず、verification用referenceはproductionの外へ置く。一般plugin frameworkやDI containerを改良の前提にしない。

canonical入力、field support/sampling、physics catalog、geometry/event、wall law、時計、結果identityは各ownerへ集約する。CLIや通常clientが数値規則や第二solverを複製しない。責務表の「所有しない」は、変更の影響先を判断するための境界である。

テストは公開利用と科学的性質を確認する。品質gate、用途別性能、COMSOL観測量、物理妥当性は別々の根拠として維持し、一つの認証へまとめない。`} />

    <Narrative id="governing-equations" title="運動と帯電を同じstateで解く"
      queryIds={["physics_scope"]}
      value={`## 運動と帯電を同じstateで解く

状態は位置x、速度v、平均電荷数Zである。選択したforce、drag、noiseと電荷rateを実際の積分stageで評価する。固定電荷と連続帯電を別engineへ分けない。

質量、drag径、電気半径、接触半径、排除体積、model weightは別のauthorityである。runtimeで一つの粒径から勝手に再構成しない。下表は現在のcatalogから取得した選択肢で、実際のfield metadata、適用域、積分器の検査が必要となる。

catalogは各primitiveのfinite・positive・nonnegative条件、選択model identity、path証明に使う入力roleを宣言する。stageで値を計算できることと、全pathのsupport・適用域・constant specializationを証明できることは別である。`} />

    {visible("physics-table") && <DataComponent id="physics-table" title="現在の物理catalogと適用条件"
      queryId="physics_scope" kind="table" sourceRows={physics}>
      <DataTable rows={physics} searchable={false} compactNumbers={false} label="物理catalog"
        columns={[{ field: "capability", label: "category / model" }, { field: "current", label: "revision" }, { field: "limits", label: "条件・制約" }]} />
    </DataComponent>}

    <Narrative id="coordinates" title="XYとRZ meridionalの意味を保つ"
      queryIds={["design_decisions", "physics_scope"]}
      value={`## XYとRZ meridionalの意味を保つ

現行の粒子自由度はCartesian XYまたはRZ meridionalの2次元である。RZのaxis foldは材料壁の反射ではない。方位成分を捨てて3DのBrownian運動を再現したと主張しない。

通常のCSV importは静的selected nodal場のXY/RZを扱い、runtimeは対応するstatic/固定topology linear-timeのregular/P1/Q1場を扱う。縮約electrostatic builderはgas-only RZ/P1であり、過去レポートに登場した全材料・浮遊導体V/Q機能が現行builderに実装済みとは扱わない。`} />

    <Narrative id="boundary-numerics" title="first hitと壁応答、残時間を分けて接続する"
      queryIds={["design_decisions", "physics_scope"]}
      value={`## first hitと壁応答、残時間を分けて接続する

proposalの経路から最初のhitを求め、同じpathのx/v/Zで応答し、反射後は残時間を継続する。endpoint判定、固定epsilonの押し戻し、同一hitへの重複した確率判定は使わない。

specular、uniform inward angleのisotropic、thermal wallは異なるlawである。確率stickのotherwiseも明記する。point contactと有限半径接触は物理設定で、toleranceによる代用ではない。stuck/heldは数値的な境界結果であり、実物のadhesionの証明ではない。`} />

    {visible("numerics-table") && <DataComponent id="numerics-table" title="数値処理と確認する証拠"
      queryId="numerical_contracts" kind="table" sourceRows={numerics}>
      <DataTable rows={numerics} searchable={false} compactNumbers={false} label="数値処理"
        columns={[{ field: "topic", label: "対象" }, { field: "decision", label: "処理" }, { field: "acceptance", label: "確認する証拠" }]} />
    </DataComponent>}

    <Narrative id="start-gate" title="自分の目的に必要な妥当性を確認する"
      queryIds={["physics_scope", "numerical_contracts", "roadmap"]}
      value={`## 自分の目的に必要な妥当性を確認する

利用前に目的となる観測量、対象source/cohort、分母・重み、windowと単位を決める。geometry・field・sourceの対応を検査し、dt、field解像度、Brownian数値path、観測clockを別々に選ぶ。worker数は実行資源であり、物理結果を合わせる調整ノブではない。

固定macro gridとfield knotの併合・丸め・window照会は共通ownerが持つ。nominal dtや全区間のmacro件数だけで選択windowの細分化を推定せず、実partitionとaccepted numerical pathを確認する。安全処理の再積分や証明の細分化も、macro refinementや観測clockとは別である。

manifestの\`prepared_stability_upper_bounds\`は要求dtと全域boundから準備した上界であり、実stageの観測最大や誤差推定ではない。checkの成功や既存出力との一致だけで精度を判断せず、目的のwindow・観測量に適した参照と解像度変化を調べる。

外部evaluatorは比較の資格とscopeを所有する。qualifiedがNOT_TESTEDでも、raw差や明示した不確かさの範囲での実用的なpopulation対応は利用できる。装置の付着fluxや汚染量には、source rate/weight、surface capture physicsと実験比較が別途必要となる。`} />

    <section className="report-section">
      <Narrative id="dataset-role" title="現在の説明と実行時の証拠を対応付ける"
        queryIds={["quality_findings", "case_audit", "trajectory_divergence", "design_decisions"]}
        value={`## 現在の説明と実行時の証拠を対応付ける

この本文と設計表は現在のauthorityを参照する。Sourceのauthority一覧はpathとexact SHA-256に束縛され、文書変更後の未更新snapshotをdocument checkerが拒否する。一方、取り込んだ実行receiptのsourceは実行時のまま保存し、現在のHEADへ書き換えない。

model_datasetや原MPHは外部入力・V&Vの証拠であり、物理定義やdefaultではない。30 nm VigusとA/P 10 nmは別caseである。品質・性能・比較の表は、各projectionの実行source、seed集合、window、測定条件と合わせて読む。同じseedの再実行や再利用native参照を、新しい独立標本とは数えない。

Brownianの個別pathを乱数レベルで合わせる必要はない。実現source、全母集団、停止先・時刻・速度・電荷と不確かさから、実用的に合う観測量を判断する。残る差を一律に乱数へ帰属せず、不明なdisappearanceをescapeと同一視しない。native RK4/time-key noiseの5/9 weak-limit解釈も、同定した方式とmicrocaseの範囲に限定する。

frozen charge比較は明示CSV・現行式・独立な残差allowanceを用いる。ion-forceはrevisionとmatched/analogousをさらに明示し、analogousはraw差とNOT_TESTEDを返す。これらの宣言や一点の残差だけで、native真正性・積分精度・収束を認定しない。

対象のbaseline result4 storeは利用者により5件とも削除済みで、入力10fileと科学export5fileのidentityは保持されている。下表の整理状態は既存の削除完了receiptに基づき、以前の保持・拒否は当時の履歴として参照する。その他の品質・科学・性能の観測は元の実行sourceに限定し、この文書更新で新たに実行したものとは扱わない。`} />
      {visible("quality-table") && <DataComponent id="quality-table" title="今回取込んだ品質証拠と未取得事項"
        queryId="quality_findings" kind="table" sourceRows={quality}>
        <DataTable rows={quality} searchable={false} compactNumbers={false} label="品質証拠"
          columns={[{ field: "severity", label: "状態" }, { field: "finding", label: "観察" }, { field: "disposition", label: "扱い" }]} />
      </DataComponent>}
    </section>

    <section className="report-section">
      <Narrative id="implementation" title="現行能力と条件付きの追加能力を分ける"
        queryIds={["roadmap"]}
        value={`## 現行能力と条件付きの追加能力を分ける

通常のwheel入力から結果利用までが現在の製品範囲である。nativeの追加抽出・再計算を必要とする作業は外部V&Vの入口へ進む。利用目的に保存済み証拠で足りる場合は、その範囲で比較・可視化を完了できる。

全材料・浮遊導体、3D、moving geometry、GPUなどは将来の別能力である。過去の実行記録に存在する機能を現行提供機能と読み替えず、追加するときに物理・数値・利用経路・性能の独立した完了条件を定める。`} />
      {visible("roadmap-table") && <DataComponent id="roadmap-table" title="現行範囲と次の完了条件"
        queryId="roadmap" kind="table" sourceRows={roadmap}>
        <DataTable rows={roadmap} searchable={false} compactNumbers={false} label="現行範囲"
          columns={[{ field: "phase", label: "区分" }, { field: "name", label: "範囲" }, { field: "exit", label: "完了条件" }]} />
      </DataComponent>}
    </section>

    <RichNarrative id="report:disclosure" className="report-disclosure" label="文書注記を編集"
      value="現行説明の参照はauthorityとproducerのhashに、実行済み数値は各assessment/quality/performance projectionの条件とhashに対応付ける。Sourceの再生成手順は実際に渡したprojectionを含み、design-only更新とは区別される。文書更新・snapshot生成・build・表示確認は別の実行記録であり、いずれも科学計算の再実行を意味しない。表示編集はreviewed rowsを変更しない。" />
  </article>;
}
