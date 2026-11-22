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
      <p className="report-kicker">CHAMBER PARTICLES 0.4.0 · CLI AND PYTHON WORKFLOW</p>
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
        value="外部の形状・場・粒子源から、入力作成、検査、計算、保存結果の解釈までをCLIとPythonで行う現行0.4.0 alphaの構成を示す。ここに示す設計能力と、実行された品質・性能・COMSOL比較の証拠は分けて読む。" />
    </header>

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

    <Narrative id="executive-decision" title="Executive Summary"
      queryIds={["design_decisions", "physics_scope"]}
      value={`## Executive Summary

**入力から結果の再利用まで、通常の操作を同じwheelへまとめる。** 計算coreは数値処理を所有し、外側のclientがCSV変換、source作成、場の生成・cache、解析・表示を所有する。CLIとPythonで処理を分岐しない。

**検査は実行と同じprepareを使う。** \`check_case\`はmodel、field support、source、境界、資源を解決するが、粒子更新やworker起動は行わない。準備完了は収束や物理妥当性の認定とは別である。

**保存結果は元caseなしで説明できる。** result schema 4のimmutable contextにgeometryと意味を保持し、解析・export・表示は公開ResultViewだけを読む。背景場と完全な実現sourceは複製しない。

**科学的な主張は証拠の範囲に留める。** COMSOLとの実用的な輸送対応、数値的な全方程式認定、実装置の予測妥当性を区別する。今回のsnapshotに実行receiptが未取込なら、件数や精度をゼロまたは成功と表示しない。`} />

    {visible("decision-table") && <DataComponent id="decision-table" title="現行の製品判断"
      queryId="design_decisions" kind="table" sourceRows={decisions}>
      <DataTable rows={decisions} searchable={false} compactNumbers={false} label="製品判断"
        columns={[{ field: "area", label: "領域" }, { field: "decision", label: "判断" }, { field: "reason", label: "理由" }]} />
    </DataComponent>}

    <Narrative id="governing-equations" title="運動と帯電を同じstateで解く"
      queryIds={["physics_scope"]}
      value={`## 運動と帯電を同じstateで解く

状態は位置x、速度v、平均電荷数Zである。選択したforce、drag、noiseと電荷rateを実際の積分stageで評価する。固定電荷と連続帯電を別engineへ分けない。

質量、drag径、電気半径、接触半径、排除体積、model weightは別のauthorityである。runtimeで一つの粒径から勝手に再構成しない。下表は現在のcatalogから取得した選択肢で、実際のfield metadata、適用域、積分器の検査が必要となる。`} />

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

    <Narrative id="single-loop" title="CLIとPythonから同じ経路を使う"
      queryIds={["design_decisions"]}
      value={`## CLIとPythonから同じ経路を使う

\`models → init / create → check → run → inspect → analyze / export / view\`

Pythonでは通常clientの\`inputs.create_case\`または\`create_from_configuration\`で公開し、coreの\`load_case → check_case → simulate → open_result\`を使う。PreparedRunやwriterを利用者へ公開せず、coreがouter clientをimportする逆依存も作らない。

worker数は実行資源である。親がmacro時計・release・single writer・checkpointを持ち、workerは排他的粒子群を更新する。出力の要求時刻でproduction stepを切らず、同じaccepted pathから観測する。`} />

    {visible("module-table") && <DataComponent id="module-table" title="責務の所有者"
      queryId="module_responsibilities" kind="table" sourceRows={modules}>
      <DataTable rows={modules} searchable={false} compactNumbers={false} label="module ownership"
        columns={[{ field: "module", label: "module" }, { field: "owns", label: "所有する" }, { field: "must_not", label: "所有しない" }]} />
    </DataComponent>}

    <section className="report-section">
      <Narrative id="implementation-sequence" title="最初の自分のsourceを作る"
        queryIds={["implementation_work_packages", "roadmap"]}
        value={`## 最初の自分のsourceを作る

recipe format 1はcanonical input_bundle、CSV sources、通常のcase settingsを持つ。publisherがdata pathとlogical hashを埋め、load/checkが通ってから新しいdirectoryへ公開する。既存の粒子caseをtemplateとして用意する必要はない。

source format 2はinternal tableまたはsurface rowを受け取る。位置またはcanonical facet rowとstrict-interior parameter、速度、ID/release、全独立物性を明示する。物性の継承は\`inherit_from\`によるexact-ID joinを要求した場合だけ行い、整数IDの一致から同じ物理sourceを推測しない。

CSV mapping format 4のproviderはcaller-owned csvとcomsolを区別する。後者のdataset/solution/member証拠を、前者へ架空に付けない。`} />
      {visible("work-package-table") && <DataComponent id="work-package-table" title="利用者フローごとの完成条件"
        queryId="implementation_work_packages" kind="table" sourceRows={workPackages}>
        <DataTable rows={workPackages} searchable={false} compactNumbers={false} label="利用者フロー"
          columns={[{ field: "package", label: "単位" }, { field: "capability", label: "機能" }, { field: "deliverable", label: "受渡し" }, { field: "gate", label: "確認" }]} />
      </DataComponent>}
    </section>

    <Narrative id="result-store" title="保存結果を移動しても解釈を保つ"
      queryIds={["result_layers"]}
      value={`## 保存結果を移動しても解釈を保つ

入力case/dataはschema 3、結果はschema 4、checkpointはlayout 2である。immutable context.h5に元case/dataのhash、geometry、境界/sourceの意味、basis/unitを一度保存する。結果consumerは元caseやprivate属性を必要としない。

final、反復interaction、first hit、terminal、保存active曲線は別量である。CSV exportと図は同じexact cohortを使う。表示のbounded sampleを統計の母集団へ置き換えない。旧結果の互換readerは作らず、当時の環境でexportするか明示して再実行する。`} />

    {visible("result-table") && <DataComponent id="result-table" title="結果から説明までの受渡し"
      queryId="result_layers" kind="table" sourceRows={results}>
      <DataTable rows={results} searchable={false} compactNumbers={false} label="result layers"
        columns={[{ field: "layer", label: "層" }, { field: "contents", label: "内容" }, { field: "rule", label: "規則" }]} />
    </DataComponent>}

    <Narrative id="complexity" title="一つのownerと一つのengineを維持する"
      queryIds={["design_decisions"]}
      value={`## 一つのownerと一つのengineを維持する

canonical writer/parser、field support/sampler、physics catalog、event、wall lawをそれぞれのownerへ集約する。normal clientが数値validatorや第二solverを複製しない。一般plugin frameworkやDI containerは追加しない。

詳細な局所probeは原因調査の外部証拠として必要な範囲だけ保存する。quality gateの成功、用途別性能、COMSOLの観測量比較、物理妥当性を一つの「認証済み」へまとめない。`} />

    <section className="report-section">
      <Narrative id="dataset-role" title="native証拠と今回の設計snapshotを分ける"
        queryIds={["quality_findings", "case_audit", "trajectory_divergence"]}
        value={`## native証拠と今回の設計snapshotを分ける

model_datasetや原MPHは外部入力・V&Vの証拠であり、coreの物理定義やdefaultではない。11月8日レポートの30 nm Vigusと今回のA/P 10 nmは別caseであり、結果やhashを流用しない。

Brownianでは個々のpathの一致を要求せず、実現source、全母集団、保存window、停止先・時刻・速度・電荷と不確かさを評価する。native RK4/time-key noiseの5/9 weak-limit解釈は、同定した実行方式と独立microcaseに対する明示したcaseであり、有限step・壁eventでの厳密同値や一般defaultではない。

実用的に合う観測量はそのように判断し、残る差を一律に乱数のせいにしない。保存参照の不明なdisappearanceをescapeと同一視しない。このreportにassessment/quality/performance入力がない項目はunavailableとして残し、古い12-caseの値を現行0.4の資格へ昇格しない。`} />
      {visible("quality-table") && <DataComponent id="quality-table" title="今回取込んだ品質証拠と未取得事項"
        queryId="quality_findings" kind="table" sourceRows={quality}>
        <DataTable rows={quality} searchable={false} compactNumbers={false} label="品質証拠"
          columns={[{ field: "severity", label: "状態" }, { field: "finding", label: "観察" }, { field: "disposition", label: "扱い" }]} />
      </DataComponent>}
    </section>

    <section className="report-section">
      <Narrative id="implementation" title="現行能力と追加評価を分けて引き継ぐ"
        queryIds={["roadmap"]}
        value={`## 現行能力と追加評価を分けて引き継ぐ

通常のwheel入力→結果利用と、native再計算を伴うV&Vを別の入口へ分離する。追加model、3D、moving geometry、全材料・浮遊導体、GPUにはそれぞれ独立した物理・数値・性能の証拠が必要であり、設定名だけを先行追加しない。`} />
      {visible("roadmap-table") && <DataComponent id="roadmap-table" title="現行範囲と次の完了条件"
        queryId="roadmap" kind="table" sourceRows={roadmap}>
        <DataTable rows={roadmap} searchable={false} compactNumbers={false} label="現行範囲"
          columns={[{ field: "phase", label: "区分" }, { field: "name", label: "範囲" }, { field: "exit", label: "完了条件" }]} />
      </DataComponent>}
    </section>

    <Narrative id="start-gate" title="利用者が独立に確認すること"
      queryIds={["physics_scope", "roadmap"]}
      value={`## 利用者が独立に確認すること

自分のgeometry/field/CSV源のunit・basis・ID・supportと物性authorityを固定し、modelsで要求を調べ、create/checkで準備する。dt、field解像度、Brownian path深度、観測clockは別々に選ぶ。worker数を物理の調整ノブにしない。

結果をanalyze/export/viewで読み、失敗coverageと対象cohortを確認する。装置の付着fluxや汚染量を求める場合は、計算粒子数だけでなくsource rate/weight、surface capture physicsと実験比較を別途用意する。`} />

    <RichNarrative id="report:disclosure" className="report-disclosure" label="文書注記を編集"
      value="このsnapshotはsource-boundの現行設計と、明示して取込んだ実行証拠を説明する。再生成scriptはanalysis/build_report_snapshot.py、authority文書とcatalog、任意のassessment/quality/performance入力はhashで対応付ける。アプリID、既存block ID、タイトルと保存済みpresentationを保持し、表示編集がreviewed rowsを変更しない。" />
  </article>;
}
