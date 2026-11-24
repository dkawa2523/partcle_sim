# 現行仕様と保存証拠を読むレポート

0.5.0 alphaの利用経路、責務、科学・数値の意味と、取り込んだ実行証拠を説明するconsumerです。
計算や比較判定の入口は[solver README](../solver/README.md)です。
本文は、機能選択 → 自前入力 → check/run → 結果の再利用 → 実装の責務 → 妥当性の順に構成します。

## 現在の説明と実行証拠

本文と設計表は現在のauthority・catalogを参照します。品質・性能・COMSOL比較の数値は、
各receiptのsource、母集団、window、実行条件に限定されます。
文書更新は過去実行のsourceをHEADへ置き換えず、科学計算の再実行を意味しません。
`generatedAt`はsnapshot作成時刻で、共通の科学的cutoffを持たない`report.asOf`はnullです。

10 query IDs、app ID・タイトル・block IDs・presentation・filtersを保持します。
編集領域と公式build契約は[AGENTS.md](AGENTS.md)を参照してください。
共通runtime、chrome、source inspector、theme、manifest、lockは本文整理の対象にしません。

## 内容とsnapshotを更新する

本文は`src/content/report/ReportContent.jsx`、reviewed datasetは`src/data.json`です。
編集開始時は`buildStatus: updating`とし、authority文書・本文・producerを確定してからsnapshotを生成します。
本文だけの修正でも、current authority hashの更新が必要です。

次は`particle_platform_redesign/solver/`から、既存の科学・性能projectionと、
削除完了だけを反映した品質projectionを取り込む手順です。品質の元の計測値は保持しています。

```console
uv run --locked python ../analysis/build_report_snapshot.py --existing ../report_app/src/data.json --output ../report_app/src/data.json --assessment evidence/foundation_2026_11_23/assessment_projection.json --quality evidence/documentation_2026_11_24/quality_projection.json --performance evidence/foundation_2026_11_23/performance_projection.json --build-status complete
uv run --locked python -m pytest tests/verification/test_documentation.py ../report_app/tests/test_reviewed_snapshot.py -q
uv run --locked python scripts/check_documentation.py
```

assessmentは`case_audit`/`trajectory_divergence`、qualityは`quality_findings`、performanceは
`implementation_work_packages`を更新します。省略した種類の実行証拠を再現したことにはなりません。
設計だけを生成する場合は別の出力先を明示します。

```console
uv run --locked python ../analysis/build_report_snapshot.py --existing ../report_app/src/data.json --output .artifacts/report_design_only/data.json --build-status updating
```

省略した比較・品質証拠はunavailable/nullとなり、過去のPASSやゼロで補いません。
performanceを省略した表は設計上の利用フローであり、実測性能ではありません。
`reproducibility.reproduction`と各queryのMethodsには、実際のoptional projection・出力先・statusを
含む引数列を保存します。Python例はsolver directoryから実行し、空白を含むpathも一つの引数として渡します。
外部の測定手順は保持し、producer自身の手順は置き換えます。

### 参照とhashの所有者

producer `analysis/build_report_snapshot.py`がcurrent authorityの正確なroster・path・SHA-256を所有し、
document checkerも同じ`validate_current_authorities`を呼びます。文書変更後の古いsnapshotは拒否されます。
過去の`execution_source`を現行sourceへ一致させる検査は行いません。

projectionは`queries`配下にrowsとsourceを持ちます。parseとhashには同じ一度のread bytesを使い、
表示に必要なrow列、source.files、unavailable/nullを検査します。
`source.files`の文字列は引用です。local fileのbyte bindingが必要なら、
`{location, availability: "bound_local", sha256, execution_source, scope}`を渡します。
relative locationは`particle_platform_redesign/`基準で、存在・hash不一致を拒否します。
`external_reference`と`historical_unavailable`は各scopeを保持し、欠損から自動推定しません。

runtimeには文字列またはlabel/hrefを渡し、構造化したavailability/hash/execution_source/scopeは
`reproducibility.reviewed_evidence`とEvidence flowに残します。履歴の条件や数値を変更せず、
文書の現在形と実行時の測定値を区別します。削除・保持も各整理記録の時点と対象に限定されます。

snapshot全体をserializeした後、同directoryの一時fileをfsyncしてatomic replaceします。
書込み・replace前の失敗では以前のdata.jsonを保持します。

## 公式buildと表示確認

ここからはrepositoryの`particle_platform_redesign/`を作業directoryにします。
次のPowerShell例では3変数をinstalled runtimeとprojectの**絶対path**へ置き換えます。
本文中の`NODE DATA_PLUGIN/...`をそのまま実行する必要はありません。

```powershell
Set-Location C:\path\to\partcle_sim-sim_rev4\particle_platform_redesign
$reportNode = 'C:\absolute\path\to\node.exe'
$reportPlugin = 'C:\absolute\path\to\data-analytics\1.2.2-e.1'
$reportApp = (Resolve-Path .\report_app).Path
& $reportNode "$reportPlugin\output_formats\html\scripts\data-app.mjs" prepare --project-dir $reportApp
& $reportNode "$reportPlugin\output_formats\html\scripts\data-app.mjs" build --project-dir $reportApp --separate-data
```

このworkspaceのWindows実行では、installed prebuiltのasset path検査が拒否する事象を確認しています。
その条件では同じsourceをWSL/Linux Nodeから同じ公式scriptへ渡す経路を利用しました。
別環境では実際のprepare結果で判断します。runtimeやguardの変更、source build、依存追加は不要です。

WSLを使う場合もPowerShellから実行でき、cwdに依存しない絶対Linux pathを3変数へ指定します。
NodeはWSL/Linux用のinstalled binaryを使い、Windows Nodeのpathを渡しません。
このworkspaceで検証したversionは24.22.3です。次のpathは配置例であり、Nodeの一時directoryを要求しません。

```powershell
$reportLinuxNode = '/absolute/linux/path/to/node'
$reportLinuxPlugin = '/mnt/c/Users/USER/.codex/plugins/cache/openai-curated-remote/data-analytics/1.2.2-e.1'
$reportLinuxApp = '/mnt/c/path/to/partcle_sim-sim_rev4/particle_platform_redesign/report_app'
wsl --exec $reportLinuxNode "$reportLinuxPlugin/output_formats/html/scripts/data-app.mjs" prepare --project-dir $reportLinuxApp
wsl --exec $reportLinuxNode "$reportLinuxPlugin/output_formats/html/scripts/data-app.mjs" build --project-dir $reportLinuxApp --separate-data
```

dist全体をloopback HTTPでserveし、desktop/mobileの読順・表・Source・Evidence flow・Copyを確認します。
全10 query、snapshot、HTML、runtimeのbindingは`dist/data-app-build.json`と生成物で確認します。
以前のdistを、新しいsourceに対する表示確認の証拠にしません。

公式export-offlineで、検査した同じアプリをsidecar HTTP server不要の単一HTMLへ書き出せます。

同じPowerShell sessionのWindows変数、またはWSL変数を使います。

```powershell
& $reportNode "$reportPlugin\output_formats\html\scripts\data-app.mjs" export-offline --project-dir $reportApp --output "$reportApp\.data-app-offline\exports\foundation-report.html"
```

```powershell
wsl --exec $reportLinuxNode "$reportLinuxPlugin/output_formats/html/scripts/data-app.mjs" export-offline --project-dir $reportLinuxApp --output "$reportLinuxApp/.data-app-offline/exports/foundation-report.html"
```

dist/offline HTMLはGitへ同梱せず、sourceとreviewed inputsから再生成します。
`buildStatus: complete`はauthoring状態であり、未実行のbuild・表示確認・公開を成功に変える値ではありません。
文書更新の実行範囲は[今回の作業記録](../solver/evidence/documentation_2026_11_24/README.md)、
以前の実装・科学計算は[foundation記録](../solver/evidence/foundation_2026_11_23/README.md)で確認し、
以前のbuildや測定receiptは当時の証拠として保持します。公開・共有設定の変更は別の操作です。
