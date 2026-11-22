# 保存結果を説明するレポート

この既存Data appは、現行0.4.0の利用フロー・責務と明示的に取込んだ実行証拠を説明するconsumerです。
solverの実行・比較判定・新しいnative抽出は行いません。app ID、タイトル、block IDsと保存済みpresentationを保持します。

sourceとreviewed snapshotの更新・検査を行い、公式のprebuilt経路で再ビルドします。Windows Nodeでは
下記のpath検査が停止しますが、同じinstalled scriptをWSL/Linux Nodeから実行したprepareは成功しました。
最終実行receiptを取込んだ公式prebuilt buildとoffline exportが成功しています。
desktop・390px狭幅、性能表の続き、品質の実行値と認定外の行、source overview/data previewを確認しました。
[最終build・表示・保護検査の記録](../solver/evidence/platform_restructure_2026_11_23/checks/report_verification.json)を参照してください。

## snapshotを再生成する

独立solver projectから実行します。

```console
uv run --locked python ../analysis/build_report_snapshot.py --output ../report_app/src/data.json --assessment evidence/platform_restructure_2026_11_23/report_inputs/assessment.json --quality evidence/platform_restructure_2026_11_23/report_inputs/quality.json --performance evidence/platform_restructure_2026_11_23/report_inputs/performance.json --build-status complete
uv run --locked pytest ../report_app/tests/test_reviewed_snapshot.py -q
```

このcommandは同梱したsource-boundのreviewed projectionsで今回のsnapshotを再生成します。コード変更後は実行証拠も更新し、過去の成功を現行sourceへ付け替えません。
`--assessment/--quality/--performance`を省略すると、その実行証拠をunavailableとする設計snapshotを作ります。
exporterは既存`src/data.json`からidentity、title、filters、presentation等を保持し、既存10 query IDsのrows/sourceを更新します。
古い固定4-query exporterを置き換え、authority文書・catalog・producerのSHA-256を保存します。
`generatedAt`はsnapshot作成時刻です。共通の科学的evidence cutoffがないため`report.asOf`はnullで、測定日を捏造しません。

任意の実行証拠は、読んで確認したprojectionをJSONで渡します。

```json
{
  "queries": {
    "quality_findings": {
      "rows": [
        {"severity": "not_tested", "finding": "説明可能な観察", "disposition": "必要な確認"}
      ],
      "source": {"label": "実際のreceipt", "files": ["相対pathと別途配布した証拠"]}
    }
  }
}
```

- `--assessment`は`case_audit`と`trajectory_divergence`だけを更新します。
- `--quality`は`quality_findings`だけを更新します。
- `--performance`は`implementation_work_packages`の明示した証拠rowsを更新します。

rowsは各consumerの列契約に合わせます。sourceには実ファイル、scope、単位、母集団、実行条件と制約を記録します。
exporterは任意の実行logから成功件数や科学的判定を推測しません。未提供の証拠はunavailable/nullで、ゼロではありません。
過去の12-case audit、30 nm Vigusと今回のA/P 10 nmを、現行版の全認定へ付け替えません。

## authored boundaryとビルド

編集領域は`src/content/`と`src/data.json`です。保護された234 runtimeファイル、manifest、package lock、chrome、
source inspector、autosave、publicationを変更せず、distは公式buildで再生成します。詳細は[AGENTS.md](AGENTS.md)を参照してください。
更新前のowned sourceとprotected/distのhashは、solverのignored artifact
`.artifacts/platform_restructure_20261123/report_before/`に保全しています。

公式のinstalled Data commandをbundled Nodeから実行します。

```console
NODE DATA_PLUGIN/output_formats/html/scripts/data-app.mjs prepare --project-dir report_app
NODE DATA_PLUGIN/output_formats/html/scripts/data-app.mjs build --project-dir report_app --separate-data
```

Windows Nodeのprepareは`Invalid prebuilt Data runtime: Invalid prebuilt source path: assets\\visualize-data\\assets\\cdn.json`で停止しました。
同じinstalled pluginと同じappをWSL/Linuxの公式Nodeから呼ぶと、prebuilt prepareが成功しています。
OSのpath表現を避けるこの経路は、rendererの差替え、guardの緩和、`--source`、Vite/node_modulesのinstallを必要としません。

```console
wsl NODE_LINUX DATA_PLUGIN_LINUX_PATH/output_formats/html/scripts/data-app.mjs prepare --project-dir APP_LINUX_PATH
wsl NODE_LINUX DATA_PLUGIN_LINUX_PATH/output_formats/html/scripts/data-app.mjs build --project-dir APP_LINUX_PATH --separate-data
```

pathは同じWindows sourceを指すWSL側のpathに置き換えます。公式split buildのHTML/snapshot/runtime identityは
`dist/data-app-build.json`、source・protected identityと確認範囲はsolverのignored artifact
`.artifacts/platform_restructure_20261123/report_content_verification.json`へ記録します。
Windows側の失敗をLinux側の失敗へ広げたり、中間buildを最終receipt更新後の確認へ流用したりしません。
protected runtime、manifest、既存presentationを維持します。source buildは今回の回復に不要です。

source/dataの作業完了時に`buildStatus: complete`へ更新します。これはauthoringの状態であり、未実施のbuild/visual QAを成功にしません。
previewを新しいverified buildへ更新するまでは旧内容・旧progress表示が残る可能性があります。公開・共有変更は実施していません。

## サーバーなしで閲覧する

検査済みsplit buildから、公式exportで単一HTMLを作れます。今回の出力は約2.49MB、CDN不要です。
この生成物はGitへ同梱せず、sourceとreviewed inputsから再生成します。

```console
NODE DATA_PLUGIN/output_formats/html/scripts/data-app.mjs export-offline --project-dir report_app --output report_app/.data-app-offline/exports/report.html
```

`.data-app-offline/exports/report.html`をブラウザで開くと、保存されたデータと出典をローカルで閲覧できます。
