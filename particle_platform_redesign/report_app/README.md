# 保存結果を説明するレポート

現行0.4.0の利用フロー・責務と、明示的に取り込んだ科学・品質・性能の証拠を説明するconsumerです。
solver実行や比較判定は行いません。app ID、タイトル、block IDs、presentation、共通runtimeを維持します。

## データを更新する

独立solver projectから実行します。

```console
uv run --locked python ../analysis/build_report_snapshot.py --output ../report_app/src/data.json --assessment evidence/retirement_2026_11_23/report_inputs/assessment.json --quality evidence/retirement_2026_11_23/report_inputs/quality.json --performance evidence/retirement_2026_11_23/report_inputs/performance.json --build-status complete
uv run --locked python -m pytest ../report_app/tests/test_reviewed_snapshot.py -q
```

reviewed projectionsを読み、現行authority・catalog・producerと入力のSHA-256を保存します。
既存identity、presentation、filters、10 query IDsを保持し、rows/sourceを更新します。
任意の実行logから判定を推測しません。未提供の証拠はunavailable/nullであり、ゼロや過去のPASSを補いません。

各projectionはqueries配下へ、対象queryのrowsと実ファイル・母集団・単位・条件を明示したsourceを持ちます。
assessmentはcase_audit/trajectory_divergence、qualityはquality_findings、performanceはimplementation_work_packagesを更新します。
A/Pの保存比較は元の入力と40実現に限定した科学参照です。新しい計算標本や現行全体の数値認定としません。
generatedAtはsnapshot作成時刻、共通の科学的cutoffがないreport.asOfはnullです。

## ビルドと閲覧

編集領域と公式buildは[AGENTS.md](AGENTS.md)を参照してください。
src/contentとsrc/data.jsonを更新し、234の共通runtimeファイル、manifest、lock、chrome、source inspector、autosave、publicationを維持します。

```console
NODE DATA_PLUGIN/output_formats/html/scripts/data-app.mjs prepare --project-dir report_app
NODE DATA_PLUGIN/output_formats/html/scripts/data-app.mjs build --project-dir report_app --separate-data
```

NODEとDATA_PLUGINはinstalled runtimeの絶対pathを使います。
このWindows環境ではinstalled prebuiltのpath検査にOS依存の問題があるため、
同じsourceをWSL/Linux Nodeから公式scriptへ渡す経路で実行します。
rendererの変更、guardの緩和、source build、node_modulesのinstallは必要ありません。

dist全体をloopback HTTPでserveします。snapshot、HTML、runtimeのidentityはdist/data-app-build.jsonにあります。
公式export-offlineで、検査済みsplit buildを同じappの.data-app-offline/exports/report.htmlへ書き出せます。
dist/offline HTMLはGitへ同梱せず、sourceとreviewed inputsから再生成します。

今回の実行範囲と結果は[整理・実行記録](../solver/evidence/retirement_2026_11_23/README.md)に記載します。
authoringのbuildStatus: completeは未実施のbuild、表示確認、公開を成功にする値ではありません。
公開先・共有設定は変更していません。
