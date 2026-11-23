# 保存結果を説明するレポート

現行0.5.0の利用フロー・責務と、明示的に取り込んだ科学・品質・性能の証拠を説明するconsumerです。
solver実行や比較判定は行いません。app ID、タイトル、block IDs、presentation、共通runtimeを維持します。

## データを更新する

独立solver projectから実行します。

```console
uv run --locked python ../analysis/build_report_snapshot.py --output ../report_app/src/data.json --assessment evidence/foundation_2026_11_23/assessment_projection.json --quality evidence/foundation_2026_11_23/quality_projection.json --build-status updating
uv run --locked python -m pytest ../report_app/tests/test_reviewed_snapshot.py -q
```

reviewed projectionsを読み、現行authority・catalog・producerと入力のSHA-256を保存します。
projectionのparseとhashは同じ一度のread bytesを使います。row object、表示に必要な列、
source.filesのlist、unavailable/nullをproducerが検査し、document checkerも同じ検査ownerを使います。
完成snapshotをserializeしてから同directoryの一時fileをfsyncし、atomic replaceで公開します。
検査・書込み・replace前の失敗では以前のdata.jsonを保持します。
既存identity、presentation、filters、10 query IDsを保持し、rows/sourceを更新します。
任意の実行logから判定を推測しません。未提供の証拠はunavailable/nullであり、ゼロや過去のPASSを補いません。

各projectionはqueries配下へ、対象queryのrowsと実ファイル・母集団・単位・条件を明示したsourceを持ちます。
`source.files`の文字列は引用として保持します。現在のlocal fileとbyte bindingを要求する場合は、
`{location, availability: "bound_local", sha256, execution_source, scope}`を明示します。
relative locationは`particle_platform_redesign/`基準です。bound_localは存在とSHAを確認し、不一致を拒否します。
外部引用は`external_reference`、現在利用できない履歴引用は`historical_unavailable`とscopeを明示できます。
欠損を自動的に別区分へ変更しません。runtimeのsource.filesには対応する文字列又はlabel/hrefを渡し、
availability/hash/execution_source/scopeは`reproducibility.reviewed_evidence`と既存Evidence flowへ同じ内容を残します。
現行designのauthority hash、過去実行source、今回のbuild identityは別の事実です。
assessmentはcase_audit/trajectory_divergence、qualityはquality_findings、performanceはimplementation_work_packagesを更新します。
A/P比較は入力receiptのsource、seed集合、実現数/各case、windowへ限定します。
今回の実行と以前のsourceに束縛した参照は、各projectionのexecution source/scopeで区別します。
同じseedでの再実行や再利用native参照を新しい独立標本とは数えず、実行済みの範囲から
現行全体の数値認定を推測しません。進行中のsnapshotでは未完了の測定を明示します。
過去のretirement projectionはその当時のsourceに対する履歴であり、変更後の品質・性能のPASSを補いません。
generatedAtはsnapshot作成時刻、共通の科学的cutoffがないreport.asOfはnullです。

recipe 2は完成bundleのsource table全量を明示し、通常bundle producerはbundle.h5/report.jsonをdirectory単位で公開します。
ResultView/ResultCohort、manifest_document/source_identity、macro window・stability boundの説明も現行ownerへ対応させます。
frozen charge/ion-force比較は明示CSVと現行式を使い、ion-forceではrevisionとmatched/analogousを選びます。
性能のC03 relaxation計測はP10の`--scenarios relaxation --relaxation-particles`で選びます。

reportのauthoring完了とfilesystem cleanupの完了を分けます。削除が自動承認reviewで拒否されたscopeは
未実行のまま、正確なblock理由・残存範囲・再開条件をreviewed品質入力へ保存します。
追跡対象の削除から物理fileやbytesの削除済みを推測しません。

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

今回の実行範囲と結果は[実装・実行記録](../solver/evidence/foundation_2026_11_23/README.md)に記載します。
authoringのbuildStatus: completeは未実施のbuild、表示確認、公開を成功にする値ではありません。
公開先・共有設定は変更していません。
