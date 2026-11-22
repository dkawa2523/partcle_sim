# 利用から結果再利用までの基盤再構成

CLIとPythonの通常フローを一つのwheelへまとめ、入力・数値処理・実行資源・結果解釈の責務を整理した。
変更前は`ade0c8e700f594d181f098259dddb4dd2cb778ac`。旧archiveは変更・importしていない。
最初の操作は[利用README](../../README.md)、現在の責務は[architecture](../../../architecture_proposal.md)、
既存要求と条件付き拡張は[実装計画](../../../implementation_plan.md)が所有する。

## 利用者に提供する流れ

```text
models → init / inputs / create → check → run → inspect → analyze / export / view
```

`chamber_particles`は`load_case/check_case/simulate/open_result`を公開し、
`chamber_particles_tools`は通常の入力作成・場変換/生成・解析・表示と唯一のCLIを所有する。
consoleと`python -m chamber_particles`は同じdispatcherへ接続する。
Windows/Linuxのruntime-only wheelを空cwdから使い、独自CSV、最初の内部/surface source、cache、
縮約電場、全5例題、元caseを移動した後の結果利用を実行した。

## 全体の対応と実行済み範囲

| 領域 | 実装・整理 | 実行した確認 |
|---|---|---|
| 公開入口・設定 | catalog由来のモデル/field metadata、実行と同じcheck | 27モデル/200 field記述をbindingへ照合。両OSで13 CLI操作、Python、5例題 |
| 入力・source | CSV4、source2、recipe1。独立物性、exact-ID継承、内部/surface、単一publisher | sourceなしから独自ケース、解析的な加速とhit、cache同値、surface/原子性/不正入力拒否 |
| 数値・容量 | field数値policyを一箇所へ、oracleはproduction外へ。具体的state/workspaceが容量のowner | 41 JIT関数、87配列構成、9 prepare例の独立baseline照合 |
| 実行資源 | engineはprepare/親時計/writer、transportはslab、executionはworker/共有transport/ACK | 実process、worker変更resume、異常終了/cleanup、出力/slab/worker科学identity |
| 結果・再開 | 必須context付きresult4。親hash・canonical identity・checkpointを結線 | 完成/recovery破損拒否、context非再生成、独立8例とresume4例、I/O障害注入 |
| 解析・export・表示 | interaction/first material hit/terminal/final capture、cohort・重み・分母を分離 | frameなし曲線、pending/欠損/periodic/finite、形状/equal aspect、CSV、SVG表示 |
| 外部V&V | native抽出/比較をcore外へ、通常producerの旧CLI/moduleを撤去 | COMSOL不要101 tests。A/P各40実現を元seed/window/clockで全量再計算 |
| レポート | reviewed snapshot再生成、app/block/presentationと234保護ファイルを維持 | 公式Linux prebuilt build、desktop/390px、出典overview/data preview。最終buildはreport記録に別掲 |
| 説明・技能・契約 | README/AGENTS/所有文書/4技能、6依存契約、軽量CI | 33文書/4技能、snapshot/helper7 tests。予定commitだけの隔離checkoutでも同検査と234保護検査 |
| 性能 | stale revision/廃止helper/誤ったnested JIT counterを整理 | P10/P11小型実行、4 harness型検査。影響するmatrix/process測定はperformance記録に別掲 |

1,201 testsが同一sourceで一括成功し、portable V&Vは101 tests成功した。
lock、Ruff format/lint、configured Pyrefly、変更4性能harnessのPyrefly、6 import契約、CC<16が成功した。
不正event budgetの拒否試験にはoverflow warningが1件あり、試験自体は成功した。
実行値・source hash・初回失敗・配布内容は[checks/verification.json](checks/verification.json)に保存する。

初回全体実行の3失敗は、optional設定表記が異なるcaseのcontext親hashまで同一とする比較と、
2件の旧memory revision固定だった。物理payload/容量/geometryの検査を維持して確認を更新し、全体を再実行した。
未知schema・破損・適用外を受理する緩和は行っていない。

## COMSOL参照との対応

[scienceの再生成記録](science/README.md)に観測値・図・比較の判断を保存した。
全80結果でfinal、release、boundary、failure、series、全frame、全probeが変更前とbitwise一致し、failureは0だった。
新result4から再集計したA/P各15群の指標と空間比較も以前と同一だった。再計算を標本数の追加とは数えない。

各287粒子の大局的な輸送、総停止率、代表terminal時間、人口曲線は適切に対応する。
境界33の停止割合差−4.72/−4.07 ppと、100 µsのsurvivor速度差+13.5/+17.5%は留保を維持する。
乱数pathの一致、formal equivalence、実機adhesion/fluxの認定には拡張しない。
seed21–24の30msと25–60の5ms、全観測clockを保持し、nativeを再solveしていない。

## 独立レビューと整理

[性能の記録](performance/README.md)には全8条件と1万粒子/320macroの実数・条件を保存した。
新旧の1-worker public時間は28.487/29.913秒、4-workerは18.310/18.184秒で、全science payloadは一致した。
短いwarm条件では時間が増えた例もある。1観測の記述値から一律の高速化・退行・scaling資格を認定しない。

利用者/実装者と科学的意味を別担当が確認し、最後に全体を独立レビューした。
必須指摘だったparent identity、ASCII JSON、電場非収束、installed CLI網羅、軽量CI、
未配布の履歴リンク、撤去済みREADME参照を修正して影響確認をやり直した。
[全体レビュー](checks/final_platform_review.json)と[科学的なclientレビュー](checks/independent_clients_review_receipt.json)は、
担当が実行したこととrootの確認を区別している。

旧normal toolsのpackage/CLI、旧core CLI、手書きQuick Start writer、result3の文書・reader経路は残していない。
入力/data3、checkpoint配列schema2を維持するが、旧result3/partialは現行readerで読まない。
旧結果は当時のreader環境と一緒に保全し、必要なexportまたは明示的な再計算を行う。

## 条件と未認定範囲

大量の入力・full resultsは外部`.artifacts/platform_restructure_20261123/`に保全した。
同梱証拠だけでnative campaignの再実行や旧resultの読込はできない。必要な引渡しはscience READMEに記す。
過去の資格を新sourceへ流用せず、完全な10^5/10^6 release matrix、一般SDE/path/mesh収束、
新native solve、実機予測の認定を今回の成功に含めない。既存の条件付きbacklogは実装計画へ保持した。
remote CIの実行状態はlocal gateから推定しない。
