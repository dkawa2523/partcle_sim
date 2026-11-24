# 変更・品質・配布の確認

本書は変更作業と確認手順のownerです。安い確認から必要な範囲へ進み、実装・文書の変更を科学的資格の更新と混同しません。
安定した実装制約は[AGENTS](AGENTS.md)、責務と影響先は[architecture](architecture_proposal.md)、
数値比較の前提は[numerics](solver/docs/numerics.md)と[V&V methodology](vv_methodology.md)が所有します。

## 一つの変更workflow

1. 利用目的、変更する事実のowner、影響する公開経路、置換・削除対象を確定する。
2. 現行実装と未実装案を区別する。新model/state/methodの採否には、必要field・状態・適用域、独立reference、容量影響を[判断履歴](solver/docs/decisions.md)へ簡潔に残す。
3. 既存の一つのproduction経路へ実装し、呼出側・例題・設定・説明を合わせる。旧経路と専用確認を同時撤去し、独立oracleは残す。
4. 下表で影響する確認を選び、原因を表す最小caseから実行する。数値変更には解析解・manufactured case・独立referenceと明示したbudgetを使う。
5. hot pathの変更は同じ科学条件で時間・memory・科学identityを測る。source、入力、環境、時計、seed、出力・worker条件を記録する。
6. 利用者・実装者・科学的妥当性の観点で独立に確認し、根拠のある必須指摘を直して影響する確認をやり直す。
7. owner文書と実行receiptを整合させる。実行済み・未実行・外部条件と再開条件を分け、過去のPASSを新sourceへ付け替えない。

新しい品質runner、承認framework、広い診断/契約suiteをこのworkflowのために追加しません。
既存の確認で足りる場合は再利用し、testを書いたことと実行したことを区別します。

## 環境とtool

`solver/pyproject.toml`と`uv.lock`をauthorityとし、独立uv projectから実行します。
通常は`uv run --locked`、CI同期は`uv sync --locked`です。`--frozen`で鮮度確認を省略せず、
依存upgradeとfeature変更を混ぜません。root環境、pip/Poetry/Conda、手書きrequirementsや`uvx`の都度取得を並存させません。

Ruffがformat/lint/import整形、Pyreflyが型、import-linterが依存方向、Radonがcomplexityを所有します。
重複toolや同じ依存規則のpytest suiteを増やしません。通常clientと性能clientも既存の設定対象に含めます。
変更関数は原則CC10以下、数値上の理由がある場合も15以下とし、既存checkerは16以上を拒否します。
MIは参考値です。薄いhelper、blanket ignore、baseline、empty facadeでgateを通しません。

## 変更に応じた確認

| 変更 | 必要な性質と確認範囲 |
|---|---|
| 文書・技能・reviewed reportのみ | 現行リンク、role/handoff、用語・単位・分母・source/scope/hash、snapshot再現、必要なら公式buildと影響するUI。数値再計算やnative campaignは追加しない |
| 入力・公開client | 自分のCSV/源→create/check/runと結果再利用、不正入力・実I/O中断、exact ID、完成物の公開。元caseなしのanalysis/export/viewも確認 |
| 物理・積分・field/event | 独立式/極限/保存則、support/適用域、連成rate/Jacobian、解析/製造解、OU共分散、時間・場・event/pathの独立resolution |
| state・出力・並列 | 固定ID/RNG、accepted commit、prefix/late error、slab/workers/output/resume identity、bounded容量、途中死と実I/O障害 |
| 外部V&V | native asset不要のproducer/正規化/比較試験。live生成・保存値比較・数値資格・実験validationを分離 |
| 性能・配布 | 科学条件を保持したcold/warm/end-to-end、owned planとRSS、clean wheel bytes・resource・metadata、空cwdでの通常一巡 |

`verification / scenarios / performance`の三層で、数学的性質、公開利用、測定を分けます。
件数・coverage率は成功条件ではありません。private名・呼出順・directory構造の固定や旧入口だけのassertは更新し、
独立式/RNG golden/未知revision拒否/resume/bounded出力/実I/O障害を確認する能力は維持します。
同じwindow partition、窓外だけの細分、非一様grid、未知clockと不足coverageを、小さい差だけで収束認定しません。

## 実行コマンド

以下は`solver/`から実行します。文書・技能・snapshotのみの変更には既存の軽い入口を使います。

```console
uv run --locked python scripts/check_documentation.py
uv run --locked python -m pytest tests/verification/test_documentation.py ../report_app/tests/test_reviewed_snapshot.py -q
```

数値・通常client・検証実装を変更した場合の標準gateは次です。関連する小型確認を先に通し、必要な全体確認へ進みます。

```console
uv lock --check
uv run --locked ruff format --check src tests tools scripts examples
uv run --locked ruff check src tests tools scripts examples
uv run --locked pyrefly check --summarize-errors
uv run --locked lint-imports
uv run --locked python scripts/check_complexity.py
uv run --locked python -m pytest tests/verification tests/scenarios -q
```

reportの公式buildとUI確認は[report README](report_app/README.md)と専用AGENTS、測定入口は
[performance README](solver/tests/performance/README.md)、外部試験は[V&V tools](solver/tools/vv/README.md)を使います。
全performance suiteや100万粒子を通常変更ごとに繰り返しません。新しい変更、失敗、未解決の懸念がない限り、合格済み確認を拡大・反復しません。

## 配布とCI

`uv build --wheel`とlocked runtime-only依存へそのwheelを入れ、source/wheel/installed bytes、METADATAと同梱resourceを照合します。
既存のrelease smokeとinstalled workflowで、checkout/PYTHONPATHなしのconsole/module/Python、全配布例、
自分のCSV geometry/field・最初のinternal/surface源、check/run、結果だけの解析/export/view、processとcleanupを確認します。
旧形式・欠損context・改変・不完全partialを成功扱いにせず、上書きやmigrationはしません。

[solver CI](../.github/workflows/solver-release.yml)と[docs CI](../.github/workflows/platform-docs.yml)の実際のtrigger/stepを確認します。
現行のsolver CIは`solver/**`へ広く反応するため、文書変更でも起動し得ます。上表の必要確認とCIのtrigger範囲は別です。
Windows/Linux、clean install、cold JIT、障害注入、規模matrixは変更・release/manualの目的に対応付けます。
host policyによる拒否を成功にも数値失敗にも読み替えません。

source SHA/dirty map、case/data/result identity、command/環境、実行と未実行をreceiptへ残します。
巨大trace/raw MPHをGitへ足さず、元native観測・実験資料は保持します。Git push、remote CI、tag、Release/registry公開は別の実施事項であり、local passから推定しません。
