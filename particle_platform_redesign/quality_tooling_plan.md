# 品質と配布の確認

## 環境と標準gate

solverの独立uv projectとlocked依存を使う。Ruffはformat/lint、Pyreflyは型、import-linterは依存、Radonはcomplexityを所有する。
root環境、重複lint/type/architecture suite、品質runner frameworkを追加しない。

```console
uv lock --check
uv run --locked ruff format --check src tests tools scripts examples
uv run --locked ruff check src tests tools scripts examples
uv run --locked pyrefly check --summarize-errors
uv run --locked lint-imports
uv run --locked python scripts/check_complexity.py
uv run --locked python -m pytest tests/verification tests/scenarios -q
```

変更関数は原則CC10以下、数値上の理由がある場合も15以下。薄いhelper、blanket ignore、baselineで失敗を隠さない。
通常clientをsrc全体のgateへ入れ、既存依存契約でcore/APIからclient/nativeへの逆依存を禁止する。
現行性能clientも同じPyrefly対象に含め、prepare/state/metadataの変更による入口の陳腐化を検出する。

## 確認する性質

| 層 | 必須の確認 |
|---|---|
| verification | 解析/製造解、保存則/極限、独立force/G/J、OU full covariance、support/適用域、geometry/event、source identity |
| scenarios | 公開API/CLI、全現行機能の代表組合せ、failure/coverage、保存/slab/workers/resumeの科学identity |
| portable V&V | 合成入力の正規化/比較/producer/viewer。native asset/ライセンス不要 |
| 性能 | 同科学条件のcold/warm/end-to-end、solver-owned planとOS RSS、process/出力/早期終端を区別 |
| 配布 | source/wheel/installed bytes/METADATA/resources、空cwdで通常一巡、console/module/Python |
| report/説明 | 実行値/単位/分母/scope/hash、missingと0、同reviewed入力の再構築 |

件数は成功条件ではない。構造固定/旧入口/重複assertは置換し、独立oracle/RNG golden/未知revision拒否/resume/bounded出力/実I/O障害注入は維持する。
数値評価は判定前提の反例も扱う。nominal dtだけ異なる同一window partition、窓外だけの細分、
非一様gridの次数、未知clock revisionと不足coverageを、数値差が小さいだけで認定しない。
ResultViewの不変性とJSON用documentの意味を確認し、内部dict/list型を固定するassertへ戻さない。

## installed wheel

`uv build --wheel`とlocked runtime-only依存＋そのwheelを使う。
既存core smokeに加え、空cwd・checkout/PYTHONPATHなしで配布例と公開clientだけから一巡する。
新CSV geometry/field→最初のinternal/surface source→create→check→run→inspect→JSON/CSV→形状付きSVGを要する。
XY/RZ、cache、縮約電場、pending/zero-event/escaped suffix、元caseなしcontext、worker変更resumeを含める。
context欠落/改変、旧result3/partialを拒否し、上書き/migrationをしない。
Windows/Linuxで実console/module/Pythonを確認し、host policy拒否は成功にも数値失敗にも読み替えない。

## CIと証拠

solver変更は全品質/科学/配布、portable V&V変更はasset不要検査へ、docs/skills/reportは軽い参照/metadata/data接続検査へ接続する。
文書変更へ重いnative campaignを強制しない。
source SHA/dirty map、case/data/result identity、command/環境、実行/未実行を保存し、巨大trace/raw MPHをGitへ入れない。
過去資格は元source/条件に限定する。schemaだけでnativeを再計算せず、候補再生成は元seed/window/clockを保持する。
利用者/実装者/科学的妥当性の独立レビューで必須指摘を修正し、影響確認をやり直す。
