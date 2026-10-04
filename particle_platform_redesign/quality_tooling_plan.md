# 品質ツール導入・運用計画

## 0. 結論

新solverの環境と品質検査は、次の五つを正式な道具として採用する。

| tool | 一つだけ持たせる責務 | 持たせない責務 |
|---|---|---|
| `uv` | Python version、dependency、lock、仮想環境、command実行 | lint、testの意味や独自workflow |
| Ruff | format、通常lint、import整形、明白なPython/NumPy上の不具合 | 型検査、architecture、循環的複雑度 |
| import-linter | 少数の安定したimport依存方向 | file配置、private symbol、全設計判断 |
| Pyrefly | Pythonの静的型検査 | array shape、単位、物理適用域、runtime入力検査 |
| Radon | 関数の循環的複雑度の検出と肥大化の停止 | 行数競争、module分割の自動判断、物理妥当性 |

`pytest`は数値verificationとscenario testの実行器として別に必要だが、品質toolを束ねる独自frameworkには
しない。旧repository rootの`noxfile.py`、`quality_tools/`、大量のbaseline、旧`pyproject.toml`設定は
clean-room solverへcopyしない。

新solverでは、`uv run --locked TOOL ...`をそのままCIと開発者の共通commandにする。Nox、Tox、Poetry、
Black、isort、Flake8、mypy、Pyright、Xenonを重ねない。pre-commitも初期導入しない。入力回数の削減が
実測上必要になった時だけ、同じcommandを呼ぶ薄い入口を検討する。

---

## 1. なぜこの分担にするか

品質toolを増やす目的は、診断件数を増やすことではない。次の失敗を実装前に止めるためである。

- dependencyの追加・upgradeが開発者ごとに異なる
- 数値式の変更にstyle変更が混ざり、差分をreviewできない
- physicsやintegratorがI/O・geometry・COMSOL toolへ逆依存する
- 型の曖昧さにより、粒子状態、field量、event結果を取り違える
- 一つのengine関数へ条件分岐が積み重なり、古い経路が消えない

一方、これらのtoolで次は証明できない。

- 運動方程式、帯電式、力モデルが正しいこと
- 時間積分の収束次数
- first-hitの時刻、facet、event順が正しいこと
- COMSOLその他の外部solverとの一致
- 10^4～10^6粒子での性能とmemory

これらは解析解microcase、収束試験、scenario、performance測定が所有する。lint合格を数値計算の
完了条件に読み替えない。

---

## 2. `uv`を唯一の環境基盤にする

### 2.1 Stage 0で作るもの

`particle_platform_redesign/solver/`に次だけを置く。

```text
solver/
├─ .python-version
├─ pyproject.toml
├─ uv.lock
├─ README.md
├─ src/chamber_particles/
├─ tests/
└─ scripts/check_complexity.py
```

初期Pythonは3.12系列へ固定する。NumPy、h5py、Numbaおよび配布先の対応を確認した上で、対応versionを
広げる変更は独立して行う。Ruffの`target-version`とPyreflyの`python-version`も同じ値にする。

`pyproject.toml`はruntime dependencyと開発dependencyを分ける。

```toml
[build-system]
requires = ["uv_build>=0.11.16,<0.12"]
build-backend = "uv_build"

[project]
requires-python = ">=3.12,<3.13"
dependencies = [
  "h5py>=3.11,<4",
  "numpy>=2.0,<3",
  "pyyaml>=6.0.2,<7",
]

[dependency-groups]
dev = [
  "import-linter",
  "pyrefly",
  "pytest",
  "radon",
  "ruff",
]

[tool.uv]
required-version = ">=0.11.16,<0.12"
```

P00ではbuild backendを`uv_build`に固定し、uvと同じminor rangeでlockする。YAML parserは
PyYAMLに一元化する。別build backend、別YAML parserは並存させない。

実際のpackage versionは`uv.lock`が固定する。uv自身もP00で採用したminor rangeを`required-version`へ
固定し、CI runnerと開発環境の挙動差を避ける。version upgradeは一つずつ行い、そのtoolの出力差分と
数値testを確認する。通常changeへlock全更新を混ぜない。

### 2.2 標準command

```console
uv python pin 3.12
uv sync --locked
uv lock --check
uv run --locked python -c "import chamber_particles"
```

規則は次のとおり。

- `pip install`や手動venv作成を手順に書かない。
- dependency追加・削除は`uv add`、`uv remove`を使う。
- project toolを`uvx`で都度取得しない。必ずlockされたdev dependencyを使う。
- CIでは最初に`uv sync --locked`または`uv lock --check`を実行し、lockを更新しない。
- `--frozen`は標準にしない。lockと`pyproject.toml`の不一致を検出する必要があるためである。
- `.venv`はcommitしない。`uv.lock`はcommitする。
- OS別requirementsや手書きconstraints fileを並存させない。

---

## 3. Ruffの活用規約

Ruffはformat、import整形、通常lintだけを所有する。循環的複雑度の`C90`は選ばずRadonへ一元化する。
Black、isort、Flake8を追加しない。

初期設定は広すぎるrule setを避け、実害のある種類から始める。

```toml
[tool.ruff]
target-version = "py312"
line-length = 100
src = ["src", "tests", "tools"]

[tool.ruff.lint]
select = [
  "E4", "E7", "E9",
  "F",
  "I",
  "UP",
  "B",
  "C4",
  "NPY",
  "PERF",
  "RUF",
]
```

標準commandは次である。

```console
uv run --locked ruff format --check src tests tools
uv run --locked ruff check src tests tools
```

開発者がformatを適用する時だけ次を使う。

```console
uv run --locked ruff format src tests tools
uv run --locked ruff check --fix src tests tools
```

運用規則：

- CIは`--fix`しない。
- blanket `noqa`、package全体ignore、巨大なper-file-ignore表を作らない。
- ignoreが不可避なら最小行へerror codeと理由を書く。
- lintを通す目的だけで物理式を多数の薄いhelperへ分解しない。
- 新rule導入は、既存code全体の機械修正とfeature変更を同じchangeへ混ぜない。
- generated data、benchmark出力、外部exportはscan対象にしない。

---

## 4. import-linterの活用規約

import-linterは、設計書の全行を機械契約へ変換するtoolではない。壊れた場合にsolverの責務分離が
本当に失われる、高価値な境界だけを検査する。

基本設定：

```toml
[tool.importlinter]
root_package = "chamber_particles"
include_external_packages = false
exclude_type_checking_imports = false
```

`unmatched_ignore_imports_alerting`はroot設定ではなくcontract単位のoptionである。P00ではcontractも
`ignore_imports`も作らない。後続stageでignoreが必要なcontractを追加する場合だけ、そのcontractで
unmatched ignoreをerrorにする（現行のdefaultもerror）。

契約はmoduleの実装と同時に追加する。契約を先に満たすためのempty moduleは作らない。v0.1で最大三つを
目安とする。

1. `physics`はgeometry、field locate、engine、output、case formatへ依存しない。
2. `integrators`はgeometry、event、boundary、engine、outputへ依存しない。
3. 低位numericsはAPI、engine、CPU orchestration、outputへ逆依存しない。

外部`tools/`をPython packageにする場合は、coreからtoolsへの依存禁止を追加する。analysisやvisualizationが
private solver moduleを読むことを防ぐ契約は、それらが実装されたstageで追加する。

```console
uv run --locked lint-imports
```

運用規則：

- file一つごとの契約や、directory treeを固定する契約を作らない。
- `ignore_imports`は初期値を空にする。
- 違反をfacade、re-export、TYPE_CHECKING importで隠さない。
- 契約が誤っているならarchitecture文書と契約を同じchangeで修正する。
- import-linterと同じ内容をpytestのarchitecture testで重複検査しない。
- 循環importを解消するために一般的な`interfaces.py`や`contracts.py`を作らない。事実の所有者を直す。

---

## 5. Pyreflyの活用規約

Pyreflyは型検査の唯一のtoolとする。clean-room projectなのでbaselineを作らず、最初のmoduleからerror zeroを
維持する。

```toml
[tool.pyrefly]
project-includes = ["src/chamber_particles", "tools"]
project-excludes = []
search-path = ["src", "."]
python-version = "3.12"
check-unannotated-defs = true
infer-return-types = "never"
preset = "default"
```

```console
uv run --locked pyrefly check --summarize-errors
```

型付けの対象：

- 三つの公開APIと公開例外
- `SimulationCase`、`PreparedRun`、`ParticleState`、`StepProposal`、event、result境界
- physics modelの入力primitiveと出力rate/acceleration
- arrayのdtype、rank、ownerが変わるmodule境界

型に持たせないもの：

- SI単位の完全な型体系
- ndarray shapeを表す巨大generic
- physics modelの適用域
- mesh topologyの正しさ
- runtimeのfinite/support検査

これらは名称、docstring、prepare時の一回の検査、verificationが所有する。

運用規則：

- 新規projectで`pyrefly suppress`やbaselineを使わない。
- `Any`をmodel/coreへ伝播させず、stub不足などの外部library境界へ局在させる。
- ignoreは最小行へ具体的error codeと理由を書く。file全体を無効化しない。
- `cast`でruntimeの不確実性を隠さない。
- Numba kernelを型checkerへ合わせるために不自然なobject階層へしない。typed Python wrapperとの境界を明確にする。
- Pyrefly upgradeで新errorが出た場合、baselineを生成せず、upgrade change内で修正またはupgradeを戻す。

---

## 6. Radonの活用規約

Radonは複雑なfunctionを早く発見するために使う。module数、行数、comment量を最適化するためには使わない。

```console
uv run --locked radon cc src/chamber_particles tools -s -a
uv run --locked radon mi src/chamber_particles tools -s
```

基準：

- 新規・変更functionは原則CC 1～10（Radon A/B）。
- CC 11～15は、数値kernelまたは単一state transitionとして一体の方が明確な場合だけ許容する。
- CC 16以上はCIで拒否する。
- Maintainability Indexは傾向を見るだけで合否に使わない。
- tests、generated code、外部adapterの変換tableはcoreのCC gateから除く。

Radon CLIには本方針の数値thresholdをそのままexit codeへする機能がないため、Stage 0で
`scripts/check_complexity.py`を一つだけ作る。このscriptはRadonの結果からCC 16以上を列挙して非zero終了する
だけとし、複数toolのorchestration、baseline、HTML report、独自scoreを持たせない。

複雑度を下げる時の規則：

- 条件分岐を別名の薄いhelperへ移すだけの修正をしない。
- lifecycle、数値step、event探索、boundary lawなど、異なる変更理由を分離する。
- branch tableやdata-driven dispatchで意味が明確になる時だけ置換する。
- performance kernelを分割する場合は、速度とmemoryを変更前後で測る。
- CC例外台帳を作らない。15以下で説明できなければ設計を見直す。

---

## 7. gateの順序

### 7.1 通常change

安い検査から実行する。

```console
uv lock --check
uv run --locked ruff format --check src tests tools
uv run --locked ruff check src tests tools
uv run --locked pyrefly check --summarize-errors
uv run --locked lint-imports
uv run --locked python scripts/check_complexity.py
uv run --locked pytest tests/verification tests/scenarios -q
```

このcommand列を包む独自runnerは作らない。CI jobにそのまま記載し、開発者も同じcommandを使う。

### 7.2 main/release

- `solver-release.yml`：Windows/Linuxで上記標準gate、warm P14 smoke、wheel、runtime-only clean install、三公開API smoke。
- release/manual：cold JIT、failure injection、10^4/10^5/10^6粒子の正式証跡。共有runnerの通常CIへ重いmatrixを重複させない。
- COMSOL比較：`tools/vv/comsol`の外部V&V。core PRの常時gateにはしない。

性能は共有runnerの絶対秒数だけでfailさせない。同一hardware、同一case、複数回median、peak memory、
event/result identityを記録する。

---

## 8. 実装work packageへの反映

### P00

- uv project、Python pin、`pyproject.toml`、`uv.lock`
- Ruff、Pyrefly、Radonの最小設定
- import-linterのroot設定
- `scripts/check_complexity.py`
- clean installと三公開API import smoke

P00では三APIの名前と公開例外だけをpackage境界に固定し、操作は明示的に
`NotImplementedError`でfail closedする。将来moduleのempty skeletonは作らない。`load_case`はP01、
`simulate` / `open_result`はP04で同じ公開関数の本実装へ置き換え、暫定分岐を同じchangeで削除する。

### P03～P08

- moduleが生まれた時点で対応するimport-linter契約を追加
- public/numerical boundaryの型を同じchangeで追加
- 数値verificationと小型scenarioを品質toolより優先して完成条件にする

### P09～P14

- Numba境界の型とsuppressionを局在化
- hot loopのRadon値を確認し、単一loopとして必要なbranchかreview
- thread/tile/output scheduleのidentity testとperformance測定

---

## 9. 旧品質基盤から継承しないもの

旧repositoryには五toolがすでに存在するが、旧packageの構造とbaseline debtを前提としている。次を新solverへ
移植しない。

- Noxを介した`quality-fast`、`quality-pr`、`quality-nightly`
- `quality_tools.runner`とtool別JSONの巨大集約
- Pyrefly baseline
- 既存違反を固定したRuff/Radon baseline
- private module配置を検査するarchitecture test
- coverage率、mutation、診断snapshotの一括必須化
- 一つの変更を何種類ものsummary/reportへ重複記録する仕組み

必要な検査は五toolの標準CLI、数値verification、公開scenario、性能測定で表現する。新しい補助scriptを
追加する場合は「標準CLIだけでは出せないexit statusか」を説明し、同等の古い補助処理を削除する。

---

## 10. 公式仕様の参照先

- uv project、lock、sync：<https://docs.astral.sh/uv/concepts/projects/sync/>
- uv project command：<https://docs.astral.sh/uv/guides/projects/>
- uv build backend：<https://docs.astral.sh/uv/concepts/build-backend/>
- Ruff configuration：<https://docs.astral.sh/ruff/configuration/>
- Ruff formatter：<https://docs.astral.sh/ruff/formatter/>
- import-linter configuration：<https://import-linter.readthedocs.io/en/stable/get_started/configure/>
- import-linter contracts：<https://import-linter.readthedocs.io/en/stable/contract_types/layers/>
- Pyrefly installation/CLI：<https://pyrefly.org/en/docs/installation/>
- Pyrefly configuration：<https://pyrefly.org/en/docs/configuration/>
- Radon command line：<https://radon.readthedocs.io/en/latest/commandline.html>

toolのversionに依存する設定を変更する時は、上記公式仕様とlockされた実versionを確認する。
