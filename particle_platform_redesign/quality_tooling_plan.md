# 品質ツールと検証gate

## 0. 責務

環境と静的品質は次の五toolを使う。設定とversionのauthorityは独立solverの`pyproject.toml`/`uv.lock`とする。
本書へ初期dependency例や完了済みbootstrap手順を複製しない。

| tool | 所有する責務 | 所有しない責務 |
|---|---|---|
| uv | Python、dependency、lock、環境、command実行 | 数値testの意味、独自workflow |
| Ruff | format、lint、import整形 | 型、architecture、循環的複雑度 |
| import-linter | 少数の安定したimport方向 | private symbol、file配置、全設計判断 |
| Pyrefly | Python静的型 | shape/unit完全体系、物理適用域、runtime検査 |
| Radon | 関数complexityの検出・上限 | 行数競争、物理妥当性、module分割の自動判断 |

pytestは解析解・manufactured・収束・公開scenarioの実行器である。
静的gate合格を数式、first-hit、COMSOL一致、性能の証明へ読み替えない。
Nox/Tox、別formatter/type checker、baseline、巨大quality runnerを重ねない。

## 1. uv

Pythonは3.12系列。build backendはuv_build、YAML parserはPyYAML、既存CPU数値実装はNumPy/Numbaである。
正確なversion/rangeはproject/lockへ一元化し、dependency upgradeとfeature変更を混ぜない。

```console
uv sync --locked
uv lock --check
uv run --locked python -c "import chamber_particles"
```

- 通常実行は`uv run --locked`、CI同期は`uv sync --locked`。`--frozen`を標準にしない。
- dependency追加/削除は`uv add`/`uv remove`。pip、Conda、Poetry、手書きrequirementsを並存させない。
- project toolをuvxで都度取得せず、lockされたdev dependencyを使う。
- `.venv`、JIT/cache、build outputをcommitしない。`uv.lock`はcommitする。
- 旧root project/環境を復元またはactivateしない。

## 2. Ruff

format/lint/import整形を一元化する。Ruff C90は選ばずcomplexityをRadonへ分ける。

```console
uv run --locked ruff format --check src tests tools scripts examples
uv run --locked ruff check src tests tools scripts examples
```

CIはfixしない。開発時のformat/auto-fixは編集対象へ限定する。
blanket noqa、directory-wide ignore、巨大per-file-ignoreを作らず、必要な最小行へruleと理由を書く。
lintのためだけに数式を薄いhelperへ分割しない。外部export/generated evidenceはscan対象にしない。

## 3. import-linter

既存の高価値三契約を維持する。実際の契約module一覧はproject設定が所有する。

1. 低位numericsからAPI/engine/outer I/Oへの逆依存禁止。
2. integratorからgeometry/event/boundary応答への依存禁止。
3. physicsはsample済みprimitiveだけを受け、field locate/geometry/I/Oへ依存しない。

```console
uv run --locked lint-imports
```

新ownerが必要となった変更でだけ契約を更新する。契約のためのempty moduleは作らない。
facade、re-export、TYPE_CHECKINGで違反を隠さず、同内容のpytest architecture testを増やさない。
ignoreは最小とし、不要になったignoreは同じ変更で削除する。

## 4. Pyrefly

型検査の唯一のtoolで、baselineを作らずerror zeroを維持する。

```console
uv run --locked pyrefly check --summarize-errors
```

型付けは三公開API/例外、case/prepared state、physics/stage/proposal/event/resultの境界、array ownerに行う。
unit/shape/適用域を巨大型体系にしない。名称、prepareの一回の検査、数値verificationがそれを所有する。
Anyやcastでruntime不確実性を隠さず、Numba/library境界の不足だけを局在化する。
ignoreは最小行へerror codeと理由を書く。upgradeでerrorが出たらその変更で修正するかupgradeを戻す。

## 5. Radon

```console
uv run --locked radon cc src/chamber_particles tools -s -a
uv run --locked radon mi src/chamber_particles tools -s
uv run --locked python scripts/check_complexity.py
```

変更functionは原則CC≤10、数値kernel/単一state transitionとして一体が明確な場合でも≤15。
CC≥16を既存`check_complexity.py`で拒否する。MIは参考値でgateにしない。
scriptはRadon thresholdのexit statusだけを補い、tool集約、baseline、HTML、独自scoreを持たせない。

scoreだけの改善、薄いmanager/facade、branchの名前替えをしない。
file分割は行数でなく独立変更理由から決める。hot kernelの分割は同じ入力で速度/memoryも確認する。

## 6. 通常changeのgate

solver directoryから安い順に実行する。

```console
uv lock --check
uv run --locked ruff format --check src tests tools scripts examples
uv run --locked ruff check src tests tools scripts examples
uv run --locked pyrefly check --summarize-errors
uv run --locked lint-imports
uv run --locked python scripts/check_complexity.py
uv run --locked pytest tests/verification tests/scenarios -q
```

独自runnerで包まず開発者とCIが同じCLIを使う。PowerShellのgate stepは一native commandで終了させ、
後続commandの成功で失敗exit codeを上書きしない。Quick Start、build、同期、install、smokeも別stepとする。
変更に適切な数値/scenarioを実行し、一度合格した重いmatrixを根拠なく繰り返さない。
文書だけの変更に実装を写したtestを追加しない。

## 7. releaseと用途評価

### 7.1 Windows/Linux入口

既存`solver-release.yml`が両OSの標準gate、warm P14 smoke、wheel、runtime-only clean installと
三API/console/moduleの実workers=2 smokeを所有する。新release runnerは作らない。
license/外部dataset不要のadapter、builder、field cache、meaning/boundary、曲面収束、F02入力試験は
portable suiteで実行する。歴史assetが必要な全COMSOL tools試験はlocal gateで、asset欠損をskip成功に変えない。

clean runtimeのPythonから既存smokeを実行する。

```console
python -m tests.release_smoke --wheel <wheel> --json <receipt>
```

環境同期/installはuv手順が所有する。smokeはsource/wheel/installed SHA、READMEとMETADATA、lock、
runtime-only dependency、三API/console/module CLI、complete resultを照合する。
実workers=2の32粒子全公開記録を同じ入力のserial参照とbyte比較し、API終了後live childゼロ、
CLI timeout/正常終了/空stderrを確認する。console launcher拒否をmodule fallback/skipで成功扱いにしない。
shared root unlink・実child突然死・checkpoint/resumeは既存cause scenarioが所有する。

### 7.2 性能・memory

release/manualはcold JIT、failure injection、10^4/10^5/10^6の用途付き証跡を追加する。
共有runnerの通常CIへ重いmatrixを重複させず、絶対秒数だけでfailさせない。
同一hardware/input/dt/seed/output、複数warm median、cold、科学identity、memory/cleanupを記録する。
process startup/JIT/IPC/writer/shutdownを省いた時間を公開APIのspeedupとして扱わない。
owned array budgetとsampled private/WSを区別し、共有pageをOS memory削減へ読み替えない。

0.3.0 v62の既存local qualification、C3/C4 source binding、Windows full tools/Linux portable subset、
installed-wheel結果は[性能文書](solver/docs/parallel_execution_plan.md)第9.27節が所有する。
Case A/P full-stepとP14-U 320stepの追加scopeは第9.28/9.29節へ分ける。
READMEなど配布本文を変更した場合は最終本文でwheel/METADATAを確認し、以前のwheel byte一致を継承しない。

### 7.3 COMSOLと公開操作

実COMSOL比較は外部`tools/vv/comsol`が所有し、licensed solveをcore PRの常時gateにしない。
保存native参照のhash、実方程式/field/release/wall、登録条件を照合し、未評価層を合格scopeへ足さない。

Git branch push、remote CI、tag、Release/registry公開は個別の実操作である。
source commit/version、実行したgateとwheel identityを結合し、receiptを保存する。
local Linux/WSL結果をremote CIの成功、branch pushをpackage registry公開へ読み替えない。

## 8. 変更を止める条件

- 同じ検査を新validator/diagnostic frameworkへ複製する。
- 旧code/settings/test/docがproductionに残る。
- private helper、file配置、mockだけで構造を固定するtestを増やす。
- coverage率やlint合格で数値verificationを代替する。
- 数値変更とtool upgrade/全体機械formatを混ぜる。

必要な検査は標準CLI、原因を表す最小数値回帰、公開scenario、用途付き性能で表現する。
追加補助scriptは標準CLIでは得られないexit statusが必要な場合に限り、同等の旧処理を削除する。

## 9. 公式仕様の参照先

- [uv project/lock/sync](https://docs.astral.sh/uv/concepts/projects/sync/)
- [uv build backend](https://docs.astral.sh/uv/concepts/build-backend/)
- [Ruff configuration](https://docs.astral.sh/ruff/configuration/)
- [Ruff formatter](https://docs.astral.sh/ruff/formatter/)
- [import-linter configuration](https://import-linter.readthedocs.io/en/stable/get_started/configure/)
- [import-linter contracts](https://import-linter.readthedocs.io/en/stable/contract_types/layers/)
- [Pyrefly installation/CLI](https://pyrefly.org/en/docs/installation/)
- [Pyrefly configuration](https://pyrefly.org/en/docs/configuration/)
- [Radon CLI](https://radon.readthedocs.io/en/latest/commandline.html)

version依存の設定を変更する時に、lockした実versionと公式仕様を確認する。
