# Chamber Particles 0.4.0

外部の形状・場・粒子源を使い、粒子の運動・帯電・境界イベントを計算するPython 3.12基盤です。
**alpha版**で、通常の入力作成・検査・実行・解析・表示を同じwheelのCLIとPythonから使えます。
対応座標は2-D Cartesian XYとRZ meridional。COMSOLやソースcheckoutは通常利用の前提ではありません。

## 例題を実行する

source checkoutでは、この独立uv projectから実行します。再実行時は新しい出力先を指定してください。

```console
uv sync --locked
uv run --locked python -m chamber_particles models --category electric
uv run --locked python -m chamber_particles init .artifacts/quickstart/input --example xy-ballistic
uv run --locked python -m chamber_particles check .artifacts/quickstart/input/case.yaml
uv run --locked python -m chamber_particles run .artifacts/quickstart/input/case.yaml -o .artifacts/quickstart/result
uv run --locked python -m chamber_particles inspect .artifacts/quickstart/result
uv run --locked python -m chamber_particles analyze .artifacts/quickstart/result -o .artifacts/quickstart/summary.json
uv run --locked python -m chamber_particles export .artifacts/quickstart/result -o .artifacts/quickstart/export
uv run --locked python -m chamber_particles view .artifacts/quickstart/result -o .artifacts/quickstart/view --detail
```

このXY例は1粒子、境界イベント1件、failure 0件、最終`stuck`です。
`case.yaml`に設定、`case.h5`にcanonical入力、`particles.csv`に実現source、`recipe.yaml`に再作成手順が入ります。
CSVやrecipeを編集したら別directoryへ作り直します。

```console
uv run --locked python -m chamber_particles create .artifacts/quickstart/input/recipe.yaml -o .artifacts/quickstart/edited
```

`chamber-particles` console commandも同じdispatcherです。通常出力は整形JSON、`--json`でcompact JSONです。
各command/subcommandで`--help`を使えます。

配布wheelが手元にある場合、空の作業directoryでも次のように使えます。wheel pathは実ファイルへ置き換えます。

```console
uv run --no-project --python 3.12 --with /path/to/chamber_particles-0.4.0-py3-none-any.whl python -m chamber_particles init input --example xy-ballistic
```

以後も同じ`uv run --no-project --python 3.12 --with WHEEL`で`check`、`run`、`analyze`等を呼べます。
これはwheelの利用手順であり、registryやReleaseへの公開済みを意味しません。

| `init --example` | 内容 |
|---|---|
| `xy-ballistic` | XY直進と付着。場は不要 |
| `rz-charged` | RZで固定電荷と一様電場 |
| `xy-time-field` | 固定topologyの線形時間電場 |
| `brownian-wall` | Epstein drag、FDT Brownian、反射壁 |
| `reduced-electrostatic` | 小型RZの縮約電場生成と輸送。編集用`thermal.h5`、`builder.yaml`も保存 |

例題は機能の説明です。粒径、物性、step、壁則を別装置のdefaultにしません。

## 自分の入力を作る

geometry・fieldのcanonical bundleをPythonで作るか、CSVを明示mappingでimportします。
`inputs import-csv`は静的XY/RZのselected nodal scalar/vector、tri3と明示Q1 quadからのP1変換を扱います。
設定format 4の`provider: csv`は自分のCSV、`provider: comsol`は同定済みCOMSOL exportです。
後者だけがdataset/solution/memberのnative lineageを要求します。両者ともunit・basis・domain/side・supportを明記し、
field欠損をゼロや最近傍で埋めません。

```console
uv run --locked python -m chamber_particles inputs import-csv mapping.yaml background.h5 --report import.json
uv run --locked python -m chamber_particles create recipe.yaml -o input
uv run --locked python -m chamber_particles check input/case.yaml --workers 1
```

recipeはformat 1で、`input_bundle`、`sources`、`settings`を持ちます。pathはrecipeからの相対pathです。
`settings`は通常のcase YAML documentで、`case`にはnameだけを指定できます。
publisherがcanonical writerのhashとdata pathを埋め、公開`load_case`と`check_case`を通してから公開します。
利用者がhashを手計算したり、既存ケースをtemplateとして用意したりする必要はありません。

```yaml
format_version: 1
input_bundle: background.h5
sources:
  - csv: particles.csv
    configuration: source.yaml
settings:
  format_version: 3
  case: {name: my_case}
  # motion/time/solver/resources/physics/sources/boundaries/outputを明記する。
  # 完全な編集例はinitで生成したrecipe.yamlを参照。
```

source configurationはformat 2、`kind: table`または`kind: surface`です。
ID、release time、位置またはcanonical facet row/strict-interior parameter、速度、電荷、質量、drag/contact/electric寸法、
排除体積、weight、materialを相関のある実現rowとして与えます。空のbundleへ最初のsourceを追加できます。
物性の暗黙継承はありません。必要な場合だけ`inherit_from: {table: existing, join: particle_id}`を要求し、
同じcohort・物性authorityであることを別途確認します。ID整数の一致だけで同じ物理sourceと認定しません。
[source producer](src/chamber_particles_tools/source_builder/README.md)と
[CSV adapter](src/chamber_particles_tools/csv_import/README.md)が詳細を所有します。

場の生成・変換も同じCLIにあります。出力bundleをrecipeの`input_bundle`へ指定してrunnable caseを作ります。

```console
uv run --locked python -m chamber_particles inputs cache cache.yaml cached.h5 --report cache.json
uv run --locked python -m chamber_particles inputs electrostatic check builder.yaml
uv run --locked python -m chamber_particles inputs electrostatic build builder.yaml electric.h5 --report electric.json
```

cacheは明示した誤差・support・資源条件を満たす変換だけを公開します。
electrostaticは静的gas-only RZ、conforming P1、明示境界電位と縮約plasma closureです。
全材料・浮遊導体のV/Q連成や一般COMSOL代替は含みません。電場変更後にcharge/ion/DEP係数の再生成が必要なmodelでは、
その依存producerも明示して更新します。電場だけを交換して旧係数を残す操作を同じ問題と扱いません。

## Pythonで使う

計算coreの公開操作は4つです。`check_case`は`simulate`と同じprepareで対応条件とmemory planを調べ、
worker起動、粒子更新、writer開始をしません。検査成功は収束精度や物理妥当性の認定とは別です。

```python
from chamber_particles import check_case, load_case, open_result, simulate
from chamber_particles_tools.inputs import init_example
from chamber_particles_tools.analysis import summarize_result, export_result
from chamber_particles_tools.visualization import render_result


def main():
    created = init_example("xy-ballistic", "input")
    case = load_case(created.case_path)
    print(check_case(case, workers=1))
    simulate(case, "result", workers=1)
    result = open_result("result")
    print(result.read_context().coordinate_system)
    print(summarize_result("result"))
    export_result("result", "export")
    render_result("result", "view", detail=True)


if __name__ == "__main__":
    main()
```

自分の`DataBundle`には`inputs.create_case(bundle, settings, sources, output_directory)`を使います。
`sources`は意図したcomplete source tupleです。`create_from_configuration(recipe_path, output_directory)`はCLI `create`と同じです。
canonical writer/parserは`chamber_particles.case_format`が所有し、clientは物理validatorや数値engineを複製しません。

`workers`の既定は1。複数processのPython scriptにはmain guardを付けます。
worker数は実行資源で、caseの数値設定に含めません。親が時計・release・single writer・checkpointを所有し、
各workerが排他的粒子群を更新します。活動粒子のmacroステップごとに同期します。
速度・memoryの条件は[性能文書](docs/parallel_execution_plan.md)を参照してください。

## 保存結果を再利用する

入力case/dataはschema 3、結果はschema 4、checkpointはlayout 2です。
結果のimmutable `context.h5`は元case/dataのhash、geometry、境界・source意味、basis/unitをbindします。
背景field、完全な初期source、full traceはcontextへ複製しません。
`open_result(...).read_context()`で読み、元caseなしで解析・geometry表示できます。
旧結果の互換readerはなく、新版で再生成するか当時の環境でexportします。

`analyze`は全選択粒子のfate、source別coverage、繰返し境界interaction、first hit、terminal結果、保存active曲線を区別します。
`--source-ids`と`--particle-ids`でexact cohortを指定できます。histogramのbin edgesはSIで明示します。
`export`はfinal、保存frames/probes、boundary interactionsとidentity receiptを出力します。
`view`はgeometryと全選択人口の最終占有、実eventを表示し、`--detail`か`--particle-id`で保存軌道を追加します。
表示点数の上限はreceiptへ記録し、統計の母集団を表示sampleへ置き換えません。

stuck/heldは数値的な境界結果です。adhesion、deposit mass、濃度・fluxには適切なsource weight/rateと物理モデルが必要です。
反射を含む全interaction数、初回到達、終端capture数は別量です。保存点の線分は未保存bounceやBrownian軌道を再構成しません。

## 仕様と検証

| 内容 | 正本 |
|---|---|
| canonical入力と設定 | [case format v3](docs/case_format_v3.md) |
| context・結果・checkpoint | [result format v4](docs/result_format_v4.md) |
| 物理式・revision・適用域 | [physics models](docs/physics_models.md) |
| 積分・event・Brownian・RNG | [numerics](docs/numerics.md) |
| 対応組合せとerror | [support and errors](docs/support_and_errors.md) |

解析解・manufactured case・独立OU reference・収束試験・公開API scenarioとinstalled-wheel flowで検証します。
live COMSOLと開発V&Vは`tools/vv/`へ分離します。保存COMSOLとの実用的な輸送・人口対応は、その観測量とcohortの主張です。
同一Brownian path、native全方程式の数値的認定、装置予測の実験妥当性へ広げません。
[COMSOL評価方法](../vv_methodology.md)が判定の意味を所有します。

開発はuvのlocked環境で行います。

```console
uv lock --check
uv run --locked ruff format --check src tests tools examples scripts
uv run --locked ruff check src tests tools examples scripts
uv run --locked pyrefly check --summarize-errors
uv run --locked lint-imports
uv run --locked python scripts/check_complexity.py
uv run --locked pytest tests/verification tests/scenarios -q
```

責務と変更規則は[AGENTS.md](../AGENTS.md)、実装工程は[implementation plan](../implementation_plan.md)、
変更と配布状況は[release notes](RELEASE_NOTES.md)を参照してください。過去の科学・性能・Release資格は当時の記録へ保持します。
