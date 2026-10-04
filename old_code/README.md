# Particle Tracer Unified

外部で用意した場・粒子初期条件・形状・境界条件から、粒子軌道を計算する
Pythonパッケージです。2D Cartesian、axisymmetric RZ、3D Cartesianを扱います。

計算の対象は、与えられた場に対する一方向連成のLagrangian point-particle
モデルです。COMSOLを内蔵したり再実行したりはせず、native入力または明示的な
COMSOL exportを同じsolver契約へ変換します。

## 必要環境とインストール

- Python 3.10以上
- runtime: NumPy、pandas、PyYAML、Numba

```console
python -m pip install -e .
```

可視化または検証用の追加依存が必要な場合だけextraを指定します。

```console
python -m pip install -e ".[viz,validation]"
```

## Quickstart

同梱の2D例を検証し、実行して、成果物を再検証します。

```console
particle-tracer check examples/v02_minimal/run_config.yaml --full
particle-tracer run examples/v02_minimal/run_config.yaml -o run_output
particle-tracer artifacts run_output
```

3D例は
[`examples/v02_minimal_3d/run_config.yaml`](examples/v02_minimal_3d/run_config.yaml)
です。

Python APIの実行動線は4操作です。

```python
from particle_tracer_unified import load_case, simulate, validate_case, write_result

case = load_case("examples/v02_minimal/run_config.yaml")
report = validate_case(case, detail="summary")
if not report.passed:
    raise ValueError(report)

result = simulate(case)
manifest = write_result(result, "run_output")
```

- `load_case()`はcanonical入力を読み、解決済みの`SimulationCase`を返します。
- `validate_case()`はファイルを書かずにpreflightを行います。
- `simulate()`は数値計算だけを行い、`SimulationResult`を返します。
- `write_result()`だけが、新規または空の出力directoryへ成果物を書きます。

## 入力と成果物

runtimeが受け付ける入力は`schema_version: 2`のcanonical YAML/CSVです。未知のkey、
曖昧な値、必要なSI単位や物性の欠落は入口で拒否します。旧形式はruntimeで解釈せず、
`particle-tracer migrate`でcanonical形式へ変換します。

standard modeの成果物は次の3ファイルです。

- `final_particles.csv`
- `run_summary.json`
- `wall_summary.csv`

debug modeではtrajectory、wall event、step summary、force contribution、詳細診断を
追加します。入力列、設定、artifact schemaの詳細は
[入力と成果物](docs/input_artifacts.md)を参照してください。

## CLI

公開console scriptは`particle-tracer`の1つです。

| command | 役割 |
|---|---|
| `run CONFIG [-o DIR]` | preflight後にcaseを実行して成果物を保存 |
| `check CONFIG [--full]` | 副作用なしのpreflight |
| `migrate CONFIG -o DIR` | legacy入力をcanonical v0.2へ変換 |
| `compare WORKFLOW ...` | field・acceleration・trajectory・boundary等の比較 |
| `artifacts DIR [--require-debug]` | artifact schemaと構成を検証 |
| `visualize ...` | optionalなgraph・animation・mechanics・boundary可視化 |
| `comsol build-case ...` | 明示的なCOMSOL exportからcaseを構築 |

`comsol build-case` はfieldの保存形式を明示入力から選びます。
`--field-node-samples` はCOMSOLのmesh節点値をそのまま使い、境界層メッシュの
細分と真空領域の境界をそのままsupportにします。`--field-bundle` は解を正則格子へ
再サンプルした従来形式です。両者は排他で、選んだ形式はmanifestのfield artifact
`format` に記録されます。詳細は[COMSOLとV&V](docs/comsol_vv.md)を参照してください。

各commandの引数は`particle-tracer COMMAND --help`で確認できます。

## 品質コマンド

品質toolはruntime依存から分離されています。

```console
uv sync --frozen --group quality
uv run --frozen nox -s quality-fast -- particle_tracer_unified/example.py tests/test_example.py
uv run --frozen nox -s quality-pr
```

- `quality-fast`: 指定した変更Python fileをRuffでformat・安全なfix後、baseline-awareな
  lint、Pyrefly、pytestを実行します。Git metadataがないsnapshotではpath指定が必須です。
- `quality-pr`: format/lint、型、architecture、複雑度、branch/変更行coverage、security、
  dependency、secret、dead-code候補をcheck-onlyで検査します。
- `quality-nightly`: PR gateに複数実行条件、性能・memory、mutation testを加えます。
- `quality-baseline`: 品質baselineを明示更新します。通常検査やCIからは実行しません。

nightlyはLinux/WSLの`fork`を必要とするmutmutを含みます。

```console
uv sync --frozen --group quality --group nightly
uv run --frozen nox -s quality-nightly
```

## 使い分ける選択肢

既定値はどれもCOMSOL比較を前提に選んであります。別の目的で使う場合だけ変更してください。

| 選択肢 | 既定 | 変える理由 |
|---|---|---|
| fieldの保存形式 | `--field-node-samples`（mesh native） | 参照用に正則格子が欲しい場合だけ `--field-bundle`。格子は最薄の物理層を自力で解像する必要があり、壁に隣接するcellはstencilが領域外nodeに触れて粒子を停止させます |
| `physics.wall_interaction.contact_sliding` | COMSOLケースは `false`、nativeは `true` | 壁へ落ち着いた粒子をその場に留めたい場合は `true`。COMSOLに点粒子の接触modelはありません |
| `physics.wall_interaction.max_hits_per_step` | 5 | 1 stepあたりのbounceが多いケースで引き上げます。予算切れはnumerical stopです |
| `time.max_substep_splits` | 4（16 substep） | シース通過や壁接近など、滑らかな自由飛行より細かい分割が要るケースで引き上げます |

`compare near-wall` は壁近傍で停止した粒子を数える診断です。mesh native fieldでは
supportがmeshと一致して停止帯が生じないため、主に正則格子ケースの評価に使います。

## 互換性の境界

- 公開Python操作は`load_case`、`validate_case`、`simulate`、`write_result`です。
- runtime入力と成果物はschema version 2を正本とします。
- legacy互換はmigration層だけが担当します。
- `write_result()`は既存のimmutable成果物を上書きしません。
- 大型assetの識別情報は[`data/assets.yaml`](data/assets.yaml)で管理します。

## 文書

- [COMSOLからケースを組み立てる手順](docs/comsol_workflow.md) — COMSOLのモデルと
  設定を出発点にした手順書。最初に読むもの
- [Architecture](docs/architecture.md)
- [入力と成果物](docs/input_artifacts.md)
- [物理モデルと数値計算](docs/physics_numerics.md)
- [COMSOLとV&V](docs/comsol_vv.md) — 手順が強制する契約

コーディングエージェント向けの入口は [`CLAUDE.md`](CLAUDE.md) です。
