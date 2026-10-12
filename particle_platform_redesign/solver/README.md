# Chamber Particles 0.3.0

外部から与えたチャンバーの場と形状を使い、粒子の運動・帯電・境界イベントを計算する
Python 3.12のsolverです。独立したuv projectで、COMSOLや`model_dataset/`を実行時に必要としません。
対応座標は2-D Cartesian XYとRZ meridionalです。開発statusはalphaです。

0.3.0では、同一runのCPU process並列、作業配列の再利用、共有メモリと出力の所有を整理しました。
serialとworkerは一つのengineと同じ数値処理を使います。入力・結果schemaは3、checkpoint schemaは2です。
変更と制約は[リリースノート](RELEASE_NOTES.md)、対応する物理・入力の組合せとエラー対処は
[対応範囲](docs/support_and_errors.md)を参照してください。

## Quick Start

solver directoryで実行します。例はCOMSOLを使わず、10 mm角のXY形状、粒子源、反射壁と付着壁を生成します。
出力を上書きしないため、再実行時は別の出力directoryを指定してください。

```console
uv sync --locked
uv run --locked python examples/quickstart/create_case.py .artifacts/quickstart/input
uv run --locked chamber-particles check .artifacts/quickstart/input/case.yaml
uv run --locked chamber-particles run .artifacts/quickstart/input/case.yaml -o .artifacts/quickstart/result
uv run --locked chamber-particles inspect .artifacts/quickstart/result
uv run --locked python -m tools.analysis .artifacts/quickstart/result --output .artifacts/quickstart/derived/summary.json
uv run --locked python -m tools.visualization .artifacts/quickstart/result --output-directory .artifacts/quickstart/derived/visualization
```

正常結果は1粒子、境界イベント2件、failure 0件、最終状態`stuck`です。
analysisは状態・到達集計を、visualizationは形状と軌道・境界イベントのSVGを保存します。
`tools/`と`examples/`はsource checkoutの付属機能であり、wheelには含みません。

## Python APIと並列実行

公開APIは`load_case`、`simulate`、`open_result`の三つです。
複数processのscriptはimport可能なファイルに保存し、main guardを付けて実行してください。

```python
from chamber_particles import load_case, open_result, simulate


def main():
    case = load_case("case.yaml")
    simulate(case, "result", workers=4)
    result = open_result("result")
    print(result.manifest["counts"])


if __name__ == "__main__":
    main()
```

CLIでは次のように指定します。

```console
uv run --locked chamber-particles run case.yaml -o result --workers 4
```

既定は`workers=1`です。worker数は実行資源であり、YAMLの数値設定には含めません。
要求worker数を自動で変更するtunerはありません。各workerは担当粒子を更新し、親が出力とcheckpointを所有します。
活動粒子があるmacroステップごとに全workerの完了を待ちます。同期間隔の設定化は将来項目で、0.3.0にはありません。

worker数を増やすと、計算条件によっては遅くなります。現在の測定例は次のとおりです。
時間は同じ入力の公開API実行を比較したwarm中央値で、入力ごとの計測条件はリンク先が正本です。

| 条件 | 1 process | 4 process | 速度比 W1/W4 |
|---|---:|---:|---:|
| P14-U・1万粒子・320ステップ、Brownianなし、軌道出力なし | 27.36 s | 13.93 s | 1.96x |
| COMSOL Case A入力・287粒子・1500ステップ、Brownianあり | 13.93 s | 32.25 s | 0.43x |
| COMSOL Case P入力・287粒子・1500ステップ、Brownianあり | 19.69 s | 33.87 s | 0.58x |

10万～100万粒子の測定、cold JIT、総CPU、実メモリ、出力量と早期終端の制約は
[並列実行・性能](docs/parallel_execution_plan.md)にまとめています。全CPU数への比例効果や、未測定入力の速度を保証しません。
`memory_limit_mb`はsolver-owned配列の予算で、Python/JITを含むOS総メモリの上限ではありません。

## 入力・数値計算・結果

入力はcanonical HDF5と厳格なYAMLです。SI単位、field support、粒子ID、source/boundary参照とhashを検査します。
producer向けのcanonical writerは`chamber_particles.case_format`にあります。
欠損や適用域外の設定を別モデルや値のclampで補いません。

| 内容 | 正本 |
|---|---|
| case・field・粒子源・境界設定 | [case format](docs/case_format_v3.md) |
| 物理式・model revision・適用域 | [physics models](docs/physics_models.md) |
| 積分・first hit・Brownian・乱数・許容差 | [numerics](docs/numerics.md) |
| final・event・frame・probe・checkpoint/resume | [result format](docs/result_format_v3.md) |
| 対応組合せ・非対応・エラーの確認先 | [support and errors](docs/support_and_errors.md) |

固定/連続電荷、対応するdrag、Coulomb、重力/浮力、熱泳動、ion drag、DEP、lift、Brownian/OUを明示選択します。
積分器は`rk4_fixed`、`exponential_midpoint`、`ou_langevin`です。境界は付着・消失・保持・反射・熱壁・確率付着と、
対応するCartesian XYの平行移動periodic topologyを扱います。正確な組合せは上の正本に従ってください。
付着・保持後は運動更新を停止し、終端状態を保持します。

fieldは静的または固定topologyの線形時間snapshotです。粒子間相互作用、場へのfeedback、moving geometry、
3-D、GPU、adaptive time stepping、任意のCOMSOLモデルとの一般同等性は対応範囲に含みません。
有限depthのBrownian pathを連続OUの厳密first-passageとして扱いません。

## 検証と外部比較

解析解・manufactured case・独立OU/Kramers reference・時間/mesh収束と公開API scenarioで検証します。
同一runのprocess数を変えて、保存科学出力、境界イベント、乱数、checkpoint/resumeとcleanupを確認します。
Windows/Linuxの品質検査と、実際にinstallしたwheelのAPI・console/module CLIによる複数process計算も検証対象です。

COMSOLは外部V&Vの一つです。現行Case A/Pの登録済み位置・終端人口分布は保存COMSOL参照との比較基準を満たしました。
全native右辺、詳細な境界応答、時間発展する速度・電荷分布、物理モデルの実験的妥当性まで認証したものではありません。
比較条件と制約は[COMSOL評価方法](../vv_methodology.md)を参照してください。

生のCSV/HDF5/MPH、結果ZIP、環境とJIT cacheはGit配布しません。検証記録には測定条件とhashを保存し、
元データはローカルのexternal evidenceとして保持します。

## 開発

```console
uv lock --check
uv run --locked ruff format --check src tests tools examples scripts
uv run --locked ruff check src tests tools examples scripts
uv run --locked pyrefly check --summarize-errors
uv run --locked lint-imports
uv run --locked python scripts/check_complexity.py
uv run --locked pytest tests/verification tests/scenarios -q
```

責務と変更規則は[AGENTS.md](../AGENTS.md)、現行の実装・将来項目は[実装計画](../implementation_plan.md)が所有します。
過去版の詳細は[v0.2.0](https://github.com/dkawa2523/partcle_sim/tree/v0.2.0/particle_platform_redesign/solver)と
[v0.1.0](https://github.com/dkawa2523/partcle_sim/tree/v0.1.0/particle_platform_redesign/solver)のtagに保持します。
本repositoryには利用ライセンスを設定していません。再利用条件は権利者の明示許諾が必要です。
