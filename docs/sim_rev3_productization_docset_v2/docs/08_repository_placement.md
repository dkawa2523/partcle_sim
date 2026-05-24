# 08. Repository Placement

## この資料群の配置

```text
docs/productization/sim_rev3/
```

## 参照レポートの配置

VIGUSレポートを置くなら:

```text
docs/productization/sim_rev3/reference/vigus_repro_report_combined.md
```

ただし、出力ディレクトリや巨大CSVをdocsに入れない。

## 生成物の配置

実行出力:

```text
outputs/
_out_*/
```

比較出力:

```text
outputs/<case>/comparison/
```

研究履歴:

```text
research/vigus_legacy/
```

## 製品コードに入れてよいもの

```text
particle_tracer_unified/
tools/
external/
examples/minimal_*/
tests/
docs/
```

## 製品コードに入れないもの

```text
outputs/_tmp_*
one-off terminal analysis
large generated videos
case-specific wall-id hacks
temporary branch configs as default behavior
```

## docsと実装の関係

この資料群は仕様書ではなく、実装を安全に進めるためのガイドである。実際の仕様は、最終的にはコード、最小テスト、比較summaryの3つで確認する。
