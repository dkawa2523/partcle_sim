# Phase 1 Checklist: Minimal Compare Rails

## 目的

物理変更前に比較summaryを安定化する。

## 作る/整理するもの

```text
compare_summary.json
optional: first_step_error.csv
optional: boundary_hit_comparison.csv
```

## 必須指標

- import/preprocess status
- final state counts
- wall interaction counts
- source-wise counts if available
- first-crossing/vacuum summary if available
- runtime counters
- acceptance flags

## 禁止

- dashboard化
- heavy fixture必須化
- endpoint-only合否
- VIGUS-specific id必須化

## Acceptance

- small sampleでbefore/after比較できる
- same JSON schemaで安定出力
- future phaseの変更前後を比較できる
