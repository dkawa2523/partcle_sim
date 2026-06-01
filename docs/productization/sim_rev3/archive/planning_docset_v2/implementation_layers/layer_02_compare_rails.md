# Layer 02: Minimal Compare Rails

## 目的

局所修正をglobal改善と誤認しないようにする。

## 最初に作るもの

```text
compare_summary.json
```

最低内容:

```json
{
  "import": {},
  "preprocess": {},
  "first_step": {},
  "events": {},
  "ensemble": {},
  "runtime": {},
  "acceptance": {}
}
```

## Optional CSV

```text
first_step_error.csv
force_contributions.csv
boundary_hit_comparison.csv
state_fraction_time_series.csv
```

## 実装方針

- 小さいCLIまたは既存compare toolの整理でよい。
- dashboardを作らない。
- normal runの常時出力にしない。
- VIGUS大規模fixtureをunit testにしない。
- synthetic small caseでsmokeする。

## Acceptance

- endpoint-only判断を禁止する。
- runtime, wall count, state count, first-crossingを同時に見る。
- before/after比較が同じフォーマットで読める。
