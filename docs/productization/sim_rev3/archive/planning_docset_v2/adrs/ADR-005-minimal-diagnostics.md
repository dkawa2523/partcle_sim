# ADR-005: Minimal Default Diagnostics

## Decision

通常runでは小さなsummaryとcounterのみを出し、詳細CSVはcompare/deep modeでのみ出す。

## Reason

過剰診断はhot pathを遅くし、出力管理と保守コストを増やす。製品コードでは必要な比較断面だけを安定的に出す。

## Consequences

- default: summary JSON + counters
- compare: first_step / force / boundary CSV
- deep: wall event trace, trajectory sample
