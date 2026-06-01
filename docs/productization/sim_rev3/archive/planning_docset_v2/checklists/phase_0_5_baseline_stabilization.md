# Phase 0.5 Checklist: Focused Tests and Baseline Smoke Stabilization

## 目的

新機能を足す前に、最小focused checksとsmokeを安定させる。

## 対象

```text
tests/test_comsol_case_builder.py
tests/test_runtime_builder_contracts.py
tests/test_boundary_runtime.py
existing compare smoke if present
```

## 許可

- helper signature driftの修正
- test messageの過度な完全一致の緩和
- 明確な小さなregression修正
- deferred failureの文書化

## 禁止

- physics tuning
- VIGUS-specific wall id追加
- 大規模fixture追加
- broad diagnostics追加
- new architecture追加

## Acceptance

- focused set green、またはdeferred理由が明記される
- 変更は小さい
- 新しいbehavior changeには最小testがある
