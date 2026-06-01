# ADR-001: Compare Rails Before Physics Changes

## Decision

物理実装、release処理、wall event処理より前に、最小比較基盤を実装する。

## Reason

元の問題は、比較評価が弱いために局所修正がglobal悪化を起こしたことだった。したがって、新しい実装を入れる前にpost-import、post-preprocess、post-first-step、event、ensemble、runtimeの比較断面を固定する。

## Consequences

- Phase 1でcompare summaryを作る。
- endpoint-only acceptanceは禁止。
- later phasesはcompare summaryでbefore/afterを確認する。
