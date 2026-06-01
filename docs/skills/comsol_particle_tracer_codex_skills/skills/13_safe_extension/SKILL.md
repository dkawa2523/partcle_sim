# Skill 13: Safe Extension Rules for Codex

## Purpose

COMSOL比較のために必要な実装を追加するとき、コードを肥大化・複雑化させないための拡張ルール。

## Preferred change locations

```text
io/ for source-specific loading and manifests
providers/ for field or geometry behavior
compare/ for comparison tools
tools/ for case builders and external conversion
solvers/ only for actual solver behavior
```

## Minimal tests only

Add tests only for changed behavior. Prefer:

```text
small synthetic field
small synthetic geometry
2-5 particles
short t_end
no large COMSOL fixture
```

## Avoid

```text
broad schemas
new helper layers with one caller
always-on logging
full trajectory dumps by default
case-specific branches
magic wall IDs
silent fallbacks
```

## When to add a new CLI

Add a CLI only if:

- it represents a reusable workflow
- it consumes existing artifacts
- it writes compact outputs
- it avoids changing solver runtime behavior

Do not add a CLI for one-off VIGUS probes.

## When to change solver physics

Only after:

```text
import passes
field compare passes
release compare passes
first-step evidence points to solver behavior
minimal synthetic test exists
```

## Required final note

Every extension must state:

```text
What failure layer it addresses
Why existing tools were insufficient
Why this is not VIGUS-specific
How it is tested
How to disable or avoid deep diagnostics
```
