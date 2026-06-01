# Skill 11: Evaluation Decision Tree

## Purpose

比較結果から、どの層を直すべきかを決める。Codexが局所数値だけで誤修正しないための判定手順。

## Decision tree

### A. Import/manifest fails

Symptoms:

```text
axis mismatch
unknown coordinate scale
field support invalid at release
missing wall law
```

Action:

```text
fix export or manifest
run preflight again
no solver physics changes
```

### B. Field compare fails

Symptoms:

```text
field values differ at same point
wrong component direction
unit scale mismatch
support mask differs
```

Action:

```text
fix field mapping, units, coordinate axes, or interpolation bundle
```

### C. Release parity fails

Symptoms:

```text
release count mismatch
release time mismatch
initial velocity mismatch
source ID unknown or repaired
```

Action:

```text
fix release export or release canonicalization
faithful mode must not snap
```

### D. First-step fails, preflight passes

Symptoms:

```text
post-preprocess close but post-first-step diverges
speed ratio p50/p90 wrong
force total mismatch
```

Action:

```text
triage drag, field sampling, initial velocity, integrator dt, stochastic terms
```

### E. First wall hit fails

Symptoms:

```text
first boundary ID differs
hit time differs
near-wall active no-hit particles
```

Action:

```text
check geometry primitives, segment hit, release grace, boundary normal, wall law
```

### F. Post-wall behavior fails

Symptoms:

```text
same first hit but different stuck/reflection/escape
```

Action:

```text
wall law mapping or stochastic reflection semantics
```

### G. Ensemble differs while deterministic layers pass

Symptoms:

```text
state fractions differ
source group outcomes differ
first passage CDF differs
```

Action:

```text
release population, stochastic seeds/distributions, wall probabilities, sampled/full scope
```

## Explicit forbidden actions

- Do not tune `stick_probability` to fix first-step force errors.
- Do not broaden boundary capture to fix wall law errors.
- Do not add same-source skip to fix field support failures.
- Do not change integrator because endpoint count is wrong before first-step compare.

## Output

Write `evaluation_decision.md` with:

```text
first failing layer
evidence
recommended fix scope
forbidden fixes
next command to run
```
