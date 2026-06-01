# Skill 10: Layered Comparison Against COMSOL

## Purpose

COMSOLと本solverを、単一の終点数ではなく、層別に比較する。

## Comparison layers

### L0 Manifest and import

Compare:

```text
coordinate system
axis names
unit scale
field support
particle counts
boundary/wall mapping
```

### L1 Field samples

Compare field quantities at matched probe points.

Outputs:

```text
field_validation_error.csv
field_compare_summary.json
```

### L2 Acceleration / force

Compare same particle state:

```text
position
velocity
particle size/mass/charge
time
force contribution
```

### L3 First-step

Compare post-preprocess and post-first-step states.

Outputs:

```text
first_step_error.csv
force_contributions.csv
first_step_compare_summary.json
```

### L4 Wall / boundary events

Compare:

```text
first hit time
first hit boundary/part
wall law outcome
wall interaction count
skip/block counters
```

### L5 Trajectory / first passage

Compare time evolution:

```text
x(t), v(t)
first passage CDF
vacuum time summary
state fractions over time
```

### L6 Ensemble

For stochastic or large populations:

```text
state fractions
source-part grouped outcomes
Wasserstein/KS-like distribution metrics if implemented
sampled vs full labels
```

## Rules

- Do not accept L5/L6 if L0-L3 fail.
- Do not compare stochastic particle-by-particle unless random streams are controlled.
- Do not compare final snapshot counts with ever-reached counts as if they were the same.

## Required outputs

```text
comparison_summary.json
residual_gap_report.md
```

## Pass criteria

- Every comparison report states scope and missing layers.
- First failing layer is identified.
- Suggested next action targets the failing layer only.

## Fail criteria

- Endpoint count improvement is accepted while first-step or wall metrics worsen.
- Sampled and full COMSOL references are mixed.
- Wall ID local fix is promoted without global metric improvement.
