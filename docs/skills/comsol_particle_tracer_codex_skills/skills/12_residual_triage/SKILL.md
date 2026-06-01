# Skill 12: Residual Triage for Hard Cases

## Purpose

VIGUSのように、一部改善後もCOMSOLとの差が残るケースで、残差を過剰診断ではなく最小断面で切り分ける。

## Inputs

```text
comparison_summary.json
first_step_compare_summary.json
force_contributions.csv
wall_events.csv if available
collision_diagnostics.json if available
final_particles.csv
COMSOL particle long CSV
release table
```

## Residual classes

Classify residuals into:

```text
import residual
release residual
first-step residual
force residual
event-location residual
wall-law residual
near-wall no-hit residual
stochastic ensemble residual
runtime/performance residual
```

## Minimal analyses

### 1. Source-part grouping

Group metrics by:

```text
source_part_id
unknown source
particle size bin
charge bin if applicable
release time bin
```

### 2. Near-wall active no-hit

Flag particles:

```text
active at final time
nearest wall distance below threshold
no wall event recorded
```

### 3. First-step outliers

Flag:

```text
top speed_ratio errors
top position errors
top force contribution mismatches
```

### 4. Sampled vs full consistency

Do not assume sampled reference equals full reference. Record both if available.

## Outputs

```text
residual_gap_report.md
residual_gap_summary.json
optional residual_particles.csv
```

## Pass criteria

- Residuals are assigned to layers.
- Recommended next fix is narrow.
- Unknown source and stochastic cases are identified.

## Fail criteria

- A single wall coefficient is blamed without first-step and source grouping evidence.
- A runtime optimization is proposed for a physics residual.
- Full and sampled references are mixed.
