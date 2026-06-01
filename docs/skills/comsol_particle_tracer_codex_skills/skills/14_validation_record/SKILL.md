# Skill 14: Validation Record for COMSOL Comparison

## Purpose

COMSOL比較を「実行した」だけで終わらせず、第三者が後で検証できるvalidation recordとして残す。

## Required sections

Create `validation_record.md` with:

```text
1. Repository branch and commit
2. Environment and command launcher
3. COMSOL version / model / component / study / dataset
4. Input asset list and hashes if feasible
5. Coordinate system and unit scale
6. Field inventory
7. Force inventory
8. Wall law map
9. Release table summary
10. Solver config summary
11. Seeds and stochastic policy
12. Preflight result
13. Field/acceleration comparison
14. First-step comparison
15. Boundary event comparison
16. Trajectory / ensemble comparison
17. Runtime summary
18. Known residuals
19. Verdict
```

## Verdict options

Use only:

```text
pass for stated scope
partial pass with residuals
fail at layer <Lx>
not comparable due to missing data
```

Do not write "COMSOL parity achieved" unless every required comparison layer for the stated scope passes.

## Artifact checklist

Attach or reference:

```text
comsol_case_manifest.yaml
preflight_report.json
field_compare_summary.json
first_step_compare_summary.json
boundary_hit_comparison.json
comparison_summary.json
residual_gap_report.md
solver_report.json
```

## Pass criteria

- Scope is explicit.
- Missing data is explicit.
- Residuals are attributed to layers.
- Commands are reproducible.

## Fail criteria

- Report contains only endpoint counts.
- Report omits sampled vs full reference distinction.
- Report claims parity while first-step or field layers were not checked.
