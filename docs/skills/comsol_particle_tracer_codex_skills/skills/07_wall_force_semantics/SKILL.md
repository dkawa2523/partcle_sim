# Skill 07: Wall Law and Force Inventory Semantics

## Purpose

COMSOLのwall feature、force feature、particle material設定を、本solverのmanifest・wall table・force inventoryへ落とす。壁則や力を曖昧にすると、結果が合っても理由が間違う。

## Wall law mapping

For each boundary/part, classify:

```text
stick / freeze
absorb / disappear
escape / open / outflow
pass through
specular bounce
diffuse scattering
mixed diffuse/specular
general/custom reflection
critical sticking velocity
unknown / unsupported
```

Faithful mode must fail on unknown or unsupported wall laws.

## Force inventory

Record enabled and disabled forces:

```text
drag model
gas properties and sources
electric force: E field vs potential vs q/m acceleration
thermophoresis
Brownian / stochastic motion
charge model
dielectrophoresis
ion drag if present
lift
pressure-gradient
virtual mass
external/body acceleration
```

## Steps

1. Create `force_inventory.yaml`.
2. Create or update `wall_law_map.csv`.
3. Identify fields required by enabled forces.
4. Mark unsupported COMSOL features explicitly.
5. Decide comparison layer allowed:
   - field only
   - acceleration only
   - deterministic trajectory
   - stochastic ensemble

## Outputs

```text
force_inventory.yaml
wall_law_map.csv
wall_force_semantics_report.md
```

## Pass criteria

- No unknown enabled force exists in faithful mode.
- Wall law table covers all selected boundaries.
- Brownian/stochastic policy is explicit.

## Fail criteria

- Treating qE/m acceleration field as E field without manifest declaration.
- Allowing unknown COMSOL wall law to fall back to specular.
- Comparing deterministic particle paths while stochastic terms are active and uncontrolled.
