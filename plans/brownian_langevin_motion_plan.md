# Brownian / Langevin Motion Implementation Plan

Status: implementation plan, not an implemented solver feature.

This plan defines how to add Brownian / Langevin stochastic motion without
turning the solver into a case-specific rescue path. The default trajectory
solver must remain deterministic and bitwise or numerically unchanged unless
the user explicitly enables the stochastic model in `solver.stochastic_motion`.

## Purpose

The current solver advances particles with deterministic drag relaxation,
body acceleration, and electric force from `(q(t)/m)E`. For low-pressure ICP cases with nano-particles,
thermal gas collisions can add Brownian velocity fluctuations. This feature is
useful for deposition-footprint sensitivity and long residence-time cases, but
it must not hide bad field support, wrong boundary definitions, or invalid
initial positions.

The implementation goal is:

- optional stochastic motion controlled by config
- no behavior change when disabled
- physically interpretable fluctuation strength
- reproducible runs with a fixed seed
- simple 2D/3D support
- minimal runtime and memory overhead

## Continuous Model

The deterministic model remains:

```text
dx/dt = v
dv/dt = (u(x,t) - v) / tau_eff + a_body + (q(t)/m)E(x,t)
```

where:

- `x` is particle position
- `v` is particle velocity
- `u(x,t)` is gas velocity
- `tau_eff` is the particle relaxation time from the selected drag model
- `a_body` is configured body acceleration
- `E(x,t)` is the sampled electric field, and `q(t)/m` is taken from the current particle state

The opt-in Langevin extension adds fluctuation-dissipation-consistent thermal
forcing:

```text
dv = [(u - v) / tau_eff + a_body + (q/m)E] dt
     + sqrt(2 k_B T_g / (m_p tau_eff)) dW
```

where:

- `k_B` is Boltzmann's constant
- `T_g(x,t)` is gas temperature
- `m_p` is particle mass
- `dW` is an independent Wiener increment per velocity component

For a frozen local `T_g` and `tau_eff`, the exact Ornstein-Uhlenbeck velocity
variance over a stochastic interval `Delta t_s` is:

```text
sigma_v^2 = (k_B T_g / m_p) * (1 - exp(-2 Delta t_s / tau_eff))
```

The velocity kick is:

```text
v <- v + sigma_v * N(0, I)
```

The long-time diffusion scale is:

```text
D = k_B T_g tau_eff / m_p
MSD_d(t) = 2 d D t
```

where `d` is the spatial dimension.

## First Implementation Scope

Implement only an underdamped velocity Langevin kick. Do not add direct
position random-walk noise in the first tranche.

Reason:

- a direct position kick can tunnel through walls unless it is tightly coupled
  to boundary root finding
- velocity kicks naturally pass through the existing collision system on the
  next step
- the implementation stays small and easy to verify

The stochastic kick should be applied only to active free-flight particles
after boundary/collision resolution for the current step. That makes the kick
affect the next step and avoids turning stochastic noise into an untracked wall
crossing inside the same step.

## Configuration Contract

Default behavior:

```yaml
solver:
  stochastic_motion:
    enabled: false
```

Opt-in Brownian/Langevin behavior:

```yaml
solver:
  stochastic_motion:
    enabled: true
    model: underdamped_langevin
    stride: 10
    seed: 12345
    temperature_source: field_T_then_gas
```

Fields:

- `enabled`: false by default. If false, no stochastic arrays are allocated and
  no random kicks are generated.
- `model`: initially only `underdamped_langevin`.
- `stride`: apply one stochastic kick every `stride` solver steps using
  `Delta t_s = stride * dt`. This reduces random-number cost while preserving
  the correct integrated variance for slowly varying fields.
- `seed`: optional. If omitted, use `solver.seed`.
- `temperature_source`: initially `field_T_then_gas`.

Rejected in v1:

- direct position Brownian displacement
- full random-walk wall crossing within one step
- per-particle stochastic model selection
- fitted or learned trajectory correction

## Temperature And Relaxation Inputs

For each active particle receiving a kick:

1. Sample gas temperature `T_g(x,t)` from field quantity `T` when available and
   valid.
2. Fall back to `gas.temperature_K`.
3. Compute `tau_eff` from the same drag model used by deterministic free-flight.
4. Clamp only to existing solver physical safeguards such as `min_tau_p_s`.

Do not use the COMSOL pressure variable as absolute chamber pressure unless the
case manifest explicitly proves its meaning. For ICP cases, prefer exported
`rho_g` and gas temperature for Epstein/Stokes relaxation, with chamber
pressure recorded as a diagnostic.

## 2D, 3D, And Time-Dependent Fields

The stochastic model is dimension-independent:

```text
xi shape = (n_active, spatial_dim)
```

For time-dependent fields:

- sample `T_g` at the same physical time used for the current solver step
- require the field time axis to cover the run time, as with velocity and
  acceleration
- do not extrapolate stochastic temperature beyond the provider time support

If `T` is absent in a transient case, `gas.temperature_K` is still acceptable
because it is explicit and deterministic.

## Implementation Steps

1. Add config parsing only.
   - Extend runtime options with a small `StochasticMotionConfig`.
   - Validate `enabled`, `model`, `stride`, `seed`, and `temperature_source`.
   - Report the parsed settings in `solver_report.json`.
   - Disabled mode must produce identical outputs to the current solver.

2. Add a focused solver helper module.
   - New module: `particle_tracer_unified/solvers/stochastic_motion.py`.
   - Keep stochastic config, sigma calculation, and vectorized kick generation
     there.
   - Do not spread Brownian logic across collision, field provider, and output
     modules.

3. Reuse existing physical inputs.
   - Reuse particle mass from `particles.csv`.
   - Reuse gas temperature fallback from `gas`.
   - Reuse the selected drag model's relaxation-time formula.
   - Use field `T` only when it is already present in the provider bundle.

4. Apply kicks after current-step collision resolution.
   - Only active particles receive a kick.
   - Stuck, absorbed, escaped, invalid-mask-stopped, and numerical-stopped
     particles receive no kick.
   - The updated velocity is used on the next deterministic free-flight step.

5. Keep random generation outside Numba in v1.
   - Generate a `(n_active, spatial_dim)` normal array only on stochastic
     stride steps.
   - Avoid storing random histories.
   - Use one `numpy.random.Generator` seeded from config for reproducibility.
   - Do not use parallel Numba RNG until profiling proves it is necessary.

6. Add diagnostics.
   - `stochastic_motion_enabled`
   - `stochastic_motion_model`
   - `stochastic_motion_stride`
   - `stochastic_motion_seed`
   - `stochastic_kick_count`
   - `stochastic_velocity_rms_mps`
   - `stochastic_temperature_source`

7. Add tests.
   - Disabled config gives identical results to no stochastic config.
   - Same seed gives identical stochastic output.
   - Different seed changes trajectories.
   - In a no-flow box, velocity variance approaches `k_B T / m`.
   - In a no-flow box, MSD is within tolerance of `2 d D t`.
   - Wall counters are not used as success criteria; contract counters must not
     be hidden by stochastic behavior.

8. Evaluate runtime.
   - 10k short check with stochastic disabled: overhead target < 1%.
   - 10k short check with `stride: 10`: overhead target < 10%.
   - Full ICP run only after short checks pass.

## Acceptance Criteria

The feature is accepted only when:

- default solver behavior is unchanged
- stochastic behavior is explicitly opt-in
- runs are reproducible with a fixed seed
- thermal variance and diffusion sanity checks pass
- 2D and 3D minimal cases both run
- time-dependent fields either sample valid `T` or clearly fall back to
  `gas.temperature_K`
- no field-support, input-contract, or boundary-event failure is converted into
  a successful deposition result

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Random position noise crosses walls without detection | v1 uses velocity kicks only, applied after collision resolution |
| Runtime increases from Gaussian generation | default disabled; use `stride`; generate only for active particles |
| Nondeterminism from parallel RNG | v1 uses a single NumPy generator outside Numba |
| Wrong pressure interpretation changes diffusion strength | use `T` and drag-model inputs already accepted by the solver; treat pressure as diagnostic unless manifest proves otherwise |
| Brownian model hides bad geometry or field support | keep provider/input contracts unchanged and fail before time integration |
| Case-specific tuning creeps into solver | expose only general physical parameters; keep source/wall/case choices in configs |

## What Not To Implement In This Tranche

- full OML particle charge time evolution
- charge-state stochasticity
- collision chemistry
- DSMC or PIC-like plasma coupling
- learned trajectory correction
- solver-side field filling, clipping, or particle push-off

Charge updates are a separate opt-in model. Electric force should always be
computed from the current particle state as `(q(t)/m)E`, not from precomputed
electric acceleration stored in the field bundle.

## Clean-Start Notes

- Existing docs already prohibit solver-side rescue logic. That is compatible
  with this plan because Brownian/Langevin is an explicit physical model, not a
  repair for invalid field or boundary data.
- Do not place this feature in COMSOL exporter code. Exporters may provide `T`
  and gas diagnostics, but the stochastic model belongs in solver config and
  runtime only when enabled.
