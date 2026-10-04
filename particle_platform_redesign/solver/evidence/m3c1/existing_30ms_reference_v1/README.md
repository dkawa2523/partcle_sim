# Existing 30 ms Case A / Case P reference characterization

Status: `CHARACTERIZED`

Decision: `SOLVER_CONVERGENCE_PASS_COMSOL_REFERENCE_NOT_A_CORE_GATE`

This package evaluates the already-saved 100 nm, 30 ms Case A and Case P
results.  It does not promote COMSOL to solver-core truth and it does not tune
the production engine to one benchmark model.

## What the source actually used

The saved COMSOL settings are not automatic/adaptive time stepping.  Both
packages declare:

- explicit fixed RK4;
- fixed step `10 us`;
- `tauto = manual`;
- `timeadaption = none`;
- relative tolerance `1e-2`.

The saved COMSOL Brownian feature is also active.  Every active saved row has a
non-zero Brownian force, whereas the candidate convergence matrix intentionally
has Brownian motion disabled.  Consequently, individual-particle path equality
is not a valid acceptance test for this comparison.

## Numerical result

The candidate's independent `h/h2/h4` study passes for both workflows.  On the
population shared by the candidate and saved COMSOL trajectories, refining the
candidate step changes the result by far less than the cross-representation
difference:

| workflow | quantity | candidate medium/fine RMS | fine vs saved COMSOL RMS | ratio |
|---|---|---:|---:|---:|
| Case A | position | `3.26643e-9 m` | `3.86065e-3 m` | `1.18192e6` |
| Case A | velocity | `2.54529e-6 m/s` | `1.92276 m/s` | `7.55421e5` |
| Case A | charge | `1.79440e-5 e` | `17.5299 e` | `9.76922e5` |
| Case P | position | `8.43597e-12 m` | `3.05410e-3 m` | `3.62033e8` |
| Case P | velocity | `6.10570e-9 m/s` | `1.52972 m/s` | `2.50540e8` |
| Case P | charge | `3.00438e-8 e` | `7.24696 e` | `2.41213e8` |

The discrepancy is therefore not explained by the candidate time step.  The
comparison mixes different stochastic physics/RNG semantics, native COMSOL FE
fields versus exported P1 fields, axis regularization, and heat-flux primitive
construction.  It cannot be used to assign the difference to the integrator.

Case P retains all 287 particles in both calculations.  Case A has 252/287
matching final fates; the saved reference has 51 active, 180 escaped, and 56
stuck particles, while the candidate has 24 active, 215 escaped, and 48 stuck.
These are useful diagnostic observations, not a solver acceptance gate.

## Why the candidate run was slow

The run receipts show two separate costs:

| workflow/run | macro steps | particle pieces | result segments | segment data |
|---|---:|---:|---:|---:|
| Case A coarse | 48,000 | 6,552,137 | 750 | 0.052 GiB |
| Case A medium | 96,000 | 13,102,527 | 1,500 | 0.090 GiB |
| Case A fine | 192,000 | 26,203,426 | 3,000 | 0.165 GiB |
| Case P coarse | 640,000 | 183,680,000 | 10,000 | 0.485 GiB |
| Case P medium | 1,280,000 | 367,360,000 | 20,000 | 0.967 GiB |
| Case P fine | 2,560,000 | 734,720,000 | 40,000 | 1.930 GiB |

Case P is constrained by the present explicit continuous-charge guard, based on
a global charge Lipschitz bound of about `1.04035e7 1/s`.  Separately, the
durable writer commits one segment every 64 macro steps, producing tens of
thousands of small files.  Filesystem timestamp spans are retained in
`comparison.json` only as operational observations; cells overlapped, so they
are not isolated timing benchmarks.

The next performance work is therefore:

1. one charge-stable coupled update that removes the global explicit charge
   stability bottleneck while preserving stage coupling and event safety;
2. one work-scaled durable commit cadence, recorded in the result identity,
   that preserves restart semantics without emitting a file every 64 steps;
3. only if still necessary, one local error controller within the existing
   engine, rather than a second general ODE framework.

Production step selection remains physics- and error-based: use the coarsest
step for which state, event time/location, terminal fate statistics, and the
requested observables converge.  Matching a COMSOL step is not a requirement.
For Brownian runs, convergence is evaluated with ensemble moments, fate and
first-passage distributions, not same-seed particle paths across different
implementations.

## Files and hashes

- `comparison.json`: `9e4ccf1ed5560736f58f6f7ca1dc8fd039e229ece1f7958570256e5731ecaa50`
- `caseA_time_history.csv`: `73fb8ef840adfc9d0a1ecda74e15174512c8cd4de36192db779d09a100a24693`
- `caseP_time_history.csv`: `16848633905399544d5b6f08e9f065c4b71f3f0ead90045590055bbcb9d15527`
- evaluator: `29bfa1dd62941ceb1ac6c9641634d114b9a15a9e3b6fe5984d2bc3b61ffc5db1`

The machine-readable authority is `comparison.json`; this README is its
human-readable summary.
