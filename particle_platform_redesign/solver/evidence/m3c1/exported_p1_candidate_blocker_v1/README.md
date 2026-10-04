# M3-C1 exported-P1 candidate blocker v1

Status: **BLOCKED; no integrated trajectory agreement result**.

## What this candidate is

The candidate field is a canonical, exact-connectivity, piecewise-linear P1
projection built from exported nodal values. It is **not** the COMSOL native
finite-element interpolant. Historical scratch files and tool revision strings
contain `native`; that word is not a scientific classification for this
evidence.

The candidate also predates the recovered PPR heat-flux export. Its preparation
report marks the thermophoretic primitive authority `NOT_TESTED`. The later PPR
evidence closes the saved-row producer formula, but does not retroactively
replace this candidate input or certify runtime field sampling.

## What is closed

The frozen-state record replayed seven producer forms. The isolated PPR
supplement replayed Waldmann thermophoresis at the same 13,202 saved active
particle/time states. Together this is 8/8 saved-row producer-form closure:

- dynamic charge;
- electric force;
- relative-flow ion drag;
- linear Epstein drag;
- Waldmann thermophoresis;
- free-molecular lift sensitivity;
- dielectrophoresis; and
- gravity with buoyancy.

This closure concerns producer formulas at saved states only. The frozen-state
production comparison has four `PASS` results, three documented physical-
constant convention differences, and one thermophoretic comparison that was
not rerun with the authoritative PPR primitive. Field interpolation, stage
wiring, continuous applicability, events, boundaries, and integrated accuracy
remain unproved.

## Why integration is blocked

At time zero, all 287 particles had field support and passed local stage
applicability. The maximum pointwise ion-relative speed was
`15354.76726984887 m/s`, the maximum neutral speed ratio was
`0.0014665428365538409`, and the minimum mean-free-path/radius ratio was
`40556.38566578311`.

Nevertheless, the production continuous-path check uses conservative global
component and acceleration bounds. Even the deepest observed
`9.765625e-9 s` enclosure piece bounded ion-relative speed by
`249539.93753588165 m/s` and neutral speed ratio by
`788.448680021989`. Every one of the 287 particles therefore received a
`model_applicability` failure at time zero for all three diagnostic step sizes.
No trajectory rows were produced.

This is evidence that the global certificate is inconclusive for this strongly
nonuniform case; it is not evidence that the actual local initial states violate
the models. Relaxing a limit or bypassing the certificate would not establish
validity.

## Claim boundary

- Saved-row producer-form closure: `PASS` (8/8, after the separate PPR supplement).
- Production pure-model and runtime wiring coverage: `PARTIAL`.
- Exported-P1 integrated candidate: `BLOCKED`.
- COMSOL trajectory agreement: `NOT_TESTED`.
- Native COMSOL mesh/interpolant parity: `NOT_CLAIMED`.
- Physical applicability and boundary behavior: `NOT_TESTED`.

The exact machine-readable values and source hashes are in
[report.json](report.json). Large diagnostic arrays are deliberately not
copied into durable evidence.
