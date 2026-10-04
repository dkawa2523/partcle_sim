# M3-C0 evidence

- [`boundary_semantics_v2/`](boundary_semantics_v2/): current `COMPLETE` /
  narrowly scoped `PASS` for the active force-free path and isolated COMSOL
  Freeze-37 and Disappear-35 semantics (56 PASS, 0 FAIL, 6 characterized);
  production-solver parity is `NOT_TESTED`.
- [`boundary_semantics_v1/`](boundary_semantics_v1/): historical evidence,
  superseded by v2 because v1 did not gate the active force-free path. Its
  original compact evidence is retained unchanged.

`reference_lock_v1/` is the immutable offline readiness result for the existing
12 COMSOL packages.  Its decision is authoritative only for that offline
snapshot; the current execution status is not inferred from it.  Its detailed decision is
`comparison_manifest.json`; tables contain the exact input hashes, formula and
field locks, package inventory, paired-variant differences, candidate steps,
and preregistered seed cohort.

The result is `PARTIAL_RERUN_REQUIRED`, not a trajectory certificate.  It was
produced without changing the production solver or launching a COMSOL study.
The COMSOL installation was separately checked read-only on 2026-10-01:
version 6.4.0.429, required COMSOL/Particle Tracing/Plasma/AC/DC features
available, source MPH hashes unchanged.

`deterministic_pilot_v5/` is retained as historical `CHARACTERIZED` evidence.
Its evaluator returned PASS for all seven rows at 1.25/0.625/0.3125
microseconds, but the compact admission decision is `INVALIDATED`: the position
relative L2 used the norm of absolute global RZ coordinates and therefore
depended on the coordinate origin.  Its original values and thresholds remain
recorded and must not be read as current acceptance evidence.

`deterministic_pilot_v6/` is the sequential confirmation on the
0.625/0.3125/0.15625 microsecond series, adding the previously unseen finest
result.  Each run has 13,202 active records.  The origin-invariant fine-pair
position-displacement, velocity, and charge relative L2 values are
`3.102727085428027e-5`, `3.92483251084038e-5`, and
`1.3511393490811483e-6`; observed orders are `0.9041136219836081`,
`0.9443119501831206`, and `1.1238380758116033`.  All seven operational gates
pass.  This confirms only pre-event operational step selection for the one
Case-A 100 nm theory pilot, not production-solver agreement, universal physical
accuracy, or a full 12-case certificate.

Do not replace this directory with a rerun.  Brownian-off deterministic
references, step probes, and boundary microcases must use a new revisioned
directory and keep the original MPH files unsaved.
