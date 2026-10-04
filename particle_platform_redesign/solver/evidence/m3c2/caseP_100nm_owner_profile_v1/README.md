# Case-P 100 nm owner discovery v1

Status: `COMPLETE_DISCOVERY_NO_PRODUCTION_CHANGE`.

This is machine-local performance evidence for the accepted common-P1 Case-P
100 nm candidate workload. It is not COMSOL evidence and does not change the
solver, its physics, numerical settings, or result schema. The accepted
`performance.json` policy still requires confirmation in an accepted-accuracy
workload with at least 10,000 particles before any bounded optimization may be
implemented.

## Locked workload and identity

- candidate seeds: `319032`, `319047`, `319063`
- particles per seed: 287
- interval and macro step: 30 ms and 20 us
- Brownian interval-tree depth: 3
- geometry tolerance: `1e-8`
- output: all particles at 121 registered times
- process isolation: one fresh child and one initially empty private Numba
  cache per seed, with `NUMBA_NUM_THREADS=1`
- order inside each child: warm-up, unprofiled public-API measurement, separate
  cProfile public-API run

For every seed, all three reruns reproduced the accepted `final`, release,
boundary, failure, lifecycle-series, and 121-frame payload bit-for-bit. Case
and data identities, algorithm revisions, and work counts also matched the
accepted result. Each run had 1,500 macro steps, 430,500 particle-macro roots,
3,444,000 OU leaves, 3,444,000 accepted pieces, and 3,444,000 candidate
queries. Refinements, wall events, axis crossings, residual splits, and
failures were all zero.

## Descriptive measurements

The three unprofiled public end-to-end wall times were 45.4924, 45.4668, and
45.2652 s. Their median was 45.4668 s and median absolute deviation was
0.0256 s. The corresponding median `simulate` time was 45.4005 s. These are
machine-local descriptive values, not portable speed gates.

Immediately after the unprofiled measurement, the warmed long-lived worker
process high-water RSS values were 371,040,256, 370,761,728, and 371,474,432
bytes. They include accepted-baseline reading, JIT warm-up, and retained
compiled state; they are not a fresh single-run allocation measurement. The
solver-owned plan was 53,258,034 bytes and each result occupied 2,423,448
bytes.

cProfile increased public end-to-end time by factors 1.5821, 1.5876, and
1.5904, so profiled seconds are excluded from the baseline timing. The same
source owner dominated all seeds: `integrators`, with self-time shares
42.58%, 42.86%, and 42.67%. The leading entry points were
`curved_chord_deviation_bounds`, `_upper_scalar_product`, and
`_lerp_controls`; `curved_chord_deviation_bounds` alone used about 12 s of
profile self time per seed. Event broad/localize accounted for about 9.5%,
while unattributed native calls accounted for about 19.0%.

The automatic decision remains `optimization_authorized=false`. `integrators`
was not one of the preregistered bounded owners, so this run does not
retroactively relabel it to pass the 25% gate. The reviewed interpretation is
that the data discovered a new, narrow hypothesis: curved-event enclosure and
chord preparation around every Brownian leaf. The JSON retains every
`chamber_particles` source/function self-time row so that owner mappings can be
audited without repeating the run.

## Conditional future performance work

M3-C2A is `CLOSED_ACCEPTED_WITH_LIMITATIONS`; a 10,000-particle run is not a
benchmark exit criterion or an automatic next step. If explicit target
hardware, output mode, wall-time, and memory limits open a separate product
performance work package, its first and only discovery workload is a
preregistered deterministic 10,000-particle derivative of the same accepted
workload. It must preserve the original 287-particle prefix, physics revisions,
step, tree, tolerance, and RNG addressing, and use low-overhead stage timing.
Only if one bounded curved-event enclosure chain explains at least 25% of
unprofiled end-to-end time may one implementation change be attempted. Retain
it only with exact scientific/work identity and a gain greater than the larger
of 5% or three same-condition baseline MADs; stop the work package either way.

This evidence does not certify negative-ion current, physical applicability,
eventful boundary parity, pathwise COMSOL RNG identity, or universal COMSOL
equivalence. Case-P physical applicability remains
`NOT_CERTIFIED_NEGATIVE_ION_CURRENT_OMITTED`.

## Files

- [`profile.json`](profile.json), SHA-256
  `746aebd05630b15c73515bbcdabb271eb953f144cd331bece91e46aad555b915`
- profile driver SHA-256
  `52218cce271161c0c2f170f45cf5908e30ae87ba545d4c213932986a9fefe83d`
