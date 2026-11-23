# Native benchmark reference inputs

These measurements from isolated unsaved COMSOL 6.4 models are independent
native input for current external verification.

* `xy`: seven Cartesian scenarios at three steps, collected on 2026-10-08.
  The original native run receipt binds all 21 raw CSV files, the compiled
  class, observed configuration log and native Java producer. The original
  native producer source identifies the equations that generated these observations;
  current execution uses the separately maintained entry below.
* `rz`: three force-free RZ particles at three steps, generated afresh on
  2026-11-23 with COMSOL 6.4.0.429. The native process log records actual
  configurations and completion. `provenance.json` records source hashes and
  no-save execution. Raw native CSV files are reference observations.

Current benchmark executors are `evaluate_xy_minimal_suite.py` and
`evaluate_m3c_critical_boundaries.py` in the parent directory. They create
current canonical input and public-API results in new output destinations,
then compare native and candidate observations independently with analytic
motion. Tests do not rewrite these inputs.

XY CSV frames contain native ID/time/x/y/vx/vy/current/final status in SI.
RZ frames contain ID/time/r/z/vr/vz/current/final status. These small benchmark
conditions and budgets belong to the respective executor/configuration;
they are not defaults for other models. Native receipts identify their origin;
each current comparison records the actual raw-file hashes and evaluator identity.
