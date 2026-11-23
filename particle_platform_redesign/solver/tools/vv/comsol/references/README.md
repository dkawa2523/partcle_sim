# Native benchmark reference inputs

These are measurements from isolated unsaved COMSOL 6.4 models, not old
candidate results, a production engine, or an alternate case/result format.

* `xy`: seven Cartesian scenarios at three steps, collected on 2026-10-08.
  The original native run receipt binds all 21 raw CSV files, the compiled
  class, observed configuration log and native Java producer. Only the native
  producer source is included; obsolete candidate/evaluator/runner snapshots
  are not needed to consume the native observations.
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
they are not defaults for other models. The current raw-file SHA inventory
and executed comparisons are in
`solver/evidence/retirement_2026_11_23/vv`.
