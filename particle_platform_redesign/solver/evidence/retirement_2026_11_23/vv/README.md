# External comparison cleanup and executed verification — 2026-11-23

## Scope and roles

The current external boundary covers native observation binding, Vigus
coefficient preparation, deterministic pre-event descriptors, saved population
description, actual-data figures, local charge/ion formula checks, reduced-field
comparison, exact-P1 export and small native/analytic benchmarks.
Normal input/run/result reuse remains in the installed package.

The audited starting tree had 200 tracked COMSOL tool/config/test files.
160 obsolete fixed campaign files were removed; their exact paths are in
`removed_paths.json`. This retires M3-C0/C1/C2/C3 and P18 campaign launchers,
seed/pilot allocation locks, fixed 287-particle/41-frame preparation and
normalization, saved-result reporting repair/projection routes, source-freeze
contracts and their dedicated configs/tests. No compatibility wrappers remain.
The orphaned campaign-specific Java readback helper also depended on the
retired CommonP1Epstein class and hard-coded its companion parameters. It was
removed with those callers; observed readback normalization and boundary
meaning remain in the generic Python tools.
Generic descriptor comparison replaces fixed matched-trajectory comparison;
installed CSV/create/run operations replace obsolete candidate orchestration.
Current core force/charge/event/stochastic verification preserves the
scientific checks. The aggregate primitive formula evaluators remain because
they compare actual native columns with independently reconstructed equations.

The inspected native dataset, MPH files, real experiment records and
environments were not deleted. The 29-file, 103615-byte native reference
subset (24 native raw CSVs plus observed settings/provenance/native source)
is retained under `tools/vv/comsol/references`; its exact identities are in
`reference_inventory.json`. XY records predate this change; RZ was freshly
generated. These are reference inputs, not old candidate outputs or callable
production snapshots. The reference directory also has a current README.

## Corrected responsibilities

* XY replay no longer requires the current evaluator/runner bytes to equal a
  historical version. It verifies the recorded native producer, raw hashes
  and actual configuration, then records the current evaluator hash and
  public candidate receipts. Obsolete Python/PowerShell snapshots are omitted.
* RZ replay writes candidate input/results under its output directory.
  Reference inputs stay unchanged, including when the test runs repeatedly.
* The force-free RZ benchmark no longer claims reused terminal-boundary
  qualification from a retired campaign. Terminal behavior is outside that
  nonterminal benchmark; XY stick/hold checks and core tests cover their own
  explicit scope.
* The previous static PASS/source-hash tests were replaced with actual public
  execution against independent native observations and closed-form motion.
  Native raw tampering is rejected before candidate execution.

## Executed checks

```console
uv run --locked python -m pytest tools/vv/comsol/tests -q
uv run --locked ruff check tools/vv/comsol
uv run --locked pyrefly check --summarize-errors
```

The full retained external suite passed **127 tests** (50.08 s on the second
whole-suite run); the final focused native-replay rerun passed 3 tests in 13.79 s after the
typed-test and immutable-reference assertions. Ruff passed. Configured Pyrefly reported zero errors.

The benchmarks were actually run through the public candidate API:

| Benchmark | Independent reference | Current result |
|---|---|---|
| XY, seven scenarios × three steps | 21 hash-bound native observations + analytic motion | All four scientific judgments PASS; linear-drag candidate position/velocity orders 4.018 / 4.941 |
| RZ surface departure / specular residual flight / axis fold, three steps | Fresh unsaved native model + piecewise analytic motion | 132/132 gates PASS; max direct position difference 1.613386e-17 m, velocity 2.288783e-16 m/s |
| Unit-circle wall, 16/32/64/128 line2 facets | Analytic circle intersection/normal/specular law | PASS; time/point order 2.003, normal 0.998, reflected velocity 0.997 |

The generated numeric tables and compact reports are `xy_metrics.csv`,
`rz_gates.csv`, `curved_metrics.csv`, and the corresponding
`*_comparison.json` files. Full current outputs are local
`.artifacts/retirement_vv/{xy_final,rz_final,curved_final}`.
Fresh native RZ launch used:

```powershell
./tools/vv/comsol/run_m3c_critical_boundaries.ps1 -OutputDirectory .artifacts/retirement_vv/rz_native -EvidenceDirectory .artifacts/retirement_vv/rz_evaluation
```

After the output-ownership correction, the same observed native inputs were
re-evaluated without another native solve:

```console
uv run --locked python -m tools.vv.comsol.evaluate_xy_minimal_suite tools/vv/comsol/references/xy/raw .artifacts/retirement_vv/xy_final --run-receipt tools/vv/comsol/references/xy/run_receipt.json
uv run --locked python -m tools.vv.comsol.evaluate_m3c_critical_boundaries tools/vv/comsol/references/rz --config tools/vv/comsol/cases/m3c_critical_boundaries_v1.json --evidence .artifacts/retirement_vv/rz_final
uv run --locked python -m tools.vv.comsol.evaluate_curved_wall_line2_convergence .artifacts/retirement_vv/curved_final
```

XY native/candidate velocity has opposite continuity at the exact event frame;
only that named row is excluded for affected wall cases. Pre/post states,
terminal tails and candidate event payload remain checked. These small tests
do not certify arbitrary native FE transfer, full plasma models, Brownian
pathwise agreement, all walls/corners, finite-radius contact or physical
adhesion/flux. The previous whole-population A/P practical comparison remains
separate scientific evidence, not a new native solve or statistical sample.

## Closure checks

Active modules were traced through imports, command entries, CI, configs,
skills and callers before deletion. No untracked retired Python/configuration
entry remained in the tools directory; preexisting retired bytecode caches
were removed after confirming their resolved scope. The full surviving suite
requires neither historical campaign registrations nor `model_dataset`.
The root completion audit owns independent residual/overdeletion review and
final integrated checks.
