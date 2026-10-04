# v0.1 release evidence

This directory preserves the machine-readable evidence used by the v0.1 release gate. It is
evidence for one source/lock revision, not a portable performance promise and not a COMSOL
comparison.

- `p14_release_v27.json`: historical accepted P14-v27 matrix, 18 conditions with three
  fresh-process observations each. Absolute timings are machine-local and non-gating; this is
  not evidence for the current production revision.
- `p14u_release_v1.json`: representative surface-release, nonuniform-field, material-wall gate,
  including 18 raw runs, six medians, convergence/parity checks, and the separate 1M profile.
- `release_platform_smoke.json`: 2026-09-29 local Windows and WSL2 Python 3.12 quality, wheel,
  clean-install, and `load_case` / `simulate` / `open_result` smoke evidence. Its lock and
  algorithm revisions are historical; it is not the remote release-CI receipt for the current
  production revision.

Both performance JSON files contain their machine fingerprint, locked-environment digest,
execution conditions, algorithm revisions, scientific payload digests, and pass/fail checks.
The historical v20 69-observation raw artifact no longer exists; it is not reconstructed from
summaries or current runs.

Regenerate P14 and P14-U with the commands documented in `tests/performance/README.md`. A new
source, lock, algorithm revision, or material measurement condition requires a new evidence file;
do not overwrite evidence while claiming equivalence.
