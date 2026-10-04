# P18-H hold / COMSOL Freeze microcase

Status: **PASS** (15 PASS, 0 FAIL).

This compact external V&V package compares the production `hold` / `held`
terminal-boundary semantics with the already locked COMSOL 6.4 boundary 37
Freeze probe.  COMSOL was not rerun.  The one particle starts at
`(r,z)=(0.23927,0.115) m` with velocity `(10,0) m/s` and reaches `r=0.24 m`
in a force-free R-Z domain.

The candidate event time is `7.2999999999998075e-05 s`; the
locked COMSOL value is `7.3000000000041497e-05 s`.  The
candidate has 30 active saved frames and 31 held saved frames.  Its held
position spread is `0.000e+00 m` and its maximum
held-position difference from the aligned COMSOL frames is
`2.776e-17 m`.

This result is deliberately narrow.  It does not make COMSOL a golden truth,
does not certify grazing/corner impacts or full physics, and does not define a
paused particle that can resume.  `hold` is terminal: position, impact velocity,
and charge remain queryable, while no later physics or charge evolution occurs.
The locked COMSOL files contain no charge column, and this external candidate
uses zero charge; its charge gate is therefore a candidate self-consistency
check, not evidence of COMSOL or nonzero-charge retention. Nonzero retention is
covered independently by the public solver scenario.
Exact gates, hashes, algorithm revisions, and exclusions are recorded in
`comparison_result.json` and `gates.csv`.  Candidate raw output remains in the
external run directory named by the comparison manifest and is not duplicated
here.
