# Cartesian XY minimal COMSOL comparison

Status: **PASS**

This package compares seven supplied normalized COMSOL microcases with candidate
runs made only through `load_case`, `simulate`, and `open_result`.  Both are
also checked against independent closed-form motion.  The three fixed steps
are [0.02, 0.01, 0.005] s; the comparison is not tied to matching private solver
internals.

The event-coincident velocity at 0.25 s is excluded for the three
wall cases because the COMSOL state export is left-continuous there and the
candidate frame is right-continuous.  Pre-event and post-event rows, terminal
tail semantics, and the candidate event payload remain gated.

The result is scoped to the seven isolated Cartesian meanings above.  In
particular it is not evidence for arbitrary COMSOL models or curved elements.
