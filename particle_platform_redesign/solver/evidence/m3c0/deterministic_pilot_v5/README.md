# M3-C0b deterministic pre-event step pilot v5

Status: `CHARACTERIZED`.  The former compact `PASS`/admission decision is
`INVALIDATED`; the completed run and its numerical values remain historical
evidence.

This is compact external V&V evidence for the Case-A, 100 nm COMSOL step
adequacy pilot.  The production solver was not changed or executed by this
pilot.

The audited theory MPH was loaded from an isolated copy with `ModelUtil.loadCopy`
and COMSOL was run with `-nosave -np 1`.  Brownian and Saffman forces were off;
dynamic charge and the electric, relative-flow ion-drag, Epstein drag,
Waldmann thermophoresis, free-molecular lift sensitivity, DEP, and
gravity/buoyancy contributions remained on.  All 287 particles were evaluated
at 46 output times from 0 through 450 microseconds.  This interval ends before
the earliest boundary event seen in the preceding v3 characterization.

The preregistered v4 series, 2.5/1.25/0.625 microseconds, completed correctly
but failed only its charge-order gate (`0.715955 < 0.75`).  Its threshold was
not relaxed.  The planned 0.3125 microsecond extension was therefore run as v5
with the same acceptance limits.  The historical v5 evaluator returned PASS
for all seven rows on the 1.25/0.625/0.3125 microsecond series.  Its observed
orders were position `1.0579673343657396`, velocity `1.000715854935095`, and
charge `1.064923502156796`.  Those raw results are retained, but they no longer
support an accepted step-selection decision.

The v5 position gate used
`||x_h - x_h/2||_2 / ||x_h/2||_2`, where `x` is the absolute global RZ
coordinate.  Translating the coordinate origin changes that ratio and can
change the decision without changing either trajectory.  Consequently the
v5 compact admission is invalid.  The historical fine-pair position value
`6.145964082982306e-7` and its original `5e-6` threshold remain in the manifest
and gate table rather than being erased.  Re-evaluating the same shared pair
with the corrected displacement denominator gives `5.8064307113296814e-5`.
Revision 5 is calibration-only; revision 6 replaces the gate and evaluates it
sequentially on a previously unseen finer result.

The velocity and charge limits were `5e-4` and `2e-5`, and the minimum observed
order was `0.75`.  All limits and raw outputs are historical characterization,
not universal physical-error tolerances or M3-C1 solver-agreement limits.

The full no-clobber output is kept at
`solver/_out_m3c0b/caseA_100nm_theory_pre_event_v5/`.  It contains 43 files
(67,558,107 bytes), including the staged Java source, configuration,
normalizer, raw exports, normalized tables, receipts, logs, and hashes.  This
directory stores only the compact decision record; it does not duplicate the
raw CSV data.

Scope of the retained characterization:

- records the completed pre-event COMSOL fixed-step run for this one Case-A
  100 nm theory case, but does not currently admit its step-selection rule;
- does not establish agreement with the production solver;
- does not certify 30 ms boundary-event convergence, the other particle sizes,
  the image-form ion-drag variant, Case P, Freeze behavior, or Brownian motion;
- does not make COMSOL or `model_dataset` a solver-core dependency.

The active compact decision is the sequential v6 confirmation, which first
separates this operational pre-event step-selection result from later
production-solver agreement.  Full 30 ms and all-case expansion must not use
the old 2.5 microsecond result or this invalidated v5 decision as golden truth.
