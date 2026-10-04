# M3-C0b deterministic pre-event step-selection confirmation v6

This is the compact external V&V decision record for the Case-A, 100 nm
theory COMSOL pilot.  Its status is `PASS` only for sequential confirmation of
the operational pre-event step-selection gate.  It is not production-solver
agreement, a universal physical-accuracy result, or a physical-applicability
certificate.

Revision 5 supplied the pilot data used to define the replacement gate, but
its former compact `PASS` decision was invalidated because its position
relative L2 divided by the norm of the absolute global RZ coordinates.  That
metric changes when the coordinate origin is translated.  Revision 6 instead
uses the origin-invariant displacement metric

```text
||x_h(t,p) - x_h/2(t,p)||_2
---------------------------------------
||x_h/2(t,p) - x_h/2(0,p)||_2
```

where each norm stacks all particle/time records and each fine-run trajectory
is referenced to that particle's own fine-run initial position.  Revision 6
evaluates it on the previously unseen 0.15625 microsecond result.  On the
shared 0.625-to-0.3125 microsecond pair, the corrected displacement-relative
value is `5.8064307113296814e-5`, versus the origin-dependent historical value
`6.145964082982306e-7`.  Revision 5 is calibration-only historical
`CHARACTERIZED` evidence; revision 6 supersedes its scientific PASS for the
position gate.

The audited theory MPH was loaded from an isolated copy with
`ModelUtil.loadCopy`; COMSOL ran with `-nosave -np 1`, and the source MPH hash
was unchanged.  Brownian and Saffman forces were off.  Dynamic charge and the
electric, relative-flow ion-drag, Epstein drag, Waldmann thermophoresis,
free-molecular lift sensitivity, DEP, and gravity/buoyancy contributions were
on.  Each fixed-step run contains 287 particles at 46 output times from 0
through 450 microseconds: 13,202 records, all active, with no observed boundary
event.

The series is 0.625/0.3125/0.15625 microseconds.  All seven operational gates
passed:

| gate | value | limit |
|---|---:|---:|
| position observed order | `0.9041136219836081` | `>= 0.75` |
| velocity observed order | `0.9443119501831206` | `>= 0.75` |
| charge observed order | `1.1238380758116033` | `>= 0.75` |
| fine-pair position displacement relative L2 | `3.102727085428027e-5` | `<= 1e-4` |
| fine-pair velocity relative L2 | `3.92483251084038e-5` | `<= 5e-4` |
| fine-pair charge relative L2 | `1.3511393490811483e-6` | `<= 2e-5` |
| all records active | `true` | `true` |

For the 0.3125-to-0.15625 microsecond fine pair, the position, velocity, and
charge RMS differences are respectively `5.338714552062531e-8 m`,
`3.6430540472116613e-4 m/s`, and `3.3741954922298354e-4 e`.  The observed
orders are about one and are not interpreted as the formal RK4 order through
piecewise finite-element and derived fields.

The shared 0.625 and 0.3125 microsecond normalized trajectory, force, RHS, and
event artifacts are bitwise identical between v5 and v6.  This confirms overlap
reproducibility; it does not rescue the origin-dependent v5 decision rule.

The full no-clobber output is at
`solver/_out_m3c0b/caseA_100nm_theory_pre_event_v6/`.  It contains 43 files
(67,561,848 bytes).  All 41 entries in its artifact hash index were rechecked
against file size and SHA-256; the index itself and the final run-status file
are the two intentionally unindexed files.  `comparison_manifest.json` records
the exact metrics, hashes, protocol, overlap check, and non-gating force-table
sanity results.

This result does not certify 30 ms boundary-event convergence, other particle
sizes, the image-form ion-drag variant, Case P, Freeze behavior, Brownian
motion, or the production solver.  Those comparisons remain later external
V&V work, and COMSOL and `model_dataset` remain outside the solver core.
