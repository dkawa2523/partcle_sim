# M3-C0 critical 2-D boundaries v1

Evaluation status: **COMPLETE**. Scientific decision:
**PASS**. All 132 registered
gates pass with 0 failures.

One from-scratch 10 mm by 20 mm axisymmetric rectangle carries three force-free
particles at the same initial states in COMSOL 6.4 and the production public
API. The three fixed RK4 steps are 1, 0.5, and 0.25 ms; 25 common frames cover
0--6 ms.

| scenario | accepted behavior |
|---|---|
| surface departure | starts at `(r,z)=(6 mm,0)` and remains active while moving into the domain |
| specular reflection | hits `r=10 mm` at 1.95 ms, reverses only radial velocity, and advances the 0.05 ms residual to the first post-hit frame |
| R-Z axis passage | reaches `r=0` at 1.95 ms, stays active, reverses chart radial velocity, and emits no material-wall event |

Maximum COMSOL analytic position/velocity errors are
`1.66615e-17 m` and
`2.28878e-16 m/s`. Maximum production-API
analytic errors are `1.93948e-18 m` and
`0 m/s`. The maximum direct
COMSOL/API differences are `1.61339e-17 m`
and `2.28878e-16 m/s`.

COMSOL boundary 1 is owned by `AxialSymmetry` and is absent from the ordinary
wall selection `[2,3,4]`. Its nonterminal `Bounce` setting is the axisymmetric
chart continuation, not a material-wall event. The production result likewise
reports one axis crossing and no boundary event for particle 3.

This evidence does not certify grazing/corners, multiple material hits,
probabilistic laws, finite-radius contact, forces, fields, or native-field
equivalence. Exact gates and provenance are in `gates.csv` and
`comparison_manifest.json`.
