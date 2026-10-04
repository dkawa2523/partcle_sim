# M3-C1 field-representation localization v1

Status: **PASS_LOCALIZED** for the locked Case-A 100 nm diagnostic slice.

This external V&V evaluation uses only existing immutable artifacts. It does
not rerun COMSOL, integrate a new trajectory, or change production code. At
the same 287 initial particle states, it samples the canonical exact-
connectivity P1 fields and compares them with the saved COMSOL native-field
primitives. It then sends each primitive set through the same production pure
charge and force functions.

The first observed difference is therefore localized before integration, at
the field-representation or field-sampling layer:

| identical `t=0` comparison | relative L2 |
|---|---:|
| electric field | `2.8620728726031667e-2` |
| electron density | `4.8928705740119774e-2` |
| positive-ion velocity | `2.6968376339439284e-2` |
| gradient of mean electric-field magnitude squared | `1.259955413662082e-1` |
| authoritative PPR heat flux | `7.921763663068737e-3` |
| resulting charge rate | `4.9101396439728405e-2` |
| resulting total acceleration | `3.989452875541078e-2` |

All nine preregistered gates pass. The 13,202 saved rows form the complete
287-particle by 46-frame active cohort and every sample is supported by the
canonical P1 mesh. The 1,987 PPR domain points bijectively match the canonical
nodes; after applying the registered axis projection, heat-flux relative L2
is `6.338106055700018e-17`. Initial particles begin at `r=0.14 m`, outside all
axis-incident cells (`r <= 0.007517953530915272 m`), so the axis policy cannot
explain the initial difference.

This result keeps earlier decisions separate: cross-representation trajectory
comparison remains `FAIL 6/6`, same-common-P1 comparison remains `PASS 9/9`,
and frozen producer-form plus later PPR closure remain passed within their
saved-row scope. It does not reconstruct COMSOL's native FE interpolant,
attribute an integrated error to individual fields, certify physical validity,
or claim universal COMSOL-equivalent accuracy.

No COMSOL rerun is needed for this localization decision. If a narrower split
between interpolant representation and saved-path sampling bias is later
required, the next minimal discriminator is a static export of native/PPR
primitives at the already saved candidate coordinates from the existing
stationary solution; it does not require a particle solve. A COMSOL particle
rerun is required only for later integrated grouped-hybrid attribution.

Reproduce into a new no-clobber directory:

```powershell
uv run --locked python -m tools.vv.comsol.evaluate_m3c1_field_localization `
  . `
  tools/vv/comsol/cases/m3c1_caseA_100nm_field_localization_v1.json `
  _out_m3c1/caseA_100nm_field_localization_v1_recheck
```

Exact field, RHS, and gate rows are in `field_metrics.csv`, `rhs_metrics.csv`,
and `gates.csv`. `report.json` records the decision, limits, hashes, and claim
boundary.
