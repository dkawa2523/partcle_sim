# M3-C1 thermophoresis PPR closure v1

Status: **PASS for saved-row producer-form closure only**.

The isolated COMSOL 6.4 rerun evaluated the Case-A 100 nm `thpf1` feature at
the accepted finest fixed step (`1.5625e-7 s`). The source feature is
Waldmann thermophoresis with `UsePPR=1`, temperature
`root.comp1.AS_Tg`, conductivity `k_mix`, and gas molar mass `Mmix`.
The corrected primitives are:

```text
ppr(d(root.comp1.AS_Tg,r))
ppr(d(root.comp1.AS_Tg,z))
-k_mix*ppr(d(root.comp1.AS_Tg,r))
-k_mix*ppr(d(root.comp1.AS_Tg,z))
```

All 13,202 particle/time rows are unique, finite, and active. Their coordinates
are bit-identical to the locked M3-C0b v6 fine-step states (maximum absolute
difference `0 m`; gate `1e-14 m`). Replaying

```text
F = (32/15) (d0/2)^2 q / sqrt(8 R T / (pi Mmix))
```

from the recovered heat flux matches `fptas.thpf1.Ftfr/Ftfz` with a maximum
component-scale normalized residual of `4.40465e-16` against a `1e-10` gate.
The rerun force is also identical to the locked v6 exported force. This closes
the v6 thermophoretic primitive gap: v6 had exported bare `d(T,r/z)`, while the
feature had used PPR.

The field export contains 33,449 `dset_AS_field` dataset mesh-point rows. The
3D-domain identifier is smoothed at 316 interface rows; the 1,987 exact domain-3
rows are unique and fully finite. Because the export has no node IDs,
connectivity, or element order, the normalized domain-3 table is deliberately
classified as `background_dataset_mesh_point_samples`, not as certified native
FE topology or the native COMSOL interpolant. The raw v1 filename contains the
legacy phrase `native_mesh_nodes`; it is retained only because the completed
run, configuration, and hashes are immutable evidence and must not be read as a
scientific classification.

This evidence does **not** certify integrated trajectory agreement, continuous
path applicability, physical applicability, boundary behavior, or a native
mesh/interpolant reconstruction. COMSOL is an external comparison producer,
not golden truth.

Recheck without overwriting normalized artifacts:

```powershell
uv run --locked python tools/vv/comsol/evaluate_m3c1_thermophoresis_ppr.py `
  _out_m3c1/caseA_100nm_thermophoresis_ppr_v1 `
  --config tools/vv/comsol/cases/m3c1_caseA_100nm_thermophoresis_ppr_v1.json `
  --check-only
```

Machine-readable details and artifact hashes are in [report.json](report.json).
