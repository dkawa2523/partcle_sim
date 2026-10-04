# M3-C2 COMSOL model-semantics probe v1

Status: `COMPLETE_READ_ONLY_CHARACTERIZATION`. No trajectory comparison or
accuracy claim was evaluated.

The retained `ModelUtil.loadCopy` probe inspected the locked theory-consistent
10/30/100 nm MPH with COMSOL Multiphysics 6.4.0.429. It did not run a study or
save a model. The source SHA-256 was unchanged before and after inspection:
`3bbf08e3469758313eac5de473a7a0dd4cc9a6f72c9722393229f0b856e9b524`.

## Exact observations

Both saved particle interfaces have the same root dimensional and RNG mode:

| Case | Physics | Formulation | Include out of plane | RNG arguments | Dataset position DOFs |
|---|---|---|---:|---|---|
| Case P 100 nm | `fpt` | `NewtonianFirstOrder` | `0` | `GenerateUnique` | `comp1.qr`, `comp1.qz` |
| Case A 100 nm | `fptas` | `NewtonianFirstOrder` | `0` | `GenerateUnique` | `comp1.q3r`, `comp1.q3z` |

Both `bf1` nodes are active on domain 3 and expose no component-selection
property. The Case A node is labeled “Brownian force consistent with Epstein
friction” and has:

- `i = AS_brownian_seed`
- `mu = root.comp1.AS_muB`, `mu_mat = userdef`
- temperature `root.comp1.AS_Tg`, source `userdef`
- pressure `root.comp1.AS_pabs`, source `userdef`
- `ParticlesToAffect = All`, `AffectedParticleProperties = pp1`

For completeness, Case P has `i = brownian_seed`,
`mu = root.comp1.muB_d`, and the saved temperature value `293.15[K]` with
temperature source `root.comp1.T`.

The saved Case A construction receipt independently lists the first-order
particle variables `q3r`, `q3z`, `v4r`, and `v4z`, with no `q3phi` or `v4phi`,
at
`model_dataset/cf4_o2_etch_caseA_nonlinear_sass/doc/run_logs/build_run_formal_iondrag_variants.log:6958-6969`.
The probe's Case A dataset is `part_AS_100nm -> sol35`; the construction-log
receipt is corroborating dimensional evidence and is not a claim that its
solution tag is identical.

## Interpretation

The saved interfaces are dynamically R/Z: out-of-plane motion is disabled and
the particle datasets select only radial and axial position DOFs. COMSOL can
still declare or export a three-component Brownian force, including a nonzero
`Fbphi`; that quantity does not by itself create an active phi position or
velocity state. With `IncludeOutOfPlane=0`, there is therefore no particle
`vphi` state from which the documented radial centrifugal term could couple
back into the R/Z trajectory.

Consequently, a built-in-Brownian companion can preserve R/Z trajectory
dynamics by keeping out-of-plane DOFs disabled. This is not the same as strict
two-component Brownian generation: `bf1` has no exposed component selector and
continues to expose its phi force component. Enabling out-of-plane DOFs would
instead add azimuthal state and the documented centrifugal coupling, so it is
not appropriate for the no-swirl companion.

The saved RNG mode is `GenerateUnique`. Although `bf1.i` references a parameter
whose saved Case A value is 21, COMSOL documents the additional Brownian random
argument as applying when the interface RNG-argument setting is `UserDefined`.
Thus 21 is a configured parameter value, not a certified effective seed for
the saved run. A future replica campaign must set `RandomNumberArgs` to
`UserDefined` in its isolated companion copy and verify seed-to-seed divergence
before locking the cohort.

Built-in per-step Brownian force also does not establish semantic parity with
the candidate's exact joint OU update. Any future comparison remains an
independent-seed ensemble comparison, not pathwise equality.

## Reproduction and files

From `particle_platform_redesign/solver`:

```powershell
& .\tools\vv\comsol\run_m3c2_model_semantics_probe.ps1
```

- `model_semantics.jsonl`: eight machine-readable model/interface records.
- `comsol_process_output.log`: exact output from this COMSOL invocation.
- `probe_manifest.json`: normalized findings and read-only execution receipt.
- `source_tool_hashes.csv`: before/after hashes for the MPH and both durable tools.
- `artifact_hashes.csv`: hashes of the generated evidence files.

The launcher is no-clobber, invokes `comsolbatch -nosave -np 1`, checks the
locked COMSOL build and MPH hash, rejects Java containing run/save calls,
validates one root/Brownian/dataset record for each interface, and removes the
compiled class and status file. No solver-core or `model_dataset` file was
modified.
