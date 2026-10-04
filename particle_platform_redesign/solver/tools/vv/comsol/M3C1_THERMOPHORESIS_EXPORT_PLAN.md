# M3-C1 thermophoresis closure export

This is an external V&V rerun plan. It does not change the production solver,
the accepted M3-C0b v6 artifacts, or their historical source hashes.

## Exact model setting and formula gap

The Case-A 100 nm reference selects COMSOL's `Waldmann` thermophoretic-force
model with:

- `UsePPR=1`;
- temperature `root.comp1.AS_Tg`;
- thermal conductivity `k_mix`;
- molar mass `Mmix = 0.0768032 kg/mol`;
- particle radius `d0/2`.

For this model, the force used for replay is

```text
F_th = -(32/15) a^2 k sqrt(pi M_g / (8 R T)) grad(T)
     =  (32/15) a^2 q / c_bar

q     = -k grad(T)
c_bar = sqrt(8 R T / (pi M_g)).
```

With `UsePPR=1`, the gradient entering the COMSOL feature is recovered rather
than the unrecovered `d(T,r)` and `d(T,z)` saved by M3-C0b v6. COMSOL documents
the expression-level recovered derivative as `ppr(derivative)`, so the first
expressions to export and validate are:

```text
ppr(d(root.comp1.AS_Tg,r))
ppr(d(root.comp1.AS_Tg,z))
-k_mix*ppr(d(root.comp1.AS_Tg,r))
-k_mix*ppr(d(root.comp1.AS_Tg,z))
```

The expressions must be treated as producer primitives only after their
reconstructed Waldmann force agrees with `fptas.thpf1.Ftfr` and
`fptas.thpf1.Ftfz` on every saved active row. If COMSOL does not permit the
direct operator in the particle dataset, define component variables with the
same expressions and export those variables. Inferring a heat flux from the
already exported force is circular and is not an acceptable closure.

M3-C0b v6 currently exports `-k_mix*d(AS_Tg,r/z)`. Its 15.2% maximum
component-scale residual is therefore diagnostic only; it is not evidence
that the production formula is wrong.

## Minimal no-clobber COMSOL rerun

1. Add a new M3-C1 Java runner and configuration revision. Do not edit the v6
   runner, configuration, output, or compact evidence.
2. Load the same source MPH with `ModelUtil.loadCopy`, run with `-nosave -np 1`,
   and verify the source SHA-256 before and after.
3. Run only the accepted finest pre-event step, `1.5625e-7 s`, at the existing
   46 output times from 0 through `4.5e-4 s`; keep Brownian off and retain all
   287 particles.
4. Export the existing state and force columns plus the two recovered
   temperature-gradient components and two recovered heat-flux components.
5. Stage the exact Java source, class hash, configuration hash, COMSOL receipt,
   source-model hashes, and raw-table hashes inside a new no-clobber output
   directory.
6. Replay the Waldmann force. Promote thermophoresis producer-form parity only
   if all rows are finite, keys are unique, the particle/time cohort is exact,
   and the configured residual gate passes. Production-model comparison and
   physical applicability remain separate statuses.

No additional three-step COMSOL sequence is needed to close this primitive
gap. A new convergence sequence is required only if the physics or accepted
time-step protocol changes.

## Native-mesh field export needed later

A separate native-field artifact should preserve interpolation ownership. At
minimum it needs:

- mesh vertex coordinates, triangle connectivity, domain/support identifiers,
  and the background solution/dataset identity;
- `AS_ugr`, `AS_ugz`, `AS_Tg`, `AS_rhog`, `AS_mug`, `AS_lambdag`, pressure,
  and `k_mix`;
- the recovered thermophoretic heat-flux vector above and the azimuthal gas
  vorticity used by the lift model;
- `AS_Er`, `AS_Ez`, `|E|^2`, and the exact recovered derivatives used by DEP;
- electron and positive-ion densities, ion velocity, effective ion mass, ion
  thermal energy, screening length, and ion-neutral mean free path;
- units, coordinate convention, mesh/solution hashes, and whether each
  derivative is direct, `pprint`, or `ppr` recovered.

If the source solution uses higher-order elements, vertex values plus linear
triangles are a projection, not the native COMSOL interpolant. Either export
the element order and required degrees of freedom, or explicitly classify the
linearized artifact as a common-field diagnostic. It must not be described as
native-field trajectory parity.
