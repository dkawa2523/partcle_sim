# F02 field-production integration evidence

`f02_closeout_v1.json` records the small, reproducible acceptance summary for
the COMSOL CSV adapter → reduced electrostatic builder → existing trajectory
solver path.  It is external field-production/V&V evidence, not a solver-core
golden file and not a claim that COMSOL is the physical truth.

The large generated HDF5 files and result directory are intentionally omitted.
Regenerate them with the commands in `tools/comsol_adapter/README.md`,
`tools/electrostatic_builder/README.md`, and `tools/vv/comsol/README.md`; the
configuration, source, canonical-output, comparison-report, and reference
hashes in the JSON are the durable identity.  Timings are descriptive
observations on the recorded machine.  The field comparison covers one
exported mesh at the same nodes; independent mesh convergence and COMSOL
trajectory/boundary parity remain explicitly untested.
