# Generic COMSOL preparation and particle comparison evidence

This campaign records the completed source/adapter/V&V work for the original
10 nm speed10000 Case A/P inputs. Case labels, 287-particle populations, clocks,
steps and margins are evidence for these runs, not defaults for another chamber.

- [Scientific run plan](scientific_run_plan.json) and
  [scope/source binding](scope_and_source_binding.json) identify the compared
  problem and the three candidate RK4 resolutions.
- [Quality and review receipt](quality_and_review_receipt.json) records the
  tested source bytes and the completed 976 verification/scenario and 61
  focused external-tool tests. Its older skill/document hashes identify the
  version reviewed then; later skill changes have their own
  [review record](../skills_general_review_2026_10_14/review.md).
- [Independent numerical review](skill_review/independent_numeric_review.json)
  and [comparison procedure](trajectory_review/analysis_plan.json) describe
  the recomputed trajectory metrics and validity masks.
- [All-particle side-by-side page](all_paths_side_by_side.html) shows the saved
  solver paths on the left and original COMSOL paths on the right, with shared
  limits/aspect and all requested particles. Its
  [receipt](all_paths_side_by_side_receipt.json) binds source and figure bytes.

Original native FE/noise/force settings and the candidate's transferred fields
and explicitly mapped models differ. Spatial similarity and measured outcomes
are useful scoped evidence; they do not certify original-model numerical parity
or physical validity. Source and evaluation limits remain in the linked records.

Git retains selected source scripts, plans, readbacks, conclusions and the
standalone side-by-side figures. The large MPH, canonical HDF5/results, raw or
normalized CSVs, historical atlases and their bulk figure assets remain external
payloads. Local absolute paths and SHA-256 values in historical receipts are
provenance, not portable defaults; do not rewrite them to claim a new execution.
Reproduction scripts need the actual bound external inputs and current native
access where a new COMSOL export/solve is requested. A Git checkout alone does
not include those payloads or prove that a native solve can run.
