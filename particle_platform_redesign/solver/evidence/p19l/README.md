# P19-L hot-path observation

[`performance_v1.json`](performance_v1.json) records a machine-local,
non-gating public-API comparison of the two intended applicability paths.  The
driver is
[`tests/performance/p19l_local_applicability.py`](../../tests/performance/p19l_local_applicability.py).

Both cases contain 4,096 identical particles in the safe left cell of the
accepted XY Stokes-Cunningham scenario and run 20 fixed general-RK4 macro
steps.  The only input difference is an unused far-right gas-velocity value:
`0 m/s` lets the global certificate prove the whole path, while `100 m/s`
makes that global proof inconclusive and invokes the bounded local-cell proof.
The benchmark uses `load_case`, `simulate`, and `open_result`; it does not
replace or mock an internal call.

Run it from `solver/` with:

```console
uv run --locked python -m tests.performance.p19l_local_applicability \
  --particles 4096 --repeats 7 --json .artifacts/p19l-performance-v1.json
```

## Observation

After warming both cases, seven `simulate` observations per mode were run in
alternating order.  The global-fast median was `0.9276381 s`; the local-fallback
median was `0.9345506 s`, for a local/global ratio of `1.00745`.  This roughly
one-percent separation is descriptive noise-scale evidence, not an absolute
speed PASS or a portable regression threshold.

A separate untimed `cProfile` run provides the structural result that timing
alone cannot:

- both modes attempted the global certificate once per macro step (20 calls);
- global-fast made zero `local_component_bounds` calls;
- local-fallback made exactly 20 `local_component_bounds` calls.

The profile's `local_continuous_applicability_batch` name aggregate is 40 in
the fallback case because the engine wrapper and physics-runtime method share
that function name.  The field-owner call count above is the unambiguous
local-range construction observation.

The final particle table plus release, boundary, failure, and lifecycle tables
were bitwise identical, with common digest
`sha256:a32d22ba3529c15d6e0b122088a38be75a9b961067ff07d85ed2deab5a917949`.
All 4,096 particles remained active; neither case emitted a boundary or failure
event.  Manifests are intentionally excluded from this identity because the
two input content hashes differ.

Separate `tracemalloc` scopes observed 9,926,561 B global-fast and 9,927,244 B
local-fallback additional peaks.  These include Python/NumPy allocations that
the tracer can see and result writing; they are not RSS and do not replace the
solver-owned plan.

## Bounded-memory audit

The two modes resolved the same one-slab `solver_owned_memory_plan_v13`:
18,538,545 planned bytes, 4,096 slab rows, 176 B/row dense path, 616 B/row
certificate work, and 2,048 B/row general stage scratch.

- The immutable dense path is exactly 176 B/row: two float64 times, `4x2`
  float64 position controls, `4x2` velocity controls, and four charge controls.
- With the benchmark split budget of two, the interval workspace uses
  `16*(2+1)+18 = 66 B/row`; the plan rounds this to 72 B/row.
- The maximum local candidate construction simultaneously holds int64
  `counts`, `bounded_counts`, `N+1` CSR offsets, and at most `64*N` candidate
  IDs: `536*N+8` B.  The named arena is `544*N` B, which contains that maximum
  for every `N >= 1`; at 4,096 rows it leaves 32,760 B headroom.
- The benchmark's four primitive fields have five components, so retained
  local lower/upper ranges use 80 B/row.  These and arithmetic temporaries are
  covered by the separate 2,048 B/row general stage scratch, rather than being
  counted again in the candidate arena.

All corresponding manifest/component equality checks are recorded as true in
the JSON.

## Production-path ownership

`enclose_rk4_path` and `RK4_ENCLOSURE_REVISION` remain production code.  They
own the conservative global enclosure used by rev3b material-wall/RZ event
geometry; they are not an obsolete applicability oracle.  The Hermite dense
path and its subinterval chord own applicability certification.  Keeping those
roles separate preserves one event-semantics owner and one certificate owner.

The formerly caller-free `restrict_rk4_dense_proposal` helper and its private
subset chain are absent from the measured tree.  No second integrator or
benchmark-only production path was retained.

This evidence is a current-tree fast-versus-local observation.  It is not a
pre-P19 result, a COMSOL comparison, a physical-validity claim, or a timing
acceptance gate.  It was captured with physics runtime v16, as recorded in the
JSON; a later unrelated revision does not retroactively relabel this
observation.
