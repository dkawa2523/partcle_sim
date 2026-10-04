# P18-R performance observation

`performance_v1.json` is a machine-local, non-gating comparison of the old
single-species revision pair and the P18-R effective-gas sensitivity revision
pair. Both configurations used the same 100,000 particle rows, sampled
primitives, `maximum_speed_ratio=0.1`, prepared inputs, and one reused
`PhysicsRuntimeWorkspace`. After both paths were warmed, the old and new
configurations were evaluated alternately nine times each.

The medians were 0.0267117 s for the old pair and 0.0267643 s for the new pair,
a new/old ratio of 1.00197. The complete stage payload was bitwise identical,
all 100,000 rows passed both stage and continuous-path applicability, prepared
bound storage was 2,500,040 B for each runtime, and the shared workspace was
6,600,000 B. The revision selection therefore adds no observed numerical or
memory divergence and no material stage-time change in this focused run.

This measurement characterizes only the same-form warm stage. It is not an
end-to-end or portable performance threshold and does not validate COMSOL
trajectories, mixture physics, or the effective-gas approximation. The older
P18-L fixed-charge/lift evidence used compiled tile v15 and is historical
context, not the direct baseline for this P18-R comparison.
