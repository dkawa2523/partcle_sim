# Case A/P full-step Brownian parallel and COMSOL comparison evidence

[Root verdict](raw/r65/verdict.json) · [Performance observations](raw/r65/performance_comparison.json) · [COMSOL comparison figures](raw/r65/derived/index.html)

This archive closes bytes/members/hashes only. Root's verdict and existing evaluators own scientific conclusions.
Current source is v62 with 27 production files and 237 bound files, unchanged from R64 C4; no new engine, equations, JIT, or COMSOL solver path was introduced.
The two canonical inputs and 64 generated YAMLs are stored separately from results.
All 32 current candidate seeds per case completed serially: 64 new results with 287 particles, 30 ms, 1500 macro steps, and normalized tables at 121 times.
Parallel performance stores six roles (Case A/P at 1, 2 and 4 workers), each with one fresh-cache cold and three fixed-seed warm runs: 24 results with sampled process metrics.
Parallel identity covers only the fixed three seeds per case, compared against matching current serial arms; this is not a parallel campaign for all 32 seeds.
One additional fresh full Case P run at seed 939043 validates the actual interruption/resume against the existing cohort result. It reuses the same YAML/canonical input and is separate from the 64 cohort and 24 timing results (89 new completed results total).
The pre-resume receipt retains hashes observed at the retired partial path as historical metadata; those past objects are not redirected or requalified as current files.
Original saved COMSOL arms with 32 seeds per case (64 total), original model MPH and existing proof remain explicit hash references only; no new native solve, old archive expansion, or modification occurs.
Previously observed fixed cohorts are a finite-algorithm regression, not a new unseen confidence family or pathwise COMSOL RNG equality.
Registered evaluator outputs, derived plots, root verdict, task logs/scripts, and separately frozen current documents are preserved as raw bytes.
Caches, environments, bytecode and MPH payload copying are excluded. Sampled WS is not unique physical memory; timing is observational.
No original file is deleted or overwritten. Formal release/remote-CI qualification is not inferred.
