# Skill 06: Particle Release and Reference Trajectory Export

## Purpose

COMSOL Particle Tracing結果と本solverを比較するため、release tableとlong trajectory tableを正しく抽出・正規化する。releaseとtrajectoryの意味がずれると、wallやforceを触っても比較が壊れる。

## Required release fields

At minimum:

```text
particle_id
release_time_s
position components in model units or meters
velocity components
particle diameter / density / mass if model dependent
charge if used
source boundary/feature ID if available
```

## Required trajectory fields

For faithful trajectory comparison:

```text
particle_id
time_s
position components
velocity components if available
state or event indicator if available
boundary hit information if available
```

## Steps

### 1. Preserve raw COMSOL release

Never overwrite raw release exports. Store normalized copies separately:

```text
particles_release_raw.csv
particles_release_canonical.csv
```

### 2. Check identity continuity

Verify:

- every trajectory particle_id has a release row
- release times match first trajectory time or documented COMSOL semantics
- sampled references are labeled sampled
- full references are labeled full

### 3. Do not apply production boundary preprocessing to faithful release

In `comsol_faithful` mode:

```text
no snap
no inferred source part
no generated particles
no boundary release preprocessing
```

If release points appear off-grid, report it in `release_export_gap_report.md`.

### 4. Handle stochastic references

If Brownian or random scattering is present:

- do not expect particle-by-particle deterministic agreement unless same random stream is available
- first run deterministic comparison with stochastic disabled if possible
- then compare distributions/ensembles

## Outputs

```text
particles_release_raw.csv
particles_release_canonical.csv
comsol_particle_long.csv
release_validation_report.json
```

## Pass criteria

- Particle IDs are stable.
- Release time and initial velocity are explicit.
- Sampled vs full reference is labeled.
- Stochastic policy is recorded.

## Fail criteria

- Comparing a generated solver particle population to COMSOL particle IDs as if they were identical.
- Inferring source part from nearest wall in faithful mode.
- Mixing sampled and full reference metrics without labels.
