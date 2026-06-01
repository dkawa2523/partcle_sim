# Layer 07: Artifacts and Diagnostics

## 目的

比較と保守に必要な成果物だけを安定的に出す。

## 通常runで許可

```text
run_summary.json
input_contract_report.json
provider_contract_report.json
source_model_summary.json
compare_summary.json
```

## compare runで許可

```text
first_step_error.csv
force_contributions.csv
boundary_hit_comparison.csv
state_fraction_time_series.csv
vacuum_time_summary.json
```

## deep/debug only

```text
wall_events.csv
collision_diagnostics.json detailed
trajectory_samples.csv
animations
```

## shard run

rootに集約すべきもの:

```text
source_particle_diagnostics.csv
run_summary.json
compare_summary.json
```

shard完走だけでは成功扱いにしない。比較ツールがroot artifactsを読めることを成功条件にする。
