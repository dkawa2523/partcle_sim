# Layer 05: Boundary Event and Release Grace

## 目的

wall hitをhit-time stateで処理し、same-source skipをrelease直後の短いgraceに限定する。

## 原則

```text
wall event = segment crossing + hit-time solve + wall policy
```

endpoint補正ではない。

## Release grace

許可条件:

```text
source_part_id > 0
hit part == source_part_id
outward normal speed > 0
hit time <= release_time + grace_time
distance/time envelope is small
```

禁止:

```text
inward reimpact skip
unrelated wall skip
unknown source skip
long-lived same-source bypass
```

## minimal diagnostics

通常:

```text
source_surface_release_skip_count
source_surface_release_skip_blocked_count
unresolved_crossing_count
max_hits_reached_count
numerical_boundary_stopped_count
```

deep modeのみ:

```text
wall_events.csv
collision trace samples
```

## 避けること

- same-source skipを性能最適化だけとして扱う。
- runtime改善だけで採用する。
- VIGUS source partだけに条件を書く。
