# Layer 03: Release Semantics

## 目的

releaseを単なる初期座標ではなく、物理的な発生源意味論として扱う。

## CanonicalReleaseに必要な情報

```text
particle_id
release_time_s
raw_position
raw_velocity
coordinate_system
source_feature_id
source_boundary_id
source_part_id
projected_boundary_id
projected_part_id
projection_distance_m
capture_tolerance_m
inward_offset_m
initial_position_after_preprocess
normal
tangent_basis
mode
```

## faithful vs production

### comsol_faithful

- raw releaseを勝手にsnapしない。
- invalidならfailまたはimport diagnostic。
- COMSOLが実際に使った位置・速度と比較する。

### surface_release_production

- boundary classificationを許す。
- projectionを許す。
- inward offsetを許す。
- projection distanceをsummaryへ記録する。

## 実装方針

- capture tolerance と inward offset は別設定。
- source id 0 / unknown は明示的に扱う。
- solver hot loopでrelease provenanceを推定し直さない。
