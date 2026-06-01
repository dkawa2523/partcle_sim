# Phase 7 Checklist: Axisymmetric RZ Completion

## 目的

RZとして必要な面積重み・axis boundary・source measureを完成させる。

## 実装候補

- ring_area_weight(r) = 2πr
- source sampling weighting
- boundary accumulator weighting
- axis boundary policy
- state design ready for v_theta

## Acceptance

- simple known ring weighting test passes
- cartesian behavior unchanged
- RZ not silently downgraded to XY
