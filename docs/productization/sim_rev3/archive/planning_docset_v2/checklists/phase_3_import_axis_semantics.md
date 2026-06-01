# Phase 3 Checklist: Import / Coordinate / Axisymmetric Minimum Semantics

## 目的

coordinate, scale, axis, field support, axisymmetric_rzを最小限固定する。

## 実装

- coordinate system preserved
- coordinate scale required in faithful mode
- field axes and geometry axes compared
- axisymmetric_rz parsed and reported
- r=0 axis not silently treated as cartesian wall

## Acceptance

- axis mismatch caught early
- axisymmetric_rz config survives runtime build
- field support status included in summary
