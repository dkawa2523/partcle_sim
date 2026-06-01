# Layer 06: Axisymmetric RZ

## 目的

axisymmetricをcartesianの別名ではなく、座標系意味論として扱う。

## Phase 3で最低限必要

```text
coordinate_system = axisymmetric_rz
axis names = r, z
r=0 axis boundary semantics
manifest/config/preflightに明示
cartesian_xyとの混同禁止
```

## Phase 7で完成させるもの

```text
ring-area weighting = 2πr
source surface sampling measure
boundary accumulator measure
axis boundary policy
optional v_theta-compatible state design
```

## 避けること

- RZをxyとして処理する。
- r=0を通常wallと混同する。
- axisymmetric source samplingで長さ重みだけを使う。
- v_thetaを急いで入れて大改造する。
