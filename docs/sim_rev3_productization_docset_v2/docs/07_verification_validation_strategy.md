# 07. Verification and Validation Strategy

## Verification

コードが意図した数値モデルを解いているかを、小さな合成ケースで検証する。COMSOLは使わない。

候補:

```text
V1 no force, constant velocity
V2 uniform Stokes drag relaxation
V3 constant electric acceleration
V4 drag + constant electric field
V5 single flat wall specular bounce
V6 stick wall
V7 release capture vs inward offset
V8 first-step deterministic force breakdown
V9 axisymmetric ring weighting utility
V10 segment hit-time detection
```

## Validation

COMSOLや他シミュレーション結果と比較する。ここでは完全な1粒子逐点一致ではなく、比較対象を分ける。

```text
C1 import/field parity
C2 force/acceleration parity
C3 first-step parity
C4 single-particle deterministic trajectory
C5 first-wall-event parity
C6 sampled ensemble distribution
C7 full release distribution
C8 production surface-release smoke
```

## 合格判断

単一metricで判断しない。最低限、次のグループを同時に見る。

```text
import:
  field support, coordinate, boundary map

preprocess:
  release classification, projection distance, initial support

first-step:
  position error, velocity error, force contribution error

event:
  first hit time, first hit boundary, wall action

ensemble:
  active/stuck/escaped fractions over time
  first-passage CDF
  source-wise distribution

runtime:
  collision time
  skipped/blocked wall events
  unresolved crossings
```

## Brownian / stochastic handling

deterministic verificationでは stochastic force をoffにする。

ensemble validationではseed policyと分布指標を使う。

```text
Do not judge stochastic trajectories by particle-by-particle exact matching.
```
