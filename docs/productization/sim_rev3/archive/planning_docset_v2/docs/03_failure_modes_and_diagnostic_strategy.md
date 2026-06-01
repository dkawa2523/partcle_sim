# 03. Failure Modes and Diagnostic Strategy

## 目的

この文書は、COMSOLとの差分を「どの層の問題か」に分解するための診断設計を示す。Codexが局所数値だけを見て修正することを防ぐ。

## 比較断面

最低限、以下を出す。

```text
post-import:
  geometry / field / particle / wall map が同じ座標系で読めているか

post-preprocess:
  release classification, projection, inward offset, field support が正しいか

post-first-step:
  release直後の速度・位置・力寄与が崩れていないか

first-wall-event:
  最初の壁hit時刻、位置、boundary_id、part_id、wall action

ensemble-time-series:
  active/stuck/escaped/vacuum fraction, first-crossing CDF

runtime:
  collision/event時間、skip/block数、unresolved crossing
```

## 症状から疑う層

### A. import段階で差がある

疑うもの:

- coordinate scale
- coordinate system
- field axis
- mesh/geometry boundary
- boundary map
- part map
- field support mask

solver coreを触らない。

### B. post-preprocessは良いがfirst-stepで崩れる

疑うもの:

- initial velocity semantics
- release normal orientation
- field sampling point
- drag update
- electric force sign/unit
- charge model
- integrator dt
- Brownian/stochastic force

wall lawより先にforce/first-stepを見る。

### C. near-wall active が多いのにwallhitが無い

疑うもの:

- same-source skipが広すぎる
- on-boundary判定が誤っている
- segment hitが抜けている
- source id / part id が不明扱い
- release graceが長すぎる

### D. 特定wallだけ直すとlocalは良くなるがglobalが悪化

疑うもの:

- wall lawは症状であって主因ではない
- release semanticsやfirst-step dynamicsが先にずれている
- sampled指標だけで判断している

mainlineへ入れない。local probeとして保持。

### E. runtimeが悪い

疑うもの:

- collision resolverが同じsource wallを何度も処理している
- max hits / numerical boundary stop が出ている
- spatial indexが効いていない
- diagnosticsが常時詳細すぎる
- shard集約やpath resolutionが詰まっている

物理指標を同時に見る。速度だけで採用しない。

## 最小compare summary

比較ツールは巨大dashboardにしない。まず1つのsummary JSONで足りる。

```json
{
  "case_id": "...",
  "mode": "comsol_faithful",
  "import": {},
  "preprocess": {},
  "first_step": {},
  "events": {},
  "ensemble": {},
  "runtime": {},
  "acceptance": {}
}
```

必要なCSVは小さくする。

```text
first_step_error.csv
force_contributions.csv
boundary_hit_comparison.csv
state_fraction_time_series.csv
```
