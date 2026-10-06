# chamber-particles 0.1.0

`0.1.0`は、半導体製造チャンバー内の粒子軌道を外部場とgeometryから計算する
clean-room solverの最初の正式公開版です。公開状態はalphaであり、対応範囲を明示して使います。

## 公開範囲

- Python 3.12、Windows/Linuxで検証したpure-Python wheel
- `load_case`、`simulate`、`open_result`の三つの公開Python APIと同じ経路を使うCLI
- static 2-D XYおよびRZ meridionalのcanonical field/geometry入力
- single-thread compiled CPU engine、bounded slab、再現可能なcounter RNG、checkpoint/resume
- P1/Q1/regular field、first-hit境界event、stick/escape/hold/specular/probabilistic wall
- 明示選択するdrag、electric、gravity、continuous charge、thermophoresis、ion drag、DEP、
  RZ lift sensitivity、2-D Brownian/OUの受理済みrevision
- source checkoutに付属するResultViewベースのanalysis/visualizationと、COMSOL非依存のQuick Start

wheelが配布するのは`chamber_particles` packageとCLIです。`tools/`、`examples/`、外部V&Vはsource checkoutの
companionでありwheelには含めません。

正確な対応組合せ、適用域、schema、失敗時の扱いは
[`support_and_errors.md`](https://github.com/dkawa2523/partcle_sim/blob/v0.1.0/particle_platform_redesign/solver/docs/support_and_errors.md)
を権威とします。各式とrevisionは
[`physics_models.md`](https://github.com/dkawa2523/partcle_sim/blob/v0.1.0/particle_platform_redesign/solver/docs/physics_models.md)、数値法は
[`numerics.md`](https://github.com/dkawa2523/partcle_sim/blob/v0.1.0/particle_platform_redesign/solver/docs/numerics.md)にあります。

## 検証と制約

正式tagは、同一commitについてWindows/Linuxの品質gate、verification/scenario、producer-neutral
Quick Start、warm performance smoke、wheel build、runtime-only clean install、三公開API smokeがすべて
成功した後だけGitHub Releaseになります。Releaseにはwheelと`SHA256SUMS.txt`を添付します。

外部COMSOL V&Vでは、明示したcommon-field/common-physicsの2-D範囲を
`2D_CRITICAL_VV_COMPLETE` / `CLOSED_ACCEPTED_WITH_LIMITATIONS`として閉じました。これはnative COMSOL FE場、
任意形状・任意条件、全物理組合せ、pathwise Brownian、普遍的なCOMSOL同等性を認定するものではありません。
また、COMSOLより高速というportableな性能保証、solver内thread/GPU speedup、3-D粒子、時間依存場、
自己無撞着plasma計算は`0.1.0`の主張に含めません。

本repositoryには現時点でOSSライセンスを設定していません。この公開は再利用許諾を意味せず、再利用条件は
権利者の明示許諾が必要です。PyPIとsdistへの公開も`0.1.0`の配布範囲外です。
