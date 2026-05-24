# Layer 04: Force and First-Step

## 目的

release直後に差が出る原因を、wall eventに入る前に切り分ける。

## Force breakdown

可能なら次の寄与を分ける。

```text
drag
electric
thermophoretic
dielectrophoretic
brownian
ion_drag
lift
external
total
```

## first-step compare

比較項目:

```text
particle_id
source_part_id
x0, v0
x1, v1
dx_error
dv_error
speed_ratio
force_total
force_components
field_support_status
```

## deterministic comparison

- Brownian off。
- stochastic seed固定。
- charge fixedまたはexplicit。
- particle properties fixed。

## 避けること

- VIGUSの結果へforce係数をチューニングする。
- wall eventを先に触ってfirst-step差分を隠す。
- 常時force contributionsを全粒子保存する。
