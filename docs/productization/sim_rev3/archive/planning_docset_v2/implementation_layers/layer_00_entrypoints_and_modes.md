# Layer 00: Entrypoints and Modes

## 役割

run configを読み、実行モードを決める。

## 必須モード

```text
comsol_faithful
surface_release_production
normal / default existing mode
```

## comsol_faithful

目的:

- COMSOLとの機械比較。
- solver側の暗黙補正を禁止。
- 不明な座標系、scale、force inventory、wall lawはfail。

禁止:

- implicit source generation
- implicit boundary snap
- implicit wall law default
- field support rescue as normal behavior
- VIGUS-specific defaults

## surface_release_production

目的:

- 装置パーツ表面からの粒子発生源解析。
- boundary release preprocessingを許す。
- capture toleranceとinward offsetを明示的に使う。

## 実装の最小方針

- runtime/config loading付近に1箇所だけvalidationを置く。
- solver hot loopにmode分岐を増やさない。
- 既存normal runは壊さない。
