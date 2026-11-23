# 独立衝突参照の実行

`regimes.csv`は検証用に明示した二つのMaxwell bath条件であり、装置測定値ではない。
実行処理は`tools.vv.gas_collision.qualify`、独立参照は`collision_reference.py`の衝突積分とjump sampler、
候補平均則は現行のfinite-speed Epstein。入力・参照・評価・出力を同じ役割とは扱わない。

```console
uv run --locked python -m tools.vv.gas_collision.qualify --regimes evidence/retirement_2026_11_23/gas_collision/regimes.csv --output NEW_OUTPUT --gas-orders 32,64,96 --jump-labels finite_slip --jump-count 200000 --durations-s 2.5e-7,1e-6 --interval-count 10000
uv run --locked python -m pytest tools/vv/gas_collision/tests -q
```

実行済み。2条件の積分、有限速度条件の20万jumpと1万intervalを計算し、[数値出力](result/qualification.json)と
[比較表](result/moment_comparison.csv)を生成した。関連7 testsは成功した。
有限速度の平均力の正規化差は`9.0649e-14`、32/64/96求積の最後の平均力変化は`6.1626e-14`。
zero-slipの二方向の拡散は線形FDTへ丸めの範囲で一致した。

有限速度ではparallel/transverseの拡散比は約12.59であり、線形FDTで全速度域を置換できるという認定ではない。
frozen bathの衝突momentだけが対象で、非線形輸送、meridionalからの3D運動、壁first-arrival、付着量の認定は行わない。
