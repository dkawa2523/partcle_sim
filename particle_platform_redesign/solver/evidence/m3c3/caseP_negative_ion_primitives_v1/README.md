# M3-C3 Case-P negative-ion primitive authority

## 結論

`PASS`。hash固定したCase-P source MPHから、aggregate three-current modelに必要な負イオンprimitiveを
common-P1の全1987節点へ作成した。これは外部V&V用のproducer companionであり、solver coreやsource MPHを
変更しない。三電流軌道のCOMSOL同等性は、このartifact単独では認定しない。

生成物は
`solver/_out_m3c3/caseP_negative_ion_primitives_v3/candidate_input_three_current.h5`
（SHA-256 `f14efc8bd453505baaa9fc1ddf71f91e5145fb07cb8bdeecd1955d40cec8d124`）である。

## primitive

- `negative_ion_number_density = n_F- + n_O-`
- `negative_ion_velocity = sum(n_s*(u_gas+V_diff,s))/sum(n_s)`、components `(r,z)`
- `effective_negative_ion_mass = sum(m_s*n_s)/sum(n_s)`
- `negative_ion_thermal_voltage = k_B*T_g/e`

F-は`0.019 kg/mol`、O-は`0.016 kg/mol`で、両speciesはsource modelの
`UseGasForIonTemperature`設定に従う。速度はnumber fluxから構成し、gas convectionを二重加算しない。

## one-sided boundary owner

in-memory copy上に4つのdomain cache PDEと4つのboundary cache PDEを作成した。boundary cacheはdomain 3に
隣接する16境界上で`side(3, raw primitive)`をsourceとする。新規stationary studyはcache physicsだけを
activateし、`createAutoSequences("all")`で新しいsolutionを作成した。interior nodeはdomain cache、boundary
nodeはboundary cacheが所有する。座標nudge、欠損補完、別domain fallbackはない。

COMSOL geometry座標の単位は`cm`で、canonical座標の単位は`m`である。正規化時に明示的に`0.01`倍し、
全節点の最大一致差は`2.7755575615628914e-17 m`だった。

RZ canonical fieldはaxis上のradial成分を厳密な`+0.0`とするため、34 axis nodeへ
`u_r(r=0)=+0.0`を一度だけ適用した。これは座標移動や欠損値補完ではなく、axisymmetric vectorの正則性射影で
ある。射影前の最大絶対値は`1919.783603268964 m/s`、射影後は`0 m/s`である。

## 再現条件

- COMSOL `6.4.0.429`
- source MPH SHA-256:
  `3bbf08e3469758313eac5de473a7a0dd4cc9a6f72c9722393229f0b856e9b524`
- input common-P1 H5 SHA-256:
  `c54b4b658213230e82307ca89018538c0240bfcc83e793206d5093f4f08908e9`
- `ModelUtil.loadCopy`、`comsolbatch -nosave -np 1`
- source MPHの実行前後hashは同一

実行入口は`tools/vv/comsol/run_m3c3_caseP_negative_ion_primitives.ps1`である。詳細な式、単位、field range、
artifact hashは`primitive_receipt.json`をauthorityとする。

