# M3-C3 Case-P three-current companion

## 結論

`PASS / COMPLETE`。Case-P source MPHを変更せず、100 nm、287粒子、30 ms、121出力時刻、
Brownian-offの代表caseについて、candidateとCOMSOLのaggregate three-current軌道を同じ
exact-connectivity common-P1場と共有初期状態で比較した。両者の3段階時間刻み自己収束、全時刻の
position/charge、両側active時のvelocity、lifecycle、141件のterminal eventがすべて登録gateを通過した。

本判定のauthorityは[`trajectory_evaluation.json`](trajectory_evaluation.json)と3段階の
`reference_dt_*_run_receipt.json`である。Case-Aの
10/30 nm relative-flow ion dragと100 nm image ion dragは
[`../../m3c1/case_a_size_ion_drag_companion_v1/`](../../m3c1/case_a_size_ion_drag_companion_v1/README.md)
が所有する。これにより、現モデルを使う優先1--4は完了した。

## 比較条件

- candidate: exponential midpoint、`dt=2.5/1.25/0.625 us`
- COMSOL: classical RK4、`dt=5/2.5/1.25 us`
- charge: `aggregate_relative_drift_regularized_three_current_v1`
- ion drag: `relative_flow_screened_collection_orbital_aggregate_ion_v1`
- drag: `epstein_linear_effective_gas_sensitivity_v1`
- thermophoresis、DEP、R-Z free-molecular lift、gravity、Freeze/Disappearを有効化
- 全6 runの終端人口: `stuck=141`、`active=146`

初期電荷は一つの共有三電流平衡artifactから設定した。source MPHのSHA-256は
`3bbf08e3469758313eac5de473a7a0dd4cc9a6f72c9722393229f0b856e9b524`で、各COMSOL runの
前後で不変である。負イオン5 primitiveのauthorityは
[`../caseP_negative_ion_primitives_v1/`](../caseP_negative_ion_primitives_v1/README.md)である。

## 数値結果

candidateのfine-pair relative L2はposition `2.848e-6`、velocity `9.743e-6`、charge
`4.168e-6`、観測RMS収束次数はそれぞれ`2.133`、`2.156`、`1.950`だった。COMSOLは
`5.132e-6`、`1.213e-5`、`6.147e-6`、収束次数は`2.103`、`2.413`、`2.407`だった。

最細解同士の全時刻比較は次のとおりである。

| 量 | RMS | maximum | relative L2 |
|---|---:|---:|---:|
| position [m] | `1.098e-7` | `1.121e-6` | `1.571e-6` |
| velocity [m/s], common-active states | `3.514e-4` | `2.596e-3` | `4.959e-6` |
| charge [e] | `1.507e-3` | `2.446e-2` | `2.254e-6` |

terminal particle/outcome/semanticは141件すべて一致した。event time差はRMS `4.702e-9 s`、
maximum `2.753e-8 s`で、candidate/COMSOL双方の刻み収束から事前定義した不確かさ内である。

## 抗力入力の修正

最初のCOMSOL companionは組込みEpstein dragの値欄だけをP1へ変更したが、速度と圧力のsource selectorが
native場を指したままで、P1密度・温度と混在していた。同一状態RHS診断でこのowner混在を特定した。
最終runは組込みdragを無効化し、candidateと同じP1速度・密度・温度、同じ分子質量、
`delta=1+0.9*pi/8`を使う明示custom force一つへ置換した。係数fit、gate緩和、solver coreの変更はない。

COMSOL 1-processの観測wall timeは5/2.5/1.25 usで`179.41/307.13/569.77 s`、peak RSSは
約`3.32--3.38 GB`だった。これは外部reference生成の記録であり、candidateとのequal-accuracy速度比較ではない。

## 主張しないこと

本結果は登録済みcommon-P1、Brownian-off、100 nm代表caseの同一model-form時間積分agreementだけを支持する。
native COMSOL FE一般同等性、Brownian pathwise一致、species-resolved負イオン、任意形状・任意条件、
物理近似自体の妥当性、普遍的COMSOL同等性、COMSOLより高速という主張には使わない。元Case-P二電流anchorも
変更しない。grazing/corner/multiple-hitとnative FE一般性は、識別可能な別モデルで評価する次工程である。
