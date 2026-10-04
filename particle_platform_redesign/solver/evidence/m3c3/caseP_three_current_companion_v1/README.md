# M3-C3 Case-P three-current companion closeout

## 結論

`BLOCKED / NOT_EVALUATED`。これは物理modelの`NOT_APPLICABLE`判定ではない。元Case-Pの
`auxq.R`は電子・正イオンの二電流式であり、負イオン密度`nm_d`を電流へ含めない。このanchorは変更せず、
M3-C3は負イオンを追加する独立した感度評価companionとしてのみ定義した。

このstatusは本artifactの三電流外部coverageだけに適用する。元二電流anchor、完了済みproduction revision、critical
boundaryを未完了へ戻さない。P21と明示scopeの2D benchmarkの終了判断は`implementation_plan.md`と
`vv_methodology.md`が所有し、このcoverageは解除入力を受領した場合だけ独立work packageとして開く。

COMSOL sourceから共通P1全1987節点の負イオンprimitiveを有限値で取得できなかったため、companion H5、
three-current平衡初期電荷、COMSOL/candidate軌道はいずれも生成・実行していない。旧Case-P trajectoryを
three-current結果として流用してはならない。

既存datasetが明示的に保持する負イオン量はtotal density `pcnm`だけであり、aggregate total velocity、mass、
thermal voltageのcanonical authorityはない。

## 確認できたsource意味論

- source MPH SHA-256: `3bbf08e3469758313eac5de473a7a0dd4cc9a6f72c9722393229f0b856e9b524`
- COMSOL: `6.4.0.429`
- F-: `z=-1`, `M=0.019 kg/mol`; O-: `z=-1`, `M=0.016 kg/mol`
- 両speciesとも`UseGasForIonTemperature`。したがって`V_T-=k_B T_g/e`。
- plasma transportはconvection、migration、mixture-diffusion correctionが有効。
- Equation Viewの`Vdr_*`/`Vdz_*`は`j/(rho*w)`で定義されたdiffusion velocityである。aggregateに使う
  total species velocityは`(u+Vdr, w+Vdz)`であり、ガス速度を二重加算しない。
- 内部点`(r,z)=(0.14,0.023) m`のread-only probeは
  `n-=1.0946545811434909e11 1/m^3`, `u-r=741.0886584096645 m/s`,
  `u-z=512003.53016202105 m/s`, `m-=2.907768130315183e-26 kg`,
  `V_T-=0.02596432649399907 V`を返した。これは式の評価可能性確認であり、P1場authorityではない。

## companion契約

負イオンはF-とO-を一つへ畳み込むsingle-aggregate approximationであり、species-resolvedな物理認定ではない。

- `n-=n_F+n_O`
- `u-=sum(n_s u_s)/sum(n_s)`（`u_s`は上記total species velocity）
- `m-=sum(m_s n_s)/sum(n_s)`
- `V_T-=k_B T_g/e`
- `n-=0`時の速度・質量fallbackは有限な式評価のためだけで、物理状態を捏造しない。
- screening lengthは既存の明示common-P1 fieldを両solverで共有し、負イオン寄与から再計算しない。
- 全release点の`Z0`は同一three-current式の単調rootとして再計算し、両solverへ同じ値を与える。
- `maximum_relative_ion_speed_m_s`はfit parameterではなく、正負ion node field extremaと明示したparticle-speed
  envelopeから作る適用証明値とする。内部点だけで約`5.12e5 m/s`であるため、既存`30000 m/s`を流用できない。

## export blocker

hash-lockした既存common-P1 H5は1987 nodes / 3779 trianglesである。以下の二つの最小経路をfail-closedで確認した。

1. selectionを付けない`CutPoint2D(data=dset1, 1987 exact coordinates)` + `EvalPoint`は座標の行数・順序確認を
   通過したが、最初の`n-`評価がcanonical node 11で非有限になった。node 11は
   `(r,z)=(0.0354181,0.022) m`、source external vertex ID 101の境界nodeである。
2. 一つの`Interp`へ5式と`double[2][1987]`を渡しdomain 3を明示した経路も
   `Undefined post expression - Feature: Interpolation`で停止した。
3. domain-3 `Eval`のprovider節点を取得し、locked `1e-14 m`でcanonical節点との一意な全単射を試みたが、
   canonical node 0で一意な対応を作れなかった。この試行は既存Python matcherそのものではなく、同じtoleranceを
   用いた探索なので、Python matcher合格を否定するclaimには使わない。

座標nudge、別domain fallback、欠損補間、近傍値代入はprimitiveのownerを変えるため採用しない。解除には、producer側で
one-sided domain-3境界値を定義した同じ1987座標順の5 primitive export、または同内容のhash-lock済みcanonical field
fileが必要である。各rowには有限な`n-`, total `u-r/u-z`, `m-`, `V_T-`を含め、source hash、座標hash、単位、
axis radial-velocity規則を記録する。

## 解除後の一回だけの比較

- 100 nm、287 particles、30 ms、121 common output times
- Brownian off、全決定論力、three-current、同一three-current `Z0`
- COMSOL RK4 `dt=10 us`を1回
- candidate `dt=10 us`と`5 us`
- candidate 10/5 us自己差を数値幅とし、全時刻の`r,z,v,Z`とevent/fateを判定

COMSOL rerunは必須である。primitive exportとcompanion準備後の目安は、COMSOLが約85秒、candidate 2本が
各約46--55秒、比較と検査を含め合計3--5分。追加seed/sweepは不要。
