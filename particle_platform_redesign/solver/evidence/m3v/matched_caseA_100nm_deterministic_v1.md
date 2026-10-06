# M3-V deterministic matched trajectory assessment v1

## 結論

既存production solverは、下記のhash固定されたCase-A 100 nm・決定論・材料境界到達前の比較範囲で、
COMSOL 6.4の粒子追跡と同じ時間刻み精度を持ち、2.5 us履歴は事前登録した数値不確かさ幅の内側で一致した。
この限定範囲の判定は`PASS`である。

これはCOMSOL一般との同等性、native finite-element場との空間離散同等性、境界、動的電荷、ion drag、
thermophoresis、Brownian pathwise一致を意味しない。COMSOL/model datasetはproduction coreへ依存させず、
本評価は`tools/vv/comsol/`だけが所有する。

## 比較範囲

| 項目 | 固定条件 |
|---|---|
| dataset/model | Case A、100 nm、theory-consistent MPH |
| 座標 | axisymmetric RZ no-swirl |
| 粒子 | 287個、release ID・初期位置・初速度・質量・直径を一致 |
| 時間 | 0..0.4 ms、10 us間隔の41 frame |
| 積分刻み | 10、5、2.5 usの固定step |
| 電荷 | fixed charge number `-1` |
| 有効な力 | Coulomb electric、linear Epstein、gravity/buoyancy |
| Epstein | `delta = 1 + 0.9*pi/8`、80/20 CF4/O2 mixture mass |
| 無効化 | Brownian、dynamic charge、ion drag、thermophoresis、lift、DEP |
| 場 | 同一node値・同一三角形connectivityのcanonical P1 |
| 境界 | 比較窓内のevent 0件。境界精度は`NOT_TESTED` |

COMSOL側は監査対象MPHを`ModelUtil.loadCopy`で隔離copyとして開き、保存しなかった。source MPHの実行前後
SHA-256はともに
`3bbf08e3469758313eac5de473a7a0dd4cc9a6f72c9722393229f0b856e9b524`である。
canonical P1場はCOMSOL sectionwise形式により元の三角形connectivityごと渡した。Epsteinで使う圧力も
同じfield ownerから`p=rho*k_B*T/(Mmix/N_A)`として再構成し、native圧力との混在を除いた。

## 判定手順

1. candidateとCOMSOLを別々に10/5/2.5 usで自己収束評価した。
2. 両自己収束reportと初期状態hashだけから、cross差を見る前に許容幅を登録した。
3. 登録済みfine trajectory hashを照合し、2.5 usの全287粒子・全41 frameを比較した。
4. 同じCOMSOL状態上でcanonical field、各力、合加速度を独立に再評価し、最初の差分層を確認した。

許容幅はcandidateとCOMSOLの5 us対2.5 us変化量の和とfloat64 representation floorであり、cross結果に
合わせて調整していない。

## 自己収束

| 指標 | candidate 5→2.5 us | COMSOL 5→2.5 us | 比率 candidate/COMSOL |
|---|---:|---:|---:|
| position RMS | `1.1759338e-12 m` | `1.1759365e-12 m` | `0.9999977` |
| position max | `1.7266157e-11 m` | `1.7266300e-11 m` | `0.9999917` |
| velocity RMS | `1.0301996e-8 m/s` | `1.0301995e-8 m/s` | `1.0000001` |
| velocity max | `9.9382844e-8 m/s` | `9.9382845e-8 m/s` | `0.99999999` |

RMS観測次数はcandidate/COMSOLで、position `2.07440/2.07440`、velocity `2.04326/2.04326`である。
P1場の要素境界を通るため、古典RK4の滑らかな場に対する理想4次を主張しない。重要なのは、両者の
refinement変化量と観測次数が独立runで一致したことである。

## 場・力の一致

| 層 | relative L2 |
|---|---:|
| electric field | `7.55e-16` |
| gas velocity | `5.56e-16` |
| gas temperature | `1.17e-16` |
| gas density | `1.04e-16` |
| COMSOL electric force対独立式 | `1.67e-15` |
| COMSOL Epstein force対独立式 | `7.02e-16` |
| COMSOL gravity/buoyancy対独立式 | `1.04e-16` |
| COMSOL total acceleration対独立式 | `1.76e-15` |
| production runtime対production独立式 | `1.17e-16` |

## 全時間軌道比較

初期position/velocity差はbitwiseで0だった。全11,767 state rowの結果は次のとおり。

| 指標 | 観測差 | 事前登録幅 | 判定 |
|---|---:|---:|---|
| position RMS | `8.6427e-16 m` | `2.8066e-12 m` | PASS |
| position max | `2.1724e-15 m` | `3.4987e-11 m` | PASS |
| velocity RMS | `1.9988e-14 m/s` | `2.0605e-8 m/s` | PASS |
| velocity max | `1.4129e-13 m/s` | `1.9877e-7 m/s` | PASS |

cross差は位置で登録幅より約3～4桁、速度で約6桁小さい。したがって、比較対象の既存production modelは
このhash固定case・時間窓・共通物理において、COMSOLと同等の時間離散精度を持つと判断する。

## native-field比較から得た区分

最初にCOMSOL native finite-element場とcanonical P1場をそのまま比較したrunは不合格だった。native場に対する
canonical P1 electric field差はrelative L2 `2.903e-2`で、trajectory差はposition RMS `1.513e-6 m`、
velocity RMS `9.526e-3 m/s`だった。一方、同じ状態での力式parityは約`1e-16`だった。

これはintegratorや力式の不一致ではなく、異なる空間場表現を同一入力と扱った比較設計の問題である。この結果は
削除せず、native場再現をfield extraction/remeshの独立空間収束gateとして扱う根拠にする。production solverへ
COMSOL合わせ込み、補正係数、比較専用branchは追加しない。

## 証拠identity

下表は、現行external tool一式で隔離COMSOL実行を再実行した`caseA_100nm_canonical_p1_sectionwise_v4`と、
旧native場を仮定しない診断revision 2をcanonical evidenceとする。正規化された場・力・軌道は
sectionwise v2/v3/v4でbyte-identicalだった。

| artifact | SHA-256 |
|---|---|
| candidate fine trajectory | `05e1020db2161ada0139fc63f3941cc1060a728e9ca395150ca2ece0bb9e92fe` |
| COMSOL fine trajectory | `e5d4308bc63497d35ac38bb2999c5ddef1fd990b8d349c97bda0be0d24559ca8` |
| preregistered budget | `ec62c68dac86a57b7c7f0e603685afb3c19eed60d27c6f47ffa7c1e19dd67099` |
| trajectory comparison | `10810d3b796750e3f26e91d70583b67adb4e64b70c76b13c69743205b176ed00` |
| field/force diagnostic | `d08cc2a8337afd081a02b76647fefbf5ba555d5401298f7fab887d70c8970be9` |
| candidate self-convergence | `517be7f4ffff78a4a97b1d7aefca1596f6cebedb7a20ae284554cbf46856a5e6` |
| COMSOL self-convergence | `f636e585a82942ba8034977bd717ff9f375ac4d5ef5c83896f76644cf0fd3b87` |
| COMSOL provenance | `72033fe34a3a417cdb4b36964b848d47efe828f67421b53aa6e4e8a51d6e131d` |
| COMSOL artifact receipt | `3b33e2011978ddbe6baa2c1681c370b132f935de9e01705442f9986836e80f6a` |

比較開始時点のproduction `src/` 25 filesのpath+content manifest digestは
`50e091682973945fa6e285f9693b19f2789ef3a0a7700b586c3e2557ecf4e5fb`である。本比較ではproduction sourceを
変更していない。

## 非対象と次の外部gate

- positive material event、stick/escape/reflectionの時刻・facet・post-state比較
- native COMSOL場からcanonical fieldを作る際のmesh/spatial convergence
- dynamic charge、relative-drift charge、ion drag、thermophoresisの個別matched slice
- Brownianのmulti-seed ensemble統計とfirst-passage分布。単一seedのpathwise一致は正解判定に使わない
- 3-D、time-dependent field、moving geometry

これらは本判定を無効にしないが、未検証範囲をCOMSOL同等と表示してはならない。
