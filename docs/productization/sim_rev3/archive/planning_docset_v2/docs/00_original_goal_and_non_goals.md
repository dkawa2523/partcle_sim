# 00. Original Goal and Non-Goals

## 目的

`sim_rev3` は、COMSOL や他シミュレーションから得た力場の空間分布、パーツジオメトリ、材料・壁条件・source設定を受け取り、任意の粒子初期条件から高速に軌道計算するエンジンである。

主用途は、半導体製造装置内のパーツ表面から発生するパーティクルの要因解析である。COMSOL Particle Tracing Module よりも高速に、多数粒子・複数条件・異なる境界物理を試せることを目指す。

ただし、速さだけを優先して境界物理やrelease意味論を壊してはいけない。最終的な狙いは次である。

```text
COMSOLと比較できる信頼性
+ 装置解析に使える柔軟性
+ 多数粒子を回せる速度
+ 第三者が保守できる製品コード
```

## 解くべき本質課題

元の課題は「COMSOLと値が合わない」だけではない。より本質的には、以下が絡んでいた。

1. COMSOL結果と自コード結果の比較軸が不十分だった。
2. 粒子飛散の時間変化を評価できていなかった。
3. 局所probeだけで良否判定し、global傾向を悪化させる修正が入り得た。
4. release、boundary、force、event、runtime が同時に変わり、原因特定が難しくなった。
5. Codexによる追加修正でhelper、診断、契約、testが増え、コードが肥大化した。
6. 速度問題の一部はアルゴリズムの遅さではなく、物理的に誤った境界skipやevent処理から生じていた。

## Non-Goals

このプロジェクトでやらないことを明確にする。

- COMSOLの完全クローンを作ること。
- `.mph` ファイルをsolver coreで直接読むこと。
- VIGUS専用のwall id、source id、geometry名をsolver coreに埋め込むこと。
- すべての中間値を常時詳細ログに出すこと。
- 巨大なcontract frameworkを作ること。
- テストを大量に増やして安心すること。
- Brownianなどの確率過程を含む1粒子軌道を逐点一致させること。
- 一回限りの出力ディレクトリや `_tmp_*` を製品baselineにすること。

## 重要な設計方針

最初に評価基盤を固定する。その後に実装を変える。

```text
評価基盤なしに修正する = 以前と同じ失敗を繰り返す
```

したがって、フェーズ順は次のようにする。

```text
repo audit
focused baseline stabilization
minimal compare rails
mode separation
import/coordinate semantics
release semantics
force/first-step parity
boundary event/release grace
axisymmetric completion
V&V productization
cleanup/performance
```
