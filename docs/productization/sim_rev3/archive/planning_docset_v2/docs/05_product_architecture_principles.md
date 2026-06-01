# 05. Product Architecture Principles

## 目標アーキテクチャ

```text
external/
  COMSOL API/export scripts only

tools/
  case builders, compare runners, sharded runner, report utilities

particle_tracer_unified/
  io/
  core/
  providers/
  sources/
  forces/
  integrators/
  events/
  solvers/
  compare/
```

## レイヤ責務

### io/

YAML/CSV/NPZを読み、runtimeに必要な正規化済み入力を作る。COMSOL APIは入れない。

### core/

座標系、粒子state、境界primitive、field support、基本diagnostics。

### providers/

geometry / field provider。solver loop内にformat-specific処理を入れない。

### sources/

release table、release canonicalization、surface release preprocessing。

### forces/

drag、electric、thermophoresis、Brownian、external force、charge model。force contributionを比較できるようにする。

### integrators/

ETD、drag relaxation、reference integrator。collision replayと同じsegment logicを使う。

### events/

wall hit detection、release grace、wall policy、persistent contact。

### compare/

field、force、first-step、boundary、ensemble、runtime比較。

## 製品コードの判断基準

良い変更:

- 責務が1つ。
- 読むファイルが少なくなる。
- compareで効果が分かる。
- VIGUS以外にも使える。
- 通常runでは診断が軽い。

悪い変更:

- VIGUS wall id を直接見る。
- 出力ディレクトリ名で挙動が変わる。
- helperが実装より難しい。
- testが実装詳細に密着しすぎる。
- loggingが常時大量。
- contractを複数箇所で重複チェックする。
