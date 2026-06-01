# 01. VIGUS Lessons Generalized

VIGUSレポートは、VIGUSに合わせた局所パッチの手順書として読むべきではない。むしろ、境界放出型粒子追跡コードで起きやすい失敗モードの実例として読む。

## 重要な教訓

### 1. 初期差分は single bug ではない

初期の支配要因は、same-source skip単独でも、specular/stick条件単独でもなかった。

主な初期要因:

- generated case が COMSOL faithful ではなかった。
- geometry axis と field axis がずれ得た。
- raw COMSOL release がsolver側でsurface releaseとして解釈されていなかった。
- boundary capture tolerance が狭すぎた。
- force/gas/charge設定がCOMSOL比較用に揃っていなかった。
- wall lawがCOMSOL条件と一致していなかった。

したがって、最初にやるべきことはsolver coreの物理式変更ではなく、import、case build、初期条件解釈、field/force parityの確定である。

### 2. capture tolerance と inward offset を混ぜない

VIGUSでは raw COMSOL release がoff-gridに見えるため、単純なon-boundary判定では放出面由来と認識できないケースがあった。

分けるべき概念:

```text
capture_tolerance_m:
  この粒子はどのboundary/part由来かを分類するための距離。
  mesh spacingやfield grid spacingに応じて広めに許容する。

inward_offset_m:
  数値的な壁上初期点を避けるためにdomain内へ入れる距離。
  物理初期位置を壊さないよう小さく保つ。
```

captureを広げることと、粒子を大きく内側へ移動することは違う。

### 3. first-step を独立断面にする

VIGUSでは、preprocess段階では比較的良いが、first-step後に速度比が崩れていた。これは、差分がrelease座標だけでなく、release直後の力評価、drag、field sampling、integration、near-wall eventに移っていることを示す。

そのため、比較断面は最低限次を持つ。

```text
post-import
post-preprocess
post-first-step
first-wall-event
ensemble-time-series
```

### 4. local probe と mainline fix を分ける

wall104のような局所壁条件補正は、原因切り分けには有効だった。しかし、それをglobal mainlineへ入れると過補正になる可能性がある。

local probeの役割:

- 現象を見つける。
- 主因候補を切り分ける。
- VIGUSの特定挙動を説明する。

mainline fixの条件:

- 複数source / 複数wall / 時系列分布で改善する。
- runtimeだけでなく物理指標が改善する。
- VIGUS固有IDなしで説明できる。

### 5. 高速化分岐は物理仮説である

same-source skipは単なる最適化ではなかった。物理境界条件をbypassする条件でもあった。

高速化分岐を入れるときは、次を同時に見る。

- runtime
- wall interaction count
- stuck count
- active/free-flight count
- first-crossing distribution
- blocked diagnostics
- source別分布

runtimeだけで採用してはいけない。

### 6. sampled と full を役割分担する

sampled run は iteration speed のために必要。full run は真値寄り検証のために必要。

方針:

```text
small synthetic:
  unit / verification

sampled COMSOL:
  fast iteration / local diagnosis

full COMSOL release:
  validation / release前確認

production run:
  speed and robustness
```

## VIGUSから抽出した汎用診断順

1. import境界を固定する。
2. release capture と inward offset を分ける。
3. post-import / post-preprocess / post-first-step を見る。
4. local probe と mainline fix を分ける。
5. 高速化分岐を物理仮説として扱う。
6. sampled と full の両方で閉ループを作る。
