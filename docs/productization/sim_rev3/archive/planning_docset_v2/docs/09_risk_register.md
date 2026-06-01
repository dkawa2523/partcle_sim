# 09. Risk Register

## R1. 評価基盤なしに物理変更する

症状:

- 局所probeでは良いがglobal parityが悪化する。
- runtimeだけ改善して物理が壊れる。

対策:

- Phase 1でminimal compare railsを作る。
- endpoint-only acceptanceを禁止する。

## R2. VIGUS専用ロジックがmainlineへ入る

症状:

- wall id 104/105/147などがsolver coreに現れる。
- output path名で分岐する。

対策:

- VIGUSはlocal probe扱い。
- mainlineはmanifest, wall law, source provenanceで汎用化する。

## R3. 契約・診断・helperが増えすぎる

症状:

- 読むべきファイルが増える。
- テストdriftが増える。
- Codexがhelperを追えなくなる。

対策:

- validationはruntime build近くの一点に寄せる。
- diagnosticsはsummary/counterをdefaultにする。
- helperは重複を減らす時だけ追加する。

## R4. source preprocessing と COMSOL faithful が混ざる

症状:

- COMSOL releaseがsolver側で勝手にsnapされる。
- faithful比較がproduction補正に依存する。

対策:

- `comsol_faithful` ではimplicit snap禁止。
- `surface_release_production` だけprojection/offsetを許す。
- import diagnostic modeでズレを報告する。

## R5. first-step差分を見ない

症状:

- preprocessは良いが1 step後に崩れる。
- wall処理を触っても残差が消えない。

対策:

- first_step_error.csvを早期導入。
- force_contributions.csvを小さく出す。
- stochastic offのdeterministic comparisonを標準化する。

## R6. axisymmetricをcartesianとして扱う

症状:

- VIGUS/RZ系でgeometry, source sampling, axis wallがずれる。

対策:

- coordinate_systemをfirst-classにする。
- axisymmetric_rzの最低意味論をPhase 3で固定する。
