# ADR-002: COMSOL Faithful and Surface-Release Production Are Separate Modes

## Decision

`comsol_faithful` と `surface_release_production` を分離する。

## Reason

COMSOL比較ではsolver側のsnapやimplicit correctionを入れると、COMSOLとの差分原因が隠れる。一方、実装置パーツ表面release解析ではboundary classificationとinward offsetが必要である。

## Consequences

- faithfulではsource preprocessing禁止。
- productionでは明示設定時のみboundary release preprocessingを許可。
- import diagnosticはズレを報告するが、勝手に修正しない。
