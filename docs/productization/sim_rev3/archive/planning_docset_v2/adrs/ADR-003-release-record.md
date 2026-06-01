# ADR-003: Release Provenance Is First-Class

## Decision

releaseは初期座標だけでなく、source feature, boundary/part, projection, capture, offsetを持つ。

## Reason

VIGUSではCOMSOL wall-origin releaseの解釈が初期差分の主要因だった。release provenanceを失うと、same-source skipやwall eventで誤判定が起こる。

## Consequences

- capture toleranceとinward offsetを分離する。
- faithfulとproductionで処理を分ける。
- source id unknownを明示的に扱う。
