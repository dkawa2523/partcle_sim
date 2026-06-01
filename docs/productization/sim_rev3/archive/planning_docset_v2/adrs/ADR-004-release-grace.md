# ADR-004: Same-Source Skip Is Release Grace, Not General Optimization

## Decision

same-source outward crossing skipは、release直後の短いgraceに限定する。

## Reason

広すぎるskipは物理的な壁条件をbypassし、runtime改善と物理悪化を同時に起こす可能性がある。

## Consequences

- inward reimpactはskipしない。
- grace外outward hitはblocked counterを増やし、通常wall eventへ戻す。
- runtimeだけで採用判断しない。
