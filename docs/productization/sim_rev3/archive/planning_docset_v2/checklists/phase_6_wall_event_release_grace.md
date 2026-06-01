# Phase 6 Checklist: Release Grace / Wall Event Simplification

## 目的

same-source skipをrelease直後の短いgraceに限定する。

## 実装

- release grace predicate
- blocked counter
- inward reimpact normal handling
- unrelated wall unaffected
- hit-time state remains source of wall action

## Acceptance

- outward same-source inside grace skipped
- outward same-source outside grace blocked/handled
- inward same-source reimpact handled
- no VIGUS-specific wall id
