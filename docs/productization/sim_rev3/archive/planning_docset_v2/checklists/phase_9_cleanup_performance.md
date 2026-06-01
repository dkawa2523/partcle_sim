# Phase 9 Checklist: Cleanup / Simplification / Performance

## 目的

正しさが守られた後で、古い記述、邪魔なhelper、過剰診断を削り、速度を改善する。

## 実装

- Phase 0 auditに基づくcleanup
- always-on diagnostics削減
- hot path dispatch削減
- spatial index/cache improvements if safe

## Acceptance

- physics metrics unchanged or improved
- compare summary unchanged within tolerance
- runtime improved or code simpler
- deleted/quarantined理由が明確
