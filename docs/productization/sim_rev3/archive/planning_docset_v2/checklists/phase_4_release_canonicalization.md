# Phase 4 Checklist: Release Canonicalization

## 目的

release provenanceを保持し、capture toleranceとinward offsetを分離する。

## 実装

- ReleaseRecord or equivalent small structure
- projection distance reporting
- faithful no snap
- production explicit snap/projection
- source id unknown handling

## Acceptance

- faithful preserves raw coordinates or fails
- production classifies near-boundary release
- capture and offset independently configurable
- projection distance appears in summary/preflight
