# Phase 2 Checklist: Mode Separation + Minimal Manifest Gate

## 目的

COMSOL faithful comparisonとproduction surface releaseを混ぜない。

## 実装

- mode parse
- central mode validation
- minimal manifest loader or validator
- faithful rejects implicit correction
- production allows explicit boundary_release preprocessing

## faithful rejects

- missing coordinate scale
- missing coordinate system
- missing release table
- missing wall law
- missing force inventory
- source preprocessing enabled
- implicit snap

## Acceptance

- unknown mode fails
- faithful + boundary_release fails
- production + boundary_release passes
- normal existing run not broken
