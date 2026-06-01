# Implementation Guardrails

## Acceptable changes

- Small dataclass for release provenance.
- One central mode validation point.
- Small compare summary writer.
- Small synthetic tests.
- Focused cleanup of dead exploratory branches.

## Suspicious changes

- New package with many abstract base classes.
- Large contract system.
- Many tests for one helper.
- Deep diagnostic CSV as default output.
- Hardcoded VIGUS part IDs.
- Catch-all tolerances with no summary reporting.
- Hidden fallback that silently changes COMSOL faithful inputs.

## Required end-of-phase summary

Each Codex task should end with:

```text
Changed files:
Tests run:
Artifacts produced:
Acceptance criteria:
Known deferrals:
```

## Stop conditions

Stop implementation and ask for review if:

- a required source/reference file is missing;
- a test failure indicates unclear physics behavior;
- a change would require VIGUS-specific logic;
- the change expands diagnostics or contracts beyond the phase scope;
- compare rails are not yet available but physics tuning is requested.
