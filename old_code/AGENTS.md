# Read-only archive guidance

Everything under `old_code/` is historical material from the implementation
that preceded the clean-room solver.

- Do not edit, repair, extend, install, import, test, or run this code.
- Do not activate an archived environment or use its `pyproject.toml`,
  `uv.lock`, commands, CI configuration, or quality tooling.
- Do not copy implementation code into the active solver.
- Historical equations, inputs, and failure cases may be inspected only as
  candidates; re-derive and independently verify anything adopted.

Active production work belongs in `../particle_platform_redesign/solver/` and
is governed by `../AGENTS.md` and
`../particle_platform_redesign/AGENTS.md`. Git history is the authority for
the original form of archived files.
