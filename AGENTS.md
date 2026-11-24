# Clean-room workspace guidance

This workspace maintains the independent Chamber Particles solver. Its current
scope and change rules are owned by `particle_platform_redesign/`; the former
implementation remains a read-only reference, not an implementation base.

## Workspace boundaries

- Read [particle_platform_redesign/AGENTS.md](particle_platform_redesign/AGENTS.md)
  before changing that project. Read the separate `report_app/AGENTS.md` before
  changing the visible report.
- New production code belongs only in the independent uv project at
  `particle_platform_redesign/solver/`. Its numerical core and installed clients
  have the dependency boundaries defined in the project architecture.
- `model_dataset/` is reference input and external V&V material. It is not a
  solver dependency or a golden definition of the physics.
- `old_code/` is a read-only archive. Do not activate its environment, import its
  package, copy its implementation, or modify it to implement the current solver.
- Do not restore an archived root project, lockfile, package, test tree or tool
  configuration. Use the solver's locked uv project.

Old code may be inspected for failure cases, candidate equations and input
provenance. Re-derive and independently verify anything adopted. Native
observations, independent scientific references, external inputs and execution
environments are distinct from disposable software-generated outputs.

Keep one production engine and one owner for each fact. COMSOL integration and
comparisons remain external to the core. Replace obsolete callers, settings,
tests and documentation in the same change; do not retain compatibility readers
or aliases. Do not add general plugin/DI frameworks, duplicated validators,
broad contract suites or diagnostic subsystems.

The [project document guide](particle_platform_redesign/README.md) routes users
and maintainers. Module ownership lives in the architecture; the project AGENTS
contains stable invariants, and the quality document owns the change/check workflow.

## Project skills

- Use `$inspect-comsol-model` first for an unfamiliar MPH or incomplete exports,
  to bind active equations, selections, saved solutions, units, and identities
  to native evidence. Reuse its extraction record across agents and environments.
- Use `$prepare-comsol-field-case` to turn inspected COMSOL geometry, fields,
  and requested or extracted particle settings into a checked solver case with
  explicit model correspondence and an independent numerical plan.
- Use `$validate-comsol-trajectories` for trajectory/outcome comparison with
  COMSOL, practical agreement of saved or analogous results, and cause
  attribution. Keep these conclusions separate from qualified solver parity.
- Use `$visualize-particle-trajectories` for geometry-overlaid trajectories,
  animations, population views, and COMSOL comparison figures.

The skill files live under `.agents/skills/`. Other coding agents can read the
matching `SKILL.md` directly when skill invocation is unavailable. Each skill
can finish independently in the requested scope: discovery, checked input,
smoke, saved-data characterization, converged comparison, or figures. Infer
that scope from the request rather than asking the user to select a mode, and
reuse verified records instead of repeating every stage.

For an unfamiliar complete comparison, discovery informs preparation's model
mapping and targeted export specification; inspection obtains those exports,
preparation creates/runs the case, and validation evaluates it. Visualize saved
evidence when it answers the question. Read supporting references only for the
selected scope. Existing case tags, chamber dimensions, boundary IDs, seeds,
and step sizes are not new-case defaults.

## Maintaining project skills

Before creating or improving a skill, review the existing project skills and
the affected references, repository authorities, and current tool contracts.
Check the change against the whole workflow for conflicting instructions,
overlapping responsibilities, inconsistent terminology or verdicts, and missing
handoffs. Do not assess the edited file in isolation.

- Give each skill a clear purpose, invocation scope, required inputs, useful
  outputs, completion criteria, and boundary with neighboring skills. Keep one
  owner for each fact or decision; link to that owner instead of duplicating
  definitions or introducing competing procedures.
- Resolve contradictions at the owning source. Update affected callers,
  references and UI descriptions in the same change, and replace or remove
  obsolete instructions. Do not append an exception while leaving conflicting
  requirements active.
- Check that the combined instructions reflect actual solver/tool capabilities
  and distinguish evidence acquisition, physical assumptions, input preparation,
  numerical qualification, practical comparison, cause attribution and proposed
  improvements. Unknown evidence must not become a default or an unsupported
  claim; a limited task must not inherit unrelated full-workflow requirements.
- Keep guidance useful across models and environments. Case-specific values,
  tags, IDs, thresholds and historical outcomes are examples or bound evidence,
  not general defaults. Preserve the user's requested scope and reuse verified
  evidence rather than requiring repeated extraction or simulation.
- Validate skill metadata, routing and reference links, and run changed helpers
  when applicable. For substantive changes to scientific judgments, ownership
  or workflow, use a proportionate independent review or realistic forward test
  covering another model or task scope. Minor wording changes do not require a
  new simulation campaign.

Finish with a coherent, usable set of skills: explain which conflicts were
resolved, verify the affected handoffs, and retain explicit capability limits
or unresolved evidence. More instructions alone do not constitute an improvement.
