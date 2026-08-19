from __future__ import annotations

import nox

nox.options.error_on_missing_interpreters = True
nox.options.reuse_existing_virtualenvs = True


def _run(session: nox.Session, command: str) -> None:
    session.run("python", "-m", "quality_tools.runner", command, *session.posargs)


@nox.session(name="quality-fast", venv_backend="none")
def quality_fast(session: nox.Session) -> None:
    """Format/fix changed Python files, then run baseline-aware fast gates."""

    _run(session, "fast")


@nox.session(name="quality-pr", venv_backend="none")
def quality_pr(session: nox.Session) -> None:
    """Run every deterministic pull-request quality gate in check-only mode."""

    _run(session, "pr")


@nox.session(name="quality-nightly", venv_backend="none")
def quality_nightly(session: nox.Session) -> None:
    """Run PR gates plus stochastic, performance, E2E, and mutation checks."""

    _run(session, "nightly")


@nox.session(name="quality-baseline", venv_backend="none")
def quality_baseline(session: nox.Session) -> None:
    """Explicitly refresh quality baselines after refusing regressions."""

    _run(session, "baseline")
