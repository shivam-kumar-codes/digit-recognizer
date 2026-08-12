# CLAUDE.md

Persistent context for Claude Code in this repository. Read this at the
start of every session.

## Project goal
"Digit Recognizer — Production Edition." A digit-drawing web app backed by
a neural network, built in deliberate phases as both an ML-fundamentals
learning project (Michael Nielsen's *Neural Networks and Deep Learning*)
and a Claude Code workflow learning project. See `PROJECT_SPEC.md` for the
full spec and v1 scope boundary.

## Roadmap & current phase status
1. **[NOT STARTED]** From-scratch numpy neural network (Nielsen-style),
   trained on MNIST
2. **[NOT STARTED]** Ship it: FastAPI backend + canvas frontend, deployed
   publicly
3. **[NOT STARTED]** Harden it: tests, CI/CD, Docker
4. **[NOT STARTED]** Reimplement in PyTorch, compare v1 vs v2
5. **[NOT STARTED]** MLOps: W&B experiment tracking, model versioning,
   monitoring
6. **[NOT STARTED]** Polish: README, demo, resume write-up

**Right now:** only the foundation docs (`PROJECT_SPEC.md`, this file)
exist. No model, backend, or frontend code has been written. Do not start
Phase 1 code until the user explicitly asks for it.

> Update the status markers above as phases start/finish. Keep this
> section current — it's the fastest way for a future session to know
> where things stand.

## Tech stack decisions
Decided so far:
- **Python dependency/env management:** `uv` (venv + deps + lockfile)
- **Phase 1 model:** pure numpy, no ML framework (the point is to
  implement backprop etc. by hand, Nielsen-style)
- **Phase 2 backend:** FastAPI
- **Phase 2 frontend:** plain HTML/JS canvas (no framework, unless a
  need emerges)
- **Phase 4 framework:** PyTorch, for the v1-vs-v2 comparison

Undecided — resolve when that phase actually starts, and record the
decision (with a one-line rationale) here once made:
- Deployment target/host for Phase 2
- Testing framework for Phase 3 (likely pytest, not yet confirmed)
- Containerization approach for Phase 3
- CI provider for Phase 3
- Monitoring approach for Phase 5

## Coding conventions
- **Phase 1 (numpy NN) only:** comments should be learning-heavy —
  explain the *why* and the math intuition (e.g. why the gradient looks
  like this, what a given matrix shape represents) liberally, since the
  explicit point of this phase is learning from first principles, not
  writing minimal production code.
- **All other phases (FastAPI, frontend, PyTorch, tests, infra):** standard
  terse style — comment only non-obvious WHY, not WHAT. Well-named code
  over explanatory comments.
- No premature abstraction: don't build config systems, plugin points, or
  generalized utilities for a single current use case. This project
  intentionally grows in scope over time (phase by phase) — resist
  pulling later-phase concerns into the current phase's code.
- Prefer small, direct scripts/modules over frameworks-within-frameworks,
  especially in Phase 1.

## Working agreements
- Don't start building a phase's code until the user explicitly asks —
  even though the roadmap is known, each phase should be a deliberate,
  discussed step, not an automatic continuation.
- When a tech-stack decision gets made during a session, add it to the
  "Tech stack decisions" section above in the same session.
