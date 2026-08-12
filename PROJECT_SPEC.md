# Digit Recognizer — Production Edition

## What it does
A web app where a user draws a digit (0–9) on a canvas and gets a real-time
prediction from a neural network trained on MNIST. The project is built in
public, in phases, going from a from-scratch numpy implementation through
a deployed product to a hardened, MLOps-instrumented, PyTorch-based v2.

## Who it's for
- **Primary:** the author — a learning vehicle for ML fundamentals
  (Nielsen's *Neural Networks and Deep Learning*) and for the full
  Claude Code workflow (planning, building, hardening, iterating).
- **Secondary:** portfolio reviewers / hiring managers who want to see
  first-principles ML understanding *and* the ability to ship and operate
  a real product, not just a notebook.

## What "v1 done" means
- A neural network implemented from scratch in numpy (no ML frameworks),
  trained on MNIST, following Nielsen's approach.
- A FastAPI backend that serves predictions from the trained model.
- A simple canvas frontend: draw a digit, submit, see the predicted digit.
- Deployed and reachable at a public URL — anyone can open the link, draw
  a digit, and get a live prediction.

v1 is considered done the moment all four of the above are true
simultaneously. Nothing else is required for v1.

## Explicitly out of scope (for now)
These are real, planned phases (2–6 of the project roadmap) but are
**not** part of v1 and should not be pulled forward:
- Automated tests, CI/CD pipelines, Docker packaging (Phase 3: harden)
- PyTorch reimplementation / v1-vs-v2 comparison (Phase 4)
- Experiment tracking (Weights & Biases), model versioning, monitoring
  (Phase 5: MLOps)
- Polished README, recorded demo, resume write-up (Phase 6)
- Any UI polish beyond "a working canvas" (styling, mobile support,
  drawing tools, undo, etc.)
- Multi-digit recognition, handwriting beyond single digits, any dataset
  other than MNIST
