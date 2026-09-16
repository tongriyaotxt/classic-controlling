# [Project] classic-controlling: zero-tuning ADRC in pure NumPy — one line does system ID + tuning; survives the nonlinear unstable plant where Z-N PID diverges

**Author: tongriyao (田晓潼)** — feedback, stars, issues and PRs welcome.

Hi r/ControlTheory — I'd like to share a small library I just released, and I'll lead with the numbers because that's what I'd want to see.

## Blind benchmark: zero-tuning ADRC vs model-tuned PID

Three plants, step load disturbance at t=20s. The PID controllers were tuned from the **same step-test data** using Z-N / IMC rules — i.e., PID gets the full advantage of "knowing the model". The ADRC was tuned by `ADRC.auto_tune(plant, dt=0.05)` with **zero human-set parameters** (automatic step experiment → FOPDT fit → bandwidth selection under delay phase-margin and sampling-rate hard caps).

| Plant | PID (model-tuned) | ADRC (zero-tuning) |
|---|---|---|
| First-order + delay | Stable, **faster** (as expected — model is accurate) | Stable, steady-state error 0.89% |
| Second-order underdamped | 6.7% overshoot | **0% overshoot**, SSE 3.21% |
| Nonlinear, open-loop unstable (ÿ = −ẏ + u + 0.5y²) | **Diverges** — growing oscillation, 757% overshoot, recovery time ∞ | **Locked**, SSE 1.03% |

The point is not "ADRC beats PID" — on well-behaved LTI plants, model-tuned PID is genuinely faster, and the README says so. The point is the trade: the auto-tuned bandwidth is calibrated against measured divergence boundaries (phase-margin cap + sampling cap), so it **trades speed for a guarantee**. Use it where modeling isn't worth it, the plant is nonlinear/time-varying/unstable, or you have a hundred loops and no appetite for hand-tuning each one.

## What's in the box

- **Self-tuning LADRC** — one-line `auto_tune`; optional RLS online refinement (marked experimental, off by default — honest note: its fixed-point sits ~0.65× low on first-order plants, so it's rate-limited and guard-railed)
- **Vectorized UKF** — sigma-point generation/propagation/statistics fully broadcast in NumPy, no per-point Python loop: **2.2× / 6.0× / 11.2× speedup** (dim 2/8/32) over the mathematically equivalent point-by-point implementation
- **Dependency-free linear MPC** — one-time QP compilation + built-in OSQP-style ADMM solver + (z,s,y) warm start (total iterations 3687 → 1935, −47%)

Pure NumPy/SciPy, MIT license, 21 tests green. **Every performance number is a real measurement** — the benchmark scripts are in the repo (`benchmarks/`, `examples/`), please run them and prove me wrong.

## Links

```bash
pip install classic-controlling
```

GitHub: **github.com/tongriyaotxt/classic-controlling**

`examples/demo_adrc_vs_pid.py` reproduces the comparison figure above (PID diverging vs ADRC locked on the nonlinear plant).

---

Motivation: classic control algorithms deserve an implementation worthy of 2026. ADRC theory is 20+ years old, but "zero tuning" was never actually finished in engineering practice — the sampling cap and phase-margin cap in the bandwidth rule were bought with measured divergence boundaries. Happy to answer questions about the tuning rule, the LESO discretization (semi-implicit Euler, z3→z2→z1 — reverting to explicit breaks RLS regression consistency), or anything else.
