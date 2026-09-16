# Contributing

Contributions are welcome — bug reports, new algorithms, better demos, docs.

## Setup

```bash
git clone <repo-url>
cd classic_controlling
pip install -e ".[dev]"
```

## Running tests

```bash
python -m pytest tests/ -v     # 18 tests, must stay green
python benchmarks/bench_ukf.py # optional: performance regression check
```

## Guidelines

- Keep the public API in `classic_controlling/__init__.py` backward compatible.
- Pure NumPy/SciPy only — no new runtime dependencies without discussion.
- Add tests for new features; the auto-tune blind tests (`tests/test_autotune.py`)
  are the quality gate for anything touching ADRC/auto-tuning.
- Demos live in `examples/` and must run standalone (`python examples/demo_*.py`),
  writing figures to `docs/figures/`.
