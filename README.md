# Accelerated Filtering on Graphs using Lanczos (AFGL)

Accelerated Filtering on Graphs using the Lanczos method (AFGL). This repo
contains a Lanczos/Arnoldi implementation for graph Laplacian filtering, plus
scripts to reproduce the experiments and plots.

[Report](report/report.pdf)

---

## Quick Start

### Setup

1. **Install uv**: `curl -LsSf https://astral-sh.uv/install.sh | sh`
2. **Clone repo**
3. **Sync Env**: `uv sync`
4. **Install Hooks**: `uv run pre-commit install`

### Commands

- **Run App**: `uv run afgl-run`
- **Test**: `uv run pytest`
- **Format**: `uv run ruff format .`
- **Lint**: `uv run ruff check . --fix`

## Project structure

```
├── out (generated files)
├── report (project report)
│   ├── assets
│   └── test
├── src (source code)
│   └── afgl
│       └── util (numerical util functions)
│       └── plot (plotting functions)
└── tests (unit tests)
```

## References

[1] Susnjara et al., Accelerated filtering on graphs using Lanczos method.

[2] https://epfl-lts2.github.io/gspbox-html/
