# Contributing to lungct-kit

Thanks for helping make lung CT research faster and more trustworthy.

## Set up

```bash
git clone https://github.com/reshamrajs/lungct-kit && cd lungct-kit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check src tests
```

Tests use small synthetic scans, so you don't need to download any dataset.

## Rules

- **Never commit patient data**, not even one scan. Use synthetic volumes (see `tests/conftest.py`).
- Every new function gets a test. Geometry code gets a test against SimpleITK.
- Keep the core dependency list at NumPy + SciPy. Anything heavier goes in an optional extra.
- Follow the conventions in the README: arrays `[z, y, x]`, CSV coordinates world `(x, y, z)` mm.

## Adding a dataset

1. Add a `DatasetInfo` entry to `src/lungct_kit/registry.py`, with licence, size, citation and
   known pitfalls. Every fact needs a source you can link in the PR.
2. If its CSV has a different column layout, map the columns in `columns`.
3. Add a synthetic fixture in `tests/conftest.py` and tests that load it.

## Good first issues

- LIDC-IDRI loader (via `pylidc`)
- Random flips and rotations for 3D patches
- Loader for the LUNA25 64x128x128 nodule blocks
- A docs site (MkDocs)
