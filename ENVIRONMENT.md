# Environment

This anonymous supplement is a standalone `uv` project.

## Minimal setup

```bash
uv sync
UV_CACHE_DIR=/tmp/uv-cache uv run python tmp/validate_rces_pairs.py
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests/test_offline_experiment_boundary.py tests/test_experiment_matrix.py tests/test_retrieval.py
```

The full GPU workflow additionally requires the omitted image directory and
public model weights described in `DATA_AVAILABILITY.md`.

## Optional secrets

Cloud annotation and model-download helper scripts read `.env` from this package
root. Use `.env.example` as the template. The offline experiments and validation
commands above do not require API keys.

## Tested interface

The package targets Python 3.13. The committed code avoids repository-level
imports outside this supplement; paths are resolved from the package root.
