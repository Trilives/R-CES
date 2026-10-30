# Environment

This repository is a standalone `uv` project.

## Minimal setup

```bash
uv sync --locked --python 3.13
sha256sum -c data/SHA256SUMS
uv run --locked python tmp/validate_rces_pairs.py
uv run --locked python -m unittest tests/test_offline_experiment_boundary.py tests/test_experiment_matrix.py tests/test_retrieval.py tests/test_evaluate_scores.py tests/test_image_release.py tests/test_metric_thresholds_and_families.py
```

These CPU checks need no images, model weights or API keys. Python 3.13 is selected
explicitly so a fresh clone uses the tested interface.

## Verify the paper on a CPU

```bash
(cd results/acml2026 && sha256sum -c SHA256SUMS)
uv run --locked python experiments/verify_paper_results.py
# Faster point-estimate check (omits the bootstrap intervals):
RCES_VERIFY_INTERVALS=0 uv run --locked python experiments/verify_paper_results.py
```

The verifier checks archive hashes, source/prompt fingerprints and annotation
judgment fields, then recomputes the six tables. The full command uses 3,000
two-level bootstrap replicates, resampling seeds and 50 original-image families;
originals and crops stay together. Risk–coverage thresholds accept complete ties
with `score >= threshold`; AURC is the right-endpoint step integral over those
groups. No image, GPU, trained head or API key is needed. The validated full CPU check
completed in 100.8 seconds; the point-estimate-only check took 0.3 seconds.

## Fixed model snapshots

[configs/models.lock.json](configs/models.lock.json) owns repository IDs, full
commit revisions and checksums of the experimental model/config/tokenizer files.
The download helper uses the revision, creates the expected `models/<name>/`
directory and verifies the recorded hashes. Its default action prints a plan:

```bash
uv run --locked python experiments/data/download_models.py
RCES_MODEL_ACTION=download uv run --locked python experiments/data/download_models.py
# One answerer, or a comma-separated selection of manifest names:
RCES_MODELS=InternVL3_5-8B-HF RCES_MODEL_ACTION=download uv run --locked python experiments/data/download_models.py
RCES_MODELS=InternVL3_5-8B-HF RCES_MODEL_ACTION=verify uv run --locked python experiments/data/download_models.py
```

The default selection downloads the two encoders, SigLIP 2 and DINOv2. Gated
models require accepting the provider's terms and Hugging Face authentication.
The Qwen3.6 archive name `Qwen3.6-35B-A3B` denotes the remote FP8 answerer; its
manifest entry is `Qwen3.6-35B-A3B-FP8`. That entry pins a public snapshot for new
runs; the historical remote service's weight revision and hardware were not
recorded. Exact Task-2 verification uses its frozen predictions in the archive.

## Validated hardware and runtime

[configs/validated_runtime.json](configs/validated_runtime.json) records the
validated archival run: Python 3.13.13, PyTorch 2.11.0 with CUDA 13.0,
Transformers 5.8.1, NumPy 2.4.4, and an NVIDIA GeForce RTX 4090 reporting
50,857,050,112 bytes of memory. The committed dependency pins and `uv.lock` match
this environment. For fresh head runs, set `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`.
Head training uses float32, deterministic math attention,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, two CPU threads, 40 epochs, AdamW (learning rate
0.001, weight decay 0.0001), hidden dimension 256, dropout 0.1, and batch size 64
(v1) or 32 (v2). Original cached features were used for archival head regeneration;
each seed's unaffected AUROCs are checked against the stored experimental values.
A timed SigLIP 2–512 seed-0 run took 119.7 seconds for head training, scoring,
calibration and persistence while other archival jobs ran concurrently; feature
extraction and VLM inference are excluded. Its test scores exactly match the
archived seed-0 scores. VLM inference is checked through frozen predictions. The runtime record describes
the validated archival run, rather than assigning this hardware to every historical
VLM run.

## Public-input check

Follow [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) to acquire the fixed upstream
photographs and generate the annotated crops locally. Then run this example:

```bash
uv run --locked python experiments/evaluate_scores.py
uv run --locked python - <<'PY'
import json
from pathlib import Path
source = Path("experiments/artifacts/test_pairs.jsonl")
destination = Path("experiments/artifacts/example_scores.jsonl")
rows = [json.loads(line) for line in source.read_text().splitlines()]
destination.write_text("".join(json.dumps({"pair_id": row["pair_id"], "score": 0.5}) + "\n" for row in rows))
PY
RCES_SCORES=experiments/artifacts/example_scores.jsonl uv run --locked python experiments/evaluate_scores.py
```

This loads the paper's test split and scores a constant predictor, which has
AUROC 0.5 on both populations and within rules. It is an interface example,
not a trained R-CES result. The full GPU workflow additionally requires the
public model weights described in `DATA_AVAILABILITY.md`.

## Optional secrets

Cloud annotation and model-download helper scripts read `.env` from this package
root. Use `.env.example` as the template. The offline experiments and validation
commands above do not require API keys.

## Tested interface

The package targets Python 3.13. The committed code avoids repository-level
imports outside this repository; paths are resolved from the package root.
