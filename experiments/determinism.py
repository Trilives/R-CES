"""Best-effort training determinism for the R-CES experiments.

The v2 head's attention made same-seed runs drift (E1 head checkpoint AUROC
0.612 vs an ablation retrain at 0.734). `set_determinism` seeds every RNG and
requests deterministic cuDNN / algorithms so a fixed seed reproduces, and the
multi-seed drivers vary the seed to report mean +/- CI on top of that.

`use_deterministic_algorithms(..., warn_only=True)` keeps unsupported CUDA ops
(some attention backward kernels) from raising; they fall back to their
non-deterministic path with a warning instead of crashing the run.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_determinism(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if not deterministic:
        return
    # Must be set before the first CUBLAS call for deterministic GEMMs; harmless
    # if CUDA is already up (we still set cudnn + algo flags below).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # The v2 head's attention drifts because the flash / mem-efficient SDPA
    # backward kernels are non-deterministic. Force the math backend, whose
    # backward is deterministic, so same-seed runs reproduce.
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        # Older torch without warn_only: best-effort, skip rather than crash.
        pass
