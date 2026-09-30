"""Kamath-style correctness calibrator over cheap, checkpoint-free features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

from baselines import common  # noqa: E402
from data import splits  # noqa: E402


def feature_matrix(rows: list[common.BaselineRow]) -> np.ndarray:
    return np.asarray(
        [
            [
                row.confidence,
                float(row.decision_scope == "requires_external_evidence"),
                min(row.n_critical, 5) / 5.0,
                1.0,
            ]
            for row in rows
        ],
        dtype=float,
    )


def fit_logistic(rows: list[common.BaselineRow], *, steps: int = 1000, lr: float = 0.1) -> np.ndarray:
    if not rows:
        raise RuntimeError("Kamath calibrator needs aligned validation predictions")
    x = feature_matrix(rows)
    y = np.asarray([row.compliance_correct for row in rows], dtype=float)
    weights = np.zeros(x.shape[1], dtype=float)
    for _ in range(steps):
        logits = np.clip(x @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = x.T @ (probabilities - y) / len(y)
        weights = weights - lr * gradient
    return weights


def predict_scores(rows: list[common.BaselineRow], weights: np.ndarray) -> list[float]:
    logits = np.clip(feature_matrix(rows) @ weights, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-logits))).tolist()


def main() -> None:
    split = splits.load_split()
    val_rows = common.build_rows(set(split["val"]))
    test_rows = common.build_rows(set(split["test"]))
    weights = fit_logistic(val_rows)
    common.report(
        "kamath",
        val_rows,
        predict_scores(val_rows, weights),
        test_rows,
        predict_scores(test_rows, weights),
    )


if __name__ == "__main__":
    main()
