"""MaxProb baseline from local Qwen3.5 restricted-label probabilities."""

from __future__ import annotations

import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

from baselines import common  # noqa: E402
from data import splits  # noqa: E402


def main() -> None:
    split = splits.load_split()
    val_rows = common.build_rows(set(split["val"]))
    test_rows = common.build_rows(set(split["test"]))
    common.report(
        "maxprob",
        val_rows,
        common.maxprob_scores(val_rows),
        test_rows,
        common.maxprob_scores(test_rows),
    )


if __name__ == "__main__":
    main()
