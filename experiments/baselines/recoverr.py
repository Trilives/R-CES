"""Reduced ReCoVERR reliability gate over existing local-Qwen confidence.

This preserves the fixed reliability decision rule but does not implement the
paper's iterative evidence-gathering loop. Reports and paper text must call this
variant ReCoVERR-reduced.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

import config  # noqa: E402
from baselines import common  # noqa: E402
from data import splits  # noqa: E402


def main() -> None:
    split = splits.load_split()
    test_rows = common.build_rows(set(split["test"]))
    common.report(
        "recoverr_reduced",
        [],
        [],
        test_rows,
        common.maxprob_scores(test_rows),
        fixed_tau=1.0 - config.TARGET_RISK,
    )


if __name__ == "__main__":
    main()
