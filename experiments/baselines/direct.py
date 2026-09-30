"""Local Qwen3.5 direct-answer full-coverage reference."""

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
    test_rows = common.build_rows(set(split["test"]))
    common.report(
        "local_qwen35_direct",
        [],
        [],
        test_rows,
        [1.0] * len(test_rows),
        fixed_tau=0.0,
    )


if __name__ == "__main__":
    main()
