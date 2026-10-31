"""Generate the paper's data-driven figure from real artifacts.

Writes a vector PDF to experiments/artifacts/figures/. Run with matplotlib available:

  UV_CACHE_DIR=/tmp/uv-cache uv run --with matplotlib \
      python experiments/figures/make_figures.py

  fig_calib.pdf -- reliability diagrams (raw / temperature / isotonic, fit on
                   validation) for Task-1 gate-on, Task-1 visual subset and
                   Task-2, pooled over the 10 seeds of
                   experiments/ablation/camera_ready.py.

The other figures are TikZ drawings or photos in the paper source, not data plots.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# experiments/figures/make_figures.py -> experiments/artifacts
ART = Path(__file__).resolve().parents[1] / "artifacts"
OUT = ART / "figures"

# Colorblind-friendly, also distinguishable in grayscale via marker.
C_V1, C_V2, C_BASE = "#0072B2", "#D55E00", "#888888"
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "figure.dpi": 150,
    "pdf.fonttype": 42,  # embed real fonts (TrueType), not Type-3
    "ps.fonttype": 42,
})

CALIB_REPORT = ART / "camera_ready_siglip2_512_cropaug.json"
CALIB_PANELS = [
    ("task1_gate_on", "Task-1, gate on (267)"),
    ("task1_visual", "Task-1, visual (217)"),
    ("task2", "Task-2, Qwen3.5-9B"),
]
CALIB_METHODS = [("raw", "raw", C_BASE, "o"), ("temperature", "temperature", C_V1, "s"),
                 ("isotonic", "isotonic", C_V2, "^")]


def fig_calib() -> None:
    """Reliability diagrams before/after validation-fit calibration."""
    archive = Path(os.environ.get("RCES_RESULTS_DIR", str(ART.parents[1] / "results/acml2026")))
    if (archive / "manifest.json").is_file():
        import sys
        sys.path.insert(0, str(ART.parent))
        from ablation.camera_ready import reliability_bins
        from verify_paper_results import score_runs
        runs = score_runs(archive / "scores/v1_siglip2_512")
        tables = json.loads((archive / "reports/paper_tables.json").read_text())
        report = {}
        for block, _ in CALIB_PANELS:
            report[block] = {"ece": tables["table_3"]["ece"][block], "reliability_pooled": {}}
            for method, *_ in CALIB_METHODS:
                scores = [s for run in runs for s in run["result"]["calibration"][block]["_scores"][method]]
                labels = [y for run in runs for y in run["frames"]["targets"][block]]
                report[block]["reliability_pooled"][method] = reliability_bins(scores, labels, 10)
    else:
        report = json.loads(CALIB_REPORT.read_text())["summary"]["calibration"]
    # Size text for the full-width manuscript figure. A shared legend stays
    # outside the data panels; ECE values are reported in the adjacent Table 3.
    fig, axes = plt.subplots(1, len(CALIB_PANELS), figsize=(7.2, 2.0), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.25, top=0.75, wspace=0.15)
    line_styles = ("-", "--", "-.")
    for ax, (block, title) in zip(axes, CALIB_PANELS):
        for (key, name, color, marker), line_style in zip(CALIB_METHODS, line_styles):
            bins = report[block]["reliability_pooled"][key]
            conf = np.array(bins["confidence"], float)
            acc = np.array(bins["accuracy"], float)
            count = np.array(bins["count"], float)
            keep = count > 0
            ax.plot(conf[keep], acc[keep], color=color, lw=1.2,
                    ls=line_style, alpha=0.8, zorder=2)
            ax.scatter(conf[keep], acc[keep], s=8 + 120 * count[keep] / count.sum(),
                       color=color, marker=marker, alpha=0.85, zorder=3)
        ax.plot([0, 1], [0, 1], color="black", ls=":", lw=1, zorder=1)
        # Include complete markers at scores 0 and 1, without changing any bins.
        ax.set_xlim(-0.055, 1.055)
        ax.set_ylim(-0.055, 1.055)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_title(title, fontsize=10, pad=7)
        ax.set_xlabel("predicted score", fontsize=10, labelpad=4)
        ax.tick_params(labelsize=9, length=3, pad=3)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("empirical rate", fontsize=10, labelpad=4)
    handles = [
        Line2D([], [], color=color, marker=marker, ls=line_style,
               lw=1.2, markersize=5, label=name.capitalize())
        for (_, name, color, marker), line_style in zip(CALIB_METHODS, line_styles)
    ]
    handles.append(Line2D([], [], color="black", ls=":", lw=1,
                          label="Perfect calibration"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 1.01),
               ncol=4, frameon=False, fontsize=9.5, handlelength=2.3,
               columnspacing=1.5)
    fig.savefig(OUT / "fig_calib.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_calib.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_calib()
