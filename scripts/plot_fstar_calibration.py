#!/usr/bin/env python3
"""Render the f* prediction/oracle scatter from the formal summary JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


COLORS = {
    "JHU/SHA/SHB": "#c0392b",
    "QNRF/SHA/SHB": "#2471a3",
    "SHA/SHB": "#5d6d7e",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="runs_real/fstar_calibration/fstar_calibration_summary.json",
    )
    parser.add_argument(
        "--out",
        default="runs_real/fstar_calibration/fstar_calibration_scatter.pdf",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(Path(args.summary).read_text())
    rows = payload["rows"]
    figure, axis = plt.subplots(figsize=(4.8, 4.4))
    axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="ideal")

    for family in COLORS:
        counts = Counter(
            (float(row["f_pred"]), float(row["f_oracle"]))
            for row in rows
            if row["family"] == family
        )
        if not counts:
            continue
        axis.scatter(
            [point[0] for point in counts],
            [point[1] for point in counts],
            s=[35 + 28 * count for count in counts.values()],
            color=COLORS[family],
            alpha=0.72,
            edgecolors="white",
            linewidths=0.7,
            label=family,
        )

    metrics = payload["overall"]
    axis.text(
        0.03,
        0.97,
        (
            f"MAE={metrics['mae_factor']:.3f}\n"
            f"Spearman={metrics['spearman_rho']:.3f}\n"
            f"exact={100*metrics['exact_match_rate']:.1f}%"
        ),
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.8"},
    )
    axis.set_xlim(0.0, 1.03)
    axis.set_ylim(0.0, 1.03)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"Train-only predicted $f^*_{\mathrm{pred}}$")
    axis.set_ylabel(
        r"Matched-MSE test oracle $f^*_{\mathrm{oracle}}$ (diagnostic)"
    )
    axis.grid(alpha=0.18)
    axis.legend(loc="lower right", fontsize=7.5)
    figure.tight_layout()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    png = output.with_suffix(".png")
    figure.savefig(png, dpi=220, bbox_inches="tight")
    print("saved ->", output)
    print("saved ->", png)


if __name__ == "__main__":
    main()
