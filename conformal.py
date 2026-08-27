from __future__ import annotations

from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robustness import Corruption, Predictor


def _conformal_quantile(residuals: np.ndarray, level: float) -> float:
    """Smallest residual covering the requested share, with the finite-sample correction."""
    rank = int(np.ceil((len(residuals) + 1) * level))
    if rank > len(residuals):
        return float("inf")
    return float(np.sort(residuals)[rank - 1])


def calibrate(
    predict: Predictor, inputs: np.ndarray, targets: np.ndarray, level: float
) -> float:
    """Half-width of the prediction interval, sized on data the model never trained on."""
    return _conformal_quantile(np.abs(targets - predict(inputs)), level)


def coverage(
    predict: Predictor, inputs: np.ndarray, targets: np.ndarray, width: float
) -> float:
    """Share of targets falling inside prediction plus or minus the given half-width."""
    return float((np.abs(targets - predict(inputs)) <= width).mean())


def sweep_coverage(
    predict: Predictor,
    inputs: np.ndarray,
    targets: np.ndarray,
    width: float,
    modes: dict[str, tuple[Corruption, list[float]]],
) -> pd.DataFrame:
    """Measure whether the interval keeps its promise as each failure mode is turned up."""
    rows = []
    for name, (corrupt, values) in modes.items():
        for value in values:
            rows.append(
                {
                    "perturbation": name,
                    "parameter": value,
                    "width": 2 * width,
                    "coverage": coverage(predict, corrupt(inputs, value), targets, width),
                }
            )
    return pd.DataFrame(rows)


def plot_coverage(frame: pd.DataFrame, level: float, path: str) -> None:
    """Save one panel per failure mode, each showing coverage against severity."""
    names = list(dict.fromkeys(frame["perturbation"]))
    figure, axes = plt.subplots(1, len(names), figsize=(11, 3.4), sharey=True)

    for axis, name in zip(axes, names):
        panel = frame[frame["perturbation"] == name]
        axis.plot(panel["parameter"], panel["coverage"], marker="o", markersize=4)
        axis.axhline(level, linewidth=1.0, linestyle="--")
        axis.set_ylim(0, 1.02)
        axis.set_title(name)
        axis.set_xlabel("severity")

    axes[0].set_ylabel("coverage")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)