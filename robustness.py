from __future__ import annotations

from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import forecasting
import perturbations

Predictor = Callable[[np.ndarray], np.ndarray]
Corruption = Callable[[np.ndarray, float], np.ndarray]


def sweep(
    predictors: dict[str, Predictor],
    inputs: np.ndarray,
    targets: np.ndarray,
    name: str,
    corrupt: Corruption,
    values: Iterable[float],
) -> pd.DataFrame:
    """Measure forecast error as one failure mode is turned up, with every model held frozen."""
    rows = []
    for value in values:
        corrupted = corrupt(inputs, value)
        for label, predict in predictors.items():
            rows.append(
                {
                    "model": label,
                    "perturbation": name,
                    "parameter": value,
                    "mae": forecasting.mean_absolute_error(targets, predict(corrupted)),
                }
            )
    return pd.DataFrame(rows)


def failure_modes(
    noise_sigma: float, lengths: np.ndarray, seed: int = 0
) -> dict[str, tuple[Corruption, list[float]]]:
    """Each failure mode paired with the severities to sweep, anchored on measured levels."""
    multiples = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    return {
        "noise": (
            lambda x, sigma: perturbations.add_noise(x, sigma, seed),
            [multiple * noise_sigma for multiple in multiples],
        ),
        "scale": (perturbations.scale, [1.0, 1.1, 1.2, 1.4, 1.7, 2.2, 3.0]),
        "dropout": (
            lambda x, blocks: perturbations.drop_blocks(x, lengths, int(blocks), seed),
            [0, 1, 2, 3, 5, 8],
        ),
    }


def degradation(
    predictors: dict[str, Predictor],
    inputs: np.ndarray,
    targets: np.ndarray,
    noise_sigma: float,
    lengths: np.ndarray,
    seed: int = 0,
) -> pd.DataFrame:
    """Sweep all three failure modes, anchored on the levels measured from the sensors."""
    frames = [
        sweep(predictors, inputs, targets, name, corrupt, values)
        for name, (corrupt, values) in failure_modes(noise_sigma, lengths, seed).items()
    ]
    return pd.concat(frames, ignore_index=True)


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Report the clean error, the worst error and the ratio between them."""
    rows = []
    for (label, name), group in frame.groupby(["model", "perturbation"], sort=False):
        clean = group["mae"].iloc[0]
        worst = group["mae"].max()
        rows.append(
            {
                "model": label,
                "perturbation": name,
                "clean_mae": clean,
                "worst_mae": worst,
                "ratio": worst / clean,
            }
        )
    return pd.DataFrame(rows)


def plot_degradation(frame: pd.DataFrame, path: str) -> None:
    """Save one panel per failure mode, each showing error against severity."""
    names = list(dict.fromkeys(frame["perturbation"]))
    figure, axes = plt.subplots(1, len(names), figsize=(11, 3.4), sharey=True)

    for axis, name in zip(axes, names):
        panel = frame[frame["perturbation"] == name]
        for label, group in panel.groupby("model", sort=False):
            axis.plot(group["parameter"], group["mae"], marker="o", markersize=4, label=label)
        axis.set_title(name)
        axis.set_xlabel("severity")

    axes[0].set_ylabel("mean absolute error")
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)