from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _days_since_start(times: pd.Series) -> pd.Series:
    """Elapsed days from the first reading, for fitting against time."""
    return (times - times.iloc[0]).dt.total_seconds() / 86400


def channel_difference(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    """Pair the two sensors on one channel and add their difference."""
    paired = df[["datetime", f"{channel}_sen55", f"{channel}_atmo"]].dropna()
    paired = paired.rename(
        columns={f"{channel}_sen55": "sen55", f"{channel}_atmo": "atmo"}
    )
    paired["difference"] = paired["sen55"] - paired["atmo"]
    return paired


def fit_drift(paired: pd.DataFrame) -> tuple[float, float]:
    """Least-squares slope and intercept of the difference against days elapsed."""
    slope, intercept = np.polyfit(
        _days_since_start(paired["datetime"]), paired["difference"], 1
    )
    return float(slope), float(intercept)


def summarise(df: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """Report mean difference, limits of agreement and drift rate per channel."""
    rows = []
    for channel in channels:
        paired = channel_difference(df, channel)
        mean = paired["difference"].mean()
        sd = paired["difference"].std()
        rows.append(
            {
                "channel": channel,
                "n": len(paired),
                "mean_difference": mean,
                "sd_difference": sd,
                "loa_lower": mean - 1.96 * sd,
                "loa_upper": mean + 1.96 * sd,
                "drift_per_day": fit_drift(paired)[0],
            }
        )
    return pd.DataFrame(rows)


def plot_difference_over_time(df: pd.DataFrame, channel: str, path: str) -> None:
    """Save a time series of the sensor difference with its fitted trend."""
    paired = channel_difference(df, channel)
    days = _days_since_start(paired["datetime"])
    slope, intercept = fit_drift(paired)

    figure, axes = plt.subplots(figsize=(9, 4))
    axes.plot(paired["datetime"], paired["difference"], linewidth=0.6)
    axes.plot(paired["datetime"], slope * days + intercept, linewidth=1.6)
    axes.axhline(0, linewidth=0.8, linestyle="--")
    axes.set_title(f"{channel}: SEN55 minus Atmocube ({slope:+.3f} per day)")
    axes.set_ylabel("difference")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_bland_altman(df: pd.DataFrame, channel: str, path: str) -> None:
    """Save a Bland-Altman plot of agreement between the two sensors."""
    paired = channel_difference(df, channel)
    average = (paired["sen55"] + paired["atmo"]) / 2
    mean = paired["difference"].mean()
    sd = paired["difference"].std()

    figure, axes = plt.subplots(figsize=(6, 4.5))
    axes.scatter(average, paired["difference"], s=3, alpha=0.25)
    axes.axhline(mean, linewidth=1.4)
    axes.axhline(mean - 1.96 * sd, linewidth=1.0, linestyle="--")
    axes.axhline(mean + 1.96 * sd, linewidth=1.0, linestyle="--")
    axes.set_title(f"{channel}: Bland-Altman")
    axes.set_xlabel("mean of both sensors")
    axes.set_ylabel("SEN55 minus Atmocube")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)




def noise_floor(df: pd.DataFrame, channels: list[str], seconds: int) -> pd.DataFrame:
    """Estimate per-sensor white noise from the change in difference between adjacent slots."""
    rows = []
    for channel in channels:
        paired = channel_difference(df, channel)
        step = paired["datetime"].diff().dt.total_seconds()
        # only adjacent slots: across a gap the slow structure moves and inflates the estimate
        change = paired["difference"].diff()[step == seconds]
        total = paired["difference"].std()
        combined = change.std() / np.sqrt(2)
        rows.append(
            {
                "channel": channel,
                "sd_difference": total,
                "combined_noise": combined,
                "per_sensor_noise": combined / np.sqrt(2),
                "noise_share": (combined / total) ** 2,
            }
        )
    return pd.DataFrame(rows)