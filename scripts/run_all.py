from __future__ import annotations

import numpy as np

import agreement
import conformal
import forecasting
import gaps
import robustness
from core_settings import (
    data_windows,
    grid_seconds,
    history_minutes,
    horizon_minutes,
    shared_channels,
    target_column,
)
from ingest import load_window

COVERAGE = 0.90


def _pooled_gap_lengths(seconds: int) -> np.ndarray:
    """Outage lengths from every window and sensor, so dropouts are sampled realistically."""
    lengths: list[int] = []
    for sen55_path, atmocube_path in data_windows().values():
        df = load_window(sen55_path, atmocube_path)
        for column in ("pm_2_5_sen55", "pm_2_5_atmo"):
            lengths.extend(gaps.gap_lengths(df, column, seconds).tolist())
    return np.array(lengths)


def main() -> None:
    """Run every stage of the pipeline for each collection window."""
    seconds = grid_seconds()
    channels = shared_channels()
    history = history_minutes()
    horizon = horizon_minutes()
    target = target_column()
    lengths = _pooled_gap_lengths(seconds)

    for name, (sen55_path, atmocube_path) in data_windows().items():
        df = load_window(sen55_path, atmocube_path)
        print(f"[{name}] {len(df)} rows, {df['datetime'].min()} to {df['datetime'].max()}")

        print(agreement.summarise(df, channels).round(3).to_string(index=False))
        print()
        noise = agreement.noise_floor(df, channels, seconds)
        print(noise.round(4).to_string(index=False))
        print()
        print(agreement.ratio_summary(df, channels).round(3).to_string(index=False))
        print()
        print(gaps.summarise(df, "pm_2_5", seconds).round(3).to_string(index=False))
        print(f"simultaneous share: {gaps.simultaneous_share(df, 'pm_2_5', seconds):.3f}")

        full = gaps.reindex_to_grid(df, seconds)
        inputs, targets = forecasting.build_windows(full[target], history, horizon)
        blocks = forecasting.split_by_time(inputs, targets, 0.70, 0.15)
        print({key: len(value[0]) for key, value in blocks.items()})

        model = forecasting.train_gru(*blocks["train"])
        test_inputs, test_targets = blocks["test"]
        predictors = {
            "gru": lambda x: forecasting.predict(model, x),
            "persistence": forecasting.persistence,
        }

        sigma = float(noise.loc[noise["channel"] == "pm_2_5", "per_sensor_noise"].iloc[0])
        frame = robustness.degradation(predictors, test_inputs, test_targets, sigma, lengths)
        print()
        print(robustness.summarise(frame).round(3).to_string(index=False))
        print()

        modes = robustness.failure_modes(sigma, lengths)
        width = conformal.calibrate(predictors["gru"], *blocks["calibration"], COVERAGE)
        clean = conformal.coverage(predictors["gru"], test_inputs, test_targets, width)
        print(f"interval +/-{width:.3f}, clean coverage {clean:.3f} against a promise of {COVERAGE}")
        intervals = conformal.sweep_coverage(
            predictors["gru"], test_inputs, test_targets, width, modes
        )
        print(intervals.round(3).to_string(index=False))
        print()

        for channel in ("temperature", "pm_2_5"):
            agreement.plot_difference_over_time(df, channel, f"figures/{name}_{channel}_time.png")
            agreement.plot_bland_altman(df, channel, f"figures/{name}_{channel}_bland_altman.png")
        gaps.plot_coverage(df, "pm_2_5", seconds, f"figures/{name}_coverage.png")
        robustness.plot_degradation(frame, f"figures/{name}_degradation.png")
        conformal.plot_coverage(intervals, COVERAGE, f"figures/{name}_conformal.png")


if __name__ == "__main__":
    main()