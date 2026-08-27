from __future__ import annotations


def grid_seconds() -> int:
    """Common resampling grid for both sensors."""
    return 60


def history_minutes() -> int:
    """Length of the input window fed to the forecaster."""
    return 60


def horizon_minutes() -> int:
    """How far ahead we predict."""
    return 30


def shared_channels() -> list[str]:
    """Channels measured by both sensors in comparable units."""
    return ["temperature", "humidity", "pm_1_0", "pm_2_5", "pm_4_0", "pm_10_0"]


def data_windows() -> dict[str, tuple[str, str]]:
    """SEN55 and Atmocube export paths for each collection window."""
    return {
        "june": (
            "data_raw/dataset/SEN55_june.csv",
            "data_raw/dataset/Atmocube_june.csv",
        ),
        "august": (
            "data_raw/dataset/SEN55_aug.csv",
            "data_raw/dataset/Atmocube_aug.csv",
        ),
    }

def target_column() -> str:
    """Series the forecaster predicts."""
    return "pm_2_5_atmo"