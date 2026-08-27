from __future__ import annotations

import pandas as pd

from core_settings import grid_seconds, shared_channels


def _timestamp_column(df: pd.DataFrame) -> str:
    """Name of the epoch column, which differs between the two export formats."""
    return "timestamp" if "timestamp" in df.columns else "timestamp_rounded"


def load_sensor_csv(path: str, channels: list[str]) -> pd.DataFrame:
    """Read one raw sensor export, keeping the timestamp and the named channels."""
    df = pd.read_csv(path)
    df = df.rename(columns={_timestamp_column(df): "timestamp"})
    df["timestamp"] = df["timestamp"].astype(float)
    return df[["timestamp"] + channels]


def align_to_grid(df: pd.DataFrame, seconds: int) -> pd.DataFrame:
    """Snap readings onto a fixed grid, averaging any that share a bucket."""
    df = df.copy()
    df["timestamp_rounded"] = (df["timestamp"] // seconds * seconds).astype(int)
    df = df.drop(columns=["timestamp"])
    return df.groupby("timestamp_rounded").mean().reset_index()


def merge_sensors(sen55: pd.DataFrame, atmocube: pd.DataFrame) -> pd.DataFrame:
    """Join two aligned sensors into one table, suffixing each channel by source."""
    merged = pd.merge(
        sen55,
        atmocube,
        on="timestamp_rounded",
        how="outer",
        suffixes=("_sen55", "_atmo"),
    )
    merged = merged.sort_values("timestamp_rounded").reset_index(drop=True)
    merged["datetime"] = pd.to_datetime(merged["timestamp_rounded"], unit="s")

    lead = ["datetime", "timestamp_rounded"]
    return merged[lead + [c for c in merged.columns if c not in lead]]


def load_window(sen55_path: str, atmocube_path: str) -> pd.DataFrame:
    """Load one collection window with both sensors on a common grid."""
    seconds = grid_seconds()
    channels = shared_channels()
    sen55 = align_to_grid(load_sensor_csv(sen55_path, channels), seconds)
    atmocube = align_to_grid(load_sensor_csv(atmocube_path, channels), seconds)
    return merge_sensors(sen55, atmocube)