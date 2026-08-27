from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def reindex_to_grid(df: pd.DataFrame, seconds: int) -> pd.DataFrame:
    """Insert an explicit row for every slot, including those absent from both sensors."""
    start = int(df["timestamp_rounded"].min())
    stop = int(df["timestamp_rounded"].max()) + seconds
    grid = pd.DataFrame({"timestamp_rounded": range(start, stop, seconds)})

    full = grid.merge(df.drop(columns=["datetime"]), on="timestamp_rounded", how="left")
    full["datetime"] = pd.to_datetime(full["timestamp_rounded"], unit="s")

    lead = ["datetime", "timestamp_rounded"]
    return full[lead + [c for c in full.columns if c not in lead]]


def _run_lengths(missing: pd.Series) -> pd.Series:
    """Length of each unbroken run of missing readings."""
    runs = missing.groupby((missing != missing.shift()).cumsum()).sum()
    return runs[runs > 0]


def summarise(df: pd.DataFrame, channel: str, seconds: int) -> pd.DataFrame:
    """Report coverage and gap structure for each sensor on one channel."""
    full = reindex_to_grid(df, seconds)
    rows = []
    for sensor in ("sen55", "atmo"):
        missing = full[f"{channel}_{sensor}"].isna()
        runs = _run_lengths(missing)
        rows.append(
            {
                "sensor": sensor,
                "slots": len(full),
                "missing": int(missing.sum()),
                "missing_pct": 100 * missing.mean(),
                "outages": len(runs),
                "longest_gap": int(runs.max()) if len(runs) else 0,
                "median_gap": float(runs.median()) if len(runs) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def simultaneous_share(df: pd.DataFrame, channel: str, seconds: int) -> float:
    """Fraction of affected slots where both sensors were silent at the same moment."""
    full = reindex_to_grid(df, seconds)
    sen55 = full[f"{channel}_sen55"].isna()
    atmocube = full[f"{channel}_atmo"].isna()
    affected = (sen55 | atmocube).sum()
    return float((sen55 & atmocube).sum() / affected) if affected else 0.0


def gap_lengths(df: pd.DataFrame, column: str, seconds: int) -> pd.Series:
    """Outage lengths in slots, for sampling realistic dropouts in perturbations."""
    full = reindex_to_grid(df, seconds)
    return _run_lengths(full[column].isna()).reset_index(drop=True)


def usable_windows(
    df: pd.DataFrame, column: str, seconds: int, history: int, horizon: int
) -> dict[str, int]:
    """Count sliding windows whose history is complete and whose target is present."""
    full = reindex_to_grid(df, seconds)
    present = full[column].notna()
    complete = present.rolling(history).sum() == history
    target = present.shift(-horizon).fillna(False).astype(bool)

    possible = max(len(full) - history - horizon + 1, 0)
    return {
        "slots": len(full),
        "possible": possible,
        "usable": int((complete & target).sum()),
    }


def plot_coverage(df: pd.DataFrame, channel: str, seconds: int, path: str) -> None:
    """Save a strip chart marking every slot in which a sensor failed to report."""
    full = reindex_to_grid(df, seconds)

    figure, axes = plt.subplots(figsize=(9, 2.4))
    for offset, sensor in enumerate(("sen55", "atmo")):
        missing = full[f"{channel}_{sensor}"].isna()
        axes.scatter(full["datetime"][missing], [offset] * int(missing.sum()), s=40, marker="|")

    axes.set_yticks([0, 1])
    axes.set_yticklabels(["SEN55", "Atmocube"])
    axes.set_ylim(-0.5, 1.5)
    axes.set_title(f"{channel}: slots with no reading")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)