from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import ingest


def _readings(timestamps: list[float], values: list[float]) -> pd.DataFrame:
    """Minimal one-channel export, built by hand so the test touches no files."""
    return pd.DataFrame({"timestamp": timestamps, "pm_2_5": values})


class TimestampColumnTest(unittest.TestCase):
    def test_june_exports_carry_a_raw_timestamp(self) -> None:
        df = pd.DataFrame({"timestamp": [1.0], "pm_2_5": [2.0]})
        self.assertEqual(ingest._timestamp_column(df), "timestamp")

    def test_august_exports_carry_a_pre_rounded_timestamp(self) -> None:
        df = pd.DataFrame({"timestamp_rounded": [1.0], "pm_2_5": [2.0]})
        self.assertEqual(ingest._timestamp_column(df), "timestamp_rounded")


class AlignToGridTest(unittest.TestCase):
    def test_readings_in_the_same_minute_share_a_bucket(self) -> None:
        # 1748782836 is :36 past the minute, 1748782851 is :51 — the real sensor offset
        aligned = ingest.align_to_grid(_readings([1748782836, 1748782851], [2.0, 4.0]), 60)
        self.assertEqual(len(aligned), 1)

    def test_bucket_is_labelled_with_its_start(self) -> None:
        aligned = ingest.align_to_grid(_readings([1748782851], [2.0]), 60)
        self.assertEqual(aligned["timestamp_rounded"].iloc[0], 1748782800)

    def test_readings_sharing_a_bucket_are_averaged(self) -> None:
        aligned = ingest.align_to_grid(_readings([1748782836, 1748782851], [2.0, 4.0]), 60)
        self.assertAlmostEqual(aligned["pm_2_5"].iloc[0], 3.0)

    def test_readings_in_different_minutes_stay_apart(self) -> None:
        aligned = ingest.align_to_grid(_readings([1748782836, 1748782896], [2.0, 4.0]), 60)
        self.assertEqual(len(aligned), 2)

    def test_the_caller_s_frame_is_left_untouched(self) -> None:
        original = _readings([1748782836], [2.0])
        ingest.align_to_grid(original, 60)
        self.assertEqual(list(original.columns), ["timestamp", "pm_2_5"])


class MergeSensorsTest(unittest.TestCase):
    def test_shared_channels_are_suffixed_by_source(self) -> None:
        sen55 = ingest.align_to_grid(_readings([1748782836], [4.6]), 60)
        atmocube = ingest.align_to_grid(_readings([1748782851], [2.9]), 60)
        merged = ingest.merge_sensors(sen55, atmocube)
        self.assertIn("pm_2_5_sen55", merged.columns)
        self.assertIn("pm_2_5_atmo", merged.columns)

    def test_both_sensors_land_on_one_row_when_they_share_a_minute(self) -> None:
        sen55 = ingest.align_to_grid(_readings([1748782836], [4.6]), 60)
        atmocube = ingest.align_to_grid(_readings([1748782851], [2.9]), 60)
        merged = ingest.merge_sensors(sen55, atmocube)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged["pm_2_5_sen55"].iloc[0], 4.6)
        self.assertAlmostEqual(merged["pm_2_5_atmo"].iloc[0], 2.9)

    def test_a_minute_only_one_sensor_reported_is_kept_with_a_blank(self) -> None:
        sen55 = ingest.align_to_grid(_readings([1748782836, 1748782896], [4.6, 4.8]), 60)
        atmocube = ingest.align_to_grid(_readings([1748782851], [2.9]), 60)
        merged = ingest.merge_sensors(sen55, atmocube)
        self.assertEqual(len(merged), 2)
        self.assertTrue(np.isnan(merged["pm_2_5_atmo"].iloc[1]))

    def test_time_columns_come_first(self) -> None:
        sen55 = ingest.align_to_grid(_readings([1748782836], [4.6]), 60)
        atmocube = ingest.align_to_grid(_readings([1748782851], [2.9]), 60)
        merged = ingest.merge_sensors(sen55, atmocube)
        self.assertEqual(list(merged.columns)[:2], ["datetime", "timestamp_rounded"])