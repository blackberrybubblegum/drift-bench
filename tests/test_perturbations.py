from __future__ import annotations

import unittest

import numpy as np

import perturbations


class AddNoiseTest(unittest.TestCase):
    def test_zero_sigma_leaves_the_input_unchanged(self) -> None:
        inputs = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(perturbations.add_noise(inputs, 0.0), inputs)

    def test_same_seed_gives_the_same_corruption(self) -> None:
        inputs = np.ones((4, 10))
        first = perturbations.add_noise(inputs, 0.5, seed=7)
        second = perturbations.add_noise(inputs, 0.5, seed=7)
        np.testing.assert_allclose(first, second)

    def test_noise_has_roughly_the_requested_spread(self) -> None:
        inputs = np.zeros((2000, 20))
        corrupted = perturbations.add_noise(inputs, 0.5, seed=1)
        self.assertAlmostEqual(corrupted.std(), 0.5, places=2)


class OffsetAndScaleTest(unittest.TestCase):
    def test_offset_shifts_the_mean_by_exactly_the_offset(self) -> None:
        inputs = np.array([[1.0, 2.0, 3.0]])
        shifted = perturbations.add_offset(inputs, 2.5)
        self.assertAlmostEqual(shifted.mean() - inputs.mean(), 2.5)

    def test_scale_multiplies_every_reading(self) -> None:
        inputs = np.array([[1.0, 2.0, 4.0]])
        np.testing.assert_allclose(
            perturbations.scale(inputs, 2.0), np.array([[2.0, 4.0, 8.0]])
        )


class DropBlocksTest(unittest.TestCase):
    def test_zero_blocks_leaves_the_input_unchanged(self) -> None:
        inputs = np.arange(12, dtype=float).reshape(3, 4)
        np.testing.assert_allclose(
            perturbations.drop_blocks(inputs, np.array([2]), blocks=0), inputs
        )

    def test_output_never_contains_missing_values(self) -> None:
        inputs = np.arange(600, dtype=float).reshape(10, 60)
        corrupted = perturbations.drop_blocks(inputs, np.array([1, 5, 16]), blocks=3, seed=3)
        self.assertFalse(np.isnan(corrupted).any())

    def test_a_blanked_run_repeats_the_reading_before_it(self) -> None:
        inputs = np.array([[10.0, 20.0, 30.0, 40.0, 50.0]])
        corrupted = perturbations.drop_blocks(inputs, np.array([2]), blocks=1, seed=0)
        repeats = corrupted[0, 1:] == corrupted[0, :-1]
        self.assertTrue(repeats.any())

    def test_shape_is_preserved(self) -> None:
        inputs = np.zeros((7, 60))
        corrupted = perturbations.drop_blocks(inputs, np.array([3]), blocks=2, seed=0)
        self.assertEqual(corrupted.shape, inputs.shape)