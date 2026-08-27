from __future__ import annotations

import numpy as np


def add_noise(inputs: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """Add zero-mean Gaussian noise of the given standard deviation to every reading."""
    generator = np.random.default_rng(seed)
    return inputs + generator.normal(0.0, sigma, size=inputs.shape)


def add_offset(inputs: np.ndarray, offset: float) -> np.ndarray:
    """Shift every reading by a constant, the additive degradation model."""
    return inputs + offset


def scale(inputs: np.ndarray, factor: float) -> np.ndarray:
    """Multiply every reading by a constant, the multiplicative degradation model."""
    return inputs * factor


def _forward_fill(inputs: np.ndarray) -> np.ndarray:
    """Carry the last reading forward across blanks, as a live system with no future would."""
    present = ~np.isnan(inputs)
    index = np.where(present, np.arange(inputs.shape[1]), 0)
    np.maximum.accumulate(index, axis=1, out=index)
    filled = np.take_along_axis(inputs, index, axis=1)

    leading = np.isnan(filled)
    if leading.any():
        # a blank at the very start has nothing behind it, so reach forward instead
        first = inputs[np.arange(len(inputs)), np.argmax(present, axis=1)]
        filled = np.where(leading, first[:, None], filled)
    return filled


def drop_blocks(
    inputs: np.ndarray, lengths: np.ndarray, blocks: int, seed: int = 0
) -> np.ndarray:
    """Blank contiguous runs of readings and refill them the way a live pipeline would."""
    if blocks == 0:
        return inputs.copy()

    generator = np.random.default_rng(seed)
    corrupted = inputs.copy().astype(float)
    width = corrupted.shape[1]

    for row in range(len(corrupted)):
        for _ in range(blocks):
            length = min(int(generator.choice(lengths)), width)
            start = int(generator.integers(0, width - length + 1))
            corrupted[row, start : start + length] = np.nan

    return _forward_fill(corrupted)