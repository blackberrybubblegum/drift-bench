from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn


def build_windows(
    series: pd.Series, history: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a gridded series into input windows and the target that follows each one."""
    values = series.to_numpy(dtype=float)
    count = len(values) - history - horizon + 1
    if count <= 0:
        return np.empty((0, history)), np.empty(0)

    inputs = sliding_window_view(values, history)[:count]
    targets = values[history + horizon - 1 :]

    keep = ~np.isnan(inputs).any(axis=1) & ~np.isnan(targets)
    return inputs[keep], targets[keep]


def split_by_time(
    inputs: np.ndarray, targets: np.ndarray, train_share: float, calibration_share: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Cut the windows into train, calibration and test blocks in chronological order."""
    train_end = int(len(inputs) * train_share)
    calibration_end = train_end + int(len(inputs) * calibration_share)
    return {
        "train": (inputs[:train_end], targets[:train_end]),
        "calibration": (inputs[train_end:calibration_end], targets[train_end:calibration_end]),
        "test": (inputs[calibration_end:], targets[calibration_end:]),
    }


def persistence(inputs: np.ndarray) -> np.ndarray:
    """Forecast the last observed value in each window."""
    return inputs[:, -1]


def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Average size of the forecast error, in the units of the target."""
    return float(np.abs(actual - predicted).mean())


class _GRUForecaster(nn.Module):
    """Single-layer GRU that normalises with the training statistics it was built from."""

    def __init__(self, hidden: int, mean: float, sd: float) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scaled = (inputs.unsqueeze(-1) - self.mean) / self.sd
        output, _ = self.gru(scaled)
        return self.head(output[:, -1, :]).squeeze(-1) * self.sd + self.mean


def train_gru(
    inputs: np.ndarray,
    targets: np.ndarray,
    hidden: int = 32,
    epochs: int = 30,
    batch_size: int = 128,
    seed: int = 0,
) -> nn.Module:
    """Fit a small GRU on the training block and return it in evaluation mode."""
    torch.manual_seed(seed)
    model = _GRUForecaster(hidden, float(inputs.mean()), float(inputs.std()))
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.L1Loss()

    x = torch.tensor(inputs, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=True
    )

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimiser.zero_grad()
            loss_function(model(batch_x), batch_y).backward()
            optimiser.step()

    model.eval()
    return model


def predict(model: nn.Module, inputs: np.ndarray) -> np.ndarray:
    """Run a trained model over a block of windows without tracking gradients."""
    if len(inputs) == 0:
        return np.empty(0)
    with torch.no_grad():
        return model(torch.tensor(inputs, dtype=torch.float32)).numpy()