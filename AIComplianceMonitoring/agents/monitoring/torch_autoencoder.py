"""PyTorch Autoencoder model used by the MonitoringAgent for anomaly detection.

This class is adapted from the proof-of-concept Autoencoder implemented in
`test_anomaly_detection_with_feedback.py` and adds:

* Optional mixed-precision (AMP) training for faster GPU utilisation.
* Automatic threshold computation (percentile of reconstruction error).
* Convenience `train`, `predict`, `save`, and `load` helpers so the parent
  `AnomalyDetectionModule` can interact with it similarly to scikit-learn
  estimators.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["AutoencoderModel"]


class _AutoencoderNet(nn.Module):
    """Simple fully-connected autoencoder."""

    def __init__(self, input_dim: int):
        super().__init__()
        dim1 = int(input_dim * 0.75)
        dim2 = int(input_dim * 0.5)
        dim3 = max(4, int(input_dim * 0.33))
        bottleneck = max(2, int(input_dim * 0.25))

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dim1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Linear(dim2, dim3),
            nn.ReLU(),
            nn.Linear(dim3, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, dim3),
            nn.ReLU(),
            nn.Linear(dim3, dim2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim2, dim1),
            nn.ReLU(),
            nn.Linear(dim1, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        encoded = self.encoder(x)
        return self.decoder(encoded)


class AutoencoderModel:
    """High-level wrapper around the neural net with training helpers."""

    def __init__(self, input_dim: int, threshold_percentile: int = 95):
        self.input_dim = input_dim
        self.threshold_percentile = threshold_percentile
        self.threshold: Optional[float] = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: _AutoencoderNet = _AutoencoderNet(input_dim).to(self.device)

        # Reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

    # ---------------------------------------------------------------------
    # Training helpers
    # ---------------------------------------------------------------------
    def train(self, X: np.ndarray, *, epochs: int = 20, batch_size: int = 64) -> None:
        """Fit the model and compute the reconstruction-error threshold."""
        if hasattr(X, "values"):
            X = X.values  # type: ignore[attr-defined]
        X = X.astype("float32")

        tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        loader = DataLoader(TensorDataset(tensor, tensor), batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        scaler = GradScaler(enabled=torch.cuda.is_available())

        self.model.train()
        for _ in range(epochs):
            for inputs, targets in loader:
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=torch.cuda.is_available()):
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # Threshold calculation
        self.model.eval()
        with torch.no_grad(), autocast(enabled=torch.cuda.is_available()):
            recon = self.model(tensor)
            mse = torch.mean((tensor - recon) ** 2, dim=1).cpu().numpy()
        self.threshold = np.percentile(mse, self.threshold_percentile)

    # ------------------------------------------------------------------
    # Compatibility wrapper (Keras-like API)
    # ------------------------------------------------------------------
    def fit(self, X, y=None, epochs: int = 30, batch_size: int = 64, **kwargs):
        """Alias to allow code written for Keras models to call `.fit()`."""
        self.train(X, epochs=epochs, batch_size=batch_size)
        return self

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return reconstruction error for each sample."""
        if hasattr(X, "values"):
            X = X.values  # type: ignore[attr-defined]
        X = X.astype("float32")
        tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad(), autocast(enabled=torch.cuda.is_available()):
            recon = self.model(tensor).cpu().numpy()
        return np.mean((X - recon) ** 2, axis=1)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "input_dim": self.input_dim,
            "threshold": self.threshold,
            "threshold_percentile": self.threshold_percentile,
        }, path)

    def load(self, path: Path | str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim = checkpoint["input_dim"]
        self.threshold = checkpoint["threshold"]
        self.threshold_percentile = checkpoint["threshold_percentile"]
        self.model = _AutoencoderNet(self.input_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # noqa: D401
        return (
            f"AutoencoderModel(input_dim={self.input_dim}, "
            f"threshold={self.threshold:.4f} if self.threshold is not None else None)"
        )
