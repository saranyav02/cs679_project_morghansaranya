# model_baseline.py
#
# Fully-connected MLP baseline for the MDM4 project.
# Architecture:
#   Input (D)
#     → BatchNorm1d(D)
#     → Linear(D → 128) + ReLU + Dropout(0.3)
#     → Linear(128 → 64) + ReLU + Dropout(0.3)
#     → Linear(64 → 32) + ReLU + Dropout(0.3)
#     → Linear(32 → 1) (logit)
#
# Training is handled in run_model_baseline.py (this file only defines the model
# and a small helper to construct it + optimizer).

from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch import optim


class BaselineMLP(nn.Module):
    """
    Simple fully-connected MLP baseline (no pathway structure).

    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (25, 20, 12),
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []

        # Can do input batch norm to stabilize training with many features.
        layers.append(nn.BatchNorm1d(input_dim))

        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = h

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        """
        out = self.feature_extractor(x)
        logits = self.classifier(out).squeeze(-1)
        return logits


def build_baseline_model(
    input_dim: int,
    lr: float = 1e-5,
    weight_decay: float = 1e-5,
    device: torch.device | str | None = None,
) -> Tuple[nn.Module, optim.Optimizer]:
    """
    Note: can call this from run_model_baseline.py using
        model, optimizer = build_baseline_model(input_dim=X_train.shape[1], device=device)
    """
    model = BaselineMLP(input_dim=input_dim)
    if device is not None:
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer


if __name__ == "__main__":
    # Make sure shapes are good
    D = 13802
    model, _ = build_baseline_model(input_dim=D)
    x = torch.randn(4, D)
    logits = model(x)
    print("Logits shape:", logits.shape)  
