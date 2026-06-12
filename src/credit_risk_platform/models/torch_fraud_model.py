"""Optional PyTorch fraud model for experimentation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from credit_risk_platform.features import FRAUD_FEATURES, require_columns


@dataclass
class TorchFraudResult:
    metrics: dict[str, float | int | str]


def train_torch_fraud_model(
    frame: pd.DataFrame,
    target: str = "fraud_label",
    epochs: int = 10,
    random_seed: int = 42,
) -> TorchFraudResult:
    """Train a compact PyTorch neural network for fraud detection."""
    try:
        import numpy as np
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError("PyTorch is required. Install requirements-full.txt.") from exc

    require_columns(frame, FRAUD_FEATURES + [target])
    torch.manual_seed(random_seed)

    x_train, x_test, y_train, y_test = train_test_split(
        frame[FRAUD_FEATURES],
        frame[target],
        test_size=0.25,
        random_state=random_seed,
        stratify=frame[target],
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype("float32")
    x_test_scaled = scaler.transform(x_test).astype("float32")
    y_train_array = y_train.to_numpy().astype("float32").reshape(-1, 1)

    dataset = TensorDataset(torch.from_numpy(x_train_scaled), torch.from_numpy(y_train_array))
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = nn.Sequential(
        nn.Linear(len(FRAUD_FEATURES), 32),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test_scaled))
        probabilities = torch.sigmoid(logits).numpy().flatten()

    return TorchFraudResult(
        metrics={
            "model_type": "PyTorchSequentialFraudNet",
            "epochs": int(epochs),
            "rows_train": int(len(x_train)),
            "rows_test": int(len(x_test)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "average_precision": float(average_precision_score(y_test, probabilities)),
            "positive_rate_test": float(np.mean(y_test)),
        }
    )
