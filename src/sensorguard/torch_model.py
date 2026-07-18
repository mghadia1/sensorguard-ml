"""Optional PyTorch multilayer-perceptron comparison on the same data split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .data import DatasetSplits, feature_target
from .modeling import binary_metrics, make_preprocessor, select_threshold


def train_torch_comparison(
    splits: DatasetSplits,
    output_dir: str | Path,
    *,
    random_state: int = 42,
    epochs: int = 40,
) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise RuntimeError("install SensorGuard with the 'torch' optional dependency") from error

    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    preprocessor = make_preprocessor(dense=True)
    train_features, train_labels = feature_target(splits.train)
    validation_features, validation_labels = feature_target(splits.validation)
    test_features, test_labels = feature_target(splits.test)
    train_array = np.asarray(preprocessor.fit_transform(train_features), dtype=np.float32)
    validation_array = np.asarray(preprocessor.transform(validation_features), dtype=np.float32)
    test_array = np.asarray(preprocessor.transform(test_features), dtype=np.float32)
    train_target = train_labels.to_numpy(dtype=np.float32).reshape(-1, 1)

    generator = torch.Generator().manual_seed(random_state)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_array), torch.from_numpy(train_target)),
        batch_size=128,
        shuffle=True,
        generator=generator,
    )
    hidden_width = 24
    model = nn.Sequential(
        nn.Linear(train_array.shape[1], hidden_width),
        nn.ReLU(),
        nn.Dropout(p=0.15),
        nn.Linear(hidden_width, 1),
    )
    positive_count = float(train_target.sum())
    negative_count = float(len(train_target) - positive_count)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((negative_count / positive_count,), dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        batch_losses: list[float] = []
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = loss_function(logits, batch_labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(batch_losses)))

    model.eval()
    with torch.no_grad():
        validation_probabilities = torch.sigmoid(model(torch.from_numpy(validation_array))).numpy().ravel()
        test_probabilities = torch.sigmoid(model(torch.from_numpy(test_array))).numpy().ravel()
    threshold = select_threshold(validation_labels, validation_probabilities)
    validation_metrics = binary_metrics(
        validation_labels,
        validation_probabilities,
        threshold=threshold,
    )
    test_metrics = binary_metrics(test_labels, test_probabilities, threshold=threshold)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "torch_model.pt")
    joblib.dump(preprocessor, output / "torch_preprocessor.joblib")
    report = {
        "architecture": {
            "input_width": int(train_array.shape[1]),
            "hidden_width": hidden_width,
            "output_width": 1,
            "activation": "ReLU",
            "dropout": 0.15,
        },
        "epochs": epochs,
        "initial_training_loss": losses[0],
        "final_training_loss": losses[-1],
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "note": "Test metrics are a comparison, not the classical model-selection result.",
    }
    (output / "torch_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

