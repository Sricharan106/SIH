import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.temporal_transformer import TemporalTransformer
from preprocressing.features import extract_labels, prepare_features
from preprocressing.load_data import load_network_data
from preprocressing.state_builder import build_network_states, create_sequences

DATA_PATH = Path("data/UNSW-NB15_4.csv")
MODEL_PATH = Path("saved_models/temporal_transformer.pt")
WINDOW_SIZE = 50
SEQUENCE_LENGTH = 8
EPOCHS = 8
BATCH_SIZE = 256


def _best_threshold(labels, probabilities):
    candidates = np.linspace(0.2, 0.8, 25)
    scores = [f1_score(labels, probabilities >= value, zero_division=0) for value in candidates]
    return float(candidates[int(np.argmax(scores))])


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    frame = load_network_data(DATA_PATH)
    labels = extract_labels(frame)
    if labels is None or len(np.unique(labels)) < 2:
        raise ValueError("Training requires a label column containing both normal and attack rows.")

    features = prepare_features(frame)
    states, state_labels = build_network_states(features, WINDOW_SIZE, labels)
    sequences, targets = create_sequences(states, state_labels, SEQUENCE_LENGTH)
    if len(sequences) < 10:
        raise ValueError("Not enough complete temporal sequences for training.")

    split = max(1, int(len(sequences) * 0.8))
    train_x, validation_x = sequences[:split], sequences[split:]
    train_y, validation_y = targets[:split], targets[split:]
    if len(np.unique(train_y)) < 2:
        raise ValueError("The chronological training split contains only one class.")

    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    scale = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_x = (train_x - mean) / scale
    validation_x = (validation_x - mean) / scale

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalTransformer(state_dim=sequences.shape[-1]).to(device)
    positives = max(float(train_y.sum()), 1.0)
    negatives = max(float(len(train_y) - train_y.sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
        print(f"Epoch {epoch + 1}/{EPOCHS} | loss={total_loss / len(train_x):.4f}")

    model.eval()
    with torch.no_grad():
        validation_logits = model(torch.tensor(validation_x, dtype=torch.float32, device=device))
        probabilities = torch.sigmoid(validation_logits).cpu().numpy()
    threshold = _best_threshold(validation_y, probabilities)
    auc = roc_auc_score(validation_y, probabilities) if len(np.unique(validation_y)) > 1 else float("nan")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "artifact_version": 1,
            "model_state": model.cpu().state_dict(),
            "state_dim": int(sequences.shape[-1]),
            "sequence_length": SEQUENCE_LENGTH,
            "feature_names": list(features.columns),
            "normalization_mean": mean.tolist(),
            "normalization_scale": scale.tolist(),
            "threshold": threshold,
            "model_config": {"d_model": 64, "nhead": 4, "num_layers": 2, "dropout": 0.1},
        },
        MODEL_PATH,
    )
    print(f"Saved {MODEL_PATH} | validation_auc={auc:.4f} | threshold={threshold:.2f}")


if __name__ == "__main__":
    main()
