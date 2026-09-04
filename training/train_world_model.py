import os

import numpy as np
import torch
import torch.nn as nn
from preprocessing.features import prepare_features
from preprocessing.load_data import load_network_data
from preprocessing.state_builder import build_network_states, create_sequences

from models.world_model import WorldModel

DATA_PATH = "data/sample_data.csv"

MODEL_PATH = "saved_models/world_model.pt"

SEQUENCE_LENGTH = 10


def main():

    os.makedirs("saved_models", exist_ok=True)

    print("Loading dataset...")

    df = load_network_data(DATA_PATH)

    features = prepare_features(df)

    states = build_network_states(features, window_size=50)

    X, y_state = create_sequences(states, SEQUENCE_LENGTH)

    if len(X) == 0:
        raise ValueError("Not enough data to create sequences.")

    input_size = X.shape[2]

    # ------------------------------------------------
    # Temporary prototype labels
    # ------------------------------------------------

    # In a real implementation these should come
    # from dataset attack timeline labels.

    attack_labels = np.zeros(len(X))

    stage_labels = np.zeros(len(X), dtype=int)

    # ------------------------------------------------

    X_tensor = torch.tensor(X, dtype=torch.float32)

    y_state_tensor = torch.tensor(y_state, dtype=torch.float32)

    attack_tensor = torch.tensor(attack_labels, dtype=torch.float32).unsqueeze(1)

    stage_tensor = torch.tensor(stage_labels, dtype=torch.long)

    model = WorldModel(input_size=input_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    state_loss_fn = nn.MSELoss()

    attack_loss_fn = nn.BCELoss()

    stage_loss_fn = nn.CrossEntropyLoss()

    epochs = 30

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()

        (predicted_state, attack_probability, stage_logits, attention) = model(X_tensor)

        # Multi-task losses

        state_loss = state_loss_fn(predicted_state, y_state_tensor)

        attack_loss = attack_loss_fn(attack_probability, attack_tensor)

        stage_loss = stage_loss_fn(stage_logits, stage_tensor)

        total_loss = state_loss + 0.5 * attack_loss + 0.5 * stage_loss

        total_loss.backward()

        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss.item():.4f}")

    torch.save(
        {"model_state": model.state_dict(), "input_size": input_size}, MODEL_PATH
    )

    print("World Model saved!")


if __name__ == "__main__":
    main()
