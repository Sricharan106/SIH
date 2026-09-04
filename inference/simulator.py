import numpy as np
import torch


def simulate_future(model, initial_sequence, steps=3):

    model.eval()

    sequence = initial_sequence.copy()

    predictions = []

    for step in range(steps):
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            (next_state, attack_probability, stage_logits, attention) = model(x)

        predicted_state = next_state.squeeze(0).numpy()

        risk = float(attack_probability.item())

        stage = int(torch.argmax(stage_logits, dim=1).item())

        predictions.append(
            {
                "step": step + 1,
                "attack_probability": risk,
                "stage": stage,
                "predicted_state": predicted_state.tolist(),
                "attention": attention.squeeze().numpy().tolist(),
            }
        )

        # Remove oldest state
        sequence = np.vstack([sequence[1:], predicted_state])

    return predictions
