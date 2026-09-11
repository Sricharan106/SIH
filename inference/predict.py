from pathlib import Path

import numpy as np
import torch

from models.temporal_transformer import TemporalTransformer

MODEL_PATH = Path(__file__).resolve().parents[1] / "saved_models" / "temporal_transformer.pt"


class NetOraclePredictor:
    def __init__(self, model_path=MODEL_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}. Run training/train_world_model.py first.")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.sequence_length = int(checkpoint["sequence_length"])
        self.feature_names = checkpoint["feature_names"]
        self.mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
        self.scale = np.asarray(checkpoint["normalization_scale"], dtype=np.float32)
        self.threshold = float(checkpoint["threshold"])
        self.model = TemporalTransformer(
            state_dim=int(checkpoint["state_dim"]), **checkpoint.get("model_config", {})
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def analyze(self, states):
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != len(self.feature_names) * 2:
            raise ValueError(f"Expected states with shape [windows, {len(self.feature_names) * 2}], received {states.shape}.")
        if len(states) < self.sequence_length:
            raise ValueError(f"At least {self.sequence_length} complete network windows are required for analysis.")
        sequences = np.asarray(
            [states[index : index + self.sequence_length] for index in range(len(states) - self.sequence_length + 1)],
            dtype=np.float32,
        )
        normalized = (sequences - self.mean) / self.scale
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(torch.tensor(normalized))).numpy()
        windows = [
            {"window": index + self.sequence_length - 1, "risk": float(probability), "status": "Suspicious" if probability >= self.threshold else "Normal"}
            for index, probability in enumerate(probabilities)
        ]
        suspicious = [window for window in windows if window["status"] == "Suspicious"]
        overall_risk = float(np.max(probabilities))
        return {
            "status": "Suspicious" if suspicious else "Normal",
            "risk": overall_risk,
            "threshold": self.threshold,
            "windows_analyzed": len(windows),
            "suspicious_windows": len(suspicious),
            "suspicious_percentage": len(suspicious) / len(windows) * 100,
            "windows": windows,
        }
