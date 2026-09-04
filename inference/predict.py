import numpy as np
import torch

from inference.simulator import simulate_future
from models.controller import AdaptiveController
from models.lightweight_detector import LightweightDetector
from models.world_model import WorldModel
from utils.mitre_mapper import get_stage_name

LIGHT_MODEL_PATH = "saved_models/lightweight_detector.pkl"

WORLD_MODEL_PATH = "saved_models/world_model.pt"


class NetOraclePredictor:
    def __init__(self):

        # Lightweight model
        self.light_model = LightweightDetector()

        self.light_model.load(LIGHT_MODEL_PATH)

        # Controller
        self.controller = AdaptiveController()

        # Heavy model
        checkpoint = torch.load(WORLD_MODEL_PATH, map_location="cpu")

        input_size = checkpoint["input_size"]

        self.world_model = WorldModel(input_size=input_size)

        self.world_model.load_state_dict(checkpoint["model_state"])

        self.world_model.eval()

    def analyze(self, states, simulation_steps=3):

        results = []

        for i, state in enumerate(states):
            anomaly_score = self.light_model.predict(state)

            mode = self.controller.update(anomaly_score)

            result = {"window": i, "anomaly_score": anomaly_score, "mode": mode}

            # Activate World Model only
            # during deep analysis

            if mode == "DEEP_ANALYSIS" and i >= 9:
                sequence = states[i - 9 : i + 1]

                future = simulate_future(
                    self.world_model, sequence, steps=simulation_steps
                )

                # Convert stage IDs to names

                for prediction in future:
                    prediction["stage"] = get_stage_name(prediction["stage"])

                result["future_predictions"] = future

            results.append(result)

        return results
