import os

from preprocessing.features import prepare_features
from preprocessing.load_data import load_network_data
from preprocessing.state_builder import build_network_states

from models.lightweight_detector import LightweightDetector

DATA_PATH = "data/sample_data.csv"

MODEL_PATH = "saved_models/lightweight_detector.pkl"


def main():

    os.makedirs("saved_models", exist_ok=True)

    print("Loading data...")

    df = load_network_data(DATA_PATH)

    print("Extracting features...")

    features = prepare_features(df)

    print("Building network states...")

    states = build_network_states(features, window_size=50)

    print("Training lightweight detector...")

    detector = LightweightDetector()

    detector.train(states)

    detector.save(MODEL_PATH)

    print("Saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
