import numpy as np
import pandas as pd


def build_network_states(features, window_size=50):
    """
    Convert traffic features into aggregated network states.

    window_size:
        Number of flows per network state.
    """

    states = []

    total_rows = len(features)

    for start in range(0, total_rows, window_size):
        end = start + window_size

        window = features.iloc[start:end]

        if len(window) < window_size // 2:
            continue

        state = []

        # Mean of each feature
        means = window.mean().values

        # Standard deviation
        stds = window.std().fillna(0).values

        # Combine statistics
        state = np.concatenate([means, stds])

        states.append(state)

    return np.array(states)


def create_sequences(states, sequence_length=10):
    """
    Creates:

    S1 S2 S3 ... S10 -> S11
    """

    X = []
    y = []

    for i in range(len(states) - sequence_length):
        X.append(states[i : i + sequence_length])

        y.append(states[i + sequence_length])

    return np.array(X), np.array(y)
