import numpy as np


def build_network_states(features, window_size=10, labels=None):
    """Aggregate flows into fixed-width states and optionally aligned labels."""
    states = []
    state_labels = []
    for start in range(0, len(features), window_size):
        window = features.iloc[start : start + window_size]
        if len(window) < window_size:
            continue
        states.append(np.concatenate([window.mean().to_numpy(), window.std().fillna(0).to_numpy()]))
        if labels is not None:
            state_labels.append(float(np.max(labels[start : start + window_size])))
    states = np.asarray(states, dtype=np.float32)
    if labels is None:
        return states
    return states, np.asarray(state_labels, dtype=np.float32)


def create_sequences(states, labels=None, sequence_length=3):
    """Create rolling state sequences and labels for the final state."""
    if len(states) < sequence_length:
        empty_x = np.empty((0, sequence_length, states.shape[1]), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.float32)
        return empty_x, empty_y
    sequences = np.asarray(
        [states[index : index + sequence_length] for index in range(len(states) - sequence_length + 1)],
        dtype=np.float32,
    )
    if labels is None:
        return sequences, None
    targets = np.asarray(labels[sequence_length - 1 :], dtype=np.float32)
    return sequences, targets
