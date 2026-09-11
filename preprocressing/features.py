import pandas as pd


FEATURE_ALIASES = {
    "bytes": ["Flow Bytes/s", "Total Length of Fwd Packets", "bytes", "total bytes", "sbytes", "dbytes"],
    "flow_duration": ["Flow Duration", "duration", "flow duration", "dur"],
    "packets": ["Total Fwd Packets", "Total Backward Packets", "packets", "packet count", "spkts", "dpkts"],
    "destination_port": ["Destination Port", "dst_port", "destination port", "dsport"],
    "syn_count": ["SYN Flag Count", "syn_count", "synack", "syn"],
    "ack_count": ["ACK Flag Count", "ack_count", "ackdat", "ack"],
    "avg_packet_size": ["Average Packet Size", "Avg Packet Size", "packet size", "smeansz"],
    "iat_mean": ["Flow IAT Mean", "iat_mean", "IAT Mean", "sintpkt"],
}


def _find_column(frame, aliases):
    columns = {str(column).strip().lower(): column for column in frame.columns}
    for alias in aliases:
        if alias.lower() in columns:
            return columns[alias.lower()]
    return None


def prepare_features(frame):
    """Return a fixed-width numeric feature frame for supported flow CSVs."""
    result = pd.DataFrame(index=frame.index)
    matched = 0
    for feature_name, aliases in FEATURE_ALIASES.items():
        source = _find_column(frame, aliases)
        if source is None:
            result[feature_name] = 0.0
        else:
            matched += 1
            result[feature_name] = pd.to_numeric(frame[source], errors="coerce").fillna(0.0)
    if matched == 0:
        raise ValueError("CSV does not contain supported network-flow columns such as bytes, duration, packets, or destination port.")
    return result.astype("float32")


def extract_labels(frame):
    """Return binary attack labels when a supported label column is present."""
    label_column = _find_column(frame, ["label", "attack", "is_attack", "malicious"])
    if label_column is None:
        return None
    values = pd.to_numeric(frame[label_column], errors="coerce").fillna(0)
    return (values > 0).astype("float32").to_numpy()
