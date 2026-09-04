import numpy as np
import pandas as pd


def find_column(df, possible_names):
    """
    Finds the first matching column from possible dataset column names.
    """

    columns_lower = {col.lower().strip(): col for col in df.columns}

    for name in possible_names:
        if name.lower() in columns_lower:
            return columns_lower[name.lower()]

    return None


def prepare_features(df):
    """
    Convert raw network traffic into numerical ML features.
    """

    result = pd.DataFrame(index=df.index)

    # -----------------------------
    # FLOW LENGTH / BYTES
    # -----------------------------

    bytes_col = find_column(
        df, ["Flow Bytes/s", "Total Length of Fwd Packets", "bytes", "total bytes"]
    )

    if bytes_col:
        result["bytes"] = pd.to_numeric(df[bytes_col], errors="coerce").fillna(0)

    # -----------------------------
    # FLOW DURATION
    # -----------------------------

    duration_col = find_column(df, ["Flow Duration", "duration", "flow duration"])

    if duration_col:
        result["flow_duration"] = pd.to_numeric(
            df[duration_col], errors="coerce"
        ).fillna(0)

    # -----------------------------
    # PACKETS
    # -----------------------------

    packet_col = find_column(
        df, ["Total Fwd Packets", "Total Backward Packets", "packets", "packet count"]
    )

    if packet_col:
        result["packets"] = pd.to_numeric(df[packet_col], errors="coerce").fillna(0)

    # -----------------------------
    # PORT
    # -----------------------------

    port_col = find_column(df, ["Destination Port", "dst_port", "destination port"])

    if port_col:
        result["destination_port"] = pd.to_numeric(
            df[port_col], errors="coerce"
        ).fillna(0)

    # -----------------------------
    # SYN FLAGS
    # -----------------------------

    syn_col = find_column(df, ["SYN Flag Count", "syn_count", "syn"])

    if syn_col:
        result["syn_count"] = pd.to_numeric(df[syn_col], errors="coerce").fillna(0)

    # -----------------------------
    # ACK FLAGS
    # -----------------------------

    ack_col = find_column(df, ["ACK Flag Count", "ack_count", "ack"])

    if ack_col:
        result["ack_count"] = pd.to_numeric(df[ack_col], errors="coerce").fillna(0)

    # -----------------------------
    # PACKET SIZE
    # -----------------------------

    packet_size_col = find_column(
        df, ["Average Packet Size", "Avg Packet Size", "packet size"]
    )

    if packet_size_col:
        result["avg_packet_size"] = pd.to_numeric(
            df[packet_size_col], errors="coerce"
        ).fillna(0)

    # -----------------------------
    # INTER ARRIVAL TIME
    # -----------------------------

    iat_col = find_column(df, ["Flow IAT Mean", "iat_mean", "IAT Mean"])

    if iat_col:
        result["iat_mean"] = pd.to_numeric(df[iat_col], errors="coerce").fillna(0)

    # Ensure numeric values
    result = result.fillna(0)

    return result
