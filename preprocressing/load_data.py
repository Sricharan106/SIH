import pandas as pd


UNSW_NB15_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit",
    "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label",
]


def _looks_like_header(columns):
    known = {"srcip", "sport", "dstip", "dsport", "dur", "sbytes", "label"}
    return any(str(column).strip().lower() in known for column in columns)


def load_network_data(file_path):
    """Load headered or headerless flow CSV data and normalize its columns."""
    frame = pd.read_csv(file_path)
    if len(frame.columns) == len(UNSW_NB15_COLUMNS) and not _looks_like_header(frame.columns):
        frame = pd.read_csv(file_path, header=None, names=UNSW_NB15_COLUMNS)
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.replace([float("inf"), float("-inf")], pd.NA)
    return frame.dropna(how="all").reset_index(drop=True)
