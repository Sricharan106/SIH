MITRE_STAGES = {
    0: "Normal",
    1: "Reconnaissance",
    2: "Initial Access",
    3: "Lateral Movement",
    4: "Command & Control",
    5: "Exfiltration",
}


def get_stage_name(stage_id):

    return MITRE_STAGES.get(int(stage_id), "Unknown")
