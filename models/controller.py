class AdaptiveController:
    def __init__(
        self, watch_threshold=0.45, deep_threshold=0.55, consecutive_required=2
    ):

        self.watch_threshold = watch_threshold
        self.deep_threshold = deep_threshold

        self.consecutive_required = consecutive_required

        self.suspicious_count = 0

        self.mode = "NORMAL"

    def update(self, anomaly_score):

        # Normal traffic
        if anomaly_score < self.watch_threshold:
            self.suspicious_count = 0

            self.mode = "NORMAL"

        # Watch mode
        elif anomaly_score < self.deep_threshold:
            self.suspicious_count += 1

            self.mode = "WATCH"

        # High suspicion
        else:
            self.suspicious_count += 1

            if self.suspicious_count >= self.consecutive_required:
                self.mode = "DEEP_ANALYSIS"

            else:
                self.mode = "WATCH"

        return self.mode
