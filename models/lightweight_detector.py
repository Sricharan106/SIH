import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class LightweightDetector:
    def __init__(self):
        self.scaler = StandardScaler()

        self.model = IsolationForest(
            n_estimators=100, contamination=0.05, random_state=42
        )

    def train(self, states):

        scaled_states = self.scaler.fit_transform(states)

        self.model.fit(scaled_states)

    def get_anomaly_score(self, state):

        state = state.reshape(1, -1)

        scaled_state = self.scaler.transform(state)

        # Higher = more suspicious
        raw_score = -self.model.score_samples(scaled_state)[0]

        # Convert roughly into 0-1
        anomaly_score = 1 / (1 + pow(2.71828, -raw_score))

        return float(anomaly_score)

    def predict(self, state):

        score = self.get_anomaly_score(state)

        return score

    def save(self, path):

        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    def load(self, path):

        data = joblib.load(path)

        self.model = data["model"]
        self.scaler = data["scaler"]
