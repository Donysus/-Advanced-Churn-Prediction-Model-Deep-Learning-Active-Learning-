import numpy as np
from scipy.stats import wasserstein_distance
from preprocess import preprocess_data

class DriftDetector:
    def __init__(self, reference_data, threshold=0.01):
        self.reference = reference_data
        self.threshold = threshold

    def check_drift(self, new_data):
        distances = [wasserstein_distance(self.reference[:, i], new_data[:, i]) for i in range(self.reference.shape[1])]
        return np.mean(distances) > self.threshold

if __name__ == "__main__":
    X, _, _ = preprocess_data()
    detector = DriftDetector(X)
    print("Drift Detector Ready")
