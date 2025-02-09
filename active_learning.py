import numpy as np
import tensorflow as tf
from preprocess import preprocess_data
from train import build_model

class ChurnUpdater:
    def __init__(self, model, scaler, threshold=0.35):
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.X_pool, self.y_pool, _ = preprocess_data()

    def update(self, new_data):
        new_data_scaled = self.scaler.transform(new_data)
        probas = self.model.predict(new_data_scaled, verbose=0)
        uncertainty = np.abs(probas - 0.5).flatten()
        uncertain_mask = uncertainty < self.threshold

        self.X_pool = np.vstack([self.X_pool, new_data_scaled[uncertain_mask]])
        self.y_pool = np.concatenate([self.y_pool, np.zeros(np.sum(uncertain_mask))])

        self.model.fit(self.X_pool, self.y_pool, epochs=20, batch_size=512, validation_split=0.1, verbose=0)

if __name__ == "__main__":
    model = tf.keras.models.load_model("models/churn_model.keras")
    _, _, scaler = preprocess_data()
    updater = ChurnUpdater(model, scaler)
    print("Active Learning Module Ready")
