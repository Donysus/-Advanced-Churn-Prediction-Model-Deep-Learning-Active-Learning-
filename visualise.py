import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import RocCurveDisplay
from preprocess import preprocess_data

X, y, _ = preprocess_data()
model = tf.keras.models.load_model("models/churn_model.keras")

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.kdeplot(model.predict(X, verbose=0).flatten(), fill=True)
plt.title('Prediction Distribution')

plt.subplot(1, 2, 2)
RocCurveDisplay.from_predictions(y, model.predict(X, verbose=0).flatten())
plt.title('ROC Curve')
plt.show()
