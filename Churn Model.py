import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

# ML Imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Deep Learning
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Recursive Training
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# %% Initialize Environment
drive.mount('/content/drive', force_remount=False)
ray.shutdown()  # Clean previous instances
ray.init(
    ignore_reinit_error=True,
    num_cpus=2,
    num_gpus=1,
    object_store_memory=2*1024*1024*1024  # 2GB limit for Colab
)

# Check GPU availability
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# %% Realistic Data Generation
def generate_realistic_data(samples=50000):  # Reduced for Colab memory
    np.random.seed(42)
    return pd.DataFrame({
        'tenure': np.random.poisson(24, samples),
        'monthly_charges': np.clip(np.random.normal(70, 15, samples), 20, 120),
        'total_charges': np.abs(np.random.normal(2500, 1000, samples)),
        'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'],
                                   samples, p=[0.6, 0.3, 0.1]),
        'online_security': np.random.choice([0, 1], samples, p=[0.65, 0.35]),
        'tech_support': np.random.choice([0, 1], samples, p=[0.6, 0.4]),
        'streaming_tv': np.random.choice([0, 1], samples, p=[0.55, 0.45]),
        'churn': np.random.choice([0, 1], samples, p=[0.75, 0.25])
    })

df = generate_realistic_data()

# %% Enhanced Preprocessing
def preprocess_data(df):
    # Convert categoricals
    df = pd.get_dummies(df, columns=['contract'], drop_first=True)

    # Feature engineering
    df['value_score'] = (df['total_charges'] / (df['tenure'] + 1)) * df['online_security']
    df['engagement_ratio'] = df['streaming_tv'] / (df['tenure'] + 1)

    # Split data
    X = df.drop('churn', axis=1)
    y = df['churn']

    # Improved SMOTE handling
    smote = SMOTE(sampling_strategy=0.67, random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X, y)

    # Robust scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    return X_scaled, y_res, scaler

X, y, scaler = preprocess_data(df)

# %% Advanced Hybrid Model
def build_enhanced_model(input_shape):
    inputs = Input(shape=(input_shape,))

    # Feature processing
    x = Dense(256, activation='swish')(inputs)
    x = BatchNormalization()(x)
    x = Dense(128, activation='swish')(x)
    x = BatchNormalization()(x)

    # Skip connection
    shortcut = Dense(64, activation='linear')(inputs)
    x = Concatenate()([x, shortcut])

    # Output layer
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(name='auc_roc', curve='ROC'),
            tf.keras.metrics.AUC(name='auc_pr', curve='PR')
        ]
    )
    return model

# %% Optimized Training Function
def train_model(config):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

    model = build_enhanced_model(X_train.shape[1])

    es = EarlyStopping(monitor='val_auc_pr', patience=7, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=config['batch_size'],
        callbacks=[es],
        verbose=0
    )

    tune.report(
        val_auc_pr=history.history['val_auc_pr'][-1],
        val_auc_roc=history.history['val_auc_roc'][-1],
        best_epoch=len(history.history['val_auc_pr'])
    )

# %% Hyperparameter Optimization (Fixed)
analysis = tune.run(
    train_model,
    config={
        'batch_size': tune.choice([512, 1024]),
        'lr': tune.loguniform(1e-4, 1e-3)
    },
    resources_per_trial={'cpu': 1, 'gpu': 0.25},  # Colab-optimized
    num_samples=8,  # Reduced for faster results
    scheduler=ASHAScheduler(
        metric='val_auc_pr',
        mode='max',
        max_t=100,
        grace_period=10,
        reduction_factor=2
    )
)

# %% Train Final Model
best_config = analysis.get_best_config(metric='val_auc_pr', mode='max')
final_model = build_enhanced_model(X.shape[1])
final_model.fit(
    X, y,
    epochs=150,
    batch_size=best_config['batch_size'],
    validation_split=0.2
)

# %% Active Learning System
class ChurnUpdater:
    def __init__(self, model, scaler, uncertainty_threshold=0.35):
        self.model = model
        self.scaler = scaler
        self.threshold = uncertainty_threshold
        self.X_pool = X.copy()
        self.y_pool = y.copy()

    def update(self, new_data):
        new_data_scaled = self.scaler.transform(new_data)
        probas = self.model.predict(new_data_scaled, verbose=0)
        uncertainty = np.abs(probas - 0.5).flatten()
        uncertain_mask = uncertainty < self.threshold

        self.X_pool = np.vstack([self.X_pool, new_data_scaled[uncertain_mask]])
        self.y_pool = np.concatenate([self.y_pool, np.zeros(np.sum(uncertain_mask))])

        self.model.fit(
            self.X_pool, self.y_pool,
            epochs=20,
            batch_size=best_config['batch_size'],
            validation_split=0.1,
            verbose=0
        )

# %% Concept Drift Monitor
class DriftDetector:
    def __init__(self, reference_data, threshold=0.01):
        self.reference = reference_data
        self.threshold = threshold

    def check_drift(self, new_data):
        from scipy.stats import wasserstein_distance
        distances = []
        for i in range(self.reference.shape[1]):
            distances.append(wasserstein_distance(self.reference[:,i], new_data[:,i]))
        return np.mean(distances) > self.threshold

# %% Save & Visualize
final_model.save('/content/drive/MyDrive/churn_model_prod.keras')

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.kdeplot(final_model.predict(X, verbose=0).flatten(), fill=True)
plt.title('Prediction Distribution')

plt.subplot(1, 2, 2)
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(y, final_model.predict(X, verbose=0).flatten())
plt.title('ROC Curve')
plt.show()
