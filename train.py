import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from preprocess import preprocess_data

X, y, scaler = preprocess_data()

def build_model(input_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(256, activation='swish')(inputs)
    x = BatchNormalization()(x)
    x = Dense(128, activation='swish')(x)
    x = BatchNormalization()(x)
    shortcut = Dense(64, activation='linear')(inputs)
    x = Concatenate()([x, shortcut])
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc_roc', curve='ROC'),
                           tf.keras.metrics.AUC(name='auc_pr', curve='PR')])
    return model

def train_model(config):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
    model = build_model(X_train.shape[1])
    es = EarlyStopping(monitor='val_auc_pr', patience=7, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=100,
                        batch_size=config['batch_size'],
                        callbacks=[es], verbose=0)

    tune.report(val_auc_pr=history.history['val_auc_pr'][-1],
                val_auc_roc=history.history['val_auc_roc'][-1])

if __name__ == "__main__":
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    analysis = tune.run(train_model,
                        config={'batch_size': tune.choice([512, 1024])},
                        resources_per_trial={'cpu': 1, 'gpu': 0.25},
                        num_samples=8,
                        scheduler=ASHAScheduler(metric='val_auc_pr', mode='max'))
    
    best_config = analysis.get_best_config(metric='val_auc_pr', mode='max')
    final_model = build_model(X.shape[1])
    final_model.fit(X, y, epochs=150, batch_size=best_config['batch_size'], validation_split=0.2)
    final_model.save("models/churn_model.keras")
