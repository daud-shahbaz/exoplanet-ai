"""
NASA Exoplanet Detection - Deep Learning Model Training
Using 3-Model Ensemble: DNN, CNN, LSTM
Author: Lead AI Engineer with 20 years experience
"""

import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("EXOPLANET DETECTION MODEL TRAINING")
print("="*80)

# Load the dataset
print("\n[1/6] Loading dataset...")
df = pd.read_csv('dataset/cumulative_2025.10.05_02.33.52.csv', comment='#')
print(f"✓ Loaded {len(df)} samples")

# Key features for exoplanet detection (based on 20 years experience)
KEY_FEATURES = [
    'koi_period',      # Orbital period - CRITICAL
    'koi_depth',       # Transit depth - CRITICAL
    'koi_duration',    # Transit duration
    'koi_prad',        # Planetary radius
    'koi_teq',         # Equilibrium temperature
    'koi_insol',       # Insolation flux
    'koi_steff',       # Stellar effective temperature
    'koi_slogg',       # Stellar surface gravity
    'koi_srad',        # Stellar radius
    'koi_impact',      # Impact parameter
]

print("\n[2/6] Preprocessing data...")

# Select features that exist
available_features = [f for f in KEY_FEATURES if f in df.columns]
print(f"✓ Using {len(available_features)} key features")

# Create feature matrix
X = df[available_features].copy()

# Handle missing values with median imputation
X = X.fillna(X.median())

# Create target variable
# Binary classification: CONFIRMED = 1, else = 0
y = (df['koi_disposition'] == 'CONFIRMED').astype(int)

print(f"✓ Total samples: {len(X)}")
print(f"✓ Confirmed planets: {y.sum()} ({y.mean()*100:.1f}%)")
print(f"✓ Non-planets: {(~y.astype(bool)).sum()} ({(~y.astype(bool)).mean()*100:.1f}%)")

# Split data - stratified to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Testing samples: {len(X_test)}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate feature importance using correlation with target
feature_importance = {}
for i, feat in enumerate(available_features):
    correlation = abs(np.corrcoef(X_train.iloc[:, i], y_train)[0, 1])
    feature_importance[feat] = float(correlation)

# Sort by importance
feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

print("\n[3/6] Feature Importance (Top 5):")
for i, (feat, imp) in enumerate(list(feature_importance.items())[:5], 1):
    print(f"  {i}. {feat}: {imp:.4f}")

# Model 1: Deep Neural Network (Primary Model)
print("\n[4/6] Training Model 1: Deep Neural Network...")
def create_dnn_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ], name='DNN_Exoplanet_Detector')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model

model1 = create_dnn_model(X_train_scaled.shape[1])

# Class weights to handle imbalance
class_weight = {0: 1.0, 1: (len(y_train) - y_train.sum()) / y_train.sum()}

history1 = model1.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=[
        EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(patience=7, factor=0.5, verbose=1)
    ],
    verbose=1
)

print("✓ DNN training complete")

# Model 2: CNN (Pattern Detection in Feature Space)
print("\n[5/6] Training Model 2: Convolutional Neural Network...")
def create_cnn_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ], name='CNN_Exoplanet_Detector')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# Reshape for CNN
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

model2 = create_cnn_model(X_train_scaled.shape[1])

history2 = model2.fit(
    X_train_cnn, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=[
        EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(patience=7, factor=0.5, verbose=1)
    ],
    verbose=1
)

print("✓ CNN training complete")

# Model 3: LSTM (Temporal Pattern Detection)
print("\n[6/6] Training Model 3: LSTM...")
def create_lstm_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ], name='LSTM_Exoplanet_Detector')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

model3 = create_lstm_model(X_train_scaled.shape[1])

history3 = model3.fit(
    X_train_cnn, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=[
        EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(patience=7, factor=0.5, verbose=1)
    ],
    verbose=1
)

print("✓ LSTM training complete")

# Ensemble Prediction
print("\n" + "="*80)
print("ENSEMBLE MODEL EVALUATION")
print("="*80)

pred1 = model1.predict(X_test_scaled, verbose=0).flatten()
pred2 = model2.predict(X_test_cnn, verbose=0).flatten()
pred3 = model3.predict(X_test_cnn, verbose=0).flatten()

# Weighted ensemble (DNN gets highest weight based on experience)
ensemble_pred = 0.5 * pred1 + 0.3 * pred2 + 0.2 * pred3
ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)

# Individual model performance
print("\n### Model 1 (DNN) ###")
pred1_binary = (pred1 > 0.5).astype(int)
print(f"Accuracy: {accuracy_score(y_test, pred1_binary):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, pred1):.4f}")

print("\n### Model 2 (CNN) ###")
pred2_binary = (pred2 > 0.5).astype(int)
print(f"Accuracy: {accuracy_score(y_test, pred2_binary):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, pred2):.4f}")

print("\n### Model 3 (LSTM) ###")
pred3_binary = (pred3 > 0.5).astype(int)
print(f"Accuracy: {accuracy_score(y_test, pred3_binary):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, pred3):.4f}")

print("\n### ENSEMBLE MODEL ###")
print(f"Accuracy: {accuracy_score(y_test, ensemble_pred_binary):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, ensemble_pred):.4f}")

print("\n" + classification_report(y_test, ensemble_pred_binary, 
                                   target_names=['Not Planet', 'Exoplanet']))

# Confusion Matrix
cm = confusion_matrix(y_test, ensemble_pred_binary)
print("\nConfusion Matrix:")
print(cm)
print(f"True Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")

# Save models
print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

model1.save('models/dnn_model.keras')
print("✓ Saved DNN model")

model2.save('models/cnn_model.keras')
print("✓ Saved CNN model")

model3.save('models/lstm_model.keras')
print("✓ Saved LSTM model")

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("✓ Saved scaler")

# Save metadata
metadata = {
    'feature_names': available_features,
    'feature_importance': feature_importance,
    'target_column': 'koi_disposition',
    'model_performance': {
        'dnn': {
            'accuracy': float(accuracy_score(y_test, pred1_binary)),
            'roc_auc': float(roc_auc_score(y_test, pred1))
        },
        'cnn': {
            'accuracy': float(accuracy_score(y_test, pred2_binary)),
            'roc_auc': float(roc_auc_score(y_test, pred2))
        },
        'lstm': {
            'accuracy': float(accuracy_score(y_test, pred3_binary)),
            'roc_auc': float(roc_auc_score(y_test, pred3))
        },
        'ensemble': {
            'accuracy': float(accuracy_score(y_test, ensemble_pred_binary)),
            'roc_auc': float(roc_auc_score(y_test, ensemble_pred))
        }
    },
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'confirmed_planets_train': int(y_train.sum()),
    'confirmed_planets_test': int(y_test.sum())
}

with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Saved metadata")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\n✓ Models ready for deployment")
print(f"✓ Final Ensemble Accuracy: {metadata['model_performance']['ensemble']['accuracy']:.2%}")
print(f"✓ Final Ensemble ROC-AUC: {metadata['model_performance']['ensemble']['roc_auc']:.4f}")
