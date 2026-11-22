# PREPROCESSING / TRAINING (RUN THIS SECOND):

from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random


import tensorflow as tf
# ########################
# Load & preprocess data
# ########################
df = pd.read_csv("all_pose_data.csv")
print(df.head())


# Drop sequences 14 & 15
df = df[~df['sequence_id'].isin([14, 15])]

# Define feature columns: all x & y columns
feature_cols = [col for col in df.columns if col.startswith('x') or col.startswith('y')]

X_list = []
y_list = []

# Group by sequence and process each
for seq_id, group in df.groupby('sequence_id'):
    group_sorted = group.reset_index(drop=True)
    n_frames = len(group_sorted)


    if n_frames < 2:
        continue

    # Select middle 5 frames
    start = max(0, (n_frames // 2) - 2)
    mid_frames = group_sorted.iloc[start:start+2]

    if len(mid_frames) < 2:
        continue

    X_list.append(mid_frames[feature_cols].to_numpy())
    y_list.append(mid_frames['label'].iloc[0])

# Normalize each sequence (zero mean, unit variance)
X_list = [(seq - np.mean(seq)) / np.std(seq) for seq in X_list]

# Convert to numpy arrays
X = np.array(X_list)  # shape: (num_samples, 5, 66)
print(f"Input shape: {X.shape}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_list)
num_classes = len(le.classes_)
print(f"Classes: {list(le.classes_)}")

# ########################
# Data Augmentation
# ########################
def augment_sequence(sequence, label):
    # Augments pose keypoints with noise, scaling, and translation
    sequence_aug = sequence.copy()

    # Add Gaussian noise
    noise = np.random.normal(loc=0.0, scale=0.01, size=sequence.shape)
    sequence_aug += noise

    # Uniform scale
    scale = np.random.uniform(0.9, 1.1)
    sequence_aug *= scale

    # Translation
    translation_x = np.random.uniform(-0.05, 0.05)
    translation_y = np.random.uniform(-0.05, 0.05)
    sequence_aug[:, ::2] += translation_x  # x-coordinates
    sequence_aug[:, 1::2] += translation_y  # y-coordinates

    return sequence_aug


# Apply augmentation
X_augmented = [augment_sequence(seq, label) for seq, label in zip(X, y_list)]
y_augmented = y_list.copy()

# Combine original and augmented
X = np.concatenate([X, np.array(X_augmented)], axis=0)
y_encoded = np.concatenate([y_encoded, le.transform(y_augmented)], axis=0)

print(f"Data augmented. New shape: {X.shape}")

# ########################
# 🔷 Training
# ########################
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
history_per_fold = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    print(f"\n Fold {fold+1}/{k}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(5, 66)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=12,
                        batch_size=32,
                        verbose=0)

    val_acc = history.history['val_accuracy'][-1]
    print(f"Fold {fold+1} Validation Accuracy: {val_acc:.4f}")
    history_per_fold.append(history.history)

model.save("action_model.h5")
print("\nModel saved as action_model.h5")

# ########################
# Plotting
# ########################
avg_acc = np.mean([h['val_accuracy'][-1] for h in history_per_fold])
print(f"\nAverage Validation Accuracy across {k} folds: {avg_acc:.4f}")

plt.figure(figsize=(12, 5))
for i, history in enumerate(history_per_fold):
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label=f"Fold {i+1}")
    plt.title("Train Accuracy per Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label=f"Fold {i+1}")
    plt.title("Val Accuracy per Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

plt.subplot(1, 2, 1)
plt.legend()
plt.subplot(1, 2, 2)
plt.legend()
plt.tight_layout()
plt.show()
