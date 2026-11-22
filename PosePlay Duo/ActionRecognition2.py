# PREPROCESSING / TRAINING (RUN THIS SECOND):
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# ########################
# Load & preprocess data
# ########################
df = pd.read_csv("all_pose_data.csv")
df2 = pd.read_csv("multi_action_data.csv")
# Drop unwanted sequences
df = df[~df['sequence_id'].isin([14, 15])]

# Define feature columns: all x/y coordinates
feature_cols = [col for col in df.columns if col.startswith('x') or col.startswith('y')]

X_list = []
y_list = []

# Group by sequence_id
for seq_id, group in df.groupby('sequence_id'):
    group_sorted = group.reset_index(drop=True)
    n_frames = len(group_sorted)

    if n_frames < 5:
        continue

    # Take middle 5 frames
    start = max(0, (n_frames // 2) - 2)
    mid_frames = group_sorted.iloc[start:start+5]

    if len(mid_frames) < 5:
        continue

    # Add features
    X_list.append(mid_frames[feature_cols].to_numpy())

    label = mid_frames['label'].iloc[0]
    y_list.append(label)

# Normalize each sequence
X_list = [(seq - np.mean(seq)) / np.std(seq) for seq in X_list]

# Convert to numpy array
X = np.array(X_list)  # (samples, 5, 66)
print(f"Input shape: {X.shape}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_list)
num_classes = len(le.classes_)
print(f"Classes: {list(le.classes_)}")
print("Class distribution:", Counter(y_list))

# ########################
# Data Augmentation
# ########################
def augment_sequence(sequence):
    noise = np.random.normal(loc=0.0, scale=0.01, size=sequence.shape)
    sequence_aug = sequence + noise

    scale = np.random.uniform(0.9, 1.1)
    sequence_aug *= scale

    translation = np.random.uniform(-0.05, 0.05)
    sequence_aug += translation

    return sequence_aug

X_aug = [augment_sequence(seq) for seq in X]
y_aug = y_encoded.copy()

# Combine
X = np.concatenate([X, X_aug], axis=0)
y_encoded = np.concatenate([y_encoded, y_aug], axis=0)
print(f"After augmentation, shape: {X.shape}")

# ########################
# Training
# ########################

k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
history_per_fold = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    print(f"\n Fold {fold+1}/{k}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    # Model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(5, 66)),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # single-label
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=2,
        batch_size=16,
        verbose=0
    )

    val_acc = history.history['val_accuracy'][-1]
    print(f"Fold {fold+1} Val Accuracy: {val_acc:.4f}")
    history_per_fold.append(history.history)

# Save model
model.save("single_action_model.h5")
print("\nModel saved as single_action_model.h5")

# ########################
# Plotting
# ########################
plt.figure(figsize=(12, 5))
for i, history in enumerate(history_per_fold):
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label=f"Fold {i+1}")
    plt.title("Train Accuracy per Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label=f"Fold {i+1}")
    plt.title("Validation Accuracy per Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

plt.subplot(1, 2, 1)
plt.legend()
plt.subplot(1, 2, 2)
plt.legend()
plt.tight_layout()
plt.show()

# ########################
# Top-2 Predictions
# ########################
print("\nTop-2 Predictions on All Data")

# Predict probabilities
probs = model.predict(X)

# Get top-1 and top-2 class indices
sorted_indices = np.argsort(probs, axis=1)
top_1_idx = sorted_indices[:, -1]
top_2_idx = sorted_indices[:, -2]

# Convert to labels
top_1_labels = le.inverse_transform(top_1_idx)
top_2_labels = le.inverse_transform(top_2_idx)

# Show predictions
for i in range(len(top_1_labels)):
    print(f"Sample {i+1}: Top-1 = {top_1_labels[i]}, Top-2 = {top_2_labels[i]}")
