import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ==========================================================
# 1. PATH SETUP
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "UCI_HAR", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "UCI_HAR", "test")

print("Training data path:", TRAIN_DIR)
print("Testing data path :", TEST_DIR)


# ==========================================================
# 2. LOAD FEATURES
# ==========================================================

def load_features(folder):
    """
    Load UCI HAR feature data.
    """

    file_path = os.path.join(
        folder,
        "X_" + os.path.basename(folder) + ".txt"
    )

    # If above path doesn't work, use standard UCI HAR names
    if not os.path.exists(file_path):
        file_path = os.path.join(folder, "X_" + os.path.basename(folder) + ".txt")

    data = np.loadtxt(file_path)

    return data


# ==========================================================
# 3. LOAD TRAIN DATA
# ==========================================================

print("\nLoading training data...")

X_train_path = os.path.join(TRAIN_DIR, "X_train.txt")
y_train_path = os.path.join(TRAIN_DIR, "y_train.txt")

X_train = np.loadtxt(X_train_path)
y_train = np.loadtxt(y_train_path).astype(int)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# ==========================================================
# 4. LOAD TEST DATA
# ==========================================================

print("\nLoading testing data...")

X_test_path = os.path.join(TEST_DIR, "X_test.txt")
y_test_path = os.path.join(TEST_DIR, "y_test.txt")

X_test = np.loadtxt(X_test_path)
y_test = np.loadtxt(y_test_path).astype(int)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ==========================================================
# 5. CONVERT LABELS
# ==========================================================

# UCI HAR labels are 1 to 6.
# Neural network uses 0 to 5.

y_train = y_train - 1
y_test = y_test - 1

y_train = to_categorical(y_train, num_classes=6)
y_test = to_categorical(y_test, num_classes=6)


# ==========================================================
# 6. BUILD DEEP LEARNING MODEL
# ==========================================================

print("\nBuilding Deep Learning model...")

model = Sequential([
    
    Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(64, activation="relu"),

    Dense(6, activation="softmax")
])


# ==========================================================
# 7. COMPILE MODEL
# ==========================================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================================
# 8. SHOW MODEL
# ==========================================================

model.summary()


# ==========================================================
# 9. TRAIN MODEL
# ==========================================================

print("\nStarting model training...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)


# ==========================================================
# 10. TEST MODEL
# ==========================================================

print("\nTesting model...")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\n===================================")
print("TEST ACCURACY:", round(accuracy * 100, 2), "%")
print("===================================")


# ==========================================================
# 11. SAVE MODEL
# ==========================================================

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "har_model.keras"
)

model.save(MODEL_PATH)

print("\nModel saved successfully!")
print("Saved at:", MODEL_PATH)