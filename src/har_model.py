import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout


DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "dataset",
    "UCI_HAR"
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "har_model.keras"
)


def load_data():
    train_x_path = os.path.join(
        DATASET_PATH, "train", "X_train.txt"
    )
    train_y_path = os.path.join(
        DATASET_PATH, "train", "y_train.txt"
    )

    test_x_path = os.path.join(
        DATASET_PATH, "test", "X_test.txt"
    )
    test_y_path = os.path.join(
        DATASET_PATH, "test", "y_test.txt"
    )

    if not all(os.path.exists(path) for path in [
        train_x_path,
        train_y_path,
        test_x_path,
        test_y_path
    ]):
        print("Dataset files are not available yet.")
        return None

    X_train = np.loadtxt(train_x_path)
    y_train = np.loadtxt(train_y_path) - 1

    X_test = np.loadtxt(test_x_path)
    y_test = np.loadtxt(test_y_path) - 1

    return X_train, y_train, X_test, y_test


def build_model(input_shape):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_shape,)),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dropout(0.3),

        Dense(32, activation="relu"),

        Dense(6, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model():
    data = load_data()

    if data is None:
        return

    X_train, y_train, X_test, y_test = data

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    model = build_model(X_train.shape[1])

    model.summary()

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.2
    )

    loss, accuracy = model.evaluate(X_test, y_test)

    print("Test Accuracy:", accuracy)

    model.save(MODEL_PATH)

    print("Model saved successfully.")


if __name__ == "__main__":
    train_model()