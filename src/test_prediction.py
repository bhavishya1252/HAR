import os
import numpy as np
import tensorflow as tf

# Project base folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model path
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "har_model.keras"
)

# Test data path
TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "UCI_HAR",
    "test"
)

X_TEST_PATH = os.path.join(TEST_DIR, "X_test.txt")
Y_TEST_PATH = os.path.join(TEST_DIR, "y_test.txt")


# Load model
model = tf.keras.models.load_model(MODEL_PATH)

print("HAR model loaded successfully!")


# Load test data
X_test = np.loadtxt(X_TEST_PATH)
y_test = np.loadtxt(Y_TEST_PATH).astype(int)

# UCI HAR labels: 1-6
activity_names = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

print("\nTotal test samples:", len(X_test))

# Take first test sample
sample = X_test[0].reshape(1, -1)

prediction = model.predict(sample, verbose=0)

predicted_class = np.argmax(prediction[0])
actual_class = y_test[0] - 1

print("\n==============================")
print("Predicted Activity:", activity_names[predicted_class])
print("Actual Activity   :", activity_names[actual_class])
print("Confidence        :", round(float(prediction[0][predicted_class]) * 100, 2), "%")
print("==============================")