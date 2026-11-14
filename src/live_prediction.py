import os
import cv2
import numpy as np
import tensorflow as tf

# ==========================================================
# 1. LOAD TRAINED MODEL
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "har_model.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)

print("HAR model loaded successfully!")


# ==========================================================
# 2. ACTIVITY NAMES
# ==========================================================

ACTIVITIES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]


# ==========================================================
# 3. START CAMERA
# ==========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera started successfully!")
print("Press Q to quit.")


# ==========================================================
# 4. LIVE CAMERA LOOP
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    # Display camera
    cv2.putText(
        frame,
        "HAR - Live Activity Recognition",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Camera working - Model loaded",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Human Activity Recognition", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================================
# 5. RELEASE CAMERA
# ==========================================================

cap.release()
cv2.destroyAllWindows()

print("Camera stopped.")