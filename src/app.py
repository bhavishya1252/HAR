import os
import numpy as np
import tensorflow as tf


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "har_model.keras"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "UCI_HAR",
    "test"
)

X_TEST_PATH = os.path.join(TEST_DIR, "X_test.txt")


ACTIVITIES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]


def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def predict_activity(model, sample):
    prediction = model.predict(sample.reshape(1, -1), verbose=0)

    predicted_index = np.argmax(prediction[0])

    activity = ACTIVITIES[predicted_index]
    confidence = float(prediction[0][predicted_index]) * 100

    return activity, confidence


def main():

    print("=" * 45)
    print("   HUMAN ACTIVITY RECOGNITION SYSTEM")
    print("=" * 45)

    print("\nLoading trained HAR model...")

    model = load_model()

    print("Model loaded successfully!")

    X_test = np.loadtxt(X_TEST_PATH)

    print("\nTest dataset loaded successfully!")
    print("Total samples:", len(X_test))

    while True:

        print("\n1. Predict Activity")
        print("2. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":

            sample_number = input(
                f"Enter sample number (1-{len(X_test)}): "
            )

            try:
                sample_number = int(sample_number)

                if sample_number < 1 or sample_number > len(X_test):
                    print("Invalid sample number.")
                    continue

                sample = X_test[sample_number - 1]

                activity, confidence = predict_activity(
                    model,
                    sample
                )

                print("\n-----------------------------")
                print("Predicted Activity :", activity)
                print(
                    "Confidence          :",
                    round(confidence, 2),
                    "%"
                )
                print("-----------------------------")

            except ValueError:
                print("Please enter a valid number.")

        elif choice == "2":

            print("\nHAR system closed.")
            break

        else:

            print("Invalid choice.")


if __name__ == "__main__":
    main()