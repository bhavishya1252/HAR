import os
import numpy as np
import pandas as pd


# Dataset path
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "dataset",
    "UCI_HAR"
)


def load_data():
    print("Dataset path:", DATASET_PATH)

    if not os.path.exists(DATASET_PATH):
        print("Dataset folder not found.")
        return

    print("Dataset folder found.")

    train_path = os.path.join(DATASET_PATH, "train")
    test_path = os.path.join(DATASET_PATH, "test")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Train folder found.")
        print("Test folder found.")
    else:
        print("Train/Test data is not available yet.")


if __name__ == "__main__":
    load_data()