from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_DIRECTORY = Path("data/motion")
MODEL_FILE = Path("models/motion_classifier.joblib")
LABELS = ["J", "Z", "OTHER"]


def extract_motion_features(sequence):
    # Convert 63 values back into 21 landmarks with x, y, z.
    landmarks = sequence.reshape(SEQUENCE_LENGTH, 21, 3)

    # J mainly uses the pinky; Z mainly uses the index finger.
    fingertip_paths = landmarks[:, [8, 20], :]

    velocity = np.diff(fingertip_paths, axis=0)
    acceleration = np.diff(velocity, axis=0)

    path_lengths = np.linalg.norm(velocity, axis=2).sum(axis=0)
    displacement = fingertip_paths[-1] - fingertip_paths[0]
    coordinate_range = (
        fingertip_paths.max(axis=0) - fingertip_paths.min(axis=0)
    )

    return np.concatenate(
        [
            fingertip_paths.flatten(),
            velocity.flatten(),
            acceleration.flatten(),
            path_lengths.flatten(),
            displacement.flatten(),
            coordinate_range.flatten(),
        ]
    )


SEQUENCE_LENGTH = 30
features = []
labels = []

for label in LABELS:
    for filename in (DATA_DIRECTORY / label).glob("*.npy"):
        sequence = np.load(filename)

        if sequence.shape != (SEQUENCE_LENGTH, 63):
            print("Skipping invalid file:", filename)
            continue

        features.append(extract_motion_features(sequence))
        labels.append(label)

X = np.asarray(features)
y = np.asarray(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        (
            "classifier",
            SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

print("Training motion classifier...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump(model, MODEL_FILE)
print("Model saved to:", MODEL_FILE)