from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


DATA_FILE = Path("data/landmarks.csv")
MODEL_FILE = Path("models/asl_classifier.joblib")

data = pd.read_csv(DATA_FILE)

X = data.drop(columns=["label"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

print("Training model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nClassification report:")
print(classification_report(y_test, predictions))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))
print("Class order:", model.classes_)

MODEL_FILE.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_FILE)

print(f"\nModel saved to {MODEL_FILE}")