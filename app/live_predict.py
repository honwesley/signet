from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import pandas as pd

from collections import Counter, deque 

HAND_MODEL_PATH = Path("models/hand_landmarker.task")
CLASSIFIER_PATH = Path("models/asl_classifier.joblib")
CONFIDENCE_THRESHOLD = 0.75
HISTORY_SIZE = 12
MIN_VOTES = 8


def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    relative_points = []

    for point in landmarks:
        relative_points.append(
            (
                point.x - wrist.x,
                point.y - wrist.y,
                point.z - wrist.z,
            )
        )

    scale = max(
        (x**2 + y**2 + z**2) ** 0.5
        for x, y, z in relative_points
    )

    if scale < 0.000001:
        return None

    features = []

    for x, y, z in relative_points:
        features.extend([x / scale, y / scale, z / scale])

    return features

model = joblib.load(CLASSIFIER_PATH)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open the webcam.")

frame_timestamp_ms = 0
prediction_history = deque(maxlen=HISTORY_SIZE)

try:
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = camera.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            frame_timestamp_ms += 33
            result = landmarker.detect_for_video(
                mp_image,
                frame_timestamp_ms,
            )

            displayed_label = "No hand"
            confidence = 0.0

            if not result.hand_landmarks:
                prediction_history.clear()
                
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                features = normalize_landmarks(hand)

                for point in hand:
                    x = int(point.x * frame.shape[1])
                    y = int(point.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                if features is not None:
                    sample = pd.DataFrame(
                        [features],
                        columns=model.feature_names_in_,
                    )

                    probabilities = model.predict_proba(sample)[0]
                    best_index = probabilities.argmax()

                    prediction = model.classes_[best_index]
                    confidence = probabilities[best_index]

                    if confidence >= CONFIDENCE_THRESHOLD:
                        prediction_history.append(prediction)
                        
                        if len(prediction_history) >= MIN_VOTES:
                            most_common, votes = Counter(
                                prediction_history
                            ).most_common(1)[0]
                            
                            if votes >= MIN_VOTES:
                                displayed_label = most_common
                            else:
                                displayed_label = "Reading..."
                        else:
                            displayed_label = "Reading..."
                    else:
                        prediction_history.clear()
                        displayed_label = "Unknown"

            cv2.putText(
                frame,
                f"Prediction: {displayed_label}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence:.0%}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )

            cv2.imshow("SIGNET ASL Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()