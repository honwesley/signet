from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd


HAND_MODEL_PATH = Path("models/hand_landmarker.task")
STATIC_MODEL_PATH = Path("models/asl_classifier.joblib")
MOTION_MODEL_PATH = Path("models/motion_classifier.joblib")

CONFIDENCE_THRESHOLD = 0.75
MOTION_CONFIDENCE_THRESHOLD = 0.70
HISTORY_SIZE = 12
MIN_VOTES = 8
SEQUENCE_LENGTH = 30


def normalize_landmarks(landmarks):
    wrist = landmarks[0]

    relative_points = [
        (
            point.x - wrist.x,
            point.y - wrist.y,
            point.z - wrist.z,
        )
        for point in landmarks
    ]

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


def extract_motion_features(sequence):
    landmarks = sequence.reshape(SEQUENCE_LENGTH, 21, 3)
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


static_model = joblib.load(STATIC_MODEL_PATH)
motion_model = joblib.load(MOTION_MODEL_PATH)
motion_classes = motion_model.named_steps["classifier"].classes_

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

typed_text = ""
last_added_letter = None
unknown_frames = 0

motion_recording = False
motion_sequence = []
motion_status = "Press M to record J or Z"
motion_cooldown = 0

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
            hand = None

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]

                for point in hand:
                    x = int(point.x * frame.shape[1])
                    y = int(point.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            if hand is None:
                prediction_history.clear()
                last_added_letter = None
                unknown_frames = 0

                if motion_recording:
                    motion_recording = False
                    motion_sequence = []
                    motion_status = "Hand lost—press M to retry"

            else:
                features = normalize_landmarks(hand)

                if features is not None:
                    if motion_recording:
                        motion_sequence.append(features)

                        displayed_label = (
                            f"Motion {len(motion_sequence)}/"
                            f"{SEQUENCE_LENGTH}"
                        )

                        if len(motion_sequence) == SEQUENCE_LENGTH:
                            sequence = np.asarray(
                                motion_sequence,
                                dtype=np.float32,
                            )

                            motion_features = extract_motion_features(
                                sequence
                            )

                            probabilities = motion_model.predict_proba(
                                [motion_features]
                            )[0]

                            best_index = probabilities.argmax()
                            motion_prediction = motion_classes[best_index]
                            confidence = probabilities[best_index]

                            if (
                                motion_prediction in {"J", "Z"}
                                and confidence
                                >= MOTION_CONFIDENCE_THRESHOLD
                            ):
                                typed_text += motion_prediction
                                displayed_label = motion_prediction
                                motion_status = (
                                    f"Added {motion_prediction} "
                                    f"({confidence:.0%})"
                                )
                            else:
                                displayed_label = "OTHER"
                                motion_status = (
                                    f"No J/Z detected "
                                    f"({confidence:.0%})"
                                )

                            motion_recording = False
                            motion_sequence = []
                            motion_cooldown = 20
                            prediction_history.clear()
                            last_added_letter = None

                    elif motion_cooldown > 0:
                        motion_cooldown -= 1
                        displayed_label = "Motion complete"

                    else:
                        sample = pd.DataFrame(
                            [features],
                            columns=static_model.feature_names_in_,
                        )

                        probabilities = static_model.predict_proba(
                            sample
                        )[0]

                        best_index = probabilities.argmax()
                        prediction = static_model.classes_[best_index]
                        confidence = probabilities[best_index]

                        if confidence >= CONFIDENCE_THRESHOLD:
                            unknown_frames = 0
                            prediction_history.append(prediction)

                            if len(prediction_history) >= MIN_VOTES:
                                most_common, votes = Counter(
                                    prediction_history
                                ).most_common(1)[0]

                                if votes >= MIN_VOTES:
                                    displayed_label = most_common

                                    if most_common != last_added_letter:
                                        typed_text += most_common
                                        last_added_letter = most_common
                                else:
                                    displayed_label = "Reading..."
                            else:
                                displayed_label = "Reading..."

                        else:
                            unknown_frames += 1
                            displayed_label = "Unknown"

                            if unknown_frames >= 5:
                                prediction_history.clear()
                                last_added_letter = None

            cv2.putText(
                frame,
                f"Prediction: {displayed_label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence:.0%}",
                (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 0, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Text: {typed_text}",
                (20, 116),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                motion_status,
                (20, 154),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                "M: J/Z | SPACE: space | C: clear | Q: quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            cv2.imshow("SIGNET Full ASL App", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 32:
                typed_text += " "

            if key == 8 and typed_text:
                typed_text = typed_text[:-1]

            if key == ord("c"):
                typed_text = ""
                last_added_letter = None

            if key == ord("m") and not motion_recording and hand is not None:
                motion_recording = True
                motion_sequence = []
                motion_status = "Recording motion..."
                prediction_history.clear()

finally:
    camera.release()
    cv2.destroyAllWindows()