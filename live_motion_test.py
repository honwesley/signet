from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np


HAND_MODEL_PATH = Path("models/hand_landmarker.task")
MOTION_MODEL_PATH = Path("models/motion_classifier.joblib")
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


model = joblib.load(MOTION_MODEL_PATH)
classes = model.named_steps["classifier"].classes_

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open the webcam.")

frame_timestamp_ms = 0
recording = False
sequence = []
prediction_text = "Press SPACE to record"
confidence = 0.0

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

            hand = None

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]

                for point in hand:
                    x = int(point.x * frame.shape[1])
                    y = int(point.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            if recording:
                if hand is None:
                    recording = False
                    sequence = []
                    prediction_text = "Hand lost—try again"
                else:
                    features = normalize_landmarks(hand)

                    if features is not None:
                        sequence.append(features)

                if len(sequence) == SEQUENCE_LENGTH:
                    motion_features = extract_motion_features(
                        np.asarray(sequence, dtype=np.float32)
                    )

                    probabilities = model.predict_proba(
                        [motion_features]
                    )[0]

                    best_index = probabilities.argmax()
                    prediction_text = classes[best_index]
                    confidence = probabilities[best_index]

                    recording = False
                    sequence = []

            status = (
                f"Recording: {len(sequence)}/{SEQUENCE_LENGTH}"
                if recording
                else prediction_text
            )

            cv2.putText(
                frame,
                f"Motion: {status}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
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

            cv2.imshow("SIGNET Motion Test", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 32 and not recording and hand is not None:
                sequence = []
                recording = True
                prediction_text = "Recording"

finally:
    camera.release()
    cv2.destroyAllWindows()