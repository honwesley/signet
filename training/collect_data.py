import csv
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp


MODEL_PATH = Path("models/hand_landmarker.task")
DATA_FILE = Path("data/landmarks.csv")
TARGET_SAMPLES = 300


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


if len(sys.argv) != 2:
    raise SystemExit("Usage: python collect_data.py <label>")

label = sys.argv[1].upper()

STATIC_LETTERS = set("ABCDEFGHIKLMNOPQRSTUVWXY")
if label not in STATIC_LETTERS:
    raise SystemExit("Choose a static ASL letter. Cannot be J and Z because they require motion.")

DATA_FILE.parent.mkdir(exist_ok=True)
file_exists = DATA_FILE.exists()
existing_samples = 0 

if file_exists:
    with DATA_FILE.open("r", newline="") as existing_file:
        reader = csv.DictReader(existing_file)
        
        existing_samples = sum(
            row["label"] == label
            for row in reader
        )
if existing_samples >= TARGET_SAMPLES:
    raise SystemExit(
        f"{label} already has {existing_samples} samples."
    )

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open the webcam.")

sample_count = existing_samples
recording = False
frame_timestamp_ms = 0

try:
    with (
        DATA_FILE.open("a", newline="") as file,
        mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker,
    ):
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(
                ["label"] + [f"feature_{number}" for number in range(63)]
            )

        while sample_count < TARGET_SAMPLES:
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
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                for point in hand:
                    x = int(point.x * frame.shape[1])
                    y = int(point.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                if recording:
                    features = normalize_landmarks(hand)

                    if features is not None:
                        writer.writerow([label] + features)
                        sample_count += 1

            status = "RECORDING" if recording else "Press SPACE to start"

            cv2.putText(
                frame,
                f"Label: {label} | Samples: {sample_count}/{TARGET_SAMPLES}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                status,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            cv2.imshow("SIGNET Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 32:
                recording = not recording

finally:
    camera.release()
    cv2.destroyAllWindows()

print(f"Saved {sample_count} samples for {label}.")