from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import sys

import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC 

if getattr(sys, "frozen", False):
    ROOT_DIRECTORY = Path(sys._MEIPASS)
else:
    ROOT_DIRECTORY = Path(__file__).resolve().parents[1]

HAND_MODEL_PATH = ROOT_DIRECTORY / "models/hand_landmarker.task"
STATIC_MODEL_PATH = ROOT_DIRECTORY / "models/asl_classifier.joblib"
MOTION_MODEL_PATH = ROOT_DIRECTORY / "models/motion_classifier.joblib"

CONFIDENCE_THRESHOLD = 0.75
MOTION_CONFIDENCE_THRESHOLD = 0.70
AUTO_MOTION_CONFIDENCE = 0.78
HISTORY_SIZE = 8
MIN_VOTES = 4
SEQUENCE_LENGTH = 30
AUTO_SEQUENCE_LENGTH = 15
AUTO_CHECK_INTERVAL = 3
MIN_MOTION_DISTANCE = 0.18
NO_HAND_SPACE_FRAMES = 3


@dataclass
class RecognitionOutput:
    frame: np.ndarray
    label: str
    confidence: float
    added_text: str
    status: str


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


def resample_sequence(sequence):
    sequence = np.asarray(sequence, dtype=np.float32)

    old_times = np.linspace(0, 1, len(sequence))
    new_times = np.linspace(0, 1, SEQUENCE_LENGTH)

    resampled = np.empty(
        (SEQUENCE_LENGTH, sequence.shape[1]),
        dtype=np.float32,
    )

    for feature_index in range(sequence.shape[1]):
        resampled[:, feature_index] = np.interp(
            new_times,
            old_times,
            sequence[:, feature_index],
        )

    return resampled


class RecognitionEngine:
    def __init__(self):
        self.static_model = joblib.load(STATIC_MODEL_PATH)
        self.static_model.n_jobs = 1
        self.motion_model = joblib.load(MOTION_MODEL_PATH)

        self.motion_classes = self.motion_model.named_steps[
            "classifier"
        ].classes_

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

        self.landmarker = (
            mp.tasks.vision.HandLandmarker.create_from_options(options)
        )

        self.frame_timestamp_ms = 0
        self.prediction_history = deque(maxlen=HISTORY_SIZE)
        self.last_added_letter = None
        self.unknown_frames = 0
        self.hand_present = False

        self.motion_recording = False
        self.motion_sequence = []
        self.motion_cooldown = 0
        self.motion_status = "Ready"

        self.auto_motion_buffer = deque(maxlen=AUTO_SEQUENCE_LENGTH)
        self.auto_check_counter = 0
        
        self.no_hand_frames = 0
        self.space_added_for_absence = False

    def reset_text_state(self):
        self.prediction_history.clear()
        self.auto_motion_buffer.clear()
        self.last_added_letter = None
        self.unknown_frames = 0

    def start_motion(self):
        if not self.hand_present:
            self.motion_status = "Show your hand before recording"
            return False

        if self.motion_recording:
            return False

        self.motion_recording = True
        self.motion_sequence = []
        self.motion_status = "Recording J/Z motion"
        self.prediction_history.clear()

        return True

    def _classify_motion(self, sequence):
        motion_features = extract_motion_features(sequence)
        probabilities = self.motion_model.predict_proba(
            [motion_features]
        )[0]

        best_index = probabilities.argmax()
        prediction = self.motion_classes[best_index]
        confidence = probabilities[best_index]

        return prediction, confidence

    def process(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        self.frame_timestamp_ms += 33
        result = self.landmarker.detect_for_video(
            mp_image,
            self.frame_timestamp_ms,
        )

        displayed_label = "â€”"
        confidence = 0.0
        added_text = ""
        hand = None

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            self.hand_present = True
            
            self.no_hand_frames = 0
            self.space_added_for_absence = False

            for point in hand:
                x = int(point.x * frame.shape[1])
                y = int(point.y * frame.shape[0])
                cv2.circle(frame, (x, y), 4, (53, 208, 127), -1)
        else:
            self.hand_present = False
            self.prediction_history.clear()
            self.auto_motion_buffer.clear()
            self.last_added_letter = None
            self.unknown_frames = 0

            self.no_hand_frames += 1
            if(
                self.no_hand_frames >= NO_HAND_SPACE_FRAMES
                and not self.space_added_for_absence
            ):
                added_text = " "
                self.space_added_for_absence = True 
                self.motion_status = "added word space"

            if self.motion_recording:
                self.motion_recording = False
                self.motion_sequence = []
                self.motion_status = "Hand lostâ€”try again"

        if hand is not None:
            features = normalize_landmarks(hand)

            if features is not None:
                auto_motion_detected = False

                if (
                    not self.motion_recording
                    and self.motion_cooldown == 0
                ):
                    self.auto_motion_buffer.append(features)
                    self.auto_check_counter += 1

                    should_check_motion = (
                        len(self.auto_motion_buffer)
                        == AUTO_SEQUENCE_LENGTH
                        and self.auto_check_counter
                        % AUTO_CHECK_INTERVAL
                        == 0
                    )

                    if should_check_motion:
                        short_sequence = np.asarray(
                            self.auto_motion_buffer,
                            dtype=np.float32,
                        )

                        short_landmarks = short_sequence.reshape(
                            AUTO_SEQUENCE_LENGTH,
                            21,
                            3,
                        )

                        fingertip_paths = short_landmarks[:, [8, 20], :]
                        fingertip_velocity = np.diff(
                            fingertip_paths,
                            axis=0,
                        )

                        motion_distance = np.linalg.norm(
                            fingertip_velocity,
                            axis=2,
                        ).sum(axis=0).max()

                        if motion_distance >= MIN_MOTION_DISTANCE:
                            full_sequence = resample_sequence(
                                short_sequence
                            )
                            motion_prediction, motion_confidence = (
                                self._classify_motion(full_sequence)
                            )

                            if (
                                motion_prediction in {"J", "Z"}
                                and motion_confidence
                                >= AUTO_MOTION_CONFIDENCE
                            ):
                                displayed_label = motion_prediction
                                confidence = motion_confidence
                                added_text = motion_prediction
                                auto_motion_detected = True

                                self.motion_status = (
                                    f"Automatically added "
                                    f"{motion_prediction}"
                                )
                                self.motion_cooldown = 20
                                self.auto_motion_buffer.clear()
                                self.prediction_history.clear()
                                self.last_added_letter = None

                if self.motion_recording:
                    self.motion_sequence.append(features)
                    displayed_label = (
                        f"{len(self.motion_sequence)}/{SEQUENCE_LENGTH}"
                    )

                    if len(self.motion_sequence) == SEQUENCE_LENGTH:
                        sequence = np.asarray(
                            self.motion_sequence,
                            dtype=np.float32,
                        )
                        motion_prediction, confidence = (
                            self._classify_motion(sequence)
                        )

                        if (
                            motion_prediction in {"J", "Z"}
                            and confidence
                            >= MOTION_CONFIDENCE_THRESHOLD
                        ):
                            displayed_label = motion_prediction
                            added_text = motion_prediction
                            self.motion_status = (
                                f"Added {motion_prediction}"
                            )
                        else:
                            displayed_label = "OTHER"
                            self.motion_status = "No J or Z detected"

                        self.motion_recording = False
                        self.motion_sequence = []
                        self.motion_cooldown = 20
                        self.auto_motion_buffer.clear()
                        self.prediction_history.clear()
                        self.last_added_letter = None

                elif auto_motion_detected:
                    pass

                elif self.motion_cooldown > 0:
                    self.motion_cooldown -= 1

                    if self.motion_cooldown == 0:
                        self.motion_status = "Ready"

                else:
                    sample = pd.DataFrame(
                        [features],
                        columns=self.static_model.feature_names_in_,
                    )

                    probabilities = self.static_model.predict_proba(
                        sample
                    )[0]

                    best_index = probabilities.argmax()
                    prediction = self.static_model.classes_[best_index]
                    confidence = probabilities[best_index]

                    if confidence >= CONFIDENCE_THRESHOLD:
                        self.unknown_frames = 0
                        self.prediction_history.append(prediction)

                        if len(self.prediction_history) >= MIN_VOTES:
                            most_common, votes = Counter(
                                self.prediction_history
                            ).most_common(1)[0]

                            if votes >= MIN_VOTES:
                                displayed_label = most_common

                                if most_common != self.last_added_letter:
                                    added_text = most_common
                                    self.last_added_letter = most_common
                            else:
                                displayed_label = "â€¦"
                        else:
                            displayed_label = "â€¦"
                    else:
                        displayed_label = "?"
                        self.unknown_frames += 1

                        if self.unknown_frames >= 5:
                            self.prediction_history.clear()
                            self.last_added_letter = None

        return RecognitionOutput(
            frame=frame,
            label=displayed_label,
            confidence=confidence,
            added_text=added_text,
            status=self.motion_status,
        )

    def close(self):
        self.landmarker.close()