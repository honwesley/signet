import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


MODEL_PATH = Path("models/hand_landmarker.task")
DATA_DIRECTORY = Path("data/motion")

SEQUENCE_LENGTH = 30
VALID_LABELS = {"J", "Z", "OTHER"}


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
        features.extend([
            x / scale,
            y / scale,
            z / scale,
        ])

    return features


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Collect J, Z, and OTHER motion data."
    )

    parser.add_argument(
        "contributor_id",
        help="Anonymous contributor ID, such as person_02",
    )

    parser.add_argument(
        "labels",
        nargs="+",
        help="One or more labels: J, Z, or OTHER",
    )

    parser.add_argument(
        "--sequences",
        type=int,
        default=10,
        help="Number of motion sequences per label",
    )

    return parser.parse_args()


def clean_contributor_id(contributor_id):
    cleaned_id = ""

    for character in contributor_id.strip():
        if character.isalnum() or character in {"_", "-"}:
            cleaned_id += character
        else:
            cleaned_id += "_"

    return cleaned_id


def ask_what_to_do(sequence_count):
    while True:
        print(f"\nCollected {sequence_count} motion sequences.")

        answer = input(
            "[Y] Keep  [N] Redo  [S] Skip  [0] Quit: "
        ).strip().lower()

        if answer in {"y", "yes"}:
            if sequence_count == 0:
                print("There are no sequences to keep.")
                continue

            return "keep"

        if answer in {"n", "no"}:
            return "redo"

        if answer in {"s", "skip"}:
            return "skip"

        if answer == "0":
            return "quit"

        print("Please enter Y, N, S, or 0.")


def get_next_file_number(output_directory, contributor_id):
    file_number = 1

    while True:
        filename = (
            output_directory
            / f"{contributor_id}_{file_number:04d}.npy"
        )

        if not filename.exists():
            return file_number

        file_number += 1


def save_sequences(
    contributor_id,
    label,
    collected_sequences,
):
    output_directory = DATA_DIRECTORY / label
    output_directory.mkdir(parents=True, exist_ok=True)

    file_number = get_next_file_number(
        output_directory,
        contributor_id,
    )

    saved_files = []

    for sequence in collected_sequences:
        filename = (
            output_directory
            / f"{contributor_id}_{file_number:04d}.npy"
        )

        np.save(
            filename,
            np.asarray(sequence, dtype=np.float32),
        )

        saved_files.append(filename)
        file_number += 1

    return saved_files


def draw_hand(frame, hand):
    for point in hand:
        x = int(point.x * frame.shape[1])
        y = int(point.y * frame.shape[0])

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1,
        )


def collect_motion_sequences(
    contributor_id,
    label,
    target_sequences,
    camera,
    landmarker,
):
    collected_sequences = []

    current_sequence = []
    recording = False
    quit_requested = False
    message = "Press SPACE to record a motion"

    last_timestamp_ms = 0

    while len(collected_sequences) < target_sequences:
        success, frame = camera.read()

        if not success:
            print("Could not read a frame from the webcam.")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        timestamp_ms = max(
            int(time.monotonic() * 1000),
            last_timestamp_ms + 1,
        )

        last_timestamp_ms = timestamp_ms

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        hand = None

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            draw_hand(frame, hand)

        if recording:
            if hand is None:
                recording = False
                current_sequence = []
                message = (
                    "Hand lost. Press SPACE to try again."
                )

            else:
                features = normalize_landmarks(hand)

                if features is not None:
                    current_sequence.append(features)

                if len(current_sequence) >= SEQUENCE_LENGTH:
                    collected_sequences.append(
                        np.asarray(
                            current_sequence,
                            dtype=np.float32,
                        )
                    )

                    recording = False
                    current_sequence = []
                    message = (
                        "Sequence complete. "
                        "Press SPACE for the next one."
                    )

        if recording:
            status = (
                f"RECORDING: "
                f"{len(current_sequence)}/{SEQUENCE_LENGTH}"
            )
        else:
            status = message

        cv2.putText(
            frame,
            f"Contributor: {contributor_id}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Label: {label} | Sequences: "
            f"{len(collected_sequences)}/{target_sequences}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            status,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            "SPACE = record next motion",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "Q = finish this label early",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "0 = quit entire collection",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        cv2.imshow(
            "SIGNET Motion Data Collector",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("0"):
            quit_requested = True
            break

        if key == ord("q"):
            break

        if key == 32 and not recording:
            if hand is None:
                message = (
                    "Show your hand before pressing SPACE."
                )
            else:
                current_sequence = []
                recording = True
                message = "Recording motion..."

    cv2.destroyAllWindows()

    return collected_sequences, quit_requested


def main():
    args = parse_arguments()

    contributor_id = clean_contributor_id(
        args.contributor_id
    )

    labels = [
        label.upper()
        for label in args.labels
    ]

    target_sequences = args.sequences

    if not contributor_id:
        raise SystemExit(
            "Contributor ID cannot be empty."
        )

    invalid_labels = [
        label
        for label in labels
        if label not in VALID_LABELS
    ]

    if invalid_labels:
        raise SystemExit(
            "Invalid motion labels: "
            + ", ".join(invalid_labels)
            + ". Choose J, Z, or OTHER."
        )

    if target_sequences <= 0:
        raise SystemExit(
            "The number of sequences must be greater than zero."
        )

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"MediaPipe model not found: {MODEL_PATH}"
        )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the webcam."
        )

    kept_labels = []
    skipped_labels = []
    quit_requested = False

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(
            options
        ) as landmarker:

            for label_number, label in enumerate(
                labels,
                start=1,
            ):
                while True:
                    print(
                        f"\nMotion label "
                        f"{label_number}/{len(labels)}: {label}"
                    )

                    (
                        collected_sequences,
                        quit_during_collection,
                    ) = collect_motion_sequences(
                        contributor_id,
                        label,
                        target_sequences,
                        camera,
                        landmarker,
                    )

                    if quit_during_collection:
                        print(
                            f"\nDiscarded the current unaccepted "
                            f"{label} sequences."
                        )

                        quit_requested = True
                        break

                    choice = ask_what_to_do(
                        len(collected_sequences)
                    )

                    if choice == "keep":
                        saved_files = save_sequences(
                            contributor_id,
                            label,
                            collected_sequences,
                        )

                        kept_labels.append(label)

                        print(
                            f"Saved {len(saved_files)} "
                            f"{label} sequences."
                        )

                        print(
                            f"Saved to: "
                            f"{DATA_DIRECTORY / label}"
                        )

                        break

                    if choice == "redo":
                        print(
                            f"Discarded the sequences. "
                            f"Redoing {label}."
                        )

                        continue

                    if choice == "skip":
                        skipped_labels.append(label)

                        print(
                            f"Discarded the sequences "
                            f"and skipped {label}."
                        )

                        break

                    if choice == "quit":
                        print(
                            f"Discarded the current unaccepted "
                            f"{label} sequences."
                        )

                        quit_requested = True
                        break

                if quit_requested:
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    if quit_requested:
        print("\nMotion collection stopped by the user.")
    else:
        print("\nAll requested motion labels are finished.")

    print(
        "Accepted labels:",
        ", ".join(kept_labels) if kept_labels else "None",
    )

    print(
        "Skipped labels:",
        ", ".join(skipped_labels)
        if skipped_labels
        else "None",
    )

    print("All previously accepted motion data was kept.")
    print("Program closed.")


if __name__ == "__main__":
    main()