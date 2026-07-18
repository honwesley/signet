import argparse
import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp


MODEL_PATH = Path("models/hand_landmarker.task")
DATA_FILE = Path("data/landmarks.csv")
STATIC_LETTERS = set("ABCDEFGHIKLMNOPQRSTUVWXY") | {"?"}


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
        description="Collect ASL data from another contributor."
    )

    parser.add_argument(
        "contributor_id",
        help="Anonymous contributor ID, such as person_02",
    )

    parser.add_argument(
        "letters",
        nargs="+",
        help="One or more static ASL letters",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples per letter",
    )

    return parser.parse_args()


def ask_what_to_do(sample_count):
    while True:
        print(f"\nCollected {sample_count} samples.")

        answer = input(
            "[Y] Keep  [N] Redo  [S] Skip  [0] Quit: "
        ).strip().lower()

        if answer in {"y", "yes"}:
            if sample_count == 0:
                print("There are no samples to keep.")
                continue

            return "keep"

        if answer in {"n", "no"}:
            return "redo"

        if answer in {"s", "skip"}:
            return "skip"

        if answer == "0":
            return "quit"

        print("Please enter Y, N, S, or 0.")


def save_samples(label, collected_samples):
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    file_exists = (
        DATA_FILE.exists()
        and DATA_FILE.stat().st_size > 0
    )

    with DATA_FILE.open("a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(
                ["label"]
                + [f"feature_{number}" for number in range(63)]
            )

        for features in collected_samples:
            writer.writerow([label] + features)


def collect_letter(
    contributor_id,
    label,
    target_samples,
    camera,
    landmarker,
):
    collected_samples = []
    recording = False
    quit_requested = False

    last_saved_time = 0.0
    last_timestamp_ms = 0
    sample_delay = 0.08

    while len(collected_samples) < target_samples:
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

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

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

            current_time = time.monotonic()

            if (
                recording
                and current_time - last_saved_time
                >= sample_delay
            ):
                features = normalize_landmarks(hand)

                if features is not None:
                    collected_samples.append(features)
                    last_saved_time = current_time

        status = (
            "RECORDING - change angle and distance"
            if recording
            else "Press SPACE to start or pause"
        )

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
            f"Letter: {label} | Samples: "
            f"{len(collected_samples)}/{target_samples}",
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
            0.65,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            "Q = finish this letter early",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "0 = quit entire collection",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        cv2.imshow(
            "SIGNET Contributor Data Collector",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("0"):
            quit_requested = True
            break

        if key == 32:
            recording = not recording

    cv2.destroyAllWindows()

    return collected_samples, quit_requested


def main():
    args = parse_arguments()

    contributor_id = args.contributor_id.strip()
    letters = [letter.upper() for letter in args.letters]
    target_samples = args.samples

    if not contributor_id:
        raise SystemExit("Contributor ID cannot be empty.")

    invalid_letters = [
        letter
        for letter in letters
        if letter not in STATIC_LETTERS
    ]

    if invalid_letters:
        raise SystemExit(
            "Invalid static letters: "
            + ", ".join(invalid_letters)
            + ". J and Z require motion data."
        )

    if target_samples <= 0:
        raise SystemExit(
            "The number of samples must be greater than zero."
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
        raise RuntimeError("Could not open the webcam.")

    kept_letters = []
    skipped_letters = []
    quit_requested = False

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(
            options
        ) as landmarker:

            for letter_number, label in enumerate(
                letters,
                start=1,
            ):
                while True:
                    print(
                        f"\nLetter {letter_number}/"
                        f"{len(letters)}: {label}"
                    )

                    (
                        collected_samples,
                        quit_during_collection,
                    ) = collect_letter(
                        contributor_id,
                        label,
                        target_samples,
                        camera,
                        landmarker,
                    )

                    if quit_during_collection:
                        print(
                            f"\nDiscarded the current unaccepted "
                            f"samples for {label}."
                        )

                        quit_requested = True
                        break

                    choice = ask_what_to_do(
                        len(collected_samples)
                    )

                    if choice == "keep":
                        save_samples(
                            label,
                            collected_samples,
                        )

                        kept_letters.append(label)

                        print(
                            f"Added {len(collected_samples)} "
                            f"samples of {label} to {DATA_FILE}."
                        )

                        break

                    if choice == "redo":
                        print(
                            f"Discarded the samples. Redoing {label}."
                        )

                        continue

                    if choice == "skip":
                        skipped_letters.append(label)

                        print(
                            f"Discarded the samples and skipped {label}."
                        )

                        break

                    if choice == "quit":
                        print(
                            f"Discarded the current unaccepted "
                            f"samples for {label}."
                        )

                        quit_requested = True
                        break

                if quit_requested:
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    if quit_requested:
        print("\nCollection stopped by the user.")
    else:
        print("\nAll requested letters are finished.")

    print(
        "Accepted letters:",
        ", ".join(kept_letters) if kept_letters else "None",
    )

    print(
        "Skipped letters:",
        ", ".join(skipped_letters) if skipped_letters else "None",
    )

    print("All previously accepted data was kept.")
    print("Program closed.")


if __name__ == "__main__":
    main()