import sys
from pathlib import Path 

import cv2 
import mediapipe as mp 
import numpy as np 

HAND_MODEL_PATH = Path("models/hand_landmarker.task")
SEQUENCE_LENGTH = 30
TARGET_SEQUENCES = 10

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
        for x,y,z in relative_points 
    )
    
    if scale < 0.000001: 
        return None 
    
    features = []
    
    for x,y,z in relative_points:
        features.extend([x/scale, y/scale, z/scale])
    
    return features

if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python collect_motion_data.py J, Z, or OTHER"
    )
    
label = sys.argv[1].upper()

if label not in {"J", "Z", "OTHER"}:
    raise SystemExit("Choose J, Z, or OTHER.")

output_directory = Path("data/motion") / label 
output_directory.mkdir(parents = True, exist_ok = True)

saved_count = len(list(output_directory.glob("*.npy")))

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options = mp.tasks.BaseOptions(
        model_asset_path = str(HAND_MODEL_PATH)
    ),
    running_mode = mp.tasks.vision.RunningMode.VIDEO,
    num_hands = 1
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

frame_timestamp_ms = 0
recording = False 
sequence = []

try: 
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while saved_count < TARGET_SEQUENCES:
            success, frame = camera.read()
            
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(
                image_format = mp.ImageFormat.SRGB,
                data = rgb_frame,
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
                    cv2.circle(frame, (x,y), 4, (0, 255, 0), -1)
            
            if recording:
                if hand is None:
                    recording = False
                    sequence = []
                else:
                    features = normalize_landmarks(hand)
                    
                    if features is not None:
                        sequence.append(features)
                
                if len(sequence) == SEQUENCE_LENGTH:
                    filename = output_directory / f"{saved_count:04d}.npy"
                    
                    np.save(
                        filename,
                        np.asarray(sequence, dtype=np.float32)
                    )
                    saved_count += 1
                    recording = False
                    sequence = []
                    
            status = (
                f"Recording: {len(sequence)}/{SEQUENCE_LENGTH}"
                if recording
                else "Press SPACE to record"
            )
            
            cv2.putText(
                frame, 
                f"{label}: {saved_count}/{TARGET_SEQUENCES}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, 
                (0,255,0),
                2,
            )
            
            cv2.putText(
                frame, 
                status, 
                (20,80), 
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, 
                (0,0,255),
                2,
            )
            
            cv2.imshow("SIGNET Motion Collector", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):
                break
            
            if key == 32 and not recording and hand is not None:
                sequence = []
                recording = True

finally: 
    camera.release()
    cv2.destroyAllWindows()
    
print(f"{label} sequences saved: {saved_count}")