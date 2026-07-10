import time
from pathlib import Path 

import cv2
import mediapipe as mp 

MODEL_PATH = Path("models/hand_landmarker.task")

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), 
    (0,5), (5,6), (6,7), (7,8), 
    (5,9), (9,10), (10, 11), (11, 12), 
    (9,13), (13,14), (14,15), (15,16), 
    (13,17), (17,18), (18,19), (19,20), 
    (0,17),
]

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options = mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode = mp.tasks.vision.RunningMode.VIDEO,
    num_hands = 2,
    min_hand_detection_confidence = 0.5,
    min_hand_presence_confidence = 0.5,
    min_tracking_confidence = 0.5,
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open the webcam.")

try: 
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = camera.read()
            
            if not success:
                break
            
            # Flip image to behave as mirror
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(
                image_format = mp.ImageFormat.SRGB,
                data = rgb_frame,
            )
            
            timestamp_ms = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            height, width = frame.shape[:2]
            
            for hand in result.hand_landmarks:
                points = [
                    (int(point.x * width), int(point.y * height))
                    for point in hand
                ]
                
                for start, end in HAND_CONNECTIONS:
                    cv2.line(
                        frame, 
                        points[start], 
                        points[end], 
                        (0, 255, 0),
                        2, 
                    )
                
                for point in points:
                    cv2.circle(frame, point, 4, (0,0,255), -1)
                    
            cv2.putText(
                frame, 
                f"Hands detected: {len(result.hand_landmarks)}", 
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, 
                (255, 0, 0), 
                2, 
            )
            
            cv2.imshow("SIGNET hand tracker", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break 
            
finally: 
    camera.release()
    cv2.destroyAllWindows()