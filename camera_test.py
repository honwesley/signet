import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open the webcam.")

while True:
    success, frame = camera.read();
    
    if not success:
        break;
    
    cv2.putText(
        frame, 
        "Press 0 to close",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,250,0),
        2,
    )
    
    cv2.imshow("SIGNET Camera Test", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
camera.release()
cv2.destroyAllWindows()