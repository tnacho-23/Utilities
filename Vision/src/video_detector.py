import cv2
from ultralytics import YOLO

video_path = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/images/FUP_Mall/FUP1/easton4.mp4"
weights_path = "C:/Users/ignac/Escritorio/DualVision/Utilities/Vision/workbench/weights/FUP_mall_y26mV1.pt"

model = YOLO(weights_path)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: could not open video {video_path}")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated = results[0].plot()

    display = cv2.resize(annotated, (1280, 720))
    cv2.imshow("Detections", display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
