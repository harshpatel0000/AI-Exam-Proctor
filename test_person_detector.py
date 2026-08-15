import cv2
import time
from modules.person_detector import PersonDetector

detector = PersonDetector(confidence_threshold=0.5, absence_duration_threshold_sec=5.0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

print("✅ Person detector test running — press 'q' to quit")
print("Try stepping out of frame (5+ sec) or having a 2nd person join briefly")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detector.detect(frame, time.time())
    frame = detector.draw_detections(frame, result)

    cv2.imshow("Person Detector Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()