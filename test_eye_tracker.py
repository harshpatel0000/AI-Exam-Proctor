import cv2
import time
from modules.eye_tracker import EyeTracker

tracker = EyeTracker(looking_away_threshold_sec=3.0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

print("✅ Eye tracker test running — press 'q' to quit")
print("Try looking left/right and holding it for 3+ seconds to trigger a violation")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = tracker.track(frame, time.time())
    frame = tracker.draw_debug(frame, result)

    cv2.imshow("Eye Tracker Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
tracker.release()
cv2.destroyAllWindows()