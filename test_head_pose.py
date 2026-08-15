import cv2
import time
from modules.head_pose import HeadPoseEstimator

estimator = HeadPoseEstimator(turned_duration_threshold_sec=3.0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

print("✅ Head pose test running — press 'q' to quit")
print("Try turning your head left/right/up/down and holding for 3+ seconds")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = estimator.estimate(frame, time.time())
    frame = estimator.draw_debug(frame, result)

    cv2.imshow("Head Pose Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
estimator.release()
cv2.destroyAllWindows()