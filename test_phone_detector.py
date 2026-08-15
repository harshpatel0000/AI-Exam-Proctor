import cv2
from modules.phone_detector import PhoneDetector

detector = PhoneDetector(confidence_threshold=0.5)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

print("✅ Phone detector test running — press 'q' to quit")
print("Hold up your phone to the camera to test detection")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detector.detect(frame)
    frame = detector.draw_detections(frame, result["detections"])

    if result["phone_detected"]:
        cv2.putText(frame, "VIOLATION: Phone detected!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Phone Detector Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()