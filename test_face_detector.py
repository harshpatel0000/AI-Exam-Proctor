import cv2
from modules.face_detector import FaceDetector

detector = FaceDetector(min_detection_confidence=0.6)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

print("✅ Face detector test running — press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detector.detect_faces(frame)
    frame = detector.draw_faces(frame, result["faces"])

    # Show face count on screen
    cv2.putText(
        frame,
        f"Faces detected: {result['face_count']}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.imshow("Face Detector Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.release()
cv2.destroyAllWindows()