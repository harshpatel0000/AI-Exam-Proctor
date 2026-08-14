import cv2
from modules.face_recognition import StudentFaceRecognizer

recognizer = StudentFaceRecognizer(tolerance=0.5)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

print("✅ Face recognition test running — press 'q' to quit")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Run recognition every 3rd frame for performance (face_recognition is slower than detection)
    if frame_count % 3 == 0:
        results = recognizer.recognize(frame)
        last_results = results
    else:
        results = last_results if frame_count > 3 else []

    frame = recognizer.draw_results(frame, results)

    cv2.imshow("Face Recognition Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()