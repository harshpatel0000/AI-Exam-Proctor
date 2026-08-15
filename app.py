import cv2
import time
import sys

from modules.face_detector import FaceDetector
from modules.face_recognition import StudentFaceRecognizer
from modules.eye_tracker import EyeTracker
from modules.head_pose import HeadPoseEstimator
from modules.phone_detector import PhoneDetector
from modules.person_detector import PersonDetector
from modules.violation_manager import ViolationManager
from modules.report_generator import ReportGenerator

from ultralytics import YOLO
from utils import config


class ExamProctorApp:
    def __init__(self, student_name):
        self.student_name = student_name
        print("🔄 Loading models... this may take a few seconds")

        # --- Load lightweight detectors ---
        self.face_detector = FaceDetector(
            min_detection_confidence=config.FACE_DETECTION_CONFIDENCE
        )
        self.recognizer = StudentFaceRecognizer(
            embeddings_path=config.FACE_EMBEDDINGS_PATH,
            tolerance=config.FACE_RECOGNITION_TOLERANCE
        )
        self.eye_tracker = EyeTracker(
            looking_away_threshold_sec=config.LOOKING_AWAY_THRESHOLD_SEC
        )
        self.head_pose = HeadPoseEstimator(
            turned_duration_threshold_sec=config.HEAD_TURNED_THRESHOLD_SEC
        )

        # --- Shared YOLO model for phone + person detection (avoids double loading) ---
        shared_yolo = YOLO(config.YOLO_MODEL_PATH)
        self.phone_detector = PhoneDetector(confidence_threshold=config.PHONE_DETECTION_CONFIDENCE)
        self.phone_detector.model = shared_yolo  # reuse shared instance
        self.person_detector = PersonDetector(
            confidence_threshold=config.PERSON_DETECTION_CONFIDENCE,
            absence_duration_threshold_sec=config.ABSENCE_THRESHOLD_SEC
        )
        self.person_detector.model = shared_yolo  # reuse shared instance

        # --- Violation manager ---
        self.violation_manager = ViolationManager()

        # --- Frame skip counters ---
        self.frame_count = 0
        self.last_recognition_results = []
        self.last_phone_result = {"phone_detected": False, "detections": []}
        self.last_person_result = {
            "person_count": 0, "detections": [],
            "multiple_people_violation": False,
            "absence_duration": 0.0, "absence_violation": False
        }

        print("✅ All models loaded successfully")

    def _draw_dashboard(self, frame, summary):
        """Draws a live status dashboard on the top-right of the frame."""
        h, w, _ = frame.shape
        overlay_x = w - 250

        cv2.rectangle(frame, (overlay_x - 10, 10), (w - 10, 130), (0, 0, 0), -1)
        cv2.rectangle(frame, (overlay_x - 10, 10), (w - 10, 130), (255, 255, 255), 1)

        risk_colors = {
            "Safe": (0, 200, 0),
            "Warning": (0, 200, 200),
            "Suspicious": (0, 140, 255),
            "High Risk": (0, 0, 255)
        }
        risk_level = summary["risk_level"]
        color = risk_colors.get(risk_level, (255, 255, 255))

        cv2.putText(frame, f"Student: {self.student_name}", (overlay_x, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"Score: {summary['total_score']}", (overlay_x, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Risk: {risk_level}", (overlay_x, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, "Press 'q' to end exam", (overlay_x, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access webcam")
            sys.exit(1)

        print(f"\n🎥 Exam monitoring started for '{self.student_name}'")
        print("Press 'q' to end the exam and generate the report.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Webcam frame not received, stopping.")
                break

            current_time = time.time()
            self.frame_count += 1

            # --- Face detection (every frame, lightweight) ---
            face_result = self.face_detector.detect_faces(frame)
            frame = self.face_detector.draw_faces(frame, face_result["faces"])

            # --- Face recognition (every Nth frame, heavier) ---
            if self.frame_count % config.FACE_RECOGNITION_SKIP_FRAMES == 0:
                self.last_recognition_results = self.recognizer.recognize(frame)
            frame = self.recognizer.draw_results(frame, self.last_recognition_results)

            # --- Eye tracking (every frame) ---
            eye_result = self.eye_tracker.track(frame, current_time)
            frame = self.eye_tracker.draw_debug(frame, eye_result)

            # --- Head pose (every frame) ---
            head_result = self.head_pose.estimate(frame, current_time)
            frame = self.head_pose.draw_debug(frame, head_result)

            # --- Phone + Person detection (every Nth frame, heaviest - shared YOLO) ---
            if self.frame_count % config.YOLO_SKIP_FRAMES == 0:
                self.last_phone_result = self.phone_detector.detect(frame)
                self.last_person_result = self.person_detector.detect(frame, current_time)
            frame = self.phone_detector.draw_detections(frame, self.last_phone_result["detections"])
            frame = self.person_detector.draw_detections(frame, self.last_person_result)

            # --- Feed everything into the violation manager ---
            self.violation_manager.process_frame(
                frame, current_time,
                face_result=face_result,
                recognition_results=self.last_recognition_results,
                eye_result=eye_result,
                head_pose_result=head_result,
                phone_result=self.last_phone_result,
                person_result=self.last_person_result
            )

            # --- Live dashboard ---
            summary = self.violation_manager.get_summary()
            frame = self._draw_dashboard(frame, summary)

            cv2.imshow("AI Smart Exam Proctor", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        self._finish_exam()

    def _finish_exam(self):
        print("\n🏁 Exam ended. Generating final report...")

        self.face_detector.release()
        self.eye_tracker.release()
        self.head_pose.release()

        summary = self.violation_manager.get_summary()
        generator = ReportGenerator(student_name=self.student_name)
        pdf_path = generator.generate_pdf(summary)
        generator.generate_summary_csv(summary)

        print(f"\n📊 FINAL RESULTS for {self.student_name}")
        print(f"   Total Score: {summary['total_score']}")
        print(f"   Risk Level: {summary['risk_level']}")
        print(f"   Report saved to: {pdf_path}")


if __name__ == "__main__":
    name = input("Enter student name for this exam session: ").strip()
    if not name:
        print("❌ Student name cannot be empty")
        sys.exit(1)

    app = ExamProctorApp(student_name=name)
    app.run()