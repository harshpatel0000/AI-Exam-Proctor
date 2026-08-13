import cv2
from ultralytics import YOLO
import os


class PersonDetector:
    """
    Detects and counts people in a video frame using YOLOv8
    (pretrained on COCO dataset — class 0 = 'person').

    Flags violations for:
        - Multiple people in frame (unauthorized assistance)
        - Zero people in frame (student left camera) — tracked with duration
    """

    COCO_PERSON_CLASS_ID = 0

    def __init__(self, model_path=os.path.join("models", "yolov8n.pt"),
                 confidence_threshold=0.5,
                 absence_duration_threshold_sec=5.0):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"❌ YOLO model not found at '{model_path}'. "
                f"Make sure yolov8n.pt is in the models/ folder."
            )
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.absence_duration_threshold_sec = absence_duration_threshold_sec
        self.absence_start_time = None

    def detect(self, frame, current_time):
        """
        Detects and counts people in the given frame.

        Args:
            frame: BGR image (OpenCV format)
            current_time: timestamp (e.g., time.time()) for absence duration tracking

        Returns:
            dict with:
                - person_count: int
                - detections: list of dicts, each with bbox and confidence
                - multiple_people_violation: bool
                - absence_duration: seconds student has been absent
                - absence_violation: bool
        """
        results = self.model(
            frame,
            classes=[self.COCO_PERSON_CLASS_ID],
            conf=self.confidence_threshold,
            verbose=False
        )

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": round(confidence, 2)
                })

        person_count = len(detections)

        # Track absence duration (0 people detected)
        if person_count == 0:
            if self.absence_start_time is None:
                self.absence_start_time = current_time
            absence_duration = current_time - self.absence_start_time
            absence_violation = absence_duration >= self.absence_duration_threshold_sec
        else:
            self.absence_start_time = None
            absence_duration = 0.0
            absence_violation = False

        return {
            "person_count": person_count,
            "detections": detections,
            "multiple_people_violation": person_count > 1,
            "absence_duration": round(absence_duration, 1),
            "absence_violation": absence_violation
        }

    def draw_detections(self, frame, result):
        """
        Draws bounding boxes around detected people and status text.
        """
        for det in result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            color = (0, 165, 255) if result["multiple_people_violation"] else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"Person {conf}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.putText(frame, f"People: {result['person_count']}", (10, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if result["multiple_people_violation"]:
            cv2.putText(frame, "VIOLATION: Multiple people detected!", (10, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if result["absence_duration"] > 0:
            color = (0, 0, 255) if result["absence_violation"] else (0, 165, 255)
            cv2.putText(frame, f"Absent: {result['absence_duration']}s", (10, 390),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if result["absence_violation"]:
            cv2.putText(frame, "VIOLATION: Student left camera!", (10, 420),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame