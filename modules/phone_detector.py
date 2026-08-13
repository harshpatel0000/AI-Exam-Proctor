import cv2
from ultralytics import YOLO
import os


class PhoneDetector:
    """
    Detects mobile phones in a video frame using YOLOv8
    (pretrained on COCO dataset — class 67 = 'cell phone').
    """

    COCO_CELL_PHONE_CLASS_ID = 67

    def __init__(self, model_path=os.path.join("models", "yolov8n.pt"),
                 confidence_threshold=0.5):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"❌ YOLO model not found at '{model_path}'. "
                f"Make sure yolov8n.pt is in the models/ folder."
            )
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        """
        Detects phones in the given frame.

        Args:
            frame: BGR image (OpenCV format)

        Returns:
            dict with:
                - phone_detected: bool
                - detections: list of dicts, each with bbox (x1,y1,x2,y2) and confidence
        """
        results = self.model(
            frame,
            classes=[self.COCO_CELL_PHONE_CLASS_ID],
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

        return {
            "phone_detected": len(detections) > 0,
            "detections": detections
        }

    def draw_detections(self, frame, detections):
        """
        Draws bounding boxes around detected phones.
        """
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"Phone {conf}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
        return frame