import cv2
import mediapipe as mp


class FaceDetector:
    """
    Detects faces in a video frame using MediaPipe Face Detection.
    Returns face count, bounding boxes, and confidence scores.
    """

    def __init__(self, min_detection_confidence=0.6):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 = short-range (best for webcam, <2m)
            min_detection_confidence=min_detection_confidence
        )

    def detect_faces(self, frame):
        """
        Detects faces in the given frame.

        Args:
            frame: BGR image (OpenCV format)

        Returns:
            dict with:
                - face_count: int
                - faces: list of dicts, each containing bbox (x, y, w, h) and confidence
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)

        faces = []
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bbox_rel = detection.location_data.relative_bounding_box
                x = int(bbox_rel.xmin * w)
                y = int(bbox_rel.ymin * h)
                box_w = int(bbox_rel.width * w)
                box_h = int(bbox_rel.height * h)
                confidence = detection.score[0]

                faces.append({
                    "bbox": (x, y, box_w, box_h),
                    "confidence": round(confidence, 2)
                })

        return {
            "face_count": len(faces),
            "faces": faces
        }

    def draw_faces(self, frame, faces):
        """
        Draws bounding boxes and confidence labels on the frame for visualization.
        """
        for face in faces:
            x, y, w, h = face["bbox"]
            conf = face["confidence"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Face {conf}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        return frame

    def release(self):
        self.face_detector.close()