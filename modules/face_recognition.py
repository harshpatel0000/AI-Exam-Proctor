import face_recognition
import pickle
import os
import numpy as np


class StudentFaceRecognizer:
    """
    Verifies student identity by comparing live webcam faces
    against registered face embeddings.
    """

    def __init__(self, embeddings_path=os.path.join("models", "face_embeddings.pkl"),
                 tolerance=0.5):
        """
        Args:
            embeddings_path: path to the saved embeddings database
            tolerance: lower = stricter match (0.5 is a good default;
                       face_recognition's own default is 0.6)
        """
        self.tolerance = tolerance
        self.known_encodings = []
        self.known_names = []

        self._load_database(embeddings_path)

    def _load_database(self, embeddings_path):
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"❌ No embeddings database found at '{embeddings_path}'. "
                f"Run register_student.py first."
            )

        with open(embeddings_path, "rb") as f:
            database = pickle.load(f)

        for name, encodings in database.items():
            for encoding in encodings:
                self.known_encodings.append(encoding)
                self.known_names.append(name)

        print(f"✅ Loaded {len(self.known_names)} face encodings "
              f"for {len(set(self.known_names))} registered student(s)")

    def recognize(self, frame):
        """
        Detects and identifies faces in the given frame.

        Args:
            frame: BGR image (OpenCV format)

        Returns:
            list of dicts, each containing:
                - name: matched student name, or "Unknown"
                - confidence: match confidence (0-1, higher = better match)
                - bbox: (top, right, bottom, left) face location
        """
        # face_recognition expects RGB
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        results = []
        for encoding, location in zip(face_encodings, face_locations):
            name = "Unknown"
            confidence = 0.0

            if self.known_encodings:
                distances = face_recognition.face_distance(self.known_encodings, encoding)
                best_match_index = np.argmin(distances)
                best_distance = distances[best_match_index]

                if best_distance <= self.tolerance:
                    name = self.known_names[best_match_index]
                    confidence = round(1 - best_distance, 2)

            results.append({
                "name": name,
                "confidence": confidence,
                "bbox": location  # (top, right, bottom, left)
            })

        return results

    def draw_results(self, frame, results):
        """
        Draws bounding boxes and identity labels on the frame.
        """
        import cv2
        for res in results:
            top, right, bottom, left = res["bbox"]
            name = res["name"]
            conf = res["confidence"]

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({conf})" if name != "Unknown" else "Unknown"

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame