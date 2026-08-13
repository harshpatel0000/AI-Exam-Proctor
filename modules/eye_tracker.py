import cv2
import mediapipe as mp
import numpy as np


class EyeTracker:
    """
    Tracks eye landmarks using MediaPipe Face Mesh to estimate
    gaze direction and detect prolonged looking-away behavior.
    """

    # MediaPipe Face Mesh landmark indices for eyes
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    # Iris landmarks (require refine_landmarks=True)
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    def __init__(self, max_num_faces=1, min_detection_confidence=0.5,
                 looking_away_threshold_sec=3.0):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,  # needed for iris tracking
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )
        self.looking_away_threshold_sec = looking_away_threshold_sec
        self.away_start_time = None

    def _landmark_to_point(self, landmark, w, h):
        return int(landmark.x * w), int(landmark.y * h)

    def _eye_aspect_ratio(self, landmarks, eye_indices, w, h):
        pts = [self._landmark_to_point(landmarks[i], w, h) for i in eye_indices]
        pts = np.array(pts)

        # Vertical distances
        vertical_1 = np.linalg.norm(pts[1] - pts[5])
        vertical_2 = np.linalg.norm(pts[2] - pts[4])
        # Horizontal distance
        horizontal = np.linalg.norm(pts[0] - pts[3])

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)
        return ear

    def _gaze_direction(self, landmarks, w, h):
        """
        Estimates horizontal/vertical gaze direction using iris position
        relative to eye corners.
        """
        left_iris = np.mean(
            [self._landmark_to_point(landmarks[i], w, h) for i in self.LEFT_IRIS],
            axis=0
        )
        right_iris = np.mean(
            [self._landmark_to_point(landmarks[i], w, h) for i in self.RIGHT_IRIS],
            axis=0
        )

        left_corner_l = self._landmark_to_point(landmarks[self.LEFT_EYE[0]], w, h)
        left_corner_r = self._landmark_to_point(landmarks[self.LEFT_EYE[3]], w, h)
        right_corner_l = self._landmark_to_point(landmarks[self.RIGHT_EYE[0]], w, h)
        right_corner_r = self._landmark_to_point(landmarks[self.RIGHT_EYE[3]], w, h)

        # Horizontal ratio (0 = looking left corner, 1 = looking right corner)
        left_ratio = (left_iris[0] - left_corner_r[0]) / (left_corner_l[0] - left_corner_r[0] + 1e-6)
        right_ratio = (right_iris[0] - right_corner_l[0]) / (right_corner_r[0] - right_corner_l[0] + 1e-6)
        horizontal_ratio = (left_ratio + right_ratio) / 2.0

        if horizontal_ratio < 0.35:
            h_direction = "Right"
        elif horizontal_ratio > 0.65:
            h_direction = "Left"
        else:
            h_direction = "Center"

        return h_direction

    def track(self, frame, current_time):
        """
        Processes a frame and returns eye tracking results.

        Args:
            frame: BGR image
            current_time: timestamp (e.g., time.time()) for duration tracking

        Returns:
            dict with:
                - eyes_detected: bool
                - ear: average eye aspect ratio (None if no face)
                - gaze_direction: "Left" / "Right" / "Center" / None
                - looking_away_duration: seconds looking away continuously
                - violation: bool (True if looking away too long)
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        output = {
            "eyes_detected": False,
            "ear": None,
            "gaze_direction": None,
            "looking_away_duration": 0.0,
            "violation": False
        }

        if not results.multi_face_landmarks:
            self.away_start_time = None
            return output

        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = self._eye_aspect_ratio(landmarks, self.LEFT_EYE, w, h)
        right_ear = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        gaze = self._gaze_direction(landmarks, w, h)

        output["eyes_detected"] = True
        output["ear"] = round(avg_ear, 3)
        output["gaze_direction"] = gaze

        # Track how long the student has been looking away
        if gaze != "Center":
            if self.away_start_time is None:
                self.away_start_time = current_time
            duration = current_time - self.away_start_time
            output["looking_away_duration"] = round(duration, 1)
            output["violation"] = duration >= self.looking_away_threshold_sec
        else:
            self.away_start_time = None

        return output

    def draw_debug(self, frame, result):
        """
        Draws EAR, gaze direction, and violation status on the frame.
        """
        if not result["eyes_detected"]:
            cv2.putText(frame, "No eyes detected", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        color = (0, 0, 255) if result["violation"] else (0, 255, 0)
        cv2.putText(frame, f"Gaze: {result['gaze_direction']}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"EAR: {result['ear']}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if result["looking_away_duration"] > 0:
            cv2.putText(frame, f"Away: {result['looking_away_duration']}s",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if result["violation"]:
            cv2.putText(frame, "VIOLATION: Looking away too long!",
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def release(self):
        self.face_mesh.close()