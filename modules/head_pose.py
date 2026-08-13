import cv2
import mediapipe as mp
import numpy as np


class HeadPoseEstimator:
    """
    Estimates head orientation (yaw, pitch, roll) using MediaPipe Face Mesh
    landmarks and OpenCV's solvePnP for 3D pose estimation.
    """

    # Landmark indices used for pose estimation (nose tip, chin, eye corners, mouth corners)
    POSE_LANDMARKS = {
        "nose_tip": 1,
        "chin": 152,
        "left_eye_corner": 263,
        "right_eye_corner": 33,
        "left_mouth_corner": 291,
        "right_mouth_corner": 61,
    }

    # Generic 3D face model points (approximate, in mm, arbitrary reference frame)
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),   # Right eye corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ], dtype=np.float64)

    def __init__(self, max_num_faces=1, min_detection_confidence=0.5,
                 yaw_threshold=20.0, pitch_threshold=15.0,
                 turned_duration_threshold_sec=3.0):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.turned_duration_threshold_sec = turned_duration_threshold_sec
        self.turned_start_time = None

    def _get_direction(self, yaw, pitch):
        if yaw > self.yaw_threshold:
            return "Left"
        elif yaw < -self.yaw_threshold:
            return "Right"
        elif pitch > self.pitch_threshold:
            return "Down"
        elif pitch < -self.pitch_threshold:
            return "Up"
        else:
            return "Center"

    def estimate(self, frame, current_time):
        """
        Processes a frame and returns head pose results.

        Args:
            frame: BGR image
            current_time: timestamp (e.g., time.time()) for duration tracking

        Returns:
            dict with:
                - face_detected: bool
                - yaw, pitch, roll: float (degrees), None if no face
                - direction: "Left"/"Right"/"Up"/"Down"/"Center"/None
                - turned_duration: seconds turned away continuously
                - violation: bool (True if turned too long)
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        output = {
            "face_detected": False,
            "yaw": None,
            "pitch": None,
            "roll": None,
            "direction": None,
            "turned_duration": 0.0,
            "violation": False
        }

        if not results.multi_face_landmarks:
            self.turned_start_time = None
            return output

        landmarks = results.multi_face_landmarks[0].landmark

        image_points = np.array([
            (landmarks[self.POSE_LANDMARKS["nose_tip"]].x * w,
             landmarks[self.POSE_LANDMARKS["nose_tip"]].y * h),
            (landmarks[self.POSE_LANDMARKS["chin"]].x * w,
             landmarks[self.POSE_LANDMARKS["chin"]].y * h),
            (landmarks[self.POSE_LANDMARKS["left_eye_corner"]].x * w,
             landmarks[self.POSE_LANDMARKS["left_eye_corner"]].y * h),
            (landmarks[self.POSE_LANDMARKS["right_eye_corner"]].x * w,
             landmarks[self.POSE_LANDMARKS["right_eye_corner"]].y * h),
            (landmarks[self.POSE_LANDMARKS["left_mouth_corner"]].x * w,
             landmarks[self.POSE_LANDMARKS["left_mouth_corner"]].y * h),
            (landmarks[self.POSE_LANDMARKS["right_mouth_corner"]].x * w,
             landmarks[self.POSE_LANDMARKS["right_mouth_corner"]].y * h),
        ], dtype=np.float64)

        # Camera internals (approximate, assuming no lens distortion)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            self.turned_start_time = None
            return output

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pose_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)

        pitch, yaw, roll = euler_angles.flatten()

        # Normalize angles to a more intuitive range
        pitch = self._normalize_angle(pitch)
        yaw = self._normalize_angle(yaw)
        roll = self._normalize_angle(roll)

        direction = self._get_direction(yaw, pitch)

        output["face_detected"] = True
        output["yaw"] = round(yaw, 1)
        output["pitch"] = round(pitch, 1)
        output["roll"] = round(roll, 1)
        output["direction"] = direction

        if direction != "Center":
            if self.turned_start_time is None:
                self.turned_start_time = current_time
            duration = current_time - self.turned_start_time
            output["turned_duration"] = round(duration, 1)
            output["violation"] = duration >= self.turned_duration_threshold_sec
        else:
            self.turned_start_time = None

        return output

    def _normalize_angle(self, angle):
        if angle > 180:
            angle -= 360
        return angle

    def draw_debug(self, frame, result):
        """
        Draws yaw/pitch/roll and violation status on the frame.
        """
        if not result["face_detected"]:
            cv2.putText(frame, "No face for head pose", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        color = (0, 0, 255) if result["violation"] else (0, 255, 0)
        cv2.putText(frame, f"Head: {result['direction']}", (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Yaw:{result['yaw']} Pitch:{result['pitch']}",
                    (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if result["turned_duration"] > 0:
            cv2.putText(frame, f"Turned: {result['turned_duration']}s",
                        (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if result["violation"]:
            cv2.putText(frame, "VIOLATION: Head turned too long!",
                        (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def release(self):
        self.face_mesh.close()