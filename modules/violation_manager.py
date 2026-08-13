import cv2
import csv
import os
import time
from datetime import datetime

from utils import constants
from utils import config


class ViolationManager:
    """
    Central module that aggregates outputs from all detectors,
    calculates the violation score, logs violations to CSV,
    and saves screenshot evidence.
    """

    def __init__(self):
        self.total_score = 0
        self.violation_counts = {v: 0 for v in constants.VIOLATION_SCORES.keys()}
        self.last_logged_time = {v: 0 for v in constants.VIOLATION_SCORES.keys()}
        self.session_start_time = datetime.now()

        self._init_csv_log()

    def _init_csv_log(self):
        file_exists = os.path.exists(config.VIOLATIONS_LOG_PATH)
        with open(config.VIOLATIONS_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Violation Type", "Score Added",
                                  "Total Score So Far", "Screenshot Path"])

    def _should_log(self, violation_type, current_time):
        """Cooldown check — avoid logging the same violation every frame."""
        last_time = self.last_logged_time.get(violation_type, 0)
        return (current_time - last_time) >= config.VIOLATION_COOLDOWN_SEC

    def _save_screenshot(self, frame, violation_type, current_time):
        timestamp_str = datetime.fromtimestamp(current_time).strftime("%Y%m%d_%H%M%S")
        safe_type = violation_type.replace(" ", "_")
        filename = f"{safe_type}_{timestamp_str}.jpg"
        filepath = os.path.join(config.SCREENSHOTS_DIR, filename)
        cv2.imwrite(filepath, frame)
        return filepath

    def _log_violation(self, frame, violation_type, current_time):
        if not self._should_log(violation_type, current_time):
            return  # still in cooldown, skip

        score_added = constants.VIOLATION_SCORES[violation_type]
        self.total_score += score_added
        self.violation_counts[violation_type] += 1
        self.last_logged_time[violation_type] = current_time

        screenshot_path = self._save_screenshot(frame, violation_type, current_time)

        with open(config.VIOLATIONS_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S"),
                violation_type,
                score_added,
                self.total_score,
                screenshot_path
            ])

        print(f"⚠️  VIOLATION: {violation_type} (+{score_added}) | Total Score: {self.total_score}")

    def process_frame(self, frame, current_time,
                       face_result=None,
                       recognition_results=None,
                       eye_result=None,
                       head_pose_result=None,
                       phone_result=None,
                       person_result=None):
        """
        Receives outputs from all detector modules for the current frame,
        determines which violations occurred, and logs them.

        Any detector result can be None if it wasn't run this frame
        (e.g., due to frame-skipping for performance).
        """

        # --- Face detection based violations ---
        if face_result is not None:
            if face_result["face_count"] == 0:
                self._log_violation(frame, constants.VIOLATION_NO_FACE, current_time)
            elif face_result["face_count"] > 1:
                self._log_violation(frame, constants.VIOLATION_MULTIPLE_FACES, current_time)

        # --- Face recognition based violations ---
        if recognition_results:
            for res in recognition_results:
                if res["name"] == "Unknown":
                    self._log_violation(frame, constants.VIOLATION_UNKNOWN_FACE, current_time)

        # --- Eye tracking violations ---
        if eye_result is not None and eye_result.get("violation"):
            self._log_violation(frame, constants.VIOLATION_LOOKING_AWAY, current_time)

        # --- Head pose violations ---
        if head_pose_result is not None and head_pose_result.get("violation"):
            self._log_violation(frame, constants.VIOLATION_HEAD_TURNED, current_time)

        # --- Phone detection violations ---
        if phone_result is not None and phone_result.get("phone_detected"):
            self._log_violation(frame, constants.VIOLATION_PHONE_DETECTED, current_time)

        # --- Person detection violations ---
        if person_result is not None:
            if person_result.get("multiple_people_violation"):
                self._log_violation(frame, constants.VIOLATION_MULTIPLE_FACES, current_time)
            if person_result.get("absence_violation"):
                self._log_violation(frame, constants.VIOLATION_STUDENT_ABSENT, current_time)

    def get_summary(self):
        """
        Returns a summary dict for report generation.
        """
        return {
            "total_score": self.total_score,
            "risk_level": constants.get_risk_level(self.total_score),
            "violation_counts": self.violation_counts,
            "session_start_time": self.session_start_time,
            "session_end_time": datetime.now()
        }