# Violation types
VIOLATION_UNKNOWN_FACE = "Unknown Face"
VIOLATION_NO_FACE = "No Face"
VIOLATION_MULTIPLE_FACES = "Multiple Faces"
VIOLATION_PHONE_DETECTED = "Phone Detected"
VIOLATION_LOOKING_AWAY = "Looking Away"
VIOLATION_HEAD_TURNED = "Head Turned"
VIOLATION_STUDENT_ABSENT = "Student Left Camera"

# Violation score weights
VIOLATION_SCORES = {
    VIOLATION_UNKNOWN_FACE: 20,
    VIOLATION_NO_FACE: 8,
    VIOLATION_MULTIPLE_FACES: 15,
    VIOLATION_PHONE_DETECTED: 10,
    VIOLATION_LOOKING_AWAY: 5,
    VIOLATION_HEAD_TURNED: 5,
    VIOLATION_STUDENT_ABSENT: 8,
}

# Risk level thresholds
RISK_SAFE_MAX = 10
RISK_WARNING_MAX = 30
RISK_SUSPICIOUS_MAX = 60
# 60+ = High Risk

def get_risk_level(score):
    if score <= RISK_SAFE_MAX:
        return "Safe"
    elif score <= RISK_WARNING_MAX:
        return "Warning"
    elif score <= RISK_SUSPICIOUS_MAX:
        return "Suspicious"
    else:
        return "High Risk"