import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")
FACE_EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "face_embeddings.pkl")
VIOLATIONS_LOG_PATH = os.path.join(LOGS_DIR, "violations.csv")
SYSTEM_LOG_PATH = os.path.join(LOGS_DIR, "system.log")

# Detection thresholds
FACE_DETECTION_CONFIDENCE = 0.6
FACE_RECOGNITION_TOLERANCE = 0.5
PHONE_DETECTION_CONFIDENCE = 0.5
PERSON_DETECTION_CONFIDENCE = 0.5

LOOKING_AWAY_THRESHOLD_SEC = 3.0
HEAD_TURNED_THRESHOLD_SEC = 3.0
ABSENCE_THRESHOLD_SEC = 5.0

# Violation cooldown — minimum seconds between logging the SAME violation type,
# to avoid spamming the log/screenshots every single frame during a sustained violation
VIOLATION_COOLDOWN_SEC = 5.0

# Frame skip intervals for performance (run heavier models every Nth frame)
FACE_RECOGNITION_SKIP_FRAMES = 3
YOLO_SKIP_FRAMES = 3

# Ensure directories exist
for directory in [MODELS_DIR, SCREENSHOTS_DIR, REPORTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)