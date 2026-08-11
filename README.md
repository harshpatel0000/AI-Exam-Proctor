# 🎓 ProctorAI — AI-Powered Online Exam Monitoring System

**ProctorAI** is a real-time, computer-vision-based exam proctoring system that uses AI to detect suspicious activity during online exams — without needing a human invigilator watching every second of footage.

Built with **OpenCV, YOLOv8, MediaPipe, and Face Recognition**, it continuously analyzes a student's webcam feed and flags cheating behaviors like phone usage, multiple people in frame, looking away from the screen, and unauthorized face swaps — then compiles everything into a professional PDF violation report at the end of the session.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-CV-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📸 Demo

<!-- Add a screenshot or GIF of the live dashboard here -->
<!-- ![Demo](docs/demo.gif) -->

---

## ✨ Features

ProctorAI performs **six simultaneous AI checks** on every exam session:

| Check | Technology | What It Catches |
|---|---|---|
| 🙂 **Face Detection** | MediaPipe Face Detection | No face / face missing from frame |
| 🪪 **Face Recognition** | `face_recognition` (Dlib 128-D embeddings) | Unknown or mismatched identity |
| 👁️ **Eye Tracking** | MediaPipe Face Mesh (Iris landmarks) | Prolonged looking away from screen |
| 🧭 **Head Pose Estimation** | OpenCV `solvePnP` + Face Mesh | Head turned left/right/up/down |
| 📱 **Phone Detection** | YOLOv8 (COCO) | Mobile phone visible in frame |
| 👥 **Multiple Person Detection** | YOLOv8 (COCO) | More than one person in frame |
| 🚪 **Absence Detection** | Derived from person detection | Student leaves the camera view |

Every violation is:
- ⏱️ Timestamped and cooldown-throttled (so a 30-second violation doesn't spam 30 log entries)
- 📸 Captured as screenshot evidence
- ➕ Scored and aggregated into a live **Risk Score**
- 📊 Compiled into a final **PDF report** with a violation-distribution pie chart

### Risk Scoring

| Violation | Score |
|---|---|
| Unknown Face | +20 |
| Multiple Faces / People | +15 |
| Phone Detected | +10 |
| Student Left Camera | +8 |
| No Face | +8 |
| Looking Away | +5 |
| Head Turned | +5 |

| Total Score | Risk Level |
|---|---|
| 0–10 | ✅ Safe |
| 10–30 | ⚠️ Warning |
| 30–60 | 🟠 Suspicious |
| 60+ | 🔴 High Risk |

---

## 🧠 Tech Stack

- **Language:** Python 3.11
- **Computer Vision:** OpenCV
- **Object Detection:** YOLOv8 (Ultralytics, COCO-pretrained)
- **Face Mesh / Landmarks:** MediaPipe (0.10.21)
- **Face Recognition:** `face_recognition` + Dlib (128-D embeddings)
- **Reporting:** FPDF, Matplotlib, Pandas
- **Numerical:** NumPy, SciPy, scikit-learn

---

## 🏗️ Project Architecture

```
ProctorAI/
│
├── dataset/
│   ├── student_faces/         # Registered student face images
│   ├── embeddings/            # Generated face embeddings
│   └── sample_images/         # Test images
│
├── models/
│   ├── yolov8n.pt              # YOLOv8 nano (person + phone detection)
│   └── face_embeddings.pkl     # Registered student embeddings database
│
├── screenshots/                # Auto-captured violation evidence
├── reports/                    # Generated PDF reports + CSV summaries
├── logs/
│   └── violations.csv          # Full violation event log
│
├── utils/
│   ├── config.py                # Paths, thresholds, tunable settings
│   └── constants.py             # Violation types, score weights, risk levels
│
├── modules/
│   ├── face_detector.py         # Face presence detection
│   ├── face_recognition.py      # Identity verification
│   ├── eye_tracker.py           # Gaze direction + EAR tracking
│   ├── head_pose.py             # Yaw / pitch / roll estimation
│   ├── phone_detector.py        # YOLOv8 phone detection
│   ├── person_detector.py       # YOLOv8 person counting + absence tracking
│   ├── violation_manager.py     # Aggregates all detectors, scores, logs
│   └── report_generator.py      # PDF/CSV report generation
│
├── register_student.py          # Student face registration script
├── app.py                       # Main application entry point
├── requirements.txt
└── README.md
```

### How it works

```
Webcam Feed
     │
     ▼
┌─────────────────────────────────────────────┐
│  Face Detection → Recognition → Eye Tracking │
│  → Head Pose → YOLO (Phone + Person)         │
└─────────────────────────────────────────────┘
     │
     ▼
Violation Manager (scores + logs + screenshots)
     │
     ▼
Live Dashboard Overlay (score, risk level)
     │
     ▼
On exam end → PDF + CSV Report Generation
```

Each detector runs as an **independent, testable module** — heavier models (face recognition, YOLO) run on a frame-skip interval to keep the app responsive on CPU-only machines, while lightweight checks (face detection, eye tracking, head pose) run every frame.

---

## ⚙️ Installation

### Prerequisites
- Python 3.11 (recommended — MediaPipe and Dlib wheels are most stable on this version)
- A working webcam
- Windows, macOS, or Linux

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ProctorAI.git
cd ProctorAI
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **⚠️ Windows users — Dlib will likely fail to build from source** unless you have Visual Studio C++ Build Tools installed. This project uses the prebuilt **`dlib-bin`** wheel instead to avoid that entirely:
>
> ```bash
> pip install dlib-bin
> pip install face_recognition --no-deps
> pip install face-recognition-models Click numpy Pillow colorama
> ```
>
> If you hit `ModuleNotFoundError: No module named 'pkg_resources'` when importing `face_recognition_models`, it's because newer `setuptools` versions dropped `pkg_resources` by default. Fix with:
>
> ```bash
> pip install "setuptools<81"
> ```

### 4. Download the YOLOv8 model

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

Move the downloaded `yolov8n.pt` into the `models/` folder.

### 5. Verify your webcam

```bash
python test_webcam.py
```

A window should open showing your live camera feed — press `q` to close.

---

## 🚀 Usage

### Step 1 — Register a student

Before running an exam session, register the student's face so the system can verify their identity:

```bash
python register_student.py
```

- Enter the student's name
- Press **SPACE** to capture ~5 photos from slightly different angles
- Embeddings are automatically generated and saved to `models/face_embeddings.pkl`

### Step 2 — Start an exam session

```bash
python app.py
```

- Enter the registered student's name
- The live monitoring dashboard opens, showing:
  - Face recognition status
  - Eye gaze & head pose indicators
  - Phone/person detection boxes
  - A running **Score** and **Risk Level** in the top-right corner
- Press **`q`** to end the exam

### Step 3 — Review the report

On exam end, a PDF report is automatically generated in `reports/`, containing:

- Student name, exam date, and duration
- A full violation breakdown table
- Total violations and final risk score
- A color-coded risk level verdict
- A pie chart visualizing the violation distribution

Screenshot evidence for every logged violation is saved in `screenshots/`, and a full timestamped event log is kept in `logs/violations.csv`.

---

## 🧪 Testing Individual Modules

Every detector was built as a standalone, independently testable component. You can run each one in isolation:

```bash
python test_face_detector.py       # Face detection only
python test_face_recognition.py    # Identity verification only
python test_eye_tracker.py         # Gaze tracking only
python test_head_pose.py           # Head orientation only
python test_phone_detector.py      # Phone detection only
python test_person_detector.py     # Person counting + absence only
python test_report_generator.py    # Report generation with sample data
```

---

## 🔧 Configuration

All thresholds and tunable parameters live in `utils/config.py`:

```python
FACE_DETECTION_CONFIDENCE = 0.6
FACE_RECOGNITION_TOLERANCE = 0.5
PHONE_DETECTION_CONFIDENCE = 0.5
PERSON_DETECTION_CONFIDENCE = 0.5

LOOKING_AWAY_THRESHOLD_SEC = 3.0
HEAD_TURNED_THRESHOLD_SEC = 3.0
ABSENCE_THRESHOLD_SEC = 5.0

VIOLATION_COOLDOWN_SEC = 5.0
FACE_RECOGNITION_SKIP_FRAMES = 3
YOLO_SKIP_FRAMES = 3
```

Violation types and score weights live in `utils/constants.py` and can be adjusted to fit different exam risk tolerances.

---

## ⚠️ Known Limitations

- **Phone detection favors screen-on phones.** YOLOv8's COCO-pretrained "cell phone" class was trained predominantly on images where the phone screen is visible/lit — a screen-off phone (dark, featureless rectangle) is harder for the model to distinguish from other small dark objects. In practice this is a reasonable trade-off, since a student needs the screen on to actually read content off it.
- **Head pose estimation uses a generic 3D face model**, not a per-person calibrated one — accuracy can vary slightly with camera angle and distance.
- **Single-camera, single-webcam setup** — the system does not currently support secondary camera angles or screen-recording/tab-switch detection (see Future Improvements).
- **CPU-only performance** — YOLOv8 and Dlib both run measurably faster with GPU (CUDA) support; this project defaults to CPU inference with frame-skipping to stay responsive on standard laptops.

---

## 🛣️ Future Improvements

- [ ] Voice/audio monitoring for suspicious conversations
- [ ] Browser tab-switch detection (desktop app or browser extension)
- [ ] Cloud database integration for centralized report storage
- [ ] Automatic email delivery of reports to administrators
- [ ] Admin dashboard with historical analytics across sessions
- [ ] QR code / OTP-based student authentication
- [ ] Live streaming of sessions to human invigilators
- [ ] Multi-student concurrent monitoring
- [ ] LMS integration (Moodle, Google Classroom)

---

## 📁 Requirements

```txt
opencv-python
numpy
mediapipe==0.10.21
ultralytics
face_recognition
dlib-bin
pandas
matplotlib
fpdf
scikit-learn
scipy
Pillow
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙋 Author

Built by **Harsh Patel** as a portfolio project demonstrating applied skills in Computer Vision, Deep Learning, Object Detection, Face Recognition, Pose Estimation, and AI-based surveillance systems.

If you found this useful, consider ⭐ starring the repo!