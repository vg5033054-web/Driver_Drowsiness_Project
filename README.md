# 🚗 Driver Drowsiness Detection System

Real-time AI Driver Safety Monitoring, Computer Vision & Biometric Telemetry Dashboard.

---

## ✨ Features & Capabilities

- **Real-Time Webcam Telemetry**: Live biometric tracking with Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) computed using MediaPipe 468+ facial landmark mesh.
- **Hybrid AI Decision Engine**: Combines continuous eye-closure duration safety thresholds with an ensemble **Random Forest Classifier** trained on facial biometrics.
- **Audible Warning System**: Instant audio warning in browser with custom track support (`funny_munni.mp3`) and synthesized emergency alarm sirens.
- **Incident Logger**: Automatic timestamped event logging saved to `logs/drowsiness_events.csv` with export capabilities.
- **Interactive Telemetry Dashboard**:
  - 🎥 **Live Video Stream**: Real-time webcam feed with biometric HUD overlays and status gauges.
  - 📸 **Instant Snapshot Test**: Test facial expressions directly with the browser camera or uploaded images.
  - 📊 **Incident History & Analytics**: Hourly incident frequency chart, event logs, and CSV download.
  - 🧠 **Model & Dataset Explorer**: Training data distribution, feature importance analysis, and interactive "What-If" driver simulator.
  - ⚙️ **Custom Calibration**: Fine-tune EAR threshold, alarm trigger delay, yawn threshold, and alert audio.

---

## 🚀 Quick Start (Streamlit Frontend)

### 1. Activate the Virtual Environment

**In PowerShell:**
```powershell
. ./.venv/bin/Activate.ps1
```

**In Bash:**
```bash
source .venv/bin/activate
```

### 2. Launch the Streamlit App

```bash
streamlit run app.py
```
Or directly using the virtual environment python:
```bash
./.venv/bin/streamlit run app.py
```

The application will be accessible at: **`http://localhost:8501`**

---

## 📁 Project Structure

```text
Mini Project/
├── app.py                                  # Streamlit Frontend Web Application
├── Driver_Drowsiness_Detection_Jupyter.ipynb # Jupyter Notebook Pipeline
├── funny_munni.mp3                         # Audio Alert Sound Track
├── requirements.txt                        # Python Dependencies
├── README.md                               # Project Documentation
├── data/
│   └── features.csv                        # Collected Training Dataset
├── models/
│   ├── drowsiness_model.joblib             # Trained Random Forest Model
│   └── face_landmarker.task               # MediaPipe Face Landmark Model
└── logs/
    └── drowsiness_events.csv               # Incident History Log
```
