"""
Driver Drowsiness Detection System - Streamlit Dashboard
A simple, clean, and real-time computer vision application to detect driver
drowsiness using MediaPipe Face Mesh, Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR),
and a Random Forest Classifier. Supports both Browser WebRTC streaming and local OpenCV capture.
"""

import os
import sys
import time
import base64
import urllib.request
from pathlib import Path
from datetime import datetime

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import av
from streamlit_webrtc import (
    VideoProcessorBase,
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
)

# --- AUTOMATIC STREAMLIT LAUNCHER ---
if __name__ == "__main__" and not st.runtime.exists():
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    sys.exit(stcli.main())

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Driver Drowsiness AI Guard",
    page_icon="🚗",
    layout="wide",
)

# Custom clean styling
st.markdown(
    """
    <style>
    .header-box {
        padding: 18px 24px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        margin-bottom: 20px;
        border-left: 6px solid #38bdf8;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .header-title {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .header-sub {
        margin: 6px 0 0 0;
        color: #94a3b8;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 1. DIRECTORIES & FILE PATHS
# ==========================================
PROJECT_DIR = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
LOG_DIR = PROJECT_DIR / "logs"

for folder in (DATA_DIR, MODEL_DIR, LOG_DIR):
    folder.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "drowsiness_model.joblib"
TASK_MODEL_PATH = MODEL_DIR / "face_landmarker.task"
EVENT_LOG_PATH = LOG_DIR / "drowsiness_events.csv"
AUDIO_PATH = PROJECT_DIR / "funny_munni.mp3"

# MediaPipe 468-point face mesh indices
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_CORNERS = (78, 308)
MOUTH_TOP_BOTTOM = (13, 14)

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ==========================================
# 2. MODEL & DETECTOR LOADING
# ==========================================
@st.cache_resource
def load_detector():
    """Load MediaPipe FaceLandmarker for extracting facial coordinates."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        if not TASK_MODEL_PATH.exists():
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, str(TASK_MODEL_PATH))

        base_options = python.BaseOptions(model_asset_path=str(TASK_MODEL_PATH))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        return {"type": "tasks", "detector": detector, "mp": mp}
    except Exception:
        try:
            import mediapipe as mp

            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                return {"type": "solutions", "detector": mesh, "mp": mp}
        except Exception:
            pass
    return None


@st.cache_resource
def load_ml_model():
    """Load the trained Random Forest classifier."""
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            return bundle.get("model"), bundle.get(
                "feature_columns",
                ["left_ear", "right_ear", "avg_ear", "mar", "eye_closed_seconds"],
            )
        except Exception:
            pass
    return None, None


DETECTOR_BUNDLE = load_detector()
ML_MODEL, FEATURE_COLUMNS = load_ml_model()


# ==========================================
# 3. BIOMETRIC CALCULATIONS (EAR & MAR)
# ==========================================
def euclidean_dist(pt1, pt2):
    """Calculates 2D Euclidean distance between two points."""
    return float(np.hypot(pt1.x - pt2.x, pt1.y - pt2.y))


def eye_aspect_ratio(landmarks, eye_indices):
    """
    Computes Eye Aspect Ratio (EAR).
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    horizontal = euclidean_dist(p1, p4)
    if horizontal < 1e-8:
        return 0.0
    vertical_1 = euclidean_dist(p2, p6)
    vertical_2 = euclidean_dist(p3, p5)
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_metrics(landmarks):
    """Calculates Left EAR, Right EAR, Average EAR, and Mouth Aspect Ratio (MAR)."""
    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_INDICES)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_INDICES)
    avg_ear = (left_ear + right_ear) / 2.0

    mouth_top = landmarks[MOUTH_TOP_BOTTOM[0]]
    mouth_bottom = landmarks[MOUTH_TOP_BOTTOM[1]]
    mouth_left = landmarks[MOUTH_CORNERS[0]]
    mouth_right = landmarks[MOUTH_CORNERS[1]]
    mouth_width = euclidean_dist(mouth_left, mouth_right)
    mar = euclidean_dist(mouth_top, mouth_bottom) / max(mouth_width, 1e-8)

    return {
        "left_ear": float(left_ear),
        "right_ear": float(right_ear),
        "avg_ear": float(avg_ear),
        "mar": float(mar),
    }


def process_frame(frame_bgr):
    """Extracts facial landmarks and calculates metrics from a BGR video frame."""
    if DETECTOR_BUNDLE is None:
        return None, None

    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if DETECTOR_BUNDLE["type"] == "tasks":
        mp_mod = DETECTOR_BUNDLE["mp"]
        mp_image = mp_mod.Image(image_format=mp_mod.ImageFormat.SRGB, data=rgb_frame)
        res = DETECTOR_BUNDLE["detector"].detect(mp_image)
        if res and res.face_landmarks and len(res.face_landmarks) > 0:
            landmarks = res.face_landmarks[0]
            return landmarks, calculate_metrics(landmarks)
    else:
        res = DETECTOR_BUNDLE["detector"].process(rgb_frame)
        if res.multi_face_landmarks and len(res.multi_face_landmarks) > 0:
            landmarks = res.multi_face_landmarks[0].landmark
            return landmarks, calculate_metrics(landmarks)

    return None, None


def draw_annotations(frame_bgr, landmarks, metrics, is_drowsy, status_text, draw_mesh=True):
    """Draws facial landmarks and HUD diagnostics directly on the OpenCV frame."""
    annotated = frame_bgr.copy()
    h, w, _ = annotated.shape

    if landmarks and draw_mesh:
        eye_color = (0, 0, 240) if is_drowsy else (0, 220, 100)
        mouth_color = (0, 165, 255) if metrics["mar"] >= 0.65 else (240, 200, 100)

        left_pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in LEFT_EYE_INDICES], np.int32)
        cv2.polylines(annotated, [left_pts], isClosed=True, color=eye_color, thickness=2)

        right_pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in RIGHT_EYE_INDICES], np.int32)
        cv2.polylines(annotated, [right_pts], isClosed=True, color=eye_color, thickness=2)

        m_pts = np.array([
            [int(landmarks[MOUTH_CORNERS[0]].x * w), int(landmarks[MOUTH_CORNERS[0]].y * h)],
            [int(landmarks[MOUTH_TOP_BOTTOM[0]].x * w), int(landmarks[MOUTH_TOP_BOTTOM[0]].y * h)],
            [int(landmarks[MOUTH_CORNERS[1]].x * w), int(landmarks[MOUTH_CORNERS[1]].y * h)],
            [int(landmarks[MOUTH_TOP_BOTTOM[1]].x * w), int(landmarks[MOUTH_TOP_BOTTOM[1]].y * h)],
        ], np.int32)
        cv2.polylines(annotated, [m_pts], isClosed=True, color=mouth_color, thickness=2)

    # Top Status Bar
    banner_color = (40, 40, 220) if is_drowsy else (30, 160, 60)
    cv2.rectangle(annotated, (0, 0), (w, 46), banner_color, -1)
    cv2.putText(annotated, status_text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Bottom Telemetry Bar
    if metrics:
        hud_info = f"EAR: {metrics['avg_ear']:.3f} | MAR: {metrics['mar']:.3f} | Closed: {metrics.get('eye_closed_seconds', 0.0):.1f}s"
        cv2.rectangle(annotated, (0, h - 32), (w, h), (20, 25, 30), -1)
        cv2.putText(annotated, hud_info, (14, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 230, 250), 1, cv2.LINE_AA)

    return annotated


# ==========================================
# 4. DROWSINESS EVALUATION & ALARM
# ==========================================
def evaluate_drowsiness(metrics, eye_closed_sec, ear_thresh, mar_thresh, alarm_delay):
    """Hybrid decision rule: Prolonged Eye Closure or ML prediction with high yawns."""
    ml_pred = "Alert"
    if ML_MODEL is not None and FEATURE_COLUMNS:
        row = pd.DataFrame(
            [[
                metrics["left_ear"],
                metrics["right_ear"],
                metrics["avg_ear"],
                metrics["mar"],
                eye_closed_sec,
            ]],
            columns=FEATURE_COLUMNS,
        )
        ml_pred = str(ML_MODEL.predict(row)[0])

    is_prolonged_closed = eye_closed_sec >= alarm_delay
    is_ml_drowsy_with_signs = (ml_pred.lower() == "drowsy") and (
        eye_closed_sec >= 0.8 or metrics["mar"] >= mar_thresh
    )
    is_drowsy = bool(is_prolonged_closed or is_ml_drowsy_with_signs)

    return is_drowsy, ml_pred


@st.cache_data
def get_audio_b64():
    """Encodes funny_munni.mp3 audio alert to base64."""
    if AUDIO_PATH.exists():
        with open(AUDIO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None


AUDIO_B64 = get_audio_b64()


def trigger_alarm():
    """Plays the audio alert sound in the browser."""
    if AUDIO_B64:
        st.markdown(
            f'<audio autoplay style="display:none;"><source src="data:audio/mp3;base64,{AUDIO_B64}" type="audio/mp3"></audio>',
            unsafe_allow_html=True,
        )


def log_incident(ml_prediction):
    """Logs timestamped drowsiness event to CSV."""
    now_str = datetime.now().isoformat(timespec="seconds")
    new_entry = pd.DataFrame([{"timestamp": now_str, "event": "Drowsy warning", "ml_prediction": ml_prediction}])
    if EVENT_LOG_PATH.exists() and os.path.getsize(EVENT_LOG_PATH) > 0:
        new_entry.to_csv(EVENT_LOG_PATH, mode="a", header=False, index=False)
    else:
        new_entry.to_csv(EVENT_LOG_PATH, mode="w", header=True, index=False)


# ==========================================
# 5. WEBRTC STREAMING PROCESSOR
# ==========================================
class DrowsinessVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.eye_closed_start = None
        self.last_drowsy_state = False
        self.metrics = None
        self.is_drowsy = False
        self.ml_pred = "N/A"
        self.closed_sec = 0.0

        # Dynamic settings updated from sidebar
        self.ear_thresh = 0.21
        self.alarm_delay = 1.5
        self.mar_thresh = 0.65
        self.show_mesh = True

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        now_mono = time.monotonic()

        landmarks, metrics = process_frame(img)
        is_drowsy = False
        status_label = "SEARCHING FOR FACE"
        ml_pred = "N/A"
        closed_sec = 0.0

        if landmarks and metrics:
            if metrics["avg_ear"] < self.ear_thresh:
                if self.eye_closed_start is None:
                    self.eye_closed_start = now_mono
                closed_sec = now_mono - self.eye_closed_start
            else:
                self.eye_closed_start = None
                closed_sec = 0.0

            metrics["eye_closed_seconds"] = float(closed_sec)
            is_drowsy, ml_pred = evaluate_drowsiness(
                metrics, closed_sec, self.ear_thresh, self.mar_thresh, self.alarm_delay
            )

            if is_drowsy:
                status_label = f"DROWSY WARNING! (ML: {ml_pred})"
            else:
                status_label = f"SAFE & ALERT (ML: {ml_pred})"

            if is_drowsy and not self.last_drowsy_state:
                log_incident(ml_pred)

            self.last_drowsy_state = is_drowsy
        else:
            self.eye_closed_start = None
            self.last_drowsy_state = False

        self.metrics = metrics
        self.is_drowsy = is_drowsy
        self.ml_pred = ml_pred
        self.closed_sec = closed_sec

        annotated = draw_annotations(
            img, landmarks, metrics, is_drowsy, status_label, draw_mesh=self.show_mesh
        )
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ==========================================
# 6. USER INTERFACE (SIDEBAR & MAIN VIEW)
# ==========================================

# Main Header
st.markdown(
    """
    <div class="header-box">
        <h1 class="header-title">🚗 Driver Drowsiness Detection System</h1>
        <p class="header-sub">
            Real-time computer vision safety monitor using MediaPipe Face Mesh & Random Forest AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Detection Sensitivity")
    ear_thresh = st.slider(
        "👁️ Eye Openness Threshold (EAR)",
        min_value=0.15,
        max_value=0.35,
        value=0.21,
        step=0.01,
        help="EAR below this value indicates closed eyes.",
    )
    alarm_delay = st.slider(
        "⏱️ Alarm Trigger Delay (Seconds)",
        min_value=0.5,
        max_value=3.5,
        value=1.5,
        step=0.1,
        help="Seconds of continuous eye closure before the alarm sounds.",
    )
    mar_thresh = st.slider(
        "🥱 Yawn Threshold (MAR)",
        min_value=0.40,
        max_value=0.90,
        value=0.65,
        step=0.05,
        help="MAR above this value indicates a wide yawn.",
    )

    st.divider()
    st.header("🔊 Audio Alarm")
    enable_sound = st.toggle("Enable Audio Alarm", value=True)
    if st.button("🔔 Test Alarm Sound", width="stretch"):
        trigger_alarm()
        st.toast("Sound test triggered in browser!", icon="🎵")

    st.divider()
    st.header("📹 Visual Overlays")
    show_mesh = st.checkbox("Show Eye & Mouth Outlines", value=True)


# Main Tabs
tab_stream, tab_snap, tab_history = st.tabs(
    [
        "🎥 Live Camera Monitoring",
        "📸 Photo Snapshot Test",
        "📋 Incident History & Logs",
    ]
)

# ----------------------------------------------------
# TAB 1: LIVE CAMERA MONITORING
# ----------------------------------------------------
with tab_stream:
    cam_mode = st.radio(
        "Select Camera Input Mode:",
        [
            "🌐 Browser Webcam (WebRTC - Recommended)",
            "🖥️ Local OpenCV Device",
        ],
        horizontal=True,
        help="Browser WebRTC streams directly without hardware lock issues on Linux.",
    )

    if "Browser Webcam" in cam_mode:
        st.caption(
            "Streams directly from your web browser. Click **START** below to activate."
        )

        col_stream, col_telemetry = st.columns([1.6, 1.0])

        with col_stream:
            webrtc_ctx = webrtc_streamer(
                key="drowsiness-live",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIG,
                video_processor_factory=DrowsinessVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            # Keep processor thresholds in sync with sidebar controls
            if webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.ear_thresh = ear_thresh
                webrtc_ctx.video_processor.alarm_delay = alarm_delay
                webrtc_ctx.video_processor.mar_thresh = mar_thresh
                webrtc_ctx.video_processor.show_mesh = show_mesh

        with col_telemetry:
            if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
                vp = webrtc_ctx.video_processor
                if vp.is_drowsy:
                    st.error("🚨 **DROWSY DETECTED! WAKE UP!**")
                    if enable_sound:
                        trigger_alarm()
                elif vp.metrics:
                    st.success("✅ **DRIVER ALERT & ATTENTIVE**")
                else:
                    st.warning("👤 **Position face in front of camera**")

                ear_val = vp.metrics["avg_ear"] if vp.metrics else 0.0
                mar_val = vp.metrics["mar"] if vp.metrics else 0.0
                st.metric("Eye Aspect Ratio (EAR)", f"{ear_val:.3f}", f"Threshold: {ear_thresh}")
                st.metric("Mouth Aspect Ratio (MAR)", f"{mar_val:.3f}", f"Threshold: {mar_thresh}")
                st.metric("Eyes Closed Duration", f"{vp.closed_sec:.1f}s", f"Alarm at: {alarm_delay}s")
                st.metric("ML Classifier Prediction", vp.ml_pred, "Random Forest")
            else:
                st.info("ℹ️ Click **START** on the video feed to begin camera stream.")
                st.metric("Eye Aspect Ratio (EAR)", "0.000", f"Threshold: {ear_thresh}")
                st.metric("Mouth Aspect Ratio (MAR)", "0.000", f"Threshold: {mar_thresh}")
                st.metric("Eyes Closed Duration", "0.0s", f"Alarm at: {alarm_delay}s")
                st.metric("ML Classifier State", "Standby", "Random Forest")

    else:
        # Local OpenCV Device Mode
        col_cv_ctrl, col_cv_dev = st.columns([2, 1])
        with col_cv_dev:
            cam_index = st.selectbox("Webcam Device ID", options=[0, 1, 2], index=0)

        col_video, col_info = st.columns([1.6, 1.0])

        with col_video:
            col_b1, col_b2 = st.columns(2)
            if "streaming_cv" not in st.session_state:
                st.session_state.streaming_cv = False

            with col_b1:
                if st.button("▶ Start Camera", type="primary", width="stretch", disabled=st.session_state.streaming_cv):
                    st.session_state.streaming_cv = True
                    st.rerun()

            with col_b2:
                if st.button("⏹ Stop Camera", width="stretch", disabled=not st.session_state.streaming_cv):
                    st.session_state.streaming_cv = False
                    st.rerun()

            cv_placeholder = st.empty()

        with col_info:
            status_box = st.empty()
            m_ear = st.empty()
            m_mar = st.empty()
            m_time = st.empty()
            m_ml = st.empty()
            alarm_box = st.empty()

        if not st.session_state.streaming_cv:
            status_box.info("ℹ️ **OpenCV Camera Idle** — Click **Start Camera** to begin.")
            m_ear.metric("Eye Aspect Ratio (EAR)", "0.000", f"Threshold: {ear_thresh}")
            m_mar.metric("Mouth Aspect Ratio (MAR)", "0.000", f"Threshold: {mar_thresh}")
            m_time.metric("Eyes Closed Duration", "0.0s", f"Alarm at: {alarm_delay}s")
            m_ml.metric("ML Classifier State", "Standby", "Random Forest")
        else:
            cap = cv2.VideoCapture(cam_index)
            if not cap.isOpened():
                cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)

            if not cap.isOpened():
                st.session_state.streaming_cv = False
                st.error(
                    f"⚠️ **Unable to open camera Device ID {cam_index}.**\n\n"
                    "**Why this happens:**\n"
                    "- Your web browser or another application is currently using the camera.\n"
                    "- On Linux, `/dev/video0` is locked exclusively by the active browser tab.\n\n"
                    "👉 **Solution:** Switch to **'🌐 Browser Webcam (WebRTC)'** above for instant streaming without conflicts!"
                )
            else:
                eye_closed_start = None
                last_drowsy_state = False
                last_alarm_time = 0.0

                while cap.isOpened() and st.session_state.streaming_cv:
                    success, frame = cap.read()
                    if not success:
                        st.warning("Webcam frame could not be read.")
                        break

                    frame = cv2.flip(frame, 1)
                    now_mono = time.monotonic()

                    landmarks, metrics = process_frame(frame)
                    is_drowsy = False
                    status_label = "SEARCHING FOR FACE"
                    ml_pred = "N/A"
                    closed_sec = 0.0

                    if landmarks and metrics:
                        if metrics["avg_ear"] < ear_thresh:
                            if eye_closed_start is None:
                                eye_closed_start = now_mono
                            closed_sec = now_mono - eye_closed_start
                        else:
                            eye_closed_start = None
                            closed_sec = 0.0

                        metrics["eye_closed_seconds"] = float(closed_sec)
                        is_drowsy, ml_pred = evaluate_drowsiness(
                            metrics, closed_sec, ear_thresh, mar_thresh, alarm_delay
                        )

                        if is_drowsy:
                            status_label = f"🚨 DROWSY WARNING! (ML: {ml_pred})"
                        else:
                            status_label = f"✅ SAFE & ALERT (ML: {ml_pred})"

                        if is_drowsy and not last_drowsy_state:
                            log_incident(ml_pred)

                        if is_drowsy and enable_sound:
                            if now_mono - last_alarm_time > 1.2:
                                with alarm_box:
                                    trigger_alarm()
                                last_alarm_time = now_mono

                        last_drowsy_state = is_drowsy
                    else:
                        eye_closed_start = None
                        last_drowsy_state = False

                    annotated = draw_annotations(
                        frame, landmarks, metrics, is_drowsy, status_label, draw_mesh=show_mesh
                    )
                    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    cv_placeholder.image(rgb_frame, channels="RGB", width="stretch")

                    if is_drowsy:
                        status_box.error("🚨 **DROWSY DETECTED! WAKE UP!**")
                    elif landmarks:
                        status_box.success("✅ **DRIVER ALERT & ATTENTIVE**")
                    else:
                        status_box.warning("👤 **Position face in front of camera**")

                    ear_val = metrics["avg_ear"] if metrics else 0.0
                    mar_val = metrics["mar"] if metrics else 0.0
                    m_ear.metric("Eye Aspect Ratio (EAR)", f"{ear_val:.3f}", f"Threshold: {ear_thresh}")
                    m_mar.metric("Mouth Aspect Ratio (MAR)", f"{mar_val:.3f}", f"Threshold: {mar_thresh}")
                    m_time.metric("Eyes Closed Duration", f"{closed_sec:.1f}s", f"Alarm at: {alarm_delay}s")
                    m_ml.metric("ML Classifier Prediction", ml_pred, "Random Forest")

                cap.release()


# ----------------------------------------------------
# TAB 2: PHOTO SNAPSHOT TEST
# ----------------------------------------------------
with tab_snap:
    st.markdown("#### Test Facial Landmark & Drowsiness Detection on an Image")
    st.caption("Capture a photo from your webcam or upload an image to inspect the biometric calculations.")

    source_method = st.radio(
        "Snapshot Source:",
        ["📁 Upload Image File", "📸 Browser Camera Snapshot"],
        horizontal=True,
    )

    photo = None
    if source_method == "📁 Upload Image File":
        photo = st.file_uploader(
            "Upload an image file", type=["jpg", "jpeg", "png", "webp"]
        )
    else:
        photo = st.camera_input("Take a photo using browser camera")

    if photo is not None:
        file_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        landmarks, metrics = process_frame(img_bgr)
        if landmarks and metrics:
            mock_closed_sec = alarm_delay if metrics["avg_ear"] < ear_thresh else 0.0
            metrics["eye_closed_seconds"] = mock_closed_sec

            is_drowsy, ml_pred = evaluate_drowsiness(
                metrics, mock_closed_sec, ear_thresh, mar_thresh, alarm_delay
            )
            status_text = "DROWSY WARNING" if is_drowsy else "SAFE & ALERT"

            annotated = draw_annotations(
                img_bgr, landmarks, metrics, is_drowsy, status_text, draw_mesh=True
            )
            col_img, col_res = st.columns([1.4, 1.0])

            with col_img:
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption="Biometric Landmark Analysis",
                    width="stretch",
                )

            with col_res:
                st.markdown("### 📊 Diagnostic Results")
                if is_drowsy:
                    st.error(f"🚨 **Drowsiness Detected** (ML Prediction: `{ml_pred}`)")
                    if enable_sound:
                        trigger_alarm()
                else:
                    st.success(f"✅ **Driver Alert & Attentive** (ML Prediction: `{ml_pred}`)")

                st.write({
                    "Average EAR": round(metrics["avg_ear"], 4),
                    "EAR Threshold": ear_thresh,
                    "Mouth Ratio (MAR)": round(metrics["mar"], 4),
                    "MAR Threshold": mar_thresh,
                    "ML Classification": ml_pred,
                })
        else:
            st.warning("⚠️ No face detected. Please ensure good lighting and that your face is clearly visible.")


# ----------------------------------------------------
# TAB 3: INCIDENT HISTORY & LOGS
# ----------------------------------------------------
with tab_history:
    st.markdown("#### 📋 Recorded Drowsiness Incidents")
    if EVENT_LOG_PATH.exists() and os.path.getsize(EVENT_LOG_PATH) > 0:
        events = pd.read_csv(EVENT_LOG_PATH)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Warnings Recorded", len(events))
        with c2:
            last_ts = events["timestamp"].iloc[-1] if not events.empty else "None"
            st.metric("Most Recent Event", str(last_ts).replace("T", " "))

        st.dataframe(events.tail(50).sort_index(ascending=False), width="stretch")

        col_dl, col_clr = st.columns([1, 1])
        with col_dl:
            csv_data = events.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Log CSV",
                data=csv_data,
                file_name="drowsiness_events.csv",
                mime="text/csv",
            )
        with col_clr:
            if st.button("🗑️ Clear Log History"):
                pd.DataFrame(columns=["timestamp", "event", "ml_prediction"]).to_csv(
                    EVENT_LOG_PATH, index=False
                )
                st.rerun()
    else:
        st.info("No drowsiness incidents recorded yet. Start camera monitoring to detect events.")
