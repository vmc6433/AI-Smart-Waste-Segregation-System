"""
======================================================================
AI POWERED WASTE SEGREGATION SYSTEM
Professional PyQt6 Camera GUI
======================================================================

LOCAL MODEL:
    EfficientNet-B0

CLASSES:
    E_WASTE
    ORGANIC
    PLASTIC
    RECYCLABLE

GEMINI:
    gemini-3.5-flash-lite

BIN SELECTION PRIORITY:
    Gemini is PRIMARY (more accurate).
    Local EfficientNet is the FALLBACK used only when
    Gemini fails, is quota-blocked, or times out (> 30 s).

CAPTURE LOGIC:
    Capture image
        |
        +--> EfficientNet prediction immediately (displayed only)
        |
        +--> Gemini prediction in background
                    |
                    +--> Gemini < 10 sec
                    |       wait until 10 sec
                    |       dispatch GEMINI class -> motor
                    |       capture next
                    |
                    +--> Gemini 10-30 sec
                    |       dispatch GEMINI class -> motor
                    |       capture immediately
                    |
                    +--> Gemini >= 30 sec
                            TIMEOUT
                            dispatch LOCAL class -> motor (fallback)
                            capture next

CONTROLS:
    SPACE = Capture
    Q     = Quit
    ESC   = Quit
======================================================================
"""

# ======================================================================
# IMPORTS
# ======================================================================

import os
import re
import sys
import time
import socket
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from dotenv import load_dotenv

from torchvision import models, transforms

from google import genai
from google.genai import types

from PyQt6.QtCore import (
    QObject,
    Qt,
    QTimer,
    pyqtSignal,
)

from PyQt6.QtGui import (
    QFont,
    QImage,
    QPixmap,
)

from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# ======================================================================
# PROJECT CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "efficientnet_b0_best.pth"
)

ENV_PATH = PROJECT_ROOT / ".env"


# ======================================================================
# MODEL CONFIGURATION
# ======================================================================

CLASSES = [
    "E_WASTE",
    "ORGANIC",
    "PLASTIC",
    "RECYCLABLE",
]


# ======================================================================
# GEMINI CONFIGURATION  (UNCHANGED)
# ======================================================================

GEMINI_MODEL = "gemini-3.5-flash-lite"

GEMINI_MAX_OUTPUT_TOKENS = 30

GEMINI_PROMPT = """
Classify this waste image.

Choose exactly ONE category:

E_WASTE
ORGANIC
PLASTIC
RECYCLABLE

Return ONLY:

Category: <category>
Confidence: <0-100>%
"""


# ======================================================================
# TIMING CONFIGURATION  (UNCHANGED)
# ======================================================================

MIN_CAPTURE_INTERVAL = 10.0

GEMINI_MAX_WAIT = 30.0


# ======================================================================
# CAMERA CONFIGURATION
# ======================================================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 1280

CAMERA_HEIGHT = 720


# ======================================================================
# IMAGE CONFIGURATION
# ======================================================================

GEMINI_IMAGE_WIDTH = 640

GEMINI_JPEG_QUALITY = 75


# ======================================================================
# PREDICTION BRIDGE  (camera_predict  ->  motor controller)
# ======================================================================

MOTOR_CONTROLLER_HOST    = "127.0.0.1"
MOTOR_CONTROLLER_PORT    = 9999
MOTOR_CONTROLLER_TIMEOUT = 2.0

# Class -> Bin mapping.  Must match CLASS_TO_BIN in motion_controller_gui.py.
CLASS_TO_BIN = {
    "E_WASTE":    1,
    "ORGANIC":    2,
    "RECYCLABLE": 3,
    "PLASTIC":    4,
}


# ======================================================================
# LOAD ENVIRONMENT
# ======================================================================

load_dotenv(ENV_PATH)


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def format_seconds(seconds):
    seconds = max(0.0, float(seconds))
    return f"{seconds:.1f}s"


# ----------------------------------------------------------------------

def frame_to_qimage(frame):
    if frame is None:
        return QImage()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    height, width, channels = rgb.shape

    bytes_per_line = channels * width

    return QImage(
        rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    ).copy()


# ----------------------------------------------------------------------

def resize_frame_for_label(frame, width, height):
    qimage = frame_to_qimage(frame)

    if qimage.isNull():
        return QPixmap()

    pixmap = QPixmap.fromImage(qimage)

    return pixmap.scaled(
        width,
        height,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


# ----------------------------------------------------------------------

def make_gemini_image_bytes(frame):
    if frame is None:
        raise ValueError("Captured frame is empty.")

    height, width = frame.shape[:2]

    if width <= 0 or height <= 0:
        raise ValueError("Captured frame has invalid dimensions.")

    target_width = GEMINI_IMAGE_WIDTH

    target_height = int(height * target_width / width)

    target_height = max(1, target_height)

    resized = cv2.resize(
        frame,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )

    success, encoded = cv2.imencode(
        ".jpg",
        resized,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            GEMINI_JPEG_QUALITY,
        ],
    )

    if not success:
        raise RuntimeError("Failed to encode image for Gemini.")

    return encoded.tobytes()


# ----------------------------------------------------------------------

def parse_gemini_reply(text):
    if text is None:
        text = ""

    text = str(text).strip()

    upper_text = text.upper()

    category = "UNKNOWN"

    category_match = re.search(
        r"\b(E_WASTE|ORGANIC|PLASTIC|RECYCLABLE)\b",
        upper_text,
    )

    if category_match:
        category = category_match.group(1)

    confidence = None

    confidence_match = re.search(
        r"Confidence\s*:\s*"
        r"(\d+(?:\.\d+)?)"
        r"\s*%",
        text,
        flags=re.IGNORECASE,
    )

    if confidence_match:
        try:
            confidence = float(confidence_match.group(1))

            confidence = max(0.0, min(100.0, confidence))

        except ValueError:
            confidence = None

    return category, confidence


# ----------------------------------------------------------------------

def is_quota_error(error):
    text = str(error).lower()

    return (
        "resource_exhausted" in text
        or "quota" in text
        or "429" in text
        or "rate limit" in text
    )


# ----------------------------------------------------------------------

def extract_retry_seconds(error):
    text = str(error)

    patterns = [
        r"retry in\s+([0-9]+(?:\.[0-9]+)?)s",
        r"retryDelay['\"]?\s*[:=]\s*['\"]?([0-9]+)s",
        r"retryDelay.*?([0-9]+(?:\.[0-9]+)?)s",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)

        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

    return None


# ----------------------------------------------------------------------

def send_prediction_to_controller(
    category,
    host=MOTOR_CONTROLLER_HOST,
    port=MOTOR_CONTROLLER_PORT,
    timeout=MOTOR_CONTROLLER_TIMEOUT,
):
    """
    Fire-and-forget: tell the motor controller which bin to go to.
    Returns True on ACK, False on any error or unmapped class.
    """

    if not category:
        return False

    if category in (
        "—",
        "ERROR",
        "UNKNOWN",
        "TIMEOUT",
        "QUOTA LIMIT",
        "ANALYZING",
    ):
        return False

    if category not in CLASS_TO_BIN:
        return False

    try:
        with socket.create_connection(
            (host, port),
            timeout=timeout,
        ) as sock:

            sock.settimeout(timeout)

            sock.sendall(f"CLASS:{category}\n".encode("utf-8"))

            try:
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass

            try:
                sock.recv(64)
            except (socket.timeout, OSError):
                pass

        return True

    except Exception:
        return False


# ======================================================================
# LOCAL EFFICIENTNET MODEL  (UNCHANGED)
# ======================================================================

class LocalWasteModel:

    def __init__(self, model_path):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                "EfficientNet model was not found:\n\n"
                f"{self.model_path}"
            )

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )

        state_dict = checkpoint

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

        if not isinstance(state_dict, dict):
            raise RuntimeError(
                "The saved model checkpoint "
                "does not contain a valid state dictionary."
            )

        cleaned_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            cleaned_state_dict[key] = value

        self.model = models.efficientnet_b0(weights=None)

        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features,
            len(CLASSES),
        )

        try:
            self.model.load_state_dict(cleaned_state_dict, strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                "EfficientNet model weights "
                "could not be loaded correctly.\n\n"
                f"{error}"
            )

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def predict(self, frame):
        if frame is None:
            raise ValueError("Cannot classify an empty frame.")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.inference_mode():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, index = torch.max(probabilities, dim=1)

        class_index = int(index.item())
        confidence_value = float(confidence.item()) * 100.0

        if class_index < 0 or class_index >= len(CLASSES):
            raise RuntimeError(
                "Model returned an invalid class index: "
                f"{class_index}"
            )

        category = CLASSES[class_index]

        return category, confidence_value


# ======================================================================
# GEMINI ENGINE  (UNCHANGED)
# ======================================================================

class GeminiEngine:

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY", "").strip()

        if not api_key or api_key == "YOUR_REAL_KEY":
            raise RuntimeError(
                "GEMINI_API_KEY is missing.\n\n"
                "Open the project's .env file and set:\n\n"
                "GEMINI_API_KEY=your_actual_key"
            )

        self.client = genai.Client(api_key=api_key)

    def classify(self, image_bytes):
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
                GEMINI_PROMPT,
            ],
            config={
                "thinking_config": {
                    "thinking_level": "minimal"
                },
                "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
            },
        )

        raw_reply = (response.text or "").strip()

        category, confidence = parse_gemini_reply(raw_reply)

        return {
            "category": category,
            "confidence": confidence,
            "raw": raw_reply,
        }


# ======================================================================
# GEMINI WORKER SIGNALS  (UNCHANGED)
# ======================================================================

class GeminiSignals(QObject):

    finished = pyqtSignal(int, object)

    failed = pyqtSignal(int, str, bool, object)


# ======================================================================
# GEMINI WORKER  (UNCHANGED)
# ======================================================================

class GeminiWorker:

    def __init__(self):
        self.signals = GeminiSignals()
        self._engine = None
        self._engine_lock = threading.Lock()

    def get_engine(self):
        if self._engine is None:
            with self._engine_lock:
                if self._engine is None:
                    self._engine = GeminiEngine()
        return self._engine

    def start(self, request_id, image_bytes):
        thread = threading.Thread(
            target=self._run,
            args=(request_id, image_bytes),
            daemon=True,
        )
        thread.start()

    def _run(self, request_id, image_bytes):
        try:
            engine = self.get_engine()
            result = engine.classify(image_bytes)
            self.signals.finished.emit(request_id, result)

        except Exception as error:
            quota = is_quota_error(error)
            retry_seconds = extract_retry_seconds(error)
            self.signals.failed.emit(
                request_id,
                str(error),
                quota,
                retry_seconds,
            )


# ======================================================================
# MAIN WINDOW
# ======================================================================

class WasteSegregationWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Powered Waste Segregation System")
        self.resize(1450, 850)
        self.setMinimumSize(1180, 700)

        self.camera = None
        self.last_frame = None

        self.local_model = None
        self.gemini_worker = None

        self.capture_number = 0
        self.captured_frame = None

        self.capture_time = None
        self.gemini_finished = False
        self.gemini_running = False
        self.gemini_timeout_handled = False
        self.current_request_id = None
        self.request_counter = 0

        self.local_category = "—"
        self.local_confidence = None
        self.gemini_category = "—"
        self.gemini_confidence = None
        self.gemini_raw = "Waiting for capture…"

        self.quota_block_until = 0.0

        # Guard so a single capture can only dispatch to the motor once.
        self.dispatched_this_capture = False

        self.history = deque(maxlen=6)

        self.build_ui()
        self.apply_styles()

        self.initialize_system()

        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_camera)
        self.camera_timer.start(30)

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_timing_display)
        self.status_timer.start(100)

    # ==================================================================
    # BUILD UI  (UNCHANGED, + motor dispatch label)
    # ==================================================================

    def build_ui(self):

        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 14)
        root.setSpacing(14)

        # ---------------- HEADER ----------------
        header = QFrame()
        header.setObjectName("HeaderCard")

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(22, 16, 22, 16)

        title_layout = QVBoxLayout()

        title = QLabel("AI POWERED WASTE SEGREGATION SYSTEM")
        title.setObjectName("Title")

        subtitle = QLabel(
            "Dual AI waste classification • "
            "EfficientNet-B0 + Google Gemini"
        )
        subtitle.setObjectName("Subtitle")

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        self.system_badge = QLabel("● STARTING")
        self.system_badge.setObjectName("SystemBadge")
        header_layout.addWidget(self.system_badge)

        root.addWidget(header)

        # ---------------- MAIN GRID ----------------
        main_grid = QGridLayout()
        main_grid.setSpacing(14)

        # ==============================================================
        # LEFT: LIVE CAMERA
        # ==============================================================
        camera_card = QFrame()
        camera_card.setObjectName("Card")

        camera_layout = QVBoxLayout(camera_card)
        camera_layout.setContentsMargins(14, 14, 14, 14)

        camera_title = QLabel("LIVE CAMERA")
        camera_title.setObjectName("SectionTitle")
        camera_layout.addWidget(camera_title)

        self.live_label = QLabel("Opening camera…")
        self.live_label.setObjectName("Video")
        self.live_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.live_label.setMinimumSize(620, 450)
        self.live_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        camera_layout.addWidget(self.live_label, 1)

        camera_footer = QHBoxLayout()

        self.camera_state = QLabel("● CAMERA")
        self.camera_state.setObjectName("SmallBadge")
        camera_footer.addWidget(self.camera_state)

        camera_footer.addStretch()

        self.capture_counter = QLabel("IMAGE 00")
        self.capture_counter.setObjectName("Counter")
        camera_footer.addWidget(self.capture_counter)

        camera_layout.addLayout(camera_footer)

        main_grid.addWidget(camera_card, 0, 0, 2, 1)

        # ==============================================================
        # RIGHT: CURRENT ANALYSIS
        # ==============================================================
        analysis_card = QFrame()
        analysis_card.setObjectName("Card")

        analysis_layout = QVBoxLayout(analysis_card)
        analysis_layout.setContentsMargins(14, 14, 14, 14)
        analysis_layout.setSpacing(10)

        analysis_title = QLabel("CURRENT ANALYSIS")
        analysis_title.setObjectName("SectionTitle")
        analysis_layout.addWidget(analysis_title)

        self.captured_label = QLabel("No captured image")
        self.captured_label.setObjectName("CapturedVideo")
        self.captured_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.captured_label.setMinimumHeight(175)
        analysis_layout.addWidget(self.captured_label)

        # -------- Motor dispatch status line --------
        self.dispatch_label = QLabel("Motor: waiting for prediction…")
        self.dispatch_label.setObjectName("DispatchText")
        self.dispatch_label.setWordWrap(True)
        analysis_layout.addWidget(self.dispatch_label)

        # -------- LOCAL RESULT --------
        local_card = QFrame()
        local_card.setObjectName("ResultCard")

        local_layout = QVBoxLayout(local_card)
        local_layout.setContentsMargins(14, 9, 14, 9)

        local_header = QLabel("LOCAL AI  •  EfficientNet-B0")
        local_header.setObjectName("ResultTitle")
        local_layout.addWidget(local_header)

        local_row = QHBoxLayout()

        self.local_result = QLabel("—")
        self.local_result.setObjectName("ResultValue")
        local_row.addWidget(self.local_result)

        local_row.addStretch()

        self.local_confidence_label = QLabel("—")
        self.local_confidence_label.setObjectName("Confidence")
        local_row.addWidget(self.local_confidence_label)

        local_layout.addLayout(local_row)

        analysis_layout.addWidget(local_card)

        # -------- GEMINI RESULT --------
        gemini_card = QFrame()
        gemini_card.setObjectName("ResultCard")

        gemini_layout = QVBoxLayout(gemini_card)
        gemini_layout.setContentsMargins(14, 9, 14, 9)

        gemini_header = QHBoxLayout()

        gemini_title = QLabel("GEMINI AI  •  PRIMARY")
        gemini_title.setObjectName("ResultTitle")
        gemini_header.addWidget(gemini_title)

        gemini_header.addStretch()

        self.gemini_badge = QLabel("READY")
        self.gemini_badge.setObjectName("GeminiBadge")
        gemini_header.addWidget(self.gemini_badge)

        gemini_layout.addLayout(gemini_header)

        gemini_row = QHBoxLayout()

        self.gemini_result = QLabel("—")
        self.gemini_result.setObjectName("ResultValue")
        gemini_row.addWidget(self.gemini_result)

        gemini_row.addStretch()

        self.gemini_confidence_label = QLabel("—")
        self.gemini_confidence_label.setObjectName("Confidence")
        gemini_row.addWidget(self.gemini_confidence_label)

        gemini_layout.addLayout(gemini_row)

        reply_title = QLabel("GEMINI REPLY")
        reply_title.setObjectName("ReplyTitle")
        gemini_layout.addWidget(reply_title)

        self.gemini_reply = QLabel("Waiting for capture…")
        self.gemini_reply.setObjectName("GeminiReply")
        self.gemini_reply.setWordWrap(True)
        self.gemini_reply.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.gemini_reply.setMinimumHeight(58)
        gemini_layout.addWidget(self.gemini_reply)

        analysis_layout.addWidget(gemini_card)

        # -------- TIMING CARD --------
        timing_card = QFrame()
        timing_card.setObjectName("TimingCard")

        timing_layout = QVBoxLayout(timing_card)
        timing_layout.setContentsMargins(14, 9, 14, 9)

        timing_header = QHBoxLayout()

        timing_title = QLabel("CAPTURE CONTROL")
        timing_title.setObjectName("ResultTitle")
        timing_header.addWidget(timing_title)

        timing_header.addStretch()

        self.timing_text = QLabel("Ready")
        self.timing_text.setObjectName("TimingText")
        timing_header.addWidget(self.timing_text)

        timing_layout.addLayout(timing_header)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        timing_layout.addWidget(self.progress)

        analysis_layout.addWidget(timing_card)

        # -------- HISTORY --------
        history_title = QLabel("RECENT RESULTS")
        history_title.setObjectName("ResultTitle")
        analysis_layout.addWidget(history_title)

        self.history_scroll = QScrollArea()
        self.history_scroll.setWidgetResizable(True)
        self.history_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.history_scroll.setMaximumHeight(135)

        self.history_widget = QWidget()
        self.history_layout = QVBoxLayout(self.history_widget)
        self.history_layout.setContentsMargins(0, 0, 0, 0)
        self.history_layout.setSpacing(5)

        self.history_scroll.setWidget(self.history_widget)
        analysis_layout.addWidget(self.history_scroll)

        main_grid.addWidget(analysis_card, 0, 1, 2, 1)

        main_grid.setColumnStretch(0, 3)
        main_grid.setColumnStretch(1, 2)

        root.addLayout(main_grid, 1)

        # ---------------- FOOTER ----------------
        footer = QHBoxLayout()

        self.device_label = QLabel("Device: detecting…")
        self.device_label.setObjectName("FooterText")
        footer.addWidget(self.device_label)

        footer.addStretch()

        controls = QLabel("SPACE  Capture     •     Q / ESC  Exit")
        controls.setObjectName("FooterText")
        footer.addWidget(controls)

        root.addLayout(footer)

    # ==================================================================
    # STYLES  (UNCHANGED, + DispatchText)
    # ==================================================================

    def apply_styles(self):

        self.setStyleSheet(
            """
            QMainWindow {
                background: #08111f;
            }

            QWidget {
                color: #e8eef7;
                font-family: "Segoe UI";
            }

            QFrame#HeaderCard,
            QFrame#Card {
                background: #101a2b;
                border: 1px solid #263752;
                border-radius: 14px;
            }

            QFrame#ResultCard,
            QFrame#TimingCard {
                background: #0c1626;
                border: 1px solid #263752;
                border-radius: 10px;
            }

            QLabel#Title {
                font-size: 23px;
                font-weight: 800;
                color: #f5f8fc;
            }

            QLabel#Subtitle {
                font-size: 12px;
                color: #8fa3be;
                margin-top: 2px;
            }

            QLabel#SectionTitle {
                font-size: 14px;
                font-weight: 800;
                color: #d2ddec;
                letter-spacing: 1px;
            }

            QLabel#SystemBadge,
            QLabel#GeminiBadge,
            QLabel#SmallBadge {
                padding: 6px 10px;
                border-radius: 8px;
                background: #17365f;
                color: #82c7ff;
                font-size: 10px;
                font-weight: 800;
            }

            QLabel#Counter {
                padding: 5px 9px;
                border-radius: 7px;
                background: #18243a;
                color: #bdcce0;
                font-size: 10px;
                font-weight: 800;
            }

            QLabel#Video,
            QLabel#CapturedVideo {
                background: #050a12;
                border: 1px solid #263752;
                border-radius: 10px;
                color: #63758e;
            }

            QLabel#ResultTitle,
            QLabel#ReplyTitle {
                font-size: 9px;
                font-weight: 800;
                color: #8296b2;
                letter-spacing: 1px;
            }

            QLabel#ResultValue {
                font-size: 22px;
                font-weight: 900;
                color: #f5f8fc;
            }

            QLabel#Confidence {
                font-size: 17px;
                font-weight: 800;
                color: #72dda0;
            }

            QLabel#GeminiReply {
                background: #111f33;
                border: 1px solid #263752;
                border-radius: 7px;
                padding: 8px;
                color: #cbd7e6;
                font-size: 10px;
            }

            QLabel#TimingText {
                color: #a4b5ca;
                font-size: 10px;
                font-weight: 700;
            }

            QLabel#DispatchText {
                background: #111f33;
                border: 1px solid #263752;
                border-radius: 7px;
                padding: 6px 8px;
                color: #a9c7ff;
                font-size: 11px;
                font-weight: 700;
            }

            QLabel#FooterText {
                color: #72859f;
                font-size: 10px;
            }

            QProgressBar {
                background: #17243a;
                border: none;
                border-radius: 4px;
                height: 7px;
            }

            QProgressBar::chunk {
                background: #3988ff;
                border-radius: 4px;
            }

            QScrollArea {
                background: transparent;
                border: none;
            }

            QScrollBar:vertical {
                background: #0c1626;
                width: 7px;
                border-radius: 3px;
            }

            QScrollBar::handle:vertical {
                background: #30425f;
                border-radius: 3px;
            }
            """
        )

    # ==================================================================
    # INITIALIZE SYSTEM  (UNCHANGED)
    # ==================================================================

    def initialize_system(self):

        try:
            self.local_model = LocalWasteModel(MODEL_PATH)

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                self.device_label.setText("Device: CUDA • " f"{gpu_name}")
            else:
                self.device_label.setText("Device: CPU")

            self.gemini_worker = GeminiWorker()

            self.gemini_worker.signals.finished.connect(
                self.on_gemini_finished
            )
            self.gemini_worker.signals.failed.connect(
                self.on_gemini_failed
            )

            api_key = os.getenv("GEMINI_API_KEY", "").strip()

            if not api_key or api_key == "YOUR_REAL_KEY":
                raise RuntimeError(
                    "GEMINI_API_KEY is not configured "
                    "in the .env file."
                )

            self.open_camera()

            self.set_system_status("● SYSTEM READY", "ready")

        except Exception as error:
            self.set_system_status("● STARTUP ERROR", "error")
            QMessageBox.critical(self, "Startup Error", str(error))

    # ==================================================================
    # SYSTEM STATUS  (UNCHANGED)
    # ==================================================================

    def set_system_status(self, text, status):

        self.system_badge.setText(text)

        if status == "ready":
            style = (
                "background:#123d2b;"
                "color:#7ee2a8;"
                "padding:6px 10px;"
                "border-radius:8px;"
                "font-weight:700;"
                "font-size:11px;"
            )
        elif status == "error":
            style = (
                "background:#54212a;"
                "color:#ff9aa8;"
                "padding:6px 10px;"
                "border-radius:8px;"
                "font-weight:700;"
                "font-size:11px;"
            )
        else:
            style = (
                "background:#17365f;"
                "color:#7fc4ff;"
                "padding:6px 10px;"
                "border-radius:8px;"
                "font-weight:700;"
                "font-size:11px;"
            )

        self.system_badge.setStyleSheet(style)

    # ==================================================================
    # CAMERA  (UNCHANGED)
    # ==================================================================

    def open_camera(self):

        self.camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = cv2.VideoCapture(CAMERA_INDEX)

        if not self.camera.isOpened():
            raise RuntimeError(
                "Could not open the camera.\n\n"
                "Please check:\n"
                "• Camera permissions\n"
                "• Camera connection\n"
                "• Another application using the camera"
            )

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        self.camera_state.setText("● CAMERA LIVE")

        self.camera_state.setStyleSheet(
            "background:#123d2b;"
            "color:#7ee2a8;"
            "padding:6px 10px;"
            "border-radius:8px;"
            "font-weight:700;"
            "font-size:11px;"
        )

    # ==================================================================
    # UPDATE CAMERA  (UNCHANGED)
    # ==================================================================

    def update_camera(self):

        if self.camera is None or not self.camera.isOpened():
            return

        success, frame = self.camera.read()

        if not success:
            self.camera_state.setText("● CAMERA ERROR")
            return

        self.last_frame = frame.copy()

        display = frame.copy()

        cv2.rectangle(
            display,
            (0, 0),
            (display.shape[1], 48),
            (5, 10, 18),
            -1,
        )

        cv2.putText(
            display,
            "LIVE CAMERA",
            (18, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (220, 235, 250),
            2,
            cv2.LINE_AA,
        )

        pixmap = resize_frame_for_label(
            display,
            self.live_label.width(),
            self.live_label.height(),
        )

        self.live_label.setPixmap(pixmap)

    # ==================================================================
    # CAPTURE PERMISSION  (UNCHANGED)
    # ==================================================================

    def can_capture(self):

        if self.capture_time is None:
            return True

        elapsed = time.monotonic() - self.capture_time

        if self.gemini_finished:
            return elapsed >= MIN_CAPTURE_INTERVAL

        return elapsed >= GEMINI_MAX_WAIT

    # ==================================================================
    # CAPTURE IMAGE
    # ==================================================================

    def capture_now(self):

        if self.last_frame is None:
            return

        if not self.can_capture():
            return

        frame = self.last_frame.copy()

        # ---------------- IMAGE NUMBER ----------------
        self.capture_number += 1

        self.capture_counter.setText(
            f"IMAGE {self.capture_number:02d}"
        )

        # ---------------- STORE FRAME ----------------
        self.captured_frame = frame.copy()

        pixmap = resize_frame_for_label(
            frame,
            self.captured_label.width(),
            self.captured_label.height(),
        )

        self.captured_label.setPixmap(pixmap)

        # ---------------- RESET TIMING ----------------
        self.capture_time = time.monotonic()

        self.gemini_finished = False
        self.gemini_running = False
        self.gemini_timeout_handled = False

        # ---------------- RESET DISPATCH GUARD ----------------
        self.dispatched_this_capture = False

        # ---------------- RESET RESULTS ----------------
        self.local_category = "—"
        self.local_confidence = None

        self.gemini_category = "ANALYZING"
        self.gemini_confidence = None
        self.gemini_raw = "Sending captured image to Gemini…"

        self.local_result.setText("ANALYZING…")
        self.local_confidence_label.setText("—")

        self.gemini_result.setText("ANALYZING…")
        self.gemini_confidence_label.setText("—")
        self.gemini_reply.setText(self.gemini_raw)

        # <-- Motor dispatch line reset (Gemini is primary now)
        self.dispatch_label.setText(
            "Motor: waiting for Gemini (local is fallback)…"
        )

        self.set_gemini_status("ANALYZING")

        # ==============================================================
        # LOCAL EFFICIENTNET — DISPLAY ONLY, do NOT dispatch.
        # Gemini (primary) will decide which bin to use.
        # Local result is kept as fallback for timeout / error cases.
        # ==============================================================

        try:
            category, confidence = self.local_model.predict(frame)

            self.local_category = category
            self.local_confidence = confidence

            self.local_result.setText(category)
            self.local_confidence_label.setText(
                f"{confidence:.1f}%"
            )

        except Exception as error:
            self.local_category = "ERROR"
            self.local_confidence = None

            self.local_result.setText("LOCAL ERROR")
            self.local_confidence_label.setText("—")

            self.gemini_reply.setText(
                "Local model error:\n"
                f"{error}"
            )

        # ==============================================================
        # PREPARE GEMINI IMAGE
        # ==============================================================

        try:
            image_bytes = make_gemini_image_bytes(frame)

        except Exception as error:
            self.gemini_finished = True
            self.gemini_category = "ERROR"

            self.gemini_result.setText("ERROR")
            self.gemini_reply.setText(
                f"Image preparation error:\n{error}"
            )

            self.set_gemini_status("ERROR")

            # Gemini cannot be reached — fall back to local for the bin.
            self.dispatch_local_fallback(
                "Gemini image prep failed"
            )

            self.add_history()

            return

        # ==============================================================
        # REQUEST ID
        # ==============================================================

        self.request_counter += 1
        request_id = self.request_counter
        self.current_request_id = request_id

        # ==============================================================
        # QUOTA COOLDOWN
        # ==============================================================

        now = time.monotonic()

        if now < self.quota_block_until:
            remaining = self.quota_block_until - now

            self.gemini_finished = True
            self.gemini_category = "QUOTA LIMIT"
            self.gemini_confidence = None

            self.gemini_result.setText("QUOTA LIMIT")
            self.gemini_confidence_label.setText("—")

            self.gemini_reply.setText(
                "Gemini quota cooldown is active.\n"
                f"Retry available in approximately "
                f"{remaining:.0f}s."
            )

            self.set_gemini_status("QUOTA LIMIT")

            # Gemini unavailable — fall back to local.
            self.dispatch_local_fallback(
                "Gemini quota cooldown"
            )

            self.add_history()

            return

        # ==============================================================
        # START GEMINI
        # ==============================================================

        self.gemini_running = True

        self.gemini_worker.start(request_id, image_bytes)

    # ==================================================================
    # MOTOR DISPATCH
    # ==================================================================

    def dispatch_class_to_motor(self, category, source="Gemini"):
        """
        Send the chosen class to the motion controller on a background
        thread so the GUI never blocks.

        `source` is only used for the on-screen status text.
        """

        bin_number = CLASS_TO_BIN.get(category)

        if bin_number is None:
            self._log_dispatch(category, None, False, source)
            return

        def _worker():
            ok = send_prediction_to_controller(category)
            QTimer.singleShot(
                0,
                lambda: self._log_dispatch(
                    category,
                    bin_number,
                    ok,
                    source,
                ),
            )

        threading.Thread(target=_worker, daemon=True).start()

    def dispatch_local_fallback(self, reason=""):
        """
        Use the Local EfficientNet result as the bin source.
        Only called when Gemini fails / times out / is quota-blocked.
        """

        if self.dispatched_this_capture:
            return

        if self.local_category in ("—", "ERROR", None):
            self.dispatch_label.setText(
                "Motor: no usable prediction (Gemini failed, "
                "local unavailable)"
            )
            self.dispatched_this_capture = True
            return

        self.dispatched_this_capture = True

        suffix = f" — fallback: {reason}" if reason else " — fallback"

        self.dispatch_label.setText(
            f"Motor: local fallback → {self.local_category}"
            f"{suffix}"
        )

        self.dispatch_class_to_motor(
            self.local_category,
            source="Local (fallback)",
        )

    def _log_dispatch(self, category, bin_number, ok, source):

        if bin_number is None:
            self.dispatch_label.setText(
                f"Motor [{source}]: '{category}' has no bin mapping"
            )
            return

        if ok:
            self.dispatch_label.setText(
                f"Motor [{source}]: sent {category} → BIN "
                f"{bin_number} ✓"
            )
        else:
            self.dispatch_label.setText(
                f"Motor [{source}]: unable to reach controller "
                f"for BIN {bin_number}"
            )

    # ==================================================================
    # GEMINI FINISHED — this is the PRIMARY dispatch path
    # ==================================================================

    def on_gemini_finished(self, request_id, result):

        if request_id != self.current_request_id:
            return

        self.gemini_running = False
        self.gemini_finished = True

        category = result.get("category", "UNKNOWN")
        confidence = result.get("confidence")
        raw_reply = result.get("raw", "")

        self.gemini_category = category
        self.gemini_confidence = confidence
        self.gemini_raw = raw_reply or "(Empty Gemini response)"

        self.gemini_result.setText(category)

        if confidence is None:
            self.gemini_confidence_label.setText("—")
        else:
            self.gemini_confidence_label.setText(
                f"{confidence:.1f}%"
            )

        self.gemini_reply.setText(self.gemini_raw)

        self.set_gemini_status("COMPLETE")

        # ----------------------------------------------------------
        # DISPATCH: Gemini is primary.
        # If Gemini returned a real class → use it.
        # Otherwise → fall back to Local.
        # ----------------------------------------------------------

        if (
            not self.dispatched_this_capture
            and category in CLASS_TO_BIN
        ):
            self.dispatched_this_capture = True
            self.dispatch_class_to_motor(
                category,
                source="Gemini",
            )
        elif not self.dispatched_this_capture:
            # Gemini returned something unusable (UNKNOWN etc.)
            self.dispatch_local_fallback(
                f"Gemini returned '{category}'"
            )

        self.add_history()

    # ==================================================================
    # GEMINI FAILED — fallback to local for bin selection
    # ==================================================================

    def on_gemini_failed(
        self,
        request_id,
        message,
        quota,
        retry_seconds,
    ):

        if request_id != self.current_request_id:
            return

        self.gemini_running = False
        self.gemini_finished = True

        if quota:
            cooldown = (
                float(retry_seconds)
                if retry_seconds is not None
                else 30.0
            )
            cooldown = max(1.0, cooldown)

            self.quota_block_until = max(
                self.quota_block_until,
                time.monotonic() + cooldown,
            )

            self.gemini_category = "QUOTA LIMIT"
            self.gemini_confidence = None

            self.gemini_result.setText("QUOTA LIMIT")
            self.gemini_confidence_label.setText("—")

            self.gemini_reply.setText(
                "Gemini API quota limit reached.\n\n"
                f"Google requested approximately "
                f"{cooldown:.0f}s before retrying.\n\n"
                "The application will NOT repeatedly "
                "send requests during the cooldown."
            )

            self.set_gemini_status("QUOTA LIMIT")

            self.dispatch_local_fallback(
                "Gemini quota limit"
            )

        else:
            self.gemini_category = "ERROR"
            self.gemini_confidence = None

            self.gemini_result.setText("ERROR")
            self.gemini_confidence_label.setText("—")

            self.gemini_reply.setText(
                "Gemini request failed:\n\n"
                f"{message}"
            )

            self.set_gemini_status("ERROR")

            self.dispatch_local_fallback(
                "Gemini error"
            )

        self.add_history()

    # ==================================================================
    # GEMINI TIMEOUT — fallback to local for bin selection
    # ==================================================================

    def handle_timeout(self):

        if self.gemini_finished:
            return

        if self.capture_time is None:
            return

        elapsed = time.monotonic() - self.capture_time

        if elapsed < GEMINI_MAX_WAIT:
            return

        self.gemini_finished = True
        self.gemini_timeout_handled = True

        self.gemini_category = "TIMEOUT"
        self.gemini_confidence = None

        self.gemini_result.setText("TIMEOUT")
        self.gemini_confidence_label.setText("—")

        self.gemini_reply.setText(
            "Gemini did not respond within "
            "the 30-second limit.\n\n"
            "Falling back to local EfficientNet for bin "
            "selection."
        )

        self.set_gemini_status("TIMEOUT")

        # ----------------------------------------------------------
        # Fallback: use local result for the bin.
        # ----------------------------------------------------------
        self.dispatch_local_fallback(
            "Gemini timeout (>30s)"
        )

        self.add_history()

    # ==================================================================
    # TIMING DISPLAY  (UNCHANGED)
    # ==================================================================

    def update_timing_display(self):

        if self.capture_time is None:
            self.timing_text.setText(
                "Ready • press SPACE to capture"
            )
            self.progress.setValue(0)
            return

        elapsed = time.monotonic() - self.capture_time

        if (
            not self.gemini_finished
            and elapsed >= GEMINI_MAX_WAIT
        ):
            self.handle_timeout()

        if self.gemini_finished:

            if elapsed < MIN_CAPTURE_INTERVAL:

                remaining = MIN_CAPTURE_INTERVAL - elapsed

                self.timing_text.setText(
                    "Gemini complete • "
                    f"next capture in "
                    f"{remaining:.1f}s"
                )

                progress = int(
                    min(
                        100.0,
                        (elapsed / MIN_CAPTURE_INTERVAL) * 100.0,
                    )
                )

                self.progress.setValue(progress)

            else:
                self.timing_text.setText(
                    "READY • press SPACE for next capture"
                )
                self.progress.setValue(100)

            return

        if elapsed < MIN_CAPTURE_INTERVAL:

            remaining = MIN_CAPTURE_INTERVAL - elapsed

            self.timing_text.setText(
                "Analyzing • "
                f"minimum interval "
                f"{remaining:.1f}s"
            )

            progress = int(
                (elapsed / MIN_CAPTURE_INTERVAL) * 100.0
            )

            self.progress.setValue(progress)

            return

        remaining = GEMINI_MAX_WAIT - elapsed

        self.timing_text.setText(
            "Waiting for Gemini • "
            f"timeout in {remaining:.1f}s"
        )

        progress = int(
            (elapsed / GEMINI_MAX_WAIT) * 100.0
        )

        self.progress.setValue(min(100, progress))

    # ==================================================================
    # GEMINI STATUS BADGE  (UNCHANGED)
    # ==================================================================

    def set_gemini_status(self, status):

        self.gemini_badge.setText(status)

        if status == "COMPLETE":
            style = (
                "background:#123d2b;"
                "color:#7ee2a8;"
                "padding:6px 10px;"
                "border-radius:8px;"
                "font-weight:700;"
                "font-size:11px;"
            )
        elif status in ("ERROR", "TIMEOUT"):
            style = (
                "background:#54212a;"
                "color:#ff9aa8;"
                "padding:6px 10px;"
                "border-radius:8px;"
                "font-weight:700;"
                "font-size:11px;"
            )
        elif status == "QUOTA LIMIT":
            style = (
                "background:#5a421b;"
                "color:#ffd477;"
                "padding:6px 10px;"
                "border-radius:8px;"
                "font-weight:700;"
                "font-size:11px;"
            )
        else:
            style = (
                "background:#17365f;"
                "color:#7fc4ff;"
                "padding:6px 10px;"
                "border-radius:8px;"
                "font-weight:700;"
                "font-size:11px;"
            )

        self.gemini_badge.setStyleSheet(style)

    # ==================================================================
    # HISTORY  (UNCHANGED)
    # ==================================================================

    def add_history(self):

        image_number = self.capture_number

        for item in self.history:
            if item["image"] == image_number:
                return

        self.history.appendleft(
            {
                "image": image_number,
                "local": self.local_category,
                "local_conf": self.local_confidence,
                "gemini": self.gemini_category,
                "gemini_conf": self.gemini_confidence,
            }
        )

        self.refresh_history()

    def refresh_history(self):

        while self.history_layout.count():

            item = self.history_layout.takeAt(0)
            widget = item.widget()

            if widget is not None:
                widget.deleteLater()

        for item in self.history:

            local_conf = (
                f"{item['local_conf']:.0f}%"
                if item["local_conf"] is not None
                else "—"
            )

            gemini_conf = (
                f"{item['gemini_conf']:.0f}%"
                if item["gemini_conf"] is not None
                else "—"
            )

            text = (
                f"#{item['image']:02d}   "
                f"Local: {item['local']} "
                f"({local_conf})   •   "
                f"Gemini: {item['gemini']} "
                f"({gemini_conf})"
            )

            label = QLabel(text)
            label.setWordWrap(True)

            label.setStyleSheet(
                """
                background:#111e32;
                border:1px solid #233653;
                border-radius:6px;
                padding:6px;
                color:#aebed3;
                font-size:10px;
                """
            )

            self.history_layout.addWidget(label)

        self.history_layout.addStretch()

    # ==================================================================
    # KEYBOARD  (UNCHANGED)
    # ==================================================================

    def keyPressEvent(self, event):

        key = event.key()

        if key == Qt.Key.Key_Space:
            self.capture_now()
            event.accept()
            return

        if key in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self.close()
            event.accept()
            return

        super().keyPressEvent(event)

    # ==================================================================
    # WINDOW CLOSE  (UNCHANGED)
    # ==================================================================

    def closeEvent(self, event):

        try:
            if hasattr(self, "camera_timer"):
                self.camera_timer.stop()
        except Exception:
            pass

        try:
            if hasattr(self, "status_timer"):
                self.status_timer.stop()
        except Exception:
            pass

        try:
            if self.camera is not None:
                self.camera.release()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        event.accept()


# ======================================================================
# APPLICATION ENTRY POINT
# ======================================================================

def main():

    print()
    print("=" * 72)
    print("AI POWERED WASTE SEGREGATION SYSTEM")
    print("PROFESSIONAL PYQT6 CAMERA GUI")
    print("=" * 72)
    print()

    print(f"Project : {PROJECT_ROOT}")
    print(f"Model   : {MODEL_PATH}")
    print(f"Gemini  : {GEMINI_MODEL}")
    print()

    print("Classes:")
    for class_name in CLASSES:
        print(f"  - {class_name}")

    print()

    print("Capture timing:")
    print(
        "  Minimum interval : "
        f"{MIN_CAPTURE_INTERVAL:.0f} seconds"
    )
    print(
        "  Gemini maximum   : "
        f"{GEMINI_MAX_WAIT:.0f} seconds"
    )

    print()

    print("Bin selection priority:")
    print("  PRIMARY  : Gemini")
    print("  FALLBACK : Local EfficientNet (only on Gemini failure/timeout)")

    print()

    print("Motor bridge:")
    print(f"  {MOTOR_CONTROLLER_HOST}:{MOTOR_CONTROLLER_PORT}")
    for cls, bin_number in CLASS_TO_BIN.items():
        print(f"  {cls:<12} -> BIN {bin_number}")

    print()

    print("Controls:")
    print("  SPACE = Capture")
    print("  Q     = Quit")
    print("  ESC   = Quit")
    print()

    app = QApplication(sys.argv)

    app.setApplicationName("AI Powered Waste Segregation System")
    app.setOrganizationName("AI Waste Segregation Project")

    window = WasteSegregationWindow()
    window.show()

    exit_code = app.exec()

    sys.exit(exit_code)


# ======================================================================
# RUN
# ======================================================================

if __name__ == "__main__":
    main()