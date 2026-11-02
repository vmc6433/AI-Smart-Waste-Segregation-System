import json
import sys
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import serial
from serial.tools import list_ports
from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtNetwork import QHostAddress, QTcpServer
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget
)

# ============================================================
#  PATHS
# ============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = SCRIPT_DIR / "config.json"

RESULTS_DIR = Path(
    r"C:\Users\vchet\OneDrive\Desktop\Major_Project"
    r"\AI_Waste_Segregation_System\results"
)
CAL_FILE = RESULTS_DIR / "calibration.json"

# ============================================================
#  CONFIG
# ============================================================
BAUD                  = 115200
POLL_MS               = 25
HANDSHAKE_RETRY_S     = 0.75
STATUS_INTERVAL_S     = 1.5
COMMAND_TIMEOUT_S     = 60.0
HOME_LOSS_CONFIRM     = 3
MAX_HOMING_RETRIES    = 3
HOMING_RETRY_DELAY_MS = 500

SERVO_HOME_ANGLE = 10
SERVO_MAX_ANGLE  = 70
MAX_STEPS        = 15000

DEFAULT_BIN_STEPS = {1: 1600, 2: 2400, 3: 0, 4: 800}

# ============================================================
#  PREDICTION BRIDGE (camera_predict  ->  motor controller)
# ============================================================
CLASS_TO_BIN = {
    "E_WASTE":    1,
    "ORGANIC":    2,
    "RECYCLABLE": 3,   # <-- requested assignment
    "PLASTIC":    4,
}

COMMAND_SERVER_HOST = "127.0.0.1"
COMMAND_SERVER_PORT = 9999


# ============================================================
#  PERSISTENT CONFIG
# ============================================================
class AppConfig:
    def __init__(self, path: Path):
        self.path = path
        self.bin_steps = dict(DEFAULT_BIN_STEPS)
        self.load()

    def load(self):
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                bs = data.get("bin_steps", {})
                for k, v in bs.items():
                    self.bin_steps[int(k)] = int(v)
        except Exception:
            pass

    def save(self):
        try:
            self.path.write_text(
                json.dumps({"bin_steps": {str(k): int(v)
                                          for k, v in self.bin_steps.items()}},
                           indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass


# ============================================================
#  DATA
# ============================================================
@dataclass
class MachineState:
    connected: bool = False
    handshake: bool = False
    busy: bool = False
    servo: int = SERVO_HOME_ANGLE
    motor: int = 0
    servo_home: bool = False
    servo_max: bool = False
    motor_home: bool = False
    motor_end: bool = False
    holding: bool = True
    fault: bool = False


@dataclass
class Calibration:
    timestamp: str = ""
    servo_home_angle: int = SERVO_HOME_ANGLE
    servo_max_angle: int = SERVO_MAX_ANGLE
    nema_home_to_end: int = 0
    nema_end_to_home: int = 0

    def save(self, path: Path) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
            return True
        except Exception:
            return False


# ============================================================
#  SERIAL
# ============================================================
class SerialController(QObject):
    line_received = pyqtSignal(str)
    connected_changed = pyqtSignal(bool)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.ser = None
        self.buffer = bytearray()

    def ports(self):
        return [p.device for p in list_ports.comports()]

    def open(self, port):
        self.close(silent=True)
        try:
            self.log.emit(f"Opening {port} @ {BAUD}...")
            self.ser = serial.Serial(
                port=port, baudrate=BAUD, timeout=0, write_timeout=1,
                dsrdtr=False, rtscts=False, xonxoff=False,
            )
            time.sleep(1.2)
            self.buffer.clear()
            self.connected_changed.emit(True)
            self.log.emit("SERIAL OPENED")
            return True
        except Exception as exc:
            self.ser = None
            self.error.emit(f"Serial open failed: {exc}")
            return False

    def close(self, silent=False):
        was_open = bool(self.ser)
        try:
            if self.ser: self.ser.close()
        except Exception:
            pass
        self.ser = None
        self.buffer.clear()
        if was_open:
            self.connected_changed.emit(False)
        if was_open and not silent:
            self.log.emit("SERIAL CLOSED")

    def is_open(self):
        return bool(self.ser and self.ser.is_open)

    def send(self, command, log_command=True):
        if not self.is_open():
            self.error.emit("Connect to the UNO first.")
            return False
        try:
            self.ser.write((command.strip() + "\n").encode("ascii"))
            if log_command:
                self.log.emit(f"> {command.strip()}")
            return True
        except Exception as exc:
            self.error.emit(f"Serial write failed: {exc}")
            self.close(silent=True)
            return False

    def poll(self):
        if not self.is_open():
            return
        try:
            n = self.ser.in_waiting
            if n <= 0:
                return
            self.buffer.extend(self.ser.read(n))
            while b"\n" in self.buffer:
                raw, _, self.buffer = self.buffer.partition(b"\n")
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    self.line_received.emit(line)
        except Exception as exc:
            self.error.emit(f"Serial read failed: {exc}")
            self.close(silent=True)


# ============================================================
#  MACHINE CONTROLLER
# ============================================================
class MachineController(QObject):
    state_changed = pyqtSignal(object)
    phase_changed = pyqtSignal(str)
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    calibration_updated = pyqtSignal(object)

    def __init__(self, serial_controller):
        super().__init__()
        self.serial = serial_controller
        self.serial.line_received.connect(self.handle_line)
        self.serial.connected_changed.connect(self.on_connection)

        self.state = MachineState()
        self.phase = "DISCONNECTED"
        self.mode = "IDLE"
        self.step = None
        self.homing_reason = ""

        self.pending = None
        self.hello_due = 0.0
        self.last_status = 0.0
        self.home_loss_counter = 0

        self.homing_retries = 0
        self.homing_generation = 0

        self.seq_steps = 0
        self.seq_label = ""

        self.cal = Calibration()

    # ---------------- connection ----------------
    def on_connection(self, connected):
        self.state = MachineState(connected=connected)
        self.mode = "IDLE"
        self.step = None
        self.pending = None
        self.home_loss_counter = 0
        self.homing_retries = 0
        self.homing_generation += 1
        self.set_phase("CONNECTED" if connected else "DISCONNECTED")
        self.emit_state()
        if connected:
            self.hello_due = 0.0

    # ---------------- periodic ----------------
    def tick(self):
        if not self.serial.is_open():
            return
        now = time.monotonic()

        if not self.state.handshake:
            if now >= self.hello_due:
                self.hello_due = now + HANDSHAKE_RETRY_S
                self.serial.send("HELLO")
            return

        if self.pending and now >= self.pending["deadline"]:
            cmd = self.pending["command"]
            self.pending = None
            self.handle_error(f"TIMEOUT:{cmd}")
            return

        if self.mode == "IDLE" and self.phase == "READY":
            if not self.state.motor_home:
                self.home_loss_counter += 1
                if self.home_loss_counter >= HOME_LOSS_CONFIRM:
                    self.home_loss_counter = 0
                    self.log.emit("NEMA HOME lost — auto-returning")
                    self.start_homing("HOME_LOST")
                    return
            else:
                self.home_loss_counter = 0

        if self.mode == "IDLE" and self.phase == "READY" and not self.pending:
            if now - self.last_status >= STATUS_INTERVAL_S:
                self.last_status = now
                self.serial.send("STATUS", log_command=False)

    # ---------------- helpers ----------------
    def set_phase(self, phase):
        self.phase = phase
        self.phase_changed.emit(phase)

    def emit_state(self):
        self.state_changed.emit(self.state)

    def send_with_timeout(self, command):
        self.pending = {
            "command": command,
            "deadline": time.monotonic() + COMMAND_TIMEOUT_S,
        }
        self.serial.send(command)

    def clear_pending(self):
        self.pending = None

    # ---------------- serial handler ----------------
    def handle_line(self, line):
        parts = line.split(",")
        head = parts[0]

        if head == "STATE":
            self.parse_state(parts)
            return

        if head.startswith("DBG"):
            self.log.emit(f"  [UNO] {line}")
            return

        if head == "UNO_READY":
            if self.state.handshake:
                self.log.emit("UNO reset detected — re-handshaking")
                self.state.handshake = False
                self.pending = None
                self.mode = "IDLE"
                self.step = None
                self.set_phase("RE-HANDSHAKING")
                self.emit_state()
            self.hello_due = 0.0
            return

        if head == "HELLO_ACK":
            if self.state.handshake:
                return
            self.state.handshake = True
            self.clear_pending()
            self.log.emit("HANDSHAKE OK — auto-homing")
            self.start_homing("STARTUP")
            return

        if head == "START":
            self.state.busy = True
            self.emit_state()
            return

        if head == "WARN":
            self.log.emit("WARN: " + ",".join(parts[1:]))
            return

        if head == "DONE":
            self.handle_done(parts)
            return

        if head == "ERROR":
            self.handle_error(",".join(parts[1:]) or "UNKNOWN")
            return

        if head == "STOPPED":
            self.clear_pending()
            self.state.busy = False
            self.mode = "IDLE"
            self.step = None
            self.set_phase("STOPPED — press HOME ALL")
            self.emit_state()
            return

        self.log.emit(f"< {line}")

    def parse_state(self, parts):
        if len(parts) != 9:
            self.error.emit(f"Malformed STATE: {','.join(parts)}")
            return
        try:
            self.state.servo      = int(parts[1])
            self.state.motor      = int(parts[2])
            self.state.servo_home = bool(int(parts[3]))
            self.state.servo_max  = bool(int(parts[4]))
            self.state.motor_home = bool(int(parts[5]))
            self.state.motor_end  = bool(int(parts[6]))
            self.state.holding    = bool(int(parts[7]))
            self.state.busy       = bool(int(parts[8]))
        except ValueError:
            self.error.emit("Malformed STATE values")
            return
        self.emit_state()

    # ============================================================
    #  HOMING
    # ============================================================
    def start_homing(self, reason, reset_retries=True):
        if not self.state.handshake:
            return
        if reset_retries:
            self.homing_retries = 0
        self.homing_generation += 1
        self.mode = "HOMING"
        self.step = "HOMING_SERVO"
        self.homing_reason = reason
        self.state.busy = True
        self.state.fault = False
        self.clear_pending()
        self.set_phase(f"HOMING ({reason}) — SERVO HOME")
        self.log.emit(f"[HOMING] {reason}: servo home first")
        self.send_with_timeout("SERVO_HOME")

    def _schedule_homing_retry(self, gen, reason, detail):
        def _fire():
            if gen != self.homing_generation:
                return
            if not self.state.handshake:
                return
            self.start_homing(reason, reset_retries=False)
        QTimer.singleShot(HOMING_RETRY_DELAY_MS, _fire)

    # ============================================================
    #  SEQUENCE  (BIN + CW)
    #  steps = 0  ->  skip NEMA move, go directly to servo cycle
    # ============================================================
    def start_sequence(self, steps, label="CW MOVE"):
        if not self.state.handshake:
            self.error.emit("Connect to the UNO first.")
            return
        if self.mode != "IDLE" or self.pending or self.state.busy:
            self.error.emit("Machine busy.")
            return
        if self.phase != "READY":
            self.error.emit("Not at HOME. Press HOME ALL first.")
            return
        if steps < 0 or steps > MAX_STEPS:
            self.error.emit(f"Steps must be 0–{MAX_STEPS}.")
            return

        self.mode = "SEQUENCE"
        self.seq_steps = steps
        self.seq_label = label
        self.state.busy = True
        self.clear_pending()

        if steps == 0:
            # BIN at HOME — NEMA stays put, just run the servo cycle.
            self.step = "SEQ_SERVO_MAX"
            self.set_phase(f"{label} — at HOME, servo MAX")
            self.log.emit(f"[{label}] NEMA at HOME (0 steps) — servo MAX")
            self.send_with_timeout("SERVO_MAX")
            return

        self.step = "SEQ_CW"
        self.set_phase(f"{label} — NEMA CW {steps} steps")
        self.log.emit(f"[{label}] NEMA CW {steps} steps")
        self.send_with_timeout(f"NEMA_CW {steps}")

    # ============================================================
    #  CALIBRATION
    # ============================================================
    def start_calibration(self):
        if not self.state.handshake:
            self.error.emit("Connect to the UNO first.")
            return
        if self.mode != "IDLE" or self.pending or self.state.busy:
            self.error.emit("Machine busy.")
            return
        if self.phase != "READY":
            self.error.emit("Not at HOME. Press HOME ALL first.")
            return

        self.mode = "CALIBRATING"
        self.step = "CAL_SERVO_HOME"
        self.cal = Calibration()
        self.state.busy = True
        self.clear_pending()
        self.set_phase("CAL — servo home")
        self.log.emit("[CAL] 1/6: servo HOME")
        self.send_with_timeout("SERVO_HOME")

    # ============================================================
    #  DONE dispatcher
    # ============================================================
    def handle_done(self, parts):
        action = parts[1] if len(parts) > 1 else ""
        payload = parts[2] if len(parts) > 2 else ""
        self.clear_pending()

        if self.mode == "HOMING":
            self._handle_homing_done(action)
            return
        if self.mode == "SEQUENCE":
            self._handle_sequence_done(action)
            return
        if self.mode == "CALIBRATING":
            self._handle_cal_done(action, payload)
            return

    def _handle_homing_done(self, action):
        if self.step == "HOMING_SERVO" and action == "SERVO_HOME":
            self.step = "HOMING_NEMA"
            self.set_phase("HOMING — NEMA HOME")
            self.log.emit("[HOMING] servo home ✓ — NEMA home")
            self.send_with_timeout("NEMA_HOME")
            return
        if self.step == "HOMING_NEMA" and action == "NEMA_HOME":
            self.mode = "IDLE"
            self.step = None
            self.state.busy = False
            self.state.motor = 0
            self.state.servo = SERVO_HOME_ANGLE
            self.home_loss_counter = 0
            self.homing_retries = 0
            self.set_phase("READY")
            self.emit_state()
            self.log.emit("[HOMING] complete — READY at HOME")
            return

    def _handle_sequence_done(self, action):
        label = self.seq_label
        # (SEQ_CW only occurs when steps > 0)
        if self.step == "SEQ_CW" and action == "NEMA_MOVE":
            self.step = "SEQ_SERVO_MAX"
            self.set_phase(f"{label} — servo MAX")
            self.log.emit(f"[{label}] NEMA positioned — servo MAX")
            self.send_with_timeout("SERVO_MAX")
            return
        if self.step == "SEQ_SERVO_MAX" and action == "SERVO_MAX":
            self.step = "SEQ_SERVO_HOME"
            self.set_phase(f"{label} — servo HOME")
            self.log.emit(f"[{label}] servo MAX ✓ — servo HOME")
            self.send_with_timeout("SERVO_HOME")
            return
        if self.step == "SEQ_SERVO_HOME" and action == "SERVO_HOME":
            self.step = "SEQ_NEMA_HOME"
            self.set_phase(f"{label} — NEMA HOME")
            self.log.emit(f"[{label}] servo HOME ✓ — NEMA home")
            self.send_with_timeout("NEMA_HOME")
            return
        if self.step == "SEQ_NEMA_HOME" and action == "NEMA_HOME":
            self.mode = "IDLE"
            self.step = None
            self.state.busy = False
            self.state.motor = 0
            self.state.servo = SERVO_HOME_ANGLE
            self.home_loss_counter = 0
            self.set_phase("READY")
            self.emit_state()
            self.log.emit(f"[{label}] COMPLETE ✓ — READY at HOME")
            return

    def _handle_cal_done(self, action, payload):
        s = self.step

        if s == "CAL_SERVO_HOME" and action == "SERVO_HOME":
            self.cal.servo_home_angle = self.state.servo
            self.step = "CAL_SERVO_MAX"
            self.set_phase(f"CAL — servo home = {self.cal.servo_home_angle}° → MAX")
            self.log.emit(f"[CAL] 2/6: servo HOME = {self.cal.servo_home_angle}°")
            self.send_with_timeout("SERVO_MAX")
            return

        if s == "CAL_SERVO_MAX" and action == "SERVO_MAX":
            self.cal.servo_max_angle = self.state.servo
            self.step = "CAL_SERVO_HOME2"
            self.set_phase(f"CAL — servo max = {self.cal.servo_max_angle}° → HOME")
            self.log.emit(f"[CAL] 3/6: servo MAX = {self.cal.servo_max_angle}°")
            self.send_with_timeout("SERVO_HOME")
            return

        if s == "CAL_SERVO_HOME2" and action == "SERVO_HOME":
            self.step = "CAL_NEMA_HOME"
            self.set_phase("CAL — NEMA HOME")
            self.log.emit("[CAL] 4/6: NEMA → HOME")
            self.send_with_timeout("NEMA_HOME")
            return

        if s == "CAL_NEMA_HOME" and action == "NEMA_HOME":
            self.step = "CAL_NEMA_FIND_END"
            self.set_phase("CAL — NEMA HOME → END (counting)")
            self.log.emit("[CAL] 5/6: counting NEMA HOME → END")
            self.send_with_timeout("NEMA_FIND_END")
            return

        if s == "CAL_NEMA_FIND_END" and action == "NEMA_FIND_END":
            try:
                self.cal.nema_home_to_end = int(payload)
            except ValueError:
                self.cal.nema_home_to_end = 0
            self.step = "CAL_NEMA_FIND_HOME"
            self.set_phase(f"CAL — END = {self.cal.nema_home_to_end} steps → HOME")
            self.log.emit(f"[CAL] home→end = {self.cal.nema_home_to_end} steps")
            self.log.emit("[CAL] 6/6: counting NEMA END → HOME")
            self.send_with_timeout("NEMA_FIND_HOME")
            return

        if s == "CAL_NEMA_FIND_HOME" and action == "NEMA_FIND_HOME":
            try:
                self.cal.nema_end_to_home = int(payload)
            except ValueError:
                self.cal.nema_end_to_home = 0
            self.cal.timestamp = datetime.now().isoformat(timespec="seconds")

            ok = self.cal.save(CAL_FILE)
            backlash = abs(self.cal.nema_home_to_end - self.cal.nema_end_to_home)

            self.mode = "IDLE"
            self.step = None
            self.state.busy = False
            self.set_phase("READY")
            self.emit_state()

            if ok:
                self.log.emit(f"[CAL] end→home = {self.cal.nema_end_to_home} steps")
                self.log.emit(f"[CAL] backlash estimate = {backlash} steps")
                self.log.emit(f"[CAL] saved → {CAL_FILE}")
            else:
                self.error.emit(f"[CAL] could not write {CAL_FILE}")

            self.calibration_updated.emit(self.cal)
            return

    # ============================================================
    #  ERROR HANDLING
    # ============================================================
    def handle_error(self, detail):
        self.clear_pending()
        self.state.busy = False
        self.error.emit(detail)

        if not self.state.handshake:
            return

        if "NO_HANDSHAKE" in detail:
            self.log.emit("UNO reports no handshake — re-handshaking")
            self.state.handshake = False
            self.mode = "IDLE"
            self.step = None
            self.set_phase("RE-HANDSHAKING")
            self.emit_state()
            return

        if self.mode == "HOMING":
            if self.homing_retries < MAX_HOMING_RETRIES:
                self.homing_retries += 1
                n = self.homing_retries
                gen = self.homing_generation
                self.log.emit(
                    f"[RECOVERY] Homing retry {n}/{MAX_HOMING_RETRIES} — {detail}"
                )
                self._schedule_homing_retry(gen, f"RETRY_{n}", detail)
                return
            self.mode = "IDLE"
            self.step = None
            self.state.fault = True
            self.set_phase("FAULT — press HOME ALL")
            self.emit_state()
            self.error.emit(f"Homing failed after {MAX_HOMING_RETRIES} retries")
            return

        self.log.emit(f"[RECOVERY] restarting homing after {self.mode} error")
        self.start_homing(f"RECOVER:{detail}", reset_retries=True)

    # ============================================================
    #  USER CONTROLS
    # ============================================================
    def manual_home(self):
        if not self.state.handshake:
            self.error.emit("Connect to the UNO first.")
            return
        if self.mode == "HOMING":
            return
        self.start_homing("MANUAL", reset_retries=True)

    def manual_servo_home(self):
        if self.mode != "IDLE" or self.state.busy:
            self.error.emit("Machine busy.")
            return
        self.state.busy = True
        self.send_with_timeout("SERVO_HOME")

    def manual_servo_max(self):
        if self.mode != "IDLE" or self.state.busy:
            self.error.emit("Machine busy.")
            return
        self.state.busy = True
        self.send_with_timeout("SERVO_MAX")

    def manual_nema_home(self):
        if self.mode != "IDLE" or self.state.busy:
            self.error.emit("Machine busy.")
            return
        self.state.busy = True
        self.send_with_timeout("NEMA_HOME")

    def manual_nema_end(self):
        if self.mode != "IDLE" or self.state.busy:
            self.error.emit("Machine busy.")
            return
        self.state.busy = True
        self.send_with_timeout("NEMA_END")

    def manual_nema_ccw(self, steps):
        if self.mode != "IDLE" or self.state.busy:
            self.error.emit("Machine busy.")
            return
        if not (1 <= steps <= MAX_STEPS):
            self.error.emit(f"Steps must be 1–{MAX_STEPS}.")
            return
        self.state.busy = True
        self.send_with_timeout(f"NEMA_CCW {steps}")

    def stop(self):
        if self.serial.is_open():
            self.serial.send("STOP")


# ============================================================
#  COMMAND SERVER  (listens for "CLASS:<name>" from camera_predict)
# ============================================================
class CommandServer(QObject):
    command_received = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, host=COMMAND_SERVER_HOST, port=COMMAND_SERVER_PORT, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = port
        self.server = QTcpServer(self)
        self.server.newConnection.connect(self._on_new_connection)
        self._clients = []

    def start(self) -> bool:
        if self.server.isListening():
            return True
        if not self.server.listen(QHostAddress(self.host), self.port):
            self.log.emit(f"[IPC] Failed to listen on {self.host}:{self.port}")
            return False
        self.log.emit(f"[IPC] Listening on {self.host}:{self.port}")
        return True

    def stop(self):
        for sock in list(self._clients):
            try:
                sock.disconnectFromHost()
            except Exception:
                pass
        self._clients.clear()
        if self.server.isListening():
            self.server.close()
        self.log.emit("[IPC] Server stopped")

    def _on_new_connection(self):
        while self.server.hasPendingConnections():
            sock = self.server.nextPendingConnection()
            self._clients.append(sock)
            sock.readyRead.connect(lambda s=sock: self._on_ready_read(s))
            sock.disconnected.connect(lambda s=sock: self._on_disconnected(s))

    def _on_ready_read(self, sock):
        raw = bytes(sock.readAll())
        if not raw:
            return
        text = raw.decode("utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            if line:
                self.command_received.emit(line)
        try:
            sock.write(b"OK\n")
            sock.flush()
        except Exception:
            pass

    def _on_disconnected(self, sock):
        try:
            self._clients.remove(sock)
        except ValueError:
            pass
        sock.deleteLater()


# ============================================================
#  GUI
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Waste Segregation — Motion Controller")
        self.resize(1150, 900)

        self.config = AppConfig(CONFIG_FILE)

        self.serial = SerialController()
        self.controller = MachineController(self.serial)

        self.port_box     = QComboBox()
        self.refresh_btn  = QPushButton("Refresh Ports")
        self.connect_btn  = QPushButton("Connect")

        self.ready_label  = QLabel("DISCONNECTED")
        self.phase_label  = QLabel("Phase: DISCONNECTED")
        self.servo_label  = QLabel("Servo: --°")
        self.motor_label  = QLabel("NEMA: -- steps")
        self.limits_label = QLabel("Limits: --")
        self.cal_label    = QLabel("Calibration: not loaded")

        self.home_all_btn  = QPushButton("HOME ALL")
        self.calibrate_btn = QPushButton("CALIBRATE")
        self.stop_btn      = QPushButton("STOP")

        self.servo_home_btn = QPushButton("Servo HOME")
        self.servo_max_btn  = QPushButton("Servo MAX")

        self.nema_home_btn = QPushButton("NEMA HOME")
        self.nema_end_btn  = QPushButton("NEMA END")

        self.cw_edit  = QLineEdit("1600")
        self.cw_btn   = QPushButton("CW MOVE  (full cycle)")
        self.ccw_edit = QLineEdit("800")
        self.ccw_btn  = QPushButton("CCW Move  (simple)")

        self.bin_edits = {}
        self.bin_buttons = {}

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.build_ui()
        self.connect_signals()
        self.refresh_ports()
        self.load_calibration_display()

        self.timer = QTimer(self)
        self.timer.setInterval(POLL_MS)
        self.timer.timeout.connect(self.serial.poll)
        self.timer.timeout.connect(self.controller.tick)
        self.timer.start()

        # --------------------------------------------------------
        # PREDICTION BRIDGE  (TCP server for camera_predict)
        # --------------------------------------------------------
        self.pending_bins = deque()
        self.command_server = CommandServer()
        self.command_server.log.connect(self.log)
        self.command_server.command_received.connect(self.on_external_command)
        if not self.command_server.start():
            self.log("WARNING: prediction bridge could not start.")

        self.update_buttons()

    def build_ui(self):
        root = QWidget()
        main = QVBoxLayout(root)

        title = QLabel("AI WASTE SEGREGATION SYSTEM")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        main.addWidget(title)

        conn = QGroupBox("UNO Connection")
        cr = QHBoxLayout(conn)
        cr.addWidget(QLabel("COM Port:"))
        cr.addWidget(self.port_box, 1)
        cr.addWidget(self.refresh_btn)
        cr.addWidget(self.connect_btn)
        main.addWidget(conn)

        st = QGroupBox("Machine Status")
        sg = QGridLayout(st)
        sg.addWidget(QLabel("State:"), 0, 0)
        sg.addWidget(self.ready_label, 0, 1)
        sg.addWidget(self.phase_label, 0, 2, 1, 2)
        sg.addWidget(self.servo_label, 1, 0)
        sg.addWidget(self.motor_label, 1, 1)
        sg.addWidget(self.limits_label, 2, 0, 1, 4)
        sg.addWidget(self.cal_label, 3, 0, 1, 4)
        main.addWidget(st)

        mc = QGroupBox("Main Controls")
        mr = QHBoxLayout(mc)
        mr.addWidget(self.home_all_btn)
        mr.addWidget(self.calibrate_btn)
        mr.addWidget(self.stop_btn)
        mr.addStretch(1)
        main.addWidget(mc)

        sv = QGroupBox("Servo Control  (holds position — no auto-detach)")
        svg = QHBoxLayout(sv)
        svg.addWidget(self.servo_home_btn)
        svg.addWidget(self.servo_max_btn)
        svg.addStretch(1)
        main.addWidget(sv)

        nm = QGroupBox("NEMA 17 Manual Control  (1/16 microstep, holding torque ON)")
        nmg = QHBoxLayout(nm)
        nmg.addWidget(self.nema_home_btn)
        nmg.addWidget(self.nema_end_btn)
        nmg.addStretch(1)
        main.addWidget(nm)

        cwg = QGroupBox("CW MOVE — Full Cycle Sequence")
        cwgl = QGridLayout(cwg)
        cwgl.addWidget(QLabel("CW Steps:"), 0, 0)
        cwgl.addWidget(self.cw_edit, 0, 1)
        cwgl.addWidget(self.cw_btn, 0, 2)
        hint = QLabel("Sequence: NEMA CW → Servo MAX → Servo HOME → NEMA HOME")
        hint.setWordWrap(True)
        cwgl.addWidget(hint, 1, 0, 1, 3)
        main.addWidget(cwg)

        ccwg = QGroupBox("CCW Move  (simple, no servo cycle)")
        ccwgl = QHBoxLayout(ccwg)
        ccwgl.addWidget(QLabel("CCW Steps:"))
        ccwgl.addWidget(self.ccw_edit)
        ccwgl.addWidget(self.ccw_btn)
        ccwgl.addStretch(1)
        main.addWidget(ccwg)

        bins = QGroupBox(
            "BIN POSITIONS — editable steps  "
            "(0 steps = servo cycle only, NEMA stays at HOME)"
        )
        bg = QGridLayout(bins)
        for col, number in enumerate((1, 2, 3, 4)):
            lbl = QLabel(f"BIN {number} — steps")
            edit = QLineEdit(str(self.config.bin_steps.get(number, 0)))
            edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            edit.editingFinished.connect(
                lambda n=number: self.on_bin_edit_changed(n)
            )
            button = QPushButton(f"GO TO BIN {number}")
            button.setMinimumHeight(60)
            button.clicked.connect(lambda _=False, n=number: self.on_bin_pressed(n))
            self.bin_edits[number] = edit
            self.bin_buttons[number] = button
            bg.addWidget(lbl, 0, col)
            bg.addWidget(edit, 1, col)
            bg.addWidget(button, 2, col)
        main.addWidget(bins)

        lg = QGroupBox("Event Log")
        lgl = QVBoxLayout(lg)
        lgl.addWidget(self.log_box)
        main.addWidget(lg, 1)

        self.setCentralWidget(root)

    def connect_signals(self):
        self.refresh_btn.clicked.connect(self.refresh_ports)
        self.connect_btn.clicked.connect(self.toggle_connection)

        self.home_all_btn.clicked.connect(self.controller.manual_home)
        self.calibrate_btn.clicked.connect(self.controller.start_calibration)
        self.stop_btn.clicked.connect(self.controller.stop)

        self.servo_home_btn.clicked.connect(self.controller.manual_servo_home)
        self.servo_max_btn.clicked.connect(self.controller.manual_servo_max)

        self.nema_home_btn.clicked.connect(self.controller.manual_nema_home)
        self.nema_end_btn.clicked.connect(self.controller.manual_nema_end)

        self.cw_btn.clicked.connect(self.on_cw_pressed)
        self.ccw_btn.clicked.connect(self.on_ccw_pressed)

        self.serial.log.connect(self.log)
        self.serial.error.connect(self.error)
        self.controller.log.connect(self.log)
        self.controller.error.connect(self.error)
        self.controller.state_changed.connect(self.update_state)
        self.controller.phase_changed.connect(self.update_phase)
        self.controller.calibration_updated.connect(self.on_cal_updated)
        self.serial.connected_changed.connect(lambda _: self.update_buttons())

    def on_cw_pressed(self):
        try:
            steps = int(self.cw_edit.text().strip())
        except ValueError:
            self.error("CW: enter a whole number.")
            return
        self.controller.start_sequence(steps, label="CW MOVE")

    def on_ccw_pressed(self):
        try:
            steps = int(self.ccw_edit.text().strip())
        except ValueError:
            self.error("CCW: enter a whole number.")
            return
        self.controller.manual_nema_ccw(steps)

    def on_bin_pressed(self, number):
        try:
            steps = int(self.bin_edits[number].text().strip())
        except ValueError:
            self.error(f"BIN {number}: enter a whole number.")
            return
        self.config.bin_steps[number] = steps
        self.config.save()
        self.controller.start_sequence(steps, label=f"BIN {number}")

    def on_bin_edit_changed(self, number):
        try:
            steps = int(self.bin_edits[number].text().strip())
            self.config.bin_steps[number] = steps
            self.config.save()
        except ValueError:
            pass

    def on_cal_updated(self, cal: Calibration):
        self.log("═══ CALIBRATION SAVED ═══")
        self.log(f"  Servo HOME  : {cal.servo_home_angle}°")
        self.log(f"  Servo MAX   : {cal.servo_max_angle}°")
        self.log(f"  HOME → END  : {cal.nema_home_to_end} steps")
        self.log(f"  END → HOME  : {cal.nema_end_to_home} steps")
        self.log("═════════════════════════")
        self.load_calibration_display()

    def load_calibration_display(self):
        try:
            if CAL_FILE.exists():
                data = json.loads(CAL_FILE.read_text(encoding="utf-8"))
                self.cal_label.setText(
                    f"Calibration: servo {data.get('servo_home_angle','?')}°–"
                    f"{data.get('servo_max_angle','?')}°  |  "
                    f"H→E {data.get('nema_home_to_end','?')}  "
                    f"E→H {data.get('nema_end_to_home','?')}  "
                    f"({data.get('timestamp','')})"
                )
            else:
                self.cal_label.setText("Calibration: not saved yet")
        except Exception:
            self.cal_label.setText("Calibration: error reading file")

    def refresh_ports(self):
        current = self.port_box.currentText()
        self.port_box.clear()
        ports = self.serial.ports()
        self.port_box.addItems(ports)
        if current in ports:
            self.port_box.setCurrentText(current)
        self.log(f"Ports: {', '.join(ports) if ports else 'none'}")

    def toggle_connection(self):
        if self.serial.is_open():
            self.serial.close()
            return
        port = self.port_box.currentText().strip()
        if not port:
            self.error("Select a COM port first.")
            return
        self.serial.open(port)

    def update_state(self, state):
        self.servo_label.setText(f"Servo: {state.servo}°")
        self.motor_label.setText(f"NEMA: {state.motor} steps")
        self.limits_label.setText(
            f"SERVO HOME: {'●' if state.servo_home else '○'}   "
            f"SERVO MAX: {'●' if state.servo_max else '○'}   "
            f"NEMA HOME: {'●' if state.motor_home else '○'}   "
            f"NEMA END: {'●' if state.motor_end else '○'}   "
            f"Holding torque: {'ON' if state.holding else 'OFF'}"
        )
        self.update_buttons()

    def update_phase(self, phase):
        self.phase_label.setText(f"Phase: {phase}")
        self.update_buttons()

    def update_buttons(self):
        connected = self.serial.is_open()
        ctrl = self.controller

        ready = (
            connected
            and ctrl.state.handshake
            and ctrl.phase == "READY"
            and ctrl.mode == "IDLE"
            and not ctrl.state.busy
        )
        active = ctrl.state.busy or ctrl.mode in {"HOMING", "SEQUENCE", "CALIBRATING"}

        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.port_box.setEnabled(not connected)
        self.refresh_btn.setEnabled(not connected)

        self.home_all_btn.setEnabled(connected and ctrl.state.handshake and ctrl.mode != "HOMING")
        self.calibrate_btn.setEnabled(ready)
        self.stop_btn.setEnabled(connected and active)

        self.servo_home_btn.setEnabled(ready)
        self.servo_max_btn.setEnabled(ready)

        self.nema_home_btn.setEnabled(ready)
        self.nema_end_btn.setEnabled(ready)
        self.cw_btn.setEnabled(ready)
        self.cw_edit.setEnabled(ready)
        self.ccw_btn.setEnabled(ready)
        self.ccw_edit.setEnabled(ready)

        for n in self.bin_buttons:
            self.bin_buttons[n].setEnabled(ready)
            self.bin_edits[n].setEnabled(ready)

        if not connected:
            self.ready_label.setText("DISCONNECTED")
        elif not ctrl.state.handshake:
            self.ready_label.setText("HANDSHAKING")
        elif ctrl.mode == "HOMING":
            self.ready_label.setText(f"HOMING  ({ctrl.homing_reason})")
        elif ctrl.mode == "SEQUENCE":
            self.ready_label.setText(f"RUNNING {ctrl.seq_label}")
        elif ctrl.mode == "CALIBRATING":
            self.ready_label.setText("CALIBRATING")
        elif ctrl.state.fault:
            self.ready_label.setText("FAULT — press HOME ALL")
        elif ctrl.state.busy:
            self.ready_label.setText("BUSY")
        elif ready:
            self.ready_label.setText("READY — HOME")
        else:
            self.ready_label.setText("WAITING")

        # Fire queued prediction-driven bins as soon as we're READY.
        self._drain_pending_bins()

    # ========================================================
    #  EXTERNAL COMMANDS  (from camera_predict)
    # ========================================================
    def _is_ready(self):
        ctrl = self.controller
        return (
            self.serial.is_open()
            and ctrl.state.handshake
            and ctrl.phase == "READY"
            and ctrl.mode == "IDLE"
            and not ctrl.state.busy
        )

    def on_external_command(self, command: str):
        cmd = command.strip().upper()

        # Accept:  CLASS:RECYCLABLE   |   BIN:3
        if cmd.startswith("CLASS:"):
            cls = cmd.split(":", 1)[1].strip()
            bin_number = CLASS_TO_BIN.get(cls)
            if bin_number is None:
                self.log(f"[IPC] Unknown class: {cls}")
                return
            self.log(f"[IPC] Predicted '{cls}' -> BIN {bin_number}")
            self._enqueue_bin(bin_number, reason=f"CLASS:{cls}")
            return

        if cmd.startswith("BIN:"):
            try:
                n = int(cmd.split(":", 1)[1].strip())
            except ValueError:
                self.log(f"[IPC] Bad BIN command: {command}")
                return
            if n not in self.bin_edits:
                self.log(f"[IPC] Unknown BIN: {n}")
                return
            self.log(f"[IPC] Direct -> BIN {n}")
            self._enqueue_bin(n, reason=f"BIN:{n}")
            return

        self.log(f"[IPC] Unrecognised command: {command}")

    def _enqueue_bin(self, bin_number: int, reason: str = ""):
        if self._is_ready():
            self._execute_bin(bin_number)
        else:
            self.pending_bins.append(bin_number)
            self.log(f"[IPC] Queued BIN {bin_number} ({reason}) — machine busy")

    def _execute_bin(self, bin_number: int):
        try:
            steps = int(self.bin_edits[bin_number].text().strip())
        except (ValueError, KeyError):
            self.error(f"BIN {bin_number}: invalid step value.")
            return
        self.config.bin_steps[bin_number] = steps
        self.config.save()
        self.log(f"[IPC] Executing BIN {bin_number} ({steps} steps)")
        self.controller.start_sequence(steps, label=f"BIN {bin_number}")

    def _drain_pending_bins(self):
        if not self.pending_bins:
            return
        if not self._is_ready():
            return
        bin_number = self.pending_bins.popleft()
        self._execute_bin(bin_number)

    # ========================================================
    #  LOGGING
    # ========================================================
    def log(self, text):
        self.log_box.append(f"[{time.strftime('%H:%M:%S')}] {text}")

    def error(self, text):
        self.log(f"ERROR: {text}")

    def closeEvent(self, event):
        try:
            self.command_server.stop()
        except Exception:
            pass
        try:
            self.config.save()
        except Exception:
            pass
        self.serial.close(silent=True)
        event.accept()


def main():
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[WARN] Could not create {RESULTS_DIR}: {e}")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())