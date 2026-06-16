"""
thinkfarm Provider Launcher (standalone)

GUI front-end that starts / stops the provider via solo.py (no FastAPI).
Configuration and status only — no log output.
"""

import asyncio
import configparser
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
import socket

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QCheckBox, QFrame, QGridLayout, QMessageBox,
    QSystemTrayIcon, QMenu
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal as Signal, QObject
from PyQt6.QtGui import QFont, QPixmap, QIcon, QAction

import solo
import context_prober
from dotenv import load_dotenv
from ollama_client import OllamaClient
from model_manager import ModelManager

# Load environment variables at startup
load_dotenv()

_CONFIG_PATH = os.path.expanduser("~/.thinkfarm/config.ini")
_PRIORITY_MODEL_PATH = os.path.expanduser("~/.thinkfarm/priority_model.txt")
_SLOPEMON_DATA_PATH = os.path.expanduser("~/.thinkfarm/_slopemon.json")
_TRIGGER_PATH = os.path.expanduser("~/.thinkfarm/solo_slope_trigger.json")
_ZERO_EVAL_FILE = os.path.expanduser("~/.thinkfarm/ZERO_EVAL")


def _write_priority_model(model: str) -> None:
    """Write the priority model to a file for solo.py to consume."""
    if not model:
        return
    config_dir = os.path.dirname(_CONFIG_PATH)
    os.makedirs(config_dir, exist_ok=True)
    with open(_PRIORITY_MODEL_PATH, "w") as f:
        f.write(model)
    print(f"[MGMT] Wrote priority model to {_PRIORITY_MODEL_PATH}: {model}")


# ---------------------------------------------------------------------------
# Blacklist
# ---------------------------------------------------------------------------
_BLACKLIST_FILE = os.path.expanduser("~/.thinkfarm/blacklisted_models.json")
_blacklisted_models: set = set()


def load_blacklist() -> None:
    """Load the blacklist from disk into ``_blacklisted_models``."""
    global _blacklisted_models
    try:
        if os.path.exists(_BLACKLIST_FILE):
            with open(_BLACKLIST_FILE, "r") as f:
                _blacklisted_models = set(json.load(f))
        else:
            _blacklisted_models = set()
        print(f"[BLACKLIST] Loaded {len(_blacklisted_models)} model(s) from {_BLACKLIST_FILE}.")
    except Exception as e:
        print(f"[BLACKLIST] Error loading blacklist: {e}")
        _blacklisted_models = set()


def save_blacklist() -> None:
    """Persist the current blacklist to disk."""
    os.makedirs(os.path.dirname(_BLACKLIST_FILE), exist_ok=True)
    try:
        with open(_BLACKLIST_FILE, "w") as f:
            json.dump(sorted(_blacklisted_models), f, indent=2)
        print(f"[BLACKLIST] Saved {len(_blacklisted_models)} model(s) to {_BLACKLIST_FILE}.")
    except Exception as e:
        print(f"[BLACKLIST] Error saving blacklist: {e}")


def add_to_blacklist(model_name: str) -> None:
    """Add *model_name* to the blacklist and persist it."""
    global _blacklisted_models
    _blacklisted_models.add(model_name)
    save_blacklist()
    print(f"[BLACKLIST] Added \"{model_name}\" to blacklist.")

# ---------------------------------------------------------------------------
# Performance-based auto-blacklist (check against global peak)
# ---------------------------------------------------------------------------
_PERF_THRESHOLD = 1 / 3   # blacklisted if baseline < 1/3 * peak
_PERF_MIN_SAMPLES = 30      # need > 30 samples before trusting the peak


def _blacklist_underperforming(model: str, baseline_slope: float, global_peak: float) -> bool:
    """Blacklist a model if its local baseline is < threshold * global peak.
    
    Returns True if the model was blacklisted, False otherwise."""
    threshold_value = _PERF_THRESHOLD * global_peak
    if baseline_slope < threshold_value:
        print(f"[BLACKLIST-PERF] {model}: baseline={baseline_slope:.1f} "
              f"tokens/s < {_PERF_THRESHOLD*100:.0f}% of global peak={global_peak:.1f} "
              f"=\u003e blacklisting.")
        add_to_blacklist(model)
        return True
    return False


def _check_slopotrigger() -> dict | None:
    try:
        if os.path.exists(_TRIGGER_PATH):
            with open(_TRIGGER_PATH, "r") as f:
                data = json.load(f)
            os.remove(_TRIGGER_PATH)
            return data
    except Exception as e:
        print(f"[SLOPE-MON] Error checking trigger file: {e}")
    return None

def _decide_action(triggered_model: str, trigger_throughput: float) -> str:
    try:
        baselines_data = context_prober.load_performance_baselines()
        all_baselines = baselines_data.get("baselines", {})
        local_entry = all_baselines.get(triggered_model, {})
        local_slope = local_entry.get("slope")
        
        if local_slope and trigger_throughput < 0.5 * local_slope:
            return "restart_ollama"
    except Exception as e:
        print(f"[SLOPE-MON] Error in _decide_action: {e}")
    return "blacklist_and_restart"


class ProviderSignals(QObject):
    ui_state_signal = Signal(str)  # "started", "stopped", "probing", "stopping"
    trigger_actual_start = Signal()
    show_message = Signal(str, str, str)  # msg_type, title, message


class ProviderGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("thinkfarm Provider Control")
        self.resize(800, 520)

        self.signals = ProviderSignals()
        self.signals.ui_state_signal.connect(self._handle_ui_state_change)
        self.signals.trigger_actual_start.connect(self._actual_start_service)
        self.signals.show_message.connect(self._handle_show_message)

        self._server_process = None
        self._stop_event = threading.Event()

        # Initialize managed model state
        self.ollama = OllamaClient()
        self.model_manager = ModelManager(self.ollama, os.environ.get("SERVER_URL", "https://app.thinkfarm.net"))
        self._mgmt_thread = threading.Thread(target=self._managed_model_loop, daemon=True)
        self._mgmt_thread.start()

        self.setup_ui()
        self._load_config()
        self.setup_tray()

    def _load_config(self):
        if not os.path.exists(_CONFIG_PATH):
            return
        config = configparser.ConfigParser()
        config.read(_CONFIG_PATH)
        if config.has_section("provider"):
            if config.has_option("provider", "provider_id"):
                self.provider_id_entry.setText(config.get("provider", "provider_id"))
            if config.has_option("provider", "ollama_url"):
                url = config.get("provider", "ollama_url")
                self.ollama_url_entry.setText(url)
                # Update Ollama client base URL dynamically
                if url:
                    self.ollama.base_url = url
                    import httpx
                    self.ollama.httpx_client = httpx.AsyncClient(base_url=url, timeout=30.0)
            if config.has_option("provider", "ollama_restart_cmd"):
                self.restart_cmd_entry.setText(config.get("provider", "ollama_restart_cmd"))
            if config.has_option("provider", "managed_storage_gb"):
                self.storage_entry.setText(config.get("provider", "managed_storage_gb"))
            if config.has_option("provider", "auto_manage"):
                val = config.getboolean("provider", "auto_manage", fallback=False)
                self.auto_manage_cb.setChecked(val)

    def _managed_model_loop(self):
        """Background loop to handle automated model management."""
        while True:
            try:
                # Load latest config for storage limit and auto-manage flag
                config = configparser.ConfigParser()
                config.read(_CONFIG_PATH)
                
                auto_manage = False
                limit_gb = 0
                if config.has_section("provider"):
                    auto_manage = config.getboolean("provider", "auto_manage", fallback=False)
                    limit_gb = float(config.get("provider", "managed_storage_gb", fallback="30"))

                # Fetch and write slopemon state
                try:
                    import httpx
                    from datetime import datetime
                    with httpx.Client(timeout=15.0) as client:
                        resp = client.get("https://www.thinkfarm.net/api/performance")
                        resp.raise_for_status()
                        perf_data = resp.json()

                    state = {"updated_at": datetime.now().isoformat(), "models": {}}
                    for entry in perf_data.get("data", []):
                        name = entry.get("model", "")
                        peak = entry.get("peak", 0)
                        n_samples = entry.get("n", 0)
                        if n_samples > _PERF_MIN_SAMPLES and peak > 0:
                            state["models"][name] = {"peak": round(peak, 1)}

                    with open(_SLOPEMON_DATA_PATH, "w") as f:
                        json.dump(state, f, indent=2)
                except Exception as e:
                    print(f"[MGMT-LOOP] Error fetching /performance for slopemon: {e}")

                if auto_manage and limit_gb > 0:
                    # Run optimization
                    newly_pulled, priority_model = asyncio.run(self.model_manager.optimize_portfolio(limit_gb))
                    
                    _write_priority_model(priority_model)
                    
                    if newly_pulled:
                        print(f"[MGMT-LOOP] New models pulled: {newly_pulled}. Restarting service for probing...")
                        
                        # Programmatically stop service
                        was_running = self._server_process is not None
                        if was_running:
                            self.stop_service()
                            # Wait for it to stop
                            while self._server_process is not None:
                                time.sleep(1)

                        # Run probing for new models
                        print(f"[MGMT-LOOP] Probing context for: {newly_pulled}")
                        existing_limits = context_prober.load_context_limits()
                        asyncio.run(context_prober.run_context_probing(
                            self.ollama.base_url,
                            newly_pulled,
                            existing_limits
                        ))

                        # Restart service if it was running
                        if was_running:
                            self.start_service()
                
            except Exception as e:
                print(f"[MGMT-LOOP] Error in managed model loop: {e}")
            
            # Wait 1 hour before next check
            time.sleep(3600)

    # ── UI ────────────────────────────────────────────────────
    def setup_ui(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
            QLabel {
                color: #1c1c1e;
                font-family: "Inter", "Ubuntu", "Segoe UI", sans-serif;
            }
            QLineEdit {
                background-color: #f5f5f7;
                border: 1px solid transparent;
                border-radius: 0px;
                padding: 8px;
                color: #1c1c1e;
                font-family: "JetBrains Mono", "Fira Code", "Monospace";
            }
            QLineEdit:focus {
                background-color: #ffffff;
                border-color: #548889;
                border: 1px solid #548889;
            }
            QCheckBox {
                color: #1c1c1e;
                font-family: "Inter", "Ubuntu", sans-serif;
                font-size: 13px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar - Light Grey
        self.sidebar_frame = QFrame()
        self.sidebar_frame.setFixedWidth(240)
        self.sidebar_frame.setStyleSheet("background-color: #f7f7f8; border: none;")
        sidebar_layout = QVBoxLayout(self.sidebar_frame)
        sidebar_layout.setContentsMargins(25, 40, 25, 20)
        sidebar_layout.setSpacing(10)

        # Logo image at top of sidebar
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "desktop.webp")
        pixmap = QPixmap(logo_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                190, 190, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.logo_label = QLabel()
            self.logo_label.setPixmap(pixmap)
            self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            self.logo_label = QLabel("thinkfarm")
            self.logo_label.setStyleSheet("color: #000000; font-size: 24px; font-weight: 500; margin-bottom: 30px;")
            self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(self.logo_label)

        self.start_btn = QPushButton("Start Provider")
        self.start_btn.setFixedHeight(45)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #548889;
                color: white;
                border-radius: 0px;
                font-weight: 500;
                font-size: 14px;
                border: none;
            }
            QPushButton:disabled {
                background-color: #d2d2d7;
                color: #8e8e93;
            }
        """)
        self.start_btn.clicked.connect(self.start_service)
        sidebar_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Provider")
        self.stop_btn.setFixedHeight(45)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #6b7280;
                color: white;
                border-radius: 0px;
                font-weight: 500;
                font-size: 14px;
                border: none;
            }
            QPushButton:hover {
                background-color: #4b5563;
            }
            QPushButton:disabled {
                background-color: #d2d2d7;
                color: #8e8e93;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_service)
        sidebar_layout.addWidget(self.stop_btn)

        sidebar_layout.addStretch()

        self.info_label = QLabel("Provider v28")
        self.info_label.setStyleSheet("color: #666666; font-size: 11px;")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(self.info_label)

        main_layout.addWidget(self.sidebar_frame)

        # Main Content
        self.content_container = QWidget()
        content_layout = QVBoxLayout(self.content_container)
        content_layout.setContentsMargins(40, 40, 40, 40)
        content_layout.setSpacing(20)
        main_layout.addWidget(self.content_container)

        # Config Card
        self.config_card = QFrame()
        self.config_card.setStyleSheet("""
            QFrame#ConfigCard {
                background-color: #ffffff;
                border-radius: 0px;
                border: 1px solid rgba(0, 0, 0, 0.1);
            }
        """)
        self.config_card.setObjectName("ConfigCard")
        config_card_layout = QVBoxLayout(self.config_card)
        config_card_layout.setContentsMargins(30, 30, 30, 30)
        config_card_layout.setSpacing(15)

        # Config Grid
        grid_layout = QGridLayout()
        grid_layout.setSpacing(15)

        self.provider_id_label = QLabel("Provider Identifier")
        self.provider_id_label.setStyleSheet("color: rgba(0, 0, 0, 0.4); font-size: 13px; border: none;")
        grid_layout.addWidget(self.provider_id_label, 0, 0)

        self.provider_id_entry = QLineEdit()
        self.provider_id_entry.setPlaceholderText("Enter unique ID...")
        grid_layout.addWidget(self.provider_id_entry, 0, 1)

        self.ollama_url_label = QLabel("Ollama URL")
        self.ollama_url_label.setStyleSheet("color: rgba(0, 0, 0, 0.4); font-size: 13px; border: none;")
        grid_layout.addWidget(self.ollama_url_label, 1, 0)

        self.ollama_url_entry = QLineEdit()
        self.ollama_url_entry.setPlaceholderText("http://127.0.0.1:11434")
        grid_layout.addWidget(self.ollama_url_entry, 1, 1)

        self.restart_cmd_label = QLabel("Ollama Restart Command")
        self.restart_cmd_label.setStyleSheet("color: rgba(0, 0, 0, 0.4); font-size: 13px; border: none;")
        grid_layout.addWidget(self.restart_cmd_label, 2, 0)

        self.restart_cmd_entry = QLineEdit()
        self.restart_cmd_entry.setPlaceholderText("podman restart ollama")
        grid_layout.addWidget(self.restart_cmd_entry, 2, 1)

        self.auto_manage_cb = QCheckBox("Manage models automatically")
        self.auto_manage_cb.setStyleSheet("color: #1c1c1e; font-size: 13px; border: none;")
        grid_layout.addWidget(self.auto_manage_cb, 3, 1)

        self.storage_label = QLabel("Managed Model Storage (GB)")
        self.storage_label.setStyleSheet("color: rgba(0, 0, 0, 0.4); font-size: 13px; border: none;")
        grid_layout.addWidget(self.storage_label, 4, 0)

        self.storage_entry = QLineEdit()
        self.storage_entry.setText("30")
        grid_layout.addWidget(self.storage_entry, 4, 1)

        config_card_layout.addLayout(grid_layout)
        config_card_layout.addStretch()

        # Save Button
        save_btn_layout = QHBoxLayout()
        save_btn_layout.addStretch()
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.setFixedSize(160, 40)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #548889;
                color: white;
                border-radius: 0px;
                font-weight: 500;
                font-size: 14px;
                border: none;
            }
        """)
        self.save_btn.clicked.connect(self.save_config)
        save_btn_layout.addWidget(self.save_btn)
        config_card_layout.addLayout(save_btn_layout)

        content_layout.addWidget(self.config_card, 1)

        # Status Bar
        self.status_bar_frame = QFrame()
        self.status_bar_frame.setFixedHeight(50)
        self.status_bar_frame.setStyleSheet("background-color: #ffffff; border-top: 1px solid rgba(0, 0, 0, 0.1); border-radius: 0px;")
        status_bar_layout = QHBoxLayout(self.status_bar_frame)
        status_bar_layout.setContentsMargins(20, 0, 20, 0)

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #8e8e93; font-size: 22px; border: none;")
        status_bar_layout.addWidget(self.status_dot)

        self.status_text = QLabel("SYSTEM STOPPED")
        self.status_text.setStyleSheet("color: #8e8e93; font-size: 12px; font-weight: 600; border: none; margin-left: 5px;")
        status_bar_layout.addWidget(self.status_text)
        status_bar_layout.addStretch()

        content_layout.addWidget(self.status_bar_frame)

    # ── Thread-Safe Signal Handlers ─────────────────────────────
    def _handle_ui_state_change(self, state):
        if state == "started":
            self.start_btn.setEnabled(False)
            self.start_btn.setStyleSheet("QPushButton { background-color: #d2d2d7; color: #8e8e93; border: none; font-weight: 500; font-size: 14px; }")
            self.stop_btn.setEnabled(True)
            self.stop_btn.setStyleSheet("QPushButton { background-color: #6b7280; color: white; border: none; font-weight: 500; font-size: 14px; } QPushButton:hover { background-color: #4b5563; }")
            self.status_text.setText("SYSTEM RUNNING")
            self.status_text.setStyleSheet("color: #34c759; font-size: 12px; font-weight: 600; border: none; margin-left: 5px;")
            self.status_dot.setStyleSheet("color: #34c759; font-size: 22px; border: none;")
        elif state == "probing":
            self.status_text.setText("SYSTEM PROBING")
            self.status_text.setStyleSheet("color: #3498db; font-size: 12px; font-weight: 600; border: none; margin-left: 5px;")
            self.status_dot.setStyleSheet("color: #3498db; font-size: 22px; border: none;")
        elif state == "stopping":
            self.stop_btn.setEnabled(False)
            self.stop_btn.setStyleSheet("QPushButton { background-color: #d2d2d7; color: #8e8e93; border: none; font-weight: 500; font-size: 14px; }")
            self.status_text.setText("FINISHING JOBS")
            self.status_text.setStyleSheet("color: #f39c12; font-size: 12px; font-weight: 600; border: none; margin-left: 5px;")
            self.status_dot.setStyleSheet("color: #f39c12; font-size: 22px; border: none;")
        elif state == "stopped":
            self.start_btn.setEnabled(True)
            self.start_btn.setStyleSheet("QPushButton { background-color: #548889; color: white; border: none; font-weight: 500; font-size: 14px; }")
            self.stop_btn.setEnabled(False)
            self.stop_btn.setStyleSheet("QPushButton { background-color: #d2d2d7; color: #8e8e93; border: none; font-weight: 500; font-size: 14px; }")
            self.status_text.setText("SYSTEM STOPPED")
            self.status_text.setStyleSheet("color: #8e8e93; font-size: 12px; font-weight: 600; border: none; margin-left: 5px;")
            self.status_dot.setStyleSheet("color: #8e8e93; font-size: 22px; border: none;")
        elif state == "stopped_error":
            self.start_btn.setEnabled(True)
            self.start_btn.setStyleSheet("QPushButton { background-color: #548889; color: white; border: none; font-weight: 500; font-size: 14px; }")
            self.stop_btn.setEnabled(False)
            self.stop_btn.setStyleSheet("QPushButton { background-color: #d2d2d7; color: #8e8e93; border: none; font-weight: 500; font-size: 14px; }")
            self.status_text.setText("STOPPED - ERROR")
            self.status_text.setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: 600; border: none; margin-left: 5px;")
            self.status_dot.setStyleSheet("color: #e74c3c; font-size: 22px; border: none;")

    def _handle_show_message(self, msg_type, title, message):
        if msg_type == "info":
            QMessageBox.information(self, title, message)
        elif msg_type == "warning":
            QMessageBox.warning(self, title, message)
        elif msg_type == "error":
            QMessageBox.critical(self, title, message)

    def _restart_ollama(self):
        """Restart Ollama via configured command and poll /api/tags until it's back."""
        cmd_raw = self.restart_cmd_entry.text().strip()
        if not cmd_raw:
            QMessageBox.critical(
                self, "Performance Fault",
                "Ollama restart command is not configured. "
                "Please set the Ollama Restart Command in Settings and restart the provider.",
            )
            self.signals.ui_state_signal.emit("stopped")
            return False

        # Split into list of arguments for subprocess
        import shlex
        try:
            cmd = shlex.split(cmd_raw)
        except ValueError as e:
            QMessageBox.warning(
                self, "Recovery Error",
                f"Invalid Ollama restart command syntax: {e}\nCommand was: {cmd_raw}",
            )
            return False

        print(f"[RECOVERY] Restarting Ollama via: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"[RECOVERY] Error restarting Ollama: {e}")
            return False
        
        return self._wait_for_ollama_ready()

    def _wait_for_ollama_ready(self):
        """Poll Ollama's /api/tags endpoint until it responds or times out."""
        import httpx
        print("[RECOVERY] Polling /api/tags for Ollama recovery...")
        url = self.ollama_url_entry.text().strip() or "http://127.0.0.1:11434"
        tags_url = f"{url.rstrip('/')}/api/tags"
        
        for _ in range(30):
            try:
                with httpx.Client(timeout=2.0) as client:
                    resp = client.get(tags_url)
                    if resp.status_code == 200:
                        print("[RECOVERY] Ollama is back up!")
                        return True
            except Exception:
                pass
            time.sleep(2)
            
        print("[RECOVERY] Timed out waiting for Ollama to come back.")
        return False

    # ── start / stop ────────────────────────────────────────────
    def start_service(self):
        """Orchestrate startup: probe context if needed, then start solo.py."""
        self.start_btn.setEnabled(False)
        self._stop_event.clear()

        def _startup_logic():
            try:
                # 0. Performance-based auto-blacklist check
                print("[STARTUP] Fetching global performance data...")
                perf_data = {"data": []}
                try:
                    with _httpx_sync.Client(timeout=15.0) as client:
                        resp = client.get("https://www.thinkfarm.net/api/performance")
                        resp.raise_for_status()
                        perf_data = resp.json()
                    print(f"[STARTUP] Global performance data loaded for "
                          f"{len(perf_data.get('data', []))} model(s).")
                except Exception as e:
                    print(f"[STARTUP] Could not fetch performance data ({e}). "
                          "Skipping auto-blacklist check.")

                # Compare each known model against the global peak
                baselines_data = context_prober.load_performance_baselines()
                all_baselines = baselines_data.get("baselines", {})

                blacklisted_any = False
                for entry in perf_data.get("data", []):
                    model_name = entry.get("model", "")
                    if not model_name:
                        continue
                    global_peak = entry.get("peak", 0)
                    n_samples = entry.get("n", 0)

                    local_entry = all_baselines.get(model_name, {})
                    local_slope = local_entry.get("slope")
                    if local_slope is None:
                        continue

                    if n_samples > _PERF_MIN_SAMPLES and \
                       local_slope < _PERF_THRESHOLD * global_peak:
                        print(f"[BLACKLIST-PERF] {model_name}: "
                              f"baseline={local_slope:.1f} tokens/s "
                              f"< {_PERF_THRESHOLD*100:.0f}% of global peak={global_peak:.1f} "
                              f"(n={n_samples}) =\u003e blacklisting.")
                        add_to_blacklist(model_name)
                        blacklisted_any = True

                if blacklisted_any:
                    print(f"[BLACKLIST-PERF] Models auto-blacklisted by performance. "
                          f"{len(_blacklisted_models)} total now:")
                    for m in sorted(_blacklisted_models):
                        print(f"  - {m}")
                else:
                    print("[STARTUP] No models auto-blacklisted by performance.")

                # 1. Check for unscanned models
                url = self.ollama_url_entry.text().strip() or "http://127.0.0.1:11434"
                print(f"[STARTUP] Checking for unscanned models at {url}...")
                
                all_models = asyncio.run(context_prober.get_ollama_models(url))
                existing_limits = context_prober.load_context_limits()
                
                unscanned = [m for m in all_models if m not in existing_limits]
                
                if unscanned:
                    print(f"[STARTUP] Found {len(unscanned)} unscanned model(s): {unscanned}")

                # Run probing & custom model verification on all models;
                # _probe_new_models returns the updated limits (existing+new)
                updated_limits = asyncio.run(context_prober.run_context_probing(
                    url,
                    all_models,
                    existing_limits
                ))

                # Emit "probing" whenever any model was actually probed,
                # whether for new context limits or baseline-only setup.
                newly_probed = set(updated_limits.keys()) - set(existing_limits.keys())
                if unscanned or newly_probed:
                    self.signals.ui_state_signal.emit("probing")

                print("[STARTUP] Probing and custom model verification complete.")

                # 2. Start solo.py via main thread
                self.signals.trigger_actual_start.emit()

            except Exception as e:
                print(f"[STARTUP] Error during startup orchestration: {e}")
                self.signals.ui_state_signal.emit("stopped")

        threading.Thread(target=_startup_logic, daemon=True).start()

    def _actual_start_service(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "solo.py")
        project_root = os.path.dirname(script_dir)

        try:
            if getattr(sys, 'frozen', False):
                # PyInstaller bundle: run the same executable with a flag
                self._server_process = subprocess.Popen(
                    [sys.executable, "--solo"],
                    cwd=project_root,
                )
            else:
                # Standard Python: run the solo.py script
                self._server_process = subprocess.Popen(
                    [sys.executable, "-u", script_path],
                    cwd=project_root,
                )

            def check_startup():
                started = False
                for _ in range(50):
                    if self._server_process and self._server_process.poll() is None:
                        self.signals.ui_state_signal.emit("started")
                        started = True
                        break
                    if self._stop_event.is_set():
                        return
                    time.sleep(0.1)
                    
                if not started:
                    self.signals.ui_state_signal.emit("stopped")
                    return
                    
                while self._server_process and self._server_process.poll() is None:
                    if self._stop_event.is_set():
                        break
                    try:
                        if os.path.exists(_ZERO_EVAL_FILE):
                            os.remove(_ZERO_EVAL_FILE)
                            print("[MGMT] Zero eval detected. Stopping solo.py.")
                            self.signals.ui_state_signal.emit("stopped_error")
                            if self._server_process:
                                self._server_process.terminate()
                                self._server_process = None
                            break
                    except Exception as e:
                        print(f"[MGMT] Error checking zero eval file: {e}")
                    time.sleep(1)

                if self._server_process:
                    self._server_process.wait()

                if self._stop_event.is_set():
                    return  # Handled by stop_service
                    
                code = self._server_process.returncode
                self._server_process = None
                
                if code == 42:
                    trigger_data = _check_slopotrigger()
                    if trigger_data:
                        model = trigger_data.get("model", "unknown")
                        throughput = trigger_data.get("throughput", 0.0)
                        action = _decide_action(model, throughput)
                        
                        self.signals.show_message.emit("warning", "Performance Alert",
                            f"Model '{model}' underperformed vs global peak.\nActioned: {action}")
                            
                        if action == "restart_ollama":
                            self.signals.ui_state_signal.emit("probing")
                            if self._restart_ollama():
                                print("[SLOPE-MON] Recovery successful. Restarting solo...")
                                self.signals.trigger_actual_start.emit()
                            else:
                                self.signals.ui_state_signal.emit("stopped")
                        else:
                            add_to_blacklist(model)
                            print(f"[SLOPE-MON] Model {model} blacklisted. Restarting solo...")
                            self.signals.trigger_actual_start.emit()
                else:
                    self.signals.ui_state_signal.emit("stopped")

            threading.Thread(target=check_startup, daemon=True).start()
        except Exception as e:
            print(f"Error starting provider: {e}")
            self.signals.ui_state_signal.emit("stopped")

    def stop_service(self):
        if self._server_process is not None:
            self._stop_event.set()
            self._server_process.terminate()
            self.signals.ui_state_signal.emit("stopping")

            def wait_join():
                try:
                    # Give it up to 180 seconds to finish active jobs
                    self._server_process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    print("Soft stop timed out, forcing termination...")
                    self._server_process.kill()
                    self._server_process.wait()
                
                self._server_process = None
                self.signals.ui_state_signal.emit("stopped")

            threading.Thread(target=wait_join, daemon=True).start()
        else:
            self.signals.ui_state_signal.emit("stopped")

    # ── config ──────────────────────────────────────────────────
    def save_config(self):
        new_id = self.provider_id_entry.text().strip()
        new_url = self.ollama_url_entry.text().strip()
        new_storage = self.storage_entry.text().strip()
        new_auto = "yes" if self.auto_manage_cb.isChecked() else "no"

        errors = []
        if not new_id:
            errors.append("Provider ID cannot be empty")
        if new_url and not new_url.startswith(("http://", "https://")):
            errors.append("Ollama URL must start with http:// or https://")
        
        try:
            float(new_storage or "30")
        except ValueError:
            errors.append("Managed Model Storage must be a number")

        if errors:
            QMessageBox.warning(self, "Configuration Error", "\n".join(errors))
            return

        try:
            config_dir = os.path.expanduser("~/.thinkfarm")
            os.makedirs(config_dir, exist_ok=True)
            config = configparser.ConfigParser()
            config.read(_CONFIG_PATH)
            if not config.has_section("provider"):
                config.add_section("provider")
            config.set("provider", "provider_id", new_id)
            config.set("provider", "ollama_url", new_url)
            config.set("provider", "ollama_restart_cmd", self.restart_cmd_entry.text().strip())
            config.set("provider", "managed_storage_gb", new_storage or "30")
            config.set("provider", "auto_manage", new_auto)
            with open(_CONFIG_PATH, "w") as f:
                config.write(f)
            
            # Update Ollama client base URL dynamically
            if new_url:
                self.ollama.base_url = new_url
                import httpx
                self.ollama.httpx_client = httpx.AsyncClient(base_url=new_url, timeout=30.0)

            QMessageBox.information(self, "Success", "Settings saved successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {e}")

    # ── system tray ─────────────────────────────────────────────
    def setup_tray(self):
        """Initialize the system tray icon and its context menu."""
        self.tray_icon = QSystemTrayIcon(self)
        
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "desktop.webp")
        if os.path.exists(logo_path):
            self.tray_icon.setIcon(QIcon(logo_path))
        else:
            self.tray_icon.setIcon(self.style().standardIcon(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps))

        self.tray_menu = QMenu()
        
        restore_action = QAction("Restore", self)
        restore_action.triggered.connect(self.restore_window)
        self.tray_menu.addAction(restore_action)
        
        self.tray_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        self.tray_menu.addAction(exit_action)
        
        self.tray_icon.setContextMenu(self.tray_menu)
        self.tray_icon.activated.connect(self._on_tray_icon_activated)
        self.tray_icon.show()

    def _on_tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.restore_window()

    def restore_window(self):
        self.show()
        self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized | Qt.WindowState.WindowActive)
        self.activateWindow()

    def changeEvent(self, event):
        if event.type() == event.Type.WindowStateChange:
            if self.isMinimized():
                if QSystemTrayIcon.isSystemTrayAvailable():
                    # Hide to tray
                    QTimer.singleShot(0, self.hide)
        super().changeEvent(event)

    def closeEvent(self, event):
        if self._server_process is not None:
            self.stop_service()
        event.accept()


if __name__ == "__main__":
    multiprocessing.freeze_support()

    # PyInstaller dispatcher
    if len(sys.argv) > 1:
        if sys.argv[1] == "--solo":
            asyncio.run(solo.main())
            sys.exit(0)
        elif sys.argv[1] == "--probe":
            context_prober.main(sys.argv[2:])
            sys.exit(0)

    app = QApplication(sys.argv)
    
    font = QFont("Inter")
    if not font.exactMatch():
        font = QFont("Ubuntu")
    if not font.exactMatch():
        font = QFont("Segoe UI")
    app.setFont(font)

    window = ProviderGUI()
    window.show()
    sys.exit(app.exec())
