"""
thinkfarm Provider Launcher (standalone)

GUI front-end that starts / stops the provider via solo.py (no FastAPI).
Configuration and status only — no log output.
"""

import asyncio
import configparser
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

                if auto_manage and limit_gb > 0:
                    # Run optimization
                    newly_pulled = asyncio.run(self.model_manager.optimize_portfolio(limit_gb))
                    
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
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thinkfarm.webp")
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

        self.info_label = QLabel("Provider v18")
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

        self.auto_manage_cb = QCheckBox("Manage models automatically")
        self.auto_manage_cb.setStyleSheet("color: #1c1c1e; font-size: 13px; border: none;")
        grid_layout.addWidget(self.auto_manage_cb, 2, 1)

        self.storage_label = QLabel("Managed Model Storage (GB)")
        self.storage_label.setStyleSheet("color: rgba(0, 0, 0, 0.4); font-size: 13px; border: none;")
        grid_layout.addWidget(self.storage_label, 3, 0)

        self.storage_entry = QLineEdit()
        self.storage_entry.setText("30")
        grid_layout.addWidget(self.storage_entry, 3, 1)

        config_card_layout.addLayout(grid_layout)
        config_card_layout.addStretch()

        # Save Button
        save_btn_layout = QHBoxLayout()
        save_btn_layout.addStretch()
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.setFixedSize(160, 40)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                color: white;
                border-radius: 0px;
                font-weight: 500;
                font-size: 14px;
                border: none;
            }
            QPushButton:hover { background-color: #548889; }
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

    def _handle_show_message(self, msg_type, title, message):
        if msg_type == "info":
            QMessageBox.information(self, title, message)
        elif msg_type == "warning":
            QMessageBox.warning(self, title, message)
        elif msg_type == "error":
            QMessageBox.critical(self, title, message)

    # ── start / stop ────────────────────────────────────────────
    def start_service(self):
        """Orchestrate startup: probe context if needed, then start solo.py."""
        self.start_btn.setEnabled(False)
        self._stop_event.clear()

        def _startup_logic():
            try:
                # 1. Check for unscanned models
                url = self.ollama_url_entry.text().strip() or "http://127.0.0.1:11434"
                print(f"[STARTUP] Checking for unscanned models at {url}...")
                
                all_models = asyncio.run(context_prober.get_ollama_models(url))
                existing_limits = context_prober.load_context_limits()
                
                unscanned = [m for m in all_models if m not in existing_limits]
                
                if unscanned:
                    print(f"[STARTUP] Found {len(unscanned)} unscanned model(s): {unscanned}")
                    self.signals.ui_state_signal.emit("probing")
                    
                    # Run probing
                    asyncio.run(context_prober.run_context_probing(
                        url,
                        unscanned,
                        existing_limits
                    ))
                    print("[STARTUP] Probing complete.")

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
                for _ in range(50):
                    if self._server_process and self._server_process.poll() is None:
                        self.signals.ui_state_signal.emit("started")
                        return
                    if self._stop_event.is_set():
                        return
                    time.sleep(0.1)

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
        
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thinkfarm.webp")
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
