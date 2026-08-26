import sys
import os

# Redirect standard streams if running without a console (e.g. PyInstaller GUI mode)
if sys.platform == "win32":
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    if sys.stdin is None:
        sys.stdin = open(os.devnull, "r", encoding="utf-8")

import threading
import asyncio
import logging
import subprocess
import socket
import time
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QCheckBox, QGroupBox,
    QFormLayout, QSplitter, QFrame, QSystemTrayIcon, QMenu, QScrollArea,
    QSlider
)
import requests
from PyQt6.QtCore import pyqtSignal, QObject, Qt, QTimer
from PyQt6.QtGui import QFont, QColor, QTextCursor, QIcon, QAction

import uvicorn
import uvicorn.loops.asyncio
import uvicorn.protocols.http.h11_impl
import uvicorn.protocols.http.auto
import uvicorn.protocols.websockets.auto
import uvicorn.lifespan.on
import uvicorn.logging
from config import ConfigManager
from client_server import create_client_app
from provider_client import ProviderClient

# Thread-safe log signal emitter
class LogEmitter(QObject):
    log_received = pyqtSignal(str, str) # message, level

class QtLogHandler(logging.Handler):
    def __init__(self, emitter):
        super().__init__()
        self.emitter = emitter

    def emit(self, record):
        msg = self.format(record)
        self.emitter.log_received.emit(msg, record.levelname)

class StatusIndicator(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self.set_status("stopped")

    def set_status(self, status):
        # status: stopped, connecting, connected, running, probing
        colors = {
            "stopped": "#8e8e93",     # Grey
            "connecting": "#f5a623",  # Yellow/Orange
            "connected": "#34c759",   # Green
            "running": "#548889",     # Teal/Accented
            "probing": "#af52de"      # Purple
        }
        color = colors.get(status, "#8e8e93")
        self.setStyleSheet(f"""
            background-color: {color};
            border-radius: 7px;
            border: none;
        """)

class ClientThread(threading.Thread):
    def __init__(self, app, host, port):
        super().__init__()
        self.app = app
        self.host = host
        self.port = port
        self.server = None
        self.loop = None

    def run(self):
        logger = logging.getLogger("thinkfarm.client")
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            config = uvicorn.Config(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info",
                loop="asyncio",
                http="h11",
                lifespan="on",
                log_config=None,
            )
            self.server = uvicorn.Server(config)
            self.loop.run_until_complete(self.server.serve())
        except Exception as e:
            logger.error(f"Client Server failed to start: {e}", exc_info=True)
        finally:
            if self.loop and not self.loop.is_closed():
                try:
                    self.loop.close()
                except Exception:
                    pass

    def stop(self):
        if self.server:
            self.server.should_exit = True

class ProviderThread(threading.Thread):
    def __init__(self, provider_client):
        super().__init__()
        self.provider_client = provider_client
        self.loop = None

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self.provider_client.run())
        except RuntimeError as e:
            if "Event loop stopped before Future completed" in str(e):
                pass
            else:
                raise
        finally:
            try:
                self.loop.close()
            except Exception:
                pass

    def stop(self):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.provider_client.soft_stop(), self.loop)

    def force_stop(self):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.provider_client.stop(), self.loop)
            self.loop.call_soon_threadsafe(self.loop.stop)


class ThinkfarmApp(QMainWindow):
    models_loaded = pyqtSignal(list)

    def _get_free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    def _kill_ollama_process(self):
        if hasattr(self, '_ollama_process') and self._ollama_process is not None:
            try:
                logging.getLogger("thinkfarm").info("Terminating child Ollama process...")
                if os.name == 'nt':
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(self._ollama_process.pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                else:
                    self._ollama_process.terminate()
                self._ollama_process.wait(timeout=5)
            except Exception as e:
                logging.getLogger("thinkfarm").error(f"Error terminating ollama process: {e}")
            self._ollama_process = None

    def restart_ollama_server(self):
        logging.getLogger("thinkfarm").info("Restarting/checking internal ollama server...")
        
        self._kill_ollama_process()
            
        if hasattr(self, '_ollama_log_file') and self._ollama_log_file is not None:
            try:
                self._ollama_log_file.close()
            except Exception:
                pass
            self._ollama_log_file = None
                
        models_path = self.config_manager.ollama_models_path.strip()

        port = self._get_free_port()
        self._ollama_url = f"http://127.0.0.1:{port}"
        logging.getLogger("thinkfarm").info(f"Starting child Ollama server on port {port}...")
        
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
        env["OLLAMA_DEBUG"] = "1"
        if models_path:
            env["OLLAMA_MODELS"] = models_path

        if os.name == 'nt':
            system_root = env.get("SystemRoot") or env.get("SYSTEMROOT") or "C:\\Windows"
            env["SystemRoot"] = system_root
            if "SystemDrive" not in env and "SYSTEMDRIVE" not in env:
                env["SystemDrive"] = "C:"
            
            path_key = next((k for k in env if k.upper() == "PATH"), "PATH")
            current_path = env.get(path_key, "")
            paths = current_path.split(os.pathsep) if current_path else []
            
            sys32 = os.path.join(system_root, "System32")
            if sys32 not in paths:
                paths.append(sys32)
                
            user_profile = env.get("USERPROFILE") or os.path.expanduser("~")
            ollama_default_path = os.path.join(user_profile, "AppData", "Local", "Programs", "Ollama")
            if ollama_default_path not in paths:
                paths.append(ollama_default_path)
                
            env[path_key] = os.pathsep.join(paths)

        creationflags = 0
        if os.name == 'nt':
            creationflags = subprocess.CREATE_NO_WINDOW
            
        log_dir = os.path.expanduser("~/.thinkfarm")
        os.makedirs(log_dir, exist_ok=True)
        try:
            self._ollama_log_file = open(os.path.join(log_dir, "ollama_internal.log"), "a", encoding="utf-8")
            stdout_target = self._ollama_log_file
            stderr_target = self._ollama_log_file
        except Exception as e:
            logging.getLogger("thinkfarm").warning(f"Could not open internal Ollama log file: {e}")
            stdout_target = subprocess.DEVNULL
            stderr_target = subprocess.DEVNULL

        try:
            self._ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                env=env,
                creationflags=creationflags,
                stdin=subprocess.DEVNULL,
                stdout=stdout_target,
                stderr=stderr_target
            )
            self.config_manager.local_ollama_url = self._ollama_url
            
            # Start background thread to await ready state and refresh models
            def wait_and_refresh():
                import httpx
                tags_url = f"{self._ollama_url.rstrip('/')}/api/tags"
                for _ in range(30):
                    try:
                        with httpx.Client(timeout=2.0) as client:
                            resp = client.get(tags_url)
                            if resp.status_code == 200:
                                logging.getLogger("thinkfarm").info("Child Ollama is ready! Refreshing models...")
                                QTimer.singleShot(0, self.refresh_models)
                                return
                    except Exception:
                        pass
                    time.sleep(1)
                logging.getLogger("thinkfarm").warning("Timed out waiting for child Ollama to become ready.")

            threading.Thread(target=wait_and_refresh, daemon=True).start()
        except Exception as e:
            logging.getLogger("thinkfarm").error(f"Failed to start child Ollama process: {e}")

    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.config_manager.managed_ollama = (os.name == 'nt')
        self._original_local_url = self.config_manager.local_ollama_url
        self._ollama_process = None
        self._ollama_log_file = None

        if self.config_manager.managed_ollama:
            self.restart_ollama_server()

        self.client_thread = None
        self.provider_thread = None
        
        self.log_emitter = LogEmitter()
        self.log_emitter.log_received.connect(self.append_log)
        
        # Setup logging
        handler = QtLogHandler(self.log_emitter)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        self.model_vars = {}
        self.models_loaded.connect(self._populate_models)
        
        self.init_ui()
        self.init_tray()
        
        QTimer.singleShot(100, self.refresh_models)

    def init_tray(self):
        # Create tray icon
        self.tray_icon = QSystemTrayIcon(self)
        
        # Load thinkfarm icon if available, otherwise use default
        icon_path = os.path.join(os.path.dirname(__file__), "thinkfarm.webp")
        if os.path.exists(icon_path):
            self.tray_icon.setIcon(QIcon(icon_path))
        else:
            self.tray_icon.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon))
            
        # Create tray menu
        tray_menu = QMenu()
        show_action = QAction("Show Dashboard", self)
        show_action.triggered.connect(self.show_normal)
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.force_exit)
        
        tray_menu.addAction(show_action)
        tray_menu.addSeparator()
        tray_menu.addAction(exit_action)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.tray_icon_activated)
        self.tray_icon.show()

    def show_normal(self):
        self.show()
        self.setWindowState(Qt.WindowState.WindowNoState)
        self.raise_()
        self.activateWindow()

    def tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self.isVisible():
                self.hide()
            else:
                self.show_normal()

    def changeEvent(self, event):
        if event.type() == event.Type.WindowStateChange:
            if self.isMinimized():
                # Hide window so it minimizes to system tray instead of the taskbar
                self.hide()
                event.ignore()
                return
        super().changeEvent(event)

    def force_exit(self):
        # Allow actual close on exit action
        self.tray_icon.hide()
        if self.client_thread and self.client_thread.is_alive():
            self.client_thread.stop()
        if self.provider_thread and self.provider_thread.is_alive():
            self.provider_thread.force_stop()
        self._kill_ollama_process()
        if hasattr(self, '_ollama_log_file') and self._ollama_log_file is not None:
            try:
                self._ollama_log_file.close()
            except Exception:
                pass
        QApplication.quit()

    def init_ui(self):
        self.setWindowTitle("thinkfarm v16")
        self.resize(1400, 750)
        
        # Stylesheet to match qclient theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
            QWidget {
                color: #1c1c1e;
                font-family: "Inter", "Ubuntu", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid rgba(0, 0, 0, 0.1);
                border-radius: 0px;
                margin-top: 12px;
                padding-top: 16px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                font-weight: bold;
                color: #548889;
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
            QPushButton {
                background-color: transparent;
                border: 1px solid rgba(0, 0, 0, 0.1);
                border-radius: 0px;
                padding: 8px 16px;
                color: #8e8e93;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(84, 136, 137, 0.08);
                color: #548889;
            }
            QPushButton#actionButton {
                background-color: #548889;
                color: white;
                border-radius: 0px;
                font-weight: bold;
                border: none;
            }
            QPushButton#actionButton:hover {
                background-color: #436d6e;
            }
            QPushButton#actionButton:disabled {
                background-color: #d2d2d7;
                color: #8e8e93;
            }
            QTextEdit {
                background-color: #f5f5f7;
                border: 1px solid rgba(0, 0, 0, 0.1);
                border-radius: 0px;
                font-family: "JetBrains Mono", "Fira Code", "Monospace", monospace;
                font-size: 12px;
                color: #1c1c1e;
            }
            QCheckBox {
                spacing: 8px;
                color: #1c1c1e;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)

        # Main splitter (Left config panel, Right logs panel)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self.main_splitter)

        # Left Widget (Config & Controls)
        left_widget = QWidget()
        left_widget.setMaximumWidth(600)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # 2. Client Settings
        self._client_collapsed = False
        client_group = QGroupBox("Client (Ollama Proxy)")
        client_layout = QVBoxLayout(client_group)

        status_row1 = QHBoxLayout()
        self.client_indicator = StatusIndicator()
        self.client_status_lbl = QLabel("Stopped")
        status_row1.addWidget(self.client_indicator)
        status_row1.addWidget(self.client_status_lbl)

        self._client_collapse_btn = QPushButton("▼ Collapse")
        self._client_collapse_btn.setMaximumSize(140, 28)
        self._client_collapse_btn.setStyleSheet("""
            QPushButton { background-color: transparent; border: none; font-size: 13px; color: #8e8e93; }
            QPushButton:hover { color: #548889; }
        """)
        self._client_collapse_btn.clicked.connect(self._toggle_client_group)
        status_row1.addWidget(self._client_collapse_btn)
        status_row1.addStretch()
        
        client_layout.addLayout(status_row1)

        # Content container (collapsible)
        self._client_content = QWidget()
        self._client_content_layout = QVBoxLayout(self._client_content)
        self._client_content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        client_form = QFormLayout()
        self.consumer_id_input = QLineEdit(self.config_manager.consumer_id)
        self.client_port_input = QLineEdit(str(self.config_manager.port))
        self.whitelist_enabled_cb = QCheckBox("Enable Model Whitelist")
        self.whitelist_enabled_cb.setChecked(self.config_manager.whitelist_enabled)

        client_form.addRow("Client ID:", self.consumer_id_input)
        client_form.addRow("Local Port:", self.client_port_input)
        client_form.addRow("", self.whitelist_enabled_cb)

        self._client_content_layout.addLayout(client_form)

        # Whitelist Section Header
        wl_header_layout = QHBoxLayout()
        self.wl_title = QLabel("Model Whitelist")
        self.wl_title.setStyleSheet("font-weight: bold; color: #548889;")
        wl_header_layout.addWidget(self.wl_title)
        wl_header_layout.addStretch()

        self.refresh_btn = QPushButton("Refresh Models")
        self.refresh_btn.setFixedSize(120, 28)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid rgba(0, 0, 0, 0.1);
                border-radius: 0px;
                font-size: 11px;
                font-weight: 500;
                color: #8e8e93;
            }
            QPushButton:hover {
                background-color: rgba(84, 136, 137, 0.08);
                color: #548889;
            }
        """)
        self.refresh_btn.clicked.connect(self.refresh_models)
        wl_header_layout.addWidget(self.refresh_btn)
        self._client_content_layout.addLayout(wl_header_layout)

        # Filter
        filter_layout = QHBoxLayout()
        self.filter_icon = QLabel("🔍")
        self.filter_icon.setStyleSheet("border: none; color: rgba(0, 0, 0, 0.4);")
        filter_layout.addWidget(self.filter_icon)

        self.filter_entry = QLineEdit()
        self.filter_entry.setPlaceholderText("Filter models...")
        self.filter_entry.setStyleSheet("""
            QLineEdit {
                background-color: #f5f5f7;
                border: 1px solid transparent;
                border-radius: 0px;
                height: 32px;
                padding: 6px 10px;
            }
        """)
        self.filter_entry.textChanged.connect(self.apply_model_filter)
        filter_layout.addWidget(self.filter_entry)
        self._client_content_layout.addLayout(filter_layout)

        # Select All/None
        toggle_layout = QHBoxLayout()
        toggle_layout.addStretch()

        self.select_none_btn = QPushButton("None")
        self.select_all_btn = QPushButton("All")
        for btn in [self.select_none_btn, self.select_all_btn]:
            btn.setFixedSize(60, 28)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: 1px solid rgba(0, 0, 0, 0.1);
                    border-radius: 0px;
                    font-size: 11px;
                    font-weight: 500;
                    color: #8e8e93;
                }
                QPushButton:hover {
                    background-color: rgba(84, 136, 137, 0.08);
                    color: #548889;
                }
            """)
        self.select_all_btn.clicked.connect(self.select_all_models)
        self.select_none_btn.clicked.connect(self.select_none_models)
        toggle_layout.addWidget(self.select_none_btn)
        toggle_layout.addWidget(self.select_all_btn)
        self._client_content_layout.addLayout(toggle_layout)

        # Scroll Area for Models
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedHeight(200)
        self.models_widget = QWidget()
        self.models_widget.setStyleSheet("background-color: #ececec;")
        self.models_layout = QVBoxLayout(self.models_widget)
        self.models_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.models_widget)
        self._client_content_layout.addWidget(self.scroll_area)

        client_save_btn = QPushButton("Save Configuration")
        client_save_btn.clicked.connect(self.save_config)
        self._client_content_layout.addWidget(client_save_btn)

        self.client_toggle_btn = QPushButton("Start Client Server")
        self.client_toggle_btn.setObjectName("actionButton")
        self.client_toggle_btn = QPushButton("Start Client Server")
        self.client_toggle_btn.setObjectName("actionButton")
        self.client_toggle_btn.clicked.connect(self.toggle_client)
        self._client_content_layout.addWidget(self.client_toggle_btn)

        # Now add content container to group
        client_layout.addWidget(self._client_content)

        left_layout.addWidget(client_group)

        # 3. Provider Settings
        self._provider_collapsed = False
        provider_group = QGroupBox("Provider (Inference Node)")
        provider_layout = QVBoxLayout(provider_group)

        status_row2 = QHBoxLayout()
        self.provider_indicator = StatusIndicator()
        self.provider_status_lbl = QLabel("Stopped")
        status_row2.addWidget(self.provider_indicator)
        status_row2.addWidget(self.provider_status_lbl)

        self._provider_collapse_btn = QPushButton("▼ Collapse")
        self._provider_collapse_btn.setMaximumSize(140, 28)
        self._provider_collapse_btn.setStyleSheet("""
            QPushButton { background-color: transparent; border: none; font-size: 13px; color: #8e8e93; }
            QPushButton:hover { color: #548889; }
        """)
        self._provider_collapse_btn.clicked.connect(self._toggle_provider_group)
        status_row2.addWidget(self._provider_collapse_btn)
        status_row2.addStretch()
        
        provider_layout.addLayout(status_row2)

        # Content container (collapsible)
        self._provider_content = QWidget()
        self._provider_content_layout = QVBoxLayout(self._provider_content)
        self._provider_content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.provider_form = QFormLayout()
        self.provider_id_input = QLineEdit(self.config_manager.provider_id)
        self.local_ollama_input = QLineEdit(self.config_manager.local_ollama_url)
        self.models_path_input = QLineEdit(self.config_manager.ollama_models_path)
        self.auto_manage_cb = QCheckBox("Auto Manage Models")
        self.auto_manage_cb.setChecked(self.config_manager.auto_manage_models)
        self.gb_allowed_input = QLineEdit(str(self.config_manager.gb_allowed))
        self.restart_cmd_input = QLineEdit(self.config_manager.ollama_restart_cmd)

        self.context_pressure_slider = QSlider(Qt.Orientation.Horizontal)
        self.context_pressure_slider.setRange(0, 100)
        self.context_pressure_slider.setValue(int(self.config_manager.context_pressure * 100))
        self.context_pressure_lbl = QLabel(f"{self.config_manager.context_pressure:.2f}")
        self.context_pressure_slider.valueChanged.connect(
            lambda val: self.context_pressure_lbl.setText(f"{val / 100.0:.2f}")
        )

        pressure_layout = QHBoxLayout()
        pressure_layout.addWidget(self.context_pressure_slider)
        pressure_layout.addWidget(self.context_pressure_lbl)

        self.provider_form.addRow("Provider ID:", self.provider_id_input)
        self.provider_form.addRow("Local Ollama URL:", self.local_ollama_input)
        self.provider_form.addRow("Model Storage Path:", self.models_path_input)
        self.provider_form.addRow("", self.auto_manage_cb)
        self.provider_form.addRow("GB Allowed:", self.gb_allowed_input)
        self.provider_form.addRow("Ollama Restart Command:", self.restart_cmd_input)
        self.provider_form.addRow("Context Pressure:", pressure_layout)

        self.toggle_managed_ollama_fields()

        provider_save_btn = QPushButton("Save Configuration")
        provider_save_btn.clicked.connect(self.save_config)
        self.provider_form.addRow("", provider_save_btn)

        self._provider_content_layout.addLayout(self.provider_form)
        
        self.provider_toggle_btn = QPushButton("Start Provider")
        self.provider_toggle_btn.setObjectName("actionButton")
        self.provider_toggle_btn.clicked.connect(self.toggle_provider)
        self._provider_content_layout.addWidget(self.provider_toggle_btn)

        # Now add content container to group
        provider_layout.addWidget(self._provider_content)

        left_layout.addWidget(provider_group)
        left_layout.addStretch()
        
        self.toggle_log_btn = QPushButton("Hide Activity Log")
        self.toggle_log_btn.clicked.connect(self.toggle_logs)
        left_layout.addWidget(self.toggle_log_btn)

        # Right Widget (Logs & Console)
        self.right_widget = QWidget()
        right_layout = QVBoxLayout(self.right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        log_label = QLabel("Activity Logs:")
        log_label.setStyleSheet("font-weight: bold; color: #548889;")
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        
        log_btns_layout = QHBoxLayout()
        clear_logs_btn = QPushButton("Clear Logs")
        clear_logs_btn.clicked.connect(self.log_area.clear)
        log_btns_layout.addWidget(clear_logs_btn)
        
        right_layout.addWidget(log_label)
        right_layout.addWidget(self.log_area)
        right_layout.addLayout(log_btns_layout)
 
        # Add widgets to splitter
        self.main_splitter.addWidget(left_widget)
        self.main_splitter.addWidget(self.right_widget)
        
        # Set proportion
        self.main_splitter.setSizes([580, 820])
 
    def save_config(self):
        is_managed = self.config_manager.managed_ollama
        old_models_path = self.config_manager.ollama_models_path

        self.config_manager.provider_id = self.provider_id_input.text()
        self.config_manager.consumer_id = self.consumer_id_input.text()
        self.config_manager.port = int(self.client_port_input.text())
        self.config_manager.whitelist_enabled = self.whitelist_enabled_cb.isChecked()
        
        selected = [name for name, cb in self.model_vars.items() if cb.isChecked()]
        # Retain saved whitelisted models that are not loaded/present in the UI list currently
        for m in self.config_manager.whitelist_models:
            if m not in self.model_vars and m not in selected:
                selected.append(m)
        self.config_manager.whitelist_models = selected
        
        self.config_manager.ollama_models_path = self.models_path_input.text()

        if is_managed:
            # If transitioning to managed mode, or models path has changed, restart internal server
            models_path_changed = old_models_path != self.models_path_input.text()
            if models_path_changed:
                self.restart_ollama_server()
        else:
            self.config_manager.local_ollama_url = self.local_ollama_input.text()
            
        self.config_manager.auto_manage_models = self.auto_manage_cb.isChecked()
        try:
            self.config_manager.gb_allowed = float(self.gb_allowed_input.text())
        except ValueError:
            self.config_manager.gb_allowed = 0.0
        self.config_manager.ollama_restart_cmd = self.restart_cmd_input.text()
        self.config_manager.context_pressure = self.context_pressure_slider.value() / 100.0
        
        self.config_manager.save()
        logging.getLogger("thinkfarm").info("Configuration saved successfully.")

    def toggle_managed_ollama_fields(self):
        is_managed = self.config_manager.managed_ollama
        self.local_ollama_input.setEnabled(not is_managed)
        self.restart_cmd_input.setEnabled(not is_managed)
        self.models_path_input.setEnabled(is_managed)
        
        self.local_ollama_input.setVisible(not is_managed)
        self.restart_cmd_input.setVisible(not is_managed)
        self.models_path_input.setVisible(is_managed)
        
        # Hide/show the corresponding labels in QFormLayout
        if hasattr(self, 'provider_form') and self.provider_form:
            label_url = self.provider_form.labelForField(self.local_ollama_input)
            if label_url:
                label_url.setVisible(not is_managed)
            label_cmd = self.provider_form.labelForField(self.restart_cmd_input)
            if label_cmd:
                label_cmd.setVisible(not is_managed)
            label_path = self.provider_form.labelForField(self.models_path_input)
            if label_path:
                label_path.setVisible(is_managed)

        if not is_managed:
            # If unchecked, restore the original local url if currently set to our managed dynamic URL
            current_url = self.local_ollama_input.text()
            if hasattr(self, '_ollama_url') and current_url == self._ollama_url:
                self.local_ollama_input.setText(self._original_local_url)
        else:
            # If checked, set to our dynamic URL
            if hasattr(self, '_ollama_url'):
                self.local_ollama_input.setText(self._ollama_url)

    def refresh_models(self):
        """Fetch available models from the central server in a background thread."""
        server_url = self.config_manager.server_url.strip().rstrip('/')
        consumer_id = self.config_manager.consumer_id.strip()
        headers = {}
        if consumer_id:
            headers["X-Consumer-ID"] = consumer_id
            
        def _fetch():
            try:
                response = requests.get(f"{server_url}/api/tags", headers=headers, timeout=10)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    self.models_loaded.emit(models)
            except Exception as e:
                logging.getLogger("thinkfarm").warning(f"Failed to fetch models from central server: {e}")
        
        threading.Thread(target=_fetch, daemon=True).start()

    def _populate_models(self, models):
        """Populate the whitelist list with models fetched from the server."""
        previous_model_vars = self.model_vars
        current_checked = {name for name, cb in previous_model_vars.items() if cb.isChecked()}

        # Clear existing checkboxes
        while self.models_layout.count():
            item = self.models_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.model_vars = {}
        filter_text = self.filter_entry.text().lower().strip()

        for m in models:
            name = m.get("name") if isinstance(m, dict) else m
            if name.lower().startswith("thinkfarm"):
                continue
            
            cb = QCheckBox(name)
            cb.setStyleSheet("margin: 5px; border: none; color: #1c1c1e; font-size: 13px;")
            self.model_vars[name] = cb
            
            if name in self.config_manager.whitelist_models:
                if name in previous_model_vars and name not in current_checked:
                    cb.setChecked(False)
                else:
                    cb.setChecked(True)
            elif name in current_checked:
                cb.setChecked(True)

            cb.setVisible(not filter_text or filter_text in name.lower())
            self.models_layout.addWidget(cb)

    def apply_model_filter(self):
        """Filter the visible models in the list based on the current filter text without changing their checked states."""
        filter_text = self.filter_entry.text().lower().strip()
        for name, cb in self.model_vars.items():
            cb.setVisible(not filter_text or filter_text in name.lower())

    def select_all_models(self):
        for cb in self.model_vars.values():
            cb.setChecked(True)

    def select_none_models(self):
        for cb in self.model_vars.values():
            cb.setChecked(False)

    def append_log(self, message, level):
        color_map = {
            "INFO": "#1c1c1e",
            "WARNING": "#b45309",
            "ERROR": "#b91c1c",
            "CRITICAL": "#b91c1c",
            "DEBUG": "#6b7280"
        }
        color = color_map.get(level, "#1c1c1e")
        self.log_area.moveCursor(QTextCursor.MoveOperation.End)
        self.log_area.insertHtml(f"<span style='color: {color};'>{message}</span><br>")
        # Prune excess lines from the top (>1000 lines ≈ 50K chars of HTML markup)
        if len(self.log_area.toHtml()) > 50_000:
            old_html = self.log_area.toHtml()
            pruned = old_html[-50_000:]
            self.log_area.setHtml(pruned)
        self.log_area.moveCursor(QTextCursor.MoveOperation.End)

    # Client server control
    def toggle_client(self):
        if self.client_thread and self.client_thread.is_alive():
            logging.getLogger("thinkfarm.client").info("Stopping Client Server...")
            self.client_thread.stop()
            self.client_thread.join(timeout=3)
            self.client_thread = None
            self.client_status_lbl.setText("Stopped")
            self.client_indicator.set_status("stopped")
            self.client_toggle_btn.setText("Start Client Server")
        else:
            self.save_config()
            if not self.config_manager.consumer_id or not self.config_manager.consumer_id.strip():
                logging.getLogger("thinkfarm.client").error("Client Server cannot be started: CONSUMER_ID is missing.")
                self.client_status_lbl.setText("Missing CONSUMER_ID")
                self.client_indicator.set_status("stopped")
                return
            logging.getLogger("thinkfarm.client").info("Starting Client Server...")
            app = create_client_app(self.config_manager)
            
            # Start in background thread
            self.client_thread = ClientThread(app, "0.0.0.0", self.config_manager.port)
            self.client_thread.daemon = True
            self.client_thread.start()
            
            self.client_status_lbl.setText("Running")
            self.client_indicator.set_status("connected")
            self.client_toggle_btn.setText("Stop Client Server")

    # Provider client control
    def toggle_provider(self):
        if self.provider_thread and self.provider_thread.is_alive():
            logging.getLogger("thinkfarm.provider").info("Stopping Provider Client...")
            self.provider_thread.stop()
            self.provider_toggle_btn.setText("Stopping...")
            self.provider_toggle_btn.setEnabled(False)
        else:
            self.save_config()
            
            def handle_provider_status(status):
                self.provider_status_lbl.setText(status)
                status_lower = status.lower()
                if "connected" in status_lower:
                    self.provider_indicator.set_status("connected")
                elif "connecting" in status_lower:
                    self.provider_indicator.set_status("connecting")
                elif "stopping" in status_lower:
                    self.provider_indicator.set_status("connecting")
                elif "probing" in status_lower:
                    self.provider_indicator.set_status("probing")
                else:
                    self.provider_indicator.set_status("stopped")
                    self.provider_toggle_btn.setText("Start Provider")
                    self.provider_toggle_btn.setEnabled(True)
                    self.provider_thread = None
                    
            p_client = ProviderClient(
                self.config_manager,
                status_callback=handle_provider_status,
                restart_callback=self.restart_ollama_server
            )
            
            self.provider_thread = ProviderThread(p_client)
            self.provider_thread.daemon = True
            self.provider_thread.start()
            
            self.provider_toggle_btn.setText("Stop Provider")

    def trigger_restart_ollama(self):
        if self.provider_thread and self.provider_thread.is_alive() and self.provider_thread.provider_client:
            p_client = self.provider_thread.provider_client
            loop = self.provider_thread.loop
            if loop:
                def done_callback(future):
                    try:
                        res = future.result()
                        logging.getLogger("thinkfarm.provider").info(f"Ollama restart completed with result: {res}")
                    except Exception as e:
                        logging.getLogger("thinkfarm.provider").error(f"Error during Ollama restart: {e}")
                
                future = asyncio.run_coroutine_threadsafe(p_client.restart_ollama(), loop)
                future.add_done_callback(done_callback)
            else:
                logging.getLogger("thinkfarm.provider").warning("Provider thread loop not ready.")
        else:
            # If provider is not running, run in a background thread with a new event loop
            logging.getLogger("thinkfarm.provider").info("Starting Ollama restart (provider not running)...")
            self.save_config()
            def run_offline_restart():
                loop = asyncio.new_event_loop()
                p_client = ProviderClient(
                    self.config_manager,
                    status_callback=self.provider_status_lbl.setText,
                    restart_callback=self.restart_ollama_server
                )
                try:
                    res = loop.run_until_complete(p_client.restart_ollama())
                    logging.getLogger("thinkfarm.provider").info(f"Ollama restart completed offline with result: {res}")
                except Exception as e:
                    logging.getLogger("thinkfarm.provider").error(f"Error during offline Ollama restart: {e}")
                finally:
                    loop.close()
            threading.Thread(target=run_offline_restart, daemon=True).start()

    def toggle_logs(self):
        is_visible = self.right_widget.isVisible()
        self.right_widget.setVisible(not is_visible)
        if is_visible:
            self.resize(580, self.height())
            self.toggle_log_btn.setText("Show Activity Log")
        else:
            self.resize(1400, self.height())
            self.main_splitter.setSizes([580, 820])
            self.toggle_log_btn.setText("Hide Activity Log")

    def _toggle_client_group(self):
        self._client_collapsed = not self._client_collapsed
        visible = not self._client_collapsed
        self._client_content.setVisible(visible)
        arrow = "\u25b6" if self._client_collapsed else "\u25bc"
        label = "Expand" if self._client_collapsed else "Collapse"
        self._client_collapse_btn.setText(f"{arrow} {label}")

    def _toggle_provider_group(self):
        self._provider_collapsed = not self._provider_collapsed
        visible = not self._provider_collapsed
        self._provider_content.setVisible(visible)
        arrow = "\u25b6" if self._provider_collapsed else "\u25bc"
        label = "Expand" if self._provider_collapsed else "Collapse"
        self._provider_collapse_btn.setText(f"{arrow} {label}")

    def closeEvent(self, event):
        # Minimize to tray instead of quitting if the tray icon is visible
        if self.tray_icon.isVisible():
            self.hide()
            event.ignore()
            return
            
        # Graceful shutdown of child threads
        if self.client_thread and self.client_thread.is_alive():
            self.client_thread.stop()
        if self.provider_thread and self.provider_thread.is_alive():
            self.provider_thread.force_stop()
        
        self._kill_ollama_process()
        if hasattr(self, '_ollama_log_file') and self._ollama_log_file is not None:
            try:
                self._ollama_log_file.close()
            except Exception:
                pass
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ThinkfarmApp()
    window.show()
    sys.exit(app.exec())
