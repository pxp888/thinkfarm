import os
import sys
import multiprocessing

# ---------------------------------------------------------------------------
# PyInstaller & Windows Console Redirection
# ---------------------------------------------------------------------------
if getattr(sys, 'frozen', False) and sys.platform == 'win32':
    # When packaged with console=False, stdout and stderr are None.
    # We must redirect them to avoid crashes in uvicorn and other libraries.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w')
    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w')

# Required for PyInstaller + multiprocessing support
multiprocessing.freeze_support()

import configparser
import socket
import threading
import requests
from dotenv import load_dotenv

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QCheckBox, QScrollArea,
    QFrame, QGridLayout, QSpacerItem, QSizePolicy, QSystemTrayIcon, QMenu
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal as Signal, QObject
from PyQt6.QtGui import QFont, QColor, QPixmap, QIcon, QAction

# Add parent directory to path for imports if running in dev
if not getattr(sys, 'frozen', False):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from main import app as fastapi_app

class ModelSignals(QObject):
    models_loaded = Signal(list)

class QConsumerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("thinkfarm Client Control")
        self.resize(800, 900)
        
        self.server = None
        self.server_thread = None
        self._stop_flag = threading.Event()
        
        self.signals = ModelSignals()
        self.signals.models_loaded.connect(self._populate_models)
        
        self.model_vars = {}  # model_name -> QCheckBox
        self.server_url = "http://127.0.0.1:11434"
        
        self.load_server_url()
        self.setup_ui()
        
        # Load current data
        self.load_consumer_id()
        self.load_server_port()
        
        # Fetch models
        QTimer.singleShot(100, self.refresh_models)
        
        self.setup_tray()

    def load_server_url(self):
        """Load SERVER_URL from .env file."""
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        load_dotenv(env_path)
        self.server_url = os.environ.get("SERVER_URL", "https://app.thinkfarm.net")

    def load_port(self):
        """Load the port from ~/.thinkfarm/config.ini, defaulting to 11434."""
        try:
            config_dir = os.path.expanduser("~/.thinkfarm")
            config_path = os.path.join(config_dir, "config.ini")

            if not os.path.exists(config_path):
                return 11434

            config = configparser.ConfigParser()
            config.read(config_path)

            if config.has_section("consumer") and config.has_option("consumer", "port"):
                return int(config.get("consumer", "port"))
        except Exception:
            pass

        return 11434

    def setup_ui(self):
        # UI Stylesheet to mimic customtkinter theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f7;
            }
            QLabel {
                color: #1d1d1f;
                font-family: "Inter", "Ubuntu", "Segoe UI", sans-serif;
            }
            QLineEdit {
                background-color: #f5f5f7;
                border: 1px solid #d2d2d7;
                border-radius: 8px;
                padding: 8px;
                color: #1d1d1f;
                font-family: "JetBrains Mono", "Fira Code", "Monospace";
            }
            QCheckBox {
                color: #1d1d1f;
                font-family: "Inter", "Ubuntu", sans-serif;
                font-size: 13px;
            }
            QScrollArea {
                border: 1px solid #d2d2d7;
                border-radius: 8px;
                background-color: #f5f5f7;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar - Pure Black
        self.sidebar_frame = QFrame()
        self.sidebar_frame.setFixedWidth(240)
        self.sidebar_frame.setStyleSheet("background-color: #e6e6e6; border: none;")
        sidebar_layout = QVBoxLayout(self.sidebar_frame)
        sidebar_layout.setContentsMargins(25, 40, 25, 20)
        sidebar_layout.setSpacing(10)

        # Logo image at top of sidebar
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thinkfarm.webp")
        pixmap = QPixmap(logo_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                160, 160, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.logo_label = QLabel()
            self.logo_label.setPixmap(pixmap)
            self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            self.logo_label = QLabel("thinkfarm")
            self.logo_label.setStyleSheet("color: #e6e6e6; font-size: 24px; font-weight: 500; margin-bottom: 30px;")
            self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(self.logo_label)

        self.start_btn = QPushButton("Start Client")
        self.start_btn.setFixedHeight(45)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
                border: none;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:disabled {
                background-color: #333333;
                color: #666666;
            }
        """)
        self.start_btn.clicked.connect(self.start_service)
        sidebar_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Client")
        self.stop_btn.setFixedHeight(45)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: #666666;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
                border: none;
            }
            QPushButton:hover {
                background-color: #e74c3c;
                color: white;
            }
            QPushButton:disabled {
                background-color: #1a1a1a;
                color: #444444;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_service)
        sidebar_layout.addWidget(self.stop_btn)

        sidebar_layout.addStretch()

        self.info_label = QLabel("Client v8")
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
                border-radius: 16px;
                border: 1px solid #e5e5e7;
            }
        """)
        self.config_card.setObjectName("ConfigCard")
        config_card_layout = QVBoxLayout(self.config_card)
        config_card_layout.setContentsMargins(30, 30, 30, 30)
        config_card_layout.setSpacing(15)

        self.header_label = QLabel("Client Configuration")
        self.header_label.setStyleSheet("font-size: 22px; font-weight: 500; border: none; margin-bottom: 10px;")
        config_card_layout.addWidget(self.header_label)

        # ID and Port Grid
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)
        
        self.id_label = QLabel("Client Identifier")
        self.id_label.setStyleSheet("color: #86868b; font-size: 13px; border: none;")
        grid_layout.addWidget(self.id_label, 0, 0)
        
        self.consumer_id_entry = QLineEdit()
        self.consumer_id_entry.setPlaceholderText("Enter client ID...")
        grid_layout.addWidget(self.consumer_id_entry, 0, 1)

        self.consumer_id_status = QLabel("")
        self.consumer_id_status.setStyleSheet("font-size: 10px; color: #555555; border: none;")
        grid_layout.addWidget(self.consumer_id_status, 1, 1, Qt.AlignmentFlag.AlignRight)

        self.port_label = QLabel("Local Server Port")
        self.port_label.setStyleSheet("color: #86868b; font-size: 13px; border: none;")
        grid_layout.addWidget(self.port_label, 2, 0)
        
        self.server_port_entry = QLineEdit()
        self.server_port_entry.setPlaceholderText("11434")
        grid_layout.addWidget(self.server_port_entry, 2, 1)

        self.server_port_status = QLabel("")
        self.server_port_status.setStyleSheet("font-size: 10px; color: #555555; border: none;")
        grid_layout.addWidget(self.server_port_status, 3, 1, Qt.AlignmentFlag.AlignRight)

        config_card_layout.addLayout(grid_layout)

        # Whitelist Section Header
        wl_header_layout = QHBoxLayout()
        self.wl_title = QLabel("Model Whitelist")
        self.wl_title.setStyleSheet("font-size: 18px; font-weight: 500; border: none;")
        wl_header_layout.addWidget(self.wl_title)
        
        wl_header_layout.addStretch()
        
        self.refresh_btn = QPushButton("Refresh Models")
        self.refresh_btn.setFixedSize(120, 28)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f7;
                border: 1px solid #d2d2d7;
                border-radius: 6px;
                font-size: 11px;
                font-weight: bold;
                color: #1d1d1f;
            }
            QPushButton:hover { background-color: #e5e5e7; }
        """)
        self.refresh_btn.clicked.connect(self.refresh_models)
        wl_header_layout.addWidget(self.refresh_btn)
        config_card_layout.addLayout(wl_header_layout)

        self.whitelist_enabled_cb = QCheckBox("Enable Model Whitelisting")
        config_card_layout.addWidget(self.whitelist_enabled_cb)

        # Filter
        filter_layout = QHBoxLayout()
        self.filter_icon = QLabel("🔍")
        self.filter_icon.setStyleSheet("border: none; color: #86868b;")
        filter_layout.addWidget(self.filter_icon)
        
        self.filter_entry = QLineEdit()
        self.filter_entry.setPlaceholderText("Filter models...")
        self.filter_entry.setStyleSheet("""
            QLineEdit {
                background-color: #ffffff;
                border: 1px solid #d2d2d7;
                border-radius: 8px;
                height: 32px;
            }
        """)
        self.filter_entry.textChanged.connect(self.apply_model_filter)
        filter_layout.addWidget(self.filter_entry)
        config_card_layout.addLayout(filter_layout)

        # Select All/None
        toggle_layout = QHBoxLayout()
        toggle_layout.addStretch()
        
        self.select_none_btn = QPushButton("Select None")
        self.select_all_btn = QPushButton("Select All")
        for btn in [self.select_none_btn, self.select_all_btn]:
            btn.setFixedSize(110, 28)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #f5f5f7;
                    border: 1px solid #d2d2d7;
                    border-radius: 6px;
                    font-size: 11px;
                    font-weight: bold;
                    color: #1d1d1f;
                }
                QPushButton:hover { background-color: #e5e5e7; }
            """)
        self.select_all_btn.clicked.connect(self.select_all_models)
        self.select_none_btn.clicked.connect(self.select_none_models)
        toggle_layout.addWidget(self.select_none_btn)
        toggle_layout.addWidget(self.select_all_btn)
        config_card_layout.addLayout(toggle_layout)

        # Scroll Area for Models
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        # self.scroll_area.setFixedHeight(200)
        self.models_widget = QWidget()
        self.models_widget.setStyleSheet("background-color: #f5f5f7;")
        self.models_layout = QVBoxLayout(self.models_widget)
        self.models_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.models_widget)
        config_card_layout.addWidget(self.scroll_area, 1)
        config_card_layout.addStretch()

        # Save Settings Button
        save_btn_layout = QHBoxLayout()
        save_btn_layout.addStretch()
        self.save_all_btn = QPushButton("Save Settings")
        self.save_all_btn.setFixedSize(160, 40)
        self.save_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #8e8e93;
                color: white;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
                border: none;
            }
            QPushButton:hover { background-color: #636366; }
        """)
        self.save_all_btn.clicked.connect(self.save_all_settings)
        save_btn_layout.addWidget(self.save_all_btn)
        config_card_layout.addLayout(save_btn_layout)

        content_layout.addWidget(self.config_card, 1)
        content_layout.addStretch()

        # Status Bar
        self.status_bar_frame = QFrame()
        self.status_bar_frame.setFixedHeight(50)
        self.status_bar_frame.setStyleSheet("background-color: #ffffff; border: 1px solid #e5e5e7; border-radius: 8px;")
        status_bar_layout = QHBoxLayout(self.status_bar_frame)
        status_bar_layout.setContentsMargins(20, 0, 20, 0)
        
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #ff3b30; font-size: 22px; border: none;")
        status_bar_layout.addWidget(self.status_dot)

        self.status_text = QLabel("SYSTEM STOPPED")
        self.status_text.setStyleSheet("color: #86868b; font-size: 12px; font-weight: bold; border: none; margin-left: 5px;")
        status_bar_layout.addWidget(self.status_text)
        status_bar_layout.addStretch()

        content_layout.addWidget(self.status_bar_frame)

    def save_all_settings(self):
        """Helper to save ID, Port, and Whitelist at once."""
        self.save_consumer_id()
        self.save_server_port()
        self.save_whitelist()

    def refresh_models(self):
        """Fetch available models from the server in a background thread."""
        def _fetch():
            try:
                response = requests.get(f"{self.server_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    self.signals.models_loaded.emit(models)
            except Exception as e:
                pass
        
        threading.Thread(target=_fetch, daemon=True).start()

    def _populate_models(self, models):
        """Populate the whitelist list with models fetched from the server."""
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
            if filter_text and filter_text not in name.lower():
                continue
            
            cb = QCheckBox(name)
            cb.setStyleSheet("margin: 5px; border: none;")
            self.model_vars[name] = cb
            self.models_layout.addWidget(cb)
            cb.show()

        # After populating, load which ones are checked
        self.load_whitelist()


    def apply_model_filter(self):
        """Re-filter the model list based on the current filter text."""
        # In a real app we might not want to re-fetch, but consumer.py does.
        self.refresh_models()

    def select_all_models(self):
        for cb in self.model_vars.values():
            cb.setChecked(True)

    def select_none_models(self):
        for cb in self.model_vars.values():
            cb.setChecked(False)

    def load_whitelist(self):
        """Load whitelist settings from config.ini."""
        try:
            config_dir = os.path.expanduser("~/.thinkfarm")
            config_path = os.path.join(config_dir, "config.ini")
            if not os.path.exists(config_path): return

            config = configparser.ConfigParser()
            config.read(config_path)

            if config.has_section("consumer"):
                enabled = config.getboolean("consumer", "whitelist_enabled", fallback=False)
                self.whitelist_enabled_cb.setChecked(enabled)
                
                models_str = config.get("consumer", "whitelist_models", fallback="")
                selected_models = [m.strip() for m in models_str.split(",") if m.strip()]
                
                for m in selected_models:
                    if m in self.model_vars:
                        self.model_vars[m].setChecked(True)
        except Exception as e:
            print(f"Error loading whitelist: {e}")

    def save_whitelist(self):
        """Save whitelist settings to config.ini."""
        try:
            config_dir = os.path.expanduser("~/.thinkfarm")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, "config.ini")

            config = configparser.ConfigParser()
            config.read(config_path)

            if not config.has_section("consumer"):
                config.add_section("consumer")

            enabled = self.whitelist_enabled_cb.isChecked()
            config.set("consumer", "whitelist_enabled", str(enabled).lower())

            selected = [name for name, cb in self.model_vars.items() if cb.isChecked()]
            config.set("consumer", "whitelist_models", ",".join(selected))

            with open(config_path, "w") as f:
                config.write(f)
            
            self.server_port_status.setText("Settings saved!")
            self.server_port_status.setStyleSheet("color: #2ecc71; font-size: 10px; border: none;")
        except Exception as e:
            print(f"Error saving whitelist: {e}")

    def _check_port(self, port):
        """Check if the server's port is open via IPv4 and IPv6."""
        addrs = [
            (socket.AF_INET, "127.0.0.1"),
            (socket.AF_INET6, "::1"),
            (socket.AF_INET, "0.0.0.0"),
        ]
        for (family, host) in addrs:
            try:
                sock = socket.socket(family, socket.SOCK_STREAM)
                sock.settimeout(0.01)
                result = sock.connect_ex((host, port))
                sock.close()
                if result == 0:
                    return True
            except Exception:
                pass
        return False

    def start_service(self):
        """Start the consumer uvicorn server in a background thread."""
        port = self.load_port()
        self._starting_port = port
        self._stop_flag.clear()

        def run_server():
            try:
                config = uvicorn.Config(
                    app=fastapi_app,
                    host="0.0.0.0",
                    port=port,
                    log_level="info",
                    access_log=True,
                    lifespan="on",
                    use_colors=False,
                )
                self.server = uvicorn.Server(config)
                self.server.run()
            except Exception as e:
                print(f"Server error: {e}")

        try:
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()

            # Start polling for port readiness
            self.poll_timer = QTimer()
            self.poll_timer.timeout.connect(self._poll_server_ready)
            self.poll_count = 0
            self.poll_timer.start(100)

        except Exception as e:
            print(f"Failed to start thread: {e}")

    def _poll_server_ready(self):
        if self._check_port(self._starting_port):
            self.poll_timer.stop()
            self._on_server_ready()
            return
        
        self.poll_count += 1
        if self.poll_count > 300: # 30 seconds timeout
            self.poll_timer.stop()
            # Just let it be, similar to consumer.py

    def _on_server_ready(self):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff3b30;
                color: white;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
                border: none;
            }
            QPushButton:hover { background-color: #ff2d55; }
        """)
        self.status_text.setText("SYSTEM RUNNING")
        self.status_text.setStyleSheet("color: #34c759; font-size: 12px; font-weight: bold; border: none; margin-left: 5px;")
        self.status_dot.setStyleSheet("color: #34c759; font-size: 22px; border: none;")

    def stop_service(self):
        """Stop the consumer uvicorn server."""
        if self.server is not None:
            self._stop_flag.set()
            self.server.should_exit = True
            
            if self.server_thread and self.server_thread.is_alive():
                self.server_thread.join(timeout=2)
            
            self.server = None
        
        self._update_ui_stopped()

    def _update_ui_stopped(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: #666666;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
                border: none;
            }
            QPushButton:hover { background-color: #e74c3c; color: white; }
            QPushButton:disabled { background-color: #1a1a1a; color: #444444; }
        """)
        self.status_text.setText("SYSTEM STOPPED")
        self.status_text.setStyleSheet("color: #86868b; font-size: 12px; font-weight: bold; border: none; margin-left: 5px;")
        self.status_dot.setStyleSheet("color: #ff3b30; font-size: 22px; border: none;")

    def load_consumer_id(self):
        try:
            config_dir = os.path.expanduser("~/.thinkfarm")
            config_path = os.path.join(config_dir, "config.ini")
            
            if not os.path.exists(config_path):
                self.consumer_id_status.setText("Config not found")
                return

            config = configparser.ConfigParser()
            config.read(config_path)

            if config.has_section("consumer") and config.has_option("consumer", "consumer_id"):
                consumer_id = config.get("consumer", "consumer_id")
                self.consumer_id_entry.setText(consumer_id)
                self.consumer_id_status.setText(f"Loaded: {consumer_id}")
            else:
                self.consumer_id_status.setText("No client_id in config")
        except Exception as e:
            self.consumer_id_status.setText(f"Error loading: {e}")

    def save_consumer_id(self):
        new_id = self.consumer_id_entry.text().strip()
        if not new_id:
            self.consumer_id_status.setText("Error: ID cannot be empty")
            self.consumer_id_status.setStyleSheet("color: #e74c3c; font-size: 10px; border: none;")
            return

        try:
            config_dir = os.path.expanduser("~/.thinkfarm")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, "config.ini")

            config = configparser.ConfigParser()
            config.read(config_path)

            if not config.has_section("consumer"):
                config.add_section("consumer")

            config.set("consumer", "consumer_id", new_id)

            with open(config_path, "w") as f:
                config.write(f)

            self.consumer_id_status.setText("Client ID saved!")
            self.consumer_id_status.setStyleSheet("color: #2ecc71; font-size: 10px; border: none;")
        except Exception as e:
            self.consumer_id_status.setText(f"Error saving: {e}")

    def load_server_port(self):
        port = self.load_port()
        self.server_port_entry.setText(str(port))

    def save_server_port(self):
        new_port = self.server_port_entry.text().strip()
        if not new_port:
            self.server_port_status.setText("Port cannot be empty")
            self.server_port_status.setStyleSheet("color: #e74c3c; font-size: 10px; border: none;")
            return

        try:
            port_num = int(new_port)
            if not (1 <= port_num <= 65535):
                raise ValueError
        except ValueError:
            self.server_port_status.setText("Invalid port number")
            self.server_port_status.setStyleSheet("color: #e74c3c; font-size: 10px; border: none;")
            return

        try:
            config_dir = os.path.expanduser("~/.thinkfarm")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, "config.ini")

            config = configparser.ConfigParser()
            config.read(config_path)

            if not config.has_section("consumer"):
                config.add_section("consumer")

            config.set("consumer", "port", str(port_num))

            with open(config_path, "w") as f:
                config.write(f)

            self.server_port_status.setText("Port saved!")
            self.server_port_status.setStyleSheet("color: #2ecc71; font-size: 10px; border: none;")
        except Exception as e:
            self.server_port_status.setText(f"Error saving: {e}")

    def setup_tray(self):
        """Initialize the system tray icon and its context menu."""
        self.tray_icon = QSystemTrayIcon(self)
        
        # Load icon from the same file used for the logo
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thinkfarm.webp")
        if os.path.exists(logo_path):
            self.tray_icon.setIcon(QIcon(logo_path))
        else:
            # Fallback icon if webp is missing
            self.tray_icon.setIcon(self.style().standardIcon(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps))

        # Create tray menu
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
        self.stop_service()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Try to set Inter font if available
    font = QFont("Inter")
    if not font.exactMatch():
        font = QFont("Ubuntu")
    if not font.exactMatch():
        font = QFont("Segoe UI")
    app.setFont(font)
    
    window = QConsumerGUI()
    window.show()
    sys.exit(app.exec())
