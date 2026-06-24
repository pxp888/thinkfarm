import sys
import os
import threading
import asyncio
import logging
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QCheckBox, QGroupBox,
    QFormLayout, QSplitter, QFrame, QSystemTrayIcon, QMenu, QScrollArea
)
import requests
from PyQt6.QtCore import pyqtSignal, QObject, Qt, QTimer
from PyQt6.QtGui import QFont, QColor, QTextCursor, QIcon, QAction

import uvicorn
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
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info", loop="asyncio")
        self.server = uvicorn.Server(config)
        try:
            self.loop.run_until_complete(self.server.serve())
        finally:
            self.loop.close()

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
        self.loop.run_until_complete(self.provider_client.run())

    def stop(self):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.provider_client.soft_stop(), self.loop)

    def force_stop(self):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.provider_client.stop(), self.loop)
            self.loop.call_soon_threadsafe(self.loop.stop)


class ThinkfarmApp(QMainWindow):
    models_loaded = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
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
        QApplication.quit()

    def init_ui(self):
        self.setWindowTitle("thinkfarm")
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
        client_group = QGroupBox("Client (Ollama Proxy)")
        client_layout = QVBoxLayout(client_group)
        
        status_row1 = QHBoxLayout()
        self.client_indicator = StatusIndicator()
        self.client_status_lbl = QLabel("Stopped")
        status_row1.addWidget(self.client_indicator)
        status_row1.addWidget(self.client_status_lbl)
        status_row1.addStretch()
        
        client_layout.addLayout(status_row1)
        
        client_form = QFormLayout()
        self.consumer_id_input = QLineEdit(self.config_manager.consumer_id)
        self.client_port_input = QLineEdit(str(self.config_manager.port))
        self.whitelist_enabled_cb = QCheckBox("Enable Model Whitelist")
        self.whitelist_enabled_cb.setChecked(self.config_manager.whitelist_enabled)
        
        client_form.addRow("Client ID:", self.consumer_id_input)
        client_form.addRow("Local Port:", self.client_port_input)
        client_form.addRow("", self.whitelist_enabled_cb)
        
        client_save_btn = QPushButton("Save Configuration")
        client_save_btn.clicked.connect(self.save_config)
        client_form.addRow("", client_save_btn)
        
        client_layout.addLayout(client_form)

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
        client_layout.addLayout(wl_header_layout)

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
        client_layout.addLayout(filter_layout)

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
        client_layout.addLayout(toggle_layout)

        # Scroll Area for Models
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedHeight(200)
        self.models_widget = QWidget()
        self.models_widget.setStyleSheet("background-color: #ececec;")
        self.models_layout = QVBoxLayout(self.models_widget)
        self.models_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.models_widget)
        client_layout.addWidget(self.scroll_area)
        
        self.client_toggle_btn = QPushButton("Start Client Server")
        self.client_toggle_btn.setObjectName("actionButton")
        self.client_toggle_btn.clicked.connect(self.toggle_client)
        client_layout.addWidget(self.client_toggle_btn)
        
        left_layout.addWidget(client_group)

        # 3. Provider Settings
        provider_group = QGroupBox("Provider (Inference Node)")
        provider_layout = QVBoxLayout(provider_group)
        
        status_row2 = QHBoxLayout()
        self.provider_indicator = StatusIndicator()
        self.provider_status_lbl = QLabel("Stopped")
        status_row2.addWidget(self.provider_indicator)
        status_row2.addWidget(self.provider_status_lbl)
        status_row2.addStretch()
        
        provider_layout.addLayout(status_row2)
        
        provider_form = QFormLayout()
        self.provider_id_input = QLineEdit(self.config_manager.provider_id)
        self.local_ollama_input = QLineEdit(self.config_manager.local_ollama_url)
        self.auto_manage_cb = QCheckBox("Auto Manage Models")
        self.auto_manage_cb.setChecked(self.config_manager.auto_manage_models)
        self.gb_allowed_input = QLineEdit(str(self.config_manager.gb_allowed))
        self.restart_cmd_input = QLineEdit(self.config_manager.ollama_restart_cmd)
        
        provider_form.addRow("Provider ID:", self.provider_id_input)
        provider_form.addRow("Local Ollama URL:", self.local_ollama_input)
        provider_form.addRow("", self.auto_manage_cb)
        provider_form.addRow("GB Allowed:", self.gb_allowed_input)
        provider_form.addRow("Ollama Restart Command:", self.restart_cmd_input)
        
        provider_save_btn = QPushButton("Save Configuration")
        provider_save_btn.clicked.connect(self.save_config)
        provider_form.addRow("", provider_save_btn)
        
        provider_layout.addLayout(provider_form)
        
        self.provider_toggle_btn = QPushButton("Start Provider")
        self.provider_toggle_btn.setObjectName("actionButton")
        self.provider_toggle_btn.clicked.connect(self.toggle_provider)
        provider_layout.addWidget(self.provider_toggle_btn)
        
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
        
        self.config_manager.local_ollama_url = self.local_ollama_input.text()
        self.config_manager.auto_manage_models = self.auto_manage_cb.isChecked()
        try:
            self.config_manager.gb_allowed = float(self.gb_allowed_input.text())
        except ValueError:
            self.config_manager.gb_allowed = 0.0
        self.config_manager.ollama_restart_cmd = self.restart_cmd_input.text()
        
        self.config_manager.save()
        logging.getLogger("thinkfarm").info("Configuration saved successfully.")

    def refresh_models(self):
        """Fetch available models from the local Ollama server in a background thread."""
        server_url = self.local_ollama_input.text().strip().rstrip('/')
        def _fetch():
            try:
                response = requests.get(f"{server_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    self.models_loaded.emit(models)
            except Exception as e:
                logging.getLogger("thinkfarm").warning(f"Failed to fetch models from Ollama: {e}")
        
        threading.Thread(target=_fetch, daemon=True).start()

    def _populate_models(self, models):
        """Populate the whitelist list with models fetched from the server."""
        is_initial_load = not self.model_vars
        current_checked = {name for name, cb in self.model_vars.items() if cb.isChecked()}

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
            
            if is_initial_load:
                if name in self.config_manager.whitelist_models:
                    cb.setChecked(True)
            else:
                if name in current_checked:
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
                status_callback=handle_provider_status
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
                    status_callback=self.provider_status_lbl.setText
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
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ThinkfarmApp()
    window.show()
    sys.exit(app.exec())
