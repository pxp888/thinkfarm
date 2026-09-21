#!/usr/bin/env python3
"""PyQt6 dashboard for the self-contained thinkfarm provider app.

Same look & feel as 2llamashare/unity/app_gui.py (config form left, activity
log right, status dots, tray icon — close minimizes to tray). The heavy lifting
is exactly what thinkfarm.sh / app.py does: this module reuses app.LlamaServer and
lcpp's ProviderClient in a worker thread. Nothing here changes provider behavior.

Usage: ./thinkfarm.sh [--model NAME]   (GUI is the default; or: python3 gui.py)
"""

import asyncio
import logging
import socket
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "lcpp"))  # reuse lcpp's provider implementation

# Importing app runs its module setup (paths) but not main(); we reuse
# LlamaServer + pick_port so child-process behavior stays identical to headless.
import app  # noqa: E402
from config import ConfigManager            # noqa: E402
from provider_client import ProviderClient  # noqa: E402
from downloader import (
    download_artifacts,
    format_bytes,
    format_eta,
    ProgressInfo,
    DownloadCancelled,
)  # noqa: E402

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QGroupBox,
    QFormLayout, QSplitter, QFrame, QSystemTrayIcon, QMenu, QComboBox,
    QProgressBar, QDialog,
)
from PyQt6.QtCore import pyqtSignal, QObject, Qt, QTimer
from PyQt6.QtGui import QIcon, QAction, QTextCursor

try:
    import httpx
except ImportError:
    httpx = None

try:
    import requests
except ImportError:
    requests = None

PROVIDER_VERSION = 25


class AppSignals(QObject):
    """Thread-safe bridge from the worker thread to the GUI."""
    log_received = pyqtSignal(str, str)   # message, level
    server_state = pyqtSignal(str)        # llama-server child state (text)
    provider_status = pyqtSignal(str)     # ProviderClient status text
    finished = pyqtSignal()               # worker exited
    download_progress = pyqtSignal(object) # ProgressInfo
    download_finished = pyqtSignal(str)   # model label
    download_error = pyqtSignal(str)      # error message
    download_cancelled = pyqtSignal()     # cancelled / paused


class StatusIndicator(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self.set_status("stopped")

    def set_status(self, status):
        colors = {
            "stopped": "#8e8e93",     # Grey
            "connecting": "#f5a623",  # Yellow/Orange (starting / loading model)
            "connected": "#34c759",   # Green
            "running": "#548889",     # Teal/Accented
            "probing": "#af52de",     # Purple
            "error": "#b91c1c",       # Red
        }
        color = colors.get(status, "#8e8e93")
        self.setStyleSheet(f"""
            background-color: {color};
            border-radius: 7px;
            border: none;
        """)


class ProviderWorker(threading.Thread):
    """Runs the exact headless flow (spawn llama-server → wait ready → provider)
    in its own event loop, reporting progress via Qt signals."""

    def __init__(self, signals, config):
        super().__init__()
        self.daemon = True
        self.signals = signals
        self.config = config
        self.loop = None
        self._shutdown = None

    def _log(self, msg, level="INFO"):
        self.signals.log_received.emit(msg, level)

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        shutdown = asyncio.Event()
        self._shutdown = shutdown
        try:
            self.loop.run_until_complete(self._run(shutdown))
        except RuntimeError as e:
            if "Event loop stopped before Future completed" in str(e):
                pass  # force_stop() interrupted the loop — expected on exit
            else:
                raise
        except Exception as e:
            self._log(f"[app] ERROR: {e}", "ERROR")
        finally:
            try:
                if not self.loop.is_closed():
                    self.loop.close()
            except Exception:
                pass
            self.signals.finished.emit()

    async def _run(self, shutdown):
        cfg = self.config
        model_name = cfg.selected_model if cfg.selected_model in app.MODELS else app.DEFAULT_MODEL
        app.apply_publish_info(cfg, model_name)  # published name/digest from the model spec
        port = app.pick_port()
        runner = app.LlamaServer(port, model_name)
        cfg.local_lcpp_url = f"http://{app.HOST}:{port}"  # point provider at the child we own

        self.signals.server_state.emit("Starting…")
        self._log(f"[app] Starting llama-server on {app.HOST}:{port}")
        runner.start()
        self._log(f"[app] llama-server pid {runner.proc.pid} — loading model (first load can take a while)...")

        try:
            await runner.wait_ready(shutdown)
        except app.ShutdownRequested:
            self._log("[app] Shutdown requested during model load.")
            runner.stop()
            return
        except Exception as e:
            self._log(f"[app] ERROR: {e}", "ERROR")
            self.signals.server_state.emit("Failed")
            runner.stop()
            return

        self.signals.server_state.emit("Ready")
        self._log("[app] llama-server ready — starting provider client...")

        p_client = ProviderClient(
            cfg,
            status_callback=lambda s: (self.signals.provider_status.emit(s), self._log(f"[status] {s}")),
            log_callback=lambda msg: self._log(msg),
        )
        task = asyncio.get_running_loop().create_task(p_client.run())

        try:
            await shutdown.wait()
            self._log("[app] Shutting down provider...")
            await p_client.stop()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        finally:
            runner.stop()

    def stop(self):
        """Graceful: set the shutdown event (works during model load too)."""
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._shutdown.set)

    def force_stop(self):
        self.stop()
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)


class DownloadWorker(threading.Thread):
    """Downloads required model artifacts in a background thread."""

    def __init__(self, signals, model_name: str):
        super().__init__()
        self.daemon = True
        self.signals = signals
        self.model_name = model_name
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        m = app.MODELS[self.model_name]
        try:
            def on_progress(p: ProgressInfo):
                self.signals.download_progress.emit(p)

            download_artifacts(
                list(m.downloads),
                m.directory,
                cancel_event=self.cancel_event,
                progress_callback=on_progress,
            )
            self.signals.download_finished.emit(m.label)
        except DownloadCancelled:
            self.signals.download_cancelled.emit()
        except Exception as e:
            self.signals.download_error.emit(str(e))


class ThinkfarmProviderApp(QMainWindow):
    VERSION_CHECK_URL = "https://thinkfarm.eu/api/version"

    def __init__(self, model_override: str | None = None):
        super().__init__()
        self.config_manager = ConfigManager(str(ROOT))
        self.model_override = model_override  # --model NAME from argv (GUI mode)
        self.worker = None

        self.signals = AppSignals()
        self.signals.log_received.connect(self.append_log)
        self.signals.server_state.connect(self._on_server_state)
        self.signals.provider_status.connect(self._on_provider_status)
        self.signals.finished.connect(self._on_worker_finished)
        self.signals.download_progress.connect(self._on_download_progress)
        self.signals.download_finished.connect(self._on_download_finished)
        self.signals.download_error.connect(self._on_download_error)
        self.signals.download_cancelled.connect(self._on_download_cancelled)
        self.download_worker = None

        # Route stdlib logging (provider logs etc.) into the activity pane.
        handler = QtLogHandler(self.signals.log_received.emit)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

        self.init_ui()
        self.init_tray()
        self._check_bundle()
        QTimer.singleShot(500, self._check_version)

    # ------------------------------------------------------------------ tray
    def init_tray(self):
        self.tray_icon = QSystemTrayIcon(self)
        icon_path = ROOT / "lcpp" / "thinkfarm.webp"
        if icon_path.exists():
            self.tray_icon.setIcon(QIcon(str(icon_path)))
        else:
            self.tray_icon.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon))

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
        if event.type() == event.Type.WindowStateChange and self.isMinimized():
            # Minimize to tray instead of the taskbar.
            self.hide()
            event.ignore()
            return
        super().changeEvent(event)

    def force_exit(self):
        self.tray_icon.hide()
        if self.download_worker and self.download_worker.is_alive():
            self.download_worker.cancel()
        self._stop_worker(join=True, timeout=25)
        QApplication.quit()

    # -------------------------------------------------------------------- ui
    def init_ui(self):
        self.setWindowTitle("thinkfarm provider v" + str(PROVIDER_VERSION))
        self.resize(1100, 680)

        # Stylesheet matching the unity app theme.
        self.setStyleSheet("""
            QMainWindow { background-color: #ffffff; }
            QWidget { color: #1c1c1e; font-family: "Inter", "Ubuntu", "Segoe UI", sans-serif; font-size: 13px; }
            QGroupBox {
                border: 1px solid rgba(0, 0, 0, 0.1);
                border-radius: 0px; margin-top: 12px; padding-top: 16px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin; subcontrol-position: top left;
                left: 10px; padding: 0 5px; font-weight: bold; color: #548889;
            }
            QLineEdit {
                background-color: #f5f5f7; border: 1px solid transparent;
                border-radius: 0px; padding: 8px; color: #1c1c1e;
                font-family: "JetBrains Mono", "Fira Code", "Monospace";
            }
            QLineEdit:focus { background-color: #ffffff; border-color: #548889; border: 1px solid #548889; }
            QPushButton {
                background-color: transparent; border: 1px solid rgba(0, 0, 0, 0.1);
                border-radius: 0px; padding: 8px 16px; color: #8e8e93; font-weight: bold;
            }
            QPushButton:hover { background-color: rgba(84, 136, 137, 0.08); color: #548889; }
            QPushButton#actionButton { background-color: #548889; color: white; border-radius: 0px; font-weight: bold; border: none; }
            QPushButton#actionButton:hover { background-color: #436d6e; }
            QPushButton#actionButton:disabled { background-color: #d2d2d7; color: #8e8e93; }
            QTextEdit {
                background-color: #f5f5f7; border: 1px solid rgba(0, 0, 0, 0.1);
                border-radius: 0px; font-family: "JetBrains Mono", "Fira Code", "Monospace", monospace;
                font-size: 12px; color: #1c1c1e;
            }
        """)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self.main_splitter)

        # ---- Left: config & controls -------------------------------------
        left_widget = QWidget()
        left_widget.setMaximumWidth(600)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)

        provider_group = QGroupBox("Provider (Inference Node)")
        provider_layout = QVBoxLayout(provider_group)

        # Status rows: the app owns two things — the child llama-server and the
        # provider connection to the central server.
        status_row1 = QHBoxLayout()
        self.server_indicator = StatusIndicator()
        self.server_status_lbl = QLabel("llama-server: Stopped")
        status_row1.addWidget(self.server_indicator)
        status_row1.addWidget(self.server_status_lbl)
        status_row1.addStretch()

        status_row2 = QHBoxLayout()
        self.provider_indicator = StatusIndicator()
        self.provider_status_lbl = QLabel("provider: Stopped")
        status_row2.addWidget(self.provider_indicator)
        status_row2.addWidget(self.provider_status_lbl)
        status_row2.addStretch()

        provider_layout.addLayout(status_row1)
        provider_layout.addLayout(status_row2)

        form = QFormLayout()
        self.provider_id_input = QLineEdit(self.config_manager.provider_id)

        # Which bundled model to load (one at a time; applied on start).
        self.model_combo = QComboBox()
        for key, spec in app.MODELS.items():
            self.model_combo.addItem(spec.label, key)
        saved = self.model_override or self.config_manager.selected_model or ""
        idx = self.model_combo.findData(saved if saved in app.MODELS else app.DEFAULT_MODEL)
        self.model_combo.setCurrentIndex(idx)

        self.model_status_lbl = QLabel("")
        self.model_status_lbl.setStyleSheet("font-size: 11px;")
        self.model_combo.currentIndexChanged.connect(self._on_model_selection_changed)

        form.addRow("Provider ID:", self.provider_id_input)
        form.addRow("Model:", self.model_combo)
        form.addRow("", self.model_status_lbl)

        provider_layout.addLayout(form)

        self.save_btn = QPushButton("Save Configuration")
        self.save_btn.clicked.connect(self.save_config)
        provider_layout.addWidget(self.save_btn)

        # Download widget & progress bar
        self.download_widget = QWidget()
        dl_layout = QVBoxLayout(self.download_widget)
        dl_layout.setContentsMargins(0, 4, 0, 4)

        self.download_btn = QPushButton("Download Model")
        self.download_btn.setObjectName("actionButton")
        self.download_btn.clicked.connect(self.start_download)
        dl_layout.addWidget(self.download_btn)

        self.download_progress_bar = QProgressBar()
        self.download_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid rgba(0, 0, 0, 0.1);
                border-radius: 0px;
                text-align: center;
                font-size: 11px;
                height: 18px;
                background-color: #f5f5f7;
            }
            QProgressBar::chunk {
                background-color: #548889;
            }
        """)
        dl_layout.addWidget(self.download_progress_bar)

        dl_sub_row = QHBoxLayout()
        self.download_info_lbl = QLabel("")
        self.download_info_lbl.setStyleSheet("font-size: 11px; color: #1c1c1e;")
        self.cancel_download_btn = QPushButton("Cancel")
        self.cancel_download_btn.setFixedWidth(80)
        self.cancel_download_btn.clicked.connect(self.cancel_download)
        dl_sub_row.addWidget(self.download_info_lbl)
        dl_sub_row.addStretch()
        dl_sub_row.addWidget(self.cancel_download_btn)
        dl_layout.addLayout(dl_sub_row)

        provider_layout.addWidget(self.download_widget)

        self.toggle_btn = QPushButton("Start Provider")
        self.toggle_btn.setObjectName("actionButton")
        self.toggle_btn.clicked.connect(self.toggle_provider)
        provider_layout.addWidget(self.toggle_btn)

        log_path_lbl = QLabel(f"llama-server log: {app.SERVER_LOG}")
        log_path_lbl.setStyleSheet("color: #8e8e93; font-size: 11px;")
        log_path_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        provider_layout.addWidget(log_path_lbl)

        self._form_fields = [self.provider_id_input, self.model_combo, self.save_btn]

        left_layout.addWidget(provider_group)
        left_layout.addStretch()

        # Bottom-left: toggle the activity log pane.
        self.logs_hidden = False
        self.hide_logs_btn = QPushButton("Hide Logs")
        self.hide_logs_btn.clicked.connect(self.toggle_log_visibility)
        left_layout.addWidget(self.hide_logs_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        self.main_splitter.addWidget(left_widget)

        # ---- Right: activity log ------------------------------------------
        self.log_panel = QWidget()
        right_layout = QVBoxLayout(self.log_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)

        log_label = QLabel("Activity Logs:")
        log_label.setStyleSheet("font-weight: bold; color: #548889;")
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)

        clear_btns = QHBoxLayout()
        clear_logs_btn = QPushButton("Clear Logs")
        clear_logs_btn.clicked.connect(self.log_area.clear)
        clear_btns.addWidget(clear_logs_btn)

        right_layout.addWidget(log_label)
        right_layout.addWidget(self.log_area)
        right_layout.addLayout(clear_btns)
        self.main_splitter.addWidget(self.log_panel)

        self.main_splitter.setSizes([480, 620])

        # Tail the child's server log into the activity pane.
        self._tail_offset = None
        self._tail_timer = QTimer(self)
        self._tail_timer.setInterval(2000)
        self._tail_timer.timeout.connect(self._poll_server_log)
        self._update_model_status()

    def toggle_log_visibility(self):
        self.logs_hidden = not self.logs_hidden
        self.log_panel.setVisible(not self.logs_hidden)
        self.hide_logs_btn.setText("Show Logs" if self.logs_hidden else "Hide Logs")

    # ----------------------------------------------------------- model / downloads
    def _on_model_selection_changed(self, idx):
        self._update_model_status()

    def _update_model_status(self):
        name = self.model_combo.currentData() or app.DEFAULT_MODEL
        status = app.model_download_status(name)
        if self.download_worker and self.download_worker.is_alive():
            return  # worker manages UI while actively downloading

        if status["installed"]:
            size_fmt = format_bytes(status["total_bytes"])
            self.model_status_lbl.setText(f"✓ Weights installed ({size_fmt})")
            self.model_status_lbl.setStyleSheet("color: #2e7d32; font-size: 11px;")
            self.download_widget.hide()
            if not (self.worker and self.worker.is_alive()):
                self.toggle_btn.setEnabled(True)
                self.toggle_btn.setToolTip("")
        else:
            if status["downloaded_bytes"] > 0:
                cur_fmt = format_bytes(status["downloaded_bytes"])
                tot_fmt = format_bytes(status["total_bytes"])
                rem_fmt = format_bytes(status["remaining_bytes"])
                self.model_status_lbl.setText(f"⏸ Download paused ({cur_fmt} / {tot_fmt})")
                self.download_btn.setText(f"Resume Download ({rem_fmt})")
            else:
                tot_fmt = format_bytes(status["total_bytes"])
                self.model_status_lbl.setText(f"⚠ Weights missing ({tot_fmt} required)")
                self.download_btn.setText(f"Download Model ({tot_fmt})")

            self.model_status_lbl.setStyleSheet("color: #b45309; font-size: 11px;")
            self.download_widget.show()
            self.download_btn.show()
            self.download_progress_bar.hide()
            self.download_info_lbl.hide()
            self.cancel_download_btn.hide()
            if not (self.worker and self.worker.is_alive()):
                self.toggle_btn.setEnabled(False)
                self.toggle_btn.setToolTip("Please download the model weights before starting.")

    def start_download(self):
        if self.download_worker and self.download_worker.is_alive():
            return
        name = self.model_combo.currentData() or app.DEFAULT_MODEL
        m = app.MODELS[name]
        self.download_btn.hide()
        self.download_progress_bar.show()
        self.download_progress_bar.setValue(0)
        self.download_info_lbl.show()
        self.download_info_lbl.setText("Starting download...")
        self.cancel_download_btn.show()
        self.cancel_download_btn.setEnabled(True)
        self.cancel_download_btn.setText("Cancel")
        self._set_form_enabled(False)
        self.toggle_btn.setEnabled(False)

        self.append_log(f"[download] Starting download for {m.label}...", "INFO")
        self.download_worker = DownloadWorker(self.signals, name)
        self.download_worker.start()

    def cancel_download(self):
        if self.download_worker and self.download_worker.is_alive():
            self.cancel_download_btn.setEnabled(False)
            self.cancel_download_btn.setText("Cancelling…")
            self.download_worker.cancel()

    def _on_download_progress(self, p: ProgressInfo):
        self.download_progress_bar.setValue(int(p.overall_percent))
        msg = (
            f"[{p.file_index}/{p.total_files}] {p.filename}: {p.file_percent:.1f}% "
            f"({format_bytes(p.file_bytes_done)}/{format_bytes(p.file_bytes_total)}) • "
            f"{format_bytes(p.speed_bps)}/s • ETA {format_eta(p.eta_seconds)}"
        )
        self.download_info_lbl.setText(msg)

    def _on_download_finished(self, label: str):
        self.download_worker = None
        self._set_form_enabled(True)
        self.append_log(f"[download] Successfully downloaded all files for {label}!", "INFO")
        self._update_model_status()

    def _on_download_cancelled(self):
        self.download_worker = None
        self._set_form_enabled(True)
        self.append_log("[download] Download paused. You can resume at any time.", "WARNING")
        self._update_model_status()

    def _on_download_error(self, err: str):
        self.download_worker = None
        self._set_form_enabled(True)
        self.append_log(f"[download] Download error: {err}", "ERROR")
        self._update_model_status()

    def _required_files(self) -> list[Path]:
        """Files needed to start, for the currently selected model."""
        name = self.model_combo.currentData() or app.DEFAULT_MODEL
        return app.required_binaries() + app.model_files(name)

    def _check_bundle(self):
        for p in self._required_files():
            if not p.exists():
                logging.getLogger("thinkfarm").error(f"required file missing: {p}")

    # ------------------------------------------------------------- config
    def save_config(self):
        cfg = self.config_manager
        text = self.provider_id_input.text().strip()
        if text:
            cfg.provider_id = text
        cfg.selected_model = self.model_combo.currentData() or app.DEFAULT_MODEL
        cfg.save()
        logging.getLogger("thinkfarm").info("Configuration saved successfully.")

    # ------------------------------------------------------------- worker
    def toggle_provider(self):
        if self.worker and self.worker.is_alive():
            logging.getLogger("thinkfarm.provider").info("Stopping provider...")
            self.toggle_btn.setText("Stopping…")
            self.toggle_btn.setEnabled(False)
            self._set_form_enabled(False)
            self.worker.stop()
            return

        self.save_config()
        try:
            # Mirror headless app.py's pre-flight check (for the selected model).
            missing = [p for p in self._required_files() if not p.exists()]
            if missing:
                raise RuntimeError("required files missing:\n  " + "\n  ".join(str(p) for p in missing))

            logging.getLogger("thinkfarm").info("Starting provider...")
            # Start tailing from the end of any existing log so only this run shows.
            self._tail_offset = app.SERVER_LOG.stat().st_size if app.SERVER_LOG.exists() else 0
            self.worker = ProviderWorker(self.signals, self.config_manager)
            self.worker.start()
            self.toggle_btn.setText("Stop Provider")
            self._set_form_enabled(False)
        except Exception as e:
            logging.getLogger("thinkfarm").error(f"Failed to start provider: {e}")

    def _stop_worker(self, join=True, timeout=25):
        if not (self.worker and self.worker.is_alive()):
            return
        self.worker.stop()
        if join:
            self.worker.join(timeout)
            if self.worker.is_alive():
                logging.getLogger("thinkfarm").warning("Graceful stop timed out — forcing.")
                self.worker.force_stop()
                self.worker.join(5)

    def _set_form_enabled(self, enabled):
        for w in self._form_fields:
            w.setEnabled(enabled)

    def _on_worker_finished(self):
        self.worker = None
        self.toggle_btn.setText("Start Provider")
        self.toggle_btn.setEnabled(True)
        self._set_form_enabled(True)
        self.server_indicator.set_status("stopped")
        self.provider_indicator.set_status("stopped")
        self.server_status_lbl.setText("llama-server: Stopped")
        self.provider_status_lbl.setText("provider: Stopped")

    # ----------------------------------------------------------- status UI
    def _on_server_state(self, state):
        s = state.lower()
        if "failed" in s or s.startswith("error"):
            self.server_indicator.set_status("error")
        elif "ready" in s:
            self.server_indicator.set_status("connected")
        else:  # Starting… / Loading model… etc.
            self.server_indicator.set_status("connecting")
        self.server_status_lbl.setText(f"llama-server: {state}")

    def _on_provider_status(self, status):
        s = status.lower()
        if "error" in s or "failed" in s:
            self.provider_indicator.set_status("error")
        elif "disconnected" in s:  # reconnects happen in the background
            self.provider_indicator.set_status("connecting")
        elif "connected" in s:
            self.provider_indicator.set_status("running")
        elif "connecting" in s or "stopping" in s:
            self.provider_indicator.set_status("connecting")
        elif "probing" in s:
            self.provider_indicator.set_status("probing")
        else:  # Stopped / other terminal states
            self.provider_indicator.set_status("stopped" if "stopped" in s else "connecting")
        self.provider_status_lbl.setText(f"provider: {status}")

    # ----------------------------------------------------------------- logs
    def append_log(self, message, level):
        color_map = {
            "INFO": "#1c1c1e",
            "WARNING": "#b45309",
            "ERROR": "#b91c1c",
            "CRITICAL": "#b91c1c",
            "DEBUG": "#6b7280",
        }
        color = color_map.get(level, "#1c1c1e")
        self.log_area.moveCursor(QTextCursor.MoveOperation.End)
        self.log_area.insertHtml(f"<span style='color: {color};'>{message}</span><br>")
        # Prune excess lines from the top (>50K chars of HTML markup).
        if len(self.log_area.toHtml()) > 50_000:
            old_html = self.log_area.toHtml()
            self.log_area.setHtml(old_html[-50_000:])

    def _poll_server_log(self):
        """Append new lines from the child llama-server's log file."""
        if not (self.worker and self.worker.is_alive()):
            return
        try:
            if not app.SERVER_LOG.exists():
                return
            size = app.SERVER_LOG.stat().st_size
            offset = self._tail_offset or 0
            if size < offset:
                offset = 0  # truncated / rotated
            if size == offset:
                return
            with open(app.SERVER_LOG, "rb") as f:
                f.seek(offset)
                chunk = f.read(size - offset)
            # Advance only past the last newline so a partial final line is re-read next poll.
            cut = chunk.rfind(b"\n")
            if cut < 0:
                return
            self._tail_offset = offset + cut + 1
            complete = chunk[:cut]
            if not complete:
                return
            for ln in complete.decode(errors="replace").splitlines():
                if ln.strip():
                    self.append_log(f"[server] {ln}", "INFO")
        except OSError:
            pass

    def _check_version(self):
        """Fetch current app version from server and check against PROVIDER_VERSION."""
        url = self.VERSION_CHECK_URL

        try:
            data = None
            if httpx is not None:
                resp = httpx.get(url, timeout=5.0)
                if resp.status_code == 200:
                    data = resp.json()
            elif requests is not None:
                resp = requests.get(url, timeout=5.0)
                if resp.status_code == 200:
                    data = resp.json()
            else:
                import json
                import urllib.request
                req = urllib.request.Request(url, headers={"User-Agent": "thinkfarm-provider"})
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode())

            if not data:
                return

            server_provider = data.get("provider")
            if server_provider is None:
                return

            if server_provider > PROVIDER_VERSION:
                download_url = data.get("download_url", "")
                self._show_update_dialog(server_provider, download_url)
        except Exception:
            pass  # network issue — silently skip

    def _show_update_dialog(self, server_version: int, download_url: str):
        """Show a modal update available dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("App Update Available")
        dialog.setFixedWidth(380)
        dialog.setWindowFlags(
            dialog.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint
        )

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)

        title = QLabel("App update available")
        title.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #548889;"
        )
        layout.addWidget(title)

        body = QLabel(
            f"A newer version ({server_version}) is available on the server."
        )
        body.setStyleSheet("color: #1c1c1e; font-size: 13px;")
        layout.addWidget(body)

        desc = QLabel("Please update the app before continuing.")
        desc.setStyleSheet("color: #8e8e93; font-size: 12px;")
        layout.addWidget(desc)

        if download_url:
            link_btn = QPushButton(f"Download v{server_version}")
            link_btn.setStyleSheet("""
                QPushButton {
                    background-color: #548889;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 10px 24px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover { background-color: #436d6e; }
            """)

            def open_link():
                import webbrowser
                webbrowser.open(download_url)

            link_btn.clicked.connect(open_link)
            layout.addSpacing(12)
            layout.addWidget(link_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        close_btn = QPushButton("Ok")
        close_btn.setFixedSize(200, 40)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #d2d2d7;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #aeaeb2; }
        """)
        close_btn.clicked.connect(dialog.close)
        layout.addSpacing(6)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        dialog.adjustSize()
        dialog.exec()

    def closeEvent(self, event):
        # Minimize to tray instead of quitting when the tray icon is visible.
        if self.tray_icon.isVisible():
            self.hide()
            event.ignore()
            return
        if self.download_worker and self.download_worker.is_alive():
            self.download_worker.cancel()
        self._stop_worker(join=True)
        event.accept()


class QtLogHandler(logging.Handler):
    """Bridge stdlib logging records into the GUI log pane via a signal emitter."""

    def __init__(self, emit_fn):
        super().__init__()
        self.emit_fn = emit_fn

    def emit(self, record):
        msg = self.format(record)
        try:
            self.emit_fn(msg, record.levelname)
        except Exception:
            pass


if __name__ == "__main__":
    window_model_arg = app.parse_model_arg(sys.argv[1:])
    app_qt = QApplication(sys.argv)
    window = ThinkfarmProviderApp(model_override=window_model_arg)
    window.show()
    sys.exit(app_qt.exec())
