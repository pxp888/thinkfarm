import configparser
import multiprocessing
import os
import socket
import sys
import threading
import customtkinter as ctk
from tkinter import messagebox
import requests
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from main import app as fastapi_app


class ConsumerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("thinkfarm Client Control")
        self.root.geometry("800x900")

        self.server = None
        self.server_thread = None
        self.stop_event = threading.Event()
        self._stop_flag = threading.Event()
        
        self.model_vars = {}             # model_name -> BooleanVar
        self.model_filter_var = ctk.StringVar()  # search filter text

        # Light theme with dark sidebar
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        # UI Elements
        self.setup_ui()
        
        # Load SERVER_URL
        self.load_server_url()
        
        # Fetch models to populate list
        self.root.after(100, self.refresh_models)

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
        ui_font = ("Inter", "Ubuntu", "Segoe UI", "sans-serif")
        mono_font = ("JetBrains Mono", "Fira Code", "Cascadia Code", "Monospace")

        self.root.configure(fg_color="#f5f5f7")
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Sidebar - Pure Black
        self.sidebar_frame = ctk.CTkFrame(self.root, width=240, corner_radius=0, fg_color="#000000", border_width=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="thinkfarm",
                                       text_color="#FF7F7F",
                                       font=ctk.CTkFont(family=ui_font[0], size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=25, pady=(40, 40))

        self.start_btn = ctk.CTkButton(self.sidebar_frame, text="Start Client", height=45, corner_radius=8,
                                       command=self.start_service, fg_color="#2ecc71", hover_color="#27ae60",
                                       text_color="#FFFFFF",
                                       font=ctk.CTkFont(family=ui_font[0], size=14, weight="bold"))
        self.start_btn.grid(row=1, column=0, padx=25, pady=10, sticky="ew")

        self.stop_btn = ctk.CTkButton(self.sidebar_frame, text="Stop Client", height=45, corner_radius=8,
                                      command=self.stop_service, state="disabled", fg_color="#333333", hover_color="#e74c3c",
                                      text_color="#FFFFFF",
                                      font=ctk.CTkFont(family=ui_font[0], size=14, weight="bold"))
        self.stop_btn.grid(row=2, column=0, padx=25, pady=10, sticky="ew")

        # Version/Info at bottom of sidebar
        self.info_label = ctk.CTkLabel(self.sidebar_frame, text="Client v0.3",
                                       text_color="#666666",
                                       font=ctk.CTkFont(family=ui_font[0], size=11))
        self.info_label.grid(row=6, column=0, padx=20, pady=20)

        # Main content
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=40, pady=40)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.config_card = ctk.CTkFrame(self.main_frame, corner_radius=16, fg_color="#ffffff", border_width=1, border_color="#e5e5e7")
        self.config_card.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        self.config_card.grid_columnconfigure(1, weight=1)

        self.header_label = ctk.CTkLabel(self.config_card, text="Client Configuration",
                                         text_color="#1d1d1f",
                                         font=ctk.CTkFont(family=ui_font[0], size=22, weight="bold"))
        self.header_label.grid(row=0, column=0, columnspan=3, sticky="w", padx=30, pady=(30, 20))

        # Consumer ID
        self.consumer_id_label = ctk.CTkLabel(self.config_card, text="Client Identifier",
                                              text_color="#86868b",
                                              font=ctk.CTkFont(family=ui_font[0], size=13, weight="normal"))
        self.consumer_id_label.grid(row=1, column=0, padx=(30, 15), pady=(0, 15), sticky="w")

        self.consumer_id_var = ctk.StringVar()
        self.consumer_id_entry = ctk.CTkEntry(self.config_card, textvariable=self.consumer_id_var,
                                              placeholder_text="Enter client ID...",
                                              height=40, corner_radius=8, border_width=1,
                                              fg_color="#f5f5f7", border_color="#d2d2d7",
                                              text_color="#1d1d1f",
                                              font=ctk.CTkFont(family=mono_font[0], size=13))
        self.consumer_id_entry.grid(row=1, column=1, padx=(0, 30), pady=(0, 15), sticky="ew")

        # Server Port
        self.port_label = ctk.CTkLabel(self.config_card, text="Local Server Port",
                                       text_color="#86868b",
                                       font=ctk.CTkFont(family=ui_font[0], size=13, weight="normal"))
        self.port_label.grid(row=2, column=0, padx=(30, 15), pady=(0, 30), sticky="w")

        self.server_port_var = ctk.StringVar()
        self.server_port_entry = ctk.CTkEntry(self.config_card, textvariable=self.server_port_var,
                                              placeholder_text="11434",
                                              height=40, corner_radius=8, border_width=1,
                                              fg_color="#f5f5f7", border_color="#d2d2d7",
                                              text_color="#1d1d1f",
                                              font=ctk.CTkFont(family=mono_font[0], size=13))
        self.server_port_entry.grid(row=2, column=1, padx=(0, 30), pady=(0, 30), sticky="ew")

        # Whitelist Section
        self.whitelist_header_frame = ctk.CTkFrame(self.config_card, fg_color="transparent")
        self.whitelist_header_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=30, pady=(20, 10))
        self.whitelist_header_frame.grid_columnconfigure(0, weight=1)

        self.whitelist_header = ctk.CTkLabel(self.whitelist_header_frame, text="Model Whitelist",
                                             text_color="#1d1d1f",
                                             font=ctk.CTkFont(family=ui_font[0], size=18, weight="bold"))
        self.whitelist_header.grid(row=0, column=0, sticky="w")

        self.refresh_btn = ctk.CTkButton(self.whitelist_header_frame, text="Refresh Models", width=100, height=28, corner_radius=6,
                                         command=self.refresh_models, fg_color="#f5f5f7", hover_color="#e5e5e7",
                                         text_color="#1d1d1f", border_width=1, border_color="#d2d2d7",
                                         font=ctk.CTkFont(family=ui_font[0], size=11, weight="bold"))
        self.refresh_btn.grid(row=0, column=1, sticky="e")

        self.whitelist_enabled_var = ctk.BooleanVar(value=False)
        self.whitelist_checkbox = ctk.CTkCheckBox(self.config_card, text="Enable Model Whitelisting",
                                                  variable=self.whitelist_enabled_var,
                                                  font=ctk.CTkFont(family=ui_font[0], size=13))
        self.whitelist_checkbox.grid(row=5, column=0, columnspan=2, padx=30, pady=(0, 10), sticky="w")

        # Model filter/search entry
        self.filter_frame = ctk.CTkFrame(self.config_card, fg_color="transparent")
        self.filter_frame.grid(row=6, column=0, columnspan=2, padx=30, pady=(0, 5), sticky="ew")
        self.filter_frame.grid_columnconfigure(1, weight=1)

        self.filter_icon_label = ctk.CTkLabel(self.filter_frame, text="🔍", text_color="#86868b",
                                               font=ctk.CTkFont(size=14))
        self.filter_icon_label.grid(row=0, column=0, padx=(0, 8), sticky="e")

        self.model_filter_var.trace_add("write", lambda *args: self._apply_model_filter())
        self.filter_entry = ctk.CTkEntry(self.filter_frame, textvariable=self.model_filter_var,
                                         placeholder_text="Filter models...",
                                         height=32, corner_radius=8, border_width=1,
                                         fg_color="#ffffff", border_color="#d2d2d7",
                                         text_color="#1d1d1f",
                                         font=ctk.CTkFont(family=ui_font[0], size=12))
        self.filter_entry.grid(row=0, column=1, sticky="ew")

        # Select All / Select None buttons
        self.toggle_frame = ctk.CTkFrame(self.config_card, fg_color="transparent")
        self.toggle_frame.grid(row=7, column=0, columnspan=2, padx=30, pady=(5, 0), sticky="e")
        self.toggle_all_btn = ctk.CTkButton(self.toggle_frame, text="Select All", width=110, height=28, corner_radius=6,
                                            command=self._select_all,
                                            fg_color="#f5f5f7", hover_color="#e5e5e7",
                                            text_color="#1d1d1f", border_width=1, border_color="#d2d2d7",
                                            font=ctk.CTkFont(family=ui_font[0], size=11, weight="bold"))
        self.toggle_all_btn.pack(side="right", padx=(8, 0))

        self.toggle_none_btn = ctk.CTkButton(self.toggle_frame, text="Select None", width=110, height=28, corner_radius=6,
                                             command=self._select_none,
                                             fg_color="#f5f5f7", hover_color="#e5e5e7",
                                             text_color="#1d1d1f", border_width=1, border_color="#d2d2d7",
                                             font=ctk.CTkFont(family=ui_font[0], size=11, weight="bold"))
        self.toggle_none_btn.pack(side="right")

        self.models_frame = ctk.CTkScrollableFrame(self.config_card, height=200, label_text="Select models to show locally",
                                                  fg_color="#f5f5f7", label_font=ctk.CTkFont(family=ui_font[0], size=12, weight="bold"))
        self.models_frame.grid(row=8, column=0, columnspan=2, padx=30, pady=(10, 30), sticky="ew")

        # Root-level mouse wheel binding for Linux (CTkScrollableFrame bind_all uses MouseWheel which doesn't work on Linux)
        self.root.bind_all("<Button-4>", self._on_root_mousewheel)
        self.root.bind_all("<Button-5>", self._on_root_mousewheel)

        # Actions
        self.actions_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.actions_frame.grid(row=1, column=0, sticky="e")

        self.save_all_btn = ctk.CTkButton(self.actions_frame, text="Save Settings", width=160, height=40, corner_radius=8,
                                          command=self.save_all_settings, fg_color="#FF7F7F", hover_color="#FF5F5F",
                                          text_color="#FFFFFF",
                                          font=ctk.CTkFont(family=ui_font[0], size=14, weight="bold"))
        self.save_all_btn.pack(side="right")

        # Status Label/Status Dot (hidden until status updates)
        self.consumer_id_status = ctk.CTkLabel(self.config_card, text="", font=ctk.CTkFont(size=10), text_color="#555555")
        self.consumer_id_status.grid(row=3, column=1, padx=(0, 30), pady=(0, 10), sticky="e")
        
        self.server_port_status = ctk.CTkLabel(self.config_card, text="", font=ctk.CTkFont(size=10), text_color="#555555")
        # Reuse status area

        # Load current data
        self.load_consumer_id()
        self.load_server_port()

        # Status bar
        self.status_frame = ctk.CTkFrame(self.root, height=50, corner_radius=0, fg_color="#ffffff", border_width=1, border_color="#e5e5e7")
        self.status_frame.grid(row=1, column=1, sticky="ew")

        self.status_dot = ctk.CTkLabel(self.status_frame, text="●", text_color="#ff3b30",
                                       font=ctk.CTkFont(family=ui_font[0], size=22))
        self.status_dot.pack(side="left", padx=(30, 10))

        self.status_text = ctk.CTkLabel(self.status_frame, text="SYSTEM STOPPED",
                                         text_color="#86868b",
                                         font=ctk.CTkFont(family=ui_font[0], size=12, weight="bold"))
        self.status_text.pack(side="left")

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
                    self.root.after(0, lambda: self._populate_models(models))
            except Exception as e:
                print(f"Error fetching models: {e}")
        
        threading.Thread(target=_fetch, daemon=True).start()

    def _on_root_mousewheel(self, event):
        """Intercept mouse wheel at root level and delegate to models_frame canvas."""
        if event.num in (4, 5):
            direction = -5 if event.num == 4 else 5
            self.models_frame._parent_canvas.yview_scroll(direction, "units")
            return "break"

    def _populate_models(self, models):
        """Populate the whitelist list with models fetched from the server."""
        # Clear existing checkboxes
        for widget in self.models_frame.winfo_children():
            widget.destroy()

        self.model_vars = {}

        filter_text = self.model_filter_var.get().lower().strip()

        for m in models:
            name = m.get("name") if isinstance(m, dict) else m
            if filter_text and filter_text not in name.lower():
                continue
            var = ctk.BooleanVar(value=False)
            self.model_vars[name] = var
            cb = ctk.CTkCheckBox(self.models_frame, text=name, variable=var,
                                    font=ctk.CTkFont(size=12))
            cb.pack(anchor="w", padx=10, pady=5)

        # After populating, load which ones are checked
        self.load_whitelist()

    def _select_all(self):
        """Select all (visible) model checkboxes."""
        for var in self.model_vars.values():
            var.set(True)
        self.toggle_all_btn.configure(fg_color="#2ecc71", hover_color="#27ae60", text_color="#ffffff")
        self._update_toggle_states()

    def _select_none(self):
        """Deselect all (visible) model checkboxes."""
        for var in self.model_vars.values():
            var.set(False)
        self.toggle_none_btn.configure(fg_color="#e74c3c", hover_color="#c0392b", text_color="#ffffff")
        self._update_toggle_states()

    def _update_toggle_states(self):
        """Reset button styling after a delay so they return to neutral."""
        self.root.after(0, lambda: (
            self.toggle_all_btn.configure(fg_color="#f5f5f7", hover_color="#e5e5e7", text_color="#1d1d1f"),
            self.toggle_none_btn.configure(fg_color="#f5f5f7", hover_color="#e5e5e7", text_color="#1d1d1f")
        ))

    def _apply_model_filter(self):
        """Re-filter the model list based on the current filter text."""
        try:
            response = requests.get(f"{self.server_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                self.root.after(0, lambda m=models: self._populate_models(m))
        except Exception:
            pass

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
                self.whitelist_enabled_var.set(enabled)
                
                models_str = config.get("consumer", "whitelist_models", fallback="")
                selected_models = [m.strip() for m in models_str.split(",") if m.strip()]
                
                for m in selected_models:
                    if m in self.model_vars:
                        self.model_vars[m].set(True)
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

            enabled = self.whitelist_enabled_var.get()
            config.set("consumer", "whitelist_enabled", str(enabled).lower())

            selected = [name for name, var in self.model_vars.items() if var.get()]
            config.set("consumer", "whitelist_models", ",".join(selected))

            with open(config_path, "w") as f:
                config.write(f)
            
            self.server_port_status.configure(text="Settings saved!", text_color="#2ecc71")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save whitelist: {e}")

    def _check_port(self, port):
        """Check if the server's port is open via IPv4 and IPv6."""
        addrs = [
            ("127.0.0.1", port),    # IPv4
            ("::1", port),          # IPv6
            ("0.0.0.0", port),      # all interfaces (IPv4)
        ]
        for (host, p) in addrs:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.01)
                result = sock.connect_ex((host, p))
                sock.close()
                if result == 0:
                    return True
            except Exception:
                pass
        return False

    def start_service(self):
        """Start the consumer uvicorn server programmatically."""
        port = self.load_port()

        # Configure uvicorn settings
        config = uvicorn.Config(
            app=fastapi_app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True,
            lifespan="on",
        )
        self.server = uvicorn.Server(config)

        # Configuration
        self._starting_port = port

        # Flag to cancel polling on stop
        self._stop_flag.clear()
        polling_remaining = [300]

        def _port_check_fn():
            """Closured check to avoid passing port around."""
            return self._check_port(self._starting_port)

        def _poll_server_ready():
            if self._stop_flag.is_set() or self.server is None:
                return
            if _port_check_fn():
                self.root.after(0, self._on_server_ready)
                return
            if polling_remaining[0] > 0 and not self._stop_flag.is_set():
                polling_remaining[0] -= 1
                self.root.after(100, _poll_server_ready)

        try:
            # Start the server in a background thread
            self.stop_event.clear()
            self.server_thread = threading.Thread(
                target=self.server.run,
                args=(),
                daemon=True,
            )
            self.server_thread.start()

            # Start port-polling in the Tk mainloop
            self.root.after(0, _poll_server_ready)

            # Start a timeout alarm if the server doesn't become ready in 30s
            self.root.after(30200, self._on_server_timeout)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start service: {e}")

    def _on_server_ready(self):
        """Called from Tk mainloop when the server port becomes reachable."""
        if self._stop_flag.is_set() or self.server is None:
            return
        self.start_btn.configure(state="disabled", fg_color="#d1d1d6")
        self.stop_btn.configure(state="normal", fg_color="#ff3b30", hover_color="#ff2d55")
        self.status_text.configure(text="SYSTEM RUNNING", text_color="#34c759")
        self.status_dot.configure(text_color="#34c759")

    def _on_server_timeout(self):
        """Called from Tk mainloop if the port is still not reachable after the timeout.
        This is expected during slow startup — just cancel polling silently."""
        if self._stop_flag.is_set() or self.server is None:
            return
        if self._check_port(self._starting_port):
            self.root.after(0, self._on_server_ready)

    def stop_service(self):
        """Stop the consumer uvicorn server."""
        if self.server is not None:
            self._stop_flag.set()  # cancel port polling first
            self.stop_event.set()

            # Stop the uvicorn server which will trigger graceful shutdown
            self.server.should_exit = True

            # Wait for server to stop (with timeout for graceful shutdown)
            try:
                # uvicorn.Server.run() runs synchronously in our thread
                if self.server_thread and self.server_thread.is_alive():
                    self.server_thread.join(timeout=5)
            except Exception as e:
                pass
            finally:
                self.server = None
                self._update_ui_stopped()

        else:
            self._update_ui_stopped()

    def _update_ui_stopped(self):
        self.start_btn.configure(state="normal", fg_color="#34c759", hover_color="#30d158")
        self.stop_btn.configure(state="disabled", fg_color="#e5e5ea")
        self.status_text.configure(text="SYSTEM STOPPED", text_color="#86868b")
        self.status_dot.configure(text_color="#ff3b30")

    def load_consumer_id(self):
        """Load the current consumer_id from config.ini in ~/.thinkfarm/."""
        try:
            config_dir = os.path.expanduser("~/.thinkfarm")
            config_path = os.path.join(config_dir, "config.ini")
            
            if not os.path.exists(config_path):
                self.consumer_id_var.set("")
                self.consumer_id_status.configure(text="Config not found")
                return

            config = configparser.ConfigParser()
            config.read(config_path)

            if config.has_section("consumer") and config.has_option("consumer", "consumer_id"):
                consumer_id = config.get("consumer", "consumer_id")
                self.consumer_id_var.set(consumer_id)
                self.consumer_id_status.configure(text=f"Loaded: {consumer_id}")
            else:
                self.consumer_id_var.set("")
                self.consumer_id_status.configure(text="No client_id in config")
        except Exception as e:
            self.consumer_id_var.set("")
            self.consumer_id_status.configure(text=f"Error loading: {e}")

    def save_consumer_id(self):
        """Save the consumer_id to config.ini in ~/.thinkfarm/."""
        new_consumer_id = self.consumer_id_var.get().strip()

        if not new_consumer_id:
            self.consumer_id_status.configure(text="Error: ID cannot be empty", text_color="#e74c3c")
            return

        try:
            config_dir = os.path.expanduser("~/.thinkfarm")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, "config.ini")

            config = configparser.ConfigParser()
            config.read(config_path)

            if not config.has_section("consumer"):
                config.add_section("consumer")

            config.set("consumer", "consumer_id", new_consumer_id)

            with open(config_path, "w") as f:
                config.write(f)

            self.consumer_id_status.configure(text="Client ID saved!", text_color="#2ecc71")
        except Exception as e:
            self.consumer_id_status.configure(text=f"Error saving: {e}", text_color="#e74c3c")
            messagebox.showerror("Error", f"Failed to save client_id: {e}")

    def load_server_port(self):
        """Load the current server port from config and populate the entry."""
        port = self.load_port()
        self.server_port_var.set(str(port))

    def save_server_port(self):
        """Save the server port to ~/.thinkfarm/config.ini."""
        new_port = self.server_port_var.get().strip()

        if not new_port:
            self.server_port_status.configure(text="Port cannot be empty", text_color="#e74c3c")
            return

        try:
            port_num = int(new_port)
            if port_num < 1 or port_num > 65535:
                raise ValueError
        except ValueError:
            self.server_port_status.configure(text="Invalid port number", text_color="#e74c3c")
            messagebox.showerror("Error", "Port must be a number between 1 and 65535.")
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

            self.server_port_status.configure(text="Port saved!", text_color="#2ecc71")
        except Exception as e:
            self.server_port_status.configure(text=f"Error saving: {e}", text_color="#e74c3c")
            messagebox.showerror("Error", f"Failed to save port: {e}")

    def on_closing(self):
        self.stop_service()
        self.root.destroy()


if __name__ == "__main__":
    # Required for PyInstaller + multiprocessing support
    multiprocessing.freeze_support()

    root = ctk.CTk()
    app = ConsumerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
