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
import customtkinter as ctk
from tkinter import messagebox
import solo
import context_prober
from dotenv import load_dotenv
from ollama_client import OllamaClient
from model_manager import ModelManager

# Load environment variables at startup
load_dotenv()

_CONFIG_PATH = os.path.expanduser("~/.thinkfarm/config.ini")

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class ProviderGUI(ctk.CTk):
    def _get_free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    def __init__(self):
        super().__init__()
        self.title("thinkfarm Provider Control")
        self.geometry("800x500")
        self.configure(padx=0, pady=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._server_process = None
        self._stop_event = threading.Event()
        self._ollama_process = None
        self._ollama_log_file = None

        # Determine Ollama server URL & restart/start internal server
        self.restart_ollama_server()

        # Initialize managed model state
        self.ollama = OllamaClient(ollama_url=self._ollama_url)
        self.model_manager = ModelManager(self.ollama, os.environ.get("SERVER_URL", "https://app.thinkfarm.net"))
        self._mgmt_thread = threading.Thread(target=self._managed_model_loop, daemon=True)
        self._mgmt_thread.start()

        self.setup_ui()
        self._load_config()

    def restart_ollama_server(self):
        print("[MGMT] Restarting/checking internal ollama server...")
        
        # Terminate existing child process if running
        if hasattr(self, '_ollama_process') and self._ollama_process is not None:
            try:
                self._ollama_process.terminate()
                self._ollama_process.wait(timeout=5)
            except Exception as e:
                print(f"[MGMT] Error terminating old ollama process: {e}")
            self._ollama_process = None
            
        if hasattr(self, '_ollama_log_file') and self._ollama_log_file is not None:
            try:
                self._ollama_log_file.close()
            except Exception:
                pass
            self._ollama_log_file = None
                
        # Read config to see if custom models path is configured
        models_path = ""
        config = configparser.ConfigParser()
        if os.path.exists(_CONFIG_PATH):
            config.read(_CONFIG_PATH)
            if config.has_section("provider") and config.has_option("provider", "ollama_models_path"):
                models_path = config.get("provider", "ollama_models_path").strip()

        # We need a child Ollama server. Ensure url is set to a non-11434 port.
        if not hasattr(self, '_ollama_url') or "11434" in self._ollama_url:
            port = self._get_free_port()
            self._ollama_url = f"http://127.0.0.1:{port}"
            
        print(f"[MGMT] Starting child Ollama server on port {self._ollama_url.split(':')[-1]}...")
        
        # Prepare environment
        env = os.environ.copy()
        port = self._ollama_url.split(":")[-1]
        env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
        env["OLLAMA_DEBUG"] = "1"
        if models_path:
            env["OLLAMA_MODELS"] = models_path

        if os.name == 'nt':
            # Ensure critical Windows environment variables are present
            system_root = env.get("SystemRoot") or env.get("SYSTEMROOT") or "C:\\Windows"
            env["SystemRoot"] = system_root
            if "SystemDrive" not in env and "SYSTEMDRIVE" not in env:
                env["SystemDrive"] = "C:"
            
            # Find the Path variable (case-insensitive search)
            path_key = next((k for k in env if k.upper() == "PATH"), "PATH")
            current_path = env.get(path_key, "")
            paths = current_path.split(os.pathsep) if current_path else []
            
            # Ensure System32 is in PATH
            sys32 = os.path.join(system_root, "System32")
            if sys32 not in paths:
                paths.append(sys32)
                
            # Ensure default Ollama installation folder is in PATH
            user_profile = env.get("USERPROFILE") or os.path.expanduser("~")
            ollama_default_path = os.path.join(user_profile, "AppData", "Local", "Programs", "Ollama")
            if ollama_default_path not in paths:
                paths.append(ollama_default_path)
                
            env[path_key] = os.pathsep.join(paths)

        creationflags = 0
        if os.name == 'nt':
            creationflags = subprocess.CREATE_NO_WINDOW
            
        # Log output to a file for troubleshooting
        log_dir = os.path.expanduser("~/.thinkfarm")
        os.makedirs(log_dir, exist_ok=True)
        try:
            self._ollama_log_file = open(os.path.join(log_dir, "ollama_internal.log"), "a", encoding="utf-8")
            stdout_target = self._ollama_log_file
            stderr_target = self._ollama_log_file
        except Exception as e:
            print(f"[MGMT] Warning: could not open internal Ollama log file: {e}")
            stdout_target = subprocess.DEVNULL
            stderr_target = subprocess.DEVNULL

        self._ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            creationflags=creationflags,
            stdin=subprocess.DEVNULL,
            stdout=stdout_target,
            stderr=stderr_target
        )

        os.environ["OLLAMA_URL"] = self._ollama_url
        if hasattr(self, 'ollama') and self.ollama is not None:
            self.ollama.base_url = self._ollama_url
            import httpx
            # Close old client if possible
            try:
                asyncio.run(self.ollama.close())
            except Exception:
                pass
            self.ollama.httpx_client = httpx.AsyncClient(base_url=self._ollama_url, timeout=30.0)

    def _load_config(self):
        if not os.path.exists(_CONFIG_PATH):
            return
        config = configparser.ConfigParser()
        config.read(_CONFIG_PATH)
        if config.has_section("provider"):
            # Ensure old ollama_url config is removed so subprocesses use the env var
            if config.has_option("provider", "ollama_url"):
                config.remove_option("provider", "ollama_url")
                with open(_CONFIG_PATH, "w") as f:
                    config.write(f)

            if config.has_option("provider", "provider_id"):
                self.provider_id_entry.delete(0, "end")
                self.provider_id_entry.insert(0, config.get("provider", "provider_id"))
            if config.has_option("provider", "ollama_models_path"):
                self.models_path_entry.delete(0, "end")
                self.models_path_entry.insert(0, config.get("provider", "ollama_models_path"))
            if config.has_option("provider", "managed_storage_gb"):
                self.storage_entry.delete(0, "end")
                self.storage_entry.insert(0, config.get("provider", "managed_storage_gb"))
            if config.has_option("provider", "auto_manage"):
                val = config.getboolean("provider", "auto_manage", fallback=False)
                if val:
                    self.auto_manage_cb.select()
                else:
                    self.auto_manage_cb.deselect()

    def _managed_model_loop(self):
        """Background loop to handle automated model management."""
        while True:
            # Wait until the service is started by the user
            if self._server_process is None:
                time.sleep(2)
                continue
                
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
            
            # Wait 1 hour before next check, checking frequently so it responds cleanly to exits
            for _ in range(1800):
                time.sleep(2)

    # ── UI ────────────────────────────────────────────────────
    def setup_ui(self):
        ui_font = ("Inter", "Ubuntu", "Segoe UI", "sans-serif")
        mono_font = ("JetBrains Mono", "Fira Code", "Cascadia Code", "Monospace")
        
        self.configure(fg_color="#f5f5f7")

        # Sidebar - Pure Black
        self.sidebar_frame = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color="#e6e6e6", border_width=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="thinkfarm",
                                       text_color="black",
                                       font=ctk.CTkFont(family=ui_font[0], size=24, weight="normal"))
        self.logo_label.grid(row=0, column=0, padx=25, pady=(40, 40))

        self.start_btn = ctk.CTkButton(self.sidebar_frame, text="Start Provider", height=45, corner_radius=8,
                                       command=self.start_service, fg_color="#2ecc71", hover_color="#27ae60",
                                       text_color="#FFFFFF",
                                       font=ctk.CTkFont(family=ui_font[0], size=14, weight="bold"))
        self.start_btn.grid(row=1, column=0, padx=25, pady=10, sticky="ew")

        self.stop_btn = ctk.CTkButton(self.sidebar_frame, text="Stop Provider", height=45, corner_radius=8,
                                      command=self.stop_service, state="disabled", fg_color="#333333", hover_color="#e74c3c",
                                      text_color="#FFFFFF",
                                      font=ctk.CTkFont(family=ui_font[0], size=14, weight="bold"))
        self.stop_btn.grid(row=2, column=0, padx=25, pady=10, sticky="ew")

        # Version/Info at bottom of sidebar
        self.info_label = ctk.CTkLabel(self.sidebar_frame, text="Provider v12",
                                       text_color="#666666",
                                       font=ctk.CTkFont(family=ui_font[0], size=11))
        self.info_label.grid(row=6, column=0, padx=20, pady=20)

        # Main content
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=40, pady=40)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.config_card = ctk.CTkFrame(self.main_frame, corner_radius=16, fg_color="#ffffff", border_width=1, border_color="#e5e5e7")
        self.config_card.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        self.config_card.grid_columnconfigure(1, weight=1)

        self.header_label = ctk.CTkLabel(self.config_card, text="Provider Configuration",
                                         text_color="#1d1d1f",
                                         font=ctk.CTkFont(family=ui_font[0], size=22, weight="bold"))
        self.header_label.grid(row=0, column=0, columnspan=3, sticky="w", padx=30, pady=(30, 20))

        self.provider_id_label = ctk.CTkLabel(self.config_card, text="Provider Identifier",
                                              text_color="#86868b",
                                              font=ctk.CTkFont(family=ui_font[0], size=13, weight="normal"))
        self.provider_id_label.grid(row=1, column=0, padx=(30, 15), pady=(0, 15), sticky="w")

        self.provider_id_entry = ctk.CTkEntry(self.config_card, placeholder_text="Enter unique ID...",
                                              height=40, corner_radius=8, border_width=1,
                                              fg_color="#f5f5f7", border_color="#d2d2d7",
                                              text_color="#1d1d1f",
                                              font=ctk.CTkFont(family=mono_font[0], size=13))
        self.provider_id_entry.grid(row=1, column=1, padx=(0, 30), pady=(0, 15), sticky="ew")

        self.models_path_label = ctk.CTkLabel(self.config_card, text="Model Storage Path",
                                              text_color="#86868b",
                                              font=ctk.CTkFont(family=ui_font[0], size=13, weight="normal"))
        self.models_path_label.grid(row=2, column=0, padx=(30, 15), pady=(0, 15), sticky="w")

        self.models_path_entry = ctk.CTkEntry(self.config_card, placeholder_text="D:\\Ollama\\Models (Optional)",
                                              height=40, corner_radius=8, border_width=1,
                                              fg_color="#f5f5f7", border_color="#d2d2d7",
                                              text_color="#1d1d1f",
                                              font=ctk.CTkFont(family=mono_font[0], size=13))
        self.models_path_entry.grid(row=2, column=1, padx=(0, 30), pady=(0, 15), sticky="ew")

        self.auto_manage_cb = ctk.CTkCheckBox(self.config_card, text="Manage models automatically",
                                              text_color="#86868b",
                                              font=ctk.CTkFont(family=ui_font[0], size=13, weight="normal"),
                                              fg_color="#FF7F7F", hover_color="#FF5F5F")
        self.auto_manage_cb.grid(row=3, column=1, padx=(0, 30), pady=(0, 15), sticky="w")

        self.storage_label = ctk.CTkLabel(self.config_card, text="Managed Model Storage (GB)",
                                         text_color="#86868b",
                                         font=ctk.CTkFont(family=ui_font[0], size=13, weight="normal"))
        self.storage_label.grid(row=4, column=0, padx=(30, 15), pady=(0, 30), sticky="w")

        self.storage_entry = ctk.CTkEntry(self.config_card, placeholder_text="30",
                                          height=40, corner_radius=8, border_width=1,
                                          fg_color="#f5f5f7", border_color="#d2d2d7",
                                          text_color="#1d1d1f",
                                          font=ctk.CTkFont(family=mono_font[0], size=13))
        self.storage_entry.insert(0, "30")
        self.storage_entry.grid(row=4, column=1, padx=(0, 30), pady=(0, 30), sticky="ew")

        self.save_btn = ctk.CTkButton(self.main_frame, text="Save Settings", width=160, height=40, corner_radius=8,
                                      command=self.save_config, fg_color="#8e8e93", hover_color="#636366",
                                      text_color="#FFFFFF",
                                      font=ctk.CTkFont(family=ui_font[0], size=14, weight="bold"))
        self.save_btn.grid(row=1, column=0, sticky="e")

        # Status bar
        self.status_frame = ctk.CTkFrame(self, height=50, corner_radius=0, fg_color="#ffffff", border_width=1, border_color="#e5e5e7")
        self.status_frame.grid(row=1, column=1, sticky="ew")

        self.status_dot = ctk.CTkLabel(self.status_frame, text="●", text_color="#ff3b30",
                                       font=ctk.CTkFont(family=ui_font[0], size=22))
        self.status_dot.pack(side="left", padx=(30, 10))

        self.status_text = ctk.CTkLabel(self.status_frame, text="SYSTEM STOPPED",
                                         text_color="#86868b",
                                         font=ctk.CTkFont(family=ui_font[0], size=12, weight="bold"))
        self.status_text.pack(side="left")

    # ── start / stop ────────────────────────────────────────────
    def start_service(self):
        """Orchestrate startup: probe context if needed, then start solo.py."""
        self.start_btn.configure(state="disabled", fg_color="#d1d1d6")
        self._stop_event.clear()

        def _startup_logic():
            try:
                # 1. Check for unscanned models
                url = self._ollama_url
                print(f"[STARTUP] Checking for unscanned models at {url}...")
                
                all_models = asyncio.run(context_prober.get_ollama_models(url))
                existing_limits = context_prober.load_context_limits()
                
                unscanned = [m for m in all_models if m not in existing_limits]
                
                if unscanned:
                    print(f"[STARTUP] Found {len(unscanned)} unscanned model(s): {unscanned}")
                    self.after(0, self._update_ui_probing)
                    
                    # Run probing
                    asyncio.run(context_prober.run_context_probing(
                        url,
                        unscanned,
                        existing_limits
                    ))
                    print("[STARTUP] Probing complete.")

                # 2. Start solo.py
                self.after(0, self._actual_start_service)

            except Exception as e:
                print(f"[STARTUP] Error during startup orchestration: {e}")
                self.after(0, self._update_ui_stopped)

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
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
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
                        self.after(0, self._update_ui_started)
                        return
                    if self._stop_event.is_set():
                        return
                    time.sleep(0.1)

            threading.Thread(target=check_startup, daemon=True).start()
        except Exception as e:
            print(f"Error starting provider: {e}")
            self._update_ui_stopped()

    def stop_service(self):
        if self._server_process is not None:
            self._stop_event.set()
            self._server_process.terminate()
            self._update_ui_stopping()

            def wait_join():
                try:
                    # Give it up to 60 seconds to finish active jobs
                    self._server_process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    print("Soft stop timed out, forcing termination...")
                    self._server_process.kill()
                    self._server_process.wait()
                
                self._server_process = None
                self.after(0, self._update_ui_stopped)

            threading.Thread(target=wait_join, daemon=True).start()
        else:
            self._update_ui_stopped()

    def _update_ui_started(self):
        self.start_btn.configure(state="disabled", fg_color="#d1d1d6")
        self.stop_btn.configure(state="normal", fg_color="#ff3b30", hover_color="#ff2d55")
        self.status_text.configure(text="SYSTEM RUNNING", text_color="#34c759")
        self.status_dot.configure(text_color="#34c759")

    def _update_ui_probing(self):
        self.status_text.configure(text="SYSTEM PROBING", text_color="#3498db")
        self.status_dot.configure(text_color="#3498db")

    def _update_ui_stopping(self):
        self.stop_btn.configure(state="disabled", fg_color="#d1d1d6")
        self.status_text.configure(text="SYSTEM STOPPING", text_color="#f39c12")
        self.status_dot.configure(text_color="#f39c12")

    def _update_ui_stopped(self):
        self.start_btn.configure(state="normal", fg_color="#34c759", hover_color="#30d158")
        self.stop_btn.configure(state="disabled", fg_color="#e5e5ea")
        self.status_text.configure(text="SYSTEM STOPPED", text_color="#86868b")
        self.status_dot.configure(text_color="#ff3b30")

    # ── config ──────────────────────────────────────────────────
    def save_config(self):
        new_id = self.provider_id_entry.get().strip()
        new_models_path = self.models_path_entry.get().strip()
        new_storage = self.storage_entry.get().strip()
        new_auto = "yes" if self.auto_manage_cb.get() else "no"

        errors = []
        if not new_id:
            errors.append("Provider ID cannot be empty")
        
        try:
            float(new_storage or "30")
        except ValueError:
            errors.append("Managed Model Storage must be a number")

        if errors:
            messagebox.showwarning("Configuration Error", "\n".join(errors))
            return

        try:
            config_dir = os.path.expanduser("~/.thinkfarm")
            os.makedirs(config_dir, exist_ok=True)
            config = configparser.ConfigParser()
            config.read(_CONFIG_PATH)
            if not config.has_section("provider"):
                config.add_section("provider")
            config.set("provider", "provider_id", new_id)
            config.set("provider", "ollama_models_path", new_models_path)
            config.set("provider", "managed_storage_gb", new_storage or "30")
            config.set("provider", "auto_manage", new_auto)
            with open(_CONFIG_PATH, "w") as f:
                config.write(f)
            
            self.restart_ollama_server()
            messagebox.showinfo("Success", "Settings saved and Ollama server restarted!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def on_closing(self):
        if self._server_process is not None:
            self.stop_service()
            
        if hasattr(self, '_ollama_process') and self._ollama_process is not None:
            try:
                self._ollama_process.terminate()
                self._ollama_process.wait(timeout=5)
            except Exception as e:
                print(f"Error terminating ollama process: {e}")
                
        if hasattr(self, '_ollama_log_file') and self._ollama_log_file is not None:
            try:
                self._ollama_log_file.close()
            except Exception:
                pass
                
        self.destroy()


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

    app = ProviderGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
