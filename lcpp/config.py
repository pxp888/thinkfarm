import os
import uuid
from configparser import ConfigParser, NoSectionError
from pathlib import Path


class ConfigManager:
    _SECTION_PROVIDER = "provider"
    _SECTION_CONSUMER = "consumer"

    def __init__(self, workspace_dir=None):
        self.workspace_dir = Path(workspace_dir or "/home/pxperrine/Documents/code/unity")
        self.env_path = self.workspace_dir / ".env"
        self.config_dir = Path(os.path.expanduser("~/.thinkfarm"))
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.config_dir / "custom.ini"

        # Defaults – never read from disk; only the loaded values survive
        self.server_url: str = "https://app.thinkfarm.net"
        self.provider_id: str = str(uuid.uuid4())
        self.consumer_id: str = ""
        self.port: int = 11435
        self.slots: int = 1
        self.whitelist_enabled: bool = False
        self.whitelist_models: list[str] = []
        self.local_lcpp_url: str = "http://localhost:8080"
        self.auto_manage_models: bool = False
        self.gb_allowed: float = 0.0
        self.lcpp_restart_cmd: str = ""
        self.managed_lcpp: bool = os.name == "nt"
        self.lcpp_models_path: str = ""
        self.model_name: str = ""
        self.model_digest: str = ""
        self.lccp_context: int = 0  # explicit context limit for job acceptance (0 = use probed cache, -1 = unlimited)

        self.load()

    def load(self):
        if not self.config_path.exists():
            return

        raw = ConfigParser()
        raw.read(self.config_path, encoding="utf-8")

        # Migrate old "client" section to "provider" and "consumer" sections
        if raw.has_section("client"):
            if not raw.has_section(self._SECTION_PROVIDER):
                raw.add_section(self._SECTION_PROVIDER)
            if not raw.has_section(self._SECTION_CONSUMER):
                raw.add_section(self._SECTION_CONSUMER)
            
            for option in raw.options("client"):
                val = raw.get("client", option)
                opt_upper = option.upper()
                if opt_upper in ["CONSUMER_ID", "CLIENT_PORT", "WHITELIST_ENABLED", "WHITELIST_MODELS"]:
                    raw.set(self._SECTION_CONSUMER, option, val)
                else:
                    raw.set(self._SECTION_PROVIDER, option, val)
            raw.remove_section("client")
            # Save migrated file
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                raw.write(f)

        # Tolerate files written without section headers (dotenv-style)
        if not raw.sections():
            raw.add_section(self._SECTION_PROVIDER)
            raw.add_section(self._SECTION_CONSUMER)
            for line in self.config_path.read_text().strip().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v_val = line.split("=", 1)
                    k_upper = k.strip().upper()
                    if k_upper in ["CONSUMER_ID", "CLIENT_PORT", "WHITELIST_ENABLED", "WHITELIST_MODELS"]:
                        raw.set(self._SECTION_CONSUMER, k.strip(), v_val.strip())
                    else:
                        raw.set(self._SECTION_PROVIDER, k.strip(), v_val.strip())
        
        env_vals: dict[str, str] = {}
        if self.env_path.exists():
            for line in self.env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                val = val.strip().strip("'\"\r")
                env_vals[key.strip()] = val

        # helper to fetch value with fallback chain
        def v(sect: str, key: str, default: str = "") -> str | None:
            try:
                val = raw.get(sect, key, fallback=None)
            except (NoSectionError, KeyError):
                val = None
            return val or env_vals.get(key) or (default if default else None)

        s_raw = v(self._SECTION_PROVIDER, "SLOTS", "1") or "1"
        try:
            self.slots = max(1, int(s_raw))
        except ValueError:
            self.slots = 1

        self.server_url = env_vals.get("CENTRAL_SERVER_URL") or "https://app.thinkfarm.net"
        self.provider_id = v(self._SECTION_PROVIDER, "PROVIDER_ID", str(uuid.uuid4())) or self.provider_id
        self.consumer_id = v(self._SECTION_CONSUMER, "CONSUMER_ID", "") or ""
        self.port = int(v(self._SECTION_CONSUMER, "CLIENT_PORT", "11435") or "11435")
        self.local_lcpp_url = v(self._SECTION_PROVIDER, "LOCAL_LCPP_URL", "") or v(self._SECTION_PROVIDER, "LOCAL_OLLAMA_URL", "http://localhost:8080") or self.local_lcpp_url
        self.lcpp_restart_cmd = v(self._SECTION_PROVIDER, "LCPP_RESTART_CMD", "") or v(self._SECTION_PROVIDER, "OLLAMA_RESTART_CMD", "") or self.lcpp_restart_cmd
        self.lcpp_models_path = v(self._SECTION_PROVIDER, "LCPP_MODELS_PATH", "") or v(self._SECTION_PROVIDER, "OLLAMA_MODELS_PATH", "") or self.lcpp_models_path
        self.model_name = v(self._SECTION_PROVIDER, "MODEL_NAME", "") or ""
        self.model_digest = v(self._SECTION_PROVIDER, "MODEL_DIGEST", "") or ""

        lc = v(self._SECTION_PROVIDER, "lccp_context", "") or ""
        try:
            self.lccp_context = int(lc) if lc else 0
        except ValueError:
            self.lccp_context = 0

        # Booleans
        wl_en = v(self._SECTION_CONSUMER, "WHITELIST_ENABLED", "false") or "false"
        self.whitelist_enabled = wl_en.lower() == "true"

        am = v(self._SECTION_PROVIDER, "AUTO_MANAGE_MODELS", "false") or "false"
        self.auto_manage_models = am.lower() == "true"

        self.managed_lcpp = (os.name == "nt")

        gb = v(self._SECTION_PROVIDER, "GB_ALLOWED", "0") or "0"
        try:
            self.gb_allowed = float(gb)
        except ValueError:
            self.gb_allowed = 0.0

        # Lists
        wl_raw = raw.get(self._SECTION_CONSUMER, "WHITELIST_MODELS", fallback=None) or env_vals.get("WHITELIST_MODELS") or ""
        self.whitelist_models = [m.strip() for m in wl_raw.split(",") if m.strip()]

    def save(self):
        raw = ConfigParser()
        raw.read(self.config_path, encoding="utf-8")

        if raw.has_section("client"):
            raw.remove_section("client")

        if not raw.has_section(self._SECTION_PROVIDER):
            raw.add_section(self._SECTION_PROVIDER)
        if not raw.has_section(self._SECTION_CONSUMER):
            raw.add_section(self._SECTION_CONSUMER)

        raw.set(self._SECTION_PROVIDER, "PROVIDER_ID", self.provider_id)
        raw.set(self._SECTION_PROVIDER, "SLOTS", str(self.slots))
        raw.set(self._SECTION_CONSUMER, "CONSUMER_ID", self.consumer_id)
        raw.set(self._SECTION_CONSUMER, "CLIENT_PORT", str(self.port))
        raw.set(self._SECTION_CONSUMER, "WHITELIST_ENABLED", "true" if self.whitelist_enabled else "false")
        raw.set(self._SECTION_CONSUMER, "WHITELIST_MODELS", ",".join(self.whitelist_models))
        raw.set(self._SECTION_PROVIDER, "LOCAL_LCPP_URL", self.local_lcpp_url)
        raw.set(self._SECTION_PROVIDER, "AUTO_MANAGE_MODELS", "true" if self.auto_manage_models else "false")
        raw.set(self._SECTION_PROVIDER, "GB_ALLOWED", str(self.gb_allowed))
        raw.set(self._SECTION_PROVIDER, "LCPP_RESTART_CMD", self.lcpp_restart_cmd)
        raw.set(self._SECTION_PROVIDER, "LCPP_MODELS_PATH", self.lcpp_models_path)
        raw.set(self._SECTION_PROVIDER, "MODEL_NAME", self.model_name)
        raw.set(self._SECTION_PROVIDER, "MODEL_DIGEST", self.model_digest)
        raw.set(self._SECTION_PROVIDER, "lccp_context", str(self.lccp_context))

        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            raw.write(f)
