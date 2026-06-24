import os
from pathlib import Path
from dotenv import dotenv_values, set_key

class ConfigManager:
    def __init__(self, workspace_dir=None):
        self.workspace_dir = Path(workspace_dir or "/home/pxperrine/Documents/code/unity")
        self.env_path = self.workspace_dir / ".env"
        self.config_dir = Path(os.path.expanduser("~/.thinkfarm"))
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.config_dir / "config.txt"
        self.blacklist_file = self.config_dir / "blacklisted_models.json"
        
        # Default config settings
        self.defaults = {
            "SERVER_URL": "https://app.thinkfarm.net",
            "CONSUMER_ID": "8b4d68ce-82f6-4583-8718-7e21a0703f43",
            "PROVIDER_ID": "89eb71b1-dafb-4eb2-a528-6315739c7832",
            "CLIENT_PORT": "11435", # Default to 11435 so it doesn't conflict with local Ollama on 11434
            "WHITELIST_ENABLED": "false",
            "WHITELIST_MODELS": "",
            "LOCAL_OLLAMA_URL": "http://localhost:11434",
            "AUTO_MANAGE_MODELS": "false",
            "GB_ALLOWED": "0",
            "OLLAMA_RESTART_CMD": ""
        }
        self.load()

    def load(self):
        # Load environment values (mainly for CENTRAL_SERVER_URL and fallback)
        env_vals = dotenv_values(str(self.env_path)) if self.env_path.exists() else {}
        # Load config values (for settings saved via GUI)
        config_vals = dotenv_values(str(self.config_path)) if self.config_path.exists() else {}
            
        self.server_url = env_vals.get("CENTRAL_SERVER_URL") or self.defaults["SERVER_URL"]
        self.consumer_id = config_vals.get("CONSUMER_ID") or env_vals.get("CONSUMER_ID") or self.defaults["CONSUMER_ID"]
        self.provider_id = config_vals.get("PROVIDER_ID") or env_vals.get("PROVIDER_ID") or self.defaults["PROVIDER_ID"]
        self.port = int(config_vals.get("CLIENT_PORT") or env_vals.get("CLIENT_PORT") or self.defaults["CLIENT_PORT"])
        
        whitelist_enabled_str = config_vals.get("WHITELIST_ENABLED") or env_vals.get("WHITELIST_ENABLED") or self.defaults["WHITELIST_ENABLED"]
        self.whitelist_enabled = whitelist_enabled_str.lower() == "true"
        
        whitelist_models_str = config_vals.get("WHITELIST_MODELS") or env_vals.get("WHITELIST_MODELS") or self.defaults["WHITELIST_MODELS"]
        self.whitelist_models = [m.strip() for m in whitelist_models_str.split(",") if m.strip()]
        
        self.local_ollama_url = config_vals.get("LOCAL_OLLAMA_URL") or env_vals.get("LOCAL_OLLAMA_URL") or self.defaults["LOCAL_OLLAMA_URL"]

        auto_manage_models_str = config_vals.get("AUTO_MANAGE_MODELS") or env_vals.get("AUTO_MANAGE_MODELS") or self.defaults["AUTO_MANAGE_MODELS"]
        self.auto_manage_models = auto_manage_models_str.lower() == "true"
        
        gb_allowed_str = config_vals.get("GB_ALLOWED") or env_vals.get("GB_ALLOWED") or self.defaults["GB_ALLOWED"]
        try:
            self.gb_allowed = float(gb_allowed_str)
        except ValueError:
            self.gb_allowed = 0.0
            
        self.ollama_restart_cmd = config_vals.get("OLLAMA_RESTART_CMD") or env_vals.get("OLLAMA_RESTART_CMD") or self.defaults["OLLAMA_RESTART_CMD"]

    def save(self):
        # Update config.txt with GUI settings
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        set_key(str(self.config_path), "PROVIDER_ID", self.provider_id)
        set_key(str(self.config_path), "CONSUMER_ID", self.consumer_id)
        set_key(str(self.config_path), "CLIENT_PORT", str(self.port))
        set_key(str(self.config_path), "WHITELIST_ENABLED", "true" if self.whitelist_enabled else "false")
        set_key(str(self.config_path), "WHITELIST_MODELS", ",".join(self.whitelist_models))
        set_key(str(self.config_path), "LOCAL_OLLAMA_URL", self.local_ollama_url)
        set_key(str(self.config_path), "AUTO_MANAGE_MODELS", "true" if self.auto_manage_models else "false")
        set_key(str(self.config_path), "GB_ALLOWED", str(self.gb_allowed))
        set_key(str(self.config_path), "OLLAMA_RESTART_CMD", self.ollama_restart_cmd)

