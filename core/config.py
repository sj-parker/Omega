import yaml
from pathlib import Path

class Config:
    _instance = None
    _config = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        config_path = Path(__file__).parent.parent / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
        else:
            # Default values
            self._config = {
                "models": {
                    "main": "qwen2.5:7b",
                    "fast": "phi3:mini",
                    "use_ollama": True,
                    "use_multi_model": True
                },
                "system": {
                    "reflection_interval": 60.0,
                    "user_id_default": "user1",
                    "port": 8000
                }
            }

    def get(self, key, default=None):
        keys = key.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

# Global config instance
config = Config()
