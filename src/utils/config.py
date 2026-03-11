import yaml
from pathlib import Path


class Config:

    def __init__(self, config_path=None):

        # Resolve project root
        project_root = Path(__file__).resolve().parents[2]

        # Default config path
        if config_path is None:
            config_path = project_root / "configs" / "default.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, *keys):
        """
        Safely access nested config values.
        """
        value = self._config

        for key in keys:
            if key not in value:
                raise KeyError(f"Config key not found: {' -> '.join(keys)}")
            value = value[key]

        return value

    def __getitem__(self, key):
        return self._config[key]

    def __repr__(self):
        return f"Config({self._config})"