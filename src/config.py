from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """
    Load the pipeline configuration from a YAML file.
    """

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
