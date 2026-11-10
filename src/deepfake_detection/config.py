
import json
from pathlib import Path


def load_config(config_path="configs/config.json"):
    """
    Load project settings from a JSON configuration file.
    """

    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8-sig") as file:
        config = json.load(file)

    required_sections = [
        "dataset",
        "labels",
        "model",
        "training"
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing configuration section: {section}"
            )

    return config
