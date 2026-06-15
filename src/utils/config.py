"""Configuration file loading and merging utilities."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.

    Example::

        config = load_config("src/configs/training_config.yaml")
        print(config["training"]["batch_size"])
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """Deep-merge two configuration dictionaries.

    Values in ``override_config`` take precedence. Nested dictionaries are
    merged recursively rather than replaced wholesale.

    Args:
        base_config: The base configuration dictionary.
        override_config: The overriding configuration dictionary.

    Returns:
        A new merged dictionary (neither input is mutated).

    Example::

        merged = merge_configs(
            {"lr": 1e-3, "optim": {"betas": [0.9, 0.999]}},
            {"lr": 5e-4, "optim": {"weight_decay": 0.01}},
        )
        # {'lr': 5e-4, 'optim': {'betas': [0.9, 0.999], 'weight_decay': 0.01}}
    """
    merged: Dict[str, Any] = {}

    all_keys = set(base_config.keys()) | set(override_config.keys())
    for key in all_keys:
        base_val = base_config.get(key)
        override_val = override_config.get(key)

        if isinstance(base_val, dict) and isinstance(override_val, dict):
            merged[key] = merge_configs(base_val, override_val)
        elif key in override_config:
            merged[key] = override_val
        else:
            merged[key] = base_val

    return merged


def save_config(config: Dict, save_path: str) -> None:
    """Save a configuration dictionary to a YAML file.

    Args:
        config: Configuration dictionary to save.
        save_path: Destination file path. Parent directories are created
            automatically if they do not exist.

    Example::

        save_config(config, "experiments/configs/run_001.yaml")
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
