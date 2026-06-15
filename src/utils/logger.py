"""Training logger for metrics, messages, and configuration tracking."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional


class Logger:
    """Unified training logger that writes to file and console.

    Manages log files under a structured directory layout::

        log_dir/
        └── experiment_name/
            ├── train.log
            └── metrics.jsonl

    Args:
        log_dir: Root directory for log output (e.g. ``outputs/logs``).
        experiment_name: Name of the current experiment run.

    Example::

        logger = Logger("outputs/logs", "experiment_001")
        logger.log_metrics({"train_loss": 0.42, "lr": 1e-4}, step=100)
        logger.log_message("Epoch 1 completed")
    """

    def __init__(self, log_dir: str, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # File handler -- detailed format
        log_file = self.log_dir / "train.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )

        # Console handler -- concise format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )

        self._logger = logging.getLogger(f"phydiff.{experiment_name}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)

        # Metrics file (JSON Lines format for easy parsing)
        self._metrics_path = self.log_dir / "metrics.jsonl"

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log a set of metrics at a given training step.

        Metrics are appended as one JSON object per line to ``metrics.jsonl``,
        making them easy to parse with pandas or jq.

        Args:
            metrics: Dictionary of metric name to value.
            step: Current training step or epoch number.

        Example::

            logger.log_metrics({"loss": 0.35, "csi": 0.78}, step=50)
        """
        record = {"step": step, **metrics}

        with open(self._metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        summary = " | ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in metrics.items())
        self._logger.info("Step %d -- %s", step, summary)

    def log_message(self, message: str, level: str = "info") -> None:
        """Log a text message at the specified level.

        Args:
            message: The message string to record.
            level: Logging level. One of ``'debug'``, ``'info'``,
                ``'warning'``, ``'error'``, ``'critical'``. Defaults to
                ``'info'``.

        Raises:
            ValueError: If ``level`` is not a valid logging level.

        Example::

            logger.log_message("Starting Stage 2 training")
            logger.log_message("CUDA OOM, reducing batch size", level="warning")
        """
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }

        if level not in level_map:
            raise ValueError(
                f"Invalid log level '{level}'. "
                f"Must be one of: {list(level_map.keys())}"
            )

        self._logger.log(level_map[level], message)

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log the full configuration dictionary.

        The configuration is written to a dedicated ``config.json`` file in the
        experiment directory and also emitted at INFO level.

        Args:
            config: Configuration dictionary to record.

        Example::

            logger.log_config(training_config)
        """
        config_file = self.log_dir / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)

        self._logger.info("Configuration saved to %s", config_file)
