"""Precipitation Forecast Visualization Tools.

Visualization utilities for precipitation forecast evaluation, including
prediction comparison plots, metric bar charts, and training curves.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional


# Standard precipitation color map (mm/h)
PRECIP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "precipitation",
    [
        (0.0, "#ffffff"),    # White: no rain
        (0.05, "#e0e0e0"),   # Light gray: drizzle
        (0.1, "#b0d0ff"),    # Light blue: light rain
        (0.2, "#0066cc"),    # Blue: moderate rain
        (0.3, "#00cc00"),    # Green: heavy rain
        (0.4, "#ffcc00"),    # Yellow: very heavy rain
        (0.5, "#ff6600"),    # Orange: extreme rain
        (0.7, "#cc0000"),    # Red: severe
        (1.0, "#800080"),    # Purple: catastrophic
    ],
)


class PrecipitationVisualizer:
    """Precipitation forecast visualization toolkit.

    Provides methods for generating publication-quality figures for
    precipitation forecast evaluation, including spatial maps, metric
    comparisons, and training diagnostics.

    Attributes:
        save_dir: Directory to save generated figures.
        dpi: Figure resolution in dots per inch.
        figsize: Default figure size (width, height) in inches.
    """

    def __init__(
        self,
        save_dir: str = "outputs/figures",
        dpi: int = 150,
        figsize: tuple = (12, 8),
    ):
        """Initialize PrecipitationVisualizer.

        Args:
            save_dir: Directory to save generated figures. Created if
                it does not exist.
            dpi: Figure resolution. Default: 150.
            figsize: Default figure size in inches. Default: (12, 8).
        """
        self.save_dir = save_dir
        self.dpi = dpi
        self.figsize = figsize
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_prediction_comparison(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        save_path: str,
        title: Optional[str] = None,
        vmin: float = 0.0,
        vmax: float = 100.0,
        colorbar_label: str = "Precipitation (mm)",
    ) -> None:
        """Plot side-by-side comparison of predicted and observed precipitation.

        Generates a figure with three panels:
        1. Observed precipitation (ground truth).
        2. Predicted precipitation.
        3. Error map (prediction - observation).

        Args:
            pred: Predicted precipitation array [H, W] or [C, H, W].
            target: Observed precipitation array [H, W] or [C, H, W].
            save_path: Path to save the figure (relative to save_dir).
            title: Optional overall figure title.
            vmin: Minimum value for color scale.
            vmax: Maximum value for color scale.
            colorbar_label: Label for the colorbar.
        """
        # Handle multi-channel input (take first channel)
        if pred.ndim == 3:
            pred = pred[0]
        if target.ndim == 3:
            target = target[0]

        error = pred - target

        fig, axes = plt.subplots(1, 3, figsize=(self.figsize[0], self.figsize[1] * 0.4))

        # Observed
        im0 = axes[0].imshow(
            target, cmap=PRECIP_CMAP, vmin=vmin, vmax=vmax, interpolation="nearest"
        )
        axes[0].set_title("Observed", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Longitude Index")
        axes[0].set_ylabel("Latitude Index")
        plt.colorbar(im0, ax=axes[0], label=colorbar_label, fraction=0.046)

        # Predicted
        im1 = axes[1].imshow(
            pred, cmap=PRECIP_CMAP, vmin=vmin, vmax=vmax, interpolation="nearest"
        )
        axes[1].set_title("Predicted", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Longitude Index")
        plt.colorbar(im1, ax=axes[1], label=colorbar_label, fraction=0.046)

        # Error map
        error_max = max(abs(error.min()), abs(error.max()))
        im2 = axes[2].imshow(
            error,
            cmap="RdBu_r",
            vmin=-error_max,
            vmax=error_max,
            interpolation="nearest",
        )
        axes[2].set_title("Error (Pred - Obs)", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Longitude Index")
        plt.colorbar(im2, ax=axes[2], label="Error (mm)", fraction=0.046)

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

        plt.tight_layout()
        full_path = os.path.join(self.save_dir, save_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        fig.savefig(full_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_metrics_by_threshold(
        self,
        metrics: Dict[str, float],
        save_path: str,
        title: Optional[str] = None,
        metric_names: Optional[List[str]] = None,
    ) -> None:
        """Plot bar chart of detection metrics across thresholds.

        Generates a grouped bar chart showing CSI, POD, FAR, and HSS
        for each precipitation threshold.

        Args:
            metrics: Dictionary of metrics as returned by
                PrecipitationMetrics.compute_all_metrics.
            save_path: Path to save the figure.
            title: Optional figure title.
            metric_names: List of metric suffixes to plot. Default:
                ['csi', 'pod', 'far', 'hss'].
        """
        if metric_names is None:
            metric_names = ["csi", "pod", "far", "hss"]

        # Extract thresholds from metric keys
        thresholds = []
        for key in sorted(metrics.keys()):
            if key.endswith("_csi") and "precip_gt_" in key:
                threshold_str = key.split("precip_gt_")[1].split("mm")[0]
                try:
                    thresholds.append(float(threshold_str))
                except ValueError:
                    continue
        thresholds.sort()

        if not thresholds:
            raise ValueError(
                "No threshold-based metrics found in the provided dictionary."
            )

        fig, ax = plt.subplots(figsize=self.figsize)
        x = np.arange(len(thresholds))
        width = 0.2
        offsets = np.arange(len(metric_names)) - (len(metric_names) - 1) / 2.0

        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

        for i, metric_name in enumerate(metric_names):
            values = []
            for threshold in thresholds:
                key = f"precip_gt_{threshold}mm_{metric_name}"
                values.append(metrics.get(key, 0.0))

            ax.bar(
                x + offsets[i] * width,
                values,
                width,
                label=metric_name.upper(),
                color=colors[i % len(colors)],
                alpha=0.85,
            )

        # Threshold labels
        labels = []
        name_map = {
            0.1: "0.1\n(drizzle)",
            5.0: "5.0\n(light)",
            10.0: "10.0\n(moderate)",
            25.0: "25.0\n(heavy)",
            50.0: "50.0\n(extreme)",
        }
        for t in thresholds:
            labels.append(name_map.get(t, f"{t}"))

        ax.set_xlabel("Precipitation Threshold (mm)", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(title or "Detection Metrics by Threshold", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        full_path = os.path.join(self.save_dir, save_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        fig.savefig(full_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_training_curves(
        self,
        train_loss: List[float],
        val_loss: List[float],
        save_path: str,
        title: Optional[str] = None,
        log_scale: bool = False,
    ) -> None:
        """Plot training and validation loss curves.

        Generates a line plot showing training and validation loss
        over epochs, with optional logarithmic y-axis and loss
        component breakdown if provided.

        Args:
            train_loss: Training loss values per epoch.
            val_loss: Validation loss values per epoch.
            save_path: Path to save the figure.
            title: Optional figure title.
            log_scale: Whether to use logarithmic y-axis. Default: False.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        epochs = range(1, len(train_loss) + 1)

        ax.plot(
            epochs, train_loss,
            color="#1f77b4", linewidth=1.5,
            label="Training Loss", alpha=0.9,
        )
        ax.plot(
            epochs, val_loss,
            color="#ff7f0e", linewidth=1.5,
            label="Validation Loss", alpha=0.9,
        )

        # Mark minimum validation loss
        if val_loss:
            min_epoch = np.argmin(val_loss) + 1
            min_val = min(val_loss)
            ax.axvline(x=min_epoch, color="#d62728", linestyle="--", alpha=0.5)
            ax.annotate(
                f"Best: {min_val:.4f}\n(epoch {min_epoch})",
                xy=(min_epoch, min_val),
                xytext=(min_epoch + max(1, len(train_loss) * 0.05), min_val * 1.1),
                arrowprops=dict(arrowstyle="->", color="#d62728"),
                fontsize=9,
                color="#d62728",
            )

        if log_scale:
            ax.set_yscale("log")

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title(title or "Training and Validation Loss", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        full_path = os.path.join(self.save_dir, save_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        fig.savefig(full_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_loss_components(
        self,
        loss_history: Dict[str, List[float]],
        save_path: str,
        title: Optional[str] = None,
    ) -> None:
        """Plot individual loss components over training epochs.

        Useful for monitoring how different loss terms evolve during
        multi-task training.

        Args:
            loss_history: Dictionary mapping loss component names to
                lists of values per epoch. Example:
                {'mse': [0.5, 0.4, ...], 'csi': [0.8, 0.7, ...]}.
            save_path: Path to save the figure.
            title: Optional figure title.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        epochs = range(1, max(len(v) for v in loss_history.values()) + 1)
        colors = plt.cm.tab10(np.linspace(0, 1, len(loss_history)))

        for (name, values), color in zip(loss_history.items(), colors):
            ax.plot(epochs[:len(values)], values, label=name, color=color, linewidth=1.5)

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title(title or "Loss Component Evolution", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        full_path = os.path.join(self.save_dir, save_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        fig.savefig(full_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
