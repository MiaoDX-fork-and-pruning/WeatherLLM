"""Evaluate a GMCP-only PhyDiff-Net checkpoint on the test split.

This script complements ``scripts/train_gmcp_only.py``. It loads a trained
checkpoint, runs diffusion sampling on the test split, denormalizes
predictions back to physical mm/6h, and computes the full metric suite
(CSI / POD / FAR / HSS / RMSE / MAE / bias / correlation / extreme F1).

Usage:
    python scripts/evaluate_gmcp_only.py \\
        --config configs/training_gmcp_only.yaml \\
        --checkpoint outputs/gmcp_only/best_model.pt
    python scripts/evaluate_gmcp_only.py --max_samples 50  # quick check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.gmcp_sequence_dataset import GMCPSequenceDataset  # noqa: E402
from src.evaluation.metrics import PrecipitationMetrics  # noqa: E402
from src.models.phydiff_net import PhyDiffNet  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = logging.getLogger(__name__)

CSI_THRESHOLDS = [0.1, 1.0, 5.0, 10.0, 25.0, 50.0]
EXTREME_THRESHOLDS = {"heavy": 25.0, "very_heavy": 50.0, "extreme": 100.0}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a GMCP-only PhyDiff-Net checkpoint."
    )
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "training_gmcp_only.yaml",
        help="Training config used to build the model and test split.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to the .pt checkpoint to evaluate.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Cap on number of test samples (for quick checks).",
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "gmcp_eval",
        help="Directory to save metrics and predictions.",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU device index.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def get_device(gpu_id: int) -> torch.device:
    """Return the compute device."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def denormalize_log_minmax(
    x: torch.Tensor, norm_min: float, norm_max: float
) -> torch.Tensor:
    """Invert the dataset's log_minmax transform: x -> expm1(x*(max-min)+min)."""
    log_x = x * (norm_max - norm_min) + norm_min
    return torch.expm1(log_x)


def load_model(
    checkpoint_path: Path, model_config: Dict, device: torch.device
) -> PhyDiffNet:
    """Build the model and load checkpoint weights."""
    model = PhyDiffNet(model_config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Loaded checkpoint %s (epoch %s, %d params)",
        checkpoint_path, ckpt.get("epoch", "?"), n_params,
    )
    return model


@torch.no_grad()
def evaluate(
    model: PhyDiffNet,
    test_dataset: GMCPSequenceDataset,
    device: torch.device,
    norm_min: Optional[float],
    norm_max: Optional[float],
    max_samples: Optional[int],
) -> Dict[str, np.ndarray]:
    """Run sampling on the test set and collect physical-space arrays.

    Returns a dict with 'predictions' and 'targets' in mm/6h.
    """
    n = len(test_dataset)
    if max_samples is not None:
        n = min(n, max_samples)
    logger.info("Evaluating on %d test samples...", n)

    all_pred = []
    all_target = []
    for idx in range(n):
        batch = test_dataset[idx]
        # input: [T_in, 1, H, W] -> [1, T_in, 1, H, W]
        inp = batch["input"].unsqueeze(0).to(device)
        target = batch["target"].to(device)  # [T, H, W] normalized

        output = model(gmcp_data=inp)
        pred = output["precipitation"]  # [1, T, H, W] normalized

        # Denormalize to physical mm/6h if normalization is active.
        if norm_min is not None and norm_max is not None:
            pred_phys = denormalize_log_minmax(pred, norm_min, norm_max)
            target_phys = denormalize_log_minmax(target, norm_min, norm_max)
        else:
            pred_phys = pred.squeeze(0)
            target_phys = target

        all_pred.append(pred_phys.squeeze(0).cpu().numpy())
        all_target.append(target_phys.cpu().numpy())

        if (idx + 1) % 10 == 0 or idx == 0:
            logger.info("  %d / %d samples done", idx + 1, n)

    return {
        "predictions": np.stack(all_pred, axis=0),
        "targets": np.stack(all_target, axis=0),
    }


def compute_extreme_f1(
    predictions: np.ndarray, targets: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Precision / recall / F1 for extreme-event detection."""
    pred_bin = (predictions > threshold).astype(np.float32)
    target_bin = (targets > threshold).astype(np.float32)
    tp = float(np.sum(pred_bin * target_bin))
    fp = float(np.sum(pred_bin * (1 - target_bin)))
    fn = float(np.sum((1 - pred_bin) * target_bin))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    set_seed(args.seed)

    config = load_config(args.config)
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    device = get_device(args.gpu_id)

    # Build the test dataset from the config's test split.
    test_split = data_config["splits"]["test"]
    test_base = {
        "data_path": data_config["data_path"],
        "input_timesteps": data_config.get("input_timesteps", 12),
        "forecast_horizon": data_config.get("forecast_horizon", 4),
        "normalize": data_config.get("normalize", None),
        "use_preprocessed": data_config.get("use_preprocessed", True),
        "region": data_config.get("region"),
    }
    test_dataset = GMCPSequenceDataset({**test_base, **test_split})

    norm_min = getattr(test_dataset, "input_min", None)
    norm_max = getattr(test_dataset, "input_max", None)
    logger.info(
        "Test set: %d samples, normalize=%s, norm_min=%s, norm_max=%s",
        len(test_dataset), data_config.get("normalize"),
        norm_min, norm_max,
    )

    model = load_model(args.checkpoint, model_config, device)

    result = evaluate(
        model=model,
        test_dataset=test_dataset,
        device=device,
        norm_min=norm_min,
        norm_max=norm_max,
        max_samples=args.max_samples,
    )
    predictions = result["predictions"]
    targets = result["targets"]
    logger.info(
        "Predictions shape: %s, targets shape: %s",
        predictions.shape, targets.shape,
    )

    # Compute metrics in physical space.
    metrics_calc = PrecipitationMetrics(thresholds=CSI_THRESHOLDS)
    metrics = metrics_calc.compute_all_metrics(predictions, targets)

    for level, thr in EXTREME_THRESHOLDS.items():
        f1 = compute_extreme_f1(predictions, targets, thr)
        metrics[f"extreme_{level}_precision"] = f1["precision"]
        metrics[f"extreme_{level}_recall"] = f1["recall"]
        metrics[f"extreme_{level}_f1"] = f1["f1"]

    # Print summary.
    print(metrics_calc.summary_table(metrics))
    print("\nExtreme Event F1:")
    for level, thr in EXTREME_THRESHOLDS.items():
        print(
            f"  {level:>12} (>{thr:>3.0f}mm): "
            f"precision={metrics[f'extreme_{level}_precision']:.4f}, "
            f"recall={metrics[f'extreme_{level}_recall']:.4f}, "
            f"f1={metrics[f'extreme_{level}_f1']:.4f}"
        )

    # Save results.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = args.output_dir / "metrics.json"
    serializable = {
        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
        for k, v in metrics.items()
    }
    serializable["checkpoint"] = str(args.checkpoint)
    serializable["config"] = str(args.config)
    serializable["n_samples"] = int(predictions.shape[0])
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    logger.info("Metrics saved to %s", metrics_file)

    np.save(args.output_dir / "predictions.npy", predictions)
    np.save(args.output_dir / "targets.npy", targets)
    logger.info("Predictions and targets saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
