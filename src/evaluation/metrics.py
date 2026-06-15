"""Precipitation Forecast Evaluation Metrics.

Comprehensive evaluation metrics for precipitation forecasting, including
detection metrics (CSI, POD, FAR, HSS), continuous metrics (RMSE, MAE),
and distributional metrics. All metrics support multiple precipitation
thresholds for evaluating performance across different rainfall intensities.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union


class PrecipitationMetrics:
    """Precipitation forecast evaluation metrics.

    Computes a comprehensive set of evaluation metrics for precipitation
    predictions across multiple intensity thresholds.

    The threshold-based detection metrics include:
    - **CSI** (Critical Success Index): Fraction of hits relative to total
      hits, misses, and false alarms.
    - **POD** (Probability of Detection): Fraction of observed events
      that were correctly predicted.
    - **FAR** (False Alarm Ratio): Fraction of predicted events that
      did not occur.
    - **HSS** (Heidke Skill Score): Agreement between prediction and
      observation relative to chance agreement.

    The continuous metrics include:
    - **RMSE** (Root Mean Square Error): Standard deviation of prediction errors.
    - **MAE** (Mean Absolute Error): Average magnitude of prediction errors.

    Attributes:
        thresholds: List of precipitation thresholds for detection metrics.
        threshold_names: Human-readable names for each threshold.
    """

    def __init__(
        self,
        thresholds: List[float] = None,
    ):
        """Initialize PrecipitationMetrics.

        Args:
            thresholds: Precipitation thresholds (mm) for detection metrics.
                Default: [0.1, 5.0, 10.0, 25.0, 50.0] corresponding to
                drizzle, light rain, moderate rain, heavy rain, and very
                heavy rain.
        """
        if thresholds is None:
            thresholds = [0.1, 5.0, 10.0, 25.0, 50.0]
        self.thresholds = thresholds

        # Human-readable names for thresholds
        self.threshold_names = {
            0.1: "drizzle",
            5.0: "light_rain",
            10.0: "moderate_rain",
            25.0: "heavy_rain",
            50.0: "very_heavy_rain",
        }

    def _to_numpy(
        self, data: Union[torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """Convert tensor or array to numpy.

        Args:
            data: Input tensor or array.

        Returns:
            Numpy array.
        """
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    def csi(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        threshold: float,
    ) -> float:
        """Compute Critical Success Index (CSI) for a given threshold.

        CSI = hits / (hits + false_alarms + misses)

        Args:
            pred: Predicted precipitation values.
            target: Observed precipitation values (same shape as pred).
            threshold: Precipitation threshold in mm.

        Returns:
            CSI value between 0 (no skill) and 1 (perfect).
        """
        pred_np = self._to_numpy(pred)
        target_np = self._to_numpy(target)

        pred_binary = (pred_np > threshold).astype(np.float32)
        target_binary = (target_np > threshold).astype(np.float32)

        hits = np.sum(pred_binary * target_binary)
        false_alarms = np.sum(pred_binary * (1 - target_binary))
        misses = np.sum((1 - pred_binary) * target_binary)

        return float(hits / (hits + false_alarms + misses + 1e-8))

    def pod(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        threshold: float,
    ) -> float:
        """Compute Probability of Detection (POD) for a given threshold.

        POD = hits / (hits + misses)

        Also known as recall or true positive rate.

        Args:
            pred: Predicted precipitation values.
            target: Observed precipitation values (same shape as pred).
            threshold: Precipitation threshold in mm.

        Returns:
            POD value between 0 (missed all events) and 1 (detected all).
        """
        pred_np = self._to_numpy(pred)
        target_np = self._to_numpy(target)

        pred_binary = (pred_np > threshold).astype(np.float32)
        target_binary = (target_np > threshold).astype(np.float32)

        hits = np.sum(pred_binary * target_binary)
        misses = np.sum((1 - pred_binary) * target_binary)

        return float(hits / (hits + misses + 1e-8))

    def far(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        threshold: float,
    ) -> float:
        """Compute False Alarm Ratio (FAR) for a given threshold.

        FAR = false_alarms / (hits + false_alarms)

        Args:
            pred: Predicted precipitation values.
            target: Observed precipitation values (same shape as pred).
            threshold: Precipitation threshold in mm.

        Returns:
            FAR value between 0 (no false alarms) and 1 (all predictions wrong).
        """
        pred_np = self._to_numpy(pred)
        target_np = self._to_numpy(target)

        pred_binary = (pred_np > threshold).astype(np.float32)
        target_binary = (target_np > threshold).astype(np.float32)

        hits = np.sum(pred_binary * target_binary)
        false_alarms = np.sum(pred_binary * (1 - target_binary))

        return float(false_alarms / (hits + false_alarms + 1e-8))

    def hss(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        threshold: float,
    ) -> float:
        """Compute Heidke Skill Score (HSS) for a given threshold.

        HSS measures the fractional improvement of the forecast over
        a random forecast. HSS = 1 for perfect forecasts, HSS = 0 for
        random forecasts, and HSS < 0 for forecasts worse than random.

        HSS = 2 * (hits * misses - false_alarms * misses) /
              ((hits + misses) * (misses + false_alarms) +
               (hits + false_alarms) * (misses + false_alarms))

        Args:
            pred: Predicted precipitation values.
            target: Observed precipitation values (same shape as pred).
            threshold: Precipitation threshold in mm.

        Returns:
            HSS value. Higher is better (1 = perfect, 0 = random).
        """
        pred_np = self._to_numpy(pred)
        target_np = self._to_numpy(target)

        pred_binary = (pred_np > threshold).astype(np.float32)
        target_binary = (target_np > threshold).astype(np.float32)

        hits = np.sum(pred_binary * target_binary)
        false_alarms = np.sum(pred_binary * (1 - target_binary))
        misses = np.sum((1 - pred_binary) * target_binary)
        correct_negatives = np.sum(
            (1 - pred_binary) * (1 - target_binary)
        )

        numerator = 2.0 * (hits * correct_negatives - false_alarms * misses)
        denominator = (
            (hits + misses) * (misses + correct_negatives)
            + (hits + false_alarms) * (false_alarms + correct_negatives)
        )

        return float(numerator / (denominator + 1e-8))

    def rmse(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
    ) -> float:
        """Compute Root Mean Square Error (RMSE).

        RMSE = sqrt(mean((pred - target)^2))

        Args:
            pred: Predicted values.
            target: Observed values (same shape as pred).

        Returns:
            RMSE value (always non-negative).
        """
        pred_np = self._to_numpy(pred)
        target_np = self._to_numpy(target)

        return float(np.sqrt(np.mean((pred_np - target_np) ** 2)))

    def mae(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
    ) -> float:
        """Compute Mean Absolute Error (MAE).

        MAE = mean(|pred - target|)

        Args:
            pred: Predicted values.
            target: Observed values (same shape as pred).

        Returns:
            MAE value (always non-negative).
        """
        pred_np = self._to_numpy(pred)
        target_np = self._to_numpy(target)

        return float(np.mean(np.abs(pred_np - target_np)))

    def bias(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
    ) -> float:
        """Compute mean bias.

        bias = mean(pred - target)

        Positive bias indicates overestimation, negative indicates
        underestimation.

        Args:
            pred: Predicted values.
            target: Observed values (same shape as pred).

        Returns:
            Mean bias value.
        """
        pred_np = self._to_numpy(pred)
        target_np = self._to_numpy(target)

        return float(np.mean(pred_np - target_np))

    def area_bias(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        threshold: float,
    ) -> float:
        """Compute area bias for a given threshold.

        Area bias compares the fraction of the domain with precipitation
        above the threshold in the forecast vs observation.

        area_bias = sum(pred > threshold) / sum(target > threshold)

        A value of 1.0 indicates unbiased area prediction. Values > 1.0
        indicate over-prediction of precipitation area, and < 1.0 indicate
        under-prediction.

        Args:
            pred: Predicted precipitation values.
            target: Observed precipitation values.
            threshold: Precipitation threshold in mm.

        Returns:
            Area bias ratio.
        """
        pred_np = self._to_numpy(pred)
        target_np = self._to_numpy(target)

        pred_area = np.sum(pred_np > threshold)
        target_area = np.sum(target_np > threshold)

        return float(pred_area / (target_area + 1e-8))

    def correlation(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
    ) -> float:
        """Compute Pearson correlation coefficient.

        Args:
            pred: Predicted values.
            target: Observed values (same shape as pred).

        Returns:
            Correlation coefficient between -1 and 1.
        """
        pred_np = self._to_numpy(pred).flatten()
        target_np = self._to_numpy(target).flatten()

        # Handle constant inputs
        if np.std(pred_np) < 1e-10 or np.std(target_np) < 1e-10:
            return 0.0

        return float(np.corrcoef(pred_np, target_np)[0, 1])

    def compute_all_metrics(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        prefix: str = "",
    ) -> Dict[str, float]:
        """Compute all evaluation metrics across all thresholds.

        Args:
            predictions: Model predictions. Can be:
                - [B, C, H, W] or [B, H, W]: Single-sample or batch.
                - Will be flattened for computation.
            targets: Ground truth values, same shape as predictions.
            prefix: Optional prefix for metric names (e.g., '0h_', '6h_').

        Returns:
            Dictionary mapping metric names to values. Metric naming:
            - '{prefix}precip_gt_{threshold}mm_csi': CSI at threshold.
            - '{prefix}precip_gt_{threshold}mm_pod': POD at threshold.
            - '{prefix}precip_gt_{threshold}mm_far': FAR at threshold.
            - '{prefix}precip_gt_{threshold}mm_hss': HSS at threshold.
            - '{prefix}precip_gt_{threshold}mm_area_bias': Area bias.
            - '{prefix}rmse': Root mean square error.
            - '{prefix}mae': Mean absolute error.
            - '{prefix}bias': Mean bias.
            - '{prefix}correlation': Pearson correlation.
        """
        metrics: Dict[str, float] = {}

        # Threshold-based detection metrics
        for threshold in self.thresholds:
            name = self.threshold_names.get(threshold, f"{threshold}mm")
            base = f"{prefix}precip_gt_{threshold}mm"

            metrics[f"{base}_csi"] = self.csi(predictions, targets, threshold)
            metrics[f"{base}_pod"] = self.pod(predictions, targets, threshold)
            metrics[f"{base}_far"] = self.far(predictions, targets, threshold)
            metrics[f"{base}_hss"] = self.hss(predictions, targets, threshold)
            metrics[f"{base}_area_bias"] = self.area_bias(
                predictions, targets, threshold
            )

        # Continuous metrics
        metrics[f"{prefix}rmse"] = self.rmse(predictions, targets)
        metrics[f"{prefix}mae"] = self.mae(predictions, targets)
        metrics[f"{prefix}bias"] = self.bias(predictions, targets)
        metrics[f"{prefix}correlation"] = self.correlation(predictions, targets)

        return metrics

    def summary_table(
        self,
        metrics: Dict[str, float],
    ) -> str:
        """Generate a formatted summary table of metrics.

        Args:
            metrics: Dictionary of metrics as returned by compute_all_metrics.

        Returns:
            Formatted string table.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("  Precipitation Forecast Evaluation Summary")
        lines.append("=" * 70)

        # Threshold-based metrics
        lines.append(f"\n{'Threshold':<20} {'CSI':>8} {'POD':>8} "
                     f"{'FAR':>8} {'HSS':>8} {'Area Bias':>10}")
        lines.append("-" * 70)

        for threshold in self.thresholds:
            base = f"precip_gt_{threshold}mm"
            name = self.threshold_names.get(threshold, f"{threshold}mm")
            csi_val = metrics.get(f"{base}_csi", 0.0)
            pod_val = metrics.get(f"{base}_pod", 0.0)
            far_val = metrics.get(f"{base}_far", 0.0)
            hss_val = metrics.get(f"{base}_hss", 0.0)
            area_val = metrics.get(f"{base}_area_bias", 0.0)
            lines.append(
                f"{name:<20} {csi_val:>8.3f} {pod_val:>8.3f} "
                f"{far_val:>8.3f} {hss_val:>8.3f} {area_val:>10.3f}"
            )

        # Continuous metrics
        lines.append(f"\nContinuous Metrics:")
        lines.append("-" * 40)
        lines.append(f"  RMSE:       {metrics.get('rmse', 0.0):>10.3f}")
        lines.append(f"  MAE:        {metrics.get('mae', 0.0):>10.3f}")
        lines.append(f"  Bias:       {metrics.get('bias', 0.0):>10.3f}")
        lines.append(f"  Correlation:{metrics.get('correlation', 0.0):>10.3f}")
        lines.append("=" * 70)

        return "\n".join(lines)
