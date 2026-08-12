"""Tests for GMCPExtremeLoss."""

from __future__ import annotations

import pytest
import torch

from src.models.losses.gmcp_extreme_loss import GMCPExtremeLoss


# Realistic log_minmax bounds derived from GMCP data (6h cumulative, China).
# log1p(0) = 0 (min), log1p(~500) ~= 6.2 (max observed extreme).
NORM_MIN = 0.0
NORM_MAX = 6.2


def _make_loss(**overrides) -> GMCPExtremeLoss:
    defaults = dict(
        norm_min=NORM_MIN,
        norm_max=NORM_MAX,
        normalize="log_minmax",
        mse_weight=1.0,
        mae_weight=0.5,
        csi_weight=0.5,
        extreme_weight=1.0,
    )
    defaults.update(overrides)
    return GMCPExtremeLoss(**defaults)


def _to_norm(phys: torch.Tensor) -> torch.Tensor:
    """Map physical mm/6h to normalized space, matching the dataset."""
    return (torch.log1p(phys) - NORM_MIN) / (NORM_MAX - NORM_MIN)


def test_perfect_prediction_has_low_loss():
    """A perfect prediction should yield near-zero regression and CSI loss."""
    target_phys = torch.tensor([[[0.0, 5.0, 25.0], [50.0, 100.0, 0.1]]])
    norm = _to_norm(target_phys)
    loss = _make_loss()
    out = loss(norm, norm)
    assert out["mse"].item() < 1e-12
    assert out["mae"].item() < 1e-12
    assert out["csi"].item() < 1e-6
    # extreme MSE is also ~0 for a perfect prediction.
    assert out["extreme"].item() < 1e-10


def test_csi_loss_decreases_as_prediction_improves():
    """CSI loss should drop as predictions move closer to targets."""
    target_phys = torch.tensor([[[0.0, 0.0, 50.0], [0.0, 0.0, 50.0]]])
    norm_target = _to_norm(target_phys)

    # Bad prediction: all zeros (misses every extreme).
    bad_pred = _to_norm(torch.zeros_like(target_phys))
    # Good prediction: matches targets.
    good_pred = norm_target

    loss = _make_loss()
    bad_csi = loss(bad_pred, norm_target)["csi"].item()
    good_csi = loss(good_pred, norm_target)["csi"].item()
    assert good_csi < bad_csi


def test_extreme_weight_amplifies_extreme_errors():
    """An error at an extreme grid point should cost more than at a dry point."""
    target_phys = torch.tensor([[[0.0, 100.0]]])
    norm_target = _to_norm(target_phys)

    # Prediction under-predicts everywhere by the same normalized amount.
    pred = norm_target + 0.1
    pred = torch.clamp(pred, min=0.0)

    loss = _make_loss()
    out = loss(pred, norm_target)
    # The extreme (100mm) point dominates the weighted MSE.
    assert out["extreme"].item() > out["mse"].item()


def test_extreme_weight_scales_with_threshold():
    """Higher extreme_weight should increase the extreme component."""
    target_phys = torch.tensor([[[0.0, 100.0]]])
    norm_target = _to_norm(target_phys)
    pred = norm_target * 0.5  # severe under-prediction

    low = _make_loss(extreme_weight=0.1)(pred, norm_target)["total"].item()
    high = _make_loss(extreme_weight=10.0)(pred, norm_target)["total"].item()
    assert high > low


def test_denormalize_inverts_log_minmax():
    """denormalize should recover physical values from normalized space."""
    loss = _make_loss()
    phys = torch.tensor([0.0, 1.0, 25.0, 100.0])
    norm = _to_norm(phys)
    recovered = loss.denormalize(norm)
    assert torch.allclose(recovered, phys, atol=1e-5)


def test_none_normalize_passes_through():
    """With normalize=None, the loss operates directly in physical space."""
    loss = GMCPExtremeLoss(normalize=None, mse_weight=1.0)
    target = torch.tensor([[[0.0, 25.0, 100.0]]])
    out = loss(target, target)
    assert out["mse"].item() < 1e-12
    # CSI thresholds are in mm, applied directly.
    assert out["csi"].item() < 1e-6


def test_total_is_weighted_sum():
    """total must equal the weighted combination of components."""
    target_phys = torch.tensor([[[0.0, 5.0, 50.0]]])
    pred_phys = torch.tensor([[[1.0, 10.0, 20.0]]])
    norm_t = _to_norm(target_phys)
    norm_p = _to_norm(pred_phys)

    w = dict(mse_weight=1.0, mae_weight=0.5, csi_weight=0.3, extreme_weight=2.0)
    loss = _make_loss(**w)
    out = loss(norm_p, norm_t)
    expected = (
        w["mse_weight"] * out["mse"]
        + w["mae_weight"] * out["mae"]
        + w["csi_weight"] * out["csi"]
        + w["extreme_weight"] * out["extreme"]
    )
    assert torch.allclose(out["total"], expected, atol=1e-6)


def test_missing_norm_bounds_raises():
    """log_minmax without bounds should raise ValueError."""
    with pytest.raises(ValueError):
        GMCPExtremeLoss(normalize="log_minmax")


def test_mismatched_extreme_lengths_raise():
    """extreme_weights length must match extreme_thresholds."""
    with pytest.raises(ValueError):
        GMCPExtremeLoss(
            norm_min=0.0,
            norm_max=6.2,
            extreme_thresholds=[25.0, 50.0],
            extreme_weights=[2.0],
        )
