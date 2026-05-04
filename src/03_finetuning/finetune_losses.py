from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def mse_loss(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((preds - target) ** 2).mean()


def pearson_corr(preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    preds_flat = preds.flatten(start_dim=1)
    target_flat = target.flatten(start_dim=1)
    preds_centered = preds_flat - preds_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    numerator = (preds_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt(
        (preds_centered.square().sum(dim=1) + eps)
        * (target_centered.square().sum(dim=1) + eps)
    )
    return numerator / denominator


def pearson_loss(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - pearson_corr(preds, target).mean()


def target_weighted_mse(
    preds: torch.Tensor,
    target: torch.Tensor,
    *,
    peak_weight: float = 4.0,
    peak_weight_cap: float = 10.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    target_scale = target.detach()
    max_per_track = target_scale.amax(dim=-1, keepdim=True).clamp_min(eps)
    normalized_target = (target_scale / max_per_track).clamp_min(0.0)
    weights = 1.0 + peak_weight * normalized_target
    if peak_weight_cap > 0:
        weights = weights.clamp(max=peak_weight_cap)
    return (weights * (preds - target).square()).mean()


def _avg_pool_track(track: torch.Tensor, bin_size: int) -> torch.Tensor:
    if bin_size <= 1:
        return track
    length = track.shape[-1]
    usable_length = (length // bin_size) * bin_size
    if usable_length <= 0:
        raise ValueError(f"bin_size={bin_size} is too large for track length={length}")
    if usable_length != length:
        track = track[..., :usable_length]
    return F.avg_pool1d(track, kernel_size=bin_size, stride=bin_size)


def multiscale_mse(
    preds: torch.Tensor,
    target: torch.Tensor,
    *,
    bins: list[int],
) -> torch.Tensor:
    losses = []
    for bin_size in bins:
        pooled_preds = _avg_pool_track(preds, int(bin_size))
        pooled_target = _avg_pool_track(target, int(bin_size))
        losses.append(mse_loss(pooled_preds, pooled_target))
    if not losses:
        return preds.new_tensor(0.0)
    return torch.stack(losses).mean()


def poisson_nll_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    rate = F.softplus(preds) + eps
    clipped_target = target.clamp_min(0.0)
    return (rate - clipped_target * torch.log(rate)).mean()


def compute_configured_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
    confidence_weight: float = 0.0,
    loss_cfg: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = loss_cfg or {}
    name = cfg.get("name", "mse")

    base_mse = mse_loss(preds, target)
    components: dict[str, torch.Tensor] = {"mse": base_mse}

    if name == "mse":
        total = base_mse
    elif name == "weighted_mse":
        weighted = target_weighted_mse(
            preds,
            target,
            peak_weight=float(cfg.get("peak_weight", 4.0)),
            peak_weight_cap=float(cfg.get("peak_weight_cap", 10.0)),
        )
        components["weighted_mse"] = weighted
        total = weighted
    elif name in {"weighted_multiscale_mse", "weighted_multiscale_mse_pearson"}:
        weighted = target_weighted_mse(
            preds,
            target,
            peak_weight=float(cfg.get("peak_weight", 4.0)),
            peak_weight_cap=float(cfg.get("peak_weight_cap", 10.0)),
        )
        multiscale = multiscale_mse(
            preds,
            target,
            bins=[int(bin_size) for bin_size in cfg.get("multiscale_bins", [8, 32, 128])],
        )
        components["weighted_mse"] = weighted
        components["multiscale_mse"] = multiscale
        total = weighted + float(cfg.get("multiscale_weight", 0.25)) * multiscale
        if name == "weighted_multiscale_mse_pearson":
            corr_loss = pearson_loss(preds, target)
            components["pearson_loss"] = corr_loss
            total = total + float(cfg.get("pearson_weight", 0.05)) * corr_loss
    elif name == "poisson_nll":
        poisson = poisson_nll_loss(preds, target)
        components["poisson_nll"] = poisson
        total = poisson
    else:
        raise ValueError(f"Unknown loss.name={name!r}")

    if confidence is not None and confidence_weight > 0:
        rmse = torch.sqrt((preds - target).square().detach() + 1e-8)
        confidence_mse = mse_loss(confidence, rmse)
        components["confidence_mse"] = confidence_mse
        total = total + confidence_weight * confidence_mse

    components["loss"] = total
    components["pearson"] = pearson_corr(preds.detach(), target.detach()).mean()
    return total, components
