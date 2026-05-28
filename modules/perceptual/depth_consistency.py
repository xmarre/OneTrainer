"""Differentiable perceptual/depth anchors adapted for OneTrainer.

The depth-consistency implementation is based on BuffaloBuffaloBuffaloBuffalo's
ai-toolkit-perceptual fork. It keeps the important invariant from that code:
the frozen depth perceptor must stay in the autograd graph for x0 predictions,
while target depth is computed under no_grad and detached.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).lower().strip()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported perceptual_loss.depth_dtype: {name!r}")


def gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma is None or sigma <= 0:
        return x
    import math

    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    k = 2 * radius + 1
    coords = torch.arange(k, device=x.device, dtype=torch.float32) - radius
    g = torch.exp(-(coords ** 2) / (2.0 * float(sigma) * float(sigma)))
    g = g / g.sum()
    kernel_1d = g.view(1, 1, 1, k)
    kernel_2d = (kernel_1d * kernel_1d.transpose(-1, -2)).to(x.dtype)
    channels = x.shape[1]
    kernel = kernel_2d.expand(channels, 1, k, k).contiguous()
    x_padded = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(x_padded, kernel, groups=channels)


class DifferentiableDepthEncoder(nn.Module):
    """Frozen Depth-Anything-V2 perceptor with tensor-only preprocessing."""

    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        input_size: int = 518,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        grad_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        from transformers import DepthAnythingForDepthEstimation

        self.model = DepthAnythingForDepthEstimation.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        if grad_checkpoint:
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.model.gradient_checkpointing_enable()

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.input_size = input_size
        if device is not None:
            self.to(device)

    def _aspect_preserving_hw(self, height: int, width: int) -> Tuple[int, int]:
        if height >= width:
            new_h = self.input_size
            new_w = max(14, int(round(width * self.input_size / height / 14)) * 14)
        else:
            new_w = self.input_size
            new_h = max(14, int(round(height * self.input_size / width / 14)) * 14)
        return new_h, new_w

    def preprocess(self, pixels: torch.Tensor, input_range: Literal["0_1", "minus1_1"] = "0_1") -> torch.Tensor:
        if input_range == "minus1_1":
            pixels = (pixels + 1.0) * 0.5
        elif input_range != "0_1":
            raise ValueError(f"Unsupported depth input range: {input_range!r}")
        pixels = pixels.clamp(0.0, 1.0)
        _, _, height, width = pixels.shape
        new_h, new_w = self._aspect_preserving_hw(height, width)
        x = F.interpolate(
            pixels,
            size=(new_h, new_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        x = x.to(self.mean.dtype)
        x = (x - self.mean) / self.std
        return x.to(next(self.model.parameters()).dtype)

    def forward(self, pixels: torch.Tensor, input_range: Literal["0_1", "minus1_1"] = "0_1") -> torch.Tensor:
        x = self.preprocess(pixels, input_range=input_range)
        out = self.model(pixel_values=x)
        return out.predicted_depth.float()


def ssi_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: Literal["mean", "none"] = "mean",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    if mask is None:
        mask = torch.ones_like(pred)
    elif mask.dim() == 2:
        mask = mask.unsqueeze(0)

    p = pred.flatten(1)
    g = target.flatten(1)
    m = mask.flatten(1).float()
    n = m.sum(dim=1).clamp_min(1.0)
    mean_p = (p * m).sum(1) / n
    mean_g = (g * m).sum(1) / n
    var_p = (p * p * m).sum(1) / n - mean_p * mean_p
    cov_pg = (p * g * m).sum(1) / n - mean_p * mean_g
    scale = cov_pg / var_p.clamp_min(1e-6)
    shift = mean_g - scale * mean_p
    aligned = scale.view(-1, 1, 1) * pred + shift.view(-1, 1, 1)
    diff = (aligned - target).abs() * mask
    loss = diff.flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1.0)
    if reduction == "mean":
        loss = loss.mean()
    elif reduction != "none":
        raise ValueError(f"Unsupported reduction: {reduction!r}")
    return loss, scale.detach(), shift.detach()


def multiscale_grad_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scales: int = 4,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    scales = int(scales)
    if scales < 1:
        raise ValueError("grad_scales must be >= 1")

    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    if mask is None:
        mask = torch.ones_like(pred)
    elif mask.dim() == 2:
        mask = mask.unsqueeze(0)

    loss = pred.new_zeros((pred.shape[0],))
    p, g, m = pred, target, mask.float()
    for scale_idx in range(scales):
        if scale_idx > 0:
            p = F.avg_pool2d(p.unsqueeze(1), 2).squeeze(1)
            g = F.avg_pool2d(g.unsqueeze(1), 2).squeeze(1)
            m = F.avg_pool2d(m.unsqueeze(1), 2).squeeze(1)
        diff = p - g
        mx = m[:, :, 1:] * m[:, :, :-1]
        my = m[:, 1:, :] * m[:, :-1, :]
        dx = (diff[:, :, 1:] - diff[:, :, :-1]).abs() * mx
        dy = (diff[:, 1:, :] - diff[:, :-1, :]).abs() * my
        loss = loss + (
            dx.flatten(1).sum(1) / mx.flatten(1).sum(1).clamp_min(1.0)
        ) + (
            dy.flatten(1).sum(1) / my.flatten(1).sum(1).clamp_min(1.0)
        )
    loss = loss / scales
    if reduction == "mean":
        return loss.mean()
    if reduction == "none":
        return loss
    raise ValueError(f"Unsupported reduction: {reduction!r}")


def compute_depth_consistency_loss(
    encoder: DifferentiableDepthEncoder,
    x0_pixels: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    ssi_weight: float = 1.0,
    grad_weight: float = 0.5,
    grad_scales: int = 4,
    input_range: Literal["0_1", "minus1_1"] = "minus1_1",
    reduction: Literal["mean", "none"] = "mean",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    d_pred = encoder(x0_pixels, input_range=input_range)

    target = gt_depth
    if target.dim() == 2:
        target = target.unsqueeze(0)
    if target.shape[-2:] != d_pred.shape[-2:]:
        target = F.interpolate(
            target.unsqueeze(1).float(),
            size=d_pred.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    target = target.to(device=d_pred.device, dtype=torch.float32).detach()

    depth_mask = None
    if mask is not None:
        if mask.dim() == 4:
            mask = mask.squeeze(1)
        depth_mask = F.interpolate(
            mask.unsqueeze(1).float(),
            size=d_pred.shape[-2:],
            mode="nearest",
        ).squeeze(1).to(d_pred.device)

    ssi, _, _ = ssi_l1(d_pred, target, depth_mask, reduction=reduction)
    grad = multiscale_grad_loss(d_pred, target, depth_mask, scales=grad_scales, reduction=reduction)
    loss = ssi * ssi_weight + grad * grad_weight
    return loss, ssi.detach(), grad.detach(), d_pred.detach(), target.detach()
