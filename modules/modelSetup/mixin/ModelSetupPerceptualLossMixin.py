from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from modules.perceptual.depth_consistency import (
    DifferentiableDepthEncoder,
    compute_depth_consistency_loss,
    dtype_from_name,
    gaussian_blur_2d,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType


class ModelSetupPerceptualLossMixin:
    """Shared x0-decoded perceptual auxiliary-loss implementation.

    Model-specific setup classes pass already-reconstructed VAE latents into
    `_add_perceptual_loss`. This keeps scheduler math close to each model's
    existing `predict` implementation and keeps the trainer loop unchanged.
    """

    def __init__(self):
        super().__init__()
        self._perceptual_depth_encoder: DifferentiableDepthEncoder | None = None
        self._perceptual_depth_key: tuple[Any, ...] | None = None
        self._last_perceptual_loss: float | None = None
        self._last_perceptual_decoded_l1_loss: float | None = None
        self._last_perceptual_decoded_mse_loss: float | None = None
        self._last_perceptual_edge_loss: float | None = None
        self._last_perceptual_depth_loss: float | None = None
        self._last_perceptual_depth_ssi: float | None = None
        self._last_perceptual_depth_grad: float | None = None

    def _perceptual_enabled(self, config: TrainConfig) -> bool:
        cfg = getattr(config, "perceptual_loss", None)
        return bool(cfg is not None and cfg.enabled and cfg.any_weight_enabled())

    def _reset_perceptual_logs(self):
        self._last_perceptual_loss = None
        self._last_perceptual_decoded_l1_loss = None
        self._last_perceptual_decoded_mse_loss = None
        self._last_perceptual_edge_loss = None
        self._last_perceptual_depth_loss = None
        self._last_perceptual_depth_ssi = None
        self._last_perceptual_depth_grad = None

    def _get_perceptual_depth_encoder(self, cfg, device: torch.device) -> DifferentiableDepthEncoder:
        key = (
            cfg.depth_model_id,
            int(cfg.depth_input_size),
            str(cfg.depth_dtype),
            bool(cfg.depth_grad_checkpoint),
            str(device),
        )
        if self._perceptual_depth_encoder is None or self._perceptual_depth_key != key:
            encoder = DifferentiableDepthEncoder(
                model_id=cfg.depth_model_id,
                input_size=int(cfg.depth_input_size),
                dtype=dtype_from_name(cfg.depth_dtype),
                device=device,
                grad_checkpoint=bool(cfg.depth_grad_checkpoint),
            )
            encoder.eval()
            self._perceptual_depth_encoder = encoder
            self._perceptual_depth_key = key
        return self._perceptual_depth_encoder

    def _decode_vae_latents(
        self,
        model,
        latents: Tensor,
        chunk_size: int,
        *,
        grad: bool,
    ) -> Tensor:
        if latents.dim() not in (4, 5) or (latents.dim() == 5 and latents.shape[2] != 1):
            raise ValueError(
                f"perceptual_loss only supports 4D image latents or singleton-frame 5D latents, "
                f"got {tuple(latents.shape)}"
            )

        vae = getattr(model, "vae", None)
        if vae is None:
            raise ValueError("perceptual_loss requires model.vae")

        # VAE normally lives on temp_device after latent caching. Perceptual
        # loss explicitly needs differentiable decode on train_device.
        try:
            param = next(vae.parameters())
            if param.device != self.train_device:
                model.vae_to(self.train_device)
        except StopIteration:
            pass
        vae.eval()

        dtype = next(vae.parameters()).dtype
        chunks = [latents] if chunk_size <= 0 else list(torch.split(latents, max(1, int(chunk_size)), dim=0))
        decoded = []
        ctx = contextlib.nullcontext() if grad else torch.no_grad()
        with ctx:
            for chunk in chunks:
                # diffusers VAEs generally decode in their own parameter dtype.
                out = vae.decode(chunk.to(device=self.train_device, dtype=dtype))
                out = out.sample if hasattr(out, "sample") else out
                if out.dim() == 5:
                    if out.shape[2] == 1:       # (B, C, 1, H, W)
                        out = out[:, :, 0]
                    elif out.shape[1] == 1:     # (B, 1, C, H, W)
                        out = out[:, 0]
                    else:
                        raise ValueError(
                            f"perceptual_loss cannot reduce decoded 5D pixels with shape {tuple(out.shape)}"
                        )
                decoded.append(out.float())
        return torch.cat(decoded, dim=0)

    @staticmethod
    def _sobel_edges(pixels: Tensor) -> Tensor:
        if pixels.shape[1] == 1:
            gray = pixels
        else:
            # Works for either [-1, 1] or [0, 1]; edge deltas are scale-stable
            # up to the configured weight.
            r, g, b = pixels[:, 0:1], pixels[:, 1:2], pixels[:, 2:3]
            gray = r * 0.299 + g * 0.587 + b * 0.114

        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=pixels.device,
            dtype=pixels.dtype,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=pixels.device,
            dtype=pixels.dtype,
        ).view(1, 1, 3, 3)
        gray = F.pad(gray, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(gray, kx)
        gy = F.conv2d(gray, ky)
        return torch.sqrt(gx * gx + gy * gy + 1e-12)

    def _perceptual_active_mask(self, batch: dict, timestep: Tensor, num_train_timesteps: int, config: TrainConfig) -> Tensor:
        cfg = config.perceptual_loss
        denom = max(1, num_train_timesteps - 1)
        t_ratio = timestep.float() / float(denom)
        active = (t_ratio >= float(cfg.min_t)) & (t_ratio <= float(cfg.max_t))

        concept_type = batch.get("concept_type")
        if concept_type is not None:
            not_prior = []
            for item in concept_type:
                try:
                    if torch.is_tensor(item):
                        item = item.item()
                    not_prior.append(ConceptType(item) != ConceptType.PRIOR_PREDICTION)
                except Exception:
                    not_prior.append(True)
            active = active & torch.tensor(not_prior, device=active.device, dtype=torch.bool)
        return active

    def _weighted_sample_mean(self, values: Tensor, batch: dict, active: Tensor) -> Tensor:
        weights = batch.get("loss_weight")
        if weights is None:
            return values.mean()
        weights = weights.to(device=values.device, dtype=values.dtype)[active]
        denom = weights.sum().clamp_min(1e-8)
        return (values * weights).sum() / denom

    def _add_perceptual_loss(
        self,
        *,
        model,
        batch: dict,
        data: dict,
        config: TrainConfig,
        base_loss: Tensor,
        predicted_latent_image: Tensor,
        target_latent_image: Tensor,
        num_train_timesteps: int,
    ) -> Tensor:
        self._reset_perceptual_logs()
        if not self._perceptual_enabled(config):
            return base_loss
        if predicted_latent_image is None or target_latent_image is None:
            return base_loss
        valid_dims = (
            (predicted_latent_image.dim() == 4 and target_latent_image.dim() == 4)
            or (
                predicted_latent_image.dim() == 5
                and target_latent_image.dim() == 5
                and predicted_latent_image.shape[2] == 1
                and target_latent_image.shape[2] == 1
            )
        )
        if not valid_dims:
            return base_loss
        if "timestep" not in data:
            return base_loss

        cfg = config.perceptual_loss
        active = self._perceptual_active_mask(batch, data["timestep"], num_train_timesteps, config)
        if not active.any():
            return base_loss

        pred_latents = predicted_latent_image[active]
        target_latents = target_latent_image.detach()[active]
        pred_pixels = self._decode_vae_latents(model, pred_latents, cfg.decode_chunk_size, grad=True).clamp(-1.0, 1.0)
        target_pixels = self._decode_vae_latents(model, target_latents, cfg.decode_chunk_size, grad=False).clamp(-1.0, 1.0)

        total = base_loss.new_zeros(())

        if cfg.decoded_l1_weight > 0.0:
            per_sample = F.l1_loss(pred_pixels.float(), target_pixels.float(), reduction="none").flatten(1).mean(1)
            raw = self._weighted_sample_mean(per_sample, batch, active)
            total = total + raw * float(cfg.decoded_l1_weight)
            self._last_perceptual_decoded_l1_loss = raw.detach().item()

        if cfg.decoded_mse_weight > 0.0:
            per_sample = F.mse_loss(pred_pixels.float(), target_pixels.float(), reduction="none").flatten(1).mean(1)
            raw = self._weighted_sample_mean(per_sample, batch, active)
            total = total + raw * float(cfg.decoded_mse_weight)
            self._last_perceptual_decoded_mse_loss = raw.detach().item()

        if cfg.edge_weight > 0.0:
            pred_edges = self._sobel_edges(pred_pixels.float())
            target_edges = self._sobel_edges(target_pixels.float())
            per_sample = F.l1_loss(pred_edges, target_edges, reduction="none").flatten(1).mean(1)
            raw = self._weighted_sample_mean(per_sample, batch, active)
            total = total + raw * float(cfg.edge_weight)
            self._last_perceptual_edge_loss = raw.detach().item()

        if cfg.depth_weight > 0.0:
            # Depth-Anything gets decoded VAE pixels in [-1, 1]. Optional blur mirrors the
            # ai-toolkit-perceptual knob used to suppress texture leakage.
            pred_for_depth = gaussian_blur_2d(pred_pixels.float(), float(cfg.depth_pixel_blur_sigma))
            target_for_depth = gaussian_blur_2d(target_pixels.float(), float(cfg.depth_pixel_blur_sigma))
            encoder = self._get_perceptual_depth_encoder(cfg, self.train_device)
            with torch.no_grad():
                target_depth = encoder(target_for_depth, input_range="minus1_1").detach()
            depth_per_sample, depth_ssi_per_sample, depth_grad_per_sample, _, _ = compute_depth_consistency_loss(
                encoder=encoder,
                x0_pixels=pred_for_depth,
                gt_depth=target_depth,
                mask=None,
                ssi_weight=float(cfg.depth_ssi_weight),
                grad_weight=float(cfg.depth_grad_weight),
                grad_scales=int(cfg.depth_grad_scales),
                input_range="minus1_1",
                reduction="none",
            )
            depth_loss = self._weighted_sample_mean(depth_per_sample, batch, active)
            total = total + depth_loss * float(cfg.depth_weight)
            self._last_perceptual_depth_loss = depth_loss.detach().item()
            self._last_perceptual_depth_ssi = self._weighted_sample_mean(
                depth_ssi_per_sample, batch, active
            ).detach().item()
            self._last_perceptual_depth_grad = self._weighted_sample_mean(
                depth_grad_per_sample, batch, active
            ).detach().item()

        if total.detach().abs().item() == 0.0:
            return base_loss

        self._last_perceptual_loss = total.detach().item()
        return base_loss + total
