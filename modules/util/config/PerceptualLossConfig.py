from typing import Any

from modules.util.config.BaseConfig import BaseConfig


class PerceptualLossConfig(BaseConfig):
    """Optional x0-decoded auxiliary losses for diffusion/flow training.

    The config is intentionally flat so existing JSON presets can opt in by
    adding a single `perceptual_loss` object. All weights default to zero, so
    adding this config changes no existing training run unless explicitly
    enabled and weighted.
    """

    enabled: bool
    min_t: float
    max_t: float
    decode_chunk_size: int

    decoded_l1_weight: float
    decoded_mse_weight: float
    edge_weight: float

    depth_weight: float
    depth_model_id: str
    depth_input_size: int
    depth_dtype: str
    depth_grad_checkpoint: bool
    depth_ssi_weight: float
    depth_grad_weight: float
    depth_grad_scales: int
    depth_pixel_blur_sigma: float

    def __init__(self, data: list[tuple[str, Any, type, bool]]):
        super().__init__(data)

    def any_weight_enabled(self) -> bool:
        return (
            self.decoded_l1_weight > 0.0
            or self.decoded_mse_weight > 0.0
            or self.edge_weight > 0.0
            or self.depth_weight > 0.0
        )

    @staticmethod
    def default_values() -> 'PerceptualLossConfig':
        data = []

        # Master switch and shared timestep gate. t is normalized to [0, 1].
        data.append(("enabled", False, bool, False))
        data.append(("min_t", 0.0, float, False))
        data.append(("max_t", 1.0, float, False))
        data.append(("decode_chunk_size", 1, int, False))

        # Lightweight decoded-image anchors. These require no extra models.
        data.append(("decoded_l1_weight", 0.0, float, False))
        data.append(("decoded_mse_weight", 0.0, float, False))
        data.append(("edge_weight", 0.0, float, False))

        # Depth-Anything-V2 anchor, ported from ai-toolkit-perceptual's
        # differentiable depth-consistency path.
        data.append(("depth_weight", 0.0, float, False))
        data.append(("depth_model_id", "depth-anything/Depth-Anything-V2-Small-hf", str, False))
        data.append(("depth_input_size", 518, int, False))
        data.append(("depth_dtype", "bfloat16", str, False))
        data.append(("depth_grad_checkpoint", True, bool, False))
        data.append(("depth_ssi_weight", 1.0, float, False))
        data.append(("depth_grad_weight", 0.5, float, False))
        data.append(("depth_grad_scales", 4, int, False))
        data.append(("depth_pixel_blur_sigma", 0.0, float, False))

        return PerceptualLossConfig(data)
