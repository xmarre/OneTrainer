from typing import Any

from modules.util.config.BaseConfig import BaseConfig


class GradientNoiseConfig(BaseConfig):
    """Optional LoRA gradient-noise regularizer.

    When enabled, Gaussian noise is added to LoRA gradients after gradient
    clipping and before optimizer.step(). Defaults keep existing training
    behavior unchanged.
    """

    enabled: bool
    mode: str
    sigma: float
    eta: float
    gamma: float
    log_every: int

    def __init__(self, data: list[tuple[str, Any, type, bool]]):
        super().__init__(data)

    @staticmethod
    def default_values() -> 'GradientNoiseConfig':
        data = []
        data.append(("enabled", False, bool, False))
        data.append(("mode", "neelakantan", str, False))
        data.append(("sigma", 1e-3, float, False))
        data.append(("eta", 0.01, float, False))
        data.append(("gamma", 0.55, float, False))
        data.append(("log_every", 50, int, False))
        return GradientNoiseConfig(data)
