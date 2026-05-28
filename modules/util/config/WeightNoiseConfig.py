from typing import Any

from modules.util.config.BaseConfig import BaseConfig


class WeightNoiseConfig(BaseConfig):
    """Optional LoRA weight-noise regularizer.

    When enabled, Gaussian noise is added directly to LoRA parameter values
    after each optimizer update. Defaults keep existing training behavior
    unchanged.
    """

    enabled: bool
    mode: str
    sigma: float
    log_every: int

    def __init__(self, data: list[tuple[str, Any, type, bool]]):
        super().__init__(data)

    @staticmethod
    def default_values() -> 'WeightNoiseConfig':
        data = []
        data.append(("enabled", False, bool, False))
        data.append(("mode", "relative", str, False))
        data.append(("sigma", 0.0125, float, False))
        data.append(("log_every", 50, int, False))
        return WeightNoiseConfig(data)
