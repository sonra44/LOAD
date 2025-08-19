import torch

class Calibration:
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits / self.temperature, dim=-1)
