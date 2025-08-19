import torch
import numpy as np
from typing import Tuple

class FeatureExtractor:
    def __init__(self, window: int = 16, in_dim: int = 32):
        self.window = window
        self.in_dim = in_dim

    def extract(self, context) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Извлекает признаки из AgentContext и возвращает тензор и маску действий.
        """
        features = []

        # FSM one-hot (предполагаем 4 состояния)
        fsm_map = {"BOOTING": 0, "IDLE": 1, "ACTIVE": 2, "ERROR_STATE": 3}
        fsm_vec = np.zeros(4)
        fsm_vec[fsm_map.get(context.fsm_state, 3)] = 1.0
        features.extend(fsm_vec)

        # BIOS статусы: температура, питание, utilization
        bios = context.bios_status
        features.extend([
            min(1.0, max(0.0, getattr(bios, 'temperature', 0) / 100.0)),
            min(1.0, max(0.0, getattr(bios, 'power_draw', 0) / 100.0)),
            min(1.0, max(0.0, getattr(bios, 'utilization', 0) / 100.0))
        ])

        # Sensor data
        sensor = context.sensor_data or {}
        features.extend([
            min(1.0, max(0.0, sensor.get("distance", 0.0) / 10.0)),
            min(1.0, max(-1.0, sensor.get("velocity", 0.0) / 5.0)),
            min(1.0, max(-1.0, sensor.get("azimuth", 0.0) / 3.14)),
            min(1.0, max(0.0, sensor.get("hazard_score", 0.0)))
        ])

        # Action history
        hist = sensor.get("action_history", [])
        hist_vec = np.zeros(5)
        for i, act in enumerate(hist[-5:]):
            hist_vec[i] = min(1.0, max(0.0, act / 10.0))
        features.extend(hist_vec)

        # Паддинг до in_dim
        while len(features) < self.in_dim:
            features.append(0.0)

        # Тензор окна (T, in_dim)
        tensor = torch.tensor(features[:self.in_dim], dtype=torch.float32).unsqueeze(0).repeat(self.window, 1).unsqueeze(0)

        # Пример маски (допустим, первые 4 действия разрешены)
        mask = torch.tensor([True, True, True, True, False, False], dtype=torch.bool).unsqueeze(0)

        return tensor, mask
