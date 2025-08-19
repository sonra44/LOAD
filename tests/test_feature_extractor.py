import unittest
import torch
from core.feature_extractor import FeatureExtractor
from dataclasses import dataclass

@dataclass
class MockBiosStatus:
    temperature: float = 50.0
    power_draw: float = 50.0
    utilization: float = 50.0
    ok: bool = True

@dataclass
class MockContext:
    bios_status: MockBiosStatus
    fsm_state: str
    sensor_data: dict = None

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor(window=16, in_dim=32)

    def test_extract_shape(self):
        context = MockContext(
            bios_status=MockBiosStatus(),
            fsm_state="ACTIVE",
            sensor_data={"distance": 5.0, "velocity": 1.0, "azimuth": 0.5, "hazard_score": 0.3}
        )
        tensor, mask = self.extractor.extract(context)
        self.assertEqual(tensor.shape, (1, 16, 32))
        self.assertEqual(mask.shape, (1, 6))

if __name__ == "__main__":
    unittest.main()
