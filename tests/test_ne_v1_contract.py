import unittest
from core.neural_engine_impl import NeuralEngineV1
from shared.models import Proposal
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class MockBiosStatus:
    ok: bool = True
    temperature: float = 50.0
    power_draw: float = 50.0
    utilization: float = 50.0

@dataclass
class MockContext:
    bios_status: MockBiosStatus
    fsm_state: str
    sensor_data: Dict = None

class TestNeuralEngineV1(unittest.TestCase):
    def setUp(self):
        self.config = {
            "window": 16,
            "in_dim": 32,
            "num_classes": 6,
            "param_dim": 4,
            "topk": 3,
            "min_confidence": 0.55,
            "time_budget_ms": 8,
            "calibration": {"temperature": 1.2},
            "action_catalog": {
                "actions": [
                    {"name": "HOLD_POSITION", "params": {}},
                    {"name": "COOLING_BOOST", "params": {}},
                ]
            }
        }
        self.engine = NeuralEngineV1(self.config)

    def test_generate_proposals(self):
        context = MockContext(bios_status=MockBiosStatus(ok=True), fsm_state="ACTIVE")
        proposals = self.engine.generate_proposals(context)
        self.assertIsInstance(proposals, list)
        if proposals:
            self.assertIsInstance(proposals[0], Proposal)

    def test_error_state_returns_empty(self):
        context = MockContext(bios_status=MockBiosStatus(ok=True), fsm_state="ERROR_STATE")
        proposals = self.engine.generate_proposals(context)
        self.assertEqual(proposals, [])

if __name__ == "__main__":
    unittest.main()
