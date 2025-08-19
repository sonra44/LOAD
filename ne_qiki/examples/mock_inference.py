from core.neural_engine_impl import NeuralEngineV1
from dataclasses import dataclass

@dataclass
class MockBiosStatus:
    ok: bool

@dataclass
class MockContext:
    bios_status: MockBiosStatus
    fsm_state: str

if __name__ == "__main__":
    config = {
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

    engine = NeuralEngineV1(config)
    context = MockContext(bios_status=MockBiosStatus(ok=True), fsm_state="ACTIVE")
    proposals = engine.generate_proposals(context)
    for p in proposals:
        print(p)
