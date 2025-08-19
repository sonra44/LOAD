import unittest
from core.safety import SafetyShield
from shared.models import Proposal, ActuatorCommand

class TestSafetyShield(unittest.TestCase):
    def setUp(self):
        self.catalog = {
            "actions": [
                {"name": "HOLD_POSITION", "params": {}},
                {"name": "THROTTLE_UP", "params": {"level": [0.0, 1.0]}}
            ]
        }
        self.shield = SafetyShield(self.catalog)

    def test_validate_ok(self):
        proposal = Proposal(
            proposal_id="test",
            source_module_id="test",
            confidence=0.9,
            priority=0.8,
            justification="test",
            proposed_actions=[ActuatorCommand("HOLD_POSITION", {})]
        )
        result = self.shield.validate([proposal], "ACTIVE", True)
        self.assertEqual(len(result), 1)

    def test_validate_error_state(self):
        proposal = Proposal(
            proposal_id="test",
            source_module_id="test",
            confidence=0.9,
            priority=0.8,
            justification="test",
            proposed_actions=[ActuatorCommand("HOLD_POSITION", {})]
        )
        result = self.shield.validate([proposal], "ERROR_STATE", True)
        self.assertEqual(len(result), 0)

if __name__ == "__main__":
    unittest.main()
