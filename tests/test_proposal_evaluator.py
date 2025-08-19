import unittest
from core.proposal_evaluator import ProposalEvaluator
from shared.models import Proposal, ActuatorCommand

class TestProposalEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = ProposalEvaluator()

    def test_evaluate(self):
        proposals = [
            Proposal("1", "test", 0.6, 0.9, "test", [ActuatorCommand("HOLD", {})]),
            Proposal("2", "test", 0.4, 0.8, "test", [ActuatorCommand("MOVE", {})]),
            Proposal("3", "test", 0.7, 0.7, "test", [ActuatorCommand("STOP", {})])
        ]
        result = self.evaluator.evaluate(proposals)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].proposal_id, "1")

if __name__ == "__main__":
    unittest.main()
