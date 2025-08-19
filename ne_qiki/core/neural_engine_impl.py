import time
import json
from typing import List
import asyncio
from core.interfaces import INeuralEngine
from shared.models import Proposal, ActuatorCommand
from models.ne_v1 import NE_v1
from core.feature_extractor import FeatureExtractor
from core.calibration import Calibration
from core.safety import SafetyShield
from core.metrics import INFERENCE_COUNT, INFERENCE_LATENCY
from core.nats_logger import NATSLogger
import torch

class NeuralEngineV1(INeuralEngine):
    def __init__(self, config):
        self.model = NE_v1(config['in_dim'], 64, config['num_classes'], config['param_dim'])
        self.extractor = FeatureExtractor(config['window'], config['in_dim'])
        self.calibrator = Calibration(config['calibration']['temperature'])
        self.safety = SafetyShield(config['action_catalog'])
        self.config = config
        self.nats_logger = NATSLogger()
        asyncio.create_task(self.nats_logger.connect())

    def generate_proposals(self, context) -> List[Proposal]:
        start = time.time()
        INFERENCE_COUNT.inc()
        try:
            tensor, mask = self.extractor.extract(context)
            logits, priority, params = self.model(tensor, mask)
            probs = self.calibrator.calibrate(logits)

            proposals = []
            top_k = torch.topk(probs, min(self.config['topk'], probs.size(-1)), dim=-1)
            for i in range(top_k.indices.size(1)):
                idx = top_k.indices[0, i].item()
                conf = top_k.values[0, i].item()
                if conf < self.config['min_confidence']:
                    continue
                action_meta = self.config['action_catalog']['actions'][idx]
                action_name = action_meta['name']
                actuator = ActuatorCommand(action_name, {})
                proposal = Proposal(
                    proposal_id=f"ne_{idx}",
                    source_module_id="NeuralEngineV1",
                    confidence=conf,
                    priority=priority.item(),
                    justification=f"Predicted by NE_v1 for {action_name}",
                    proposed_actions=[actuator]
                )
                proposals.append(proposal)

            # Логирование
            self._log_proposals(proposals, context)

            proposals = self.safety.validate(proposals, context.fsm_state, context.bios_status.ok)
        except Exception as e:
            print(f"[NE] Exception: {e}")
            proposals = []

        elapsed = (time.time() - start) * 1000
        INFERENCE_LATENCY.observe(elapsed / 1000.0)
        if elapsed > self.config['time_budget_ms']:
            print(f"[NE] Timeout: {elapsed:.2f} ms")
            return []
        return proposals

    def _log_proposals(self, proposals, context):
        log_data = {
            "timestamp": time.time(),
            "fsm_state": context.fsm_state,
            "bios_ok": context.bios_status.ok,
            "proposals": [
                {
                    "id": p.proposal_id,
                    "confidence": p.confidence,
                    "priority": p.priority,
                    "action": p.proposed_actions[0].name if p.proposed_actions else None
                }
                for p in proposals
            ]
        }
        print(json.dumps(log_data))
        # NATS logging
        asyncio.create_task(self.nats_logger.log_proposal(log_data))
