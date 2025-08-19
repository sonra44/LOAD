from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ActuatorCommand:
    name: str
    params: Dict[str, float]

@dataclass
class Proposal:
    proposal_id: str
    source_module_id: str
    confidence: float
    priority: float
    justification: str
    proposed_actions: List[ActuatorCommand]
