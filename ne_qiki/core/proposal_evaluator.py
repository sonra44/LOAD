from shared.models import Proposal
from typing import List

class ProposalEvaluator:
    def evaluate(self, proposals: List[Proposal]) -> List[Proposal]:
        # Фильтр по confidence
        filtered = [p for p in proposals if p.confidence >= 0.5]
        # Сортировка по priority + confidence
        filtered.sort(key=lambda x: (x.priority, x.confidence), reverse=True)
        return filtered[:3]
