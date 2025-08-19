from abc import ABC, abstractmethod
from typing import List, Any
from shared.models import Proposal

class INeuralEngine(ABC):
    """
    Abstract interface for the Neural Engine, responsible for generating proposals based on ML models.
    """

    @abstractmethod
    def generate_proposals(self, context: Any) -> List[Proposal]:
        """Generates a list of proposals based on the current agent context using ML models."""
        pass
