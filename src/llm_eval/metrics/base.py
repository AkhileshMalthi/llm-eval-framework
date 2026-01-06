from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Optional

class MetricResult(BaseModel):
    name: str
    score: float
    reasoning: Optional[str] = None

class BaseMetric(ABC):
    @abstractmethod
    def compute(self, query: str, response: str, contexts: Optional[list[str]] = None, reference: Optional[str] = None) -> MetricResult:
        """All metrics must implement this exact method signature."""
        pass