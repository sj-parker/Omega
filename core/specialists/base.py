from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field

@dataclass
class SpecialistResult:
    """Result returned by a specialist."""
    data: Any
    confidence: float  # 0.0 - 1.0
    source: str        # Name of the specialist
    meta: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class SpecialistMetadata:
    """Metadata for specialist registration."""
    id: str
    name: str
    description: str
    keywords: List[str]
    version: str = "1.0.0"

class BaseSpecialist(ABC):
    """
    Abstract base class for all Specialists.
    
    A Specialist is a deterministic or semi-deterministic module 
    designed to handle specific types of intents (e.g., getting weather, 
    finding movies, calculating specific formulas).
    """
    
    def __init__(self):
        self._execution_count = 0
    
    @property
    @abstractmethod
    def metadata(self) -> SpecialistMetadata:
        """Return metadata for registration."""
        pass
        
    @abstractmethod
    def can_handle(self, query: str) -> float:
        """
        Check if this specialist can handle the query.
        Returns a score between 0.0 (no) and 1.0 (perfect match).
        
        This should be fast (keyword based + simple regex).
        """
        pass
        
    @abstractmethod
    async def execute(self, query: str, context: Optional[Any] = None) -> SpecialistResult:
        """
        Execute the specialist's task.
        """
        pass
        
    def help_text(self) -> str:
        """Return a string explaining what this specialist does (for the LLM/Router)."""
        return f"{self.metadata.name}: {self.metadata.description}"
