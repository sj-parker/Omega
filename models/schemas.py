# Cognitive LLM System
# Models and Schemas

from dataclasses import dataclass, field
from typing import Optional, Literal, Any
from datetime import datetime
from enum import Enum
import json
import uuid


# ============================================================
# SECURITY / IDENTITY GATE
# ============================================================

@dataclass
class UserIdentity:
    """Результат работы Security Gate."""
    user_id: str
    trust_level: float  # 0.0 - 1.0
    risk_flag: bool = False
    history_length: int = 0
    consistency_score: float = 1.0
    anomaly_detected: bool = False

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "trust_level": self.trust_level,
            "risk_flag": self.risk_flag,
            "history_length": self.history_length,
            "consistency_score": self.consistency_score,
            "anomaly_detected": self.anomaly_detected,
        }


# ============================================================
# DECISION DEPTH (Fast / Medium / Deep paths)
# ============================================================

class DecisionDepth(Enum):
    FAST = "fast"       # 1 LLM call
    MEDIUM = "medium"   # LLM + memory
    DEEP = "deep"       # experts + critic


@dataclass
class DecisionObject:
    """Результат решения ОМ."""
    action: Literal["respond", "ask", "ignore"]
    confidence: float  # 0.0 - 1.0
    depth_used: DecisionDepth = DecisionDepth.FAST
    cost: dict = field(default_factory=lambda: {"time_ms": 0, "experts_used": 0})
    policy_snapshot: dict = field(default_factory=dict)
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "confidence": self.confidence,
            "depth_used": self.depth_used.value,
            "cost": self.cost,
            "policy_snapshot": self.policy_snapshot,
            "reasoning": self.reasoning,
        }


# ============================================================
# POLICY SPACE (bounded parameters for homeostasis)
# ============================================================

@dataclass
class PolicySpace:
    """Ограниченный набор 'рычагов' для гомеостаза."""
    fast_path_bias: float = 0.65        # Preference for fast path
    expert_call_threshold: float = 0.4  # Confidence below this -> call experts
    creative_range: tuple = (0.6, 0.9)  # Temperature range for creative expert
    memory_write_threshold: float = 0.3 # Only write to memory if importance > this
    
    # Homeostasis bounds (wide initially)
    confidence_mean_range: tuple = (0.4, 0.9)
    expert_disagreement_range: tuple = (0.0, 0.6)
    
    def to_dict(self) -> dict:
        return {
            "fast_path_bias": self.fast_path_bias,
            "expert_call_threshold": self.expert_call_threshold,
            "creative_range": list(self.creative_range),
            "memory_write_threshold": self.memory_write_threshold,
            "confidence_mean_range": list(self.confidence_mean_range),
            "expert_disagreement_range": list(self.expert_disagreement_range),
        }

    def apply_update(self, update: dict) -> None:
        """Apply homeostatic adjustments."""
        for key, delta in update.items():
            if hasattr(self, key):
                current = getattr(self, key)
                if isinstance(current, (int, float)):
                    setattr(self, key, current + delta)


# ============================================================
# CONTEXT & MEMORY
# ============================================================

@dataclass
class ContextEvent:
    """Единичное событие в контексте."""
    timestamp: datetime
    event_type: str  # user_input, system_response, expert_output, etc.
    content: Any
    importance: float = 0.5
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "content": self.content,
            "importance": self.importance,
        }


@dataclass
class ContextSlice:
    """Срез контекста для ОМ (после Memory Gate)."""
    user_input: str
    user_identity: UserIdentity
    recent_events: list[ContextEvent] = field(default_factory=list)
    active_goal: Optional[str] = None
    emotional_state: str = "neutral"
    system_mode: str = "normal"
    
    def to_dict(self) -> dict:
        return {
            "user_input": self.user_input,
            "user_identity": self.user_identity.to_dict(),
            "recent_events": [e.to_dict() for e in self.recent_events],
            "active_goal": self.active_goal,
            "emotional_state": self.emotional_state,
            "system_mode": self.system_mode,
        }


# ============================================================
# EXPERIENCE (3-level representation)
# ============================================================

@dataclass
class RawTrace:
    """Level 1: Raw storage - полная трасса мышления."""
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    user_input: str = ""
    context_snapshot: dict = field(default_factory=dict)
    expert_outputs: list[dict] = field(default_factory=list)
    critic_output: dict = field(default_factory=dict)
    decision: dict = field(default_factory=dict)
    final_response: str = ""
    user_reaction: Optional[str] = None  # follow-up, satisfaction, etc.
    
    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp.isoformat(),
            "user_input": self.user_input,
            "context_snapshot": self.context_snapshot,
            "expert_outputs": self.expert_outputs,
            "critic_output": self.critic_output,
            "decision": self.decision,
            "final_response": self.final_response,
            "user_reaction": self.user_reaction,
        }


@dataclass
class EpisodeSummary:
    """Level 2: For reflection - сжатое описание эпизода."""
    episode_id: str
    summary: str  # e.g., "High disagreement → low confidence → user repeat"
    key_metrics: dict = field(default_factory=dict)  # confidence, disagreement, cost
    outcome: str = ""  # success, partial, failure
    
    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "summary": self.summary,
            "key_metrics": self.key_metrics,
            "outcome": self.outcome,
        }


@dataclass
class ExtractedPattern:
    """Level 3: For policy - паттерн/принцип."""
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""  # e.g., "Need clarification earlier"
    source_episodes: list[str] = field(default_factory=list)
    confidence: float = 0.5
    times_validated: int = 0
    
    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "source_episodes": self.source_episodes,
            "confidence": self.confidence,
            "times_validated": self.times_validated,
        }


# ============================================================
# EXPERT OUTPUTS
# ============================================================

@dataclass
class ExpertResponse:
    """Ответ одного эксперта."""
    expert_type: Literal["neutral", "creative", "conservative"]
    response: str
    confidence: float
    temperature_used: float
    reasoning: str = ""
    
    def to_dict(self) -> dict:
        return {
            "expert_type": self.expert_type,
            "response": self.response,
            "confidence": self.confidence,
            "temperature_used": self.temperature_used,
            "reasoning": self.reasoning,
        }


@dataclass
class CriticAnalysis:
    """Анализ критика."""
    inconsistencies: list[str] = field(default_factory=list)
    hallucination_risk: float = 0.0
    recommended_response: Optional[str] = None
    disagreement_score: float = 0.0  # Between experts
    
    def to_dict(self) -> dict:
        return {
            "inconsistencies": self.inconsistencies,
            "hallucination_risk": self.hallucination_risk,
            "recommended_response": self.recommended_response,
            "disagreement_score": self.disagreement_score,
        }


# ============================================================
# HOMEOSTASIS
# ============================================================

@dataclass
class HomeostasisMetrics:
    """Метрики для гомеостатического контроллера."""
    avg_confidence: float = 0.7
    avg_expert_disagreement: float = 0.2
    avg_decision_cost_ms: float = 500
    repeat_question_rate: float = 0.1
    risk_level: float = 0.1
    
    def to_dict(self) -> dict:
        return {
            "avg_confidence": self.avg_confidence,
            "avg_expert_disagreement": self.avg_expert_disagreement,
            "avg_decision_cost_ms": self.avg_decision_cost_ms,
            "repeat_question_rate": self.repeat_question_rate,
            "risk_level": self.risk_level,
        }


@dataclass
class PolicyUpdate:
    """Корректировка политики от Homeostasis."""
    updates: dict = field(default_factory=dict)
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "updates": self.updates,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }
