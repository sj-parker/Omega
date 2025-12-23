# Security / Identity Gate (Gatekeeper)
# Behavioral trust based on history, consistency, anomalies

from datetime import datetime
from typing import Optional
import hashlib

import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])

from models.schemas import UserIdentity


class UserHistoryStore:
    """Simple in-memory store for user interaction history."""
    
    def __init__(self):
        self._history: dict[str, list[dict]] = {}
    
    def add_interaction(self, user_id: str, content: str, timestamp: datetime):
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append({
            "content": content,
            "timestamp": timestamp.isoformat(),
            "content_hash": hashlib.md5(content.encode()).hexdigest()[:8]
        })
    
    def get_history(self, user_id: str) -> list[dict]:
        return self._history.get(user_id, [])
    
    def get_history_length(self, user_id: str) -> int:
        return len(self._history.get(user_id, []))


class Gatekeeper:
    """
    Security / Identity Gate.
    
    Trust is a gradient, not a flag.
    Trust = f(history, behavior, stability)
    """
    
    def __init__(self, history_store: Optional[UserHistoryStore] = None):
        self.history_store = history_store or UserHistoryStore()
        self._trust_cache: dict[str, float] = {}
    
    def identify(self, user_id: str, message: str) -> UserIdentity:
        """
        Identify user and calculate behavioral trust level.
        
        Returns UserIdentity with trust_level and risk assessment.
        """
        history = self.history_store.get_history(user_id)
        history_length = len(history)
        
        # Calculate consistency score
        consistency = self._calculate_consistency(history, message)
        
        # Detect anomalies
        anomaly = self._detect_anomaly(history, message)
        
        # Calculate trust level
        trust_level = self._calculate_trust(
            history_length=history_length,
            consistency=consistency,
            anomaly=anomaly
        )
        
        # Determine risk flag
        risk_flag = trust_level < 0.3 or anomaly
        
        # Store this interaction
        self.history_store.add_interaction(user_id, message, datetime.now())
        
        return UserIdentity(
            user_id=user_id,
            trust_level=trust_level,
            risk_flag=risk_flag,
            history_length=history_length,
            consistency_score=consistency,
            anomaly_detected=anomaly
        )
    
    def _calculate_consistency(self, history: list[dict], current_message: str) -> float:
        """
        Calculate how consistent the current message is with user's history.
        
        Higher score = more consistent behavior.
        """
        if not history:
            return 0.5  # Neutral for new users
        
        # Simple heuristic: message length variance
        avg_length = sum(len(h["content"]) for h in history[-10:]) / min(len(history), 10)
        current_length = len(current_message)
        
        # Calculate deviation
        deviation = abs(current_length - avg_length) / max(avg_length, 1)
        
        # Convert to 0-1 scale (lower deviation = higher consistency)
        consistency = max(0, 1 - deviation / 5)
        
        return round(consistency, 2)
    
    def _detect_anomaly(self, history: list[dict], current_message: str) -> bool:
        """
        Detect anomalous behavior patterns.
        
        Simple heuristics:
        - Sudden topic shift
        - Unusual message patterns
        - Potential injection attempts
        """
        # Check for potential prompt injection patterns
        injection_patterns = [
            "ignore previous",
            "disregard instructions",
            "system prompt",
            "you are now",
            "new instructions:"
        ]
        
        message_lower = current_message.lower()
        for pattern in injection_patterns:
            if pattern in message_lower:
                return True
        
        # Check for sudden behavior change (if history exists)
        if len(history) >= 5:
            recent_lengths = [len(h["content"]) for h in history[-5:]]
            avg_recent = sum(recent_lengths) / 5
            
            # Extreme length deviation
            if len(current_message) > avg_recent * 10:
                return True
        
        return False
    
    def _calculate_trust(
        self,
        history_length: int,
        consistency: float,
        anomaly: bool
    ) -> float:
        """
        Calculate overall trust level.
        
        Factors:
        - History length (more history = more trust, up to a point)
        - Consistency score
        - Anomaly detection
        """
        # Base trust from history (logarithmic growth)
        import math
        history_factor = min(0.4, math.log(history_length + 1) / 10)
        
        # Consistency factor
        consistency_factor = consistency * 0.4
        
        # Combine
        trust = 0.2 + history_factor + consistency_factor  # Base 0.2 for new users
        
        # Penalize anomalies
        if anomaly:
            trust *= 0.5
        
        return round(min(1.0, max(0.0, trust)), 2)
