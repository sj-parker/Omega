import json
import os
import asyncio
from typing import List, Dict, Optional, Tuple
from core.specialists.base import BaseSpecialist, SpecialistResult

class SpecialistBroker:
    """
    Smart Broker that manages, selects, and learns from Specialists.
    
    Features:
    - Dynamic Registry: Stores available specialists.
    - Smart Selection: Combines keyword matching with 'Reliability Score'.
    - Learning: Updates scores based on success/failure feedback.
    """
    
    def __init__(self, storage_path: str = "learning_data/specialist_stats.json"):
        self.specialists: Dict[str, BaseSpecialist] = {}
        self.stats_path = storage_path
        self.stats: Dict[str, float] = self._load_stats()
        
        # Default score for new specialists
        self.DEFAULT_SCORE = 1.0
        
    def register(self, specialist: BaseSpecialist):
        """Register a specialist instance."""
        meta = specialist.metadata
        self.specialists[meta.id] = specialist
        
        # Initialize stats if new
        if meta.id not in self.stats:
            self.stats[meta.id] = self.DEFAULT_SCORE
            
        print(f"[Broker] Registered: {meta.name} (Score: {self.stats[meta.id]:.2f})")
        
    def _load_stats(self) -> Dict[str, float]:
        """Load reliability scores from JSON."""
        if not os.path.exists(self.stats_path):
            return {}
        try:
            with open(self.stats_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Broker] Error loading stats: {e}")
            return {}
            
    def _save_stats(self):
        """Persist reliability scores."""
        # Ensure dir exists
        os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
        try:
            with open(self.stats_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            print(f"[Broker] Error saving stats: {e}")

    def get_selection(self, query: str, threshold: float = 0.5) -> Optional[BaseSpecialist]:
        """
        Select the best specialist for the query.
        
        Algorithm:
        1. Ask each specialist for 'can_handle' score (Match Score).
        2. Multiply by 'Reliability Score' (Learning).
        3. Return the best one if > threshold.
        """
        best_specialist = None
        best_score = -1.0
        
        query_lower = query.lower()
        
        for sid, specialist in self.specialists.items():
            # 1. Match Score (from specialist itself)
            match_score = specialist.can_handle(query_lower)
            
            # 2. Reliability Score (from learning)
            reliability = self.stats.get(sid, self.DEFAULT_SCORE)
            
            # 3. Final Score
            final_score = match_score * reliability
            
            if final_score > best_score:
                best_score = final_score
                best_specialist = specialist
                
        if best_score >= threshold and best_specialist:
            print(f"[Broker] Selected {best_specialist.metadata.name} (Match: {best_score:.2f})")
            return best_specialist
            
        return None

    def feedback(self, specialist_id: str, success: bool):
        """
        Feedback loop for learning.
        
        success=True  -> Reward (+0.05)
        success=False -> Penalty (-0.20)
        """
        if specialist_id not in self.stats:
            return
            
        current = self.stats[specialist_id]
        
        if success:
            # Gentle reward, cap at 2.0
            new_score = min(2.0, current + 0.05)
        else:
            # Harsh penalty, floor at 0.1
            new_score = max(0.1, current - 0.20)
            
        self.stats[specialist_id] = new_score
        self._save_stats()
        print(f"[Broker] Updated score for {specialist_id}: {current:.2f} -> {new_score:.2f}")

    def list_specialists(self) -> List[Dict]:
        """List all specialists and their stats."""
        return [
            {
                "id": s.metadata.id,
                "name": s.metadata.name,
                "score": self.stats.get(s.metadata.id, 0.0)
            }
            for s in self.specialists.values()
        ]
