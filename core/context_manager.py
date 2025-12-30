# Context Manager and Memory Gate
# Context ≠ Memory. Context is a projection, not an archive.

from datetime import datetime
from typing import Optional
from collections import deque

import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])

from models.schemas import (
    UserIdentity, ContextEvent, ContextSlice, DecisionObject, LongTermFact, WorldState,
    ContextScope
)


class ShortContextStore:
    """
    Fast context (Short Context Store).
    
    Contains:
    - Recent events
    - Active goal
    - Emotional and dialog status
    - Current system mode
    """
    
    def __init__(self, max_events: int = 20):
        self.max_events = max_events
        self.events: deque[ContextEvent] = deque(maxlen=max_events)
        self.active_goal: Optional[str] = None
        self.emotional_state: str = "neutral"
        self.system_mode: str = "normal"
    
    def add_event(self, event: ContextEvent):
        self.events.append(event)
    
    def get_recent_events(self, n: int = 10) -> list[ContextEvent]:
        return list(self.events)[-n:]
    
    def set_goal(self, goal: str):
        self.active_goal = goal
    
    def set_emotional_state(self, state: str):
        self.emotional_state = state
    
    def set_mode(self, mode: str):
        self.system_mode = mode


class FullContextStore:
    """
    Full context store.
    
    Contains:
    - All interaction history
    - Multimodal events
    - Decisions and reasons
    - System states
    
    Used on demand, not always.
    """
    
    def __init__(self):
        self.all_events: list[ContextEvent] = []
        self.decisions: list[DecisionObject] = []
        self.states: list[dict] = []
    
    def add_event(self, event: ContextEvent):
        self.all_events.append(event)
    
    def add_decision(self, decision: DecisionObject):
        self.decisions.append(decision)
    
    def save_state(self, state: dict):
        self.states.append({
            "timestamp": datetime.now().isoformat(),
            **state
        })
    
    def get_events_by_type(self, event_type: str) -> list[ContextEvent]:
        return [e for e in self.all_events if e.event_type == event_type]
    
    def get_recent_decisions(self, n: int = 10) -> list[DecisionObject]:
        return self.decisions[-n:]


class MemoryGate:
    """
    Memory Gate.
    
    Purpose: Limit the amount of information passed to OM.
    
    Functions:
    - Context filtering
    - Prioritization
    - Protection from OM overload
    
    OM receives only a slice.
    Learning circuit receives everything.
    """
    
    def __init__(
        self,
        max_context_events: int = 10,
        min_importance: float = 0.3
    ):
        self.max_context_events = max_context_events
        self.min_importance = min_importance
    
    def filter_context(
        self,
        events: list[ContextEvent],
        current_importance: float = 0.5
    ) -> list[ContextEvent]:
        """
        Filter and prioritize context events for OM.
        
        Criteria:
        - Importance threshold
        - Recency
        - Relevance to current input
        """
        # Filter by importance
        filtered = [e for e in events if e.importance >= self.min_importance]
        
        # Sort by importance * recency weight
        now = datetime.now()
        
        def score(event: ContextEvent) -> float:
            # Recency weight: newer events get higher weight
            age_seconds = (now - event.timestamp).total_seconds()
            recency = max(0.1, 1 - age_seconds / 3600)  # Decay over 1 hour
            return event.importance * recency
        
        sorted_events = sorted(filtered, key=score, reverse=True)
        
        # Limit to max
        return sorted_events[:self.max_context_events]

    def rank_facts(self, facts: list[LongTermFact], query: str) -> list[LongTermFact]:
        """Rank long-term facts based on relevance to query."""
        if not facts:
            return []
            
        # Simple heuristic: word overlap + importance
        query_words = set(query.lower().split())
        
        def fact_score(fact: LongTermFact) -> float:
            content_words = set(fact.content.lower().split())
            overlap = len(query_words.intersection(content_words))
            
            # Entity bonus
            for entity in fact.entities:
                if entity.lower() in query.lower():
                    overlap += 2
                    
            return (overlap * 0.5) + (fact.importance * 0.5)
            
        sorted_facts = sorted(facts, key=fact_score, reverse=True)
        return [f for f in sorted_facts if fact_score(f) > 0.4][:3] # Return top 3 relevant facts


class ContextManager:
    """
    Context Manager (KM).
    
    Purpose: Collect and reflect the current state of the entire system.
    
    Receives data from:
    - User
    - Security Gate
    - OM (decisions made)
    - Experts / Critic
    - External sources (future: audio, video)
    
    KM does NOT make decisions.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.short_store = ShortContextStore()
        self.full_store = FullContextStore()
        self.memory_gate = MemoryGate()
        self.long_term_facts: list[LongTermFact] = []
        self.storage_path = storage_path
        
        # Load facts if storage_path is provided (future)
        
    def add_fact(self, content: str, entities: list[str] = None, importance: float = 0.8):
        """Add a distilled fact to long-term memory."""
        fact = LongTermFact(
            content=content,
            entities=entities or [],
            importance=importance
        )
        self.long_term_facts.append(fact)
        print(f"[KM] New fact stored: {content[:50]}...")
        
    def search_facts(self, query: str) -> list[LongTermFact]:
        """Search for relevant facts in long-term memory."""
        return self.memory_gate.rank_facts(self.long_term_facts, query)
    
    def record_user_input(self, user_input: str, importance: float = 0.7):
        """Record user input as a context event."""
        event = ContextEvent(
            timestamp=datetime.now(),
            event_type="user_input",
            content=user_input,
            importance=importance
        )
        self.short_store.add_event(event)
        self.full_store.add_event(event)
    
    def record_system_response(self, response: str, importance: float = 0.5):
        """Record system response."""
        event = ContextEvent(
            timestamp=datetime.now(),
            event_type="system_response",
            content=response,
            importance=importance
        )
        self.short_store.add_event(event)
        self.full_store.add_event(event)
    
    def record_expert_output(self, expert_type: str, output: str, importance: float = 0.4):
        """Record expert output."""
        event = ContextEvent(
            timestamp=datetime.now(),
            event_type=f"expert_{expert_type}",
            content=output,
            importance=importance
        )
        self.full_store.add_event(event)  # Only to full store
    
    def record_decision(self, decision: DecisionObject):
        """Record OM decision."""
        self.full_store.add_decision(decision)
    
    def get_context_slice(
        self,
        user_input: str,
        user_identity: UserIdentity,
        world_state: WorldState
    ) -> ContextSlice:
        """
        Get a context slice for OM.
        
        This is filtered by Memory Gate.
        """
        # Get recent events from short store
        recent = self.short_store.get_recent_events(20)
        
        # Filter through Memory Gate
        filtered = self.memory_gate.filter_context(recent)
        
        return ContextSlice(
            user_input=user_input,
            user_identity=user_identity,
            world_state=world_state,
            recent_events=filtered,
            active_goal=self.short_store.active_goal,
            emotional_state=self.short_store.emotional_state,
            system_mode=self.short_store.system_mode
        )
    
    def get_context_for_recall(self, user_input: str) -> str:
        """Get formatted facts relevant to the query for retrieval."""
        relevant_facts = self.search_facts(user_input)
        if not relevant_facts:
            return ""
            
        fact_str = "\nRelevant Background Information (Long-term Memory):\n"
        for i, fact in enumerate(relevant_facts):
            fact_str += f"- {fact.content}\n"
        return fact_str
    
    def get_full_context(self) -> dict:
        """
        Get full context for Learning Decoder.
        
        This bypasses Memory Gate.
        """
        return {
            "all_events": [e.to_dict() for e in self.full_store.all_events],
            "decisions": [d.to_dict() for d in self.full_store.decisions],
            "states": self.full_store.states,
            "current": {
                "goal": self.short_store.active_goal,
                "emotional_state": self.short_store.emotional_state,
                "mode": self.short_store.system_mode
            }
        }
    
    def get_scoped_context(
        self,
        scope: ContextScope,
        n_recent: int = 5,
        semantic_filter: Optional[str] = None
    ) -> Optional[dict]:
        """
        Get context based on requested scope.
        
        Used for Task-Based Context Slicing - each task specifies
        how much context it needs.
        
        Args:
            scope: Level of context needed (NONE, RECENT, RELEVANT, FULL)
            n_recent: Number of recent events for RECENT scope
            semantic_filter: Query string for RELEVANT scope
        
        Returns:
            Context dict or None for NONE scope
        """
        if scope == ContextScope.NONE:
            return None
        
        elif scope == ContextScope.RECENT:
            events = self.short_store.get_recent_events(n_recent)
            return {
                "recent_events": [e.to_dict() for e in events],
                "active_goal": self.short_store.active_goal
            }
        
        elif scope == ContextScope.RELEVANT:
            facts = self.search_facts(semantic_filter or "")
            return {
                "relevant_facts": [f.to_dict() for f in facts],
                "query": semantic_filter
            }
        
        elif scope == ContextScope.FULL:
            return self.get_full_context()
