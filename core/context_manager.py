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
    
    def __init__(self, max_events: int = 100):
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

    def save(self, filepath: str):
        """Save context to disk using atomic write."""
        import json
        import os
        
        data = {
            "events": [e.to_dict() for e in self.events],
            "active_goal": self.active_goal,
            "emotional_state": self.emotional_state,
            "system_mode": self.system_mode
        }
        
        # Atomic write with retry for Windows locking
        temp_path = f"{filepath}.tmp"
        
        max_retries = 5
        for i in range(max_retries):
            try:
                # Write to temp
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                
                # Atomic replace
                if os.path.exists(filepath):
                    os.replace(temp_path, filepath)
                else:
                    os.rename(temp_path, filepath)
                return # Success
                
            except PermissionError:
                # Windows file locking race condition
                import time
                time.sleep(0.01 * (i + 1)) # Backoff
            except Exception as e:
                print(f"[Context] Save failed: {e}")
                break
        
        # Cleanup if failed
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

    def load(self, filepath: str):
        """Load context from disk."""
        import json
        import os
        if not os.path.exists(filepath):
            return
            
        max_retries = 5
        for i in range(max_retries):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                break # Success
            except (PermissionError, json.JSONDecodeError):
                import time
                time.sleep(0.01 * (i + 1))
            except Exception as e:
                print(f"[Context] Load failed: {e}")
                return
        else:
             print(f"[Context] Load failed after {max_retries} retries")
             return

        self.active_goal = data.get("active_goal")
        self.emotional_state = data.get("emotional_state", "neutral")
        self.system_mode = data.get("system_mode", "normal")
        
        self.events.clear()
        for e_data in data.get("events", []):
            if isinstance(e_data.get("timestamp"), str):
                e_data["timestamp"] = datetime.fromisoformat(e_data["timestamp"])
            self.events.append(ContextEvent(**e_data))
        # print(f"[Context] Loaded {len(self.events)} events from disk.")


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

    def save(self, filepath: str):
        """Atomic save with Windows lock handling."""
        import json
        import os
        import tempfile
        import time

        data = {
            "all_events": [e.to_dict() for e in self.all_events],
            "decisions": [d.to_dict() for d in self.decisions],
            "states": self.states
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        max_retries = 5
        for i in range(max_retries):
            try:
                fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(filepath), suffix=".tmp")
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                if os.path.exists(filepath):
                    os.remove(filepath)
                os.rename(temp_path, filepath)
                return
            except Exception as e:
                if i == max_retries - 1:
                    print(f"[FullStore] Final save attempt failed: {e}")
                time.sleep(0.01 * (i + 1))

    def load(self, filepath: str):
        """Atomic load with Windows lock handling."""
        import json
        import os
        import time

        if not os.path.exists(filepath):
            return

        max_retries = 5
        data = None
        for i in range(max_retries):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                break
            except Exception as e:
                if i == max_retries - 1:
                    print(f"[FullStore] Final load attempt failed: {e}")
                    return
                time.sleep(0.01 * (i + 1))
        
        if not data:
            return

        self.all_events = []
        for e_data in data.get("all_events", []):
            if isinstance(e_data.get("timestamp"), str):
                e_data["timestamp"] = datetime.fromisoformat(e_data["timestamp"])
            self.all_events.append(ContextEvent(**e_data))
        
        # Decisions and states loading could be added here if needed for recall
        self.states = data.get("states", [])
    
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
        selected = sorted_events[:self.max_context_events]
        
        # CRITICAL: Re-sort by timestamp to preserve conversation flow
        # The LLM needs Chronological order, not importance order
        selected.sort(key=lambda e: e.timestamp)
        
        # Deduplicate repetitive system responses
        # Loop backwards and remove adjacent duplicates from the SAME source (system_response)
        if len(selected) > 1:
            deduped = []
            prev = None
            for event in selected:
                if event.event_type == "system_response" and prev and prev.event_type == "system_response":
                     # Check for near-identical content (simple string match for now)
                     if event.content.strip() == prev.content.strip():
                         continue # Skip duplicate
                
                deduped.append(event)
                prev = event
            selected = deduped
        
        return selected

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
        self.full_storage_path = storage_path.replace(".json", "_full.json") if storage_path else None
        
        # Load stores if paths are provided
        if self.storage_path:
             self.short_store.load(self.storage_path)
             if self.full_storage_path:
                 self.full_store.load(self.full_storage_path)
             
    def add_fact(self, content: str, entities: list[str] = None, importance: float = 0.8):
        """Add a distilled fact to long-term memory."""
        # CRITICAL: Do NOT store facts about Omega's history or internal states
        # This prevents 'Narrative Runaway' from being persisted.
        banned_terms = ["omega", "омега", "intervention", "вмешательство", "history", "история"]
        content_lower = content.lower()
        if any(term in content_lower for term in banned_terms):
            print(f"[KM] Fact rejected (Narrative Protection): {content[:50]}...")
            return

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
        
        if self.storage_path:
            self.short_store.save(self.storage_path)
            if self.full_storage_path:
                self.full_store.save(self.full_storage_path)
    
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

        if self.storage_path:
            self.short_store.save(self.storage_path)
            if self.full_storage_path:
                self.full_store.save(self.full_storage_path)
    
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
        # Sync with disk to ensure we have latest data from other workers/processes
        if self.storage_path:
             self.short_store.load(self.storage_path)

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
        n_recent: int = 20,
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
            # Increased from 20 to 50 for deeper recall
            events = self.short_store.get_recent_events(50) 
            return {
                "relevant_facts": [f.to_dict() for f in facts],
                "recent_events": [e.to_dict() for e in events],
                "query": semantic_filter
            }
        
        elif scope == ContextScope.FULL:
            # Sync full store before retrieval
            if self.full_storage_path:
                self.full_store.load(self.full_storage_path)
            return self.get_full_context()
