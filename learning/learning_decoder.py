# Learning Decoder
# 3-level experience representation: Raw Trace → Episode Summary → Pattern

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])

from models.schemas import RawTrace, EpisodeSummary, ExtractedPattern
from models.llm_interface import LLMInterface


SUMMARIZE_PROMPT = """You are a system analyst. Summarize the following episode into a concise description.

Focus on:
1. What was the key challenge or question
2. How the system handled it (depth, experts used)
3. What was the outcome (confidence, cost)

Output format:
SUMMARY: [1-2 sentence summary]
OUTCOME: [success/partial/failure]
KEY_INSIGHT: [optional learning point]"""


class LearningDecoder:
    """
    Learning Decoder.
    
    Purpose: Store the complete trace of system thinking.
    
    3-level representation:
    1. Raw Trace - full storage
    2. Episode Summary - for reflection
    3. Extracted Pattern - for policy
    
    This is memory of experience, not memory of facts.
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        llm: Optional[LLMInterface] = None,
        auto_load: bool = True  # Load existing data on startup
    ):
        self.storage_path = storage_path or Path("./learning_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        
        # In-memory caches
        self.raw_traces: list[RawTrace] = []
        self.summaries: list[EpisodeSummary] = []
        self.patterns: list[ExtractedPattern] = []
        
        # Load existing data from disk
        if auto_load:
            self.load_from_disk()
    
    def load_from_disk(self) -> int:
        """
        Load saved traces and summaries from disk.
        
        Call this on startup to restore learning from previous sessions.
        Returns the number of items loaded.
        """
        loaded = 0
        
        # Load trace files
        for trace_file in sorted(self.storage_path.glob("trace_*.json")):
            try:
                with open(trace_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Reconstruct RawTrace (simplified - just store as dict for now)
                trace = RawTrace(
                    episode_id=data.get("episode_id", ""),
                    user_input=data.get("user_input", ""),
                    context_snapshot=data.get("context_snapshot", {}),
                    expert_outputs=data.get("expert_outputs", []),
                    critic_output=data.get("critic_output", {}),
                    decision=data.get("decision", {}),
                    final_response=data.get("final_response", ""),
                    user_reaction=data.get("user_reaction")
                )
                self.raw_traces.append(trace)
                
                # Create a basic summary from the loaded trace
                summary = EpisodeSummary(
                    episode_id=trace.episode_id,
                    summary=f"Loaded: {trace.user_input[:50]}...",
                    key_metrics={
                        "confidence": trace.decision.get("confidence", 0.5),
                        "cost_ms": trace.decision.get("cost", {}).get("time_ms", 0),
                        "experts_used": len(trace.expert_outputs),
                        "depth": trace.decision.get("depth_used", "unknown")
                    },
                    outcome="partial"  # Default for loaded data
                )
                self.summaries.append(summary)
                loaded += 1
                
            except Exception as e:
                print(f"[Learning] Error loading {trace_file}: {e}")
        
        # Load pattern files if they exist
        pattern_file = self.storage_path / "patterns.json"
        if pattern_file.exists():
            try:
                with open(pattern_file, 'r', encoding='utf-8') as f:
                    patterns_data = json.load(f)
                for p in patterns_data:
                    pattern = ExtractedPattern(
                        pattern_id=p.get("pattern_id", ""),
                        description=p.get("description", ""),
                        source_episodes=p.get("source_episodes", []),
                        confidence=p.get("confidence", 0.5),
                        times_validated=p.get("times_validated", 0)
                    )
                    self.patterns.append(pattern)
                    loaded += 1
            except Exception as e:
                print(f"[Learning] Error loading patterns: {e}")
        
        if loaded > 0:
            print(f"[Learning] Loaded {loaded} items from previous sessions")
        
        return loaded
    
    def save_patterns(self):
        """Save extracted patterns to disk for persistence."""
        pattern_file = self.storage_path / "patterns.json"
        patterns_data = [p.to_dict() for p in self.patterns]
        with open(pattern_file, 'w', encoding='utf-8') as f:
            json.dump(patterns_data, f, ensure_ascii=False, indent=2)
    
    def record_trace(self, trace: RawTrace):
        """Record a raw trace (Level 1)."""
        self.raw_traces.append(trace)
        
        # Persist to disk
        trace_file = self.storage_path / f"trace_{trace.episode_id}.json"
        with open(trace_file, 'w', encoding='utf-8') as f:
            json.dump(trace.to_dict(), f, ensure_ascii=False, indent=2)
    
    async def create_summary(self, trace: RawTrace) -> EpisodeSummary:
        """
        Create episode summary from raw trace (Level 2).
        
        This is what goes to reflection, not the raw trace.
        """
        if self.llm:
            # Use LLM to generate summary
            prompt = f"""Episode to summarize:

User input: {trace.user_input}
Experts used: {len(trace.expert_outputs)}
Decision: {trace.decision.get('action', 'respond')}
Confidence: {trace.decision.get('confidence', 0)}
Cost (ms): {trace.decision.get('cost', {}).get('time_ms', 0)}
Response excerpt: {trace.final_response[:200]}..."""
            
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=SUMMARIZE_PROMPT,
                temperature=0.3
            )
            
            # Parse response
            summary_text = trace.user_input[:50] + "..."
            outcome = "partial"
            
            for line in response.split('\n'):
                if line.startswith("SUMMARY:"):
                    summary_text = line.split(":", 1)[1].strip()
                elif line.startswith("OUTCOME:"):
                    outcome = line.split(":", 1)[1].strip().lower()
        else:
            # Simple fallback
            summary_text = f"Query: {trace.user_input[:50]}... → {trace.decision.get('depth_used', 'fast')}"
            outcome = "partial" if trace.decision.get('confidence', 0) < 0.6 else "success"
        
        summary = EpisodeSummary(
            episode_id=trace.episode_id,
            summary=summary_text,
            key_metrics={
                "confidence": trace.decision.get("confidence", 0),
                "cost_ms": trace.decision.get("cost", {}).get("time_ms", 0),
                "experts_used": len(trace.expert_outputs),
                "depth": trace.decision.get("depth_used", "fast")
            },
            outcome=outcome
        )
        
        self.summaries.append(summary)
        return summary
    
    def extract_pattern(
        self,
        episodes: list[EpisodeSummary],
        description: str
    ) -> ExtractedPattern:
        """
        Extract a pattern from multiple episodes (Level 3).
        
        Patterns are used by policy, not sent to LLM directly.
        """
        pattern = ExtractedPattern(
            description=description,
            source_episodes=[e.episode_id for e in episodes],
            confidence=0.5,
            times_validated=0
        )
        
        self.patterns.append(pattern)
        return pattern
    
    def get_recent_traces(self, n: int = 10) -> list[RawTrace]:
        """Get recent raw traces."""
        return self.raw_traces[-n:]
    
    def get_recent_summaries(self, n: int = 20) -> list[EpisodeSummary]:
        """Get recent summaries for reflection."""
        return self.summaries[-n:]
    
    def get_patterns(self) -> list[ExtractedPattern]:
        """Get all extracted patterns."""
        return self.patterns
    
    def get_metrics_aggregates(self, n: int = 50) -> dict:
        """
        Get aggregated metrics from recent episodes.
        
        Used by Homeostasis Controller.
        """
        recent = self.summaries[-n:] if self.summaries else []
        
        if not recent:
            return {
                "avg_confidence": 0.7,
                "avg_cost_ms": 500,
                "success_rate": 0.5,
                "expert_usage_rate": 0.3
            }
        
        confidences = [s.key_metrics.get("confidence", 0.5) for s in recent]
        costs = [s.key_metrics.get("cost_ms", 500) for s in recent]
        successes = sum(1 for s in recent if s.outcome == "success")
        with_experts = sum(1 for s in recent if s.key_metrics.get("experts_used", 0) > 0)
        
        return {
            "avg_confidence": sum(confidences) / len(confidences),
            "avg_cost_ms": sum(costs) / len(costs),
            "success_rate": successes / len(recent),
            "expert_usage_rate": with_experts / len(recent)
        }
    
    def clear_all_memory(self) -> int:
        """
        Clear ALL stored traces, summaries, and patterns.
        
        Returns the number of items deleted.
        """
        import shutil
        
        count = len(self.raw_traces) + len(self.summaries) + len(self.patterns)
        
        # Clear in-memory
        self.raw_traces.clear()
        self.summaries.clear()
        self.patterns.clear()
        
        # Clear disk storage
        if self.storage_path.exists():
            for f in self.storage_path.glob("*.json"):
                f.unlink()
        
        return count
    
    def sanitize_memory(self) -> int:
        """
        Filter identity mentions from all stored data.
        
        Returns the number of items sanitized.
        """
        try:
            from core.identity_filter import filter_data, filter_llm_response
        except ImportError:
            return 0
        
        sanitized = 0
        
        # Sanitize raw traces
        for trace in self.raw_traces:
            trace.final_response = filter_llm_response(trace.final_response)
            trace.user_input = filter_llm_response(trace.user_input)
            
            # Sanitize expert outputs
            for i, exp in enumerate(trace.expert_outputs):
                if isinstance(exp, dict) and "response" in exp:
                    exp["response"] = filter_llm_response(exp["response"])
            
            sanitized += 1
        
        # Sanitize summaries
        for summary in self.summaries:
            summary.summary = filter_llm_response(summary.summary)
            sanitized += 1
        
        # Sanitize patterns
        for pattern in self.patterns:
            pattern.description = filter_llm_response(pattern.description)
            sanitized += 1
        
        # Re-save sanitized traces to disk
        for trace in self.raw_traces:
            trace_file = self.storage_path / f"trace_{trace.episode_id}.json"
            if trace_file.exists():
                with open(trace_file, 'w', encoding='utf-8') as f:
                    json.dump(trace.to_dict(), f, ensure_ascii=False, indent=2)
        
        return sanitized
