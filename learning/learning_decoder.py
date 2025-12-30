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
                
                # Reconstruct RawTrace
                from models.schemas import TraceStep
                
                steps_data = data.get("steps", [])
                steps = []
                for s in steps_data:
                    steps.append(TraceStep(
                        module=s.get("module", ""),
                        name=s.get("name", ""),
                        description=s.get("description", ""),
                        data_in=s.get("data_in"),
                        data_out=s.get("data_out"),
                        timestamp=datetime.fromisoformat(s.get("timestamp")) if s.get("timestamp") else datetime.now()
                    ))

                trace = RawTrace(
                    episode_id=data.get("episode_id", ""),
                    timestamp=datetime.fromisoformat(data.get("timestamp")) if data.get("timestamp") else datetime.now(),
                    user_input=data.get("user_input", ""),
                    context_snapshot=data.get("context_snapshot", {}),
                    expert_outputs=data.get("expert_outputs", []),
                    critic_output=data.get("critic_output", {}),
                    decision=data.get("decision", {}),
                    final_response=data.get("final_response", ""),
                    user_reaction=data.get("user_reaction"),
                    thoughts=data.get("thoughts", ""),
                    validation_report=data.get("validation_report", {}),
                    world_state_snapshot=data.get("world_state_snapshot", {}),
                    steps=steps
                )
                self.raw_traces.append(trace)
                loaded += 1
                
            except Exception as e:
                print(f"[Learning] Error loading {trace_file}: {e}")
        
        # Sort traces by timestamp to maintain chronological order
        self.raw_traces.sort(key=lambda x: x.timestamp)
        
        # Rebuild summaries in correct order
        self.summaries = []
        for trace in self.raw_traces:
            confidence = trace.decision.get("confidence", 0.5)
            # Determine outcome based on confidence
            if confidence >= 0.7:
                outcome = "success"
            elif confidence >= 0.4:
                outcome = "partial"
            else:
                outcome = "failure"
            
            summary = EpisodeSummary(
                episode_id=trace.episode_id,
                summary=f"Loaded: {trace.user_input[:50]}...",
                key_metrics={
                    "confidence": confidence,
                    "cost_ms": trace.decision.get("cost", {}).get("time_ms", 0),
                    "experts_used": len(trace.expert_outputs),
                    "depth": trace.decision.get("depth_used", "unknown")
                },
                outcome=outcome
            )
            self.summaries.append(summary)
        
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
                        times_validated=p.get("times_validated", 0),
                        suggested_update=p.get("suggested_update")
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

    def record_reflection_trace(self, trace: RawTrace):
        """Record a reflection trace (same as normal trace, but semantically distinct)."""
        self.record_trace(trace)
        
        # Also create a summary for the history list
        summary = EpisodeSummary(
            episode_id=trace.episode_id,
            summary=f"System Reflection: {trace.decision.get('action', 'Analyzed patterns')}",
            key_metrics={
                "confidence": trace.decision.get("confidence", 1.0),
                "cost_ms": trace.decision.get("cost", {}).get("time_ms", 0),
                "experts_used": len(trace.expert_outputs),
                "depth": "reflection" # Special depth tag
            },
            outcome="success"
        )
        self.summaries.append(summary)
    
    async def create_summary(self, trace: RawTrace) -> EpisodeSummary:
        """
        Create episode summary from raw trace (Level 2).
        
        This is what goes to reflection, not the raw trace.
        """
        # Build prompt from trace
        history_str = ""
        for s in trace.steps:
            if s.module in ["gatekeeper", "context_manager", "experts_module", "expert_neutral", "expert_creative", "expert_conservative", "expert_physics"]:
                history_str += f"- [{s.module}] {s.name}: {s.description}\n"
        
        prompt = SUMMARIZE_PROMPT + f"\n\nEPISODE:\nUser Input: {trace.user_input}\nSteps:\n{history_str}\nFinal Response: {trace.final_response}"
        
        summary_text = "Analysis failed"
        outcome = "failure"
        
        if self.llm:
            try:
                # Use quality_llm if available, otherwise just llm
                llm_for_summary = getattr(self.llm, 'main_llm', self.llm)
                response = await llm_for_summary.generate(prompt)
                
                # Simple parsing of response
                if "SUMMARY:" in response:
                    summary_text = response.split("SUMMARY:")[1].split("OUTCOME:")[0].strip()
                if "OUTCOME:" in response:
                    outcome_line = response.split("OUTCOME:")[1].split("KEY_INSIGHT:")[0].strip().lower()
                    if "success" in outcome_line: outcome = "success"
                    elif "partial" in outcome_line: outcome = "partial"
            except Exception as e:
                print(f"[Learning] Summary generation error: {e}")
        
        summary = EpisodeSummary(
            episode_id=trace.episode_id,
            summary=summary_text,
            key_metrics={
                "confidence": trace.decision.get("confidence", 0.0),
                "cost_ms": trace.decision.get("cost", {}).get("time_ms", 0),
                "experts_used": len(trace.expert_outputs),
                "depth": trace.decision.get("depth_used", "unknown")
            },
            outcome=outcome
        )
        
        self.summaries.append(summary)
        
        # Save summary to disk
        summary_file = self.storage_path / f"summary_{trace.episode_id}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)
            
        return summary
    
    def _pattern_similarity(self, desc1: str, desc2: str) -> float:
        """
        Calculate simple similarity between two pattern descriptions.
        Uses word overlap (Jaccard similarity).
        """
        words1 = set(desc1.lower().split())
        words2 = set(desc2.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'to', 'in', 'for', 
                      'on', 'with', 'and', 'or', 'that', 'this', 'it', '-', '–', '—'}
        words1 -= stop_words
        words2 -= stop_words
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _find_similar_pattern(self, description: str, threshold: float = 0.6) -> Optional[ExtractedPattern]:
        """
        Find an existing pattern similar to the given description.
        
        Returns the most similar pattern if similarity > threshold.
        """
        best_match = None
        best_score = 0.0
        
        for pattern in self.patterns:
            score = self._pattern_similarity(description, pattern.description)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = pattern
        
        return best_match
    
    def extract_pattern(
        self,
        episodes: list[EpisodeSummary],
        description: str
    ) -> ExtractedPattern:
        """
        Extract a pattern from multiple episodes (Level 3).
        
        If a similar pattern already exists, increments its validation count
        instead of creating a duplicate.
        """
        # Check for similar existing pattern
        existing = self._find_similar_pattern(description)
        
        if existing:
            # Increment validation count and update confidence
            existing.times_validated += 1
            existing.confidence = min(0.95, existing.confidence + 0.05)
            # Add new source episodes
            for ep in episodes:
                if ep.episode_id not in existing.source_episodes:
                    existing.source_episodes.append(ep.episode_id)
            return existing
        
        # Create new pattern
        pattern = ExtractedPattern(
            description=description,
            source_episodes=[e.episode_id for e in episodes],
            confidence=0.5,
            times_validated=1
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
