# Reflection Mode
# Background analysis of past episodes

import asyncio
from datetime import datetime
from typing import Optional

import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])

from models.schemas import EpisodeSummary, ExtractedPattern, PolicySpace
from models.llm_interface import LLMInterface
from learning.learning_decoder import LearningDecoder
from learning.homeostasis import HomeostasisController


REFLECTION_PROMPT = """You are a system reflection agent. Analyze these episodes and identify patterns.

Focus on:
1. Recurring issues or challenges
2. Successful strategies
3. Areas for improvement
4. Potential optimizations

Output format:
PATTERN: [description of identified pattern]
CONFIDENCE: [0.0-1.0]
RECOMMENDATION: [suggested policy adjustment]"""


class ReflectionController:
    """
    Reflection Mode Controller.
    
    Activates when there are no user requests.
    
    Process:
    1. Select episodes from Learning Decoder
    2. Send episodes to experts for analysis
    3. Identify improvements/errors
    4. Pass conclusions to OM (in analysis mode)
    5. Write generalizations back to Learning Decoder
    
    The system thinks at different times in different ways - like a human.
    """
    
    def __init__(
        self,
        learning_decoder: LearningDecoder,
        homeostasis: HomeostasisController,
        llm: Optional[LLMInterface] = None
    ):
        self.learning_decoder = learning_decoder
        self.homeostasis = homeostasis
        self.llm = llm
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start_background(self, interval_seconds: float = 60.0):
        """Start background reflection loop."""
        self._running = True
        self._task = asyncio.create_task(self._reflection_loop(interval_seconds))
    
    async def stop_background(self):
        """Stop background reflection loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _reflection_loop(self, interval: float):
        """Main reflection loop."""
        from datetime import datetime
        print(f"[Reflection] Background loop started (interval: {interval}s)")
        
        while self._running:
            try:
                print(f"[Reflection] {datetime.now().strftime('%H:%M:%S')} - Running background reflection...")
                pattern = await self._do_reflect(verbose=False)
                if pattern:
                    print(f"[Reflection] {datetime.now().strftime('%H:%M:%S')} - Pattern: {pattern.description[:80]}...")
                else:
                    print(f"[Reflection] {datetime.now().strftime('%H:%M:%S')} - No pattern (not enough data)")
            except Exception as e:
                print(f"[Reflection] {datetime.now().strftime('%H:%M:%S')} - Error: {e}")
            
            await asyncio.sleep(interval)
    
    async def reflect_once(self) -> Optional[ExtractedPattern]:
        """
        Perform a single reflection cycle (user-triggered).
        
        Returns extracted pattern if any.
        """
        return await self._do_reflect(verbose=True)
    
    async def _do_reflect(self, verbose: bool = False) -> Optional[ExtractedPattern]:
        """
        Internal reflection implementation.
        
        Args:
            verbose: If True, print status messages (for manual /reflect)
        """
        # Get recent summaries (not raw traces)
        summaries = self.learning_decoder.get_recent_summaries(n=10)
        
        if len(summaries) < 3:
            # Not enough data to reflect
            return None
        
        # Get aggregated metrics
        metrics = self.learning_decoder.get_metrics_aggregates()
        
        # Run homeostasis analysis
        policy_update = self.homeostasis.analyze(metrics)
        if policy_update:
            self.homeostasis.apply_update(policy_update)
            if verbose:
                print(f"[Homeostasis] Policy adjusted: {policy_update.reason}")
        
        # Try LLM reflection first, fallback to simple
        pattern = None
        
        if self.llm:
            try:
                pattern = await self._llm_reflection(summaries)
            except Exception as e:
                if verbose:
                    print(f"[Reflection] LLM error: {e}")
        
        # Fallback to simple reflection if LLM didn't produce a pattern
        if pattern is None:
            pattern = self._simple_reflection(summaries)
        
        if pattern:
            # Save pattern to disk
            self.learning_decoder.save_patterns()
            if verbose:
                print(f"[Reflection] Pattern: {pattern.description}")
        
        return pattern
    
    async def _llm_reflection(
        self,
        summaries: list[EpisodeSummary]
    ) -> Optional[ExtractedPattern]:
        """Use LLM to identify patterns."""
        
        # Format summaries for LLM
        summaries_text = "\n\n".join([
            f"Episode {i+1}:\n{s.summary}\nOutcome: {s.outcome}\nMetrics: {s.key_metrics}"
            for i, s in enumerate(summaries)
        ])
        
        response = await self.llm.generate(
            prompt=f"Analyze these recent episodes:\n\n{summaries_text}",
            system_prompt=REFLECTION_PROMPT,
            temperature=0.4
        )
        
        # Parse response
        pattern_desc = None
        confidence = 0.5
        
        for line in response.split('\n'):
            if line.startswith("PATTERN:"):
                pattern_desc = line.split(":", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except:
                    pass
        
        if pattern_desc:
            return self.learning_decoder.extract_pattern(
                episodes=summaries,
                description=pattern_desc
            )
        
        return None
    
    def _simple_reflection(
        self,
        summaries: list[EpisodeSummary]
    ) -> Optional[ExtractedPattern]:
        """Simple pattern detection - always creates a summary."""
        
        # Count outcomes
        outcomes = {"success": 0, "partial": 0, "failure": 0}
        for s in summaries:
            outcomes[s.outcome] = outcomes.get(s.outcome, 0) + 1
        
        total = len(summaries)
        
        # Calculate metrics
        avg_confidence = sum(
            s.key_metrics.get("confidence", 0.5) for s in summaries
        ) / total
        
        expert_usage = sum(
            1 for s in summaries
            if s.key_metrics.get("experts_used", 0) > 0
        ) / total
        
        depths = {}
        for s in summaries:
            d = s.key_metrics.get("depth", "unknown")
            depths[d] = depths.get(d, 0) + 1
        most_common_depth = max(depths, key=depths.get) if depths else "unknown"
        
        # Check for specific patterns first
        if outcomes.get("failure", 0) / total > 0.3:
            description = f"⚠️ High failure rate ({outcomes['failure']}/{total}). Need more careful responses."
        elif outcomes.get("success", 0) / total > 0.8:
            description = f"✅ High success rate ({outcomes['success']}/{total}). Strategy is working well."
        elif expert_usage > 0.7:
            description = f"🔄 High expert usage ({expert_usage:.0%}). Consider raising fast_path_bias."
        else:
            # Always create a summary pattern
            description = (
                f"📊 Summary of {total} episodes: "
                f"success={outcomes.get('success', 0)}, "
                f"partial={outcomes.get('partial', 0)}, "
                f"failure={outcomes.get('failure', 0)}. "
                f"Avg confidence: {avg_confidence:.0%}. "
                f"Most used depth: {most_common_depth}. "
                f"Expert usage: {expert_usage:.0%}."
            )
        
        return self.learning_decoder.extract_pattern(
            episodes=summaries,
            description=description
        )
