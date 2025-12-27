# Verification Test for Pattern-Driven Policy Updates

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.schemas import (
    PolicySpace, ContextSlice, UserIdentity, RawTrace, 
    EpisodeSummary, ExtractedPattern, DecisionDepth
)
from models.llm_interface import MockLLM
from core.operational_module import OperationalModule
from learning.learning_decoder import LearningDecoder
from learning.homeostasis import HomeostasisController
from learning.reflection import ReflectionController

class ForcedMockLLM(MockLLM):
    """Custom Mock LLM to return specific behavior for reflection."""
    async def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=1024):
        if "analyze these recent episodes" in prompt.lower():
            return """PATTERN: Frequent failures in complex analytical queries.
CONFIDENCE: 0.9
RECOMMENDATION: Increase depth for analytical tasks."""
            
        if "translate a system behavior pattern" in (system_prompt or "").lower():
            return """{
    "policy_deltas": {
        "fast_path_bias": -0.02,
        "expert_call_threshold": 0.02
    },
    "semantic_rules": {
        "complex": "deep"
    }
}"""
        return await super().generate(prompt, system_prompt, temperature, max_tokens)

async def test_pattern_impact():
    print("--- Starting Pattern-Driven Updates Verification ---")
    
    # 1. Setup
    llm = ForcedMockLLM()
    policy = PolicySpace()
    decoder = LearningDecoder(llm=llm, auto_load=False)
    homeostasis = HomeostasisController(policy)
    reflection = ReflectionController(decoder, homeostasis, llm)
    om = OperationalModule(llm, policy)
    
    # Initial policy state
    print(f"Initial Fast Path Bias: {policy.fast_path_bias}")
    print(f"Initial Semantic Rules: {policy.semantic_rules}")
    
    # 2. Mock 3 failure episodes
    for i in range(3):
        summary = EpisodeSummary(
            episode_id=f"fail_{i}",
            summary=f"Episode {i} failed to handle complex query",
            key_metrics={"confidence": 0.3, "cost_ms": 1000, "experts_used": 0},
            outcome="failure"
        )
        decoder.summaries.append(summary)
    
    print(f"Added {len(decoder.summaries)} failure episodes.")
    
    # 3. Trigger reflection
    print("Triggering reflection...")
    pattern = await reflection.reflect_once()
    
    if pattern:
        print(f"Pattern found: {pattern.description}")
        print(f"Suggested Update: {pattern.suggested_update}")
    else:
        print("FAILED: No pattern found")
        return

    # 4. Verify policy updates
    print(f"Updated Fast Path Bias: {policy.fast_path_bias}")
    print(f"Updated Semantic Rules: {policy.semantic_rules}")
    
    assert policy.fast_path_bias < 0.65, "Fast path bias should have decreased"
    assert "complex" in policy.semantic_rules, "Semantic rule 'complex' should be present"
    assert policy.semantic_rules["complex"] == "deep", "Semantic rule 'complex' should point to 'deep'"
    
    # 5. Verify OperationalModule respects semantic rules
    print("Verifying OperationalModule behavior...")
    
    # A query that matches the rule "complex"
    from models.schemas import WorldState
    context = ContextSlice(
        user_input="Explain complex quantum physics",
        user_identity=UserIdentity(user_id="test", trust_level=1.0),
        world_state=WorldState()
    )
    
    # We need to simulate intent classification since MockLLM is generic
    # But OM._decide_depth uses the intent returned by _classify_intent
    # Let's mock _classify_intent or just see how it handles "complex"
    
    intent, conf = await om._classify_intent(context.user_input)
    depth = om._decide_depth(intent, conf, context)
    
    print(f"Query: '{context.user_input}' -> Intent: {intent}, Depth: {depth.value}")
    
    # If the rule trigger "complex" is in the intent or input, it should be DEEP
    assert depth == DecisionDepth.DEEP, f"Should have triggered DEEP path for 'complex' query, got {depth.value}"

    print("--- Verification Successful! ---")

if __name__ == "__main__":
    asyncio.run(test_pattern_impact())
