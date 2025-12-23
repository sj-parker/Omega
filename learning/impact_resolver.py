# Pattern Impact Resolver
# Maps natural language patterns to policy deltas and semantic rules

import json
from typing import Optional, Dict, Any

from models.schemas import ExtractedPattern, PolicyUpdate
from models.llm_interface import LLMInterface

IMPACT_RESOLVER_PROMPT = """You are a system architect. Translate a system behavior pattern into specific policy adjustments or semantic rules.

POLICY PARAMETERS:
- fast_path_bias: (float) Preference for fast path (±0.05). Increase if system is too slow/expensive, decrease if too shallow/inaccurate.
- expert_call_threshold: (float) Confidence below this -> call experts (±0.05). Increase to be more cautious, decrease to be more confident.

SEMANTIC RULES (Routing Hints):
Format: "trigger_condition": "target_depth"
Example: "complex_analytical_queries": "deep"
Example: "low_trust_users": "deep"

OUTPUT FORMAT (JSON only):
{
    "policy_deltas": {
        "fast_path_bias": 0.02,
        "expert_call_threshold": -0.01
    },
    "semantic_rules": {
        "pattern_category": "target_depth"
    }
}

Cap all numerical deltas at ±0.02 to ensure stability."""

class PatternImpactResolver:
    """
    Pattern Impact Resolver.
    
    Translates ExtractedPattern (text) into PolicyUpdate (numerical)
    and Semantic Rules (routing hints).
    """
    
    def __init__(self, llm: Optional[LLMInterface] = None):
        self.llm = llm
        
    async def resolve_impact(self, pattern: ExtractedPattern) -> Dict[str, Any]:
        """
        Resolve the impact of a pattern on the system policy.
        
        Returns a dict containing:
        - policy_deltas: Dict[str, float]
        - semantic_rules: Dict[str, str]
        """
        if not self.llm:
            return self._simple_resolve(pattern)
            
        try:
            response = await self.llm.generate(
                prompt=f"Pattern: {pattern.description}",
                system_prompt=IMPACT_RESOLVER_PROMPT,
                temperature=0.1
            )
            
            # Find the JSON block in the response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > 0:
                impact_data = json.loads(response[start:end])
                
                # Sanitize and cap deltas
                deltas = impact_data.get("policy_deltas", {})
                for key in deltas:
                    if isinstance(deltas[key], (int, float)):
                        deltas[key] = max(-0.02, min(0.02, deltas[key]))
                
                impact_data["policy_deltas"] = deltas
                return impact_data
                
        except Exception as e:
            print(f"[ImpactResolver] Error resolving impact for pattern {pattern.pattern_id}: {e}")
            
        return self._simple_resolve(pattern)
        
    def _simple_resolve(self, pattern: ExtractedPattern) -> Dict[str, Any]:
        """Fallback for resolving impact without LLM."""
        desc = pattern.description.lower()
        impact = {"policy_deltas": {}, "semantic_rules": {}}
        
        if "failure" in desc or "⚠️" in desc:
            # Increase caution
            impact["policy_deltas"]["expert_call_threshold"] = 0.02
            impact["policy_deltas"]["fast_path_bias"] = -0.01
        elif "success" in desc or "✅" in desc:
            # Slight bias towards efficiency
            impact["policy_deltas"]["fast_path_bias"] = 0.01
        elif "expert usage" in desc:
            if "high" in desc:
                impact["policy_deltas"]["fast_path_bias"] = 0.01
            else:
                impact["policy_deltas"]["expert_call_threshold"] = 0.01
                
        return impact
