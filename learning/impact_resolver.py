# Pattern Impact Resolver
# Maps natural language patterns to policy deltas and semantic rules

import json
import re
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

OUTPUT FORMAT (JSON only, no extra text):
{
    "policy_deltas": {
        "fast_path_bias": 0.02,
        "expert_call_threshold": -0.01
    },
    "semantic_rules": {
        "pattern_category": "target_depth"
    }
}

CRITICAL: Return ONLY the JSON object. No explanations. Cap deltas at ±0.02."""

class PatternImpactResolver:
    """
    Pattern Impact Resolver.
    
    Translates ExtractedPattern (text) into PolicyUpdate (numerical)
    and Semantic Rules (routing hints).
    """
    
    def __init__(self, llm: Optional[LLMInterface] = None):
        self.llm = llm
        
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Robust JSON extraction from LLM response.
        
        Tries multiple strategies:
        1. Direct parse (if response is pure JSON)
        2. Find JSON block between { }
        3. Regex extraction of JSON-like structure
        4. raw_decode for partial JSON
        """
        # Strategy 1: Direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Find outermost { } braces
        start = text.find('{')
        if start >= 0:
            # Find matching closing brace
            depth = 0
            for i, char in enumerate(text[start:], start):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i+1])
                        except json.JSONDecodeError:
                            break
        
        # Strategy 3: Regex for JSON-like structure
        json_pattern = r'\{[^{}]*"policy_deltas"[^{}]*\{[^{}]*\}[^{}]*\}'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: raw_decode (handles trailing garbage)
        try:
            decoder = json.JSONDecoder()
            start_idx = text.find('{')
            if start_idx >= 0:
                obj, _ = decoder.raw_decode(text[start_idx:])
                return obj
        except json.JSONDecodeError:
            pass
        
        return None
        
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
                temperature=0.0  # Deterministic for JSON parsing
            )
            
            # Use robust JSON extraction
            impact_data = self._extract_json(response)
            
            if impact_data:
                # Sanitize and cap deltas
                deltas = impact_data.get("policy_deltas", {})
                for key in list(deltas.keys()):
                    if isinstance(deltas[key], (int, float)):
                        deltas[key] = max(-0.02, min(0.02, deltas[key]))
                    else:
                        del deltas[key]  # Remove non-numeric values
                
                impact_data["policy_deltas"] = deltas
                
                # Ensure semantic_rules is a dict
                if not isinstance(impact_data.get("semantic_rules"), dict):
                    impact_data["semantic_rules"] = {}
                    
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
