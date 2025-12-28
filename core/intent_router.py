import yaml
import json
import os
from typing import Tuple, Dict, Any, Optional
from core.ontology import is_internal_query, entity_exists, extract_entity_name, should_block_search

class IntentRouter:
    """
    Decoupled Intent Classification Module.
    
    Responsibilities:
    1. Load classification rules from config.
    2. Check ontology/safety gates.
    3. Apply keyword-based heuristics (Fast Path).
    4. Fallback to LLM with Structured Output (Slow Path).
    """
    
    def __init__(self, llm_interface, config_path: str = "config/intent_rules.yaml"):
        self.llm = llm_interface
        self.config_path = config_path
        self.rules = self._load_rules()
        
    def _load_rules(self) -> Dict[str, Any]:
        """Load keyword rules from YAML config."""
        # Adjust path if running from different working dir
        # Assuming e:\agi2 is root
        base_path = os.path.dirname(os.path.dirname(__file__)) # e:\agi2
        full_path = os.path.join(base_path, self.config_path)
            
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data.get('rules', {})
        except FileNotFoundError:
            print(f"[IntentRouter] WARNING: Config file not found at {full_path}. Using empty rules.")
            return {}
        except Exception as e:
            print(f"[IntentRouter] Error loading rules: {e}")
            return {}

    async def classify(self, user_input: str) -> Tuple[str, float]:
        """
        Classify user intent.
        
        Returns:
            (intent_name, confidence_score)
        """
        input_lower = user_input.lower()
        
        # 1. Ontology & Safety Gates
        gate_result = self._check_ontology_gates(user_input)
        if gate_result:
            return gate_result
            
        # 2. Keyword Rule Matching (Fast Path)
        rule_result = self._check_keyword_rules(input_lower)
        if rule_result:
            return rule_result
            
        # 3. LLM Classification (Slow Path - Structured JSON)
        return await self._llm_classify(user_input)

    def _check_ontology_gates(self, user_input: str) -> Optional[Tuple[str, float]]:
        """Check internal ontology and blocked search terms."""
        
        # Internal Architecture Queries
        if is_internal_query(user_input):
            entity_name = extract_entity_name(user_input)
            if entity_name and not entity_exists(entity_name):
                print(f"[IntentRouter] ONTOLOGY GATE: Entity '{entity_name}' not found -> blocking fabrication")
                return "unknown_internal", 0.99
            print(f"[IntentRouter] ONTOLOGY GATE: Valid internal query about '{entity_name or 'Omega'}'")
            return "internal_query", 0.90
            
        # Blocked Search Topics
        block_search, reason = should_block_search(user_input)
        if block_search:
            print(f"[IntentRouter] SEARCH BLOCKED: Reason='{reason}'")
            if reason == "math_expression":
                return "calculation_simple", 0.95
            elif reason == "self_analysis":
                return "self_reflection", 0.90
            else:
                return "internal_query", 0.90
                
        return None

    def _check_keyword_rules(self, input_lower: str) -> Optional[Tuple[str, float]]:
        """Check input against loaded YAML rules."""
        
        # Priority order could be important, but dict iteration order is insertion order in py3.7+
        # We can enforce specific order if needed.
        
        # Pre-defined priority list to match original logic's precedence
        priority_order = ["realtime_data", "calculation", "analytical", "philosophical", "physics"]
        
        for intent in priority_order:
            if intent not in self.rules:
                continue
                
            rule_config = self.rules[intent]
            keywords = rule_config.get('keywords', [])
            threshold = rule_config.get('threshold', 1)
            
            match_count = sum(1 for kw in keywords if kw.lower() in input_lower)
            
            if match_count >= threshold:
                print(f"[IntentRouter] KEYWORD RULE: Detected '{intent}' ({match_count} matches)")
                # Map specific intents to original return values if needed, 
                # or just return the intent name from config
                return intent, 0.95
                
        return None

    async def _llm_classify(self, user_input: str) -> Tuple[str, float]:
        """Use LLM with JSON output to classify intent."""
        
        # Simplified JSON-focused prompt
        prompt = f"""Classify this user message into a JSON object.
User message: "{user_input}"

Categories:
- memorize: (Save facts/instructions)
- recall: (Ask about past/history)
- smalltalk: (Casual chat)
- factual: (General static knowledge)
- analytical: (Logic, reasoning, analysis)
- creative: (Ideas, writing)
- complex: (Multi-step)
- confirmation: (Simple agreement/acknowledgement)
- realtime_data: (Live data: price, weather, news - REQUIRES SEARCH)

Output ONLY valid JSON:
{{
  "intent": "category_name",
  "confidence": 0.0-1.0,
  "reasoning": "short explanation"
}}"""

        try:
            # Use generate_fast if possible
            if hasattr(self.llm, 'generate_fast'):
                response_text = await self.llm.generate_fast(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=150
                )
            else:
                response_text = await self.llm.generate(
                    prompt=prompt,
                    temperature=0.1
                )
                
            # Parse JSON
            # Clean up potential markdown code blocks
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            # Find the first { and last }
            start = clean_text.find('{')
            end = clean_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = clean_text[start:end]
                data = json.loads(json_str)
                
                intent = data.get("intent", "factual").lower()
                start_conf = data.get("confidence", 0.5)
                # Ensure confidence is float
                try:
                    confidence = float(start_conf)
                except:
                    confidence = 0.5
                    
                return intent, confidence
                
        except Exception as e:
            print(f"[IntentRouter] JSON Parsing Failed: {e}. Defaulting to factual.")
            pass
            
        # Fallback
        return "factual", 0.5
