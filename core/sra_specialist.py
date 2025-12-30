# SRA Specialist (Semantic Requirements Analyser)
# Identifies exact data requirements from a query using LLM analysis

import json
import re
from typing import List, Dict, Any, Optional
from models.llm_interface import LLMInterface
from models.schemas import WorldState

SRA_PROMPT = """[SYSTEM MESSAGE]
You are OMEGA-SRA (Semantic Requirements Analyser).
Your job is to identify every piece of EXTERNAL or DYNAMIC information needed to solve the user's problem.

🚨 GOAL: Create a "Shopping List" of facts.
DO NOT solve the problem.
DO NOT invent data.
ONLY identify what is MISSING and needed for an accurate calculation/answer.

### RULES:
1. **ENTITY**: What/Who are we looking for? (e.g., "Bitcoin", "Poltava").
2. **VARIABLE**: What specific property? (e.g., "price", "distance").
3. **DOMAIN**: geography, finance, temporal, technical, general.
4. **PRIORITY**: `high` (essential) or `low` (enrichment).
5. **LOGIC vs DATA (CRITICAL)**: 
   - If the request is a LOGIC PUZZLE, RIDDLE, or MATH WORD PROBLEM (e.g., "A > B and B > C..."), and ALL numbers/rules are provided in the text, return an EMPTY LIST `[]`.
   - ONLY identify requirements for missing EXTERNAL data (e.g., "What is the price of BTC?").
   - DO NOT identify search terms for solving puzzles.

### OUTPUT FORMAT (JSON ONLY):
```json
[
  {
    "entity": "Bitcoin",
    "variable": "current price",
    "domain": "finance",
    "priority": "high",
    "volatility": "high"
  }
]
```

### VOLATILITY GUIDE:
- `high`: Changes hourly/daily (Prices, Weather, News, Showtimes, Crypto).
- `medium`: Changes monthly/yearly (Policies, Features, Versions).
- `low`: Static or historical facts (Distances, Birthdays, History).

### EXAMPLES:
User: "If price > 100, buy. Search price of BTC."
SRA: [{"entity": "BTC", "variable": "current price", "domain": "finance", "priority": "high", "volatility": "high"}]

User: "What famous events happened on this day?" (Context: Today is 30.12.2025)
SRA: [{"entity": "December 30", "variable": "historical events", "domain": "general", "priority": "high", "volatility": "low"}]

User: "Showtimes for Spider-Man in NYC."
SRA: [{"entity": "Spider-Man in NYC", "variable": "showtimes", "domain": "general", "priority": "high", "volatility": "high"}]
"""

class SRASpecialist:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    async def identify_requirements(self, query: str, context: str = "") -> List[Dict[str, Any]]:
        """Analyse query to find missing data requirements."""
        prompt = f"Query: {query}\nContext: {context}\n\nList requirements:"
        
        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=SRA_PROMPT,
                temperature=0.1 # High precision
            )
            
            # Extract JSON from response
            match = re.search(r"(\[.*\])", response, re.DOTALL)
            if match:
                raw_json = match.group(1)
                return json.loads(raw_json)
            
            return []
        except Exception as e:
            print(f"[SRA] Error: {e}")
            return []

    def filter_existing_facts(self, requirements: List[Dict[str, Any]], world_state: WorldState) -> List[Dict[str, Any]]:
        """Remove requirements that are already present in the world state."""
        missing = []
        data_keys_lower = [k.lower() for k in world_state.data.keys()]
        
        for req in requirements:
            # Simple check: is entity or variable in world_state keys?
            found = False
            target = f"{req.get('entity', '')} {req.get('variable', '')}".lower()
            for key in data_keys_lower:
                if key in target or target in key:
                    found = True
                    break
            
            if not found:
                missing.append(req)
        
        return missing
