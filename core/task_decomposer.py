# Task Decomposer
# Breaks complex multi-step problems into subtasks

from dataclasses import dataclass, field
from typing import Optional, List, Any, TYPE_CHECKING
from enum import Enum
import re

if TYPE_CHECKING:
    from models.llm_interface import LLMInterface


class SubtaskType(Enum):
    """Type of subtask."""
    EXTRACT_DATA = "extract_data"       # Extract given data from problem
    IDENTIFY_RULES = "identify_rules"   # Identify rules/constraints
    CALCULATE = "calculate"              # Perform calculation
    COMPARE = "compare"                  # Compare options
    PRIORITIZE = "prioritize"            # Determine priority/order
    REASON = "reason"                    # Logical reasoning
    LOOKUP = "lookup"                    # Need external data
    VALIDATE = "validate"                # Check if answer is valid


@dataclass
class Subtask:
    """A single step in a problem solving plan."""
    task_id: int
    task_type: SubtaskType
    description: str
    depends_on: List[int] = field(default_factory=list)
    missing_data: List[str] = field(default_factory=list) # Data needed but not in context
    is_volatile: bool = False # Whether this lookup requires real-time data
    result: Any = None
    status: str = "pending"  # pending, completed, blocked
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "type": self.task_type.value,
            "description": self.description,
            "depends_on": self.depends_on,
            "given_data": self.given_data,
            "missing_data": self.missing_data,
            "status": self.status,
            "result": self.result
        }


@dataclass
class DecomposedProblem:
    """A complex problem broken into subtasks."""
    original_query: str
    entities: List[str]                   # Key entities in the problem
    given_facts: dict                     # Data explicitly provided
    missing_facts: List[str]              # Data NOT provided (don't hallucinate!)
    rules: List[str]                      # Rules/constraints to apply
    subtasks: List[Subtask]               # Ordered subtasks
    final_goal: str                       # What the answer should provide
    
    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "entities": self.entities,
            "given_facts": self.given_facts,
            "missing_facts": self.missing_facts,
            "rules": self.rules,
            "subtasks": [s.to_dict() for s in self.subtasks],
            "final_goal": self.final_goal
        }
    
    def get_next_subtask(self) -> Optional[Subtask]:
        """Get the next subtask that can be executed."""
        for task in self.subtasks:
            if task.status == "pending":
                # Check if all dependencies are completed
                deps_done = all(
                    self.subtasks[dep_id].status == "completed" 
                    for dep_id in task.depends_on
                    if dep_id < len(self.subtasks)
                )
                if deps_done:
                    return task
        return None
    
    def to_prompt(self) -> str:
        """Generate a structured prompt for LLM."""
        prompt_parts = [
            "## PROBLEM ANALYSIS",
            "",
            f"**Goal:** {self.final_goal}",
            "",
            "### GIVEN DATA (use ONLY this data):",
        ]
        
        for key, value in self.given_facts.items():
            prompt_parts.append(f"- {key}: {value}")
        
        if self.missing_facts:
            prompt_parts.append("")
            prompt_parts.append("### MISSING DATA (DO NOT INVENT!):")
            for fact in self.missing_facts:
                prompt_parts.append(f"- {fact} → use formula/variable instead")
        
        prompt_parts.append("")
        prompt_parts.append("### RULES TO APPLY:")
        for rule in self.rules:
            prompt_parts.append(f"- {rule}")
        
        prompt_parts.append("")
        prompt_parts.append("### ENTITIES:")
        for entity in self.entities:
            prompt_parts.append(f"- {entity}")
        
        return "\n".join(prompt_parts)


class TaskDecomposer:
    """
    Task Decomposer - Breaks complex problems into structured subtasks.
    
    Solves problems like:
    - Multi-entity priority decisions (EV charging)
    - Resource allocation with constraints
    - Multi-step calculations with rules
    
    Key principles:
    - Explicitly identify GIVEN vs MISSING data
    - Prevent hallucination of missing data
    - Create dependency graph of subtasks
    """
    
    # Patterns that indicate complex multi-step problems
    COMPLEX_INDICATORS = [
        r"(?:одновременно|simultaneously|at the same time)",
        r"(?:правило|rule|rules|constraint)",
        r"(?:исключение|exception|except)",
        r"(?:приоритет|priority|prioritize)",
        r"(?:очередь|order|queue|sequence)",
        r"(?:порт|port|slot|space).*\d+",
        r"(?:скидк|discount)",
        r"(?:час.*пик|peak.*hour)",
        r"\d+%.*\d+%.*\d+%",  # Multiple percentages
        # NEW PATTERNS FOR SEARCH & CONDITIONS
        r"(?:найди|найти|поиск|search|find|lookup)",
        r"(?:ставка|rate|цена|price|курс|exchange)",
        r"(?:кредит|loan|debt|mortgage)",
        r"(?:если|if|when|когда|condition|услови)",
        r"(?:налог|tax|fee)",
        r"(?:сколько.*стоит|how.*much|what.*is.*the.*price)"
    ]
    
    # Keywords to extract entities
    ENTITY_PATTERNS = [
        r"(?:машина|car|vehicle|truck|грузовик|tesla|ambulance|скорая)",
        r"(?:компания|company)[\s:]+[\"']?(\w+)[\"']?",
    ]
    
    def __init__(self, llm: Optional['LLMInterface'] = None):
        self.llm = llm
        self._stats = {
            "problems_decomposed": 0,
            "avg_subtasks": 0
        }
    
    def is_complex_problem(self, query: str) -> bool:
        """Check if the query is a complex multi-step problem."""
        query_lower = query.lower()
        matches = 0
        
        for pattern in self.COMPLEX_INDICATORS:
            if re.search(pattern, query_lower):
                matches += 1
        
        # Complex if enough indicators OR special keywords
        if matches >= 2:
            return True
        
        # 1. ALWAYS Trigger on calculation/resource problems
        CALCULATION_INDICATORS = [
            r"(?:расход|consumption|потреблен)",  # consumption
            r"(?:заряд|charge|battery|аккумулятор)",  # charging  
            r"(?:формул|formula)",  # explicit formula request
            r"(?:рассчит|вычисл|посчитай|calculate|compute)",  # explicit calculation
            r"(?:бюджет|budget|смета)",  # budget calculation
        ]
        
        for pattern in CALCULATION_INDICATORS:
            if re.search(pattern, query_lower):
                return True
            
        # 2. Trigger on conditional logic with parameters (if X then Y)
        if re.search(r"(?:если|if).*(?:\d+|%)", query_lower):
            return True
        
        # 3. Trigger on multi-intent queries (e.g., "fact AND calculation")
        conjunctions = [r"\s+и\s+", r"\s+а также\s+", r"\s+and\s+", r"\s+also\s+"]
        if any(re.search(c, query_lower) for c in conjunctions) and len(re.findall(r'\d+', query_lower)) >= 1:
            return True
            
        # 4. Trigger on long queries with multiple numbers
        if len(query) > 400 and len(re.findall(r'\d+', query)) >= 3:
            return True

        # 5. Check for simple lookups (False if NO calculation markers above)
        if matches < 3:
            SIMPLE_QUESTION_INDICATORS = [
                r"находит",                  
                r"штат",                     
                r"страна",                   
                r"город",                    
                r"weather|погода",           
                r"курс|price|цена",          
            ]
            for pattern in SIMPLE_QUESTION_INDICATORS:
                if re.search(pattern, query_lower):
                    return False  # Simple question, don't decompose
        
        # 6. Fallback to match count
        if matches >= 2:
            return True
            
        return False
    
    async def split_query(self, query: str) -> List[str]:
        """Split a query into independent sub-queries if needed."""
        if not self.llm:
            return [query]
            
        # Only split if query is long or has conjunctions
        conjunctions = [", а ", ", но ", " и ", " but ", " and ", " also "]
        if len(query) < 50 and not any(c in query.lower() for c in conjunctions):
            return [query]
            
        prompt = f"""Split this user query into a list of INDEPENDENT requests.
If it's a single request with multiple conditions/facts, return it as a list with ONE item.
Independent requests are UNRELATED questions (e.g., "How are you? AND what's the weather?").

⚠️ DO NOT split a single calculation or search task into sentences. 
Example: "Find BTC price. We have 15k budget. Calculate robots." -> This is ONE request.
Example: "Fact about Mars. Also, calculate 2+2." -> These are TWO requests.

User query: "{query}"

Output ONLY a JSON list of strings in the ORIGINAL query language: ["req1", "req2", ...]"""

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt="You are a query analysis expert. Split unrelated intents.",
                temperature=0.0
            )
            
            # Robust JSON extraction
            import json
            import re
            
            clean_res = response.strip()
            # Find the first [ and last ]
            match = re.search(r'\[.*\]', clean_res, re.DOTALL)
            if match:
                json_str = match.group(0)
                parts = json.loads(json_str)
                if isinstance(parts, list) and parts:
                    return parts
        except Exception as e:
            print(f"[TaskDecomposer] Query split failed: {e}. Raw response: {response[:100]}")
            
        return [query]

    async def decompose(self, query: str) -> DecomposedProblem:
        """
        Decompose a complex problem into structured subtasks.
        Uses recursive decomposition for multi-intent support.
        """
        # Multi-intent splitting is handled by Orchestrator calling split_query first.
        # This method handles structural decomposition of a SINGLE intent.
        
        self._stats["problems_decomposed"] += 1
        return self._do_stuctural_decompose(query)

    def _do_stuctural_decompose(self, query: str) -> DecomposedProblem:
        """The actual logic for breaking down a complex query."""
        # Step 1: Extract entities
        entities = self._extract_entities(query)
        
        # Step 2: Extract given facts
        given_facts = self._extract_given_facts(query)
        
        # Step 3: Identify rules
        rules = self._extract_rules(query)
        
        # Step 4: Identify missing data
        missing_facts = self._identify_missing_data(query, given_facts)
        
        # Step 5: Create subtasks
        subtasks = self._create_subtasks(query, entities, rules, given_facts)
        
        # Step 6: Determine final goal
        final_goal = self._extract_goal(query)
        
        problem = DecomposedProblem(
            original_query=query,
            entities=entities,
            given_facts=given_facts,
            missing_facts=missing_facts,
            rules=rules,
            subtasks=subtasks,
            final_goal=final_goal
        )
        
        self._stats["avg_subtasks"] = (
            self._stats["avg_subtasks"] * 0.9 + len(subtasks) * 0.1
        )
        
        return problem
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract key entities from the problem."""
        entities = []
        
        # Look for vehicle types
        vehicle_patterns = [
            (r"(?:грузовик|truck)\s*(?:GreenLog)?", "GreenLog Truck"),
            (r"(?:tesla|тесла)", "Tesla"),
            (r"(?:скорая|ambulance|скорой помощи)", "Ambulance"),
            (r"(?:полиц|police)", "Police"),
        ]
        
        query_lower = query.lower()
        for pattern, name in vehicle_patterns:
            if re.search(pattern, query_lower):
                entities.append(name)
        
        # Look for percentages with context
        percent_matches = re.findall(r'(\w+)[^\d]{0,30}(\d+)\s*%', query)
        for context, pct in percent_matches:
            if "заряд" in context.lower() or "charge" in context.lower() or "battery" in context.lower():
                # This is battery level, already captured with entity
                pass
        
        return list(set(entities))
    
    def _extract_given_facts(self, query: str) -> dict:
        """Extract explicitly given data from the problem."""
        facts = {}
        
        # Time
        time_match = re.search(r'(\d{1,2}:\d{2})', query)
        if time_match:
            facts["current_time"] = time_match.group(1)
        
        # Peak hours
        peak_match = re.search(r'(?:пик|peak)[^)]*\((\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})\)', query)
        if peak_match:
            facts["peak_hours"] = f"{peak_match.group(1)}-{peak_match.group(2)}"
        
        # Battery levels - look for "заряд X%" pattern near entity names
        # Pattern: Entity ... (заряд X%) or Entity (X%)
        battery_patterns = [
            (r"(?:GreenLog|грузовик)[^()]*\((?:заряд\s*)?(\d+)\s*%\)", "greenlog_battery"),
            (r"(?:Tesla|тесла)[^()]*\((?:заряд\s*)?(\d+)\s*%\)", "tesla_battery"),
            (r"(?:скорая|ambulance)[^()]*\((?:заряд\s*)?(\d+)\s*%\)", "ambulance_battery"),
        ]
        for pattern, key in battery_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                facts[key] = f"{match.group(1)}%"
        
        # Also try simpler pattern if parentheses didn't match
        if "greenlog_battery" not in facts:
            match = re.search(r"GreenLog.*?(\d+)\s*%", query, re.IGNORECASE)
            if match:
                # Make sure it's not the discount (which is typically 30%)
                pct = int(match.group(1))
                if pct < 20:  # Battery levels are typically low values
                    facts["greenlog_battery"] = f"{pct}%"
        
        # Discounts
        discount_match = re.search(r'скидк\w*\s*(\d+)\s*%', query)
        if discount_match:
            facts["greenlog_discount"] = f"{discount_match.group(1)}%"
        
        # Threshold for discount
        threshold_match = re.search(r'ниже\s*(\d+)\s*%', query)
        if threshold_match:
            facts["discount_threshold"] = f"{threshold_match.group(1)}%"
        
        # Number of ports
        port_match = re.search(r'(\d+)\s*(?:свободных?\s*)?(?:порт|port)', query)
        if port_match:
            facts["available_ports"] = int(port_match.group(1))
        
        # Price multiplier
        if re.search(r'удваивается|double|×2|x2', query, re.IGNORECASE):
            facts["peak_price_multiplier"] = 2

        # NEW: General unit extraction (Money, Crypto, Quantities)
        money_matches = re.findall(r'(\d{1,3}(?:[,\s]\d{3})*(?:[\.,]\d+)?)\s*(USD|EUR|GBP|BTC|ETH|\$|€|£|грн|₴|руб|₽)', query, re.I)
        for val, unit in money_matches:
            # Clean value (15,000 -> 15000)
            clean_val = val.replace(',', '').replace(' ', '')
            facts[f"amount_{unit.lower()}"] = float(clean_val)
            
        # Specific cost/budget extraction
        budget_match = re.search(r'(?:бюджет|budget|всего|total|есть|have)\s*(?:составляет)?\s*(\d{1,3}(?:[,\s]\d{3})*(?:[\.,]\d+)?)\s*(USD|EUR|GBP|BTC|ETH|\$|€|£|грн|₴|руб|₽)', query, re.I)
        if budget_match:
            facts["budget"] = float(budget_match.group(1).replace(',', '').replace(' ', ''))
            facts["budget_unit"] = budget_match.group(2).upper()
            
        cost_match = re.search(r'(?:стоит|цена|cost|price)\s*(\d{1,3}(?:[,\s]\d{3})*(?:[\.,]\d+)?)\s*(USD|EUR|GBP|BTC|ETH|\$|€|£|грн|₴|руб|₽)', query, re.I)
        if cost_match:
            facts["unit_cost"] = float(cost_match.group(1).replace(',', '').replace(' ', ''))
            facts["unit_cost_unit"] = cost_match.group(2).upper()

        return facts
    
    def _extract_rules(self, query: str) -> List[str]:
        """Extract rules and constraints from the problem."""
        rules = []
        
        # Priority rules
        if re.search(r'скорой|ambulance|полиц|police.*приоритет|priority', query, re.IGNORECASE):
            rules.append("PRIORITY: Emergency vehicles (ambulance, police) have priority and charge FREE")
        
        # Peak hour pricing
        if re.search(r'час.*пик.*удва|peak.*double', query, re.IGNORECASE):
            rules.append("PRICING: Price per kWh doubles during peak hours")
        
        # Conditional discount
        if re.search(r'скидк.*ниже|discount.*below|если.*%', query, re.IGNORECASE):
            rules.append("DISCOUNT: GreenLog gets 30% discount ONLY if battery < 10%")
        
        # Port limit
        if re.search(r'всего.*порт|only.*port', query, re.IGNORECASE):
            rules.append("CONSTRAINT: Limited charging ports available")
        
        return rules
    
    def _identify_missing_data(self, query: str, given: dict) -> List[str]:
        """Identify data that is NOT provided and should NOT be invented."""
        missing = []
        
        # 1. Dynamic extraction from explicit requests ("Find X", "Search for Y")
        # Improved regex to handle commas and optional words
        search_requests = re.findall(r"(?:узнай|найди|поиск|search|find|узнать|найти|какой|какая|какое)[\s,]+(?:чтобы\s+узнать\s+)?([^.!?\n]+)", query, re.I)
        for req in search_requests:
            # Clean up: remove "current", "price", etc. to get the core entity
            clean = re.sub(r"текущу\w*|current|live|now|сейчас|цену|цената|rate|price|курс|температуру|погоду|weather", "", req, flags=re.I).strip()
            # Also remove trailing prepositions or verbs
            clean = re.sub(r"\s+(?:в|на|for|at|in|is)\s*$", "", clean, flags=re.I).strip()
            
            if clean and len(clean) < 50 and clean not in missing:
                missing.append(clean)

        # 2. Heuristic extraction for common volatile data (symbols, rates)
        volatile_patterns = [
            (r"(?:цена|курс|price|rate)\s+([A-Z]{3,5})", "price of \1"),
            (r"(?:цена|курс|price|rate)\s+(?:биткоина|bitcoin|btc|эфира|ethereum|eth)", "Bitcoin price"),
            (r"(?:погода|weather)\s+(?:в|на|in|at)\s+([А-ЯA-Z][а-яa-z]+)", "weather in \1"),
        ]
        for pattern, label in volatile_patterns:
            match = re.search(pattern, query, re.I)
            if match:
                # Use simplified label if it's a known pattern
                if r"\1" in label:
                    fact = label.replace(r"\1", match.group(1).upper())
                else:
                    fact = label
                if fact not in missing:
                    missing.append(fact)

        # 3. Domain-specific (Charging) - only if relevant keywords exist
        charging_keywords = ["заряд", "квтч", "kwh", "станция", "charging", "tesla"]
        if any(k in query.lower() for k in charging_keywords):
            # Price per kWh - ONLY if user asks about money/cost
            if any(k in query.lower() for k in ["цена", "стоимость", "usd", "грн", "₴", "руб", "₽", "сколько стоит", "cost", "price"]):
                if not re.search(r'\d+[\.,]?\d*\s*(?:USD|EUR|грн|₴|руб|\$|€)/?\s*(?:кВт|kWh)', query):
                    missing.append("base_price_per_kwh (use variable P)")
            
            # Battery capacity - only if explicitly mentioned without value
            if any(k in query.lower() for k in ["ёмкость", "capacity", "объем"]):
                if not re.search(r'(?:ёмкость|capacity|объем)\s*\d+\s*(?:кВт|kWh|л|l|гал|gal)', query):
                    missing.append("battery_capacity_kwh (use variable C)")
            
            # Charging speed - only if explicitly mentioned without value
            if any(k in query.lower() for k in ["скорость", "speed", "мощность", "power"]):
                if not re.search(r'(?:скорость|speed|мощность|power)\s*\d+\s*(?:кВт|kW)', query):
                    missing.append("charging_speed_kw (use variable S)")
        
        # 4. Geographical Data (Distance)
        # Matches "из Полтавы в Одессу", "от Киева до Львова", "from London to Paris"
        geo_trip_match = re.search(r"(?:из|от|from)\s+([А-ЯA-Z][а-яa-z]+)\s+(?:в|до|to)\s+([А-ЯA-Z][а-яa-z]+)", query)
        if geo_trip_match:
            missing.append(f"расстояние между {geo_trip_match.group(1)} и {geo_trip_match.group(2)}")
            
        # Explicit distance request
        dist_req_match = re.search(r"(?:расстояние|distance).*(?:между|between)\s+([А-ЯA-Z][а-яa-z]+)\s+(?:и|and)\s+([А-ЯA-Z][а-яa-z]+)", query, re.I)
        if dist_req_match:
            missing.append(f"расстояние между {dist_req_match.group(1)} и {dist_req_match.group(2)}")

        return missing
    
    def _create_subtasks(
        self, 
        query: str,
        entities: List[str], 
        rules: List[str], 
        given: dict
    ) -> List[Subtask]:
        """Create ordered subtasks based on problem structure."""
        subtasks = []
        task_id = 0
        # Step 1: Add LOOKUP tasks for missing facts ALWAYS
        missing_data = self._identify_missing_data(query, given)
        lookup_task_ids = []
        for fact in missing_data:
            # MARK volatile factual data (price, weather, distance etc)
            volatile_kws = ["price", "цена", "курс", "rate", "weather", "погода", "btc", "bitcoin", "crypto", "расстояние", "distance"]
            is_vol = any(kw in fact.lower() for kw in volatile_kws)
            
            subtasks.append(Subtask(
                task_id=task_id,
                task_type=SubtaskType.LOOKUP,
                description=f"Search for: {fact}",
                depends_on=[],
                is_volatile=is_vol
            ))
            lookup_task_ids.append(task_id)
            task_id += 1

        # Step 2: Domain-specific Subtasks
        # Subtasks for Charging Domain
        charging_keywords = ["заряд", "квтч", "kwh", "станция", "charging"]
        is_charging_task = any(k in " ".join(entities).lower() or k in query.lower() for k in charging_keywords)

        if is_charging_task:
            subtasks.append(Subtask(
                task_id=task_id,
                task_type=SubtaskType.CALCULATE,
                description="Express charging costs as formulas",
                depends_on=lookup_task_ids, # Wait for prices if needed
                missing_data=["base_price_per_kwh"]
            ))
            task_id += 1
            
        # Step 3: Generic logic-math subtasks
        # Analyze phase (depends on lookups)
        subtasks.append(Subtask(
            task_id=task_id,
            task_type=SubtaskType.REASON,
            description="Analyze rules, conditions and retrieved data",
            depends_on=lookup_task_ids # Wait for all lookups
        ))
        reason_task_id = task_id
        task_id += 1
        
        # CALCULATE MUST depend on REASON
        subtasks.append(Subtask(
            task_id=task_id,
            task_type=SubtaskType.CALCULATE,
            description="Perform final calculation based on determined values",
            depends_on=[reason_task_id]
        ))
        calc_task_id = task_id
        task_id += 1
        
        # Final answer MUST depend on CALCULATE
        subtasks.append(Subtask(
            task_id=task_id,
            task_type=SubtaskType.REASON,
            description="Compile final answer with queue order and cost formulas",
            depends_on=[calc_task_id]
        ))
        task_id += 1
            
        return subtasks
    
    def _extract_goal(self, query: str) -> str:
        """Extract what the final answer should provide."""
        goals = []
        
        if re.search(r'кого.*заряжа|who.*charg', query, re.IGNORECASE):
            goals.append("who gets charged")
        if re.search(r'очеред|order|queue', query, re.IGNORECASE):
            goals.append("charging order/queue")
        if re.search(r'стоимость|cost|price|сколько', query, re.IGNORECASE):
            goals.append("cost for each (formula if base price unknown)")
        if re.search(r'обоснуй|explain|justify', query, re.IGNORECASE):
            goals.append("justification based on rules")
        
        return " + ".join(goals) if goals else "Provide a complete answer"
    
    async def decompose_with_llm(self, query: str) -> DecomposedProblem:
        """Use LLM to enhance decomposition for very complex problems."""
        # First do rule-based decomposition
        problem = self.decompose(query)
        
        if not self.llm:
            return problem
        
        # Optionally refine with LLM
        refinement_prompt = f"""Analyze this problem and identify any MISSING constraints or entities.

Problem: {query}

Currently identified:
- Entities: {problem.entities}
- Given facts: {problem.given_facts}
- Missing data (DO NOT INVENT): {problem.missing_facts}
- Rules: {problem.rules}

If anything is missing, output:
MISSING_ENTITY: [entity]
MISSING_RULE: [rule]
MISSING_FACT: [fact that should NOT be invented]

Otherwise output: COMPLETE"""

        try:
            response = await self.llm.generate(
                prompt=refinement_prompt,
                system_prompt="You are a problem analysis expert. Be precise about what data is GIVEN vs MISSING.",
                temperature=0.2
            )
            
            # Parse refinements
            for line in response.split('\n'):
                if line.startswith("MISSING_ENTITY:"):
                    entity = line.split(":", 1)[1].strip()
                    if entity not in problem.entities:
                        problem.entities.append(entity)
                elif line.startswith("MISSING_RULE:"):
                    rule = line.split(":", 1)[1].strip()
                    if rule not in problem.rules:
                        problem.rules.append(rule)
                elif line.startswith("MISSING_FACT:"):
                    fact = line.split(":", 1)[1].strip()
                    if fact not in problem.missing_facts:
                        problem.missing_facts.append(fact)
        except Exception as e:
            print(f"[TaskDecomposer] LLM refinement failed: {e}")
        
        return problem
    
    def get_structured_prompt(self, problem: DecomposedProblem) -> str:
        """Generate a prompt that prevents hallucination."""
        return problem.to_prompt() + """

## CRITICAL INSTRUCTIONS

⚠️ DO NOT INVENT DATA that is not explicitly given above.
⚠️ If base price is unknown, express cost as FORMULA: Cost = P × kWh × multiplier × (1 - discount)
⚠️ If battery capacity is unknown, do NOT calculate exact kWh amounts.

Answer the problem by:
1. Processing subtasks in order
2. Using ONLY given data
3. Expressing unknown values as formulas/variables
"""
    
    def get_stats(self) -> dict:
        """Get decomposer statistics."""
        return self._stats
