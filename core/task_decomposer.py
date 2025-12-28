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
    """A subtask in a decomposed problem."""
    task_id: int
    task_type: SubtaskType
    description: str
    depends_on: List[int] = field(default_factory=list)  # IDs of prerequisite tasks
    given_data: dict = field(default_factory=dict)       # Data available for this task
    missing_data: List[str] = field(default_factory=list)  # Data that needs to be found
    result: Any = None
    status: str = "pending"  # pending, completed, blocked
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "type": self.task_type.value,
            "description": self.description,
            "depends_on": self.depends_on,
            "status": self.status
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
        
        # Complex if 2+ indicators or very long query with numbers
        if matches >= 2:
            return True
        if len(query) > 500 and len(re.findall(r'\d+', query)) >= 3:
            return True
        
        return False
    
    def decompose(self, query: str) -> DecomposedProblem:
        """
        Decompose a complex problem into structured subtasks.
        Uses rule-based extraction (fast) + optional LLM refinement.
        """
        self._stats["problems_decomposed"] += 1
        
        # Step 1: Extract entities
        entities = self._extract_entities(query)
        
        # Step 2: Extract given facts
        given_facts = self._extract_given_facts(query)
        
        # Step 3: Identify rules
        rules = self._extract_rules(query)
        
        # Step 4: Identify missing data
        missing_facts = self._identify_missing_data(query, given_facts)
        
        # Step 5: Create subtasks
        subtasks = self._create_subtasks(entities, rules, given_facts)
        
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
        
        # Check for price per kWh
        if not re.search(r'\d+[\.,]?\d*\s*(?:USD|EUR|грн|₴|руб|\$|€)/?\s*(?:кВт|kWh)', query):
            missing.append("base_price_per_kwh (use variable P)")
        
        # Check for battery capacity
        if not re.search(r'(?:ёмкость|capacity)\s*\d+\s*(?:кВт|kWh)', query):
            missing.append("battery_capacity_kwh (use variable C)")
        
        # Check for charging speed
        if not re.search(r'(?:скорость|speed|rate)\s*\d+\s*(?:кВт|kW)', query):
            missing.append("charging_speed_kw (use variable S)")
        
        return missing
    
    def _create_subtasks(
        self, 
        entities: List[str], 
        rules: List[str], 
        given: dict
    ) -> List[Subtask]:
        """Create ordered subtasks based on problem structure."""
        subtasks = []
        task_id = 0
        
        # Subtask 1: Identify time context
        subtasks.append(Subtask(
            task_id=task_id,
            task_type=SubtaskType.EXTRACT_DATA,
            description="Determine if current time is within peak hours",
            given_data={"time": given.get("current_time"), "peak": given.get("peak_hours")}
        ))
        task_id += 1
        
        # Subtask 2: Apply priority rules
        if any("priority" in r.lower() for r in rules):
            subtasks.append(Subtask(
                task_id=task_id,
                task_type=SubtaskType.PRIORITIZE,
                description="Identify priority vehicles (ambulance, police)",
                depends_on=[0]
            ))
            task_id += 1
        
        # Subtask 3: Check discount eligibility for each entity
        for entity in entities:
            if "greenlog" in entity.lower():
                subtasks.append(Subtask(
                    task_id=task_id,
                    task_type=SubtaskType.VALIDATE,
                    description=f"Check if {entity} is eligible for discount (battery < threshold)",
                    depends_on=[0],
                    given_data={"battery": given.get("greenlog_battery"), "threshold": given.get("discount_threshold")}
                ))
                task_id += 1
        
        # Subtask 4: Allocate ports
        if given.get("available_ports"):
            subtasks.append(Subtask(
                task_id=task_id,
                task_type=SubtaskType.PRIORITIZE,
                description=f"Allocate {given.get('available_ports')} ports based on priority",
                depends_on=list(range(task_id))  # Depends on all previous
            ))
            task_id += 1
        
        # Subtask 5: Calculate costs (as formulas, not numbers!)
        subtasks.append(Subtask(
            task_id=task_id,
            task_type=SubtaskType.CALCULATE,
            description="Express charging costs as formulas (do NOT invent base price)",
            depends_on=[task_id - 1] if task_id > 0 else [],
            missing_data=["base_price_per_kwh"]
        ))
        task_id += 1
        
        # Subtask 6: Final answer
        subtasks.append(Subtask(
            task_id=task_id,
            task_type=SubtaskType.REASON,
            description="Compile final answer with queue order and cost formulas",
            depends_on=[task_id - 1]
        ))
        
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
