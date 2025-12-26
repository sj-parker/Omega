# Semantic Validator Module
# Purpose: Prevent "Concept Drift" by cross-checking responses against Source of Truth.

from typing import Optional, List, Dict
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.llm_interface import LLMInterface

VALIDATOR_PROMPT = """Role: Technical Semantic Auditor. 
Task: You are a critic for a Multi-Agent System. Your goal is to prevent "Concept Drift".

Process:
1. Read Context Documents.
2. Read Draft Response.
3. EXTRACT JSON STATE if present.
4. Check Consistency: Does JSON 'result' match the Text's conclusion? (e.g. JSON says "overflow", Text says "All good" -> FAIL).
5. Identify Drift and Logic/Math errors.

Constraint: Do not be pedantic about style. Focus on:
- Consistency (JSON vs Text)
- Technical definitions (Drift)
- Resource constraints (Math)"""

CORRECTION_PROMPT = """The semantic auditor found some discrepancies in your response. 
Please rephrase the response to correctly use terms as defined in the Source of Truth.

Discrepancies:
{conflicts}

Source of Truth:
{source_of_truth}

Original Draft:
{draft}

Rephrased Response:"""

class SemanticValidator:
    """
    Semantic Guardrail / Validator.
    
    Functions:
    - Extract Key Entities (implicitly by LLM)
    - Cross-Check against context
    - Conflict Resolution
    """
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
    
    async def validate(self, draft_response: str, context_text: str) -> tuple[str, dict]:
        """
        Validate draft response against context.
        
        Returns:
        - Validated (potentially corrected) response
        - Validation report (dict)
        """
        if not context_text:
            return draft_response, {"status": "skipped", "reason": "no_context"}
            
        # Step 1: Check for drift/conflicts
        check_prompt = f"""Source of Truth:
{context_text}

Agent's Draft Response:
{draft_response}

Check for:
1. "Concept Drift"
2. CONSISTENCY: Compare any JSON state in the Draft with the Text. If JSON shows failure/overflow but Text says success => CONSISTENCY_FAIL.
3. Logic/Math errors.
4. NEGATIVE CONSTRAINTS: If context implies "Single" or "Exclusive", check if Output uses "Simultaneous" or "Both".

Output format:
DRIFT_FOUND: [YES/NO]
CONSISTENCY_FAIL: [YES/NO]
LOGIC_FAIL: [YES/NO]
CONSTRAINT_FAIL: [YES/NO]
CONFLICTS: [List of errors]
REASONING: [Brief explanation]"""

        analysis = await self.llm.generate(
            prompt=check_prompt,
            system_prompt=VALIDATOR_PROMPT,
            temperature=0.2
        )
        
        drift_found = False
        logic_fail = False
        consistency_fail = False
        constraint_fail = False
        conflicts = ""
        
        for line in analysis.split('\n'):
            if line.startswith("DRIFT_FOUND:"):
                drift_found = "YES" in line.upper()
            elif line.startswith("LOGIC_FAIL:"):
                logic_fail = "YES" in line.upper()
            elif line.startswith("CONSISTENCY_FAIL:"):
                consistency_fail = "YES" in line.upper()
            elif line.startswith("CONSTRAINT_FAIL:"):
                constraint_fail = "YES" in line.upper()
            elif line.startswith("CONFLICTS:"):
                conflicts = line.split(":", 1)[1].strip()
        
        failed = drift_found or logic_fail or consistency_fail or constraint_fail
        
        report = {
            "status": "passed" if not failed else "failed",
            "drift_found": drift_found,
            "logic_fail": logic_fail,
            "consistency_fail": consistency_fail,
            "constraint_fail": constraint_fail,
            "conflicts": conflicts,
            "raw_analysis": analysis
        }
        
        if not failed:
            return draft_response, report
            
        # Step 2: Correct response if drift found
        print(f"[Validator] Issues detected: {conflicts[:50]}...")
        
        correction_full_prompt = CORRECTION_PROMPT.format(
            conflicts=conflicts,
            source_of_truth=context_text[:1000], # Trucated for prompt
            draft=draft_response
        )
        
        corrected_response = await self.llm.generate(
            prompt=correction_full_prompt,
            system_prompt="You are a precise technical re-writer.",
            temperature=0.3
        )
        
        return corrected_response, report
