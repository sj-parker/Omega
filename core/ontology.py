"""
Omega Ontology Registry

Defines the actual components of the Omega system.
Used to prevent fabrication of non-existent modules.
"""

import re
from typing import Optional

# Self-Identity: What the system knows about itself
SELF_IDENTITY = """You are Omega - a cognitive AI system with modular architecture.
Your modules: OperationalModule, ExpertsModule, CriticModule, ContextManager.
You do NOT have: emotions, consciousness, subjective experience, influence on human history.
If asked about 'Omega' without context, assume they mean YOU (the system), not something else.
"""

# Registry of REAL Omega components
OMEGA_REGISTRY = {
    "modules": [
        "OperationalModule", "ExpertsModule", "CriticModule",
        "ContextManager", "Gatekeeper", "Validator", "LearningDecoder",
        "ReflectionController", "SearchEngine", "ToolsRegistry"
    ],
    "experts": [
        "neutral", "creative", "conservative", 
        "adversarial", "forecaster", "physics"
    ],
    "tools": [
        "search_and_extract", "verify_fact", 
        "calculate_linear_change", "calculate_resource_allocation"
    ],
    "concepts": [
        "DecisionDepth", "PolicySpace", "ContextSlice", 
        "WorldState", "ExpertResponse", "CriticAnalysis"
    ],
    "paths": [
        "fast", "medium", "deep"
    ]
}

# Keywords that indicate query is about Omega internals
INTERNAL_QUERY_PATTERNS = [
    r"модуль\s+omega", r"omega\s+модуль", r"module\s+omega",
    r"внутренн\w+\s+omega", r"internal\s+omega", r"omega\s+internal",
    r"архитектур\w+\s+omega", r"omega\s+architec", r"omega\s+component",
    r"omega\s+систем", r"система\s+omega", r"omega\s+system",
    r"omega-\w+", r"омега\s+модуль", r"модуль\s+омега",
    # Expanded: Omega alone when talking about influence/history/actions
    r"омега\s+повлиял", r"omega\s+влия", r"как\s+омега", r"how\s+omega",
    r"омега\s+сделал", r"omega\s+did", r"что\s+омега", r"what\s+omega",
    r"история\s+омега", r"omega\s+histor", r"расскажи.*омега"
]

# Patterns that should NEVER trigger search
SEARCH_BLOCKED_PATTERNS = [
    # Math expressions (e.g., "178 * 24", "5 + 3")
    r"\d+\s*[\*\+\-\/\×\÷]\s*\d+",
    
    # Self-analysis (expanded)
    r"\bсебе\b", r"\bсебя\b", r"\bтвоя\b", r"\bты сам\b", 
    r"\byourself\b", r"\bwhat are you\b", r"\bwho are you\b",
    r"внутренн\w+\s+параметр", r"твои\s+параметр", r"свои\s+параметр",
    r"почему\s+ты\s+не", r"можешь\s+ли\s+ты", r"ты\s+не\s+можешь",
    r"автономн", r"autonomous", r"self-aware", r"сознани",
    r"субъективн", r"subjective", r"experience",
    
    # Internal architecture (when combined with omega-like words)
    r"omega.*модуль", r"модуль.*omega", r"omega.*компонент",
    r"omega.*архитектур", r"архитектур.*omega",
    r"опиши.*omega", r"omega.*описа", r"explain.*omega"
]


def is_internal_query(text: str) -> bool:
    """Check if query is asking about Omega internal components."""
    text_lower = text.lower()
    for pattern in INTERNAL_QUERY_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def extract_entity_name(text: str) -> Optional[str]:
    """Try to extract the entity name being asked about."""
    text_lower = text.lower()
    
    # Common patterns: "модуль X", "X module", "компонент Y"
    patterns = [
        r"модуль\s+[\"']?(\w+)[\"']?",
        r"module\s+[\"']?(\w+)[\"']?",
        r"компонент\s+[\"']?(\w+)[\"']?",
        r"component\s+[\"']?(\w+)[\"']?"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    return None


def entity_exists(name: str) -> bool:
    """Check if entity exists in Omega registry."""
    name_lower = name.lower()
    
    # Check all registry categories
    for category, items in OMEGA_REGISTRY.items():
        for item in items:
            if item.lower() == name_lower or name_lower in item.lower():
                return True
    
    return False


def should_block_search(text: str) -> tuple[bool, str]:
    """
    Check if search should be blocked for this query.
    Returns (should_block, reason).
    """
    text_lower = text.lower()
    
    for pattern in SEARCH_BLOCKED_PATTERNS:
        if re.search(pattern, text_lower):
            # Determine reason
            if re.search(r"\d+\s*[\*\+\-\/\×\÷]\s*\d+", text_lower):
                return True, "math_expression"
            elif re.search(r"себ|yourself|who are you", text_lower):
                return True, "self_analysis"
            elif re.search(r"omega|омега", text_lower):
                return True, "internal_architecture"
            return True, "blocked_pattern"
    
    return False, ""


def get_ontology_response(entity_name: str) -> str:
    """Generate response for unknown internal entity."""
    return f"""Модуль "{entity_name}" не существует в архитектуре Omega.

Реальные модули Omega: {', '.join(OMEGA_REGISTRY['modules'][:5])}...

Если вы хотите узнать о реальной архитектуре Omega, уточните запрос."""
