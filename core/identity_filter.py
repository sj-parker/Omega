# Identity Filter
# Removes LLM self-identification from responses

import re
from typing import Optional


# Patterns to detect and remove
IDENTITY_PATTERNS = [
    # Model names
    r"\b(qwen|qwen2\.?5?|phi3?|phi-?3|gemma|llama|claude|gpt|openai|anthropic)\b",
    # Self-identification
    r"(я|i)\s+(—|am|являюсь)\s+(языков\w+\s+модел\w+|language\s+model|an?\s+ai|искусственн\w+\s+интеллект\w*)",
    r"(as\s+an?\s+ai|как\s+ии|как\s+языковая\s+модель)",
    r"(i\'?m\s+an?\s+ai|i\s+am\s+an?\s+ai)",
    # Company references are removed to allow source attribution (Google, etc.)
]

# Phrases to remove entirely
REMOVE_PHRASES = [
    "I'm Qwen",
    "I am Qwen",
    "я Qwen",
    "Меня зовут Qwen",
    "My name is Qwen",
    "created by Alibaba",
    "developed by Alibaba",
]


class IdentityFilter:
    """
    Filters LLM responses to remove self-identification.
    
    Prevents the cognitive system from learning that it's Qwen/Phi/etc.
    """
    
    def __init__(self, replacement: str = "[система]"):
        self.replacement = replacement
        self.patterns = [re.compile(p, re.IGNORECASE) for p in IDENTITY_PATTERNS]
        self.remove_phrases = [p.lower() for p in REMOVE_PHRASES]
    
    def filter_response(self, response: str) -> str:
        """Filter identity mentions from a response."""
        result = response
        
        # Remove exact phrases first
        for phrase in self.remove_phrases:
            # Case-insensitive replace
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            result = pattern.sub("", result)
        
        # Apply regex patterns
        for pattern in self.patterns:
            result = pattern.sub(self.replacement, result)
        
        # Clean up multiple spaces and orphaned punctuation
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'\s*,\s*,', ',', result)
        result = re.sub(r'^\s*[,\.]\s*', '', result)
        
        return result.strip()
    
    def filter_dict(self, data: dict) -> dict:
        """Recursively filter all string values in a dictionary."""
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.filter_response(value)
            elif isinstance(value, dict):
                result[key] = self.filter_dict(value)
            elif isinstance(value, list):
                result[key] = self.filter_list(value)
            else:
                result[key] = value
        return result
    
    def filter_list(self, data: list) -> list:
        """Recursively filter all string values in a list."""
        result = []
        for item in data:
            if isinstance(item, str):
                result.append(self.filter_response(item))
            elif isinstance(item, dict):
                result.append(self.filter_dict(item))
            elif isinstance(item, list):
                result.append(self.filter_list(item))
            else:
                result.append(item)
        return result
    
    def check_contains_identity(self, text: str) -> bool:
        """Check if text contains identity mentions."""
        for pattern in self.patterns:
            if pattern.search(text):
                return True
        return False


# Global filter instance
_identity_filter = IdentityFilter()


def filter_llm_response(response: str) -> str:
    """Convenience function to filter a response."""
    return _identity_filter.filter_response(response)


def filter_data(data: dict) -> dict:
    """Convenience function to filter a dictionary."""
    return _identity_filter.filter_dict(data)
