# Response Sanitizer
# Prevents leakage of sensitive data in responses

import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from datetime import datetime


@dataclass
class SanitizationResult:
    """Result of sanitization."""
    original_length: int
    sanitized_length: int
    redactions_count: int
    redaction_types: list[str] = field(default_factory=list)
    sanitized_text: str = ""
    
    @property
    def was_modified(self) -> bool:
        return self.redactions_count > 0


class ResponseSanitizer:
    """
    Response Sanitizer - Prevents leakage of sensitive data.
    
    Features:
    - Detects and redacts sensitive patterns (passwords, API keys, emails)
    - Filters irrelevant context from responses
    - Prevents "dumping everything" when search fails
    """
    
    # Sensitive patterns with their types
    SENSITIVE_PATTERNS = [
        # Passwords (various formats)
        (r"(?:password|пароль|pass|pwd)\s*[:=\-]\s*[^\s,;]+", "password"),
        (r"(?:password|пароль|pass|pwd)\s+(?:is|это|равен)\s+[^\s,;]+", "password"),
        
        # API Keys and Tokens
        (r"(?:api[_-]?key|apikey)\s*[:=]\s*[a-zA-Z0-9_\-]{16,}", "api_key"),
        (r"(?:secret[_-]?key|secretkey)\s*[:=]\s*[a-zA-Z0-9_\-]{16,}", "secret_key"),
        (r"(?:access[_-]?token|token)\s*[:=]\s*[a-zA-Z0-9_\-\.]{20,}", "token"),
        (r"(?:bearer)\s+[a-zA-Z0-9_\-\.]{20,}", "bearer_token"),
        
        # Private keys
        (r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", "private_key"),
        
        # Connection strings
        (r"(?:mongodb|mysql|postgresql|redis)://[^\s]+", "connection_string"),
        (r"(?:jdbc|odbc):[^\s]+", "connection_string"),
        
        # Credit card patterns (basic)
        (r"\b(?:\d{4}[\s\-]?){3}\d{4}\b", "credit_card"),
        
        # CVV
        (r"(?:cvv|cvc|cvc2|cvv2)\s*[:=]\s*\d{3,4}", "cvv"),
        
        # Phone numbers (various formats) - marked as PII
        (r"\+?\d{1,3}[\s\-]?\(?\d{2,4}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}", "phone"),
        
        # Social security / Tax ID patterns (US, RU)
        (r"\b\d{3}[\s\-]?\d{2}[\s\-]?\d{4}\b", "ssn"),  # US SSN
        (r"\b\d{12}\b", "tax_id"),  # RU INN
        
        # AWS Keys
        (r"(?:AKIA|A3T|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}", "aws_key"),
        
        # Generic secrets
        (r"(?:secret|token|key)\s*[:=]\s*['\"][^'\"]{8,}['\"]", "generic_secret"),
    ]
    
    # Patterns that indicate the response is "dumping everything"
    DUMP_INDICATORS = [
        r"(?:вот всё|here is everything|всё что есть|all data|all information)",
        r"(?:нашёл следующее|found the following).*?(?:\n.*?){5,}",  # Many items listed
    ]
    
    # Low relevance patterns (filler content)
    LOW_RELEVANCE_PATTERNS = [
        r"(?:как я уже говорил|as I mentioned|ранее упоминалось)",
        r"(?:в контексте нашего разговора|in the context of our conversation)",
        r"(?:если вернуться к|going back to)",
    ]
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize sanitizer.
        
        Args:
            strict_mode: If True, redact all sensitive patterns. 
                        If False, only redact high-confidence patterns.
        """
        self.strict_mode = strict_mode
        self._compiled_patterns: List[Tuple[re.Pattern, str]] = []
        self._compile_patterns()
        
        # Stats
        self._stats = {
            "total_sanitized": 0,
            "total_redactions": 0,
            "by_type": {}
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for performance."""
        for pattern, pattern_type in self.SENSITIVE_PATTERNS:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                self._compiled_patterns.append((compiled, pattern_type))
            except re.error as e:
                print(f"[Sanitizer] Failed to compile pattern '{pattern}': {e}")
    
    def sanitize(self, text: str, context: str = "") -> SanitizationResult:
        """
        Sanitize response text by removing sensitive data.
        
        Args:
            text: The response text to sanitize
            context: Optional context for relevance filtering
            
        Returns:
            SanitizationResult with sanitized text and metadata
        """
        if not text:
            return SanitizationResult(
                original_length=0,
                sanitized_length=0,
                redactions_count=0,
                sanitized_text=""
            )
        
        self._stats["total_sanitized"] += 1
        
        result_text = text
        redaction_types = []
        redactions_count = 0
        
        # Apply all sensitive patterns
        for compiled_pattern, pattern_type in self._compiled_patterns:
            matches = compiled_pattern.findall(result_text)
            if matches:
                for match in matches:
                    # Redact with type indicator
                    redacted = f"[REDACTED:{pattern_type.upper()}]"
                    result_text = compiled_pattern.sub(redacted, result_text, count=1)
                    redactions_count += 1
                    if pattern_type not in redaction_types:
                        redaction_types.append(pattern_type)
                    
                    # Update stats
                    self._stats["total_redactions"] += 1
                    self._stats["by_type"][pattern_type] = self._stats["by_type"].get(pattern_type, 0) + 1
        
        # Check for dump indicators
        if self._is_dumping(result_text):
            # If dumping, filter to only relevant content or truncate
            result_text = self._filter_dump(result_text, context)
        
        return SanitizationResult(
            original_length=len(text),
            sanitized_length=len(result_text),
            redactions_count=redactions_count,
            redaction_types=redaction_types,
            sanitized_text=result_text
        )
    
    def _is_dumping(self, text: str) -> bool:
        """Check if the response is dumping all available data."""
        # Only check explicit dump indicators, NOT length
        # Long responses are often legitimate (complex queries like EV charging)
        for pattern in self.DUMP_INDICATORS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # DISABLED: Length-based truncation was too aggressive
        # Complex queries (like EV charging priority) produce long valid responses
        # if len(text) > 2000:
        #     sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        #     if len(sentences) > 20:
        #         return True
        
        return False
    
    def _filter_dump(self, text: str, context: str) -> str:
        """Filter dumped content to relevant parts only."""
        # MUCH higher limit - only truncate truly excessive responses
        if not context:
            if len(text) > 10000:  # Increased from 1500
                return text[:10000] + "\n\n[Ответ сокращён для краткости]"
            return text
        
        # Split into paragraphs and filter by relevance
        paragraphs = text.split('\n\n')
        context_words = set(context.lower().split())
        
        relevant_paragraphs = []
        for para in paragraphs:
            para_words = set(para.lower().split())
            # Calculate simple overlap
            overlap = len(context_words & para_words)
            if overlap >= 1 or len(para_words) < 15:  # More permissive: was overlap >= 2
                relevant_paragraphs.append(para)
        
        if relevant_paragraphs:
            return '\n\n'.join(relevant_paragraphs[:15])  # Increased from 5 to 15
        
        # Fallback: return most of the text
        return text[:5000] + "\n\n[Содержимое отфильтровано по релевантности]"
    
    def detect_sensitive_patterns(self, text: str) -> List[dict]:
        """
        Detect sensitive patterns without sanitizing.
        Useful for logging/auditing.
        """
        detections = []
        
        for compiled_pattern, pattern_type in self._compiled_patterns:
            matches = compiled_pattern.findall(text)
            if matches:
                detections.append({
                    "type": pattern_type,
                    "count": len(matches),
                    # Don't include the actual match for security
                    "pattern_hint": pattern_type.upper()
                })
        
        return detections
    
    def filter_irrelevant_context(self, response: str, query: str) -> str:
        """
        Remove context that doesn't relate to the query.
        Prevents "all the things" responses.
        """
        if not query or not response:
            return response
        
        # Extract query keywords
        query_words = set(word.lower() for word in query.split() if len(word) > 3)
        
        # Split response into chunks (sentences or paragraphs)
        chunks = self._split_into_chunks(response)
        
        relevant_chunks = []
        for chunk in chunks:
            chunk_words = set(word.lower() for word in chunk.split() if len(word) > 3)
            
            # Calculate relevance score
            if query_words:
                overlap = len(query_words & chunk_words)
                relevance = overlap / len(query_words)
            else:
                relevance = 1.0  # Keep if no query words
            
            if relevance >= 0.2 or len(chunk_words) < 5:
                relevant_chunks.append(chunk)
        
        if not relevant_chunks:
            return response  # Keep original if nothing matches
        
        return ' '.join(relevant_chunks)
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into meaningful chunks."""
        # First try paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            return paragraphs
        
        # Fallback to sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        return sentences
    
    def get_stats(self) -> dict:
        """Get sanitizer statistics."""
        return {
            **self._stats,
            "patterns_loaded": len(self._compiled_patterns)
        }
    
    def add_custom_pattern(self, pattern: str, pattern_type: str):
        """Add a custom sensitive pattern."""
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            self._compiled_patterns.append((compiled, pattern_type))
            print(f"[Sanitizer] Added custom pattern for '{pattern_type}'")
        except re.error as e:
            print(f"[Sanitizer] Invalid pattern '{pattern}': {e}")
