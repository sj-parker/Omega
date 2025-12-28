# Fallback Generator
# Generates honest "I don't know" responses when information is insufficient

import random
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class UncertaintyLevel(Enum):
    """Level of uncertainty in the response."""
    LOW = "low"         # Some info found, but not complete
    MEDIUM = "medium"   # Little info found
    HIGH = "high"       # No reliable info found
    TOTAL = "total"     # Complete failure to find anything


@dataclass
class FallbackResponse:
    """A generated fallback response."""
    text: str
    level: UncertaintyLevel
    suggested_actions: list[str]
    is_apologetic: bool = True


class FallbackGenerator:
    """
    Fallback Generator - Creates honest uncertainty responses.
    
    Instead of hallucinating or dumping irrelevant data,
    the system admits when it doesn't know something and
    offers constructive alternatives.
    
    This is a key component for trustworthy AI behavior.
    """
    
    # Templates for uncertainty (Russian)
    UNCERTAINTY_TEMPLATES_RU = {
        UncertaintyLevel.LOW: [
            "Я нашёл частичную информацию по теме '{topic}', но полной уверенности нет. {suggestion}",
            "По запросу '{topic}' есть неполные данные. {suggestion}",
            "Информация о '{topic}' ограничена. {suggestion}",
        ],
        UncertaintyLevel.MEDIUM: [
            "К сожалению, достоверной информации по '{topic}' мне найти не удалось. {suggestion}",
            "Мне не удалось найти надёжных источников по '{topic}'. {suggestion}",
            "Запрос '{topic}' не дал достаточных результатов. {suggestion}",
        ],
        UncertaintyLevel.HIGH: [
            "Я не нашёл информации по теме '{topic}'. {suggestion}",
            "По запросу '{topic}' у меня нет данных. {suggestion}",
            "'{topic}' — информация недоступна. {suggestion}",
        ],
        UncertaintyLevel.TOTAL: [
            "Мне не удалось обработать этот запрос. Попробуйте переформулировать.",
            "Произошла ошибка при обработке. Можете уточнить вопрос?",
            "Не могу ответить на этот вопрос. {suggestion}",
        ],
    }
    
    # Templates for uncertainty (English)
    UNCERTAINTY_TEMPLATES_EN = {
        UncertaintyLevel.LOW: [
            "I found some partial information about '{topic}', but I'm not fully confident. {suggestion}",
            "There's limited data on '{topic}'. {suggestion}",
            "Information about '{topic}' is incomplete. {suggestion}",
        ],
        UncertaintyLevel.MEDIUM: [
            "Unfortunately, I couldn't find reliable information on '{topic}'. {suggestion}",
            "I wasn't able to find trustworthy sources for '{topic}'. {suggestion}",
            "The search for '{topic}' didn't yield sufficient results. {suggestion}",
        ],
        UncertaintyLevel.HIGH: [
            "I couldn't find any information on '{topic}'. {suggestion}",
            "I don't have data for '{topic}'. {suggestion}",
            "'{topic}' — information unavailable. {suggestion}",
        ],
        UncertaintyLevel.TOTAL: [
            "I was unable to process this request. Please try rephrasing.",
            "An error occurred during processing. Could you clarify your question?",
            "I cannot answer this question. {suggestion}",
        ],
    }
    
    # Suggestions for what to do next
    SUGGESTIONS_RU = [
        "Попробуйте уточнить запрос.",
        "Могу помочь с чем-то другим?",
        "Можете переформулировать вопрос?",
        "Попробуйте задать более конкретный вопрос.",
        "Возможно, есть другой способ получить эту информацию.",
    ]
    
    SUGGESTIONS_EN = [
        "Try refining your query.",
        "Can I help with something else?",
        "Could you rephrase your question?",
        "Try asking a more specific question.",
        "There might be another way to get this information.",
    ]
    
    # Clarification question templates
    CLARIFICATION_TEMPLATES_RU = [
        "Не могли бы вы уточнить, что именно вы имеете в виду под '{topic}'?",
        "Для более точного ответа мне нужно больше контекста. Можете пояснить '{topic}'?",
        "'{topic}' может означать разное. Что конкретно вас интересует?",
        "Уточните, пожалуйста: вы спрашиваете о {option_a} или о {option_b}?",
    ]
    
    CLARIFICATION_TEMPLATES_EN = [
        "Could you clarify what you mean by '{topic}'?",
        "For a more accurate answer, I need more context. Can you explain '{topic}'?",
        "'{topic}' could mean different things. What specifically interests you?",
        "Please clarify: are you asking about {option_a} or about {option_b}?",
    ]
    
    def __init__(self, default_language: str = "ru"):
        """
        Initialize fallback generator.
        
        Args:
            default_language: Default language for responses ("ru" or "en")
        """
        self.default_language = default_language
        
        # Stats
        self._stats = {
            "total_generated": 0,
            "by_level": {},
            "clarifications": 0
        }
    
    def admit_uncertainty(
        self,
        topic: str,
        level: UncertaintyLevel = UncertaintyLevel.MEDIUM,
        language: Optional[str] = None
    ) -> FallbackResponse:
        """
        Generate an honest uncertainty response.
        
        Args:
            topic: What the query was about
            level: How uncertain we are
            language: Response language ("ru" or "en")
            
        Returns:
            FallbackResponse with text and suggestions
        """
        lang = language or self._detect_language(topic) or self.default_language
        
        # Select appropriate templates
        if lang == "ru":
            templates = self.UNCERTAINTY_TEMPLATES_RU[level]
            suggestions = self.SUGGESTIONS_RU
        else:
            templates = self.UNCERTAINTY_TEMPLATES_EN[level]
            suggestions = self.SUGGESTIONS_EN
        
        # Generate response
        template = random.choice(templates)
        suggestion = random.choice(suggestions)
        
        # Clean topic for display
        clean_topic = self._clean_topic(topic)
        
        text = template.format(topic=clean_topic, suggestion=suggestion)
        
        # Update stats
        self._stats["total_generated"] += 1
        self._stats["by_level"][level.value] = self._stats["by_level"].get(level.value, 0) + 1
        
        return FallbackResponse(
            text=text,
            level=level,
            suggested_actions=suggestions[:3],
            is_apologetic=level in [UncertaintyLevel.MEDIUM, UncertaintyLevel.HIGH]
        )
    
    def suggest_clarification(
        self,
        topic: str,
        options: Optional[List[str]] = None,
        language: Optional[str] = None
    ) -> FallbackResponse:
        """
        Generate a clarification request.
        
        Args:
            topic: The ambiguous topic
            options: Optional list of possible meanings
            language: Response language
            
        Returns:
            FallbackResponse with clarification question
        """
        lang = language or self._detect_language(topic) or self.default_language
        
        if lang == "ru":
            templates = self.CLARIFICATION_TEMPLATES_RU
        else:
            templates = self.CLARIFICATION_TEMPLATES_EN
        
        clean_topic = self._clean_topic(topic)
        
        if options and len(options) >= 2:
            # Use options template
            if lang == "ru":
                text = f"Уточните, пожалуйста: вы спрашиваете о {options[0]} или о {options[1]}?"
            else:
                text = f"Please clarify: are you asking about {options[0]} or about {options[1]}?"
        else:
            template = random.choice(templates[:2])  # Skip options-based templates
            text = template.format(topic=clean_topic)
        
        self._stats["clarifications"] += 1
        
        return FallbackResponse(
            text=text,
            level=UncertaintyLevel.LOW,
            suggested_actions=options or [],
            is_apologetic=False
        )
    
    def offer_alternatives(
        self,
        failed_topic: str,
        alternatives: List[str],
        language: Optional[str] = None
    ) -> FallbackResponse:
        """
        Offer alternative topics when the original query fails.
        
        Args:
            failed_topic: The topic that couldn't be answered
            alternatives: List of alternative topics we can help with
            language: Response language
        """
        lang = language or self._detect_language(failed_topic) or self.default_language
        
        clean_topic = self._clean_topic(failed_topic)
        
        if lang == "ru":
            alts_text = ", ".join(alternatives[:3])
            text = f"Информация о '{clean_topic}' недоступна, но я могу помочь с: {alts_text}."
        else:
            alts_text = ", ".join(alternatives[:3])
            text = f"Information about '{clean_topic}' is unavailable, but I can help with: {alts_text}."
        
        return FallbackResponse(
            text=text,
            level=UncertaintyLevel.MEDIUM,
            suggested_actions=alternatives,
            is_apologetic=True
        )
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Simple language detection based on character ranges."""
        if not text:
            return None
        
        # Count Cyrillic vs Latin characters
        cyrillic_count = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        latin_count = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        
        if cyrillic_count > latin_count:
            return "ru"
        elif latin_count > 0:
            return "en"
        
        return None
    
    def _clean_topic(self, topic: str) -> str:
        """Clean and shorten topic for display."""
        # Remove excess whitespace
        topic = ' '.join(topic.split())
        
        # Truncate if too long
        if len(topic) > 50:
            topic = topic[:47] + "..."
        
        return topic
    
    def get_stats(self) -> dict:
        """Get generator statistics."""
        return self._stats
