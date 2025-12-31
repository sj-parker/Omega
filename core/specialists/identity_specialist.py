from core.specialists.base import BaseSpecialist, SpecialistResult, SpecialistMetadata
from typing import Any, Optional

class GeneralIdentitySpecialist(BaseSpecialist):
    def __init__(self):
        super().__init__()

    @property
    def metadata(self) -> SpecialistMetadata:
        return SpecialistMetadata(
            id="general_identity",
            name="General Identity Specialist",
            description="Handles questions about the system's identity, personality, humor, and general chitchat.",
            keywords=["who are you", "what are you", "your name", "sarcasm", "joke", "funny", "mars", "robot", "ai"]
        )

    def can_handle(self, query: str) -> float:
        query_lower = query.lower()
        
        # Identity keywords
        identity_keywords = ["who are you", "what is your name", "are you an ai", "are you a human", "are you a robot", "ты кто", "как тебя зовут", "ты человек", "ты робот"]
        for k in identity_keywords:
            if k in query_lower:
                return 0.95
        
        # Conversational/Emotional keywords
        conv_keywords = ["sarcasm", "joke", "funny", "feel", "emotion", "doubt", "сарказм", "шутка", "чувствуешь", "эмоции", "сомневаешься"]
        for k in conv_keywords:
            if k in query_lower:
                return 0.8
                
        # "Mars" specific (for the user's running joke)
        if "mars" in query_lower or "марс" in query_lower:
             return 0.85

        return 0.1

    async def execute(self, query: str, context: Optional[Any] = None) -> SpecialistResult:
        # This specialist is actually just a router metadata provider in a full implementation,
        # but for now we can return a "Persona Instruction" or handle it directly if simple.
        
        # Ideally, this returns an instruction to the LLM "Answer as Omega, a friendly AI..."
        # But per the BaseSpecialist contract, it returns a RESULT.
        
        # Let's return a specific context injection to guide the LLM's persona without forcing a canned response.
        
        return SpecialistResult(
            data={"instruction": "Respond as Omega. Be friendly, verified, and humble. If asked about being human or from Mars, clarify you are an AI but enjoy the humor. DO NOT include unrelated facts like Bitcoin prices unless explicitly asked."},
            confidence=1.0,
            source="general_identity"
        )
