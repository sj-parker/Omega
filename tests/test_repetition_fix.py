import unittest
from datetime import datetime
from dataclasses import dataclass

# Mock ContextEvent
@dataclass
class ContextEvent:
    timestamp: datetime
    event_type: str
    content: str
    importance: float

# Copy of the fixed filter_context logic
def filter_context(selected):
    selected.sort(key=lambda e: e.timestamp)
    
    # Deduplicate repetitive system responses
    if len(selected) > 1:
        deduped = []
        prev = None
        for event in selected:
            if event.event_type == "system_response" and prev and prev.event_type == "system_response":
                    # Check for near-identical content
                    if event.content.strip() == prev.content.strip():
                        continue # Skip duplicate
            
            deduped.append(event)
            prev = event
        selected = deduped
    
    return selected

class TestRepetitionFix(unittest.TestCase):
    def test_deduplication(self):
        now = datetime.now()
        events = [
            ContextEvent(now, "user_input", "Hi", 0.5),
            ContextEvent(now, "system_response", "Hello!", 0.5),
            ContextEvent(now, "system_response", "Hello!", 0.5), # Repetition
            ContextEvent(now, "user_input", "How are you?", 0.5),
            ContextEvent(now, "system_response", "I am fine.", 0.5),
            ContextEvent(now, "system_response", "I am fine.", 0.5), # Repetition
            ContextEvent(now, "system_response", "I am fine.", 0.5), # Triple repetition
        ]
        
        filtered = filter_context(events)
        
        self.assertEqual(len(filtered), 4)
        self.assertEqual(filtered[0].content, "Hi")
        self.assertEqual(filtered[1].content, "Hello!")
        self.assertEqual(filtered[2].content, "How are you?")
        self.assertEqual(filtered[3].content, "I am fine.")
        print("\nTest passed: Successfully pruned repetitive responses.")

    def test_non_adjacent_repetition(self):
        """Ensure we don't prune if there's a user message in between."""
        now = datetime.now()
        events = [
            ContextEvent(now, "system_response", "Hello!", 0.5),
            ContextEvent(now, "user_input", "What?", 0.5),
            ContextEvent(now, "system_response", "Hello!", 0.5), # Valid repetition (clarification)
        ]
        
        filtered = filter_context(events)
        self.assertEqual(len(filtered), 3)
        print("Test passed: Preserved non-adjacent repetition.")

if __name__ == '__main__':
    unittest.main()
