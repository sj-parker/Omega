import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.specialist_broker import SpecialistBroker
from core.specialists.movie_specialist import MovieSpecialist

async def test_broker_flow():
    print("=== Testing Specialist Broker ===")
    
    # 1. Setup
    broker = SpecialistBroker(storage_path="tests/temp_specialist_stats.json")
    
    # Mock search engine
    class MockSearch:
        async def search(self, query, volatility):
            print(f"  [MockSearch] searching for: {query}")
            return [] # Return empty for now, just checking flow
            
    broker.register(MovieSpecialist(search_engine=MockSearch()))
    
    # 2. Test Selection (Positive Case)
    query = "what movies are playing in theaters?"
    print(f"\nQuery: '{query}'")
    specialist = broker.get_selection(query, threshold=0.8)
    
    if specialist:
        print(f"[OK] Selected: {specialist.metadata.name}")
        result = await specialist.execute(query)
        print(f"  Result: {result.data[:50]}...")
    else:
        print("[FAIL] FAILED: Should have selected Movie Specialist")
        
    # 3. Test Selection (Negative Case)
    query_neg = "how to make soup?"
    print(f"\nQuery: '{query_neg}'")
    specialist_neg = broker.get_selection(query_neg, threshold=0.8)
    
    if specialist_neg is None:
        print("[OK] Correctly rejected irrelevant query")
    else:
        print(f"[FAIL] FAILED: Should NOT have selected anyone, got {specialist_neg.metadata.name}")

    # 4. Test Learning (Feedback)
    print("\n=== Testing Learning Mechanism ===")
    mid = "movie_v1"
    initial_score = broker.stats.get(mid, 1.0)
    print(f"Initial Score: {initial_score}")
    
    # Send NEGATIVE feedback
    broker.feedback(mid, success=False)
    new_score = broker.stats.get(mid)
    print(f"After Penalty: {new_score}")
    
    if new_score < initial_score:
        print("[OK] Score decreased correctly")
    else:
        print("[FAIL] FAILED: Score did not decrease")
        
    # Send POSITIVE feedback multiple times
    broker.feedback(mid, success=True)
    broker.feedback(mid, success=True)
    final_score = broker.stats.get(mid)
    print(f"After Rewards: {final_score}")
    
    if final_score > new_score:
        print("[OK] Score recovered correctly")
    else: 
        print("[FAIL] FAILED: Score did not recover")

    # Cleanup
    if os.path.exists("tests/temp_specialist_stats.json"):
        os.remove("tests/temp_specialist_stats.json")

if __name__ == "__main__":
    asyncio.run(test_broker_flow())
