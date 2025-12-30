import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web import CognitiveSystemWeb
from models.schemas import UserIdentity

async def test_deep_context():
    # Initialize system
    system = CognitiveSystemWeb()
    await system.start()
    user_id = "test_user_deep"
    identity = UserIdentity(user_id=user_id, trust_level=1.0)
    
    print("\n--- Starting Deep Context Test ---")
    
    # 1. Start a conversation with a specific fact
    first_msg = "My favorite color is emerald green and my dog's name is Barky."
    print(f"User (Turn 1): {first_msg}")
    data = await system.process(user_id, first_msg)
    print(f"System: {data['response'][:50]}...")
    
    # 2. Add 10 filler turns to push the first message out of a small (3-5) window
    for i in range(10):
        msg = f"Tell me a short fact about planet {i+1}."
        print(f"User (Turn {i+2}): {msg}")
        await system.process(user_id, msg)
    
    # 3. Ask about the first turn fact
    final_msg = "Remind me, what is my dog's name and what is my favorite color?"
    print(f"\nUser (Turn 12): {final_msg}")
    data = await system.process(user_id, final_msg)
    resp = data['response']
    print(f"System: {resp}")
    
    # Verify
    resp_l = resp.lower()
    success = "barky" in resp_l and "emerald green" in resp_l
    if success:
        print("\n[PASS] Deep Context Retained!")
    else:
        print("\n[FAIL] System forgot Turn 1 info.")
    
    await system.stop()

if __name__ == "__main__":
    asyncio.run(test_deep_context())
