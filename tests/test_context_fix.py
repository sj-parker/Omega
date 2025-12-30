import asyncio
import os
import sys
from datetime import datetime, timedelta
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.context_manager import ContextManager, ContextEvent
from models.schemas import UserIdentity, WorldState

def test_context_ordering():
    print("=== Testing Context Ordering ===")
    
    km = ContextManager()
    
    # 1. Simulate a conversation trace
    # T1: Old high importance event
    t0 = datetime.now() - timedelta(minutes=5)
    e1 = ContextEvent(timestamp=t0, event_type="user_input", content="Hi, I am User.", importance=0.9)
    km.short_store.add_event(e1)
    
    # T2: Middle event
    t1 = datetime.now() - timedelta(minutes=3)
    e2 = ContextEvent(timestamp=t1, event_type="system_reponse", content="Hello User!", importance=0.5)
    km.short_store.add_event(e2)
    
    # T3: Recent low importance event (e.g. "ok")
    t2 = datetime.now() - timedelta(minutes=1)
    e3 = ContextEvent(timestamp=t2, event_type="user_input", content="ok", importance=0.3)
    km.short_store.add_event(e3)
    
    # T4: Very recent question
    t3 = datetime.now()
    e4 = ContextEvent(timestamp=t3, event_type="user_input", content="What did I just say?", importance=0.8)
    km.short_store.add_event(e4)
    
    # 2. Get Context Slice
    print("\nRetrieving Slice...")
    us = UserIdentity(user_id="test", trust_level=1.0)
    ws = WorldState()
    
    # We expect all 4 events to be returned, in order e1 -> e2 -> e3 -> e4
    # Even though e3 has low importance, it is recent.
    # The KEY check is that they are CHRONOLOGICAL.
    
    slice_obj = km.get_context_slice("What did I just say?", us, ws)
    
    events = slice_obj.recent_events
    print(f"Retrieved {len(events)} events.")
    
    last_time = datetime.min
    for i, e in enumerate(events):
        print(f"{i}: [{e.timestamp.strftime('%H:%M:%S')}] ({e.importance}) {e.content}")
        if e.timestamp < last_time:
            print("❌ FAILED: Events are OUT OF ORDER!")
            return
        last_time = e.timestamp
        
    print("✅ Context is correctly ordered chronologically.")
    
    # check that "ok" (low importance) wasn't filtered out if we have space
    has_ok = any(e.content == "ok" for e in events)
    if has_ok:
         print("✅ Low importance recent event preserved.")
    else:
         print("⚠️ Low importance event filtered (might be expected if strict filtering).")

if __name__ == "__main__":
    test_context_ordering()
