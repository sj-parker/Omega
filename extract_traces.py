
import json
import os
from pathlib import Path

trace_ids = [
    "a90b70df-8906-4330-b4f0-fce9dcdb6ba0",
    "2ee754a0-5a4e-45e8-85ff-bc88ea0a6b35",
    "08908013-3872-4e5b-a623-afeb62caf843",
    "48d72c51-d3ef-4d45-83d8-772570087246",
    "0ab20d66-1b59-450e-98fc-79a599faa96b"
]

base_path = Path("e:/agi2/learning_data")

for i, tid in enumerate(trace_ids, 1):
    path = base_path / f"trace_{tid}.json"
    print(f"\n{'='*20} MESSAGE {i} ({tid}) {'='*20}")
    if not path.exists():
        print(f"File {path} not found")
        continue
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            user_input = data.get("user_input", "N/A")
            final_response = data.get("final_response", "N/A")
            decision = data.get("decision", {})
            
            print(f"USER: {user_input}")
            print(f"SYSTEM: {final_response}")
            print(f"DECISION: action={decision.get('action')}, intent={decision.get('intent')}, depth={decision.get('depth_used')}")
            
            # Check for memory context in recent_events if available
            context = data.get("context_snapshot", {})
            events = context.get("recent_events", [])
            print(f"CONTEXT HISTORY (Last 3):")
            for evt in events[-3:]:
                print(f"  - [{evt.get('event_type')}]: {evt.get('content')[:100]}...")
                
    except Exception as e:
        print(f"Error reading {tid}: {e}")
