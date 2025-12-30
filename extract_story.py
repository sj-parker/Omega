
import json
from pathlib import Path

trace_id = "a90b70df-8906-4330-b4f0-fce9dcdb6ba0"
base_path = Path("e:/agi2/learning_data")
path = base_path / f"trace_{trace_id}.json"

if path.exists():
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"USER: {data.get('user_input')}")
            print(f"SYSTEM: {data.get('final_response')}")
            print(f"INTENT: {data.get('decision', {}).get('intent')}")
            # Show history at that point
            events = data.get('context_snapshot', {}).get('recent_events', [])
            print("RECENT HISTORY:")
            for e in events:
                print(f"  - {e.get('event_type')}: {e.get('content')[:50]}...")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("Trace not found")
