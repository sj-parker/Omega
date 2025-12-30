
import json
from pathlib import Path

trace_ids = [
    "a90b70df-8906-4330-b4f0-fce9dcdb6ba0",
    "2ee754a0-5a4e-45e8-85ff-bc88ea0a6b35",
    "08908013-3872-4e5b-a623-afeb62caf843",
    "48d72c51-d3ef-4d45-83d8-772570087246",
    "0ab20d66-1b59-450e-98fc-79a599faa96b"
]

base_path = Path("e:/agi2/learning_data")

def clean_text(text):
    if not text: return ""
    return text.replace('\n', ' ').strip()

print(f"{'Trace ID':<8} | {'User Input':<40} | {'System Response':<40}")
print("-" * 100)

for i, tid in enumerate(trace_ids, 1):
    path = base_path / f"trace_{tid}.json"
    if not path.exists():
        print(f"TR-{i} Not Found")
        continue
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            user_in = clean_text(data.get("user_input", ""))
            sys_out = clean_text(data.get("final_response", ""))
            
            print(f"TR-{i}      | {user_in[:40]:<40} | {sys_out[:40]:<40}")
            
            # Deep dive for Trace 4
            if i == 4:
                print("\n--- TRACE 4 DEEP DIVE ---")
                print(f"Full Input: {user_in}")
                print(f"Full Output: {sys_out}")
                print("History (Cleaned):")
                history = data.get("context_snapshot", {}).get("recent_events", [])
                for evt in history[-5:]: # Last 5 events
                    content = clean_text(evt.get("content", ""))
                    if "planet" not in content.lower():
                        print(f"  [{evt.get('event_type')}]: {content[:100]}")
                
                print("Expert/Thoughts:")
                experts = data.get("expert_outputs", [])
                for exp in experts:
                    print(f"  Expert ({exp.get('expert_type')}): {clean_text(str(exp))[:150]}...")
                print("-------------------------\n")

    except Exception as e:
        print(f"Error reading {tid}: {e}")
