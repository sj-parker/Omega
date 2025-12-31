import asyncio
import re
import sys
import os
import uuid  # Added to prevent the error
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from main import CognitiveSystem
    from core.config import config
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

async def run_tests():
    input_file = "test_questions_modules.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Reading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Parse categories
    categorized_questions = {}
    current_category = "UNCATEGORIZED"
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("[") and line.endswith("]"):
            current_category = line[1:-1]
            if current_category not in categorized_questions:
                categorized_questions[current_category] = []
        else:
            if current_category not in categorized_questions:
                categorized_questions[current_category] = []
            categorized_questions[current_category].append(line)

    total_questions = sum(len(qs) for qs in categorized_questions.values())
    print(f"Found {total_questions} questions in {len(categorized_questions)} categories.")

    print("Initializing CognitiveSystem...")
    system = CognitiveSystem()
    
    output_path = "module_weakness_report.md"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Module Weakness Stress Test Report\n\n")
        f.write(f"**Date:** {os.times}\n")
        f.write(f"**Total Questions:** {total_questions}\n\n")
        f.write("---\n\n")

    q_count = 0
    for category, questions in categorized_questions.items():
        print(f"\n--- Category: {category} ---")
        
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"## Category: {category}\n\n")
            
        for q in questions:
            q_count += 1
            print(f"[{q_count}/{total_questions}] {q}")
            
            try:
                # Use a module-tester user ID to isolate history if needed, 
                # but for Memory tests we might want persistence.
                # Let's use "tester_module"
                response = await system.process("tester_module", q)
                
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(f"### Q: {q}\n")
                    f.write(f"**Response:**\n{response}\n\n")
                    f.write("---\n\n")
                    
            except Exception as e:
                print(f"Error executing question '{q}': {e}")
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(f"### Q: {q}\n")
                    f.write(f"**Error:** {e}\n\n")
                    f.write("---\n\n")
    
    if hasattr(system, 'stop_reflection'):
        await system.stop_reflection()

    print(f"\nExecution complete. Results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(run_tests())
