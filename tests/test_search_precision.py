import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.search_engine import SearchEngine
from models.llm_interface import OllamaLLM, MockLLM
from core.config import config

async def test_precision_search():
    print("=== Search Precision Test (Qwen + Regex) ===")
    
    # Initialize Search model
    search_model = config.get("models.search", "qwen2.5:7b")
    llm = OllamaLLM(model=search_model)
    
    # Initialize SearchEngine
    engine = SearchEngine(llm=llm)
    
    test_cases = [
        {"query": "Ethereum price in USD", "target": "current price of ETH"},
        {"query": "Tokyo population 2024", "target": "population number"},
        {"query": "distance from Mars to Earth today", "target": "distance in km or miles"}
    ]
    
    for case in test_cases:
        print(f"\n[Test] Query: {case['query']}")
        print(f"[Test] Target: {case['target']}")
        
        try:
            result = await engine.search_and_extract(case['query'], case['target'])
            print(f"[Result] Method: {result.get('method')}")
            print(f"[Result] Fact: {result.get('fact')}")
            print(f"[Result] Confidence: {result.get('confidence')}")
            print(f"[Result] Source: {result.get('source')}")
        except Exception as e:
            print(f"[Error] {e}")

if __name__ == "__main__":
    asyncio.run(test_precision_search())
