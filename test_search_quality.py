
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.search_engine import SearchEngine

async def test_search():
    engine = SearchEngine(max_results=5)
    query = "время в пути Одесса Полтава на машине"
    print(f"Testing search for: {query}")
    
    # Test standard search
    results = await engine.search(query)
    for i, res in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Title: {res.title}")
        print(f"Snippet: {res.snippet}")
        print(f"URL: {res.url}")
        
    # Test extraction
    print("\n" + "="*50)
    print("Testing extraction...")
    extracted = await engine.search_and_extract(query, target="время")
    print(f"Fact: {extracted['fact']}")
    print(f"Confidence: {extracted['confidence']}")

if __name__ == "__main__":
    asyncio.run(test_search())
