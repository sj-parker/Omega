# Test Omega Searcher
# Verifies SearchEngine and ToolsRegistry integration using DuckDuckGo

import asyncio
import sys
sys.path.insert(0, "e:\\agi2")

from core.tools import ToolsRegistry, SearchAndExtractArgs, VerifyFactArgs

def test_search_tool():
    print("--- Test Search Tool ---")
    query = "current price of bitcoin USD"
    print(f"Query: {query}")
    
    args = SearchAndExtractArgs(query=query, target="price")
    result = ToolsRegistry.search_and_extract(args)
    
    print(f"Result: {result.message}")
    
    if "Fact:" in result.message:
        print("[SUCCESS] Search returned a fact.")
        return result.message
    else:
        print("[FAIL] Search failed.")
        return None

def test_verify_tool(original_fact_msg):
    print("\n--- Test Verify Tool ---")
    
    # Extract fact and source from message for testing
    # Msg format: "Fact: ... (Confidence: ..., Source: URL)"
    import re
    fact_match = re.search(r"Fact: (.*?) \(", original_fact_msg)
    source_match = re.search(r"Source: (.*?)\)", original_fact_msg)
    
    if not (fact_match and source_match):
        print("[SKIP] Could not parse previous result for verification test.")
        return

    fact = fact_match.group(1)
    original_source = source_match.group(1)
    
    print(f"Verifying Fact: '{fact}' from Source: '{original_source}'")
    
    args = VerifyFactArgs(fact=fact, original_source=original_source)
    result = ToolsRegistry.verify_fact(args)
    
    print(f"Verify Result: {result.message}")
    
    if "Status:" in result.message:
        print("[SUCCESS] Verification executed.")
    else:
        print("[FAIL] Verification failed.")

if __name__ == "__main__":
    msg = test_search_tool()
    if msg:
        test_verify_tool(msg)
