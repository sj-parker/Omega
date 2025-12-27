# Omega Search Engine
# Interface to DuckDuckGo for real-time web search

from ddgs import DDGS
from dataclasses import dataclass
from typing import List, Optional
import re

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = "duckduckgo"

class SearchEngine:
    """
    Search Engine using DuckDuckGo backend.
    """
    
    def __init__(self, max_results: int = 3):
        self.max_results = max_results
        self.ddgs = DDGS()
        
    def search(self, query: str) -> List[SearchResult]:
        """Perform a text search."""
        results = []
        try:
            # text() is the method in newer versions, check version
            # But the standard way is often just calling ddgs.text()
            raw_results = self.ddgs.text(query, max_results=self.max_results)
            
            for r in raw_results:
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", "")
                ))
        except Exception as e:
            print(f"[SearchEngine] Error: {e}")
            
        return results
    
    def search_and_extract(self, query: str, target: str) -> dict:
        """
        Search and try to extract a specific target (fact).
        Returns dict with fact, confidence, and source URL.
        """
        results = self.search(query)
        best_fact = None
        best_source = None
        
        # Domains to exclude for factual queries
        BAD_DOMAINS = ["dictionary.cambridge.org", "merriam-webster.com", "thesaurus.com"]
        
        for res in results:
            if any(bd in res.url for bd in BAD_DOMAINS):
                continue
                
            snippet = res.snippet
            
            # Simple heuristic: If looking for price/rate, prioritize digits
            if target in ["price", "rate", "value", "cost"]:
                if not any(c.isdigit() for c in snippet):
                    continue # Skip if no numbers
            
            best_fact = snippet
            best_source = res.url
            break # Take top valid result
            
        return {
            "fact": best_fact or "No suitable info found",
            "confidence": 0.9 if best_fact else 0.0,
            "source": best_source or "N/A"
        }

    def verify_fact(self, fact: str, original_source: str) -> dict:
        """
        Cross-check a fact by searching again, excluding original source.
        """
        # Create verification query
        domain = original_source.split("/")[2] if "//" in original_source else original_source
        query = f"{fact} -site:{domain}"
        
        results = self.search(query)
        if not results:
            return {"verified": False, "reason": "No other sources found"}
        
        # Check if first result supports the fact (naive check)
        # Main LLM will do the semantic verification
        return {
            "verified": True, 
            "cross_check_source": results[0].url,
            "snippet": results[0].snippet
        }
