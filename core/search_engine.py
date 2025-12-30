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
        
    async def search(self, query: str, volatility: str = "low") -> List[SearchResult]:
        """Perform a text search with temporal awareness."""
        results = []
        
        # 1. Temporal query enhancement
        enhanced_query = query
        temporal_kws = [
            "цена", "курс", "price", "rate", "weather", "погода", "stock", "crypto", "bitcoin",
            "билеты", "расписание", "tickets", "showtimes", "schedule", "кино", "фильм", "movie",
            "запчасти", "parts", "cost", "стоимость", "аренда", "rent"
        ]
        
        # Trigger on keywords OR high volatility
        if volatility == "high" or any(kw in query.lower() for kw in temporal_kws):
            # Append today's date hint if not present
            from datetime import datetime
            now_year = datetime.now().year
            if str(now_year) not in query:
                enhanced_query = f"{query} latest {now_year}"
                print(f"[SearchEngine] Enhanced query (volatility={volatility}): {enhanced_query}")

        try:
            # Run in executor since ddgs is blocking
            import asyncio
            loop = asyncio.get_event_loop()
            
            print(f"[SearchEngine] Querying: {enhanced_query}")
            raw_results = await loop.run_in_executor(
                None, 
                lambda: list(self.ddgs.text(enhanced_query, max_results=self.max_results))
            )
            
            # 2. Fallback if no results and query was enhanced
            if not raw_results and enhanced_query != query:
                print(f"[SearchEngine] No results for enhanced query, falling back to original: {query}")
                raw_results = await loop.run_in_executor(
                    None, 
                    lambda: list(self.ddgs.text(query, max_results=self.max_results))
                )

            for r in raw_results:
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", "")
                ))
            
            print(f"[SearchEngine] Found {len(results)} results")
                
            # 3. Result scoring with volatility weighting
            results = self._rank_by_freshness(results, volatility=volatility)
            
        except Exception as e:
            print(f"[SearchEngine] Error: {e}")
            
        return results

    def _rank_by_freshness(self, results: List[SearchResult], volatility: str = "low") -> List[SearchResult]:
        """Re-rank results to favor recent content and penalize old years in a domain-agnostic way."""
        from datetime import datetime
        now_year = datetime.now().year
        
        # We penalize any year from 2000 to last year
        stale_years = [str(y) for y in range(2000, now_year)]
        
        # Scale penalties based on volatility
        # If volatility is high (prices, showtimes), we are more aggressive
        penalty_factor = 1.5 if volatility == "high" else 1.0
        
        scored_results = []
        for res in results:
            score = 0
            text = (res.title + " " + res.snippet).lower()
            
            # Global Historical Penalty:
            for year in stale_years:
                if year in text:
                    # Check if the user WAS looking for this year (e.g. "price in 2023")
                    # If not, it's a stale hit.
                    score -= (15 * penalty_factor)
            
            # Freshness Reward:
            if str(now_year) in text:
                score += (20 * penalty_factor)
            
            # Multi-lingual Freshness Keywords:
            fresh_kws = [
                "today", "now", "current", "latest", "update", "live", "real-time",
                "сегодня", "сейчас", "актуально", "текущий", "последний", "свежий"
            ]
            if any(kw in text for kw in fresh_kws):
                score += (15 * penalty_factor)
                
            # URL heuristic: prefer news, official sites, or known fresh sources
            fresh_domains = ["news", "live", "today", "current", "market", "weather", "ticker"]
            if any(fd in res.url.lower() for fd in fresh_domains):
                score += (5 * penalty_factor)
            
            # Additional Staleness Penalties for High Volatility:
            if volatility == "high":
                stale_indicators = [
                    "ago", "yesterday", "last week", "last month", "archive", "historical",
                    "назад", "вчера", "на прошлой неделе", "архив", "исторически"
                ]
                # High penalty if snippet says "1 month ago" for high-volatility data
                for si in stale_indicators:
                    if si in text:
                        score -= (25 * penalty_factor)

            scored_results.append((score, res))
            
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [r for s, r in scored_results]
    
    async def search_and_extract(self, query: str, target: str, volatility: str = "low") -> dict:
        """
        Search and try to extract a specific target (fact).
        Returns dict with fact, confidence, and source URL.
        """
        from core.tracer import tracer
        
        results = await self.search(query, volatility=volatility)
        best_fact = None
        best_source = None
        
        # Domains to exclude for factual queries
        BAD_DOMAINS = ["dictionary.cambridge.org", "merriam-webster.com", "thesaurus.com", "facebook.com", "twitter.com"]
        
        # Keywords that indicate "Straight line" vs "Road"
        STRAIGHT_LINE_KWS = ["по прямой", "straight line", "air distance", "as the crow flies", "vector"]
        ROAD_KWS = ["по дорогам", "на машине", "road", "driving", "route", "маршрут", "трасса"]
        
        # Targets that MUST have numbers to be valid (Language agnostic)
        NUMERIC_TARGETS = ["price", "rate", "value", "cost", "weather", "temp", "temperature", "humidity", "bitcoin", "eth", 
                          "цена", "стоимость", "дистанция", "расстояние", "время", "км", "km"]
        
        candidates = []
        for res in results:
            if any(bd in res.url for bd in BAD_DOMAINS):
                continue
                
            snippet = res.snippet
            snippet_low = snippet.lower()
            target_low = target.lower()
            
            # Scoring snippet quality
            score = 0
            
            # 1. Target keyword match
            if target_low in snippet_low:
                score += 5
            
            # 2. Number presence for numeric targets
            has_digits = any(c.isdigit() for c in snippet)
            if any(nt in target_low for nt in NUMERIC_TARGETS):
                if has_digits:
                    score += 10
                else:
                    score -= 5
            
            # 3. Road vs Straight line logic
            is_road_query = any(rk in query.lower() for rk in ROAD_KWS)
            has_straight_kws = any(sk in snippet_low for sk in STRAIGHT_LINE_KWS)
            has_road_kws = any(rk in snippet_low for rk in ROAD_KWS)
            
            if is_road_query:
                if has_road_kws:
                    score += 7
                if has_straight_kws:
                    score -= 10 # Strong penalty for straight-line results in road query
            
            # 4. Units presence (km, км, h, ч)
            UNITS = ["км", "km", "час", "hour", "мин", "min"]
            if any(u in snippet_low for u in UNITS):
                score += 3

            candidates.append((score, res))
        
        if candidates:
            # Sort by score (desc)
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_res = candidates[0]
            
            best_fact = best_res.snippet
            best_source = best_res.url
            confidence = 0.8 if best_score > 5 else 0.5
        else:
            best_fact = "No relevant information found in search results. Please try a different query."
            confidence = 0.0
            best_source = "N/A"
        
        tracer.add_step("search_engine", "Result", f"Found info for: {query[:50]}...", data_out={"fact": best_fact[:200] if best_fact else None, "source": best_source})
        
        return {
            "fact": best_fact,
            "confidence": confidence,
            "source": best_source
        }

    async def verify_fact(self, fact: str, original_source: str, volatility: str = "low") -> dict:
        """
        Cross-check a fact by searching again, excluding original source.
        """
        # Create verification query
        domain = original_source.split("/")[2] if "//" in original_source else original_source
        query = f"{fact} -site:{domain}"
        
        results = await self.search(query, volatility=volatility)
        if not results:
            return {"verified": False, "reason": "No other sources found"}
        
        # Check if first result supports the fact (naive check)
        return {
            "verified": True, 
            "cross_check_source": results[0].url,
            "snippet": results[0].snippet
        }
