# Info Broker
# Unified interface for information retrieval with fallback chain

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, List, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from core.context_manager import ContextManager
    from core.search_engine import SearchEngine
    from core.experts import ExpertsModule
    from models.schemas import WorldState


class InfoSource(Enum):
    """Source of information."""
    MEMORY = "memory"           # Local long-term facts
    SEARCH = "search"           # Web search
    EXPERT = "expert"           # LLM reasoning
    FALLBACK = "fallback"       # Admitted uncertainty
    CACHE = "cache"             # Short-term cache
    SPECIALIST = "specialist"   # Autonomous logic (Geo, Temporal, etc.)


@dataclass
class InfoResult:
    """Result of an information request."""
    source: InfoSource
    data: Any
    confidence: float  # 0.0 - 1.0
    query: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    
    @property
    def is_sufficient(self) -> bool:
        """Returns True if the information is likely sufficient."""
        if self.confidence < 0.7:
            return False
            
        # Semantic check: if query has keywords, they should be in snippets
        if not self.data:
            return False
            
        # Very basic keyword overlap check
        # We look for stems to handle basic Russian/English morphology
        stop_words = {"расстояние", "километр", "км", "цена", "стоит", "курс", "погода", "температура", "distance", "km", "price", "weather"}
        query_words = set(re.findall(r'[а-яa-z]+', self.query.lower()))
        data_words = set(re.findall(r'[а-яa-z]+', self.data.lower()))
        
        # Intersection of query descriptors and data
        descriptors = query_words.intersection(stop_words)
        if descriptors:
            # If user asked for distance/price, we MUST find it (or similar word) in data
            found_any = any(d[:4] in self.data.lower() for d in descriptors) # stem match
            if not found_any:
                return False
                
        return True
    
    def to_dict(self) -> dict:
        return {
            "source": self.source.value,
            "data": str(self.data)[:500] if self.data else None,
            "confidence": self.confidence,
            "query": self.query,
            "is_sufficient": self.is_sufficient
        }


@dataclass
class FallbackChainConfig:
    """Configuration for the fallback chain."""
    # Minimum confidence to accept result from each source
    memory_min_confidence: float = 0.7
    search_min_confidence: float = 0.6
    expert_min_confidence: float = 0.5
    
    # Timeouts
    search_timeout_seconds: float = 10.0
    expert_timeout_seconds: float = 15.0
    
    # Behavior
    parallel_search_expert: bool = False  # Try search and expert in parallel
    max_search_results: int = 3
    cache_ttl_seconds: float = 300.0  # 5 minutes


class InfoBroker:
    """
    Information Broker - Unified interface for information retrieval.
    
    Implements a fallback chain:
    1. Cache (fast, ephemeral)
    2. Memory (ContextManager.search_facts)
    3. Web Search (SearchEngine)
    4. Expert Reasoning (ExpertsModule)
    5. Fallback (admit uncertainty)
    
    This solves the "lost" problem - system always has a coherent response.
    """
    
    def __init__(
        self,
        context_manager: Optional['ContextManager'] = None,
        search_engine: Optional['SearchEngine'] = None,
        experts: Optional['ExpertsModule'] = None,
        config: Optional[FallbackChainConfig] = None
    ):
        self.context_manager = context_manager
        self.search_engine = search_engine
        self.experts = experts
        self.config = config or FallbackChainConfig()
        
        # Specialists
        from core.temporal_specialist import TemporalSpecialist
        from core.geography_tool import GeographyTool
        self.temporal = TemporalSpecialist()
        self.geography = GeographyTool()
        
        # Short-term cache
        self._cache: dict[str, tuple[InfoResult, datetime]] = {}
        
        # Stats
        self._stats = {
            "requests": 0,
            "cache_hits": 0,
            "memory_hits": 0,
            "search_hits": 0,
            "expert_hits": 0,
            "fallbacks": 0
        }
    
    async def request_info(
        self,
        query: str,
        min_confidence: float = 0.5,
        sources: Optional[List[InfoSource]] = None,
        world_state: Optional['WorldState'] = None,
        **kwargs
    ) -> InfoResult:
        """
        Request information using the fallback chain.
        
        Args:
            query: The information query
            min_confidence: Minimum acceptable confidence
            sources: Optional list of sources to try (default: all)
            world_state: Current world state for expert reasoning
            
        Returns:
            InfoResult with the best available information
        """
        self._stats["requests"] += 1
        
        # Default to all sources
        if sources is None:
            sources = [InfoSource.CACHE, InfoSource.SPECIALIST, InfoSource.MEMORY, InfoSource.SEARCH, InfoSource.EXPERT]
        
        # 0. Try specialist (autonomous logic) first if domain is known
        domain = kwargs.get("domain")
        if InfoSource.SPECIALIST in sources and domain:
            specialist_result = await self._try_specialist(query, domain)
            if specialist_result and specialist_result.confidence >= 0.8:
                return specialist_result
        
        # 1. Try cache
        if InfoSource.CACHE in sources:
            cache_result = self._check_cache(query)
            if cache_result and cache_result.confidence >= min_confidence:
                self._stats["cache_hits"] += 1
                print(f"[InfoBroker] Cache hit for '{query[:30]}...' (conf={cache_result.confidence:.2f})")
                return cache_result
        
        # 2. Try memory
        if InfoSource.MEMORY in sources and self.context_manager:
            memory_result = await self._try_memory(query)
            if memory_result.confidence >= self.config.memory_min_confidence:
                self._cache_result(query, memory_result)
                self._stats["memory_hits"] += 1
                print(f"[InfoBroker] Memory hit for '{query[:30]}...' (conf={memory_result.confidence:.2f})")
                return memory_result
        
        # 3. Try web search
        if InfoSource.SEARCH in sources and self.search_engine:
            try:
                search_result = await asyncio.wait_for(
                    self._try_search(query, volatility=kwargs.get("volatility", "low")),
                    timeout=self.config.search_timeout_seconds
                )
                if search_result.confidence >= self.config.search_min_confidence:
                    self._cache_result(query, search_result)
                    self._stats["search_hits"] += 1
                    print(f"[InfoBroker] Search hit for '{query[:30]}...' (conf={search_result.confidence:.2f})")
                    return search_result
            except asyncio.TimeoutError:
                print(f"[InfoBroker] Search timeout for '{query[:30]}...'")
        
        # 4. Try expert reasoning
        if InfoSource.EXPERT in sources and self.experts:
            try:
                expert_result = await asyncio.wait_for(
                    self._try_expert(query, world_state),
                    timeout=self.config.expert_timeout_seconds
                )
                if expert_result.confidence >= self.config.expert_min_confidence:
                    self._cache_result(query, expert_result)
                    self._stats["expert_hits"] += 1
                    print(f"[InfoBroker] Expert hit for '{query[:30]}...' (conf={expert_result.confidence:.2f})")
                    return expert_result
            except asyncio.TimeoutError:
                print(f"[InfoBroker] Expert timeout for '{query[:30]}...'")
        
        # 5. Fallback - admit uncertainty
        self._stats["fallbacks"] += 1
        print(f"[InfoBroker] Fallback for '{query[:30]}...'")
        return self._create_fallback(query)
    
    def _check_cache(self, query: str) -> Optional[InfoResult]:
        """Check short-term cache."""
        cache_key = self._normalize_query(query)
        if cache_key in self._cache:
            result, cached_at = self._cache[cache_key]
            age = (datetime.now() - cached_at).total_seconds()
            if age < self.config.cache_ttl_seconds:
                return result
            else:
                # Expired
                del self._cache[cache_key]
        return None
    
    def _cache_result(self, query: str, result: InfoResult):
        """Add result to cache."""
        cache_key = self._normalize_query(query)
        self._cache[cache_key] = (result, datetime.now())
        
        # Cleanup old entries (simple LRU-like)
        if len(self._cache) > 100:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for cache key."""
        return query.lower().strip()[:200]
    
    async def _try_memory(self, query: str) -> InfoResult:
        """Try to find information in long-term memory."""
        if not self.context_manager:
            return InfoResult(
                source=InfoSource.MEMORY,
                data=None,
                confidence=0.0,
                query=query
            )
        
        try:
            facts = self.context_manager.search_facts(query)
            if facts:
                # Combine relevant facts
                fact_texts = [f.content for f in facts[:3]]
                combined = "\n".join(fact_texts)
                
                # Confidence based on relevance and recency
                confidence = 0.85 if len(facts) >= 2 else 0.7
                
                return InfoResult(
                    source=InfoSource.MEMORY,
                    data=combined,
                    confidence=confidence,
                    query=query,
                    metadata={"fact_count": len(facts)}
                )
        except Exception as e:
            print(f"[InfoBroker] Memory error: {e}")
        
        return InfoResult(
            source=InfoSource.MEMORY,
            data=None,
            confidence=0.0,
            query=query
        )
    
    async def _try_search(self, query: str, volatility: str = "low", target: str = None) -> InfoResult:
        """Try web search with optional high-precision extraction."""
        if not self.search_engine:
            return InfoResult(
                source=InfoSource.SEARCH,
                data=None,
                confidence=0.0,
                query=query
            )
        
        try:
            # Use high-precision extraction if target is available
            if target:
                result = await self.search_engine.search_and_extract(query, target, volatility=volatility)
                return InfoResult(
                    source=InfoSource.SEARCH,
                    data=result["fact"],
                    confidence=result["confidence"],
                    query=query,
                    metadata={
                        "source": result["source"],
                        "method": result.get("method", "legacy")
                    }
                )

            # Standard search
            results = await self.search_engine.search(query, volatility=volatility)
            if results:
                snippets = [f"[{r.title}]: {r.snippet}" for r in results[:self.config.max_search_results]]
                combined = "\n\n".join(snippets)
                
                has_digits = any(c.isdigit() for c in combined)
                confidence = 0.8 if has_digits else 0.65
                
                return InfoResult(
                    source=InfoSource.SEARCH,
                    data=combined,
                    confidence=confidence,
                    query=query,
                    metadata={
                        "result_count": len(results),
                        "sources": [r.url for r in results[:3]]
                    }
                )
        except Exception as e:
            print(f"[InfoBroker] Search error: {e}")
        
        return InfoResult(
            source=InfoSource.SEARCH,
            data=None,
            confidence=0.0,
            query=query
        )
    
    async def _try_expert(self, query: str, world_state: Optional['WorldState']) -> InfoResult:
        """Try expert reasoning."""
        if not self.experts:
            return InfoResult(
                source=InfoSource.EXPERT,
                data=None,
                confidence=0.0,
                query=query
            )
        
        try:
            from models.schemas import WorldState as WS
            ws = world_state or WS()
            
            response = await self.experts.consult_expert(
                expert_type="neutral",
                prompt=query,
                world_state=ws,
                context=""
            )
            
            return InfoResult(
                source=InfoSource.EXPERT,
                data=response.response,
                confidence=response.confidence,
                query=query,
                metadata={"expert_type": response.expert_type}
            )
        except Exception as e:
            print(f"[InfoBroker] Expert error: {e}")
        
        return InfoResult(
            source=InfoSource.EXPERT,
            data=None,
            confidence=0.0,
            query=query
        )
    
    async def _try_specialist(self, query: str, domain: str) -> Optional[InfoResult]:
        """Try autonomous logic specialists."""
        try:
            if domain == "temporal":
                info = self.temporal.parse_date_and_calculate(query)
                if not info:
                    # Fallback to general info
                    now_info = self.temporal.get_current_info()
                    if "сегодня" in query.lower() or "today" in query.lower():
                        info = f"Сегодня {now_info['date']}, {now_info['day_of_week_ru']}."
                
                if info:
                    return InfoResult(source=InfoSource.SPECIALIST, data=info, confidence=1.0, query=query)
            
            elif domain == "geography":
                # If query is just a snippet of text, try to extract distance
                dist = self.geography.extract_distance(query)
                if dist:
                    return InfoResult(source=InfoSource.SPECIALIST, data=f"{dist} km", confidence=0.9, query=query)
            
            return None
        except Exception as e:
            print(f"[InfoBroker] Specialist error: {e}")
            return None
        """Create a fallback response (admitted uncertainty)."""
        return InfoResult(
            source=InfoSource.FALLBACK,
            data=None,
            confidence=0.0,
            query=query,
            metadata={"reason": "No reliable information found"}
        )
    
    async def multi_source_fetch(self, query: str, world_state: Optional['WorldState'] = None) -> List[InfoResult]:
        """
        Fetch from all sources in parallel and return all results.
        Useful when you want to compare or aggregate information.
        """
        tasks = []
        
        if self.context_manager:
            tasks.append(self._try_memory(query))
        if self.search_engine:
            tasks.append(self._try_search(query))
        if self.experts:
            tasks.append(self._try_expert(query, world_state))
        
        if not tasks:
            return [self._create_fallback(query)]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for r in results:
            if isinstance(r, InfoResult):
                valid_results.append(r)
            elif isinstance(r, Exception):
                print(f"[InfoBroker] Multi-fetch error: {r}")
        
        return valid_results if valid_results else [self._create_fallback(query)]
    
    def is_info_sufficient(self, results: List[InfoResult], min_confidence: float = 0.5) -> bool:
        """Check if we have sufficient information to answer."""
        return any(r.confidence >= min_confidence and r.data for r in results)
    
    def get_best_result(self, results: List[InfoResult]) -> InfoResult:
        """Get the best result from a list of results."""
        if not results:
            return self._create_fallback("unknown")
        
        # Sort by confidence, prefer non-fallback sources
        def score(r: InfoResult) -> tuple:
            source_priority = {
                InfoSource.MEMORY: 0,
                InfoSource.SEARCH: 1,
                InfoSource.EXPERT: 2,
                InfoSource.CACHE: 0,
                InfoSource.FALLBACK: 9
            }
            return (source_priority.get(r.source, 5), -r.confidence)
        
        return min(results, key=score)
    
    def get_stats(self) -> dict:
        """Get broker statistics."""
        total = self._stats["requests"] or 1
        return {
            **self._stats,
            "cache_hit_rate": self._stats["cache_hits"] / total,
            "fallback_rate": self._stats["fallbacks"] / total,
            "cache_size": len(self._cache)
        }
    
    def clear_cache(self):
        """Clear the short-term cache."""
        self._cache.clear()
