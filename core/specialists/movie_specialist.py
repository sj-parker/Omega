from typing import Any, List, Optional
from core.specialists.base import BaseSpecialist, SpecialistResult, SpecialistMetadata

class MovieSpecialist(BaseSpecialist):
    """
    Specialist for Movie/Cinema related queries.
    """
    
    def __init__(self, search_engine=None):
        super().__init__()
        self.search_engine = search_engine

    @property
    def metadata(self) -> SpecialistMetadata:
        return SpecialistMetadata(
            id="movie_v1",
            name="Movie Specialist",
            description="Finds movies playing in theaters, cinema schedules, and film info.",
            keywords=["movie", "cinema", "theater", "film", "фильм", "кино", "сеанс", "кинотеатр"]
        )

    def can_handle(self, query: str) -> float:
        """
        Returns high score if query contains movie keywords.
        """
        query = query.lower()
        
        # Check basic keywords
        matches = sum(1 for kw in self.metadata.keywords if kw in query)
        
        if matches >= 2:
            return 0.95  # Strong match
        elif matches == 1:
            return 0.8   # Decent match
            
        return 0.0

    async def execute(self, query: str, context: Optional[Any] = None) -> SpecialistResult:
        """
        Executes the movie search.
        """
        # 1. Refine the query for the search engine (mock or real)
        search_query = f"what movies are playing in theaters {query}"
        if "фильм" in query or "кино" in query:
             search_query = f"какие фильмы идут в кино {query}"

        # 2. Use Search Engine if available
        data = f"[MovieSpecialist] Searching for: {search_query}..."
        
        if self.search_engine:
            try:
                # We assume search_engine.search returns a list of results
                results = await self.search_engine.search(search_query, volatility="high")
                if results:
                    snippets = [f"- {r.title}: {r.snippet}" for r in results[:3]]
                    data = "Found these current movies/showtimes:\n" + "\n".join(snippets)
                else:
                    data = "I checked for movies but didn't find specific showtimes instantly."
            except Exception as e:
                data = f"Error searching for movies: {e}"
                
        return SpecialistResult(
            data=data,
            confidence=0.9,
            source=self.metadata.name
        )
