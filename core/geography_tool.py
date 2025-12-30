# Geography Tool
# Autonomous processing of geographical data (distances, locations)

import re
import math
from typing import Dict, Any, List, Optional

class GeographyTool:
    def __init__(self):
        pass

    def extract_distance(self, text: str) -> Optional[float]:
        """Extract distance in km from text using autonomous regex/parsing."""
        # Look for patterns like "123 km", "456 километров", "расстояние ... 789"
        patterns = [
            r"(\d+[\.,]?\d*)\s*(?:км|km|километр)",
            r"(?:расстояние|дистанция).{0,20}?(\d+[\.,]?\d*)"
        ]
        
        for p in patterns:
            match = re.search(p, text, re.IGNORECASE)
            if match:
                val = match.group(1).replace(",", ".")
                try:
                    return float(val)
                except:
                    continue
        return None

    def calculate_air_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine formula to calculate distance between two points autonomously."""
        R = 6371.0 # Earth radius in km
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat / 2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon / 2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c

    def find_locations(self, text: str) -> List[str]:
        """Identify potential city names in text (autonomous heuristic)."""
        # Look for capitalized words that are not at the start of sentences
        # This is a simple heuristic; a real system would use a local db of cities
        cities = re.findall(r"\b[А-ЯA-Z][а-яa-z]+\b", text)
        return list(set(cities))

    def validate_route(self, from_city: str, to_city: str, distance: float) -> bool:
        """Sanity check for a route. If distance is too small/large, it might be an error."""
        # Autonomous logic: no city in Europe is > 5000km from another
        if distance < 1 or distance > 10000:
            return False
        return True
