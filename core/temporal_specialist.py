# Temporal Specialist
# Handles date, time, and day-of-week calculations autonomously

from datetime import datetime
import calendar
from typing import Dict, Any, Optional

class TemporalSpecialist:
    def __init__(self):
        self.days_ru = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
        self.days_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def get_day_of_week(self, day: int, month: int, year: int, lang: str = "ru") -> str:
        """Calculate day of week for a specific date."""
        try:
            dt = datetime(year, month, day)
            idx = dt.weekday()
            if lang == "ru":
                return self.days_ru[idx]
            return self.days_en[idx]
        except Exception as e:
            return f"Error: {e}"

    def get_current_info(self) -> Dict[str, str]:
        """Get full current temporal context."""
        now = datetime.now()
        idx = now.weekday()
        return {
            "date": now.strftime("%d.%m.%Y"),
            "time": now.strftime("%H:%M:%S"),
            "day_of_week_ru": self.days_ru[idx],
            "day_of_week_en": self.days_en[idx],
            "iso": now.isoformat()
        }

    def parse_date_and_calculate(self, text: str) -> Optional[str]:
        """Attempt to find a date in text and return its day of week."""
        import re
        # Pattern for DD.MM.YYYY or YYYY-MM-DD
        patterns = [
            r"(\d{1,2})[\./-](\d{1,2})[\./-](\d{4})", # 28.12.2025
            r"(\d{4})[\./-](\d{1,2})[\./-](\d{1,2})"  # 2025-12-28
        ]
        
        for p in patterns:
            match = re.search(p, text)
            if match:
                groups = match.groups()
                if len(groups[0]) == 4: # YYYY-MM-DD
                    y, m, d = int(groups[0]), int(groups[1]), int(groups[2])
                else: # DD.MM.YYYY
                    d, m, y = int(groups[0]), int(groups[1]), int(groups[2])
                
                day_name = self.get_day_of_week(d, m, y)
                return f"{d:02d}.{m:02d}.{y} — это {day_name}."
        
        return None
