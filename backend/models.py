from pydantic import BaseModel
from typing import Optional

class GamePrediction(BaseModel):
    id: int
    date: str
    team: str
    opponent: str
    location: str       # "home" or "away"
    line: float
    prediction: float
    edge: float
    result: Optional[str] = None  # "W", "L", or None if pending
    score: Optional[str] = None
    confidence: str     # "High", "Medium", "Low"

class DashboardData(BaseModel):
    summary: list[dict] # We can be stricter here later
    games: list[GamePrediction]