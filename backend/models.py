from pydantic import BaseModel, field_validator
from datetime import date

class GamePrediction(BaseModel):
    home: str
    away: str
    game_date: str
    spread_prediction: float

    @field_validator('game_date', mode='before')
    @classmethod
    def format_date_to_text(cls, v):
        # If the DB returns a date object, format it
        if isinstance(v, date):
            return v.strftime("%b %d, %Y")
        # If it's already a string, just return it
        return v

class DashboardData(BaseModel):
    summary: list[dict] # We can be stricter here later
    games: list[GamePrediction]