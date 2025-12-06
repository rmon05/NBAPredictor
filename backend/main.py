from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.models import GamePrediction, DashboardData

app = FastAPI()

origins = [
    "http://localhost:5173" # Vite default port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mock_games = [
    GamePrediction(
        id=1, date="Nov 30, 2025", team="Celtics", opponent="Cavaliers",
        location="away", line=-7.5, prediction=-10.2, edge=2.7,
        result="W", score="112 - 101", confidence="High"
    ),
    GamePrediction(
        id=2, date="Nov 30, 2025", team="Lakers", opponent="Warriors",
        location="home", line=3.5, prediction=1.0, edge=2.5,
        result="L", score="105 - 110", confidence="Medium"
    ),
     GamePrediction(
        id=3, date="Nov 29, 2025", team="Nuggets", opponent="Suns",
        location="home", line=-3.5, prediction=-8.0, edge=4.5,
        result="W", score="120 - 105", confidence="High"
    ),
]

@app.get("/")
def read_root():
    return {"message": "Welcome to the NBA Predictor API"}

@app.get("/api/dashboard", response_model=DashboardData)
def get_dashboard_data():
    # Construct the response using our Pydantic model
    return DashboardData(
        summary=[
            {"label": "Model ROI", "value": "+12.4%", "trend": "up"},
            {"label": "Win Rate", "value": "58.2%", "trend": "up"},
        ],
        games=mock_games
    )