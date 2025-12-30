from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import GamePrediction, DashboardData
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

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

@app.get("/")
def read_root():
    return {"message": "Welcome to the NBA Predictor API"}

@app.get("/api/dashboard", response_model=DashboardData)
def get_dashboard_data():
    # load games from DB
    conn = psycopg2.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME
    )
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT home, away, game_date, spread_prediction 
        FROM predictions
    """)
    rows = cur.fetchall()

    # Convert to GamePrediction objects
    curr_games = [GamePrediction(**row) for row in rows]

    # Construct the response using our Pydantic model
    return DashboardData(
        summary=[
            {"label": "Model ROI", "value": "+12.4%", "trend": "up"},
            {"label": "Win Rate", "value": "58.2%", "trend": "up"},
        ],
        games=curr_games
    )