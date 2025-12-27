CREATE TABLE predictions (
    home TEXT,
    away TEXT,
    game_date DATE,
    spread_prediction FLOAT,
    PRIMARY KEY (home, game_date)
);
