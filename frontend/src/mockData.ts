//simulate the JSON response we will be getting from backend
export const SUMMARY_STATS = [
  { label: "Model ROI", value: "+12.4%", trend: "up" as const },
  { label: "Win Rate (ATS)", value: "58.2%", trend: "up" as const },
  { label: "Predicted Profit", value: "$2,450", trend: "up" as const },
  { label: "Active Alerts", value: "3", trend: "neutral" as const },
];

export const GAMES_DATA = [
  {
    id: 1,
    date: "Nov 30, 2025",
    team: "Celtics",
    opponent: "Cavaliers",
    location: "away", // 'home' or 'away'
    line: -7.5,
    prediction: -10.2, // Model thinks Celtics win by 10.2
    edge: 2.7, // Difference between line and prediction
    result: "W", // Win/Loss for ATS (Against The Spread)
    score: "112 - 101",
    confidence: "High",
  },
  {
    id: 2,
    date: "Nov 30, 2025",
    team: "Lakers",
    opponent: "Warriors",
    location: "home",
    line: +3.5,
    prediction: +1.0,
    edge: 2.5,
    result: "L",
    score: "105 - 110",
    confidence: "Medium",
  },
  {
    id: 3,
    date: "Nov 29, 2025",
    team: "Nuggets",
    opponent: "Suns",
    location: "home",
    line: -3.5,
    prediction: -8.0,
    edge: 4.5,
    result: "W",
    score: "120 - 105",
    confidence: "High",
  },
  {
    id: 4,
    date: "Nov 29, 2025",
    team: "Heat",
    opponent: "Pistons",
    location: "home",
    line: -11.5,
    prediction: -12.0,
    edge: 0.5,
    result: "L",
    score: "101 - 95", // Won game but lost spread
    confidence: "Low",
  },
  {
    id: 5,
    date: "Nov 28, 2025",
    team: "Bucks",
    opponent: "Nets",
    location: "away",
    line: -5.5,
    prediction: -2.1,
    edge: -3.4,
    result: "P", // Pending or Pass
    score: "Upcoming",
    confidence: "Low",
  },
];