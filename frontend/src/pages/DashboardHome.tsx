import { useEffect, useState } from 'react';
import StatCard from '../components/StatCard';
import GamesTable from '../components/GamesTable';
import { SUMMARY_STATS } from '../mockData';

interface GamePrediction {
  id: number;
  date: string;
  team: string;
  opponent: string;
  location: string;
  line: number;
  prediction: number;
  edge: number;
  result: string;
  score: string;
  confidence: string;
}

export default function DashboardHome() {
  const [games, setGames] = useState<GamePrediction[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://localhost:8000/api/dashboard')
      .then((res) => res.json())
      .then((data) => {
        // data.games comes from our Python DashboardData model
        setGames(data.games); 
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching dashboard data:", error);
        setLoading(false);
      });
  }, []);

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
        <div>
            <h1 className="text-2xl font-bold mb-6 text-zinc-900 dark:text-white">Overview</h1>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {SUMMARY_STATS.map((stat, i) => (
                    <StatCard key={i} {...stat} />
                ))}
            </div>
        </div>

        <div className="space-y-4">
            <div className="flex items-center justify-between">
                <h2 className="text-xl font-bold text-zinc-900 dark:text-white">
                    Upcoming & Recent Games
                    {loading && <span className="ml-2 text-sm font-normal text-gray-500">(Loading...)</span>}
                </h2>
                <button className="text-sm text-indigo-600 dark:text-indigo-400 hover:underline font-medium">View All History</button>
            </div>
            {/* Pass the data fetched from Python */}
            <GamesTable games={games} />
        </div>
    </div>
  );
}