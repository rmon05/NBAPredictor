import { useEffect, useState } from 'react';
import StatCard from '../components/StatCard';
import GamesTable from '../components/GamesTable';
import { Game } from '../components/GamesTable';
import { supabase } from '../lib/supabase'
import { SUMMARY_STATS } from '../mockData';

export default function DashboardHome() {
  const [games, setGames] = useState<Game[]>([]);
  const [loading, setLoading] = useState(true);

  // useEffect(() => {
  //   fetch('http://localhost:8000/api/dashboard')
  //     .then((res) => res.json())
  //     .then((data) => {
  //       // data.games comes from our Python DashboardData model
  //       setGames(data.games); 
  //       setLoading(false);
  //     })
  //     .catch((error) => {
  //       console.error("Error fetching dashboard data:", error);
  //       setLoading(false);
  //     });
  // }, []);

  useEffect(() => {
    // This safely queries the Supabase PaaS backend --> DB
    async function loadDashboard() {
      setLoading(true);

      // Force Eastern Time
      const today = new Date().toLocaleDateString('en-CA', {
        timeZone: 'America/New_York', 
        year: 'numeric',
        month: '2-digit',
        day: '2-digit'
      });

      // Query supabase directly with RLS in place
      const { data, error } = await supabase
        .from('predictions')
        .select('home, away, game_date, spread_prediction')
        .eq('game_date', today);

      if (error) {
        console.error("Error fetching dashboard data:", error);
      } else {
        setGames(data || []); 
      }

      setLoading(false);
    }

    loadDashboard();
  }, []);

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
        {/* <div>
            <h1 className="text-2xl font-bold mb-6 text-zinc-900 dark:text-white">Overview</h1>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {SUMMARY_STATS.map((stat, i) => (
                    <StatCard key={i} {...stat} />
                ))}
            </div>
        </div> */}

        <div className="space-y-4">
          <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold text-zinc-900 dark:text-white">
                  NBA Games
                  <span>
                    {" - "}
                    {new Date().toLocaleDateString('en-US', { 
                        month: 'short', 
                        day: 'numeric', 
                        year: 'numeric' 
                    })}
                  </span>
                  {loading && <span className="ml-2 text-sm font-normal text-gray-500">(Loading...)</span>}
              </h2>
              {/* <button className="text-sm text-indigo-600 dark:text-indigo-400 hover:underline font-medium">
                  View All History
              </button> */}
          </div>

          {/* Check if games list exists and has items */}
          {!loading && games.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 px-4 border-2 border-dashed border-zinc-200 dark:border-zinc-800 rounded-xl">
                  <p className="text-zinc-500 dark:text-zinc-400 font-medium">No games scheduled for today.</p>
              </div>
          ) : (
              <GamesTable games={games} />
          )}
        </div>
    </div>
  );
}