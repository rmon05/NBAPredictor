import StatCard from '../components/StatCard';
import GamesTable from '../components/GamesTable';
import { SUMMARY_STATS, GAMES_DATA } from '../mockData';

export default function DashboardHome() {
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
                <h2 className="text-xl font-bold text-zinc-900 dark:text-white">Upcoming & Recent Games</h2>
                <button className="text-sm text-indigo-600 dark:text-indigo-400 hover:underline font-medium">View All History</button>
            </div>
            <GamesTable games={GAMES_DATA} />
        </div>
    </div>
  );
}