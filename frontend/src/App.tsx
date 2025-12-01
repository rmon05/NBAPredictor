import React, { useState, useEffect } from 'react';
import { LayoutDashboard, BarChart3, Settings, Search, Bell, Sun, Moon } from 'lucide-react';
import StatCard from './components/StatCard';
import GamesTable from './components/GamesTable';
import { SUMMARY_STATS, GAMES_DATA } from './mockData';

function App() {
  const [activeTab, setActiveTab] = useState('NBA');
  const [isDarkMode, setIsDarkMode] = useState(true);

  // Toggle Dark Mode Class on HTML element
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 font-sans transition-colors duration-200">
      
      {/* SIDEBAR */}
      <aside className="fixed left-0 top-0 h-full w-64 bg-white dark:bg-zinc-900 border-r border-zinc-200 dark:border-zinc-800 p-6 hidden md:flex flex-col z-20 transition-colors duration-200">
        <div className="mb-10 flex items-center gap-3">
          <div className="h-8 w-8 bg-indigo-600 rounded-lg flex items-center justify-center font-bold text-xl text-white">P</div>
          <span className="text-xl font-bold tracking-tight text-zinc-900 dark:text-white">NBAPredictor</span>
        </div>

        <nav className="flex-1 space-y-2">
          <NavItem icon={<LayoutDashboard size={20} />} label="Dashboard" active />
          <NavItem icon={<BarChart3 size={20} />} label="Model Performance" />
          <NavItem icon={<Search size={20} />} label="Query Tool" />
          <NavItem icon={<Settings size={20} />} label="Settings" />
        </nav>
      </aside>

      {/* MAIN CONTENT AREA */}
      <main className="md:ml-64 min-h-screen flex flex-col">
        
        {/* HEADER */}
        <header className="h-16 border-b border-zinc-200 dark:border-zinc-800 bg-white/50 dark:bg-zinc-900/50 backdrop-blur-xl sticky top-0 z-10 flex items-center justify-between px-8 transition-colors duration-200">
            <div className="flex items-center gap-6">
                <span className="text-zinc-500 dark:text-zinc-400 text-sm">Season 2024-2025</span>
                <div className="h-4 w-[1px] bg-zinc-300 dark:bg-zinc-700"></div>
                <div className="flex gap-4">
                    {['NBA', 'NFL', 'MLB'].map(league => (
                        <button 
                            key={league}
                            onClick={() => setActiveTab(league)}
                            className={`text-sm font-medium transition-colors ${
                                activeTab === league 
                                ? 'text-zinc-900 dark:text-white font-bold' 
                                : 'text-zinc-500 hover:text-zinc-700 dark:text-zinc-500 dark:hover:text-zinc-300'
                            }`}
                        >
                            {league}
                        </button>
                    ))}
                </div>
            </div>
            
            <div className="flex items-center gap-4">
                <button 
                  onClick={() => setIsDarkMode(!isDarkMode)}
                  className="p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-full text-zinc-500 dark:text-zinc-400 transition"
                >
                    {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
                </button>
                <div className="h-8 w-8 bg-indigo-600 rounded-full flex items-center justify-center font-medium text-sm text-white">
                    JD
                </div>
            </div>
        </header>

        {/* DASHBOARD CONTENT */}
        <div className="p-8 max-w-7xl mx-auto space-y-8 w-full">
            
            {/* 1. HERO STATS */}
            <div>
                <h1 className="text-2xl font-bold mb-6 text-zinc-900 dark:text-white">Overview</h1>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    {SUMMARY_STATS.map((stat, i) => (
                        <StatCard key={i} {...stat} />
                    ))}
                </div>
            </div>

            {/* 2. RECENT GAMES TABLE */}
            <div className="space-y-4">
                <div className="flex items-center justify-between">
                    <h2 className="text-xl font-bold text-zinc-900 dark:text-white">Upcoming & Recent Games</h2>
                    <button className="text-sm text-indigo-600 dark:text-indigo-400 hover:underline font-medium">View All History</button>
                </div>
                <GamesTable games={GAMES_DATA} />
            </div>

        </div>
      </main>
    </div>
  );
}

// Helper Component for Sidebar Items
interface NavItemProps {
    icon: React.ReactNode;
    label: string;
    active?: boolean;
}

function NavItem({ icon, label, active }: NavItemProps) {
    return (
        <button className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
            active 
            ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-900/20' 
            : 'text-zinc-600 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-800 hover:text-zinc-900 dark:hover:text-white'
        }`}>
            {icon}
            {label}
        </button>
    )
}

export default App;