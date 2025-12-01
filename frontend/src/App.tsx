import React, { useState } from 'react';
import { LayoutDashboard, BarChart3, Settings, Search, Bell } from 'lucide-react';
import StatCard from './components/StatCard';
import GamesTable from './components/GamesTable';
import { SUMMARY_STATS, GAMES_DATA } from './mocData';

function App() {
  const [activeTab, setActiveTab] = useState('NBA');

  return (
    <div className="min-h-screen bg-gray-950 text-white font-sans selection:bg-indigo-500/30">
      
      {/* SIDEBAR */}
      <aside className="fixed left-0 top-0 h-full w-64 bg-gray-900 border-r border-gray-800 p-6 hidden md:flex flex-col">
        <div className="mb-10 flex items-center gap-2">
          <div className="h-8 w-8 bg-indigo-600 rounded-lg flex items-center justify-center font-bold text-xl">P</div>
          <span className="text-xl font-bold tracking-tight">NBAPredictor</span>
        </div>

        <nav className="flex-1 space-y-2">
          <NavItem icon={<LayoutDashboard size={20} />} label="Dashboard" active />
          <NavItem icon={<BarChart3 size={20} />} label="Model Performance" />
          <NavItem icon={<Search size={20} />} label="Query Tool" />
          <NavItem icon={<Settings size={20} />} label="Settings" />
        </nav>

        <div className="mt-auto pt-6 border-t border-gray-800">
           <div className="text-xs text-gray-500 uppercase font-bold mb-4">Saved Models</div>
           <div className="space-y-3 text-sm text-gray-400">
             <div className="flex items-center gap-2 cursor-pointer hover:text-white"><div className="w-2 h-2 rounded-full bg-emerald-500"></div> XGBoost Light</div>
             <div className="flex items-center gap-2 cursor-pointer hover:text-white"><div className="w-2 h-2 rounded-full bg-amber-500"></div> RNN Deep</div>
           </div>
        </div>
      </aside>

      {/* MAIN CONTENT AREA */}
      <main className="md:ml-64 min-h-screen">
        
        {/* HEADER */}
        <header className="h-16 border-b border-gray-800 bg-gray-900/50 backdrop-blur-xl sticky top-0 z-10 flex items-center justify-between px-8">
            <div className="flex items-center gap-6">
                <span className="text-gray-400 text-sm">Season 2024-2025</span>
                <div className="h-4 w-[1px] bg-gray-700"></div>
                <div className="flex gap-4">
                    {['NBA', 'NFL', 'MLB'].map(league => (
                        <button 
                            key={league}
                            onClick={() => setActiveTab(league)}
                            className={`text-sm font-medium transition-colors ${
                                activeTab === league ? 'text-white' : 'text-gray-500 hover:text-gray-300'
                            }`}
                        >
                            {league}
                        </button>
                    ))}
                </div>
            </div>
            
            <div className="flex items-center gap-4">
                <button className="p-2 hover:bg-gray-800 rounded-full text-gray-400 transition">
                    <Bell size={20} />
                </button>
                <div className="h-8 w-8 bg-indigo-500 rounded-full flex items-center justify-center font-medium text-sm">
                    JD
                </div>
            </div>
        </header>

        {/* DASHBOARD CONTENT */}
        <div className="p-8 max-w-7xl mx-auto space-y-8">
            
            {/* 1. HERO STATS */}
            <div>
                <h1 className="text-2xl font-bold mb-6">Overview</h1>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    {SUMMARY_STATS.map((stat, i) => (
                        <StatCard key={i} {...stat} />
                    ))}
                </div>
            </div>

            {/* 2. RECENT GAMES TABLE */}
            <div className="space-y-4">
                <div className="flex items-center justify-between">
                    <h2 className="text-xl font-bold">Upcoming & Recent Games</h2>
                    <button className="text-sm text-indigo-400 hover:text-indigo-300 font-medium">View All History</button>
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
            : 'text-gray-400 hover:bg-gray-800 hover:text-white'
        }`}>
            {icon}
            {label}
        </button>
    )
}

export default App;