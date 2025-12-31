import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation, Outlet } from 'react-router-dom';
import { 
  LayoutDashboard, 
  BarChart3, 
  Settings, 
  Search, 
  Sun, 
  Moon, 
  ChevronLeft, 
  ChevronRight,
  PanelLeftClose,
  PanelLeftOpen
} from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import DashboardHome from './pages/DashboardHome';
import Performance from './pages/Performance';
import QueryTool from './pages/QueryTool';
import SettingsPage from './pages/SettingsPage';

// --- LAYOUT COMPONENT ---
function Layout() {
  const [activeTab, setActiveTab] = useState('NBA');
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [isCollapsed, setIsCollapsed] = useState(false); // New State for Sidebar
  const location = useLocation();

  useEffect(() => {
    if (isDarkMode) document.documentElement.classList.add('dark');
    else document.documentElement.classList.remove('dark');
  }, [isDarkMode]);

  return (
    <TooltipProvider delayDuration={0}>
      <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 font-sans transition-colors duration-200 flex">
        
        {/* SIDEBAR (Now Retractable) */}
        {/* <aside 
          className={`fixed left-0 top-0 h-full bg-white dark:bg-zinc-900 border-r border-zinc-200 dark:border-zinc-800 transition-all duration-300 ease-in-out z-20 flex flex-col ${
            isCollapsed ? 'w-20' : 'w-64'
          }`}
        > */}
          {/* LOGO AREA */}
          {/* <div className={`h-20 flex items-center ${isCollapsed ? 'justify-center' : 'px-6 gap-3'}`}>
            <div className="h-8 w-8 bg-indigo-600 rounded-lg flex items-center justify-center font-bold text-xl text-white shrink-0">
              P
            </div>
            {!isCollapsed && (
              <span className="text-xl font-bold tracking-tight text-zinc-900 dark:text-white whitespace-nowrap overflow-hidden transition-opacity duration-300">
                NBAPredictor
              </span>
            )}
          </div> */}

          {/* NAV ITEMS */}
          {/* <nav className="flex-1 px-4 space-y-2 py-4">
            <NavItem to="/" icon={<LayoutDashboard size={20} />} label="Dashboard" active={location.pathname === '/'} collapsed={isCollapsed} />
            <NavItem to="/performance" icon={<BarChart3 size={20} />} label="Model Performance" active={location.pathname === '/performance'} collapsed={isCollapsed} />
            <NavItem to="/query" icon={<Search size={20} />} label="Query Tool" active={location.pathname === '/query'} collapsed={isCollapsed} />
            <NavItem to="/settings" icon={<Settings size={20} />} label="Settings" active={location.pathname === '/settings'} collapsed={isCollapsed} />
          </nav> */}

          {/* COLLAPSE TOGGLE BUTTON */}
          {/* <div className="p-4 border-t border-zinc-200 dark:border-zinc-800 flex justify-end">
             <button 
               onClick={() => setIsCollapsed(!isCollapsed)}
               className={`p-2 rounded-lg hover:bg-zinc-100 dark:hover:bg-zinc-800 text-zinc-500 transition-all ${isCollapsed ? 'mx-auto' : ''}`}
             >
                {isCollapsed ? <PanelLeftOpen size={20} /> : <PanelLeftClose size={20} />}
             </button>
          </div> */}
        {/* </aside> */}

        {/* MAIN CONTENT AREA */}
        <div 
          className={`flex-1 flex flex-col min-h-screen transition-all duration-300 ease-in-out`}
        >
          <header className="h-16 border-b border-zinc-200 dark:border-zinc-800 bg-white/50 dark:bg-zinc-900/50 backdrop-blur-xl sticky top-0 z-10 flex items-center justify-between px-8">
              <div className="flex items-center gap-6">
                  <span className="text-zinc-500 dark:text-zinc-400 text-sm hidden sm:block">Season 2025-2026</span>
                  {/* <div className="h-4 w-[1px] bg-zinc-300 dark:bg-zinc-700 hidden sm:block"></div>
                  <div className="flex gap-4">
                      {['NBA', 'NFL', 'MLB'].map(league => (
                          <button key={league} onClick={() => setActiveTab(league)}
                              className={`text-sm font-medium transition-colors ${activeTab === league ? 'text-zinc-900 dark:text-white font-bold' : 'text-zinc-500 hover:text-zinc-700 dark:text-zinc-500 dark:hover:text-zinc-300'}`}>
                              {league}
                          </button>
                      ))}
                  </div> */}
              </div>
              <div className="flex items-center gap-4">
                  <button onClick={() => setIsDarkMode(!isDarkMode)} className="p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-full text-zinc-500 dark:text-zinc-400 transition">
                      {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
                  </button>
                  {/* <div className="h-8 w-8 bg-indigo-600 rounded-full flex items-center justify-center font-medium text-sm text-white">JD</div> */}
              </div>
          </header>

          <main className="flex-1 p-8 overflow-auto">
              <Outlet />
          </main>
        </div>
      </div>
    </TooltipProvider>
  );
}

// --- HELPER: Smart Nav Item ---
interface NavItemProps { 
  to: string; 
  icon: any; 
  label: string; 
  active: boolean; 
  collapsed: boolean;
}

function NavItem({ to, icon, label, active, collapsed }: NavItemProps) {
    const content = (
        <Link to={to}>
            <button className={`w-full flex items-center gap-3 px-3 py-3 rounded-lg text-sm font-medium transition-all ${
                active 
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-900/20' 
                : 'text-zinc-600 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-800 hover:text-zinc-900 dark:hover:text-white'
            } ${collapsed ? 'justify-center' : ''}`}>
                {icon}
                {/* Only show label if NOT collapsed */}
                {!collapsed && <span className="whitespace-nowrap overflow-hidden transition-all duration-300">{label}</span>}
            </button>
        </Link>
    );

    // If collapsed, wrap in Tooltip
    if (collapsed) {
        return (
            <Tooltip>
                <TooltipTrigger asChild>
                    {content}
                </TooltipTrigger>
                <TooltipContent side="right" className="bg-zinc-900 text-white border-zinc-800">
                    <p>{label}</p>
                </TooltipContent>
            </Tooltip>
        );
    }

    return content;
}

// --- MAIN ROUTER ---
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<DashboardHome />} />
         <Route path="performance" element={<Performance />} />
          <Route path="query" element={<QueryTool />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}