import React from 'react';
import { Search, Filter } from 'lucide-react';
import { Card } from "@/components/ui/card";

export default function QueryTool() {
  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      <h1 className="text-3xl font-bold text-zinc-900 dark:text-white">Historical Query</h1>
      
      {/* SEARCH BAR */}
      <Card className="p-6 bg-white dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800">
        <div className="flex gap-4">
            <div className="relative flex-1">
                <Search className="absolute left-3 top-3 text-zinc-400" size={20} />
                <input 
                    type="text" 
                    placeholder="e.g. 'Celtics home underdog vs West'" 
                    className="w-full pl-10 pr-4 py-2 bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-lg text-zinc-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
            </div>
            <button className="px-6 py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-lg transition">
                Run Query
            </button>
            <button className="p-2 border border-zinc-200 dark:border-zinc-800 rounded-lg text-zinc-500 hover:bg-zinc-50 dark:hover:bg-zinc-800">
                <Filter size={20} />
            </button>
        </div>
      </Card>

      {/* EMPTY STATE */}
      <div className="h-64 flex flex-col items-center justify-center text-zinc-400">
        <p>Run a query to see historical trends and model predictions.</p>
      </div>
    </div>
  );
}