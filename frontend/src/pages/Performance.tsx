import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function Performance() {
  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-zinc-900 dark:text-white">Model Performance</h1>
        <div className="flex gap-2">
            <span className="px-3 py-1 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 rounded-full text-sm font-medium border border-emerald-200 dark:border-emerald-800">
                Live Model v2.1
            </span>
        </div>
      </div>

      {/* KPI GRID */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <KPICard title="Global Accuracy" value="62.4%" sub="Across 1,204 games" />
        <KPICard title="Current ROI" value="+14.2%" sub="Season to date" />
        <KPICard title="Mean Abs Error" value="4.1" sub="Points off spread" />
      </div>

      {/* CHART PLACEHOLDERS */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="h-96 bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 flex flex-col">
            <h3 className="text-zinc-500 font-medium mb-4">Profit over Time (Season)</h3>
            <div className="flex-1 border-2 border-dashed border-zinc-100 dark:border-zinc-800 rounded-lg flex items-center justify-center text-zinc-400">
                Line Chart will go here
            </div>
        </div>
        <div className="h-96 bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 flex flex-col">
            <h3 className="text-zinc-500 font-medium mb-4">Accuracy by Confidence</h3>
            <div className="flex-1 border-2 border-dashed border-zinc-100 dark:border-zinc-800 rounded-lg flex items-center justify-center text-zinc-400">
                Bar Chart will go here
            </div>
        </div>
      </div>
    </div>
  );
}

function KPICard({ title, value, sub }: { title: string, value: string, sub: string }) {
    return (
        <Card className="bg-white dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800">
            <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-zinc-500 dark:text-zinc-400 uppercase">{title}</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="text-3xl font-bold text-zinc-900 dark:text-white">{value}</div>
                <p className="text-xs text-zinc-500 mt-1">{sub}</p>
            </CardContent>
        </Card>
    )
}