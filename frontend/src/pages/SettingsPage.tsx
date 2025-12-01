import React from 'react';
import { Card } from "@/components/ui/card";

export default function Settings() {
  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-zinc-900 dark:text-white">Settings</h1>
      
      <div className="space-y-6">
        <Section title="Appearance">
            <SettingRow label="Dark Mode" desc="Toggle system dark theme" toggle />
            <SettingRow label="Compact Mode" desc="Decrease spacing in tables" toggle />
        </Section>

        <Section title="Model Configuration">
             <SettingRow label="Confidence Threshold" desc="Minimum confidence to show a pick (current: 60%)" />
             <SettingRow label="Default Book" desc="Sportsbook used for line comparison" value="DraftKings" />
        </Section>

        <Section title="Data Source">
            <SettingRow label="Last Update" desc="Timestamp of last scrape" value="Today, 4:00 AM" />
             <div className="flex justify-end pt-2">
                 <button className="text-sm text-indigo-500 font-medium hover:underline">Force Refresh Data</button>
             </div>
        </Section>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string, children: React.ReactNode }) {
    return (
        <div className="space-y-4">
            <h3 className="text-lg font-medium text-zinc-900 dark:text-white">{title}</h3>
            <Card className="bg-white dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800 divide-y divide-zinc-100 dark:divide-zinc-800">
                {children}
            </Card>
        </div>
    )
}

function SettingRow({ label, desc, value, toggle }: { label: string, desc: string, value?: string, toggle?: boolean }) {
    return (
        <div className="p-4 flex items-center justify-between">
            <div>
                <div className="font-medium text-zinc-900 dark:text-white">{label}</div>
                <div className="text-sm text-zinc-500">{desc}</div>
            </div>
            {toggle ? (
                <div className="w-10 h-6 bg-zinc-200 dark:bg-zinc-700 rounded-full relative cursor-pointer">
                    <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full shadow-sm"></div>
                </div>
            ) : (
                <span className="text-sm text-zinc-500">{value}</span>
            )}
        </div>
    )
}