import { ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

interface StatCardProps {
  label: string;
  value: string;
  trend: 'up' | 'down' | 'neutral';
}

const StatCard = ({ label, value, trend }: StatCardProps) => {
  const getTrendColor = () => {
    if (trend === 'up') return 'text-emerald-600 dark:text-emerald-400';
    if (trend === 'down') return 'text-rose-600 dark:text-rose-400';
    return 'text-zinc-500 dark:text-zinc-400';
  };

  const getIcon = () => {
    if (trend === 'up') return <ArrowUpRight className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />;
    if (trend === 'down') return <ArrowDownRight className="h-4 w-4 text-rose-600 dark:text-rose-400" />;
    return <Minus className="h-4 w-4 text-zinc-500 dark:text-zinc-400" />;
  };

  return (
    <Card className="bg-white dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800 shadow-sm hover:shadow-md transition-shadow">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-xs font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">
          {label}
        </CardTitle>
        {getIcon()}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold text-zinc-900 dark:text-white">{value}</div>
        <p className={`text-xs ${getTrendColor()} mt-1 flex items-center gap-1 font-medium`}>
          {trend === 'up' ? '+2.5% from last week' : 'No change'}
        </p>
      </CardContent>
    </Card>
  );
};

export default StatCard;