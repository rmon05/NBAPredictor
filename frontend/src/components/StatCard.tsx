import { ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

interface StatCardProps {
  label: string;
  value: string;
  trend: 'up' | 'down' | 'neutral';
}

const StatCard = ({ label, value, trend }: StatCardProps) => {
  const getTrendColor = () => {
    if (trend === 'up') return 'text-emerald-500';
    if (trend === 'down') return 'text-rose-500';
    return 'text-muted-foreground';
  };

  const getIcon = () => {
    if (trend === 'up') return <ArrowUpRight className="h-4 w-4 text-emerald-500" />;
    if (trend === 'down') return <ArrowDownRight className="h-4 w-4 text-rose-500" />;
    return <Minus className="h-4 w-4 text-muted-foreground" />;
  };

  return (
    <Card className="bg-card/50 backdrop-blur-sm border-muted/40 hover:bg-muted/50 transition-colors">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
          {label}
        </CardTitle>
        {getIcon()}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold text-foreground">{value}</div>
        <p className={`text-xs ${getTrendColor()} mt-1 flex items-center gap-1`}>
          {trend === 'up' ? '+2.5% from last week' : 'No change'}
        </p>
      </CardContent>
    </Card>
  );
};

export default StatCard;