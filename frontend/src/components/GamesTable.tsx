import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";

export interface Game {
  home: string;
  away: string;
  game_date: string;
  spread_prediction: number;
}

interface GamesTableProps {
  games: Game[];
}

const GamesTable = ({ games }: GamesTableProps) => {
  return (
    <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 overflow-hidden shadow-sm">
      <Table>
        <TableHeader>
          <TableRow className="bg-zinc-50 dark:bg-zinc-900/50 hover:bg-zinc-50 dark:hover:bg-zinc-900/50 border-zinc-200 dark:border-zinc-800">
            <TableHead className="w-[140px] text-zinc-500 dark:text-zinc-400 font-semibold">Date</TableHead>
            <TableHead className="text-zinc-500 dark:text-zinc-400 font-semibold">Matchup</TableHead>
            <TableHead className="text-center text-zinc-500 dark:text-zinc-400 font-semibold">Spread Prediction</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {games.map((game, index) => {            
            return (
              <TableRow 
                key={`${game.home}-${game.away}-${index}`} 
                className="border-zinc-100 dark:border-zinc-800 hover:bg-zinc-50 dark:hover:bg-zinc-800/50 transition-colors"
              >
                <TableCell className="font-medium text-zinc-600 dark:text-zinc-400">
                  {game.game_date}
                </TableCell>
                
                <TableCell>
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-zinc-900 dark:text-zinc-100">{game.away}</span>
                    <span className="text-zinc-400 text-xs font-normal">@</span>
                    <span className="font-semibold text-zinc-900 dark:text-zinc-100">{game.home}</span>
                  </div>
                </TableCell>

                <TableCell className="text-center font-mono font-bold text-indigo-600 dark:text-indigo-400 text-lg">
                  {game.spread_prediction > 0 ? `+${game.spread_prediction}` : game.spread_prediction}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
};

export default GamesTable;