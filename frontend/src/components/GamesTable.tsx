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
  id: number;
  date: string;
  team: string;
  opponent: string;
  location: string;
  line: number;
  prediction: number;
  edge: number;
  result: string;
  score: string;
  confidence: string;
}

interface GamesTableProps {
  games: Game[];
}

const GamesTable = ({ games }: GamesTableProps) => {
  return (
    // Added solid background colors and borders for better separation
    <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 overflow-hidden shadow-sm">
      <Table>
        <TableHeader>
          <TableRow className="bg-zinc-50 dark:bg-zinc-900/50 hover:bg-zinc-50 dark:hover:bg-zinc-900/50 border-zinc-200 dark:border-zinc-800">
            <TableHead className="w-[120px] text-zinc-500 dark:text-zinc-400 font-semibold">Date</TableHead>
            <TableHead className="text-zinc-500 dark:text-zinc-400 font-semibold">Matchup</TableHead>
            <TableHead className="text-center text-zinc-500 dark:text-zinc-400 font-semibold">Location</TableHead>
            <TableHead className="text-center text-zinc-500 dark:text-zinc-400 font-semibold">Line</TableHead>
            <TableHead className="text-center text-zinc-500 dark:text-zinc-400 font-semibold">My Pred</TableHead>
            <TableHead className="text-center text-zinc-500 dark:text-zinc-400 font-semibold">Edge</TableHead>
            <TableHead className="text-center text-zinc-500 dark:text-zinc-400 font-semibold">Result</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {games.map((game) => (
            <TableRow key={game.id} className="border-zinc-100 dark:border-zinc-800 hover:bg-zinc-50 dark:hover:bg-zinc-800/50 transition-colors">
              <TableCell className="font-medium text-zinc-900 dark:text-zinc-200">{game.date}</TableCell>
              
              <TableCell>
                <div className="flex flex-col">
                  <span className="font-bold text-zinc-900 dark:text-zinc-100">{game.team}</span>
                  <span className="text-xs text-zinc-500 dark:text-zinc-400">vs {game.opponent}</span>
                </div>
              </TableCell>

              <TableCell className="text-center">
                 <Badge variant="outline" className={game.location === 'home' 
                    ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800' 
                    : 'bg-orange-50 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300 border-orange-200 dark:border-orange-800'}>
                     {game.location.toUpperCase()}
                 </Badge>
              </TableCell>

              <TableCell className="text-center font-mono text-zinc-700 dark:text-zinc-300">
                {game.line > 0 ? `+${game.line}` : game.line}
              </TableCell>

              <TableCell className="text-center font-mono font-bold text-indigo-600 dark:text-indigo-400">
                {game.prediction > 0 ? `+${game.prediction}` : game.prediction}
              </TableCell>

              <TableCell className="text-center">
                  <span className={`font-bold ${game.edge > 2 ? 'text-emerald-600 dark:text-emerald-400' : 'text-zinc-400 dark:text-zinc-500'}`}>
                      {game.edge > 0 ? `+${game.edge}` : game.edge}
                  </span>
              </TableCell>

              <TableCell className="text-center">
                <Badge className={`w-8 h-8 flex items-center justify-center mx-auto p-0 rounded-full border-0 ${
  game.result === 'W' ? 'bg-emerald-100 dark:bg-emerald-500/20 text-emerald-700 dark:text-emerald-400' : 
  game.result === 'L' ? 'bg-rose-100 dark:bg-rose-500/20 text-rose-700 dark:text-rose-400' : 
  'bg-zinc-100 dark:bg-zinc-700 text-zinc-500 dark:text-zinc-400'
}`}>
                  {game.result}
                </Badge>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
};

export default GamesTable;