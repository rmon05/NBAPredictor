import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "./ui/badge";

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
    <div className="rounded-md border bg-card/50 backdrop-blur-sm">
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent border-muted/40">
            <TableHead className="w-[120px]">Date</TableHead>
            <TableHead>Matchup</TableHead>
            <TableHead className="text-center">Loc</TableHead>
            <TableHead className="text-center">Line</TableHead>
            <TableHead className="text-center">My Pred</TableHead>
            <TableHead className="text-center">Edge</TableHead>
            <TableHead className="text-center">Result</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {games.map((game) => (
            <TableRow key={game.id} className="border-muted/40 hover:bg-muted/50">
              <TableCell className="font-medium">{game.date}</TableCell>
              
              <TableCell>
                <div className="flex flex-col">
                  <span className="font-bold">{game.team}</span>
                  <span className="text-xs text-muted-foreground">vs {game.opponent}</span>
                </div>
              </TableCell>

              <TableCell className="text-center">
                 <Badge variant="outline" className={game.location === 'home' ? 'bg-blue-500/10 text-blue-500 border-blue-500/20' : 'bg-orange-500/10 text-orange-500 border-orange-500/20'}>
                     {game.location}
                 </Badge>
              </TableCell>

              <TableCell className="text-center font-mono">
                {game.line > 0 ? `+${game.line}` : game.line}
              </TableCell>

              <TableCell className="text-center font-mono text-indigo-400">
                {game.prediction > 0 ? `+${game.prediction}` : game.prediction}
              </TableCell>

              <TableCell className="text-center">
                  <span className={`font-bold ${game.edge > 2 ? 'text-emerald-500' : 'text-muted-foreground'}`}>
                      {game.edge > 0 ? `+${game.edge}` : game.edge}
                  </span>
              </TableCell>

              <TableCell className="text-center">
                <Badge className={`w-8 h-8 flex items-center justify-center p-0 rounded-full ${
                  game.result === 'W' ? 'bg-emerald-500 hover:bg-emerald-600' : 
                  game.result === 'L' ? 'bg-rose-500 hover:bg-rose-600' : 
                  'bg-gray-500 hover:bg-gray-600'
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