#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <algorithm>
#include <iomanip>


// do not generate training data for a year until this many games played
const int buffer_games = 400;

struct DataRow{
    // game metadata
    int homeTeam;
    int awayTeam;
    std::string date;
    int result;
    // home stats
    int home2pt;
    int awayDefended2ptPct;
    int home3pt;
    int awayDefended3ptPct;
    int homeFT;
    int homePts;
    // away stats
    int away2pt;
    int homeDefended2ptPct;
    int away3pt;
    int homeDefended3ptPct;
    int awayFT;
    int awayPts;
};

struct cumuStats{
    int wins = 0;
    int losses = 0;
    int totPointDiff = 0;
    int totFT = 0;
    int tot2pt = 0;
    int tot3pt = 0;
    int totDef2ptPct = 0;
    int totDef3ptPct = 0;
};

bool compareDataRowByDateAsc(const DataRow* a, const DataRow* b) {
    return a->date < b->date;
}

void cleanSeasonStats(const std::string& filePath, std::vector<DataRow*>& seasonData, std::unordered_map<std::string, int>& teamIndex) {
    std::ifstream file(filePath);
    if(!file.is_open()){
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return;
    }

    std::string line;
    bool isHeader = true;

    // Process each line
    while(std::getline(file, line)){
        if(isHeader){
            isHeader = false;
            continue;
        }
        // Process data in each line
        DataRow* currRow = new DataRow;
        std::stringstream lineStream(line);
        std::string cell;
        int cellNumber = 0;
        while(std::getline(lineStream, cell, ',')){
            if(cellNumber == 1){
                // Home team name
                currRow->homeTeam = teamIndex[cell];
            } else if(cellNumber == 2){
                // Date
                currRow->date = cell;
            } else if(cellNumber == 5){
                // Away team name
                currRow->awayTeam = teamIndex[cell];
            } else if(cellNumber == 6){
                // Result
                currRow->result = cell[0]=='W' ? 1 : 0;
            } else if(cellNumber == 11){
                // Home 2pt 
                currRow->home2pt = 2*std::stoi(cell);
            } else if(cellNumber == 13){
                // Home 2pt fg% = Away defended 2pt fg%
                currRow->awayDefended2ptPct = std::stoi(cell.substr(1));
            } else if(cellNumber == 14){
                // Home 3pt 
                currRow->home3pt = 3*std::stoi(cell);
            } else if(cellNumber == 16){
                // Home 3pt fg% = Away defended 3pt fg%
                currRow->awayDefended3ptPct = std::stoi(cell.substr(1));
            } else if(cellNumber == 17){
                // Home FT 
                currRow->homeFT = std::stoi(cell);
            } else if(cellNumber == 20){
                // Home pts
                currRow->homePts = std::stoi(cell);                
            } else if(cellNumber == 24){
                // Away 2pt
                currRow->away2pt = 2*std::stoi(cell);
            } else if(cellNumber == 26){
                // Away 2pt fg% = Home defended 2pt fg%
                currRow->homeDefended2ptPct = std::stoi(cell.substr(1));
            } else if(cellNumber == 27){
                // Away 3pt
                currRow->away3pt = 3*std::stoi(cell);
            } else if(cellNumber == 29){
                // Away 3pt fg% = Home defended 3pt fg%
                currRow->homeDefended3ptPct = std::stoi(cell.substr(1));
            } else if(cellNumber == 30){
                // Away FT
                currRow->awayFT = std::stoi(cell);
            } else if(cellNumber == 33){
                // Away pts
                currRow->awayPts = std::stoi(cell);    
            }
            cellNumber++;
        }
        seasonData.push_back(currRow);
    }
    // sort seasonData by ascending date
    std::sort(seasonData.begin(), seasonData.end(), compareDataRowByDateAsc);
    file.close();
}

void generateProcessedData(const std::vector<DataRow*>& seasonData, const std::string& outputFilePath){
    std::ofstream outFile(outputFilePath, std::ios::app); // write all processed data in one file
    if(!outFile.is_open()){
        std::cerr << "Error: Could not open file for writing: " << outputFilePath << std::endl;
        return;
    }

    int processed_games = 0;

    // Process and write data if past buffer
    std::vector<cumuStats*> teamStats(30);
    for(int i = 0; i < 30; i++){
        teamStats[i] = new cumuStats{};
    }
    for(const auto& row : seasonData){
        processed_games++;
        int home = row->homeTeam;
        int away = row->awayTeam;
        int homePlayed = teamStats[home]->wins + teamStats[home]->losses;
        int awayPlayed = teamStats[away]->wins + teamStats[away]->losses;
        // Process game result if applicable
        if(processed_games > buffer_games){
            int won = row->result;
            int homeCourt = 1;
            double homePointDiff = double(teamStats[home]->totPointDiff) / double(homePlayed);
            double awayPointDiff = double(teamStats[away]->totPointDiff) / double(awayPlayed);
            double homeFT = double(teamStats[home]->totFT)/double(homePlayed);
            double awayFT = double(teamStats[away]->totFT)/double(awayPlayed);
            double home2pt = double(teamStats[home]->tot2pt) / double(homePlayed);
            double home3pt = double(teamStats[home]->tot3pt) / double(homePlayed);
            double def2pt = double(teamStats[home]->totDef2ptPct) / double(homePlayed);
            double def3pt = double(teamStats[home]->totDef3ptPct) / double(homePlayed);        
            double away2pt = double(teamStats[away]->tot2pt) / double(awayPlayed);
            double away3pt = double(teamStats[away]->tot3pt) / double(awayPlayed);
            double awayDef2pt = double(teamStats[away]->totDef2ptPct) / double(awayPlayed);
            double awayDef3pt = double(teamStats[away]->totDef3ptPct) / double(awayPlayed);
            // Output data
            outFile << std::fixed << std::setprecision(0) 
            << won << "," << homeCourt << "," << homePointDiff << "," << homeFT << "," << home2pt << ","
            << home3pt << "," << def2pt << "," << def3pt << ","
            << awayPointDiff << "," << awayFT << "," << away2pt << "," << away3pt << "," << awayDef2pt << "," << awayDef3pt << "\n";
        }
        // Update cumulative stats
        teamStats[home]->wins += row->result;
        teamStats[home]->losses += !row->result;
        teamStats[away]->wins += !row->result;
        teamStats[away]->losses += row->result;
        teamStats[home]->totPointDiff += row->homePts - row->awayPts;
        teamStats[home]->totFT += row->homeFT;
        teamStats[home]->tot2pt += row->home2pt;
        teamStats[home]->tot3pt += row->home3pt;
        teamStats[home]->totDef2ptPct += row->homeDefended2ptPct;
        teamStats[home]->totDef3ptPct += row->homeDefended3ptPct;
        teamStats[away]->totPointDiff += row->awayPts - row->homePts;
        teamStats[away]->totFT += row->awayFT;
        teamStats[away]->tot2pt += row->away2pt;
        teamStats[away]->tot3pt += row->away3pt;
        teamStats[away]->totDef2ptPct += row->awayDefended2ptPct;
        teamStats[away]->totDef3ptPct += row->awayDefended3ptPct;
    }
    outFile.close();
}

int main() {
    // index all 30 teams 
    // note basketball reference differs from nba.com and uses BRK not BKN, CHO not CHA
    std::vector<std::string> teams = {"ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", 
                                        "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", 
                                        "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"};
    std::unordered_map<std::string, int> teamIndex;
    for (int i = 0; i < teams.size(); i++){
        teamIndex[teams[i]] = i;
    }
    
    // clear existing data and output header
    std::string processedPath = "./processed/all_processed_data.csv";
    std::ofstream outFile(processedPath, std::ios::trunc);
    // Header
    outFile << "Won,HomeCourt,HomePointDiff,HomeFT,Home2pt,Home3pt,Def2pt,Def3pt,";
    outFile << "AwayPointDiff,AwayFT,Away2pt,Away3pt,OppDef2pt,OppDef3pt\n";
    outFile.close();
    // read all data files in stathead folder    
    std::string rawStatheadFolder = "./raw/stathead";
    for(const auto& seasonStats : std::filesystem::directory_iterator(rawStatheadFolder)){
        if(seasonStats.is_regular_file() && seasonStats.path().extension() == ".csv"){
            // process season by season
            std::vector<DataRow*> seasonData;
            cleanSeasonStats(seasonStats.path().string(), seasonData, teamIndex);
            // then write processed data out
            generateProcessedData(seasonData, processedPath);
        }
    }
    return 0;
}
