#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <algorithm>


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

bool compareDataRowByDateAsc(const DataRow* a, const DataRow* b) {
    return a->date < b->date;
}

void processSeasonStats(const std::string& filePath, std::vector<DataRow*>& seasonData, std::unordered_map<std::string, int>& teamIndex) {
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

void writeProcessedData(const std::vector<DataRow*>& seasonData, const std::string& outputFilePath){
    std::ofstream outFile(outputFilePath);
    if(!outFile.is_open()){
        std::cerr << "Error: Could not open file for writing: " << outputFilePath << std::endl;
        return;
    }

    // Header
    outFile << "Home,Away,Date,Won, ,";
    outFile << "PointDiff,FTDiff,Home2pt,Away2pt,Home3pt,Away3pt,Def2pt,OppDef2pt,Def3pt,OppDef3pt\n";

    // Write each row of processed data
    for(const auto& row : seasonData){
        outFile << row->homeTeam << "," << row->awayTeam << "," << row->date << "," << row->result << ", ,"; 
        int pointDiff = row->homePts - row->awayPts;
        int ftDiff = row->homeFT - row->awayFT;
        int home2pt = row->home2pt;
        int away2pt = row->away2pt;
        int home3pt = row->home3pt;
        int away3pt = row->away3pt;
        int def2pt = row->homeDefended2ptPct;
        int oppDef2pt = row->awayDefended2ptPct;
        int def3pt = row->homeDefended3ptPct;
        int oppDef3pt = row->awayDefended3ptPct;
        outFile << pointDiff << "," << ftDiff << "," << home2pt << "," << away2pt << "," << home3pt << "," << away3pt
                << "," << def2pt << "," << oppDef2pt << "," << def3pt << "," << oppDef3pt << "\n";
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

    // read all data files in stathead folder    
    std::string rawStatheadFolder = "./raw/stathead";
    std::string processedFolder = "./processed";
    for(const auto& seasonStats : std::filesystem::directory_iterator(rawStatheadFolder)){
        if(seasonStats.is_regular_file() && seasonStats.path().extension() == ".csv"){
            // process season by season
            std::vector<DataRow*> seasonData;
            processSeasonStats(seasonStats.path().string(), seasonData, teamIndex);
            // then write processed data out
            std::string baseName = seasonStats.path().stem().string();
            std::string outputFilePath = processedFolder + "/" + baseName + "_processed.csv";
            writeProcessedData(seasonData, outputFilePath);
        }
    }
    return 0;
}
