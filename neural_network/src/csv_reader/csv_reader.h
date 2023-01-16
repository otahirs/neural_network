#ifndef NEURAL_NETWORK_CSV_READER_H
#define NEURAL_NETWORK_CSV_READER_H

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

class CsvReader {
    static vector<vector<float>> readStreamToMatrix(istream &istream);
    static vector<float> readStreamToVector(istream &istream);
    static void processLine(const string &line, vector<vector<float>>& data);
    static void checkFileExists(const fstream& file, const string &filePath);
public:
    static vector<vector<float>> readStringToMatrix(const string& s);
    static vector<vector<float>> readFileToMatrix(const string& filePath);
    static vector<float> readFileToVector(const string &filePath);
};

#endif //NEURAL_NETWORK_CSV_READER_H
