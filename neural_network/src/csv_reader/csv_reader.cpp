#include <vector>
#include <fstream>
#include <sstream>
#include <logger.h>
#include "csv_reader.h"

using namespace std;

class FileNotExistsException : public std::runtime_error {
public:
    explicit FileNotExistsException(const std::string &msg) : runtime_error(msg) {}
};

vector<vector<float>> CsvReader::readStreamToMatrix(istream &istream) {
    vector<vector<float>> result;
    string line;
    while (std::getline(istream, line)) {
        processLine(line, result);
    }
    return result;
}

vector<float> CsvReader::readStreamToVector(istream &istream) {
    vector<float> result;
    string line;
    while (std::getline(istream, line)) {
        result.push_back(stof(line));
    }
    return result;
}

void CsvReader::processLine(const string &line, vector<vector<float>> &data) {
    vector<float> values;
    istringstream line_stream(line);
    string value;
    while (getline(line_stream, value, ',')) {
        values.push_back(stof(value));
    }
    line_stream.get();
    data.push_back(values);
}

vector<vector<float>> CsvReader::readStringToMatrix(const string &s) {
    std::istringstream istr(s);
    return readStreamToMatrix(istr);
}

vector<vector<float>> CsvReader::readFileToMatrix(const string &filePath) {
    Logger::info("Reading file " + filePath + " to matrix.");
    fstream file(filePath);
    checkFileExists(file, filePath);
    return readStreamToMatrix(file);
}

vector<float> CsvReader::readFileToVector(const string &filePath) {
    Logger::info("Reading file " + filePath + " to vector.");
    fstream file(filePath);
    checkFileExists(file, filePath);
    return readStreamToVector(file);
}

void CsvReader::checkFileExists(const fstream &file, const string &filePath) {
    if (!file) {
        throw FileNotExistsException("File " + filePath + " does not exist.");
    }
}
