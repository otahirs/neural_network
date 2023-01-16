#include <vector>
#include <fstream>
#include <sstream>
#include <linalg.h>
#include <logger.h>
#include "csv_writer.h"

using namespace std;

void CsvWriter::writeMatrixToFile(const Matrix &matrix, const string &filePath, bool trailingComma /* = false */) {
    std::ofstream file(filePath);
    for (const Vector &line : matrix) {
        bool firstValue = true;
        for (const float &value : line) {
            if (!firstValue) {
                file << ',';
            }
            file << value;
            firstValue = false;
        }
        if (trailingComma) {
            file << ',';
        }
        file << '\n';
    }
}

void CsvWriter::writeVectorToFile(const Vector &vector, const string &filePath, bool trailingComma /* = false */) {
    std::ofstream file(filePath);
    for (const float &value : vector) {
        file << value;
        if (trailingComma) {
            file << ',';
        }
        file << '\n';
    }
}
