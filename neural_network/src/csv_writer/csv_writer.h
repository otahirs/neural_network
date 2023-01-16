#ifndef NEURAL_NETWORK_CSV_WRITER_H
#define NEURAL_NETWORK_CSV_WRITER_H

#include <iostream>
#include <stdexcept>
#include <vector>
#include <linalg.h>

using namespace std;

class CsvWriter {
public:
    static void writeMatrixToFile(const Matrix &matrix, const string &filePath, bool trailingComma = false);
    static void writeVectorToFile(const Vector &vector, const string &filePath, bool trailingComma = false);
};

#endif //NEURAL_NETWORK_CSV_WRITER_H
