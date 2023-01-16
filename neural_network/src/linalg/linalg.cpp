#include "linalg.h"
#include <vector>
#include <sstream>
#include <string>
#include <cmath>

using namespace std;

Vector::Vector() : Vector(0) {}

Vector::Vector(int dimension) {
    _vector = std::vector<float>(dimension, 0);
}

Vector::Vector(const vector<float> &v) {
    _vector = v;
}

void Vector::checkIndexInRange(int index) const {
    //if (index < 0 || index >= size()) {
    //    throw OutOfRangeException();
    //}
}

const float &Vector::operator[](int index) const {
    checkIndexInRange(index);
    return _vector[index];
}

float &Vector::operator[](int index) {
    checkIndexInRange(index);
    return _vector[index];
}

int Vector::size() const {
    return _vector.size();
}

template<typename Lambda>
Vector Vector::mapper(const Vector &a, Lambda f) {
    Vector result(a.size());
    for (int i = 0; i < a.size(); ++i) {
        result[i] = f(a, i);
    }
    return result;
}

template<typename Lambda>
void Vector::mapperInPlace(Lambda f) {
    for (int i = 0; i < size(); ++i) {
        (*this)[i] = f(*this, i);
    }
}

void Vector::checkSameSize(const Vector &a, const Vector &b, const string &operation) {
    //if (a.size() != b.size()) {
    //    throw Vector::DifferentSizeException(operation);
    //}
}

Vector operator+(const Vector &a, const Vector &b) {
    Vector::checkSameSize(a, b, "+");
    return Vector::mapper(a, [&b](const Vector &a, int i) -> float { return a[i] + b[i]; });
}

Vector &Vector::add(const Vector &b) {
    Vector::checkSameSize(*this, b, "+");
    Vector::mapperInPlace([&b](const Vector &a, int i) -> float { return a[i] + b[i]; });
    return *this;
}

Vector operator-(const Vector &a, const Vector &b) {
    Vector::checkSameSize(a, b, "-");
    return Vector::mapper(a, [&b](const Vector &a, int i) -> float { return a[i] - b[i]; });
}

Vector operator*(const Vector &a, const float &scalar) {
    return Vector::mapper(a, [&scalar](const Vector &a, int i) -> float { return a[i] * scalar; });
}

Vector operator*(const float &scalar, const Vector &a) {
    return a * scalar;
}

float operator*(const Vector &a, const Vector &b) {
    Vector::checkSameSize(a, b, "dot product");
    float result = 0;
    for (int i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

bool operator==(const Vector &a, const Vector &b) {
    return a._vector == b._vector;
}

bool operator!=(const Vector &a, const Vector &b) {
    return !(a == b);
}

ostream &operator<<(ostream &os, const Vector &vector) {
    bool firstValue = true;
    for (float num : vector) {
        if (!firstValue) {
            os << ' ';
        }
        os << num;
        firstValue = false;
    }
    return os;
}

string Vector::toString() const {
    ostringstream oss;
    oss << *this;
    return oss.str();
}

float Vector::max() const {
    auto it = max_element(_vector.begin(), _vector.end());
    if (it == _vector.end()) {
        return 0;
    }
    return *it;
}

int Vector::argmax() const {
    auto it = max_element(_vector.begin(), _vector.end());
    if (it == _vector.end()) {
        return -1;
    }
    return it - _vector.begin();
}

float Vector::sum() const {
    float result = 0;
    for (const float &num : _vector) {
        result += num;
    }
    return result;
}

Vector Vector::exp() const {
    return Vector::mapper(*this, [](const Vector &a, int i) -> float { return std::exp(a[i]); });
}

void Vector::checkNotEmpty(const Vector &vector) {
    //if (vector.size() == 0) {
    //    throw InvalidSizeException("Vector must not be empty.");
    //}
}

Vector operator-(const Vector &a, float scalar) {
    return Vector::mapper(a, [scalar](const Vector &a, int i) -> float { return a[i] - scalar; });
}

Matrix::Matrix() : Matrix(0, 0) {}

Matrix::Matrix(int rows, int cols) {
    _matrix = vector<Vector>(rows, Vector(cols));
}

Matrix::Matrix(const vector<vector<float>> &m) {
    vector<Vector> result;
    for (const vector<float> &row: m) {
        result.emplace_back(row);
    }
    _matrix = result;
}

Matrix::Matrix(const vector<Vector> &m) {
    _matrix = m;
}

void Matrix::checkIndexInRange(int index) const {
    //if (index < 0 || index >= rows()) {
    //    throw OutOfRangeException();
    //}
}

const Vector &Matrix::operator[](int index) const {
    checkIndexInRange(index);
    return _matrix[index];
}

Vector &Matrix::operator[](int index) {
    checkIndexInRange(index);
    return _matrix[index];
}

int Matrix::cols() const {
    return _matrix.empty() ? 0 : _matrix[0].size();
}

int Matrix::rows() const {
    return _matrix.size();
}

template<typename Lambda>
Matrix Matrix::mapper(const Matrix &a, Lambda f) {
    Matrix result(a.rows(), a.cols());
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < a.cols(); ++j) {
            result[i][j] = f(a, i, j);
        }
    }
    return result;
}

void Matrix::checkSameSize(const Matrix &a, const Matrix &b, const string &operation) {
    //if (a.rows() != b.rows() || a.cols() != b.cols()) {
    //    throw DifferentSizeException(operation);
    //}
}

Matrix operator+(const Matrix &a, const Matrix &b) {
    Matrix::checkSameSize(a, b, "+");
    return Matrix::mapper(a, [&b](const Matrix &a, int i, int j) -> float { return a[i][j] + b[i][j]; });
}

Matrix operator-(const Matrix &a, const Matrix &b) {
    Matrix::checkSameSize(a, b, "-");
    return Matrix::mapper(a, [&b](const Matrix &a, int i, int j) -> float { return a[i][j] - b[i][j]; });

}

Matrix operator*(const Matrix &a, const float &scalar) {
    return Matrix::mapper(a, [&scalar](const Matrix &a, int i, int j) -> float { return a[i][j] * scalar; });
}

Matrix operator*(const float &scalar, const Matrix &a) {
    return a * scalar;
}

Matrix operator*(const Matrix &a, const Matrix &b) {
    if (a.cols() != b.rows()) {
        throw IncompatibleSizesException(
                "Number of columns of the first matrix must be equal to the number of rows of the second matrix.");
    }

    Matrix result(a.rows(), b.cols());
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < b.cols(); ++j) {
            result[i][j] = a.row(i) * b.col(j);
        }
    }
    return result;
}

bool operator==(const Matrix &a, const Matrix &b) {
    return a._matrix == b._matrix;
}

bool operator==(const Vector &vec, const Matrix &mat) {
    if (mat.rows() == 1) {
        return vec == mat.row(0);
    }
    if (mat.cols() == 1) {
        return vec == mat.col(0);
    }
    return false;
}

bool operator==(const Matrix &mat, const Vector &vec) {
    return vec == mat;
}

bool operator!=(const Matrix &a, const Matrix &b) {
    return !(a == b);
}

bool operator!=(const Vector &vec, const Matrix &mat) {
    return !(vec == mat);
}

bool operator!=(const Matrix &mat, const Vector &vec) {
    return vec != mat;
}

Vector Matrix::row(int n) const {
    return (*this)[n];
}

Vector Matrix::col(int n) const {
    Vector result(rows());
    for (int i = 0; i < rows(); ++i) {
        result[i] = _matrix[i][n];
    }

    return result;
}

float Matrix::gauss() {
    float determinant = 1;

    for (int lead = 0; lead < rows(); ++lead) {
        /* calculate divisor */
        float d = _matrix[lead][lead];
        if (d == 0) {
            continue;
        }

        for (int r = 0; r < rows(); ++r) { // for each row ...
            /* calculate multiplier */
            float m = _matrix[r][lead] / _matrix[lead][lead];

            /* adjust determinant */
            if (r == lead) {
                determinant = determinant * d;
            }

            for (int c = 0; c < cols(); ++c) { // for each column ...
                if (r == lead) {
                    _matrix[r][c] = _matrix[r][c] / d;      // make pivot = 1
                } else {
                    _matrix[r][c] -= _matrix[lead][c] * m;  // make other = 0
                }
            }
        }
    }

    /* create new Matrix with zero rows at the bottom */
    Vector zero_vector(cols());
    vector<Vector> new_matrix(rows(), Vector(cols()));
    int i = 0;
    for (int r = 0; r < rows(); ++r) {
        if (_matrix[r] != zero_vector) {
            new_matrix[i] = _matrix[r];
            ++i;
        }
    }
    _matrix = new_matrix;

    return determinant;
}

int Matrix::rank() const {
    Matrix mat = *this;
    mat.gauss();
    int result = 0;
    for (int i = mat.rows() - 1; i >= 0; --i) {
        if (mat.row(i) != Vector(mat.cols())) {
            ++result;
        }
    }
    return result;
}

float Matrix::det() const {
    Matrix mat = *this;
    float determinant = mat.gauss();
    return mat.rank() == mat.rows() ? determinant : 0;
}

Matrix Matrix::inv() const {
    Matrix mat = *this;

    /* create augmented Matrix */
    int dimension = mat.rows();
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            mat[i]._vector.emplace_back(i == j);
        }
    }

    mat.gauss();

    /* read the inverse of Matrix */
    vector<Vector> result;
    for (int i = 0; i < dimension; ++i) {
        vector<float> vec;
        vec.reserve(dimension);
        for (int j = 0; j < dimension; ++j) {
            vec.push_back(mat[i][dimension + j]);
        }
        result.emplace_back(vec);
    }

    return Matrix(result);
}

ostream &operator<<(ostream &os, const Matrix &m) {
    bool firstRow = true;
    for (int r = 0; r < m.rows(); ++r) {
        if (!firstRow) {
            os << endl;
        }
        os << m[r];
        firstRow = false;
    }
    return os;
}

string Matrix::toString() const {
    ostringstream oss;
    oss << *this;
    return oss.str();
}

Vector operator*(const Matrix &mat, const Vector &vec) {
    if (mat.cols() != vec.size()) {
        throw IncompatibleSizesException("Number of columns of matrix must be equal to the length of the vector.");
    }

    Vector result(mat.rows());
    for (int i = 0; i < mat.rows(); ++i) {
        result[i] = vec * mat.row(i);
    }
    return result;
}

void Matrix::multiply(const Matrix &mat, const Vector &vec, Vector &result) {
    //if (mat.cols() != vec.size()) {
    //    throw IncompatibleSizesException("Number of columns of matrix must be equal to the length of the vector.");
    //}
    //if (result.size() != mat.rows()) {
    //    throw IncompatibleSizesException("Resulting vector's size must be equal to number of rows in matrix.");
    //}

    for (int i = 0; i < mat.rows(); ++i) {
        result[i] = vec * mat.row(i);
    }
}

Vector operator*(const Vector &vec, const Matrix &mat) {
    if (mat.rows() != vec.size()) {
        throw IncompatibleSizesException("Number of rows of matrix must be equal to the length of the vector.");
    }

    Vector result(mat.cols());
    for (int i = 0; i < mat.cols(); ++i) {
        result[i] = vec * mat.col(i);
    }
    return result;
}

void Matrix::checkNotEmpty(const Matrix &mat) {
    //if (mat.rows() == 0) {
    //    throw InvalidSizeException("Number of rows of matrix must not be zero");
    //}
}
