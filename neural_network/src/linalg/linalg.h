#ifndef NEURAL_NETWORK_LINALG_H
#define NEURAL_NETWORK_LINALG_H

#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

class OutOfRangeException : runtime_error {
public:
    explicit OutOfRangeException() : runtime_error("Index is out of range.") {}
};

class IncompatibleSizesException : runtime_error {
public:
    explicit IncompatibleSizesException(const string &msg) : runtime_error(msg) {}
};

class InvalidSizeException : runtime_error {
public:
    explicit InvalidSizeException(const string &msg) : runtime_error(msg) {}
};

class Matrix;

class Vector {
    friend class Matrix;

    vector<float> _vector;

    template<typename Lambda>
    static Vector mapper(const Vector &a, Lambda f);

    template<typename Lambda>
    void mapperInPlace(Lambda f);


    void checkIndexInRange(int index) const;

public:
    /* For tests */
    typedef float value_type;
    typedef vector<float>::const_iterator const_iterator;

    /* -------- */

    class DifferentSizeException : runtime_error {
    public:
        explicit DifferentSizeException(const string &operation) : runtime_error(
                "The operation '" + operation + "' must be performed only on vectors of the same size.") {}
    };

    Vector();

    /**
     * @param dimension - must be positive
     */
    explicit Vector(int dimension);

    explicit Vector(const vector<float> &v);

    const float &operator[](int index) const;

    float &operator[](int index);

    int size() const;

    static void checkSameSize(const Vector &a, const Vector &b, const string &operation);

    static void checkNotEmpty(const Vector &vector);

    auto begin() const {
        return _vector.begin();
    }

    auto end() const {
        return _vector.end();
    }

    auto begin() {
        return _vector.begin();
    }

    auto end() {
        return _vector.end();
    }

    /**
     * Both vectors a and b have the same dimension
     */
    friend Vector operator+(const Vector &a, const Vector &b);

    /**
     * MODIFIES the Vector it is called on by adding values of elements in Vector b
     * @param b Vector to be added
     * @return reference to the modified Vector it has been called on
     */
    Vector &add(const Vector &b);

    friend Vector operator-(const Vector &a, const Vector &b);

    friend Vector operator-(const Vector &a, float scalar);

    friend Vector operator*(const Vector &a, const float &scalar);

    friend Vector operator*(const float &scalar, const Vector &a);

    friend float operator*(const Vector &a, const Vector &b);

    friend bool operator==(const Vector &a, const Vector &b);

    friend bool operator!=(const Vector &a, const Vector &b);

    friend ostream &operator<<(ostream &os, const Vector &vector);

    string toString() const;

    /**
     * @return max element in vector, 0 if vector is empty
     */
    float max() const;

    /**
     * @return index of max element in vector, -1 if vector is empty
     */
    int argmax() const;

    /**
     * @return sum of elements in vector, 0 if vector is empty
     */
    float sum() const;

    /**
     * @return new Vector containing exponents of each element in the Vector it has been called on
     */
    Vector exp() const;
};

class Matrix {

    friend class Vector;

    vector<Vector> _matrix;

    template<typename Lambda>
    static Matrix mapper(const Matrix &a, Lambda f);

    void checkIndexInRange(int index) const;

public:

    class DifferentSizeException : runtime_error {
    public:
        explicit DifferentSizeException(const string &operation) : runtime_error(
                "The operation '" + operation + "' must be performed only on matrices of the same size.") {}
    };

    Matrix();

    /**
     * @param cols - must be positive
     * @param rows - must be positive
     */
    Matrix(int rows, int cols);

    explicit Matrix(const vector<vector<float>> &m);

    explicit Matrix(const vector<Vector> &m);

    const Vector &operator[](int index) const;

    Vector &operator[](int index);

    int cols() const;

    int rows() const;

    auto begin() const {
        return _matrix.begin();
    }

    auto end() const {
        return _matrix.end();
    }

    auto begin() {
        return _matrix.begin();
    }

    auto end() {
        return _matrix.end();
    }

    friend Matrix operator+(const Matrix &a, const Matrix &b);

    friend Matrix operator-(const Matrix &a, const Matrix &b);

    friend Matrix operator*(const Matrix &a, const float &scalar);

    friend Matrix operator*(const float &scalar, const Matrix &a);

    /**
     * @param mat - m row, n columns
     * @param mat - n rows, o columns
     * @return mat m x o
     */
    friend Matrix operator*(const Matrix &a, const Matrix &b);

    /**
     * @param mat - m rows, n columns
     * @param vec - n rows, 1 column
     * @return vec m x 1
     */
    friend Vector operator*(const Matrix &mat, const Vector &vec);

    /**
     * Multiply mat by vec and store multiplication result in result
     * @param mat - m rows, n columns
     * @param vec - n rows, 1 column
     * @param result - Vector m x 1
     */
    static void multiply(const Matrix &mat, const Vector &vec, Vector &result);

    /**
     * @param vec - 1 row, m columns
     * @param mat - m rows, n columns
     * @return vec 1 x n
     */
    friend Vector operator*(const Vector &vec, const Matrix &mat);

    friend bool operator==(const Matrix &a, const Matrix &b);

    friend bool operator==(const Vector &vec, const Matrix &mat);

    friend bool operator==(const Matrix &mat, const Vector &vec);

    friend bool operator!=(const Matrix &a, const Matrix &b);

    friend bool operator!=(const Vector &vec, const Matrix &mat);

    friend bool operator!=(const Matrix &mat, const Vector &vec);

    Vector row(int n) const;

    Vector col(int n) const;

    /**
     * After this operation, the Matrix adheres to these rules:
     *  - The first non-zero element in each row, called the leading coefficient, is 1.
     *  - Each leading coefficient is in a column to the right of the previous row leading coefficient.
     *  - Rows with all zeros are below rows with at least one non-zero element.
     *  - The leading coefficient in each row is the only non-zero entry in its column.
     *
     *  @return determinant of Matrix, does not take into account null rows
     */
    float gauss();

    /**
     * Number of non-zero rows after Gauss-Jordan elimination
     */
    int rank() const;

    /**
     * Undefined when called on a non-square Matrix.
     *
     * @return determinant of Matrix which is zero,
     * if there is any dependent column
     */
    float det() const;

    /**
     * Undefined when called on a non-square Matrix.
     * Undefined when called on a singular Matrix.
     */
    Matrix inv() const;

    friend ostream &operator<<(ostream &os, const Matrix &m);

    string toString() const;

    static void checkSameSize(const Matrix &a, const Matrix &b, const string &operation);

    static void checkNotEmpty(const Matrix &matrix);
};

#endif //NEURAL_NETWORK_LINALG_H
