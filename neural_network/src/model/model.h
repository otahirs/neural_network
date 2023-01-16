#ifndef NEURAL_NETWORK_MODEL_H
#define NEURAL_NETWORK_MODEL_H

#include <vector>
#include <linalg.h>
#include <functional>
#include <random>

using namespace std;

class Layer;

class Model;

typedef function<float(int)> weightInitFunctionType;
typedef function<float(float, const Vector &)> activationFunctionType;

class IllegalArgumentException : runtime_error {
public:
    explicit IllegalArgumentException(const string &msg) : runtime_error(msg) {}
};

/**
 * A configuration class
 */
class LayerConfig {
    friend Layer;
    friend Model;

    int _neuronsCount;
    float _biasInitValue;
    const weightInitFunctionType &_weightInitFunction;
public:
    class WrongLayerConfigException : runtime_error {
    public:
        explicit WrongLayerConfigException(const string &msg) : runtime_error(msg) {}
    };

    explicit LayerConfig(int neuronsCount, float biasInitValue, const weightInitFunctionType &weightInitFunction);
};

/**
 * A Layer class that holds the logic
 */
class Layer {

    LayerConfig _layerConfig;

    inline static mt19937 gen = mt19937{random_device{}()};
    inline static normal_distribution<float> standardNormalDistribution = normal_distribution<float>(0, 1);

    static float randn();

public:

    Matrix _weights;
    Vector _biases;
    Vector _values;
    class WrongInputNeuronsCountException : runtime_error {
    public:
        explicit WrongInputNeuronsCountException(const string &msg) : runtime_error(msg) {}
    };


    Layer(int inputNeuronsCount, LayerConfig layerConfig);

    // Weight initialization functions
    /**
     * used for ReLU
     */
    static float getWeightHe(int inputCount);

    /**
     * used for tanh()
     */
    static float getWeightXavier(int inputCount);

    // Activation functions
    static float unitStepFunction(const float &x, const Vector &_vec = Vector());

    static float ReLU(const float &x, const Vector &_vec = Vector());

    static float sigmoid(const float &x, const Vector &_vec = Vector());

    static float tanh(const float &x, const Vector &_vec = Vector());

    static float SoftMax(const float &x, const Vector &vec);

    int getNeuronsCount() const;

    void setBiases(Vector biases);

    void setWeights(Matrix weights);
};

class Model {
    vector<LayerConfig> _layerConfigs;
    vector<Layer> _layers;

    void initializeLayers(int inputNeuronsCount);

public:
    explicit Model(const vector<LayerConfig> &configLayers);

    /**
     * @param x - Input data
     * @param y - Target data
     * @param epochs - Number of epochs to train the model.
     *                 An epoch is an iteration over the entire x and y data provided.
     * @param batchSize - Number of samples per gradient update.
     */
    void fit(Matrix &xTrain, Vector &yTrain, const Matrix &xTest, const Vector &yTest, int epochs, int batchSize, float learningRate, float beta);

    Matrix predict(const Matrix &x);


    static float multipleOutputsError(const Vector &expected, const Vector &predicted);

    static float meanSquaredError(const Matrix &expected, const Matrix &predicted);

    /**
     * @param expected Matrix where each row is a sample and each column is expected output of one neuron
     *        usually all output neurons are 0 apart from one which is 1
     * @param predicted - Matrix where each row is a sample and each column is output of one neuron
     *        output of a neuron must be in range (0,1), i.e. SoftMax must be applied before crossEntropyError
     * @return cross entropy error value
     */

    float SoftMax(const float &x, const Vector &vec);

    float ReLU(const float &x);

    float crossEntropyError(const Vector &expected, const Matrix &predicted);

    float accuracy(const Vector &expected, const Matrix &predicted);

    float SoftMax(const float &x, float &vecMax, float &denominator);
};

#endif //NEURAL_NETWORK_MODEL_H
