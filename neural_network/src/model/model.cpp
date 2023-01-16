#include "model.h"
#include <logger.h>
#include <functional>
#include <omp.h>

using namespace std;

LayerConfig::LayerConfig(int neuronsCount, float biasInitValue,
                         const weightInitFunctionType &weightInitFunction) :
        _biasInitValue(biasInitValue),
        _weightInitFunction(weightInitFunction) {
    if (neuronsCount < 1) {
        throw WrongLayerConfigException("Neurons count must larger or equal to one.");
    }
    _neuronsCount = neuronsCount;
}

Layer::Layer(int inputNeuronsCount, LayerConfig layerConfig) :
        _layerConfig(layerConfig),
        _biases(Vector(layerConfig._neuronsCount)),
        _values(Vector(layerConfig._neuronsCount)) {
    if (inputNeuronsCount < 1) {
        throw WrongInputNeuronsCountException("Input neurons count must larger or equal to one.");
    }
    _weights = Matrix(layerConfig._neuronsCount, inputNeuronsCount);
}

float Layer::randn() { return standardNormalDistribution(gen); }

float Layer::getWeightHe(int inputsCount) {
    return randn() * static_cast<float>(sqrt(2.0 / inputsCount));
}

float Layer::getWeightXavier(int inputsCount) {
    return randn() * static_cast<float>(sqrt(1.0 / inputsCount));
}

float Layer::unitStepFunction(const float &x, const Vector &_vec /* = Vector() */) {
    return (x >= 0) ? 1 : 0;
}

float Layer::ReLU(const float &x, const Vector &_vec /* = Vector() */) {
    return (x < 0) ? 0 : x;
}
float Model::ReLU(const float &x) {
    return (x < 0) ? 0 : x;
}

float Layer::sigmoid(const float &x, const Vector &_vec /* = Vector() */) {
    return 1 / (1 + exp(-x));
}

float Layer::tanh(const float &x, const Vector &_vec /* = Vector() */) {
    return (2 / (1 + exp(-2 * x))) - 1;
}

float Model::SoftMax(const float &x, float &vecMax, float &denominator) {
    // This approach does not overflow, unlike "return exp(x) / vec.exp().sum();"
    // see https://stackoverflow.com/questions/42599498/numercially-stable-softmax

    float numerator = exp((x - vecMax));
    return numerator/denominator;
}

int Layer::getNeuronsCount() const {
    return _weights.rows();
}

void Layer::setWeights(Matrix weights) {
    _weights = weights;
}

void Layer::setBiases(Vector biases) {
    _biases = biases;
}


void Model::fit(Matrix &xTrain, Vector &yTrain, const Matrix &xVal, const Vector &yVal, int epochs, int batchSize, float learningRate, float beta) {
    Logger::info("batchsize: " + to_string(batchSize));
    Logger::info("learningRate: " + to_string(learningRate));
    Logger::info("beta: " + to_string(beta));

    initializeLayers(xTrain.cols());

    struct batch {
        Vector outputs;
        Vector derivatives;
    };

    struct layer {
        int neuronCount;
        bool is_first;
        bool is_last;
        Matrix weights;
        Matrix wMomentum;
        Vector biases;
        Vector bMomentum;
        vector<struct batch> batch;
    };

    // init memory
    int lastLayerIndex = _layerConfigs.size() - 1;
    vector<layer> layers (_layerConfigs.size());
    for(int l = 0; l < _layerConfigs.size(); l++) {
        layers[l].neuronCount = _layers[l].getNeuronsCount();
        if (l == 0) {
            layers[l].is_first = true;
            layers[l].weights = Matrix(_layers[l].getNeuronsCount(), xTrain.cols());
            layers[l].wMomentum = Matrix(_layers[l].getNeuronsCount(), xTrain.cols());
        }
        else {
            layers[l].is_first = false;
            layers[l].weights = Matrix(_layers[l].getNeuronsCount(), _layers[l - 1].getNeuronsCount());
            layers[l].wMomentum = Matrix(_layers[l].getNeuronsCount(), _layers[l - 1].getNeuronsCount());

        }
        layers[l].biases = Vector(_layers[l].getNeuronsCount());
        layers[l].bMomentum = Vector(_layers[l].getNeuronsCount());
        layers[l].batch = vector<batch>(batchSize);
        if (l == lastLayerIndex)
            layers[l].is_last = true;
        else
            layers[l].is_last = false;

    }

    for(int b=0; b < batchSize; b++)
    {
        for(int l = 0; l < _layerConfigs.size(); l++) {
            layers[l].batch[b].outputs = Vector(_layers[l].getNeuronsCount());
            layers[l].batch[b].derivatives = Vector(_layers[l].getNeuronsCount());
        }
    }

    for(int l = 0; l < layers.size(); l++) {
        // init weights
        int neuronCount = layers[l].is_first ? xTrain.cols() : _layers[l-1].getNeuronsCount();
        for (Vector &row: layers[l].weights) {
            for (float &num: row) {
                num = _layerConfigs[l]._weightInitFunction(neuronCount);
            }
        }
        // init biases
        for (float &bias: layers[l].biases) {
            bias = _layerConfigs[l]._biasInitValue;
        }
    }
    // norm input
    for (Vector &row: xTrain) {
        for (float &num: row) {
            num /= 255;
        }
    }

    // train
    //Logger::info("Init complete");

    for (int e = 0; e < epochs; e++) {
        Logger::info("---");
        Logger::info("Starting Epoch " + to_string(e + 1) + "/" + to_string(epochs));
        float loss = 0;

        std::shuffle(xTrain.begin(), xTrain.end(), std::default_random_engine(e));
        std::shuffle(yTrain.begin(), yTrain.end(), std::default_random_engine(e));

        for (int batchStartIndex = 0; batchStartIndex + batchSize <= xTrain.rows(); batchStartIndex += batchSize) {
#pragma omp parallel for num_threads(16)
            for (int b = 0; b < batchSize; b++) {
                for (int l = 0; l < _layers.size(); l++) {
                    struct layer &layer = layers[l];
                    Vector inputNeurons = layer.is_first ? xTrain[batchStartIndex + b] : layers[l - 1].batch[b].outputs;
                    // compute inner potential
                    Matrix::multiply(layer.weights, inputNeurons, layer.batch[b].outputs);
                    layer.batch[b].outputs.add(layer.biases); // Vector::add(layer.biases, layer.batch[b].outputs)
                    // apply activation function
                    if (!layer.is_last) {
                        for (float &value : layer.batch[b].outputs) {
                            value = ReLU(value);
                        }
                    }
                    else {
                        Vector &vec = layer.batch[b].outputs;
                        float vecMax = vec.max();
                        float denominator = (vec - vecMax).exp().sum();
                        for (float &value : layer.batch[b].outputs) {
                            value = SoftMax(value, vecMax, denominator);
                        }
                    }

                }

            }
            // calculate error
            Vector logprobs = Vector(batchSize);
            for (int b = 0; b < batchSize; b++) {
                for( int c = 0; c < 10; c++) { // For each class
                    if(yTrain[batchStartIndex + b] == c) {
                        logprobs[b] += -log(layers[lastLayerIndex].batch[b].outputs[c]);
                    }
                }
            }
            loss +=  logprobs.sum();
            // backpropagation
            for (int l = lastLayerIndex; l >= 0; l--) {
                layer & layer = layers[l];

                if (layer.is_last) {
                    // derivatives output layer
#pragma omp parallel for num_threads(16)
                    for (int b = 0; b < batchSize; b++) {
                        batch &batch = layer.batch[b];
                        // get vector of expected output
                        Vector expected_outputs = Vector(layer.neuronCount);
                        for (int i = 0; i < expected_outputs.size(); i++) {
                            if (i == yTrain[batchStartIndex + b]) {
                                expected_outputs[i] = 1;
                            }
                            else {
                                expected_outputs[i] = 0;
                            }
                        }
                        // cross entropy softmax derivation => output - expected output // (expected output = 0 or 1)
                        //Vector::subtract(batch.outputs, expected_outputs, batch.derivatives)
                        batch.derivatives = batch.outputs - expected_outputs; // subtract as vectors
                    }
                }
                else {
                    // derivatives hidden layer
#pragma omp parallel for num_threads(16)
                    for (int b = 0; b < batchSize; b++) {
                        struct layer &nextLayer = layers[l + 1];
                        // batch.derivatives[x] =  sum(last_layer_der * weight) * relu_der
                        // ReLU_der => 0 if input <= 0, else 1 // relu input = 0 if output == 0
                        batch &batch = layer.batch[b];
                        batch.derivatives = nextLayer.batch[b].derivatives * nextLayer.weights;
                        for (int i = 0; i < layer.neuronCount; i++) {
                            if (batch.outputs[i] <= 0)
                                batch.derivatives[i] = 0;
                        }

                    }

                }
                // update biases
                // bias der = sum(next_layer_der * weight) * relu_der * 1
                for (int i = 0; i < layer.neuronCount; i++) {
                    float gradientSum = 0;
                    for (batch &b: layer.batch) {
                        // derivative of output function * 1
                        gradientSum += b.derivatives[i]; // sum gradients of each batch into one sum for one neuron
                    }
                    gradientSum /= batchSize;
                    layer.bMomentum[i] = beta * layer.bMomentum[i] + (1-beta) * gradientSum;
                    layer.biases[i] -= learningRate * layer.bMomentum[i];
                }

                // update weights
                // weight der = sum(next_layer_der * weight) * relu_der * input
                for (int i = 0; i < layer.weights.rows(); i++) { // 10
                    for (int j = 0; j < layer.weights.cols(); j++) { // 256
                        float gradientSum = 0;
                        for (int b = 0; b < batchSize; b++) {
                            Vector &inputNeurons = layer.is_first ? xTrain[batchStartIndex + b] : layers[l - 1].batch[b].outputs;
                            // derivative of output function * output of neuron from last layer
                            gradientSum += layer.batch[b].derivatives[i] * inputNeurons[j]; // sum gradients of batch for one neuron
                        }
                        gradientSum /= batchSize;
                        layer.wMomentum[i][j] = beta * layer.wMomentum[i][j] + (1-beta) * gradientSum;
                        layer.weights[i][j] -= learningRate * layer.wMomentum[i][j];
                    }
                }
            }
        }
        //Logger::info("innerPo:" + layers[lastLayerIndex].batch[0].innerPotentials.toString());
        //Logger::info("outputs:" + layers[lastLayerIndex].batch[0].outputs.toString());
        // save trained stuff
        for (int i = 0; i < _layerConfigs.size(); i++) {
            _layers[i].setWeights(layers[i].weights);
            _layers[i].setBiases(layers[i].biases);
        }
        const Matrix yPredicted = predict(xVal);
        float accuracy = Model::accuracy(yVal, yPredicted);

        Logger::info("Loss: " + to_string(loss/xTrain.rows()));
        Logger::info("Validation data accu: " + to_string(accuracy));
        Logger::info("Validation data loss: " + to_string(crossEntropyError(yVal, yPredicted)));
    }
    Logger::info("Fitting model has ended.");
}

void Model::initializeLayers(int inputNeuronsCount) {
    for (int i = 0; i < _layerConfigs.size(); ++i) {
        int previousLayerNeuronsCount = (i == 0) ? inputNeuronsCount : _layerConfigs[i - 1]._neuronsCount;
        _layers.emplace_back(previousLayerNeuronsCount, _layerConfigs[i]);
    }
}

Matrix Model::predict(const Matrix &x) {
    //Logger::info("Predicting values:");
    Matrix data = x;
    for (Vector &row: data) {
        for (float &num: row) {
            num /= 255;
        }
    }
    Matrix predictions(x.rows(), 10);
    for (int i = 0; i < x.rows(); ++i) {
        //Logger::info("Predicting for " + x[i].toString());

        Vector inputNeurons = data[i];
        for (int j = 0; j < _layers.size(); ++j) {
            Matrix::multiply(_layers[j]._weights, inputNeurons, _layers[j]._values);
            _layers[j]._values.add(_layers[j]._biases);
            if (_layers.size() - 1 == j) {
                Vector &vec = _layers[j]._values;
                float vecMax = vec.max();
                float denominator = (vec - vecMax).exp().sum();
                for (float &value : _layers[j]._values) {
                    value = SoftMax(value, vecMax, denominator);
                }
            }
            else {
                for (float &value : _layers[j]._values) {
                    value = ReLU(value);
                }
            }
            inputNeurons =  _layers[j]._values;
        }

        predictions[i] = inputNeurons;
    }
    //Logger::info("Predicting done.");

    return predictions;
}

Model::Model(const vector<LayerConfig> &configLayers) : _layerConfigs(configLayers) {}

float Model::accuracy(const Vector &expected, const Matrix &predicted) {

    int correct = 0;
    int total = predicted.rows();
    for (int i = 0; i < predicted.rows(); ++i) {
        if (predicted[i].argmax() == expected[i]) {
            ++correct;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(total);
}

float Model::multipleOutputsError(const Vector &expected, const Vector &predicted) {
    Vector::checkSameSize(predicted, expected, "multiple outputs error");
    Vector::checkNotEmpty(expected);

    float sum = 0;
    for (int i = 0; i < expected.size(); ++i) {
        sum += pow(expected[i] - predicted[i], 2);
    }

    return sum / 2;
}

float Model::meanSquaredError(const Matrix &expected, const Matrix &predicted) {
    Matrix::checkSameSize(predicted, expected, "mean squared error");
    Matrix::checkNotEmpty(expected);

    float sum = 0;
    for (int i = 0; i < expected.rows(); ++i) {
        sum += Model::multipleOutputsError(expected[i], predicted[i]);
    }

    return sum / static_cast<float>(expected.rows());
}

float Model::crossEntropyError(const Vector &expected, const Matrix &predicted) {

    float sum = 0;
    for (int i = 0; i < predicted.rows(); ++i) {
        for (int j = 0; j < predicted.cols(); ++j) {
            if (j == expected[i]) {
                sum -= log(predicted[i][j]);
            }
        }
    }
    return  (sum / static_cast<float>(expected.size()));
}