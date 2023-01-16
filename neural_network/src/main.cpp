#include <csv_reader.h>
#include <csv_writer.h>
#include <model.h>
#include <linalg.h>
#include <logger.h>

using namespace std;
static const char *const HEADER = "\nNeural Network PRO © 2020 Petr Janik and Otakar Hirs.\n\n";
static const char *const USAGE = "Usage:\n\tneural_network <train_vectors> <train_labels> <test_vectors> <output_file>\n\n"
                                 "Description:\n\t"
                                 "Trains a model using train_vectors and train_labels.\n\t"
                                 "Makes a prediction on test_vectors and writes the results to output_file.\n";

int main(int argc, const char *argv[]) {
    std::cout << HEADER;

    // ensure the correct number of parameters are used.
    if (argc != 5) {
        std::cout << USAGE;
        return 1;
    }

    const string trainVectorsFileName = argv[1];
    const string trainLabelsFileName = argv[2];
    const string testVectorsFileName = argv[3];
    const string outputFile = argv[4];

    Logger::info(
            "neural_network " + trainVectorsFileName + " " + trainLabelsFileName + " " + testVectorsFileName + " " +
            outputFile);

    auto xTrainData = CsvReader::readFileToMatrix(trainVectorsFileName);
    auto yTrainData = CsvReader::readFileToVector(trainLabelsFileName);

    std::shuffle(xTrainData.begin(), xTrainData.end(), std::default_random_engine(0));
    std::shuffle(yTrainData.begin(), yTrainData.end(), std::default_random_engine(0));

    // allocate 1/5 of the training data for validation
    std::size_t const split = yTrainData.size() / 5 * 4;
    Matrix xTrain = Matrix(vector<vector<float>>(xTrainData.begin(), xTrainData.begin() + split));
    Matrix xVal = Matrix(vector<vector<float>>(xTrainData.begin() + split, xTrainData.end()));
    Vector yTrain = Vector(vector<float>(yTrainData.begin(), yTrainData.begin() + split));
    Vector yVal = Vector(vector<float>(yTrainData.begin() + split, yTrainData.end()));

    const Matrix xTest = Matrix(CsvReader::readFileToMatrix(testVectorsFileName));

    //test xor
    //const Matrix xTrain = Matrix({Vector({1, 1}),
    //                              Vector({0, 0}),
    //                              Vector({1, 0}),
    //                              Vector({0, 1})});
    //const Vector yTrain = Vector({0, 0, 1, 1});
    //const Matrix xVal = xTrain;
    //const Vector yVal = yTrain;

    Model model({
        LayerConfig(128, 0.01, Layer::getWeightHe),
        LayerConfig(10, 0, Layer::getWeightXavier)
    });

    // Validation data are used only for logging current progress, they are not used for learning
    model.fit(xTrain, yTrain, xVal, yVal, 13, 16, 0.05, 0.9);

    const Matrix yPredicted = model.predict(xTest);
    Vector yPredictedLabels = Vector(yPredicted.rows());
    for(int i = 0; i < yPredictedLabels.size(); i++) {
        yPredictedLabels[i] = yPredicted[i].argmax();
    }

    CsvWriter::writeVectorToFile(yPredictedLabels, outputFile);
    Logger::info("Predictions have been written to " + outputFile);

    return 0;
}
