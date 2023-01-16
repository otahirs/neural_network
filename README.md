## Fashion MNIST 
neural network written from scratch in C++ as part of the [PV021 Neural Networks](https://is.muni.cz/course/fi/PV021) course

### Authors
- Otakar Hirš 485661
- Petr Janík 485122

### How to run
explore and use the `RUN` script.

### Building Manually
```bash
cd neural_network
rm -rf build && mkdir build
cd build
cmake ..
make && make install
cd ..
```

### Running
```bash
cd ..
./neural_network/bin/neural_network data/fashion_mnist_train_vectors.csv data/fashion_mnist_train_labels.csv data/fashion_mnist_test_vectors.csv actualPredictions
```

## Known issues
Running target `./neural_network/bin/all_tests` will not work, 
because some tests depend on external files and the path to these files
is wrong when run this way.
The tests expect working directory to be `neural_network\test`.
