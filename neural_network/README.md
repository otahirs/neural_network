### Implement
- basic Matrix class
    - +, -, *, /, matmul, random initialization, softmax, relu, sigmoid,...
- CSVReader/Writer
- Trainer wrapper to store all the matrices in
- output statistics / logging
- Should work:
    - initialize random Matrices => CSVRead  => Forwardpass (random projection) => cout << stats => CSVWrite
        - make sure your stats are true, the outputs are correct, etc.
- backpropagation
    - might take 90 % of the implementation time (yet just 15 % of the code), 
    but it should be feasible as you will already have the whole pipeline set up == you will see the correct stats for nice debugging
- neural network tools (maybe)
    - layers with activation functions
    - automatic differentiation
    - equation/linear-program solvers

### Task
- implement a feed-forward neural network and train it on a given dataset using a backpropagation algorithm
- Fashion MNIST dataset (Zalando's article images)
    - training set of 60,000 examples
    - test set of 10,000 examples
    - each example is a 28x28 grayscale image, associated with a label from 10 classes
    - CSV format
    - four data files
        - two data files as input vectors
        - two data files with a list of expected predictions
- deadline on January 6th, 2021 23:59
- rules:
    - must be compilable and runnable on the Aisa server
    - must contain a runnable script called "RUN" which compiles,
      executes and exports everything in "one-click"
    - must export vectors of train and test predictions
    - exported predictions must be in the same format as is
      "actualPredictionsExample" file ‒ on each line is only one float present
    - such float on i-th line represents predicted class index (there are classes
      0 - 9 for Fashion MNIST) for i-th input vector => prediction order is relevant
    - name the exported files "trainPredictions" and "actualTestPredictions"
    - the implementation will take both train and test input vectors, but it must
      not touch test data except the evaluation of the already trained model
    - write doc-strigs where reasonable (high-level functions, complicated functions with unusual names)
        - not judged, only advised
    - reach at least 88% of correct test predictions (overall accuracy)
    - at most half an hour of training time on the Aisa server
    - implementations will be executed for a little longer (35 minutes)
        - load the data
        - process them
        - train the model
        - export train/test predictions to files
    - correctness will be checked using an independent program, which is provided for your own purposes
    - don't use high-level libraries
    - can use basic math functions like exp, sqrt, log, rand, etc.
    - what you do internally with the training dataset is up to you
    - Pack all data with your implementations and put them on the right path so
      your program will load them correctly on AISA (project dir is fine).
    - team of two
    - don't post your code openly on git[hub|lab], don't read solutions already there
    - network must be trained completely from random initialization - you can't use pre-trained models (weights) and just load them
    - update the main README with the "UČO" of both teammates working on the project

### Tips
- solve the XOR problem first
    - XOR is a very nice example as a benchmark of the working learning process with at least one hidden layer
    - the presented network solving XOR in the lecture is minimal and it can be hard to find, so consider more neurons in the hidden layer
    - if you can't solve the XOR problem, you can't solve Fashion MNIST.
- your code might be tested on other datasets of the same format to see whether it works reasonably well (MNIST, augmented FashionMNIST, etc.)
- reuse memory
    - you are implementing an iterative process, so don't allocate new vectors and matrices all the time
    - an immutable approach is nice but very inappropriate
    - don't copy data in some cache all the time; use indices
- objects are fine, but be careful about the depth of object hierarchy you are
  going to create
- use floats
- simple SGD is not strong and fast enough
    - you need to implement some heuristics as well (or maybe not, but it's highly recommended)
    - suggested heuristics: momentum, weight decay, dropout. If you are brave enough, you can try RMSProp/AdaGrad/Adam.
    - start with smaller networks and increase network topology carefully
    - consider validation of the model using part of the train dataset
    - play with hyperparameters to increase your internal validation accuracy
- Aisa has 4x16 cores, OpenMP or similar easy parallelism may help
- execute your RUN script on AISA before your submission
- if your implementation requires modules, your RUN script must include them as well
- do not shuffle testing data - it won't fit expected predictions
- usual implementations need up to a few days of fine-tuning hyperparameters (using validation set, not test set!) to achieve their max potential 

### Useful links
- [How to use modules on Aisa](https://www.fi.muni.cz/tech/unix/modules.html.en)
- [About Fashion MNIST dataset](https://arxiv.org/pdf/1708.07747.pdf) 
- [Tips for implementation](https://www.youtube.com/watch?v=fm8Scoih3nc&feature=youtu.be&ab_channel=RonaldLuc)
    - "hacks" for efficient single-core and inefficient (but easy to implement) multi-core matrix & gradient computation
- Project structure by https://github.com/kigster/cmake-project-template
- This [Google Colab](https://colab.research.google.com/drive/1cvAvtd6O2c3SHLpQLLOGfx8kgjxHphrT?usp=sharing) should help you know, what accuracies to expect from different architectures.

