#include "read_and_process_csv.h"
#include "neuralnetwork.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <matplotlibcpp.h>  

namespace plt = matplotlibcpp;

int main() {
    // const int batchSize = 512;
    const int batchSize = 2;
    std::vector<std::vector<float>> inputs;
    std::vector<float> targets;
    
    loadData("Student_Performance.csv", inputs, targets, batchSize);
    std::vector<std::vector<float>> inputBatches, targetBatches;
    createBatches(inputs, targets, inputBatches, targetBatches, batchSize);
    
    NeuralNetwork network({5}, {1}, 0.01f, batchSize);
    // NeuralNetwork network({5, 3, 2}, {3, 2, 1}, 0.01f, batchSize);
    
    int num_epochs = 5;
    float decay = 0.9;
    float loss;
    std::vector<float> history;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "=== Epoch " << (epoch) << " ===" << std::endl;
        float tot_loss = 0;
        for (size_t batch = 0; batch < inputBatches.size(); ++batch) {
            std::cout << "=== Batch " << (batch) << " ===" << std::endl;
            loss = network.forwardAndBackwardPass(inputBatches[batch].data(), targetBatches[batch].data());
            tot_loss += loss;
        }
        history.push_back(tot_loss);
        network.learningRate *= decay;
    }
    
    std::vector<int> epochs(num_epochs);
    std::iota(epochs.begin(), epochs.end(), 0);
    
    plt::plot(epochs, history);
    plt::xlabel("Epochs");
    plt::ylabel("Loss");
    plt::title("Loss Over Epochs");
    plt::show();  
    return 0;
}