#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <random>
#include <chrono>

std::vector<std::string> split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

void loadData(const std::string &filename, std::vector<std::vector<float>> &inputs, std::vector<float> &targets, int batchSize) {
    std::ifstream file(filename);
    std::string line;
    
    std::getline(file, line);
    
    size_t batchCount = 0;
    size_t maxBatches = 5000 / batchSize;
    
    while (std::getline(file, line)) {
        std::vector<std::string> row = split(line, ',');
        if (row.size() != 6) continue;
        
        std::vector<float> sample;
        sample.push_back(std::stof(row[0])); // Hours Studied
        sample.push_back(std::stof(row[1])); // Previous Scores
        sample.push_back(row[2] == "Yes" ? 1.0f : 0.0f); // Extracurr Activities
        sample.push_back(std::stof(row[3])); // Sleep Hours
        sample.push_back(std::stof(row[4])); // Sample Papers Practiced
        
        inputs.push_back(sample);
        targets.push_back(std::stof(row[5])); // Performance Index
    }
    
    // Normalize inputs
    std::vector<float> min_features(5, 0.0f);
    std::vector<float> max_features(5, 0.0f);
    
    for (size_t i = 0; i < 5; ++i) {
        float min_val = 1e30f;
        float max_val = -1e30f;
        for (const auto &sample : inputs) {
            min_val = std::min(min_val, sample[i]);
            max_val = std::max(max_val, sample[i]);
        }
        min_features[i] = min_val;
        max_features[i] = max_val;
    }
    
    for (auto &sample : inputs) {
        for (size_t i = 0; i < 5; ++i) {
            if (max_features[i] != min_features[i])
                sample[i] = (sample[i] - min_features[i]) / (max_features[i] - min_features[i]);
            else
                sample[i] = 0.0f;
        }
    }
    
    size_t totalSamples = inputs.size();
    size_t newNumSamples = (totalSamples / batchSize) * batchSize;
    inputs.resize(newNumSamples);
    targets.resize(newNumSamples);
}

// Split data into batches
void createBatches(const std::vector<std::vector<float>> &inputs, const std::vector<float> &targets, 
                   std::vector<std::vector<float>> &inputBatches, std::vector<std::vector<float>> &targetBatches, int batchSize) {
    size_t numBatches = inputs.size() / batchSize;
    
    for (size_t i = 0; i < numBatches; ++i) {
        size_t start = i * batchSize;
        size_t end = start + batchSize;
        
        std::vector<float> inputBatch(batchSize * 5);
        std::vector<float> targetBatch(batchSize);
        
        for (size_t j = 0; j < batchSize; ++j) {
            std::copy(inputs[start + j].begin(), inputs[start + j].end(), inputBatch.begin() + j * 5);
            targetBatch[j] = targets[start + j];
        }
        
        inputBatches.push_back(inputBatch);
        targetBatches.push_back(targetBatch);
    }
}