#ifndef CSV_READER_H
#define CSV_READER_H

#include <vector>
#include <string>

void loadData(const std::string &filename, std::vector<std::vector<float>> &inputs, std::vector<float> &targets, int batchSize);
void createBatches(const std::vector<std::vector<float>> &inputs, const std::vector<float> &targets, 
                   std::vector<std::vector<float>> &inputBatches, std::vector<std::vector<float>> &targetBatches, int batchSize);
#endif // CSV_READER_H
