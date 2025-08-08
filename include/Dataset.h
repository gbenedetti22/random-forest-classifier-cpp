//
// Created by gabriele on 13/07/25.
//

#ifndef DATASET_H
#define DATASET_H
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

class Dataset {
public:
    Dataset();
    static std::pair<std::vector<std::vector<float> >, std::vector<int>> load(std::string filename, const std::string &directory = ".", size_t max_lines = SIZE_MAX);

    static std::tuple<std::vector<std::vector<float>>, std::vector<int>, std::vector<std::vector<float>>, std::vector<int>> train_test_split(std::vector<std::vector<float>> &X, std::vector<int> &y, float train_perc = 0.8, bool stratified = false);

private:
    static void process_line(std::vector<std::vector<float>> &X, std::vector<int> &y,
                      const std::string &dataset_name,
                      const std::string &line, std::unordered_map<std::string, int>& labels, int &label_id);

    static std::vector<std::string> split(const std::string &s, char delim = ',');

    static void shuffle_data(std::vector<std::vector<float>> &X, std::vector<int> &y);
};

#endif //DATASET_H
