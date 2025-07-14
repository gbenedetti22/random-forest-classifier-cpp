//
// Created by gabriele on 13/07/25.
//

#ifndef DATASET_H
#define DATASET_H
#include <string>
#include <vector>

struct Sample {
    std::vector<double> features;
    int label;
};

class Dataset {
public:
    Dataset();

    static std::vector<Sample> load_classification(const std::string &filename);

    static std::tuple<std::vector<Sample>, std::vector<Sample> > train_test_split(const std::vector<Sample>& dataset, double train_perc=0.8);

private:
    static std::vector<std::string> split(const std::string &s, char delim = ',');
    static std::vector<Sample> load_iris(const std::string &filename);

    static std::vector<Sample> load_susy(const std::string &filename);
};

#endif //DATASET_H
