//
// Created by gabriele on 13/07/25.
//

#include "../include/Dataset.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_map>
using namespace std;

Dataset::Dataset() = default;

std::vector<Sample> Dataset::load_classification(const std::string &filename) {
    if (filename == "iris") {
        return load_iris("../dataset/iris.data");
    }

    if (filename == "susy") {
        return load_susy("../dataset/SUSY.csv");
    }

    cerr << "Dataset not found: " << filename << endl;
    exit(1);
}

std::tuple<std::vector<Sample>, std::vector<Sample>> Dataset::train_test_split(const std::vector<Sample>& dataset, const double train_perc) {
    const int train_size = static_cast<int>(dataset.size() * train_perc);

    std::vector<Sample> train_set;
    std::vector<Sample> test_set;

    train_set.reserve(train_size);
    test_set.reserve(dataset.size() - train_size);

    for (int i = 0; i < dataset.size(); i++) {
        if (i < train_size) {
            train_set.push_back(dataset[i]);
        } else {
            test_set.push_back(dataset[i]);
        }
    }

    return std::make_tuple(train_set, test_set);
}

std::vector<std::string> Dataset::split(const std::string &s, const char delim) {
    stringstream ss(s);
    vector<string> tokens;
    string value;
    while (getline(ss, value, delim)) {
        tokens.push_back(value);
    }

    return tokens;
}

std::vector<Sample> Dataset::load_iris(const std::string &filename) {
    ifstream file(filename);
    if (!file) {
        std::cerr << "Could not open file " << filename << endl;
        std::cout << "Current directory: " << std::filesystem::current_path() << std::endl;

        exit(1);
    }

    vector<Sample> dataset;
    std::unordered_map<string, int> labels;
    int label_id = 0;

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;

        vector<string> tokens = split(line);
        string label = tokens.back();
        tokens.pop_back();

        vector<double> features;
        for (string& token : tokens) {
            features.push_back(stod(token));
        }

        if (!labels.contains(label)) {
            labels[label] = label_id++;
        }

        dataset.push_back({features, labels[label]});
    }

    file.close();
    return dataset;
}
std::vector<Sample> Dataset::load_susy(const std::string &filename) {
    ifstream file(filename);
    if (!file) {
        std::cerr << "Could not open file " << filename << endl;
        std::cout << "Current directory: " << std::filesystem::current_path() << std::endl;

        exit(1);
    }

    vector<Sample> dataset;

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;

        vector<string> tokens = split(line);
        string label = tokens.front();
        tokens.erase(tokens.begin());

        vector<double> features;
        for (string& token : tokens) {
            features.push_back(stod(token));
        }

        dataset.push_back({features, stoi(label)});
    }

    file.close();
    return dataset;
}