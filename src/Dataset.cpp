//
// Created by gabriele on 13/07/25.
//

#include "../include/Dataset.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <tuple>
#include <tuple>
#include <tuple>
#include <tuple>
#include <unordered_map>
using namespace std;

Dataset::Dataset() = default;

pair<vector<vector<double>>, vector<int>> Dataset::load_classification(const string &filename) {
    if (filename == "iris") {
        return load_iris("../dataset/iris.data");
    }

    if (filename == "susy") {
        return load_susy("../dataset/SUSY.csv");
    }

    cerr << "Dataset not found: " << filename << endl;
    exit(1);
}

tuple<vector<vector<double>>, vector<int>, vector<vector<double>>, vector<int>>
Dataset::train_test_split(vector<vector<double>> &X,
                          vector<int> &y,
                          const double train_perc, const bool stratified) {
    const int n_samples = X.size();

    vector<vector<double>> X_train, X_test;
    vector<int> y_train, y_test;

    if (!stratified) {
        shuffle_data(X, y);

        const int train_size = static_cast<int>(n_samples * train_perc);

        X_train.reserve(train_size);
        y_train.reserve(train_size);
        X_test.reserve(n_samples - train_size);
        y_test.reserve(n_samples - train_size);

        for (int i = 0; i < n_samples; i++) {
            if (i < train_size) {
                X_train.push_back(X[i]);
                y_train.push_back(y[i]);
            } else {
                X_test.push_back(X[i]);
                y_test.push_back(y[i]);
            }
        }
    } else {
        unordered_map<int, vector<int>> class_indices;
        for (int i = 0; i < y.size(); ++i) {
            class_indices[y[i]].push_back(i);
        }

        random_device rd;
        mt19937 gen(rd());

        for (auto& [label, indices] : class_indices) {
            ranges::shuffle(indices, gen);
            const int train_count = static_cast<int>(indices.size() * train_perc);

            for (int i = 0; i < indices.size(); ++i) {
                const int idx = indices[i];
                if (i < train_count) {
                    X_train.push_back(X[idx]);
                    y_train.push_back(y[idx]);
                } else {
                    X_test.push_back(X[idx]);
                    y_test.push_back(y[idx]);
                }
            }
        }
    }

    return make_tuple(X_train, y_train, X_test, y_test);
}

vector<string> Dataset::split(const string &s, const char delim) {
    stringstream ss(s);
    vector<string> tokens;
    string value;
    while (getline(ss, value, delim)) {
        tokens.push_back(value);
    }
    return tokens;
}

pair<vector<vector<double>>, vector<int>> Dataset::load_iris(const string &filename) {
    ifstream file(filename);
    if (!file) {
        cerr << "Could not open file " << filename << endl;
        cout << "Current directory: " << filesystem::current_path() << endl;
        exit(1);
    }

    vector<vector<double>> X;
    vector<int> y;
    unordered_map<string, int> labels;
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

        X.push_back(features);
        y.push_back(labels[label]);
    }

    file.close();
    return {X, y};
}

pair<vector<vector<double>>, vector<int>> Dataset::load_susy(const string &filename) {
    ifstream file(filename);
    if (!file) {
        cerr << "Could not open file " << filename << endl;
        cout << "Current directory: " << filesystem::current_path() << endl;
        exit(1);
    }

    vector<vector<double>> X;
    vector<int> y;

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

        X.push_back(features);
        y.push_back(stoi(label));
    }

    file.close();
    return {X, y};
}

void Dataset::shuffle_data(vector<vector<double>> &X, vector<int> &y) {
    for (int i = y.size() - 1; i >= 0; i--) {

        int j = rand() % (i + 1);
        swap(X[i], X[j]);
        swap(y[i], y[j]);
    }
}