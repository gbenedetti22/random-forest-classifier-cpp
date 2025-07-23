//
// Created by gabriele on 13/07/25.
//

#include "../include/RandomForestClassifier.h"

#include <iostream>

#include "indicators.hpp"
using namespace std;

void transpose(vector<vector<double> > &matrix) {
    const int m = static_cast<int>(matrix.size());
    const int n = static_cast<int>(matrix[0].size());

    if (m == n) {
        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < n; j++) {
                swap(matrix[i][j], matrix[j][i]);
            }
        }
        return;
    }

    vector<double> flat(m * n);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            flat[i * n + j] = matrix[i][j];
        }
    }

    vector visited(m * n, false);

    for (int start = 0; start < m * n; start++) {
        if (visited[start]) continue;

        int current = start;
        double temp = flat[current];

        do {
            visited[current] = true;

            const int row = current / n;
            const int col = current % n;

            const int next = col * m + row;

            const double nextTemp = flat[next];
            flat[next] = temp;
            temp = nextTemp;

            current = next;
        } while (current != start);
    }

    matrix.resize(n);
    for (int i = 0; i < n; i++) {
        matrix[i].resize(m);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            matrix[i][j] = flat[i * m + j];
        }
    }
}
vector<int> sort_labels_by_features(const vector<int> &labels, const vector<double> &features) {
    assert(labels.size() == features.size() && "Labels should have the same size of Features");

    vector<int> indices(features.size());
    iota(indices.begin(), indices.end(), 0);

    ranges::stable_sort(indices, [&features](const int i, const int j) {
        return features[i] < features[j];
    });

    vector<int> sorted_labels;
    sorted_labels.reserve(labels.size());
    for (const int index : indices) {
        sorted_labels.push_back(labels[index]);
    }

    return sorted_labels;
}

void RandomForestClassifier::fit(const vector<vector<double>> &X, const vector<int> &y) {
    if (X.empty() || y.empty()) {
        cerr << "Cannot build the tree on dataset" << endl;
        return;
    }
    const int num_trees = trees.capacity();

    for (int i = 0; i < num_trees; i++) {
        DecisionTreeClassifier tree(params.split_criteria, params.min_samples_split, params.max_features, params.random_seed);

        if (params.bootstrap) {
            vector<vector<double>> X_sample;
            vector<int> y_sample;
            bootstrap_sample(X, y, X_sample, y_sample);

            transpose(X_sample);

            vector<vector<int>> labels_mapping;
            // labels_mapping.resize(X_sample.size());
            // for (int j = 0; j < X_sample.size(); ++j) {
            //     labels_mapping[j] = sort_labels_by_features(y_sample, X_sample[j]);
            //     ranges::stable_sort(X_sample[j]);
            // }

            tree.train(X_sample, y_sample, labels_mapping);
        }else {
            assert(false && "TODO");
            // tree.train(X, y, TODO);
        }
        
        trees.push_back(tree);
    }
}

int RandomForestClassifier::predict(const vector<double> &x) const {
    unordered_map<int, int> vote_counts;
    for (const auto &tree : trees) {
        int pred = tree.predict(x);
        vote_counts[pred]++;
    }
    int majority_class = -1, max_votes = -1;
    for (const auto &[label, count] : vote_counts) {
        if (count > max_votes) {
            max_votes = count;
            majority_class = label;
        }
    }
    return majority_class;
}

double RandomForestClassifier::evaluate(const vector<vector<double>> &X, const vector<int> &y) const {
    int classified = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        if (predict(X[i]) == y[i]) {
            classified++;
        }
    }

    return static_cast<double>(classified) / X.size();
}

void RandomForestClassifier::bootstrap_sample(const vector<vector<double>> &X, const vector<int> &y, vector<vector<double>> &X_sample, vector<int> &y_sample) const {
    X_sample.clear();
    y_sample.clear();

    if (params.random_seed.has_value()) {
        std::srand(params.random_seed.value());
    }else {
        std::srand(std::time(nullptr));
    }

    for (size_t i = 0; i < X.size(); ++i) {
        const int idx = rand() % X.size();

        X_sample.push_back(X[idx]);
        y_sample.push_back(y[idx]);
    }

}
