//
// Created by gabriele on 13/07/25.
//

#include "../include/RandomForestClassifier.h"

#include <iostream>
#include <set>
#include <unordered_set>

#include "indicators.hpp"
#include "../include/Timer.h"
#include "../include/BS_thread_pool.hpp"
using namespace std;

void transpose(vector<vector<float> > &matrix) {
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

    vector<float> flat(m * n);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            flat[i * n + j] = matrix[i][j];
        }
    }

    vector visited(m * n, false);

    for (int start = 0; start < m * n; start++) {
        if (visited[start]) continue;

        int current = start;
        float temp = flat[current];

        do {
            visited[current] = true;

            const int row = current / n;
            const int col = current % n;

            const int next = col * m + row;

            const float nextTemp = flat[next];
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

void RandomForestClassifier::fit(vector<vector<float> > &X, const vector<int> &y) {
    if (X.empty() || y.empty()) {
        cerr << "Cannot build the tree on dataset" << endl;
        return;
    }
    const int num_trees = trees.capacity();
    timer.start("transpose");
    cout << "Transposing.." << endl;
    transpose(X);
    timer.stop("transpose");
    timer.set_active(false);

    trees.reserve(num_trees);
    for (int i = 0; i < num_trees; i++) {
        cout << "Training tree n. " << i + 1 << endl;

        DecisionTreeClassifier tree(params.split_criteria, params.min_samples_split, params.max_features,
                                    params.random_seed);

        if (params.bootstrap) {
            vector<int> indices;

            timer.start("bootstrap");
            bootstrap_sample(X[0].size(), indices);
            timer.stop("bootstrap");

            tree.train(X, y, indices);
        }else {
            assert(false && "TODO");
            // tree.train(X, y, TODO);
        }

        trees.push_back(tree);
    }
}

int RandomForestClassifier::predict(const vector<float> &x) const {
    unordered_map<int, int> vote_counts;
    for (const auto &tree: trees) {
        int pred = tree.predict(x);
        vote_counts[pred]++;
    }
    int majority_class = -1, max_votes = -1;
    for (const auto &[label, count]: vote_counts) {
        if (count > max_votes) {
            max_votes = count;
            majority_class = label;
        }
    }
    return majority_class;
}

float RandomForestClassifier::evaluate(const vector<vector<float> > &X, const vector<int> &y) const {
    int classified = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        if (predict(X[i]) == y[i]) {
            classified++;
        }
    }

    return static_cast<float>(classified) / X.size();
}

void RandomForestClassifier::bootstrap_sample(const int n_samples, vector<int> &indices) const {
    indices.clear();
    indices.reserve(n_samples);

    if (params.random_seed.has_value()) {
        std::srand(params.random_seed.value());
    }else {
        std::srand(std::time(nullptr));
    }

    for (size_t i = 0; i < n_samples; ++i) {
        const int idx = rand() % n_samples;

        indices.push_back(idx);
    }

}
