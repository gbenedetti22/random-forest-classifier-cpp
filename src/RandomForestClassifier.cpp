//
// Created by gabriele on 13/07/25.
//

#include "../include/RandomForestClassifier.h"

#include <cassert>
#include <iostream>
#include <omp.h>
#include <set>

#include "../include/Timer.h"
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
        exit(EXIT_FAILURE);
    }
    if (params.njobs == 0 || params.njobs < -1) {
        cerr << "Thread count cannot be 0 or less then -1: " << params.njobs << endl;
        exit(EXIT_FAILURE);
    }

    const int num_trees = trees.capacity();
    timer.set_active(false);
    timer.start("transpose");
    cout << "Transposing.." << endl;
    transpose(X);
    timer.stop("transpose");

    trees.reserve(num_trees);
    const int t = params.njobs;
    int threads_count = t == -1 ? omp_get_max_threads() : t;

    #pragma omp parallel for if(threads_count > 1) num_threads(threads_count)
    for (int i = 0; i < num_trees; i++) {
        #pragma omp critical(output)
        {
            cout << "Thread " << omp_get_thread_num() + 1 << "/" << omp_get_num_threads()
                 << ": Training tree n. " << i + 1 << endl;
        }

        DecisionTreeClassifier tree(params.split_criteria, params.min_samples_split, params.max_features,
                                    params.random_seed, params.min_samples_ratio, params.nworkers);

        if (params.bootstrap) {
            vector<int> indices;

            timer.start("bootstrap");
            bootstrap_sample(X[0].size(), indices);
            timer.stop("bootstrap");

            tree.train(X, y, indices);
        } else {
            assert(false && "TODO");
            // tree.train(X, y, TODO);
        }

        #pragma omp critical
        {
            trees.push_back(tree);
        }
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

pair<float, float> RandomForestClassifier::evaluate(const vector<vector<float> > &X, const vector<int> &y) const {
    int classified = 0;
    vector<int> y_pred;
    y_pred.reserve(X.size());
    for (int i = 0; i < X.size(); ++i) {
        int prediction = predict(X[i]);
        y_pred.push_back(prediction);

        if (prediction == y[i]) {
            classified++;
        }
    }

    return make_pair(static_cast<float>(classified) / X.size(), f1_score(y, y_pred));
}

float RandomForestClassifier::f1_score(const vector<int> &y, const vector<int> &y_pred) {
    assert(y.size() == y_pred.size());

    const int maxLabel = *ranges::max_element(y);
    const int numClasses = maxLabel + 1;

    vector TP(numClasses, 0);
    vector FP(numClasses, 0);
    vector FN(numClasses, 0);

    for (int i = 0; i < y_pred.size(); ++i) {
        const int pred = y_pred[i];
        const int trueLabel = y[i];

        if (pred == trueLabel) {
            TP[trueLabel]++;
        } else {
            FP[pred]++;
            FN[trueLabel]++;
        }
    }

    double macro_f1 = 0.0;
    for (int label = 0; label < numClasses; ++label) {
        const double precision = TP[label] + FP[label] > 0
                                     ? static_cast<double>(TP[label]) / (TP[label] + FP[label])
                                     : 0.0;
        const double recall = TP[label] + FN[label] > 0
                                  ? static_cast<double>(TP[label]) / (TP[label] + FN[label])
                                  : 0.0;

        const double f1 = precision + recall > 0
                              ? 2.0 * precision * recall / (precision + recall)
                              : 0.0;

        macro_f1 += f1;
    }

    return static_cast<float>(macro_f1 / numClasses);
}


void RandomForestClassifier::bootstrap_sample(const int n_samples, vector<int> &indices) const {
    indices.clear();
    indices.reserve(n_samples);

    if (params.random_seed.has_value()) {
        srand(params.random_seed.value());
    } else {
        srand(time(nullptr));
    }

    for (size_t i = 0; i < n_samples; ++i) {
        const int idx = rand() % n_samples;

        indices.push_back(idx);
    }
}
