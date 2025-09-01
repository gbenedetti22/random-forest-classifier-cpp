//
// Created by gabriele on 13/07/25.
//

#include "../include/DecisionTreeClassifier.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <omp.h>
#include <unordered_map>
#include <random>
#include <ranges>
#include <stack>
#include <tuple>
#include <tuple>
#include <tuple>
#include "../include/Timer.h"
#include "../include/radix_sort_indices.h"
#include "../include/splitters/SplitterFF.hpp"
#include "splitters/SequentialSplitter.hpp"
#include <Eigen/Core>
#include <pdqsort.h>

using namespace std;

void DecisionTreeClassifier::train(const vector<vector<float>> &X, const vector<int> &y, vector<int> &samples) {
    build_tree(X, y, samples);
}

int DecisionTreeClassifier::predict(const vector<float> &x) const {
    const TreeNode *node = root;
    while (!node->is_leaf) {
        if (x[node->feature_index] < node->threshold)
            node = node->left;
        else
            node = node->right;
    }
    return node->predicted_class;
}

void DecisionTreeClassifier::build_tree(const vector<vector<float>> &X, const vector<int> &y, vector<int> &samples) {
    root = new TreeNode();

    const int total_features = static_cast<int>(X[0].size());
    unordered_map<string, int> op_value = {{"sqrt", sqrt(total_features)}, {"log2", log2(total_features)}};

    stack<tuple<vector<int>, TreeNode *> > stack;
    stack.emplace(samples, root);

    int n_features = 0;
    if (std::holds_alternative<int>(max_features)) {
        n_features = std::get<int>(max_features);
    } else if (std::holds_alternative<string>(max_features)) {
        const auto op = std::get<string>(max_features);
        n_features = op_value[op];
    }

    auto compute_fn = [this](const vector<vector<float>> &X,
                         const vector<int> &y,
                         vector<int> &indices,
                         const int f,
                         const unordered_map<int,int> &label_counts,
                         const int num_classes) -> pair<float,float> {
        return this->compute_threshold(X, y, indices, f, label_counts, num_classes);
    };

    auto split_fn = [](const vector<vector<float>> &X,
                           const vector<int> &indices,
                           const float th,
                           const int f) -> tuple<vector<int>, vector<int>> {
        return split_left_right(X, indices, th, f);
    };

    unique_ptr<BaseSplitter> splitter;
    assert(nworkers > 0 || nworkers == -1);

    if (nworkers == 1) {
        splitter = make_unique<SequentialSplitter>(compute_fn, split_fn);
    }else {
        nworkers = nworkers == -1 ? omp_get_max_threads() : nworkers;
        splitter = make_unique<SplitterFF>(compute_fn, split_fn, nworkers);
    }

    while (!stack.empty()) {
        auto [indices, node] = std::move(stack.top());
        stack.pop();

        int num_classes = 1;

        timer.start("label counts");
        unordered_map<int, int> label_counts;
        for (const int i: indices) {
            label_counts[y[i]]++;
            if (y[i] > num_classes) {
                num_classes = y[i];
            }
        }
        timer.stop("label counts");

        if (label_counts.size() == 1 || indices.size() < min_samples_split) {
            node->is_leaf = true;
            node->predicted_class = compute_majority_class(label_counts);
            continue;
        }

        assert(n_features > 0 && "Invalid max_feature parameter");
        vector<int> selected_features = sample_features(total_features, n_features);

        const SplitterResult best_split = splitter->find_best_split(
            X, y, indices, selected_features, label_counts, num_classes, min_samples_ratio
        );

        const int best_feature = best_split.best_feature;
        const float best_threshold = best_split.best_threshold;

        const vector<int>& best_left_X = best_split.left_indices;
        const vector<int>& best_right_X = best_split.right_indices;

        if (best_feature == -1) {
            node->is_leaf = true;
            node->predicted_class = compute_majority_class(label_counts);
            continue;
        }

        auto *left_node = new TreeNode();
        auto *right_node = new TreeNode();

        node->is_leaf = false;
        node->feature_index = best_feature;
        node->threshold = best_threshold;
        node->left = left_node;
        node->right = right_node;

        stack.emplace(best_left_X, left_node);
        stack.emplace(best_right_X, right_node);
    }
}

tuple<vector<int>, vector<int>> DecisionTreeClassifier::split_left_right(const vector<vector<float>> &X,
                                              const vector<int> &indices,
                                              const float th,
                                              const int f) {
    const auto it = ranges::partition_point(indices, [&X, th, f](const int i) {
        return X[i][f] < th;
    });

    vector left_indices(indices.begin(), it);
    vector right_indices(it, indices.end());

    return std::move(tuple{left_indices, right_indices});
}

pair<float, float> DecisionTreeClassifier::compute_threshold(const vector<vector<float>> &X, const vector<int> &y,
                                                             vector<int> &indices, const int f,
                                                             const unordered_map<int, int> &label_counts,
                                                             const int num_classes) const {
    timer.start("treshold: sorting");
    RADIX_SORT_INDICES(indices, X, f);
    timer.stop("treshold: sorting");

    float best_threshold = 0.0;
    float best_impurity = numeric_limits<float>::max();
    float prev_impurity = numeric_limits<float>::max();
    float impurity_tol = 1e-4;

    vector<int> left_counts, right_counts;
    left_counts.resize(num_classes);
    right_counts.resize(num_classes);

    timer.start("threshold: label counts");
    for (auto &[label, count]: label_counts) {
        right_counts[label] = count;
    }
    timer.stop("threshold: label counts");

    int left_total = 0;
    int right_total = static_cast<int>(indices.size());
    int offset = 1;

    timer.start("treshold: main");
    for (int i = 0; i < indices.size() - 1; i += offset) {
        i = min(i, static_cast<int>(indices.size() - 1));

        for (int j = max(0, i - offset + 1); j <= i; ++j) {
            const int current_idx = indices[j];
            const int current_label = y[current_idx];

            left_counts[current_label]++;
            right_counts[current_label]--;
            left_total++;
            right_total--;
        }

        const int current_idx = indices[i];
        const int next_idx = indices[i + 1];

        const float current_val = X[current_idx][f];
        const float next_val = X[next_idx][f];

        if (current_val == next_val) continue;

        const float threshold = (current_val + next_val) * 0.5f;

        const float gini_left = get_impurity(left_counts, left_total);
        const float gini_right = get_impurity(right_counts, right_total);

        const float inv_total = 1.0f / static_cast<float>(left_total + right_total);
        const float weighted_impurity =
        (static_cast<float>(left_total) * gini_left +
         static_cast<float>(right_total) * gini_right) * inv_total;

        if (weighted_impurity < best_impurity) {
            best_impurity = weighted_impurity;
            best_threshold = threshold;
        }

        const float impurity_delta = abs(weighted_impurity - prev_impurity);

        if (impurity_delta < impurity_tol) {
            offset *= 2;
        } else if (impurity_delta > 0.05) {
            offset = max(offset / 2, 1);
        }

        prev_impurity = weighted_impurity;
    }

    timer.stop("treshold: main");

    return std::move(pair{best_threshold, best_impurity});
}

float DecisionTreeClassifier::gini(const vector<int> &counts, const int total) {
    if (total == 0) return 0.0;
    const float inv_total = 1.0f / static_cast<float>(total);

    float gini = 1.0;
    for (const int count: counts) {
        const float p = static_cast<float>(count) * inv_total;
        gini -= p * p;
    }
    return gini;
}

float DecisionTreeClassifier::entropy(const vector<int> &counts, const int total) {
    float entropy = 0.0;
    const float inv_total = 1.0f / static_cast<float>(total);

    for (const int count: counts) {
        const float prob = static_cast<float>(count) * inv_total;
        if (prob > 0) {
            entropy -= prob * std::log2(prob);
        }
    }
    return entropy;
}


float DecisionTreeClassifier::get_impurity(const vector<int> &counts, const int total) const {
    if (split_criteria == "gini") {
        return gini(counts, total);
    }

    if (split_criteria == "entropy") {
        return entropy(counts, total);
    }

    cerr << "Unknown split criteria " << split_criteria << endl;
    exit(1);
}

vector<int> DecisionTreeClassifier::sample_features(const int total_features, const int n_features) {
    vector<int> all_features(total_features);
    iota(all_features.begin(), all_features.end(), 0);
    if (total_features == n_features) return all_features;

    for (int i = 0; i < n_features; i++) {
        const int j = uniform_int_distribution(i, total_features - 1)(rng);
        swap(all_features[i], all_features[j]);
    }

    all_features.resize(n_features);
    return all_features;
}

int DecisionTreeClassifier::compute_majority_class(const unordered_map<int, int> &counts) {
    timer.start("majority");
    int majority_class = -1, max_count = -1;
    for (const auto &[label, count]: counts) {
        if (count > max_count) {
            max_count = count;
            majority_class = label;
        }
    }
    timer.stop("majority");
    return majority_class;
}
