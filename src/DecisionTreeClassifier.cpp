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
#include "../include/Timer.h"
#include "../include/splitters/SplitterFF.hpp"
#include "splitters/SequentialSplitter.hpp"
#include  <radix_sort_indices.h>


using namespace std;

void DecisionTreeClassifier::train(const vector<float> &X, const pair<long, long>& shape, const vector<int> &y, vector<int> &indices) {
    timer.start("matrix");
    TrainMatrix X_train(X, shape, indices);
    timer.stop("matrix");

    vector<int> y_train;
    y_train.reserve(indices.size());

    for (const int idx: indices) {
        y_train.push_back(y[idx]);
    }

    build_tree(X_train, y_train);
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

void DecisionTreeClassifier::build_tree(TrainMatrix &X, vector<int> &y) {
    root = new TreeNode();

    const size_t total_features = X.nFeatures();
    const size_t total_samples = X.nSamples();
    unordered_map<string, int> op_value = {{"sqrt", sqrt(total_features)}, {"log2", log2(total_features)}};

    // Stack now stores ranges (start, end) instead of indices
    stack<tuple<size_t, size_t, TreeNode *> > stack;
    stack.emplace(0, total_samples, root);

    int n_features = 0;
    if (std::holds_alternative<int>(max_features)) {
        n_features = std::get<int>(max_features);
    } else if (std::holds_alternative<string>(max_features)) {
        const auto op = std::get<string>(max_features);
        n_features = op_value[op];
    }

    auto compute_fn = [this](const TrainMatrix &X,
                         const vector<int> &y,
                         const size_t start, const size_t end,
                         const int f,
                         const unordered_map<int,int> &label_counts,
                         const int num_classes) -> tuple<float,float, size_t> {
        return this->compute_threshold(X, y, start, end, f, label_counts, num_classes);
    };

    unique_ptr<BaseSplitter> splitter;
    assert(nworkers > 0 || nworkers == -1);

    if (nworkers == 1) {
        splitter = make_unique<SequentialSplitter>(compute_fn);
    }else {
        nworkers = nworkers == -1 ? omp_get_max_threads() : nworkers;
        splitter = make_unique<SplitterFF>(compute_fn, nworkers);
    }

    int it=0;
    while (!stack.empty()) {
        auto [start, end, node] = std::move(stack.top());
        stack.pop();

        int num_classes = 0;

        unordered_map<int, int> label_counts;
        for (size_t i = start; i < end; ++i) {
            label_counts[y[i]]++;
            if (y[i] > num_classes) {
                num_classes = y[i];
            }
        }
        num_classes++;

        if (label_counts.size() == 1 || (end - start) < min_samples_split) {
            node->is_leaf = true;
            node->predicted_class = compute_majority_class(label_counts);
            node->left = nullptr;
            node->right = nullptr;
            continue;
        }

        assert(n_features > 0 && "Invalid max_feature parameter");
        vector<int> selected_features = sample_features(total_features, static_cast<size_t>(n_features));

        // float best_impurity = std::numeric_limits<float>::max();
        // float best_threshold = 0.0f;
        // int best_feature = -1;

        // for (const int f : selected_features) {
        //     auto [threshold, impurity, split_point] = compute_threshold(X, y, start, end, f, label_counts, num_classes);
        //
        //     if (impurity < best_impurity) {
        //         size_t total_left = split_point - start;
        //         size_t total_right = end - split_point;
        //         const float ratio = static_cast<float>(min(total_left, total_right)) /
        //                             static_cast<float>(end - start);
        //
        //         if (ratio > min_samples_ratio) {
        //             best_impurity = impurity;
        //             best_feature = f;
        //             best_threshold = threshold;
        //         }
        //     }
        // }

        const SplitterResult best_split = splitter->find_best_split(
            X, y, start, end, selected_features, label_counts, num_classes, min_samples_ratio
        );

        const int best_feature = best_split.best_feature;
        const float best_threshold = best_split.best_threshold;

        if (best_feature == -1) {
            node->is_leaf = true;
            node->predicted_class = compute_majority_class(label_counts);
            continue;
        }

        timer.start("split");
        size_t split_point = split_left_right(X, y, start, end, best_threshold, best_feature);
        timer.stop("split");

        auto *left_node = new TreeNode();
        auto *right_node = new TreeNode();

        node->is_leaf = false;
        node->feature_index = best_feature;
        node->threshold = best_threshold;
        node->left = left_node;
        node->right = right_node;

        stack.emplace(start, split_point, left_node);
        stack.emplace(split_point, end, right_node);
    }
}

size_t DecisionTreeClassifier::split_left_right(TrainMatrix &X,
                                              vector<int> &y,
                                              const size_t start, const size_t end,
                                              const float th,
                                              const size_t f) {
    constexpr int BUFFER_SIZE = 256;
    size_t left = start;
    size_t right = end - 1;

    pair<size_t, size_t> buffer[BUFFER_SIZE];
    int buffer_count = 0;

    while (left <= right) {
        while (left <= right && X.getValue(f, left) < th) {
            left++;
        }

        while (left <= right && X.getValue(f, right) >= th) {
            right--;
        }

        if (left < right) {
            buffer[buffer_count] = pair{left, right};
            buffer_count++;

            swap(y[left], y[right]);

            if (buffer_count == BUFFER_SIZE) {
                for (size_t feature = 0; feature < X.nFeatures(); ++feature) {
                    for (int i = 0; i < buffer_count; ++i) {
                        swap(X(feature, buffer[i].first), X(feature, buffer[i].second));
                    }
                }
                buffer_count = 0;
            }

            left++;
            right--;
        }
    }

    if (buffer_count > 0) {
        for (size_t feature = 0; feature < X.nFeatures(); ++feature) {
            for (size_t i = 0; i < buffer_count; ++i) {
                swap(X(feature, buffer[i].first), X(feature, buffer[i].second));
            }
        }
    }

    return left;
}

tuple<float, float, size_t> DecisionTreeClassifier::compute_threshold(const TrainMatrix &X,
                                                                             const vector<int> &y,
                                                                             const size_t start, const size_t end,
                                                                             const int f,
                                                                             const unordered_map<int, int> &
                                                                             label_counts,
                                                                             const int num_classes) const {
    timer.start("threshold - sorting");
    vector<pair<uint8_t, int>> temp_data;
    auto getKey = [&X, f, start](const size_t i) { return X(f, i + start); };
    auto getValue = [start](const size_t i) { return i + start; };
    COUNTING_SORT_CALLBACK(getKey, getValue, end - start, temp_data);
    timer.stop("threshold - sorting");

    float best_threshold = 0.0;
    float best_impurity = numeric_limits<float>::max();
    float prev_impurity = numeric_limits<float>::max();
    float impurity_tol = 1e-4;
    size_t best_split_point = start;

    vector<int> left_counts, right_counts;
    left_counts.resize(num_classes);
    right_counts.resize(num_classes);

    for (auto &[label, count]: label_counts) {
        right_counts[label] = count;
    }

    size_t left_total = 0;
    size_t right_total = end - start;
    int offset = 1;

    timer.start("threshold - main");
    for (size_t i = 0; i < temp_data.size() - 1; i += offset) {
        i = min(i, temp_data.size() - 1);

        for (size_t j = max(static_cast<size_t>(0), i - offset + 1); j <= i; ++j) {
            const int current_label = y[temp_data[j].second];

            left_counts[current_label]++;
            right_counts[current_label]--;
            left_total++;
            right_total--;
        }

        const float current_val = X.toValue(temp_data[i].first);
        const float next_val = X.toValue(temp_data[i + 1].first);

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
            best_split_point = start + left_total;
        }

        const float impurity_delta = abs(weighted_impurity - prev_impurity);

        if (impurity_delta < impurity_tol) {
            offset *= 2;
        } else if (impurity_delta > 0.05) {
            offset = max(offset / 2, 1);
        }

        prev_impurity = weighted_impurity;
    }
    timer.stop("threshold - main");

    return tuple{best_threshold, best_impurity, best_split_point};
}

float DecisionTreeClassifier::gini(const vector<int> &counts, const size_t total) {
    if (total == 0) return 0.0;
    const float inv_total = 1.0f / static_cast<float>(total);

    float gini = 1.0;
    for (const int count: counts) {
        const float p = static_cast<float>(count) * inv_total;
        gini -= p * p;
    }
    return gini;
}

float DecisionTreeClassifier::entropy(const vector<int> &counts, const size_t total) {
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


float DecisionTreeClassifier::get_impurity(const vector<int> &counts, const size_t total) const {
    if (split_criteria == "gini") {
        return gini(counts, total);
    }

    if (split_criteria == "entropy") {
        return entropy(counts, total);
    }

    cerr << "Unknown split criteria " << split_criteria << endl;
    exit(1);
}

vector<int> DecisionTreeClassifier::sample_features(const size_t total_features, const size_t n_features) {
    vector<int> all_features(total_features);
    iota(all_features.begin(), all_features.end(), 0);
    if (total_features == n_features) return all_features;

    for (int i = 0; i < n_features; i++) {
        const size_t j = static_cast<size_t>(uniform_int_distribution(i, static_cast<int>(total_features - 1))(rng));
        swap(all_features[i], all_features[j]);
    }

    all_features.resize(n_features);
    return all_features;
}

int DecisionTreeClassifier::compute_majority_class(const unordered_map<int, int> &counts) {
    if (counts.empty()) {
        return 0; // Default to class 0 if no counts
    }

    int majority_class = -1, max_count = -1;
    for (const auto &[label, count]: counts) {
        if (count > max_count) {
            max_count = count;
            majority_class = label;
        }
    }
    return majority_class;
}

