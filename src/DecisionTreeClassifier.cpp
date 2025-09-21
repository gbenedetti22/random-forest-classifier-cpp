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
#include "../include/radix_sort_indices.h"
#include <pdqsort.h>
#include <TrainMatrix.hpp>

#include "splitters/BaseSplitter.hpp"
#include "splitters/HistogramSplitter.hpp"

using namespace std;

static void destroy_tree(const TreeNode* node) {
    if (!node) return;
    destroy_tree(node->left);
    destroy_tree(node->right);
    delete node;
}

DecisionTreeClassifier::~DecisionTreeClassifier() {
    destroy_tree(root);
    root = nullptr;
}

void DecisionTreeClassifier::train(const vector<float> &X, const pair<size_t, size_t>& shape, const vector<int> &y, vector<int> &samples) {
    pdqsort_branchless(samples.begin(), samples.end());

    timer.start("matrix");
    const TrainMatrix m(X, shape);
    timer.stop("matrix");

    build_tree(m, y, samples);
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

void DecisionTreeClassifier::build_tree(const TrainMatrix &X, const vector<int> &y, vector<int> &samples) {
    root = new TreeNode();
    int total_nodes = 0;

    const size_t total_features = X.nFeatures();
    unordered_map<string, int> op_value = {{"sqrt", sqrt(total_features)}, {"log2", log2(total_features)}};

    int n_features = 0;
    if (std::holds_alternative<int>(max_features)) {
        n_features = std::get<int>(max_features);
    } else if (std::holds_alternative<string>(max_features)) {
        const auto op = std::get<string>(max_features);
        n_features = op_value[op];
    }

    auto compute_fn = [this](const TrainMatrix &X,
                         const vector<int> &y,
                         vector<int> &indices,
                         const int f,
                         const unordered_map<int,int> &label_counts,
                         const int num_classes) -> tuple<float,float,size_t> {
        return this->compute_threshold(X, y, indices, f, label_counts, num_classes);
    };

    unique_ptr<BaseSplitter> splitter;
    assert(nworkers > 0 || nworkers == -1);

    splitter = make_unique<HistogramSplitter>(compute_fn);
    // if (nworkers == 1) {
    //     splitter = make_unique<SequentialSplitter>(compute_fn, split_fn);
    // }else {
    //     nworkers = nworkers == -1 ? omp_get_max_threads() : nworkers;
    //     splitter = make_unique<SplitterFF>(compute_fn, split_fn, nworkers);
    // }
    stack<tuple<vector<int>, TreeNode *> > stack;
    stack.emplace(samples, root);

    while (!stack.empty()) {
        auto [indices, node] = std::move(stack.top());
        stack.pop();

        int num_classes = 1;

        unordered_map<int, int> label_counts;
        for (const int i: indices) {
            label_counts[y[i]]++;
            if (y[i] > num_classes) {
                num_classes = y[i];
            }
        }

        if (label_counts.size() == 1 || indices.size() < min_samples_split) {
            node->is_leaf = true;
            node->predicted_class = compute_majority_class(label_counts);
            continue;
        }

        assert(n_features > 0 && "Invalid max_feature parameter");
        vector<int> selected_features = sample_features(total_features, n_features);

        timer.start("histogram");
        const SplitterResult best_split = splitter->find_best_split(
            X, y, indices, selected_features, label_counts, num_classes, min_samples_ratio
        );
        timer.stop("histogram");

        const int best_feature = best_split.best_feature;
        const float best_threshold = best_split.best_threshold;
        if (best_feature == -1) {
            node->is_leaf = true;
            node->predicted_class = compute_majority_class(label_counts);
            continue;
        }

        timer.start("split");
        auto [best_left_X, best_right_X] = split_left_right(X, indices, best_threshold, best_feature);
        timer.stop("split");

        auto *left_node = new TreeNode();
        auto *right_node = new TreeNode();
        total_nodes += 2;

        node->is_leaf = false;
        node->feature_index = best_feature;
        node->threshold = best_threshold;
        node->left = left_node;
        node->right = right_node;

        stack.emplace(best_left_X, left_node);
        stack.emplace(best_right_X, right_node);
    }

    cout << "Nodes created: " << total_nodes << endl;
}

tuple<vector<int>, vector<int>> DecisionTreeClassifier::split_left_right(const TrainMatrix &X,
                                              const vector<int> &indices,
                                              const float th,
                                              const int f) {
    vector<int> left_indices, right_indices;
    left_indices.reserve(indices.size());
    right_indices.reserve(indices.size());

    for (const int idx: indices) {
        if (X(f, idx) < th) {
            left_indices.push_back(idx);
        }else {
            right_indices.push_back(idx);
        }
    }

    return std::move(tuple{left_indices, right_indices});
}

tuple<float, float, size_t> DecisionTreeClassifier::compute_threshold(const TrainMatrix &X,
                                                                        const std::vector<int> &y,
                                                                        const std::vector<int>& indices,
                                                                        const int f,
                                                                        const std::unordered_map<int, int> &label_counts,
                                                                        const int num_classes) const {
    thread_local std::vector histogram(256, std::vector(num_classes, 0));
        thread_local int bin_counts[256] = {};
        thread_local uint8_t active_bins[256];
        thread_local int active_bins_size = 0;

        std::ranges::fill(bin_counts, 0);
        active_bins_size = 0;
        for (auto& h : histogram) {
            std::ranges::fill(h, 0);
        }

        timer.start("histogram - creation");
        for (const int idx : indices) {
            const uint8_t bin = X.getQuantized(f, idx);
            const int label = y[idx];
            if (bin_counts[bin] == 0) {
                active_bins[active_bins_size++] = bin;
            }

            histogram[bin][label]++;
            bin_counts[bin]++;
        }
        timer.stop("histogram - creation");

        pdqsort_branchless(active_bins, active_bins + active_bins_size);

        constexpr int start = 0;
        const size_t end = indices.size();
        float best_threshold = 0.0f;
        float best_impurity = std::numeric_limits<float>::max();
        size_t best_left_total = 0;

        std::vector<int> left_counts, right_counts;
        left_counts.resize(num_classes);
        right_counts.resize(num_classes);

        for (const auto &[label, count] : label_counts) {
            right_counts[label] = count;
        }

        size_t left_total = 0;
        size_t right_total = end - start;

        timer.start("histogram - main loop");
        for (size_t i = 0; i < active_bins_size - 1; ++i) {
            const int bin = active_bins[i];
            const int next_bin = active_bins[i + 1];

            for (int label = 0; label < num_classes; ++label) {
                const int value = histogram[bin][label];

                left_counts[label] += value;
                right_counts[label] -= value;
            }
            left_total += bin_counts[bin];
            right_total -= bin_counts[bin];

            if (left_total == 0 || right_total == 0) continue;

            const float current_val = X.toValue(static_cast<uint8_t>(bin));
            const float next_val = X.toValue(static_cast<uint8_t>(next_bin));

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
                best_left_total = left_total;
            }
        }
        timer.stop("histogram - main loop");

        return std::make_tuple(best_threshold, best_impurity, best_left_total);
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
    if (counts.empty()) {
        return 0; // Default to class 0 if no counts
    }

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
