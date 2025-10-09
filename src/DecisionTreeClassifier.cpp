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
#include <pdqsort.h>

#include "spdlog/spdlog.h"
#include "splitters/BaseSplitter.hpp"
#include "splitters/HistogramSplitter.hpp"
#include "splitters/SplitterFF.hpp"


using namespace std;

static void destroy_tree(const TreeNode *node) {
    if (!node) return;
    destroy_tree(node->left);
    destroy_tree(node->right);
    delete node;
}

DecisionTreeClassifier::~DecisionTreeClassifier() {
    destroy_tree(root);
    root = nullptr;
}

void DecisionTreeClassifier::train(const vector<float> &X, const pair<size_t, size_t> &shape, const vector<int> &y,
                                   vector<int> &samples) {
    pdqsort_branchless(samples.begin(), samples.end());

    const TrainMatrix m(X, shape);

    build_tree(m, y, samples);
}

int DecisionTreeClassifier::predict(const float* sample) const {
    const TreeNode *node = root;
    while (!node->is_leaf) {
        if (sample[node->feature_index] < node->threshold)
            node = node->left;
        else
            node = node->right;
    }
    return node->predicted_class;
}

void DecisionTreeClassifier::build_tree(const TrainMatrix &X, const vector<int> &y, vector<int> &indices) {
    root = new TreeNode();

    const size_t total_features = X.nFeatures();
    unordered_map<string, int> op_value = {{"sqrt", sqrt(total_features)}, {"log2", log2(total_features)}};

    int n_features = 0;
    if (std::holds_alternative<int>(params.max_features)) {
        n_features = std::get<int>(params.max_features);
    } else if (std::holds_alternative<string>(params.max_features)) {
        const auto op = std::get<string>(params.max_features);
        n_features = op_value[op];
    }

    auto compute_fn = [this](const TrainMatrix &X,
                             const vector<int> &y,
                             const vector<int> &indices,
                             const size_t start, const size_t end,
                             const int f,
                             const unordered_map<int, int> &label_counts,
                             const int num_classes) -> tuple<float, float, size_t> {
        return this->compute_threshold(X, y, indices, start, end, f, label_counts, num_classes);
    };

    assert(params.nworkers > 0 || params.nworkers == -1);

    int nworkers = params.nworkers;
    unique_ptr<BaseSplitter> splitter = make_unique<HistogramSplitter>(compute_fn);
    if (nworkers == 1) {
        splitter = make_unique<HistogramSplitter>(compute_fn);
    }else {
        nworkers = nworkers == -1 ? omp_get_max_threads() : nworkers;
        splitter = make_unique<SplitterFF>(compute_fn, nworkers);
    }

    stack<tuple<size_t, size_t, TreeNode *, int> > stack;
    stack.emplace(0, indices.size(), root, 0);
    size_t n_leaf_nodes = 0;
    int max_depth = 0;
    while (!stack.empty()) {
        auto [start, end, node, depth] = std::move(stack.top());
        stack.pop();
        if (depth > max_depth) {
            max_depth = depth;
        }

        int num_classes = 0;
        unordered_map<int, int> label_counts;
        for (size_t i = start; i < end; ++i) {
            const int idx = indices[i];

            label_counts[y[idx]]++;
            if (y[idx] > num_classes) {
                num_classes = y[idx];
            }
        }
        num_classes++;

        if (label_counts.size() == 1 || end - start < params.min_samples_split || depth >= params.max_depth ||
            n_leaf_nodes >= params.max_leaf_nodes) {
            node->is_leaf = true;
            node->predicted_class = compute_majority_class(label_counts);
            n_leaf_nodes++;
            continue;
        }

        assert(n_features > 0 && "Invalid max_feature parameter");
        vector<int> selected_features = sample_features(total_features, n_features);

        const SplitterResult best_split = splitter->find_best_split(
            X, y, indices, start, end, selected_features, label_counts, num_classes, params.min_samples_ratio
        );

        const int best_feature = best_split.best_feature;
        const float best_threshold = best_split.best_threshold;

        if (best_feature == -1 || n_leaf_nodes >= params.max_leaf_nodes) {
            node->is_leaf = true;
            node->predicted_class = compute_majority_class(label_counts);
            n_leaf_nodes++;
            continue;
        }

        size_t split_point = split_left_right(X, indices, start, end, best_threshold, best_feature);

        auto *left_node = new TreeNode();
        auto *right_node = new TreeNode();

        node->is_leaf = false;
        node->feature_index = best_feature;
        node->threshold = best_threshold;
        node->left = left_node;
        node->right = right_node;

        stack.emplace(start, split_point, left_node, depth + 1);
        stack.emplace(split_point, end, right_node, depth + 1);
    }

    // cout << "Max depth: " << max_depth << endl;
    // cout << "N. leaf nodes: " << n_leaf_nodes << endl;
}

size_t DecisionTreeClassifier::split_left_right(
    const TrainMatrix &X,
    vector<int> &indices,
    const size_t start,
    const size_t end,
    const float th,
    const int f) {
    size_t left = start;
    size_t right = end - 1;

    while (left <= right) {
        while (left <= right && X(f, indices[left]) < th) {
            left++;
        }

        while (left <= right && X(f, indices[right]) >= th) {
            right--;
        }

        if (left < right) {
            std::swap(indices[left], indices[right]);
            left++;
            right--;
        }
    }

    return left;
}

tuple<float, float, size_t> DecisionTreeClassifier::compute_threshold(const TrainMatrix &X,
                                                                      const std::vector<int> &y,
                                                                      const std::vector<int> &indices,
                                                                      const size_t start, const size_t end,
                                                                      const int f,
                                                                      const std::unordered_map<int, int> &label_counts,
                                                                      const int num_classes) const {
    thread_local std::vector histogram(256 * num_classes, 0);
    thread_local int bin_counts[256] = {};
    thread_local uint8_t active_bins[256];
    thread_local int active_bins_size = 0;

    const size_t histogram_size = 256 * static_cast<size_t>(num_classes);
    if (histogram.size() != histogram_size) {
        histogram.assign(histogram_size, 0);
    } else {
        ranges::fill(histogram, 0);
    }

    std::ranges::fill(bin_counts, 0);
    active_bins_size = 0;

    for (size_t i = start; i < end; ++i) {
        const int idx = indices[i];

        const uint8_t bin = X.getQuantized(f, idx);
        const int label = y[idx];
        if (bin_counts[bin] == 0) {
            active_bins[active_bins_size++] = bin;
        }

        histogram[static_cast<size_t>(bin) * static_cast<size_t>(num_classes) + static_cast<size_t>(label)]++;
        bin_counts[bin]++;
    }

    pdqsort_branchless(active_bins, active_bins + active_bins_size);

    float best_threshold = 0.0f;
    float best_impurity = std::numeric_limits<float>::max();
    size_t best_split_point = start;

    std::vector<int> left_counts, right_counts;
    left_counts.resize(num_classes);
    right_counts.resize(num_classes);

    for (const auto &[label, count]: label_counts) {
        right_counts[label] = count;
    }

    int left_total = 0;
    int right_total = end - start;

    for (size_t i = 0; i < active_bins_size - 1; ++i) {
        const int bin = active_bins[i];
        const int next_bin = active_bins[i + 1];

        for (int label = 0; label < num_classes; ++label) {
            const int value = histogram[bin * num_classes + label];

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
            best_split_point = start + left_total;
        }
    }

    return std::make_tuple(best_threshold, best_impurity, best_split_point);
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
    if (params.split_criteria == "gini") {
        return gini(counts, total);
    }

    if (params.split_criteria == "entropy") {
        return entropy(counts, total);
    }

    cerr << "Unknown split criteria " << params.split_criteria << endl;
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

    int majority_class = -1, max_count = -1;
    for (const auto &[label, count]: counts) {
        if (count > max_count) {
            max_count = count;
            majority_class = label;
        }
    }
    return majority_class;
}
