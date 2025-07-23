//
// Created by gabriele on 13/07/25.
//

#include "../include/DecisionTreeClassifier.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <ranges>
#include <stack>

using namespace std;

void DecisionTreeClassifier::train(const vector<vector<double> > &X, const vector<int> &y, vector<vector<int>>& labels_mapping) {
    build_tree(X, y, labels_mapping);
}

int DecisionTreeClassifier::predict(const vector<double> &x) const {
    const TreeNode *node = root;
    while (!node->is_leaf) {
        if (x[node->feature_index] < node->threshold)
            node = node->left;
        else
            node = node->right;
    }
    return node->predicted_class;
}

void DecisionTreeClassifier::build_tree(const vector<vector<double> > &X, const vector<int> &y, vector<vector<int>>& labels_mapping) {
    root = new TreeNode();

    const int total_features = X.size();
    unordered_map<string, int> op_value = {{"sqrt", sqrt(total_features)}, {"log2", log2(total_features)}};

    stack<tuple<vector<vector<double> >, vector<int>, TreeNode *> > stack;
    stack.emplace(X, y, root);

    while (!stack.empty()) {
        auto [data_X, data_y, node] = stack.top();
        stack.pop();

        int best_feature = -1;
        double best_threshold = 0.0;
        double best_error = numeric_limits<int>::max();
        vector<vector<double> > best_left_X, best_right_X;
        vector<int> best_left_y, best_right_y;

        map<int, int> label_counts;
        for (const auto &label: data_y) label_counts[label]++;

        if (label_counts.size() == 1 || y.size() < min_samples_split) {
            node->is_leaf = true;
            node->predicted_class = compute_majority_class(label_counts);
            continue;
        }

        int n_features;
        if (std::holds_alternative<int>(max_features)) {
            n_features = std::get<int>(max_features);
        } else if (std::holds_alternative<string>(max_features)) {
            auto op = std::get<string>(max_features);
            n_features = op_value[op];
        }

        assert(n_features > 0 && "Invalid max_feature parameter");
        vector<int> selected_features = sample_features(total_features, n_features);

        for (int f: selected_features) {
            auto [threshold, impurity] = compute_treshold(data_X, data_y, f);

            if (impurity < best_error) {
                auto [left_X, right_X, left_y, right_y] =
                        split_left_right(data_X, data_y, threshold, f);

                if (!left_X.empty() && !right_X.empty()) {
                    best_error = impurity;
                    best_feature = f;
                    best_threshold = threshold;

                    best_left_X = left_X;
                    best_right_X = right_X;
                    best_left_y = left_y;
                    best_right_y = right_y;
                }
            }
        }

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

        stack.emplace(move(best_left_X), move(best_left_y), left_node);
        stack.emplace(move(best_right_X), move(best_right_y), right_node);
    }
}

auto DecisionTreeClassifier::split_left_right(const vector<vector<double> > &X,
                                              const vector<int> &y,
                                              const double th,
                                              const int f) -> SplitResult {
    int num_features = X.size();
    int num_samples = X[0].size();

    // Pre-alloca spazio vuoto per performance
    vector<int> left_indices, right_indices;

    // 1. Dividi gli indici in base alla soglia sulla feature f
    for (int i = 0; i < num_samples; ++i) {
        if (X[f][i] < th) {
            left_indices.push_back(i);
        } else {
            right_indices.push_back(i);
        }
    }

    // 2. Costruisci left_X e right_X mantenendo la struttura trasposta
    vector<vector<double> > left_X(num_features), right_X(num_features);
    for (int feat = 0; feat < num_features; ++feat) {
        left_X[feat].reserve(left_indices.size());
        right_X[feat].reserve(right_indices.size());

        for (int i: left_indices) {
            left_X[feat].push_back(X[feat][i]);
        }
        for (int i: right_indices) {
            right_X[feat].push_back(X[feat][i]);
        }
    }

    // 3. Costruisci left_y e right_y
    vector<int> left_y, right_y;
    left_y.reserve(left_indices.size());
    right_y.reserve(right_indices.size());

    for (int i: left_indices) {
        left_y.push_back(y[i]);
    }
    for (int i: right_indices) {
        right_y.push_back(y[i]);
    }

    return make_tuple(left_X, right_X, left_y, right_y);
}


pair<double, double> DecisionTreeClassifier::compute_treshold(const vector<vector<double> > &X, const vector<int> &y,
                                                              const int f) {
    const int num_samples = X[0].size();
    vector<pair<double, int> > feature_label_pairs;
    feature_label_pairs.reserve(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        feature_label_pairs.emplace_back(X[f][i], y[i]);
    }

    ranges::sort(feature_label_pairs);

    double best_threshold = 0.0;
    double best_impurity = numeric_limits<double>::max();

    map<int, int> left_counts, right_counts;

    for (const auto &label: feature_label_pairs | views::values) {
        right_counts[label]++;
    }

    int left_total = 0;
    int right_total = feature_label_pairs.size();

    for (int i = 0; i < feature_label_pairs.size() - 1; ++i) {
        const auto &[current_val, current_label] = feature_label_pairs[i];
        const auto &[next_val, next_label] = feature_label_pairs[i + 1];

        left_counts[current_label]++;
        right_counts[current_label]--;
        if (right_counts[current_label] == 0) {
            right_counts.erase(current_label);
        }
        left_total++;
        right_total--;

        if (current_val == next_val) continue;

        const double threshold = (current_val + next_val) / 2.0;

        const double gini_left = get_impurity(left_counts, left_total);
        const double gini_right = get_impurity(right_counts, right_total);
        const double weighted_impurity = (left_total * gini_left + right_total * gini_right) /
                                         (left_total + right_total);

        if (weighted_impurity < best_impurity) {
            best_impurity = weighted_impurity;
            best_threshold = threshold;
        }
    }

    return {best_threshold, best_impurity};
}

double DecisionTreeClassifier::gini(const map<int, int> &counts, const int total) {
    if (total == 0) return 0.0;

    double gini = 1.0;
    for (const auto &count: counts | views::values) {
        const double p = static_cast<double>(count) / total;
        gini -= p * p;
    }
    return gini;
}

double DecisionTreeClassifier::entropy(const std::map<int, int> &counts, const int total) {
    double entropy = 0.0;
    for (const int count: counts | std::views::values) {
        double prob = static_cast<double>(count) / total;
        if (prob > 0) {
            entropy -= prob * std::log2(prob);
        }
    }
    return entropy;
}


double DecisionTreeClassifier::get_impurity(const map<int, int> &counts, const int total) {
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

int DecisionTreeClassifier::compute_majority_class(const map<int, int> &counts) {
    int majority_class = -1, max_count = -1;
    for (const auto &[label, count]: counts) {
        if (count > max_count) {
            max_count = count;
            majority_class = label;
        }
    }
    return majority_class;
}

int DecisionTreeClassifier::compute_error(const map<int, int> &counts, const vector<int> &y_test) {
    if (y_test.empty()) return 0;
    const int majority = compute_majority_class(counts);
    int error = 0;
    for (const auto &label: y_test)
        if (label != majority) error++;
    return error;
}
