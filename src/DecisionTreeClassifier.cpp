//
// Created by gabriele on 13/07/25.
//

#include "../include/DecisionTreeClassifier.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <random>
#include <ranges>
#include <stack>
#include <tuple>
#include "../include/pdqsort.h"
#include "../include/RadixSort.h"
#include "../include/Timer.h"

using namespace std;

void DecisionTreeClassifier::train(const vector<vector<double> > &X, const vector<int> &y, vector<int>& samples) {
    build_tree(X, y, samples);
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
int skipped_cycles = 0, total_cycles = 0;

void DecisionTreeClassifier::build_tree(const vector<vector<double> > &X, const vector<int> &y, vector<int>& samples) {
    root = new TreeNode();

    const int total_features = X.size();
    unordered_map<string, int> op_value = {{"sqrt", sqrt(total_features)}, {"log2", log2(total_features)}};

    stack<tuple<vector<int>, TreeNode *> > stack;
    stack.emplace(samples, root);

    while (!stack.empty()) {
        auto [indices, node] = std::move(stack.top());
        stack.pop();

        int best_feature = -1;
        double best_threshold = 0.0;
        double best_error = numeric_limits<int>::max();
        vector<int> best_left_X, best_right_X;

        timer.start("label counts");
        unordered_map<int, int> label_counts;
        // for (const auto &label: data_y) label_counts[label]++;
        for (const int i : indices) {
            label_counts[y[i]]++;
        }
        timer.stop("label counts");

        if (label_counts.size() == 1 || indices.size() < min_samples_split) {
            node->is_leaf = true;
            node->predicted_class = compute_majority_class(label_counts);
            continue;
        }

        int n_features = 0;
        if (std::holds_alternative<int>(max_features)) {
            n_features = std::get<int>(max_features);
        } else if (std::holds_alternative<string>(max_features)) {
            auto op = std::get<string>(max_features);
            n_features = op_value[op];
        }

        assert(n_features > 0 && "Invalid max_feature parameter");
        timer.start("sample features");
        vector<int> selected_features = sample_features(total_features, n_features);
        timer.stop("sample features");

        for (const int f: selected_features) {
            auto [threshold, impurity] = compute_treshold(X, y, indices, f);

            if (impurity < best_error) {
                timer.start("split");
                auto [left_X, right_X] = split_left_right(X, indices, threshold, f);
                timer.stop("split");

                if (!left_X.empty() && !right_X.empty()) {
                    best_error = impurity;
                    best_feature = f;
                    best_threshold = threshold;

                    best_left_X = left_X;
                    best_right_X = right_X;
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

        stack.emplace(move(best_left_X), left_node);
        stack.emplace(move(best_right_X), right_node);
    }

    std::ios_base::fmtflags oldFlags = std::cout.flags();
    std::streamsize oldPrecision = std::cout.precision();

    const double perc = (total_cycles - skipped_cycles) / static_cast<double>(total_cycles) * 100;
    cout << "Percentage of completed cycles during threshold: " << fixed << setprecision(3) << perc  << "%" << endl;
    cout << "Skipped Cycles: " << skipped_cycles << " over " << total_cycles << endl;

    std::cout.flags(oldFlags);
    std::cout.precision(oldPrecision);
}

auto DecisionTreeClassifier::split_left_right(const vector<vector<double> > &X,
                                              const vector<int> &indices,
                                              const double th,
                                              const int f) -> SplitResult {
    vector<int> left_indices, right_indices;

    for (int i : indices) {
        if (X[f][i] < th) {
            left_indices.push_back(i);
        } else {
            right_indices.push_back(i);
        }
    }

    return make_tuple(left_indices, right_indices);
}

pair<double, double> DecisionTreeClassifier::compute_treshold(const vector<vector<double>> &X, const vector<int> &y,
                                                              vector<int> &indices, const int f) const {
    const int num_samples = indices.size();
    vector<pair<double, int>> feature_label_pairs;
    feature_label_pairs.reserve(num_samples);

    int num_classes = 0;
    for (const int i : indices) {
        feature_label_pairs.emplace_back(X[f][i], y[i]);
        if (y[i] > num_classes) {
            num_classes = y[i];
        }
    }

    timer.start("treshold: sorting");

    pdqsort_branchless(feature_label_pairs.begin(), feature_label_pairs.end());

    timer.stop("treshold: sorting");

    double best_threshold = 0.0;
    double best_impurity = numeric_limits<double>::max();

    vector<int> left_counts, right_counts;
    left_counts.resize(num_classes);
    right_counts.resize(num_classes);

    for (const auto &label: feature_label_pairs | views::values) {
        right_counts[label]++;
    }

    int left_total = 0;
    int right_total = feature_label_pairs.size();

    timer.start("treshold: main");
    for (int i = 0; i < feature_label_pairs.size() - 1; ++i) {
        const auto &[current_val, current_label] = feature_label_pairs[i];
        const auto &[next_val, next_label] = feature_label_pairs[i + 1];
        total_cycles++;

        left_counts[current_label]++;
        right_counts[current_label]--;
        left_total++;
        right_total--;

        if (current_val == next_val) {
            skipped_cycles++;
            continue;
        }

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
    timer.stop("treshold: main");

    return {best_threshold, best_impurity};
}

double DecisionTreeClassifier::gini(const vector<int> &counts, const int total) {
    if (total == 0) return 0.0;

    double gini = 1.0;
    for (const int count : counts) {
        const double p = static_cast<double>(count) / total;
        gini -= p * p;
    }
    return gini;
}

double DecisionTreeClassifier::entropy(const vector<int> &counts, const int total) {
    double entropy = 0.0;
    for (const int count: counts) {
        const double prob = static_cast<double>(count) / total;
        if (prob > 0) {
            entropy -= prob * std::log2(prob);
        }
    }
    return entropy;
}


double DecisionTreeClassifier::get_impurity(const vector<int> &counts, const int total) const {
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

int DecisionTreeClassifier::compute_error(const unordered_map<int, int> &counts, const vector<int> &y_test) {
    if (y_test.empty()) return 0;
    const int majority = compute_majority_class(counts);
    int error = 0;
    for (const auto &label: y_test)
        if (label != majority) error++;
    return error;
}
