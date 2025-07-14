//
// Created by gabriele on 13/07/25.
//

#include "../include/DecisionTreeClassifier.h"

#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <stack>

using namespace std;

void DecisionTreeClassifier::train(const std::vector<Sample> &data) {
    build_tree(data);
}

int DecisionTreeClassifier::predict(const Sample &s) const {
    TreeNode* node = root;
    while (!node->is_leaf) {
        if (s.features[node->feature_index] < node->threshold)
            node = node->left;
        else
            node = node->right;
    }
    return node->predicted_class;
}

void DecisionTreeClassifier::build_tree(const std::vector<Sample> &dataset) {
    root = new TreeNode();

        const int num_features = dataset[0].features.size();

        stack<tuple<vector<Sample>, TreeNode*>> stack;
        stack.emplace(dataset, root);

        while (!stack.empty()) {
            auto [data, node] = stack.top();
            stack.pop();

            int best_feature = -1;
            double best_threshold = 0.0;
            int best_error = numeric_limits<int>::max();
            vector<Sample> best_left, best_right;

            map<int, int> label_counts;
            for (const auto&[_, label] : data) label_counts[label]++;

            if (label_counts.size() == 1) {
                node->is_leaf = true;
                node->predicted_class = compute_majority_class(label_counts);
                continue;
            }

            for (int f = 0; f < num_features; f++) {
                const double th = compute_treshold(data, f);

                auto [left, left_counts, right, right_counts] = split_left_right(data, th, f);

                int error = compute_error(left_counts, left) + compute_error(right_counts, right);

                if (error < best_error && !left.empty() && !right.empty()) {
                    best_error = error;
                    best_feature = f;
                    best_threshold = th;

                    best_left = left;
                    best_right = right;
                }
            }

            if (best_feature == -1) {
                node->is_leaf = true;
                node->predicted_class = compute_majority_class(label_counts);
                continue;
            }

            auto* left_node = new TreeNode();
            auto* right_node = new TreeNode();

            node->is_leaf = false;
            node->feature_index = best_feature;
            node->threshold = best_threshold;
            node->left = left_node;
            node->right = right_node;

            stack.emplace(std::move(best_left), left_node);
            stack.emplace(std::move(best_right), right_node);
        }
}


auto DecisionTreeClassifier::split_left_right(const vector<Sample> &data, const double th, const int f)->SplitResult {
    vector<Sample> left, right;
    map<int, int> left_counts, right_counts;
    for (const Sample& s : data) {
        if (s.features[f] < th) {
            left.push_back(s);
            left_counts[s.label]++;
        } else {
            right.push_back(s);
            right_counts[s.label]++;
        }
    }

    return make_tuple(left, left_counts, right, right_counts);
}

double DecisionTreeClassifier::compute_treshold(const vector<Sample>& data, const int i) {
    double sum = 0;
    for (const Sample& s : data) {
        sum += s.features[i];
    }

    return sum / data.size();
}

int DecisionTreeClassifier::compute_majority_class(const map<int, int>& counts) {
    int majority_class = -1, max_count = -1;
    for (const auto& [label, count] : counts) {
        if (count > max_count) {
            max_count = count;
            majority_class = label;
        }
    }
    return majority_class;
}

int DecisionTreeClassifier::compute_error(const map<int, int>& counts, const vector<Sample>& test) {
    if (test.empty()) return 0;
    const int majority = compute_majority_class(counts);
    int error = 0;
    for (const auto&[features, label] : test)
        if (label != majority) error++;
    return error;
}