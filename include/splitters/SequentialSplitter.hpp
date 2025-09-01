//
// Created by gabriele on 14/08/25.
//

#ifndef SEQUENTIALSPLITTER_HPP
#define SEQUENTIALSPLITTER_HPP
#include "BaseSplitter.hpp"
#include "Timer.h"
using namespace std;

class SequentialSplitter final : public BaseSplitter {
public:
    SequentialSplitter(const ComputeThresholdFn &compute_threshold_fn, const SplitLeftRightFn &split_left_right_fn)
        : BaseSplitter(compute_threshold_fn, split_left_right_fn) {
    }

    SplitterResult find_best_split(const vector<vector<float>> &X, const vector<int> &y, vector<int> &indices,
                                   const vector<int> &selected_features, unordered_map<int, int> &label_counts,
                                   const int num_classes, const float min_samples_ratio) override {
        SplitterResult best_split;
        for (const int f: selected_features) {
            timer.start("threshold");
            auto [threshold, impurity] = compute_threshold_fn(X, y, indices, f, label_counts, num_classes);
            timer.stop("threshold");

            if (impurity < best_split.best_impurity) {
                timer.start("split");
                auto [left_X, right_X] = split_left_right_fn(X, indices, threshold, f);
                timer.stop("split");

                const float ratio = static_cast<float>(min(left_X.size(), right_X.size())) /
                                    static_cast<float>(indices.size());

                // ratio = min_samples_leaf (scikit learn)
                if (ratio > min_samples_ratio) {
                    best_split.best_impurity = impurity;
                    best_split.best_feature = f;
                    best_split.best_threshold = threshold;

                    best_split.left_indices = left_X;
                    best_split.right_indices = right_X;
                }
            }
        }

        return best_split;
    }
};
#endif //SEQUENTIALSPLITTER_HPP
