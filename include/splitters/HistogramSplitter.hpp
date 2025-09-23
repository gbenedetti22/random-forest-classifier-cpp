//
// Created by gabriele on 14/08/25.
//

#ifndef HISTOGRAMSPLITTER_HPP
#define HISTOGRAMSPLITTER_HPP
#include <filesystem>
#include <iostream>  // For logging

#include "BaseSplitter.hpp"
#include "Timer.h"
#include "pdqsort.h"

class HistogramSplitter final : public BaseSplitter {
public:
    explicit HistogramSplitter(const ComputeThresholdFn &compute_threshold_fn)
        : BaseSplitter(compute_threshold_fn) {
    }

    SplitterResult find_best_split(const TrainMatrix &X, const std::vector<int> &y, std::vector<int> &indices, size_t start, size_t end,
                                   const std::vector<int> &selected_features, std::unordered_map<int, int> &label_counts,
                                   const int num_classes, const float min_samples_ratio) override {
        SplitterResult best_split;
        for (const int f: selected_features) {
            auto [threshold, impurity, split_point] = compute_threshold_fn(X, y, indices, start, end, f, label_counts, num_classes);

            if (impurity < best_split.best_impurity) {
                size_t total_left = split_point - start;
                size_t total_right = end - split_point;
                const float ratio = static_cast<float>(std::min(total_left, total_right)) /
                                    static_cast<float>(end - start);

                // ratio = min_samples_leaf (scikit learn)
                if (ratio > min_samples_ratio) {
                    best_split.best_feature = f;
                    best_split.best_impurity = impurity;
                    best_split.best_threshold = threshold;
                }
            }
        }

        return best_split;
    }
};

#endif //HISTOGRAMSPLITTER_HPP
