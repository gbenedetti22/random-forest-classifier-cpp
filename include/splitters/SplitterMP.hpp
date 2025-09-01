//
// Created by gabriele on 14/08/25.
//

#ifndef SPLITTERMP_HPP
#define SPLITTERMP_HPP
#include "BaseSplitter.hpp"
#include "Timer.h"
using namespace std;

class SplitterMP : public BaseSplitter {
    int nworkers;

public:
    SplitterMP(const ComputeThresholdFn &compute_threshold_fn, const SplitLeftRightFn &split_left_right_fn,
               const int workers = 4)
        : BaseSplitter(compute_threshold_fn, split_left_right_fn), nworkers(workers) {
    }

    SplitterResult find_best_split(const vector<vector<float>> &X, const vector<int> &y, vector<int> &indices,
                                   const vector<int> &selected_features, unordered_map<int, int> &label_counts,
                                   const int num_classes, const float min_samples_ratio) override {
        SplitterResult best_split;

#pragma omp parallel num_threads(nworkers)
        {
            SplitterResult local_best;
            vector<int> local_indices = indices;

#pragma omp for nowait
            for (const int f : selected_features) {
                auto [threshold, impurity] = compute_threshold_fn(X, y, local_indices, f, label_counts, num_classes);

                if (impurity < local_best.best_impurity) {
                    auto [left_X, right_X] = split_left_right_fn(X, local_indices, threshold, f);

                    const float ratio = static_cast<float>(std::min(left_X.size(), right_X.size())) /
                                        static_cast<float>(local_indices.size());

                    if (ratio > min_samples_ratio) {
                        local_best.best_impurity = impurity;
                        local_best.best_feature = f;
                        local_best.best_threshold = threshold;

                        local_best.left_indices = left_X;
                        local_best.right_indices = right_X;
                    }
                }
            }

#pragma omp critical
            {
                if (local_best.best_impurity < best_split.best_impurity) {
                    best_split.best_impurity = local_best.best_impurity;
                    best_split.best_feature = local_best.best_feature;
                    best_split.best_threshold = local_best.best_threshold;

                    best_split.left_indices = local_best.left_indices;
                    best_split.right_indices = local_best.right_indices;
                }
            }
        }

        return best_split;
    }
};
#endif
