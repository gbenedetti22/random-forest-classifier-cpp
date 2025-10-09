//
// Created by gabriele on 17/09/25.
//

#ifndef DECISION_TREE_SPLITTERMP_HPP
#define DECISION_TREE_SPLITTERMP_HPP

#include "BaseSplitter.hpp"
#include "TrainMatrix.hpp"

class SplitterMP final : public BaseSplitter {
    const int nworkers;
public:
    explicit SplitterMP(const ComputeThresholdFn &compute_threshold_fn, const int nworkers)
        : BaseSplitter(compute_threshold_fn), nworkers(nworkers) {
    }

    SplitterResult find_best_split(const TrainMatrix &X, const std::vector<int> &y, std::vector<int> &indices, size_t start, size_t end,
                                   const std::vector<int> &selected_features, std::unordered_map<int, int> &label_counts,
                                   const int num_classes, const float min_samples_ratio) override {
        SplitterResult best_split;

        #pragma omp parallel num_threads(nworkers)
        {
            SplitterResult local_best;

            #pragma omp for nowait
            for (const int f : selected_features) {
                auto [threshold, impurity, split_point] = compute_threshold_fn(X, y, indices, start, end, f, label_counts, num_classes);

                if (impurity < local_best.best_impurity) {
                    size_t total_left = split_point - start;
                    size_t total_right = end - split_point;
                    const float ratio = static_cast<float>(std::min(total_left, total_right)) /
                                        static_cast<float>(end - start);

                    if (ratio > min_samples_ratio) {
                        local_best.best_impurity = impurity;
                        local_best.best_feature = f;
                        local_best.best_threshold = threshold;
                    }
                }
            }

#pragma omp critical
            {
                if (local_best.best_impurity < best_split.best_impurity) {
                    best_split.best_impurity = local_best.best_impurity;
                    best_split.best_feature = local_best.best_feature;
                    best_split.best_threshold = local_best.best_threshold;
                }
            }
        }

        return best_split;
    }
};
#endif //DECISION_TREE_SPLITTERMP_HPP