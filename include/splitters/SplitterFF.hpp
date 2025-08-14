#ifndef SPLITTER_FF_H
#define SPLITTER_FF_H

#include <utility>
#include <functional>
#include <tuple>

#include <ff/ff.hpp>
#include <ff/parallel_for.hpp>

#include "BaseSplitter.hpp"

class SplitterFF final : public BaseSplitter {
    ff::ParallelForReduce<SplitterResult> pfr;
    const SplitterResult identity_value;

    struct Worker {
        const SplitterFF &parent;
        const std::vector<std::vector<float> > &X;
        const std::vector<int> &y;
        const std::vector<int> &indices;
        const std::vector<int> &selected_features;
        std::unordered_map<int, int> &label_counts;
        int num_classes;
        const float min_samples_ratio;

        Worker(const SplitterFF &p,
               const std::vector<std::vector<float> > &X,
               const std::vector<int> &y,
               const std::vector<int> &indices,
               const std::vector<int> &selected_features,
               std::unordered_map<int, int> &label_counts,
               const int num_classes, const float min_samples_ratio) : parent(p), X(X), y(y), indices(indices),
                                        selected_features(selected_features),
                                        label_counts(label_counts), num_classes(num_classes), min_samples_ratio(min_samples_ratio) {
        }

        void operator()(const long i, SplitterResult &local_result) const {
            const int f = selected_features[i];

            std::vector<int> thread_local_indices = indices;
            auto [threshold, impurity] = parent.compute_threshold_fn(
                X, y, thread_local_indices, f, label_counts, num_classes
            );
            if (impurity < local_result.best_impurity) {
                auto [left_X, right_X] = parent.split_left_right_fn(X, thread_local_indices, threshold, f);

                const float ratio = static_cast<float>(std::min(left_X.size(), right_X.size())) /
                                    static_cast<float>(indices.size());

                if (ratio > min_samples_ratio) {
                    local_result = SplitterResult(impurity, threshold, f, move(left_X), move(right_X));
                }
            }
        }
    };

    struct Reducer {
        SplitterResult &operator()(SplitterResult &s1, const SplitterResult &s2) const {
            if (s2.best_impurity < s1.best_impurity) {
                s1 = s2;
            }
            return s1;
        }
    };

public:
    SplitterFF(ComputeThresholdFn compute_fn, SplitLeftRightFn split_fn, const int workers = 4)
        : BaseSplitter(std::move(compute_fn), std::move(split_fn)),
          pfr(workers) {
    }

    SplitterResult find_best_split(
        const std::vector<std::vector<float> > &X,
        const std::vector<int> &y,
        std::vector<int> &indices,
        const std::vector<int> &selected_features,
        std::unordered_map<int, int> &label_counts,
        const int num_classes, const float min_samples_ratio) override {
        SplitterResult final_result = identity_value;

        const Worker worker(*this, X, y, indices, selected_features, label_counts, num_classes, min_samples_ratio);
        constexpr Reducer reducer;

        pfr.parallel_reduce(
            final_result,
            identity_value,
            0, selected_features.size(),
            worker,
            reducer
        );

        return final_result;
    }
};

#endif // SPLITTER_FF_H
