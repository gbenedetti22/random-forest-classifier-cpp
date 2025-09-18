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
        const TrainMatrix &X;
        const std::vector<int> &y;
        const size_t start;
        const size_t end;
        const std::vector<int> &selected_features;
        std::unordered_map<int, int> &label_counts;
        int num_classes;
        const float min_samples_ratio;

        Worker(const SplitterFF &p,
               const TrainMatrix &X,
               const std::vector<int> &y,
               const size_t start, const size_t end,
               const std::vector<int> &selected_features,
               std::unordered_map<int, int> &label_counts,
               const int num_classes, const float min_samples_ratio) : parent(p), X(X), y(y), start(start), end(end),
                                                                       selected_features(selected_features),
                                                                       label_counts(label_counts),
                                                                       num_classes(num_classes),
                                                                       min_samples_ratio(min_samples_ratio) {
        }

        void operator()(const long i, SplitterResult &local_result) const {
            const int f = selected_features[i];

            auto [threshold, impurity, split_point] = parent.compute_threshold_fn(X, y, start, end, f, label_counts, num_classes);

            if (impurity < local_result.best_impurity) {
                const size_t total_left = split_point - start;
                const size_t total_right = end - split_point;
                const float ratio = static_cast<float>(std::min(total_left, total_right)) /
                                    static_cast<float>(end - start);

                if (ratio > min_samples_ratio) {
                    local_result = SplitterResult(impurity, threshold, f);
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
    explicit SplitterFF(const ComputeThresholdFn& compute_fn, const int workers = 4)
        : BaseSplitter(compute_fn),
          pfr(workers) {
    }

    SplitterResult find_best_split(
        const TrainMatrix &X,
        const std::vector<int> &y,
        const size_t start, const size_t end,
        const std::vector<int> &selected_features,
        std::unordered_map<int, int> &label_counts,
        const int num_classes, const float min_samples_ratio) override {
        SplitterResult final_result = identity_value;

        const Worker worker(*this, X, y, start, end, selected_features, label_counts, num_classes, min_samples_ratio);
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
