#ifndef SPLITTERMP_HPP
#define SPLITTERMP_HPP

#include <vector>
#include <tuple>
#include <limits>
#include <unordered_map>
#include <functional>
#include <omp.h>

struct Candidate {
    float error;
    int feature;
    float threshold;
    std::vector<int> left_X;
    std::vector<int> right_X;
    bool valid;
};

#pragma omp declare reduction(merge_candidate : Candidate : \
    omp_out = (omp_in.valid && (!omp_out.valid || omp_in.error < omp_out.error)) ? omp_in : omp_out \
) initializer(omp_priv = Candidate{std::numeric_limits<float>::max(), -1, 0.0f, {}, {}, false})

class SplitterMP {
public:
    float best_error;
    int best_feature;
    float best_threshold;
    std::vector<int> best_left_X;
    std::vector<int> best_right_X;

    using ComputeThresholdFn = std::function<std::pair<float, float>(
        const std::vector<std::vector<float> > &,
        const std::vector<int> &,
        std::vector<int> &,
        int,
        std::unordered_map<int, int> &,
        int)>;

    using SplitLeftRightFn = std::function<std::tuple<std::vector<int>, std::vector<int> >(
        const std::vector<std::vector<float> > &,
        const std::vector<int> &,
        float,
        int)>;

    SplitterMP(const std::vector<int> &selected_features,
               const std::vector<std::vector<float> > &X,
               const std::vector<int> &y,
               const std::vector<int> &indices,
               std::unordered_map<int, int> &label_counts,
               int num_classes,
               ComputeThresholdFn compute_threshold_fn,
               SplitLeftRightFn split_left_right_fn)
        : selected_features(selected_features),
          X(X), y(y), indices(indices),
          label_counts(label_counts), num_classes(num_classes),
          compute_threshold_fn(std::move(compute_threshold_fn)),
          split_left_right_fn(std::move(split_left_right_fn)) {
        resetResults();
    }

    void run(int nthreads = 4) {
        omp_set_nested(1);

        #pragma omp parallel for num_threads(nthreads) schedule(dynamic) \
        firstprivate(indices) \
        shared(best_error, best_feature, best_threshold, best_left_X, best_right_X)
        for (const int f : selected_features) {
            auto [threshold, impurity] = compute_threshold_fn(X, y, indices, f, label_counts, num_classes);

            if (impurity < best_error) {
                auto [left_X, right_X] = split_left_right_fn(X, indices, threshold, f);

                float ratio = static_cast<float>(std::min(left_X.size(), right_X.size())) /
                              static_cast<float>(indices.size());

                if (ratio > 0.2f) {
        #pragma omp critical
                    {
                        if (impurity < best_error) {
                            best_error = impurity;
                            best_feature = f;
                            best_threshold = threshold;
                            best_left_X = left_X;
                            best_right_X = right_X;
                        }
                    }
                }
            }
        }
    }

private:
    void resetResults() {
        best_error = std::numeric_limits<float>::max();
        best_feature = -1;
        best_threshold = 0.0f;
        best_left_X.clear();
        best_right_X.clear();
    }

    const std::vector<int> &selected_features;
    const std::vector<std::vector<float> > &X;
    const std::vector<int> &y;
    std::vector<int> indices;
    std::unordered_map<int, int> &label_counts;
    int num_classes;

    ComputeThresholdFn compute_threshold_fn;
    SplitLeftRightFn split_left_right_fn;
};

#endif // SPLITTERMP_HPP
