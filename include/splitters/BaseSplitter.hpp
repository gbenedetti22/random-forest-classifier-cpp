#ifndef BASESPLITTER_HPP
#define BASESPLITTER_HPP

#include <functional>
#include <limits>
#include <utility>
#include <vector>
#include <unordered_map>

using ComputeThresholdFn = std::function<std::pair<float, float>(
    const std::vector<std::vector<float>> &,
    const std::vector<int> &,
    std::vector<int> &,
    int,
    std::unordered_map<int, int> &,
    int)>;

using SplitLeftRightFn = std::function<std::tuple<std::vector<int>, std::vector<int>>(
    const std::vector<std::vector<float>> &,
    const std::vector<int> &,
    float,
    int)>;

struct SplitterResult {
    float best_impurity = std::numeric_limits<float>::max();
    float best_threshold = 0.0f;
    int best_feature = -1;
    std::vector<int> left_indices;
    std::vector<int> right_indices;

    SplitterResult() = default;

    SplitterResult(float imp, float thresh, int feat, std::vector<int> left, std::vector<int> right)
        : best_impurity(imp), best_threshold(thresh), best_feature(feat),
          left_indices(std::move(left)), right_indices(std::move(right)) {}
};

class BaseSplitter {
public:
    virtual ~BaseSplitter() = default;

    ComputeThresholdFn compute_threshold_fn;
    SplitLeftRightFn split_left_right_fn;

    BaseSplitter(ComputeThresholdFn compute_threshold_fn, SplitLeftRightFn split_left_right_fn)
        : compute_threshold_fn(std::move(compute_threshold_fn)),
          split_left_right_fn(std::move(split_left_right_fn)) {}

    virtual SplitterResult find_best_split(
        const std::vector<std::vector<float>> &X,
        const std::vector<int> &y,
        std::vector<int> &indices,
        const std::vector<int> &selected_features,
        std::unordered_map<int, int> &label_counts,
        int num_classes,
        const float min_samples_ratio) = 0;
};

#endif // BASESPLITTER_HPP