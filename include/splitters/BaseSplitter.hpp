#ifndef BASESPLITTER_HPP
#define BASESPLITTER_HPP

#include <functional>
#include <limits>
#include <utility>
#include <vector>
#include <unordered_map>

#include "utils.h"

using ComputeThresholdFn = std::function<std::tuple<float, float, size_t>(
    const TrainMatrix &,
    const std::vector<int> &,
    std::vector<int> &,
    int,
    std::unordered_map<int, int> &,
    int)>;

using SplitLeftRightFn = std::function<std::tuple<std::vector<int>, std::vector<int>>(
    const TrainMatrix &,
    const std::vector<int> &,
    float,
    int)>;

struct SplitterResult {
    float best_impurity = std::numeric_limits<float>::max();
    float best_threshold = 0.0f;
    int best_feature = -1;

    SplitterResult() = default;
};

class BaseSplitter {
public:
    virtual ~BaseSplitter() = default;

    ComputeThresholdFn compute_threshold_fn;

    explicit BaseSplitter(ComputeThresholdFn compute_threshold_fn)
        : compute_threshold_fn(std::move(compute_threshold_fn)) {}

    virtual SplitterResult find_best_split(
        const TrainMatrix &X,
        const std::vector<int> &y,
        std::vector<int> &indices,
        const std::vector<int> &selected_features,
        std::unordered_map<int, int> &label_counts,
        int num_classes,
        float min_samples_ratio) = 0;
};

#endif // BASESPLITTER_HPP