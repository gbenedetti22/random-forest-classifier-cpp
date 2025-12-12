//
// Created by gabriele on 03/09/25.
//

#ifndef DECISION_TREE_TRAINMATRIX_HPP
#define DECISION_TREE_TRAINMATRIX_HPP
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "pdqsort.h"
#include "Timer.h"

// Wrapper around the feature matrix optimized for memory and cache efficiency.
// Uses quantization (mapping float values to 0-255 uint8) to speed up histogram calculation during training.
class TrainMatrix {
public:
    // computation of global min and max for quantization range.
    explicit TrainMatrix(const std::vector<float>& X, const std::pair<size_t, size_t>& shape)
        : X(X), n_samples(shape.first), n_features(shape.second) {
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::min();

        for (const float value: X) {
            if (value < min) min = value;
            if (value > max) max = value;
        }

        global_min = min;
        range = max - min;
    }

    // Returns the quantized 8-bit representation of the value at (i, j).
    // Maps the float value to a bin in [0, 255] based on the feature's range.
    [[nodiscard]] uint8_t getQuantized(const size_t i, const size_t j) const {
        const float norm = (X[i * n_samples + j] - global_min) / range;
        const auto q = static_cast<uint8_t>(std::round(norm * 255.0f));
        return q;
    }

    const float& operator()(const size_t i, const size_t j) const {
        return X[i * n_samples + j];
    }

    [[nodiscard]] float getValue(const size_t i, const size_t j) const {
        return getApprox(i, j);
    }

    // Reconstructs the approximate float value from the quantized representation.
    // Used when retrieving split thresholds from bins.
    [[nodiscard]] float getApprox(const size_t i, const size_t j) const {
        const float norm = static_cast<float>(X[i * n_samples + j]) / 255.0f;
        return norm * range + global_min;
    }

    // Converts a raw uint8 bin value back to the feature's float scale.
    [[nodiscard]] float toValue(const uint8_t rawValue) const {
        const float norm = static_cast<float>(rawValue) / 255.0f;
        return norm * range + global_min;
    }

    [[nodiscard]] size_t rows() const { return n_samples; }
    [[nodiscard]] size_t cols() const { return n_features; }

    [[nodiscard]] size_t nFeatures() const { return n_features; }
    [[nodiscard]] size_t nSamples() const { return n_samples; }

private:
    const std::vector<float>& X;
    float range;
    float global_min;

    const size_t n_samples;
    const size_t n_features;
};
#endif //DECISION_TREE_TRAINMATRIX_HPP