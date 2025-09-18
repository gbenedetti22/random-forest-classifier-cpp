//
// Created by gabriele on 03/09/25.
//

#ifndef DECISION_TREE_TRAINMATRIX_HPP
#define DECISION_TREE_TRAINMATRIX_HPP
#include <algorithm>
#include <cmath>
#include <vector>

#include "pdqsort.h"
#include "Timer.h"

class TrainMatrix {
public:
    explicit TrainMatrix(const std::vector<float>& X, const std::pair<size_t, size_t> &shape, std::vector<int> &indices)
    :  n_samples(shape.first), n_features(shape.second) {
        data.resize(n_samples * n_features, 0);

        pdqsort_branchless(indices.begin(), indices.end());

        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::min();

        for (const float value : X) {
            if (value < min) min = value;
            if (value > max) max = value;
        }

        global_min = min;
        range = max - min;

        for (size_t f = 0; f < n_features; ++f) {
            int offset = 0;

            for (const int idx: indices) {
                const float value = X[idx * n_features + f];

                const float norm = (value - min) / range;
                const auto q = static_cast<uint8_t>(std::round(norm * 255.0f));
                data[f * n_samples + offset] = q;
                offset++;
            }
        }
    }

    [[nodiscard]] uint8_t getQuantized(const size_t i, const size_t j) const {
        return data[i * n_samples + j];
    }

    uint8_t& operator()(const size_t i, const size_t j) {
        return data[i * n_samples + j];
    }

    const uint8_t& operator()(const size_t i, const size_t j) const {
        return data[i * n_samples + j];
    }

    [[nodiscard]] float getValue(const size_t i, const size_t j) const {
        return getApprox(i, j);
    }

    [[nodiscard]] float getApprox(const size_t i, const size_t j) const {
        const float norm = static_cast<float>(data[i * n_samples + j]) / 255.0f;
        return norm * range + global_min;
    }

    [[nodiscard]] float toValue(const uint8_t rawValue) const {
        const float norm = static_cast<float>(rawValue) / 255.0f;
        return norm * range + global_min;
    }

    [[nodiscard]] size_t rows() const { return n_features; }
    [[nodiscard]] size_t cols() const { return n_samples; }

    [[nodiscard]] size_t nFeatures() const { return n_features; }
    [[nodiscard]] size_t nSamples() const { return n_samples; }

private:
    std::vector<uint8_t> data;
    float range;
    float global_min;

    const size_t n_samples;
    const size_t n_features;
};
#endif //DECISION_TREE_TRAINMATRIX_HPP