//
// Created by gabriele on 03/09/25.
//

#ifndef DECISION_TREE_TRAINMATRIX_HPP
#define DECISION_TREE_TRAINMATRIX_HPP
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <array>

class TrainMatrix {
public:
    explicit TrainMatrix(const std::vector<std::vector<float>>& X, const std::vector<int> &indices) {
        if (X.empty() || X[0].empty())
            throw std::invalid_argument("Matrice vuota!");

        const int n_samples = indices.size();
        const int n_features = X[0].size();

        data.assign(n_features, std::vector<uint8_t>(n_samples));

        minVals.resize(n_features);
        ranges.resize(n_features);
        std::vector<float> X_transposed(n_samples, 0);

        for (int f = 0; f < n_features; ++f) {
            int offset = 0;
            float min = std::numeric_limits<float>::max();
            float max = std::numeric_limits<float>::min();

            for (const int idx: indices) {
                const float value = X[idx][f];

                X_transposed[offset] = value;
                offset++;

                if (value < min) {
                    min = value;
                }

                if (value > max) {
                    max = value;
                }
            }

            minVals[f] = min;
            ranges[f] = max - min;

            const float range = std::max(max - min, 1.0f);

            for (int i = 0; i < n_samples; ++i) {
                const float norm = (X_transposed[i] - min) / range;
                const auto q = static_cast<uint8_t>(std::round(norm * 255.0f));
                data[f][i] = q;
            }
        }

    }

    [[nodiscard]] uint8_t getQuantized(const int i, const int j) const {
        return data[i][j];
    }

    uint8_t& operator()(const int i, const int j) {
        return data[i][j];
    }

    const uint8_t& operator()(const int i, const int j) const {
        return data[i][j];
    }

    [[nodiscard]] float getValue(const int i, const int j) const {
        return getApprox(i, j);
    }

    [[nodiscard]] float getApprox(const int i, const int j) const {
        const float norm = static_cast<float>(data[i][j]) / 255.0f;
        return norm * ranges[i] + minVals[i];
    }

    [[nodiscard]] float toValue(const int f, const uint8_t rawValue) const {
        const float norm = static_cast<float>(rawValue) / 255.0f;
        return norm * ranges[f] + minVals[f];
    }

    [[nodiscard]] int rows() const { return data.size(); }
    [[nodiscard]] int cols() const { return data[0].size(); }

    [[nodiscard]] int nFeatures() const { return rows(); }
    [[nodiscard]] int nSamples() const { return cols(); }

private:
    std::vector<std::vector<uint8_t>> data;
    std::vector<float> ranges;
    std::vector<float> minVals;
};
#endif //DECISION_TREE_TRAINMATRIX_HPP