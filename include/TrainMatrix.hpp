//
// Created by gabriele on 03/09/25.
//

#ifndef DECISION_TREE_TRAINMATRIX_HPP
#define DECISION_TREE_TRAINMATRIX_HPP
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

class TrainMatrix {
public:
    TrainMatrix(const int rows, const int cols) {
        data.assign(rows, std::vector<uint8_t>(cols));
    }

    explicit TrainMatrix(const std::vector<std::vector<float>>& X, const std::vector<int> &indices) {
        if (X.empty() || X[0].empty())
            throw std::invalid_argument("Matrice vuota!");

        const int n_samples = indices.size();
        const int n_features = X[0].size();

        std::vector X_transposed(n_features, std::vector<float>(n_samples));
        data.assign(n_features, std::vector<uint8_t>(n_samples));

        for (int i = 0; i < n_samples; ++i) {
            const int idx = indices[i];

            for (int f = 0; f < n_features; ++f) {
                X_transposed[f][i] = X[idx][f];
            }
        }

        minVals.resize(n_features);
        maxVals.resize(n_features);

        int f_index = 0;
        for (auto f = X_transposed.begin(); f != X_transposed.end();) {
            auto [min, max] = std::ranges::minmax(*f);

            minVals[f_index] = min;
            maxVals[f_index] = max;

            const float range = std::max(max - min, 1.0f);

            for (int i = 0; i < n_samples; ++i) {
                const float norm = ((*f)[i] - min) / range;
                const auto q = static_cast<uint8_t>(std::round(norm * 255.0f));
                data[f_index][i] = q;
            }

            f = X_transposed.erase(f);
            f_index++;
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

    // Method to get dequantized value for threshold comparisons
    [[nodiscard]] float getValue(const int i, const int j) const {
        return getApprox(i, j);
    }

    [[nodiscard]] float getApprox(const int i, const int j) const {
        const float minVal = minVals[i];
        const float maxVal = maxVals[i];
        const float norm = static_cast<float>(data[i][j]) / 255.0f;
        return norm * (maxVal - minVal) + minVal;
    }

    [[nodiscard]] int rows() const { return data.size(); }
    [[nodiscard]] int cols() const { return data[0].size(); }

    [[nodiscard]] int nFeatures() const { return rows(); }
    [[nodiscard]] int nSamples() const { return cols(); }

private:
    std::vector<std::vector<uint8_t>> data;
    std::vector<float> minVals;
    std::vector<float> maxVals;
};
#endif //DECISION_TREE_TRAINMATRIX_HPP