//
// Created by gabriele on 15/08/25.
//

#ifndef UTILS_H
#define UTILS_H
#include <Eigen/Core>
using ColMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

inline std::vector<float> flatten(const std::vector<std::vector<float> > &X) {
    std::vector<float> result;
    result.reserve(X.size() * X[0].size());

    for (const auto &v: X) {
        result.insert(result.end(), v.begin(), v.end());
    }

    return result;
}
#endif //UTILS_H
