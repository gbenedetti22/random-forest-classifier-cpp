//
// Created by gabriele on 15/08/25.
//

#ifndef UTILS_H
#define UTILS_H
#include <Eigen/Core>
#include <chrono>
using ColMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

inline std::vector<float> flatten(const std::vector<std::vector<float> > &X) {
    std::vector<float> result;
    result.reserve(X.size() * X[0].size());

    for (const auto &v: X) {
        result.insert(result.end(), v.begin(), v.end());
    }

    return result;
}

inline double now() {
    using namespace std::chrono;
    const auto tp = steady_clock::now();
    const auto dur = tp.time_since_epoch();
    return duration_cast<duration<double>>(dur).count();
}
#endif //UTILS_H
