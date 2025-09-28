//
// Created by gabriele on 15/08/25.
//

#ifndef UTILS_H
#define UTILS_H
#include <chrono>
#include <fstream>
#include <execution>
#include <random>

static std::vector<float> flatten(const std::vector<std::vector<float> > &X, const size_t totalSize) {
    std::vector<float> result;
    result.reserve(totalSize);

    for (auto& v : X) {
        if (!v.empty()) {
            result.insert(result.end(),
                          std::make_move_iterator(v.begin()),
                          std::make_move_iterator(v.end()));
        }
    }

    return result;
}
static std::vector<float> flatten(const std::vector<std::vector<float> > &X) {
    return flatten(X, X.size() * X[0].size());
}

static void writeToFile(const std::string& filename,
                        const std::string& content,
                        const bool append = true)
{
    std::ios_base::openmode mode = std::ios::out;
    mode |= append ? std::ios::app : std::ios::trunc;

    std::ofstream file(filename, mode);
    if (!file) {
        throw std::runtime_error("Impossibile aprire o creare il file: " + filename);
    }

    file << content << std::endl;
    if (!file) {
        throw std::runtime_error("Errore durante la scrittura nel file: " + filename);
    }
}

static void writeToFile(const std::string& filename, const int value, const bool append = true) {
    writeToFile(filename, std::to_string(value), append);
}

static void writeToFile(const std::string& filename, const long value, const bool append = true) {
    writeToFile(filename, std::to_string(value), append);
}

static void writeToFile(const std::string& filename, const float value, const bool append = true) {
    writeToFile(filename, std::to_string(value), append);
}

static void writeToFile(const std::string& filename, const double value, const bool append = true) {
    writeToFile(filename, std::to_string(value), append);
}

static void writeToFile(const std::string& filename, const char value, const bool append = true) {
    writeToFile(filename, std::string(1, value), append);
}

static void writeToFile(const std::string& filename,
                            const std::chrono::steady_clock::duration& dur,
                            const bool append = true)
{
    using namespace std::chrono;

    const auto h = duration_cast<hours>(dur);
    const auto m = duration_cast<minutes>(dur - h);
    const auto s = duration_cast<seconds>(dur - h - m);

    const std::string formatted = std::to_string(h.count()) + "h " +
                            std::to_string(m.count()) + "m " +
                            std::to_string(s.count()) + "s";

    writeToFile(filename, formatted, append);
}

static double now() {
    using namespace std::chrono;
    const auto tp = steady_clock::now();
    const auto dur = tp.time_since_epoch();
    return duration_cast<duration<double>>(dur).count();
}

static std::pair<std::vector<float>, std::vector<int>> generate_matrix(
    const long n_features,
    const long long bytes,
    const int random_seed = 42
) {
    const size_t rows = bytes / (sizeof(float) * n_features);
    const size_t cols = n_features;

    if (rows <= 0) {
        throw std::invalid_argument("Insufficient bytes for the given number of features");
    }

    std::vector<float> matrix(rows * cols, 0);
    std::vector<int> labels(rows);

    std::vector<float> coefficients(n_features);
    {
        std::mt19937 gen(random_seed);
        std::uniform_real_distribution coeff_dist(-2.0f, 2.0f);
        for (long i = 0; i < n_features; ++i) {
            coefficients[i] = coeff_dist(gen);
        }
    }

    std::vector<long> indices(rows);
    iota(indices.begin(), indices.end(), 0);

    for_each(std::execution::par_unseq, indices.begin(), indices.end(),
        [&](const long i) {
            thread_local std::mt19937 gen(random_seed ^ i);

            thread_local std::normal_distribution feature_dist(0.0f, 1.0f);
            thread_local std::normal_distribution noise_dist(0.0f, 0.1f);
            thread_local std::uniform_int_distribution<long> feature_idx_dist(0, std::max(1L, std::min(n_features/2 - 1, n_features - 3)));

            float linear_combination = 0.0f;

            for (long j = 0; j < n_features; ++j) {
                matrix[j * rows + i] = feature_dist(gen);

                linear_combination += coefficients[j] * matrix[j * rows + i];
            }

            linear_combination += noise_dist(gen);

            labels[i] = linear_combination > 0.0f ? 1 : 0;

            if (n_features >= 2) {
                if (labels[i] == 1) {
                    matrix[0 * rows + i] += 0.5f;
                    matrix[1 * rows + i] -= 0.3f;
                } else {
                    matrix[0 * rows + i] -= 0.5f;
                    matrix[1 * rows + i] += 0.3f;
                }
            }

            if (n_features >= 4) {
                long base_idx = feature_idx_dist(gen);
                if (base_idx + 2 < n_features) {
                    matrix[(n_features - 2) * rows + i] = 0.7f * matrix[base_idx * rows + i] +
                                               0.3f * matrix[base_idx * rows + i] +
                                               noise_dist(gen);

                }
                if (n_features >= 3) {
                    matrix[(n_features - 1) * rows + i] = 0.5f * matrix[0 * rows + i] + noise_dist(gen);
                }
            }

        }
    );

    return std::pair{matrix, labels};
}

inline std::pair<size_t, size_t> transpose(const std::vector<std::vector<float>>& src, std::vector<float>& dst, const size_t block_size = 128) {
    const size_t rows = src.size();
    const size_t cols = src[0].size();
    dst.resize(rows * cols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i += block_size) {
        for (size_t j = 0; j < cols; j += block_size) {
            const size_t max_i = std::min(i + block_size, rows);
            const size_t max_j = std::min(j + block_size, cols);

            for (size_t ii = i; ii < max_i; ++ii) {
                for (size_t jj = j; jj < max_j; ++jj) {
                    dst[jj * rows + ii] = src[ii][jj];
                }
            }
        }
    }

    return std::pair{src.size(), src[0].size()};
}

#endif //UTILS_H
