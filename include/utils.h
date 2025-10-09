//
// Created by gabriele on 15/08/25.
//

#ifndef UTILS_H
#define UTILS_H
#include <cassert>
#include <chrono>
#include <fstream>
#include <execution>
#include <random>

#include "cxxopts.hpp"
#include "Logger.hpp"
#include "RandomForestClassifier.h"


static std::vector<float> flatten(const std::vector<std::vector<float> > &X, const size_t totalSize) {
    std::vector<float> result;
    result.reserve(totalSize);

    for (auto &v: X) {
        if (!v.empty()) {
            result.insert(result.end(),
                          std::make_move_iterator(v.begin()),
                          std::make_move_iterator(v.end()));
        }
    }

    return std::move(result);
}

static std::vector<float> flatten(const std::vector<std::vector<float> > &X) {
    return std::move(flatten(X, X.size() * X[0].size()));
}

static double now() {
    using namespace std::chrono;
    const auto tp = steady_clock::now();
    const auto dur = tp.time_since_epoch();
    return duration_cast<duration<double> >(dur).count();
}

static std::pair<std::vector<float>, std::vector<int> > generate_matrix(
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

    std::vector<float> coefficients(n_features); {
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
                 thread_local std::uniform_int_distribution<long> feature_idx_dist(
                     0, std::max(1L, std::min(n_features / 2 - 1, n_features - 3)));

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

inline std::pair<size_t, size_t> transpose(const std::vector<std::vector<float> > &src, std::vector<float> &dst,
                                           const size_t block_size = 128) {
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

inline std::pair<size_t, size_t> transpose(const std::vector<float> &src,
                                           std::vector<float> &dst,
                                           const std::pair<size_t, size_t> &shape,
                                           const size_t block_size = 128) {
    const size_t rows = shape.first;
    const size_t cols = shape.second;
    dst.resize(rows * cols);

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i += block_size) {
        for (size_t j = 0; j < cols; j += block_size) {
            const size_t max_i = std::min(i + block_size, rows);
            const size_t max_j = std::min(j + block_size, cols);

            for (size_t ii = i; ii < max_i; ++ii) {
                for (size_t jj = j; jj < max_j; ++jj) {
                    dst[jj * rows + ii] = src[ii * cols + jj];
                }
            }
        }
    }

    return std::pair{cols, rows};
}


static float accuracy(const std::vector<int> &y, const std::vector<int> &y_pred) {
    assert(y.size() == y_pred.size());

    float classified = 0;
    for (int sample_id = 0; sample_id < y_pred.size(); ++sample_id) {
        const int prediction = y_pred[sample_id];

        if (prediction == y[sample_id]) {
            classified++;
        }
    }

    return classified / static_cast<float>(y_pred.size());
}

static float f1_score(const std::vector<int> &y, const std::vector<int> &y_pred) {
    assert(y.size() == y_pred.size());

    const int maxLabel = *std::ranges::max_element(y);
    const int numClasses = maxLabel + 1;

    std::vector TP(numClasses, 0);
    std::vector FP(numClasses, 0);
    std::vector FN(numClasses, 0);

    for (int i = 0; i < y_pred.size(); ++i) {
        const int pred = y_pred[i];
        const int trueLabel = y[i];

        if (pred == trueLabel) {
            TP[trueLabel]++;
        } else {
            FP[pred]++;
            FN[trueLabel]++;
        }
    }

    double macro_f1 = 0.0;
    for (int label = 0; label < numClasses; ++label) {
        const double precision = TP[label] + FP[label] > 0
                                     ? static_cast<double>(TP[label]) / (TP[label] + FP[label])
                                     : 0.0;
        const double recall = TP[label] + FN[label] > 0
                                  ? static_cast<double>(TP[label]) / (TP[label] + FN[label])
                                  : 0.0;

        const double f1 = precision + recall > 0
                              ? 2.0 * precision * recall / (precision + recall)
                              : 0.0;

        macro_f1 += f1;
    }

    return static_cast<float>(macro_f1 / numClasses);
}

static std::pair<float, float> compute_metrics(const std::vector<int> &y, const std::vector<int> &y_pred) {
    return std::pair{accuracy(y, y_pred), f1_score(y, y_pred)};
}

inline std::tuple<RandomForestClassifier::RandomForestParams, std::string, size_t> parse_args(int argc, char **argv) {
    cxxopts::Options options("Random Forest Classifier",
                             "A C++ fast implementation of Random Forest algorithm");

    options.add_options()
            ("t,trees", "Number of trees", cxxopts::value<int>()->default_value("10"))
            ("c,criteria", "Split criteria (gini|entropy)", cxxopts::value<std::string>()->default_value("gini"))
            ("mss,min-samples-split", "Minimum samples per split", cxxopts::value<int>()->default_value("2"))
            ("f,max-features", "Max features (int or string: sqrt/log2/all)",
             cxxopts::value<std::string>()->default_value("sqrt"))
            ("b,bootstrap", "Use bootstrap sampling", cxxopts::value<bool>()->default_value("true"))
            ("s,seed", "Random seed (optional)", cxxopts::value<int>())
            ("msr,min-samples-ratio", "Min samples ratio", cxxopts::value<float>()->default_value("0.2"))
            ("d,max-depth", "Maximum depth", cxxopts::value<int>()->default_value(std::to_string(INT_MAX)))
            ("mln,max-leaf-nodes", "Maximum leaf nodes",
             cxxopts::value<size_t>()->default_value(std::to_string(SIZE_MAX)))
            ("ms,max-samples", "Max samples (float<1 or int>=1)", cxxopts::value<std::string>()->default_value("-1.0"))
            ("j,njobs", "Number of parallel jobs", cxxopts::value<int>()->default_value("1"))
            ("w,nworkers", "Number of FastFlow workers", cxxopts::value<int>()->default_value("1"))
            ("dataset", "Dataset name (susy|iris)", cxxopts::value<std::string>()->default_value("iris"))
            ("max-lines", "Maximum number of lines to read",
             cxxopts::value<size_t>()->default_value(std::to_string(SIZE_MAX)))
            ("v,verbose", "Enable logging", cxxopts::value<bool>()->default_value("false"))
            ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        std::exit(0);
    }

    // Parsing preliminare
    int n_trees = result["trees"].as<int>();
    auto& split_criteria = result["criteria"].as<std::string>();
    int min_samples_split = result["min-samples-split"].as<int>();

    // --- max_features (int o string)
    std::variant<int, std::string> max_features; {
        auto& f = result["max-features"].as<std::string>();
        try {
            int v = std::stoi(f);
            max_features = v;
        } catch (...) {
            max_features = f;
        }
    }

    bool bootstrap = result["bootstrap"].as<bool>();
    bool log_enabled = result["verbose"].as<bool>();
    Logger::set_enable(log_enabled);

    // --- random_seed (optional)
    std::optional<int> random_seed = std::nullopt;
    if (result.count("seed"))
        random_seed = result["seed"].as<int>();

    float min_samples_ratio = result["min-samples-ratio"].as<float>();
    int max_depth = result["max-depth"].as<int>();
    size_t max_leaf_nodes = result["max-leaf-nodes"].as<size_t>();

    // --- max_samples (float o size_t)
    std::variant<size_t, float> max_samples; {
        auto& val = result["max-samples"].as<std::string>();
        try {
            if (val.find('.') != std::string::npos)
                max_samples = std::stof(val);
            else
                max_samples = static_cast<size_t>(std::stoull(val));
        } catch (...) {
            max_samples = -1.0f;
        }
    }

    int njobs = result["njobs"].as<int>();
    int nworkers = result["nworkers"].as<int>();
    auto& dataset = result["dataset"].as<std::string>();
    size_t max_lines = result["max-lines"].as<size_t>();

    RandomForestClassifier::RandomForestParams params{
        .n_trees = n_trees,
        .split_criteria = split_criteria,
        .min_samples_split = min_samples_split,
        .max_features = max_features,
        .bootstrap = bootstrap,
        .random_seed = random_seed,
        .min_samples_ratio = min_samples_ratio,
        .max_depth = max_depth,
        .max_leaf_nodes = max_leaf_nodes,
        .max_samples = max_samples,
        .njobs = njobs,
        .nworkers = nworkers
    };

    return std::tuple{params, dataset, max_lines};
}

inline void print_duration(const std::chrono::steady_clock::duration duration) {
    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    const long hours = total_ms / (1000 * 60 * 60);
    const long minutes = total_ms % (1000 * 60 * 60) / (1000 * 60);
    const long seconds = total_ms % (1000 * 60) / 1000;
    const long ms = total_ms % 1000;

    bool printed_something = false;

    if (hours > 0) {
        std::cout << hours << "h ";
        printed_something = true;
    }

    if (minutes > 0 || printed_something) {
        std::cout << minutes << "m ";
    }

    if (ms > 0) {
        std::cout << seconds << "." << std::setfill('0') << std::setw(3) << ms << "s";
    } else {
        std::cout << seconds << "s";
    }
}

#endif //UTILS_H
