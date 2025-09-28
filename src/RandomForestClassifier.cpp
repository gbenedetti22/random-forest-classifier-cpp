//
// Created by gabriele on 13/07/25.
//

#include "../include/RandomForestClassifier.h"

#include <cassert>
#include <iostream>
#include <omp.h>
#include <pcg32/pcg_random.hpp>

#include "Logger.hpp"

#ifdef MPI_AVAILABLE
#include  <mpi.h>
#endif

#include "utils.h"
#include "../include/Timer.h"
#include <spdlog/spdlog.h>
#include <random>

using namespace std;

void RandomForestClassifier::fit(const vector<vector<float> > &X, const vector<int> &y) {
    const vector<float> flat = flatten(X);
    fit(flat, y, pair{X.size(), X[0].size()});
}

void RandomForestClassifier::fit(const vector<float> &X, const vector<int> &y, const pair<size_t, size_t> &shape) {
    if (X.empty() || y.empty()) {
        cerr << "Cannot build the tree on dataset" << endl;
        exit(EXIT_FAILURE);
    }
    if (params.njobs == 0 || params.njobs < -1) {
        cerr << "Thread count cannot be 0 or less then -1: " << params.njobs << endl;
        exit(EXIT_FAILURE);
    }

    num_classes = ranges::max(y) + 1;

    const int t = params.njobs;
    int threads_count = t == -1 ? omp_get_max_threads() : t;
    int num_trees = params.n_trees;
    int rank, size;
    int master_seed = rand();

    if (params.random_seed.has_value()) {
        master_seed = *params.random_seed;
    }

    seed_seq seq{ master_seed };
    vector<uint32_t> seeds(num_trees);
    seq.generate(seeds.begin(), seeds.end());

    size_t n_samples = 0;
    if (std::holds_alternative<size_t>(params.max_samples)) {
        n_samples = std::get<size_t>(params.max_samples);
    } else {
        if (const float perc = std::get<float>(params.max_samples); perc == -1.0F) {
            n_samples = shape.first;
        }else {
            if (perc < 0.0 || perc > 1.0) {
                cerr << "max_samples (float) must be between 0 and 1" << endl;
                exit(EXIT_FAILURE);
            }
            n_samples = static_cast<size_t>(shape.first * perc);
        }
    }

    spdlog::info("Number of samples used: {} / {}\n", n_samples, shape.first);

    const DTreeParams dtp(
        params.split_criteria,
        params.min_samples_split,
        params.max_features,
        params.min_samples_ratio,
        params.nworkers,
        params.max_depth,
        params.max_leaf_nodes
    );

    if (params.mpi) {
#ifdef MPI_AVAILABLE
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        num_trees /= size;
        if (rank == 0)
            cout << "Number of trees per node: " << num_trees << endl << endl;
#else
        cerr << "MPI not available" << endl;
        exit(EXIT_FAILURE);
#endif
    }

    for (int i = 0; i < num_trees; ++i) {
        trees.emplace_back(dtp, seeds[i]);
    }

    spdlog::set_pattern("[%t %H:%M:%S] [%^%l%$] %v");

    #pragma omp parallel for schedule(guided) num_threads(threads_count)
    for (int i = 0; i < num_trees; i++) {
        if (params.mpi) {
            const int global_idx = rank * num_trees + i;
            spdlog::debug("Rank {} - Thread {} / {} : Training tree n. {}", rank, omp_get_thread_num() + 1,
                          omp_get_thread_num(), global_idx + 1);
        } else {
            spdlog::info("Thread {} / {} : Training tree n. {}", omp_get_thread_num() + 1, omp_get_num_threads(),
                          i + 1);
        }

        vector<int> indices(n_samples);
        if (params.bootstrap) {
            bootstrap_sample(n_samples, shape.first, indices);

            trees[i].train(X, shape, y, indices);
        } else {
            iota(indices.begin(), indices.end(), 0);
            trees[i].train(X, shape, y, indices);
        }
    }
}

int RandomForestClassifier::predict(const vector<float> &x) const {
    vector vote_counts(num_classes, 0);

    const int t = params.njobs;
    int threads_count = t == -1 ? omp_get_max_threads() : t;

#pragma omp parallel for if(threads_count > 1) num_threads(threads_count)
    for (int i = 0; i < trees.size(); i++) {
        const int pred = trees[i].predict(x);
#pragma omp atomic
        vote_counts[pred]++;
    }

    int rank = 0;
    int majority_class = -1;

    if (params.mpi) {
#ifdef MPI_AVAILABLE
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == 0) {
            vector total_votes(num_classes, 0);
            MPI_Reduce(vote_counts.data(), total_votes.data(),
                       num_classes, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

            int max_votes = -1;
            for (int label = 0; label < num_classes; label++) {
                if (total_votes[label] > max_votes) {
                    max_votes = total_votes[label];
                    majority_class = label;
                }
            }
        } else {
            MPI_Reduce(vote_counts.data(), nullptr,
                       num_classes, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        }

#endif
    } else {
        int max_votes = -1;
        for (int label = 0; label < num_classes; label++) {
            if (vote_counts[label] > max_votes) {
                max_votes = vote_counts[label];
                majority_class = label;
            }
        }
    }

    return majority_class;
}

pair<float, float> RandomForestClassifier::evaluate(const vector<vector<float>> &X, const vector<int> &y) const {
    int classified = 0;
    vector<int> y_pred;
    y_pred.reserve(X.size());

    if (params.mpi) {
#ifdef MPI_AVAILABLE
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        params.mpi = false;
        for (const auto & i : X) {
            int prediction = predict(i);
            y_pred.push_back(prediction);
        }
        params.mpi = true;

        vector<int> y_pred_total;
        if (rank == 0) {
            y_pred_total.resize(size * X.size());
        }
        MPI_Gather(y_pred.data(), y_pred.size(), MPI_INT, y_pred_total.data(), y_pred.size(), MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0) return make_pair(0.0f, 0.0f);

        vector<int> y_pred_final;
        y_pred_final.reserve(X.size());

        for (int i = 0; i < X.size(); ++i) {
            unordered_map<int,int> votes;
            for (int r = 0; r < size; ++r) {
                int pred = y_pred_total[r * X.size() + i];
                votes[pred]++;
            }

            int maj_class = -1, max_votes = -1;
            for (auto &[label, count] : votes) {
                if (count > max_votes) {
                    max_votes = count;
                    maj_class = label;
                }
            }

            y_pred_final.push_back(maj_class);
            if (maj_class == y[i]) classified++;
        }


        return make_pair(static_cast<float>(classified) / X.size(), f1_score(y, y_pred));
#endif
    }


    for (int i = 0; i < X.size(); ++i) {
        int prediction = predict(X[i]);
        y_pred.push_back(prediction);

        if (prediction == y[i]) {
            classified++;
        }
    }

    return make_pair(static_cast<float>(classified) / X.size(), f1_score(y, y_pred));
}

float RandomForestClassifier::f1_score(const vector<int> &y, const vector<int> &y_pred) {
    assert(y.size() == y_pred.size());

    const int maxLabel = *ranges::max_element(y);
    const int numClasses = maxLabel + 1;

    vector TP(numClasses, 0);
    vector FP(numClasses, 0);
    vector FN(numClasses, 0);

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



void RandomForestClassifier::bootstrap_sample(const size_t n_samples,
                                              const size_t total_features,
                                              std::vector<int> &indices) const
{
    thread_local pcg32 rng(std::random_device{}());
    thread_local bool initialized = false;

    if (params.random_seed.has_value() && !initialized) {
        rng.seed(params.random_seed.value() + omp_get_thread_num());
        initialized = true;
    }

    for (size_t i = 0; i < n_samples; ++i) {
        indices[i] = rng(total_features);
    }
}

