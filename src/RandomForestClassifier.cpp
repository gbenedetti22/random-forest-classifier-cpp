//
// Created by gabriele on 13/07/25.
//

#include "../include/RandomForestClassifier.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <omp.h>
#include <Eigen/Dense>

#ifdef MPI_AVAILABLE
#include  <mpi.h>
#endif

#include "utils.h"
#include "../include/Timer.h"
using namespace std;

void RandomForestClassifier::fit(const vector<vector<float> > &X, const vector<int> &y) {
    const vector<float> flat = flatten(X);
    fit(flat, y, {X.size(), X[0].size()});
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
    const size_t rows = shape.first;

    const int t = params.njobs;
    int threads_count = t == -1 ? omp_get_max_threads() : t;
    int num_trees = params.n_trees;
    int rank, size;
    timer.set_active(false);
    int master_seed = rand();

    if (params.random_seed.has_value()) {
        master_seed = *params.random_seed;
    }

    seed_seq seq{ master_seed };
    vector<uint32_t> seeds(num_trees);
    seq.generate(seeds.begin(), seeds.end());

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

#pragma omp parallel for if(threads_count > 1) num_threads(threads_count)
    for (int i = 0; i < num_trees; i++) {

#pragma omp critical(output)
        {
            if (params.mpi) {
                const int global_idx = rank * num_trees + i;
                cout << "Rank: " << rank << " - Thread " << omp_get_thread_num() + 1 << "/" << omp_get_num_threads()
                    << ": Training tree n. " << global_idx + 1 << "\n";
            }else {
                cout << "Thread " << omp_get_thread_num() + 1 << "/" << omp_get_num_threads()
                    << ": Training tree n. " << i + 1 << "\n";
            }
        }

        auto tree = DecisionTreeClassifier(params.split_criteria, params.min_samples_split, params.max_features,
                                    seeds[i], params.min_samples_ratio, params.nworkers);

        if (params.bootstrap) {
            vector<int> indices;

            bootstrap_sample(rows, indices);

            tree.train(X, shape, y, indices);
        } else {
            vector<int> indices(X.size());

            iota(indices.begin(), indices.end(), 0);
            tree.train(X, shape, y, indices);
        }

        #pragma omp critical
        {
            trees.push_back(std::move(tree));
        }
    }
}

int RandomForestClassifier::predict(const vector<float> &x) const {
    vector vote_counts(num_classes, 0);

    const int t = params.njobs;
    int threads_count = t == -1 ? omp_get_max_threads() : t;

#pragma omp parallel for if(threads_count > 1) num_threads(threads_count)
    for (size_t i = 0; i < trees.size(); i++) {
        const int pred = trees[i].predict(x);
        if (pred < 0 || pred >= num_classes) {
            cerr << "Warning: prediction " << pred << " is outside range [0, " << num_classes-1 << "]" << endl;
            continue;
        }
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

pair<float, float> RandomForestClassifier::score(const vector<vector<float>> &X, const vector<int> &y) const {
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

        for (size_t i = 0; i < X.size(); ++i) {
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


    for (size_t i = 0; i < X.size(); ++i) {
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

    for (size_t i = 0; i < y_pred.size(); ++i) {
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


void RandomForestClassifier::bootstrap_sample(const size_t n_samples, vector<int> &indices) const {
    indices.clear();
    indices.reserve(n_samples);

    if (params.random_seed.has_value()) {
        srand(params.random_seed.value());
    } else {
        srand(time(nullptr));
    }

    for (size_t i = 0; i < n_samples; ++i) {
        const int idx = rand() % static_cast<int>(n_samples);

        indices.push_back(idx);
    }
}
