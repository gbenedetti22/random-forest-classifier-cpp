#include <vector>
#include <fstream>
#include <iostream>
#include <mpi.h>

#include "include/Dataset.h"
#include "include/RandomForestClassifier.h"
#include "include/Timer.h"

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vector<float> X_train_flat;
    vector<int> y_train;
    vector<float> X_test_flat;
    vector<int> y_test;
    int rows = 0, cols = 0, X_size = 0, y_size = 0;
    int test_rows = 0, test_cols = 0, X_test_size = 0, y_test_size = 0;

    if (rank == 0) {
        cout << "Loading dataset.." << endl;
        auto [X, y] = Dataset::load("iris", "../dataset");

        auto [X_train, y_train_local, X_test_local, y_test_local] =
                Dataset::train_test_split(X, y, 0.7);

        X_train_flat = flatten(X_train);
        y_train = y_train_local;
        X_test_flat = flatten(X_test_local);
        y_test = y_test_local;

        rows = X_train.size();
        cols = X_train[0].size();
        X_size = X_train_flat.size();
        y_size = y_train.size();

        test_rows = X_test_local.size();
        test_cols = X_test_local[0].size();
        X_test_size = X_test_flat.size();
        y_test_size = y_test.size();

        cout << "Training set size: " << X_train.size() << endl;
        cout << "Test set size: " << X_test_local.size() << endl << endl;
    }

    int meta[8] = {rows, cols, X_size, y_size, test_rows, test_cols, X_test_size, y_test_size};
    MPI_Bcast(meta, 8, MPI_INT, 0, MPI_COMM_WORLD);

    rows = meta[0];
    cols = meta[1];
    X_size = meta[2];
    y_size = meta[3];
    test_rows = meta[4];
    test_cols = meta[5];
    X_test_size = meta[6];
    y_test_size = meta[7];

    if (rank != 0) {
        X_train_flat.resize(X_size);
        y_train.resize(y_size);
        X_test_flat.resize(X_test_size);
        y_test.resize(y_test_size);
    }

    MPI_Bcast(X_train_flat.data(), X_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(y_train.data(), y_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(X_test_flat.data(), X_test_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(y_test.data(), y_test_size, MPI_INT, 0, MPI_COMM_WORLD);

    vector X_test(test_rows, vector<float>(test_cols));
    for (int i = 0; i < test_rows; i++) {
        for (int j = 0; j < test_cols; j++) {
            X_test[i][j] = X_test_flat[i * test_cols + j];
        }
    }

    const int seed = 8 + rank * 100;
    RandomForestClassifier model({.n_trees = 10, .random_seed = 8, .njobs = -1, .mpi = true});

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    model.fit(X_train_flat, y_train, {rows, cols});

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double local_time = end - start;

    cout << "Rank " << rank << " total time: " << local_time << " seconds" << endl;

    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        cout << endl << "Training time (MAX): " << max_time << " seconds (" << (max_time / 60) << " minutes)" << endl;
    }

    double start_eval = MPI_Wtime();
    auto [accuracy, f1] = model.evaluate(X_test, y_test, TODO);
    MPI_Barrier(MPI_COMM_WORLD);
    double end_eval = MPI_Wtime();

    double local_time_eval = end_eval - start_eval;
    MPI_Reduce(&local_time_eval, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        cout << "Accuracy: " << accuracy << endl;
        cout << "F1 (Macro): " << f1 << endl;
        cout << endl << "Eval time (MAX): " << max_time << " seconds (" << (max_time / 60) << " minutes)" << endl;
    }

    MPI_Finalize();
    return 0;
}