#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <unordered_map>

#include "Dataset.h"
#include "RandomForestClassifier.h"
#include "utils.h"

using namespace std;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    vector<float> X_flat; // conterrà X_train + X_test
    vector<int> y_full; // conterrà y_train + y_test

    // Array di metadati: [n_train_samples, n_test_samples, n_features]
    size_t metadata[3] = {0, 0, 0};

    if (rank == 0) {
        pair<size_t, size_t> test_shape, train_shape;
        cout << "Master node (rank 0) loading dataset..." << endl;

        auto [X, y] = Dataset::load("susy", "../dataset");
        auto [X_train, y_train, X_test, y_test] = Dataset::train_test_split(X, y, 0.7);

        vector<float> X_train_flat;
        vector<float> X_test_flat = flatten(X_test);
        test_shape = pair(X_test.size(), X_test[0].size());
        train_shape = transpose(X_train, X_train_flat);

        size_t n_train_samples = train_shape.first;
        size_t n_test_samples = test_shape.first;
        size_t n_features = train_shape.second;

        // Combina train e test in un unico buffer
        X_flat.reserve(X_train_flat.size() + X_test_flat.size());
        X_flat.insert(X_flat.end(), X_train_flat.begin(), X_train_flat.end());
        X_flat.insert(X_flat.end(), X_test_flat.begin(), X_test_flat.end());

        // Combina y_train e y_test
        y_full.reserve(y_train.size() + y_test.size());
        y_full.insert(y_full.end(), y_train.begin(), y_train.end());
        y_full.insert(y_full.end(), y_test.begin(), y_test.end());

        // Metadati
        metadata[0] = n_train_samples;
        metadata[1] = n_test_samples;
        metadata[2] = n_features;

        cout << "Train samples: " << n_train_samples
                << ", Test samples: " << n_test_samples
                << ", Features: " << n_features << endl;
    }

    // Broadcast dei metadati a tutti
    MPI_Bcast(metadata, 3, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    size_t n_train_samples = metadata[0];
    size_t n_test_samples = metadata[1];
    size_t n_features = metadata[2];
    size_t total_samples = n_train_samples + n_test_samples;
    size_t total_floats = total_samples * n_features;

    if (rank != 0) {
        X_flat.resize(total_floats);
        y_full.resize(total_samples);
    }

    // Broadcast dei dati (dataset completo)
    MPI_Bcast(X_flat.data(), total_floats, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(y_full.data(), total_samples, MPI_INT, 0, MPI_COMM_WORLD);

    // Estrazione locale di training e test set
    vector X_train_flat(X_flat.begin(), X_flat.begin() + n_train_samples * n_features);
    vector X_test_flat(X_flat.begin() + n_train_samples * n_features, X_flat.end());
    X_flat.clear();
    X_flat.shrink_to_fit();

    vector y_train(y_full.begin(), y_full.begin() + n_train_samples);
    vector y_test(y_full.begin() + n_train_samples, y_full.end());
    y_full.clear();
    y_full.shrink_to_fit();

    pair train_shape_local = {n_train_samples, n_features};
    pair test_shape_local = {n_test_samples, n_features};


    // Ogni nodo allena un sottoinsieme di alberi
    int n_trees_total = 1000;
    int n_trees_local = n_trees_total / world;
    if (rank == 0) {
        cout << "N. of Trees per node: " << n_trees_local << endl;
    }

    RandomForestClassifier model({
        .n_trees = n_trees_local,
        .random_seed = 24 + rank,
        .njobs = -1,
        .nworkers = 1
    });

    cout << "Training start: " << rank << endl;
    auto start_train = chrono::steady_clock::now();
    model.fit(X_train_flat, y_train, train_shape_local);
    auto end_train = chrono::steady_clock::now();

    cout << "Rank " << rank << " finished training in "
            << chrono::duration_cast<chrono::seconds>(end_train - start_train).count()
            << "s" << endl;

    // Predizione sul test set
    vector<int> local_preds = model.predict(X_test_flat, test_shape_local);

    MPI_Barrier(MPI_COMM_WORLD);

    // Raccolta di tutte le predizioni in un’unica chiamata
    vector<int> all_preds;
    if (rank == 0)
        all_preds.resize(world * n_test_samples);

    MPI_Gather(local_preds.data(), n_test_samples, MPI_INT,
               all_preds.data(), n_test_samples, MPI_INT,
               0, MPI_COMM_WORLD);

    // Master node: majority voting e metriche finali
    if (rank == 0) {
        vector<int> final_preds;
        final_preds.reserve(n_test_samples);

        for (size_t i = 0; i < n_test_samples; ++i) {
            unordered_map<int, int> votes;
            for (int r = 0; r < world; ++r) {
                int pred = all_preds[r * n_test_samples + i];
                votes[pred]++;
            }

            int maj_class = -1, max_votes = -1;
            for (auto &[label, count]: votes) {
                if (count > max_votes) {
                    max_votes = count;
                    maj_class = label;
                }
            }

            final_preds.push_back(maj_class);
        }

        auto [accuracy, f1] = compute_metrics(y_test, final_preds);

        cout << "Accuracy: " << accuracy << endl;
        cout << "F1 (Macro): " << f1 << endl;
    }

    MPI_Finalize();
    return 0;
}
