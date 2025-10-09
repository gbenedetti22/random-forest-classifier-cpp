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
    auto [params, dataset, max_lines] = parse_args(argc, argv);

    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    vector<float> X_flat; // conterrà X_train + X_test
    vector<int> y_full; // conterrà y_train + y_test

    // Array di metadati: [n_train_samples, n_test_samples, n_features]
    size_t metadata[3] = {0, 0, 0};

    if (rank == 0) {
        cout << "PID: " << getpid() << endl;
        cout << "Dataset: " << dataset << endl;
        cout << "N. Threads: " << params.njobs << endl;
        cout << "N. Threads (FF): " << params.nworkers << endl << endl;

        pair<size_t, size_t> test_shape, train_shape;
        cout << "Master node (rank 0) loading dataset..." << endl;

        auto [X, y] = Dataset::load(dataset, "../dataset", max_lines);
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

        cout << "Training set size: " << n_train_samples << endl;
        cout << "Test set size: " << n_test_samples << endl;
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

    params.n_trees /= world;
    if (rank == 0) {
        cout << "N. of Trees per node: " << params.n_trees << endl;
    }

    if (params.random_seed.has_value()) {
        params.random_seed = *params.random_seed + rank;
    }

    RandomForestClassifier model(params);

    double start_train = MPI_Wtime();
    model.fit(X_train_flat, y_train, train_shape_local, true);
    double end_train = MPI_Wtime();
    double local_train_time = end_train - start_train;

    double train_time = 0.0;
    MPI_Reduce(&local_train_time, &train_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Predizione sul test set
    double start_pred = MPI_Wtime();
    vector<int> local_preds = model.predict(X_test_flat, test_shape_local);

    // Raccolta di tutte le predizioni in un’unica chiamata
    vector<int> all_preds;
    if (rank == 0)
        all_preds.resize(world * n_test_samples);

    MPI_Gather(local_preds.data(), n_test_samples, MPI_INT,
               all_preds.data(), n_test_samples, MPI_INT,
               0, MPI_COMM_WORLD);

    double end_pred = MPI_Wtime();
    double local_pred_time = end_pred - start_pred;

    double pred_time = 0.0;
    MPI_Reduce(&local_pred_time,  &pred_time,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Master node: majority voting e metriche finali
    if (rank == 0) {
        double start_find_majority_time = MPI_Wtime();
        vector<int> final_preds;
        final_preds.resize(n_test_samples);

        #pragma omp parallel for
        for (size_t i = 0; i < n_test_samples; ++i) {
            std::unordered_map<int, int> votes;

            for (int r = 0; r < world; ++r) {
                int pred = all_preds[r * n_test_samples + i];
                votes[pred]++;
            }

            int maj_class = -1, max_votes = -1;
            for (auto &[label, count] : votes) {
                if (count > max_votes) {
                    max_votes = count;
                    maj_class = label;
                }
            }

            final_preds[i] = maj_class;
        }


        auto [accuracy, f1] = compute_metrics(y_test, final_preds);
        double end_find_majority_time = MPI_Wtime();

        double pred_time_total = pred_time + (end_find_majority_time - start_find_majority_time);

        cout << "Train Time: " << train_time << "s" << endl;
        cout << "Predicate Time: " << pred_time << "s" << endl;
        cout << "Accuracy: " << accuracy << endl;
        cout << "F1 (Macro): " << f1 << endl;

        ofstream file("results.csv", ios::app);
        if (file.tellp() == 0) {
            file << "n_nodes,n_threads,train_time,predict_time\n";
        }
        file << world << "," << params.njobs << ","
             << train_time << "," << pred_time_total << "\n";
        file.close();
    }

    MPI_Finalize();
    return 0;
}
