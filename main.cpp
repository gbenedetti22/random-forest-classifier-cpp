#include <random>
#include <vector>
#include <fstream>
#include <iostream>
// #include <execution>

// #include "utils.h"
#include <omp.h>

#include "include/Dataset.h"
#include "include/RandomForestClassifier.h"
#include "include/Timer.h"

using namespace std;

void print_duration(chrono::steady_clock::duration duration) {
    const auto total_ms = chrono::duration_cast<chrono::milliseconds>(duration).count();

    const long hours = total_ms / (1000 * 60 * 60);
    const long minutes = (total_ms % (1000 * 60 * 60)) / (1000 * 60);
    const long seconds = (total_ms % (1000 * 60)) / 1000;
    const long ms = total_ms % 1000;

    cout << "Time: ";

    bool printed_something = false;

    if (hours > 0) {
        cout << hours << "h ";
        printed_something = true;
    }

    if (minutes > 0 || printed_something) {
        cout << minutes << "m ";
    }

    if (ms > 0) {
        cout << seconds << "." << setfill('0') << setw(3) << ms << "s";
    } else {
        cout << seconds << "s";
    }
}

// int main() {
//     // constexpr int dataset_size[] = {2, 7, 10, 20, 30};
//     timer.set_active(false);
//
//     for (const int gb : {4}) {
//         cout << "Dataset size: " << gb << "Gb" << endl;
//
//         const float GB = gb; // 2, 7, 10, 20, 30
//         const long long bytes = GB * 1024 * 1024 * 1024;
//         constexpr int n_features = 20;
//         const long rows = bytes / (sizeof(float) * n_features);
//         constexpr int cols = n_features;
//
//         cout << "Loading dataset.." << endl;
//         auto [X_train, y_train] = generate_matrix(n_features, bytes);
//
//         cout << "Training start " << endl;
//         RandomForestClassifier model({
//             .n_trees = 100, .random_seed = 0, .njobs = -1, .nworkers = 1
//         });
//
//         const auto start = chrono::steady_clock::now();
//         model.fit(X_train, y_train, {rows, cols});
//         const auto end = chrono::steady_clock::now();
//
//         cout << "Training end! :)" << endl;
//         print_duration(end - start);
//         cout << endl;
//
//         // auto [accuracy, f1] = model.evaluate(X_train, y_train);
//         // cout << "Accuracy: " << accuracy << endl;
//         // cout << "F1 (Macro): " << f1 << endl;
//     }
//
//     return 0;
// }

int main() {
    cout << "PID: " << getpid() << endl << endl;
    timer.set_active(false);

    cout << "Loading dataset.." << endl;
    auto [X, y] = Dataset::load("susy", "../dataset");
    auto [X_train, y_train, X_test, y_test] =
    Dataset::train_test_split(X, y, 0.7);

    cout << "Training set size: " << X_train.size() << endl;
    cout << "Test set size: " << X_test.size() << endl << endl;
    vector<float> X_train_cm;
    const auto shape = transpose(X_train, X_train_cm);
    X_train.clear();
    X_train.shrink_to_fit();

    RandomForestClassifier model({
        .n_trees = 100, .random_seed = 24, .njobs = -1, .nworkers = 1 // seed = 50
    });

    const auto start = chrono::steady_clock::now();
    model.fit(X_train_cm, y_train, shape);
    const auto end = chrono::steady_clock::now();

    // cout << "Training end! :)" << endl;
    print_duration(end - start);
    cout << endl << endl;

    vector<float> X_test_flat = flatten(X_test);

    const auto start2 = chrono::steady_clock::now();
    auto [accuracy_new, f1] = model.score(X_test_flat, y_test, pair{X_test.size(), X_test[0].size()});
    const auto end2 = chrono::steady_clock::now();
    cout << "Accuracy: " << accuracy_new << endl;
    cout << "F1 (Macro): " << f1 << endl;
    print_duration(end2 - start2);

    timer.summary();
    return 0;
}