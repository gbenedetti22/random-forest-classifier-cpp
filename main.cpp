#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include <execution>

#include "utils.h"
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
//     constexpr int dataset_size[] = {2, 7, 10, 20, 30};
//
//     for (const int gb : dataset_size) {
//         const float GB = gb; // 2, 7, 10, 20, 30
//         const long long bytes = GB * 1024 * 1024 * 1024;
//         constexpr int n_features = 20;
//         const long rows = bytes / (sizeof(float) * n_features);
//         constexpr int cols = n_features;
//
//         cout << "Loading dataset.." << endl;
//         auto [X_train, y_train] = generate_matrix(n_features, bytes);
//         cout << "Training set size: " << X_train.size() << endl;
//
//         cout << "Training start " << endl;
//         RandomForestClassifier model({.n_trees = 40, .random_seed = 47, .njobs = -1, .nworkers = 1});
//
//         const auto start = chrono::steady_clock::now();
//         model.fit(X_train, y_train, {rows, cols});
//         const auto end = chrono::steady_clock::now();
//
//         cout << "Training end! :)" << endl;
//         print_duration(end - start);
//         cout << endl;
//
//         writeToFile("times_openmp.txt", gb);
//         writeToFile("times_openmp.txt", end - start);
//     }
//
//     return 0;
// }

int main() {
     // constexpr float GB = 30; // 2, 7, 10, 20, 30
     // constexpr long long bytes = GB * 1024 * 1024 * 1024;
     // constexpr int n_features = 20;
     // constexpr long rows = bytes / (sizeof(float) * n_features);
     // constexpr int cols = n_features;

     cout << "Loading dataset.." << endl;
     auto [X, y] = Dataset::load("iris", "../dataset");

     auto [X_train, y_train, X_test, y_test] =
     Dataset::train_test_split(X, y, 0.7);
    // auto [X_train, y_train] = generate_matrix(n_features, bytes);

    cout << "Training set size: " << X_train.size() << endl;
    cout << "Test set size: " << X_test.size() << endl << endl;

    cout << "Training start " << endl;
    RandomForestClassifier model({.n_trees = 1, .random_seed = 47, .njobs = 1, .nworkers = 1});

    const auto start = chrono::steady_clock::now();
    model.fit(X_train, y_train);
    const auto end = chrono::steady_clock::now();

    cout << "Training end! :)" << endl;
    print_duration(end - start);
    cout << endl;

    auto [accuracy, f1] = model.score(X_test, y_test);
    cout << "Accuracy: " << accuracy << endl;
    cout << "F1 (Macro): " << f1 << endl;

    // timer.summary();
    return 0;
}
