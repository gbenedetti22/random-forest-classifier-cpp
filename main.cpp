#include <vector>
#include <fstream>
#include <iostream>

#include "include/Dataset.h"
#include "include/RandomForestClassifier.h"
#include <cxxopts.hpp>

#include "utils.h"

using namespace std;

int main(int argc, char** argv) {
    cout << "PID: " << getpid() << endl;

    auto [params, dataset, max_lines] = parse_args(argc, argv);

    auto [X, y] = Dataset::load(dataset, "../dataset", max_lines);
    auto [X_train, y_train, X_test, y_test] =
        Dataset::train_test_split(X, y, 0.7);

    cout << "Dataset: " << dataset << endl;
    cout << "N. Threads: " << params.njobs << endl;
    cout << "N. Threads (FF): " << params.nworkers << endl;
    cout << "Training set size: " << X_train.size() << endl;
    cout << "Test set size: " << X_test.size() << endl;

    // --- Training ---
    RandomForestClassifier model(params);
    const auto start_train = chrono::steady_clock::now();
    model.fit(X_train, y_train);
    const auto end_train = chrono::steady_clock::now();

    // --- Valutazione ---
    const auto start_pred = chrono::steady_clock::now();
    auto [accuracy, f1] = model.score(X_test, y_test);
    const auto end_pred = chrono::steady_clock::now();

    // --- Report finale ---
    cout << endl;
    cout << "Training time: ";
    print_duration(end_train - start_train);

    cout << endl;
    cout << "Prediction time: ";
    print_duration(end_pred - start_pred);

    cout << endl;
    cout << "Accuracy: " << accuracy << endl;
    cout << "F1 (Macro): " << f1 << endl << endl;

    return 0;
}
