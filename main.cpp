#include <random>
#include <vector>
#include <fstream>
#include <iostream>

#include "include/Dataset.h"
#include "include/RandomForestClassifier.h"
#include "include/Timer.h"
#include <matplot/matplot.h>

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

int main() {
    cout << "Loading dataset.." << endl;
    auto [X, y] = Dataset::load("susy", "../dataset");

    auto [X_train, y_train, X_test, y_test] =
        Dataset::train_test_split(X, y, 0.7);

    cout << "Training set size: " << X_train.size() << endl;
    cout << "Test set size: " << X_test.size() << endl << endl;

    cout << "Training start " << endl;
    RandomForestClassifier model({.n_trees = 1, .random_seed = 8, .njobs = -1});

    const auto start = chrono::steady_clock::now();
    model.fit(X_train, y_train);
    const auto end = chrono::steady_clock::now();

    cout << "Training end! :)" << endl;
    print_duration(end - start);
    cout << endl << endl;

    auto [accuracy, f1] = model.evaluate(X_test, y_test);
    cout << "Accuracy: " << accuracy << endl;
    cout << "F1 (Macro): " << f1 << endl;

    timer.summary();
    return 0;
}