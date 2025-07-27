#include <vector>
#include <fstream>
#include <iostream>

#include "include/Dataset.h"
#include "include/RadixSort.h"
#include "include/RandomForestClassifier.h"
#include "include/Timer.h"
#include "utils/indicators.hpp"

using namespace std;

void generateTestData(vector<vector<double> > &features, vector<int> &labels, const int numSamples = 100, const int numFeatures = 10, const int numLabels = 2) {
    features.clear();
    labels.clear();

    for (int i = 0; i < numSamples; ++i) {
        vector<double> sample;

        for (int j = 0; j < numFeatures; ++j) {
            sample.push_back(static_cast<double>(rand()) / RAND_MAX);
        }

        int label = rand() % (numLabels + 1);

        features.push_back(sample);
        labels.push_back(label);
    }
}

void printData(const vector<vector<double>> &features, const vector<int> &labels) {
    for (size_t i = 0; i < features.size(); ++i) {
        for (const double j : features[i]) {
            cout << fixed << setprecision(3) << setw(6) << j << " ";
        }
        cout << "| " << labels[i] << endl;
    }
}

void print_duration(std::chrono::steady_clock::duration duration) {
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    long hours = total_ms / (1000 * 60 * 60);
    long minutes = (total_ms % (1000 * 60 * 60)) / (1000 * 60);
    long seconds = (total_ms % (1000 * 60)) / 1000;
    long ms = total_ms % 1000;

    cout << "Training time: ";

    bool printed_something = false;

    if (hours > 0) {
        cout << hours << "h ";
        printed_something = true;
    }

    if (minutes > 0 || printed_something) {
        cout << minutes << "m ";
        printed_something = true;
    }

    if (ms > 0) {
        cout << seconds << "." << setfill('0') << setw(3) << ms << "s";
    } else {
        cout << seconds << "s";
    }
}


int main() {
    cout << "Loading dataset.." << endl;
    auto [X, y] = Dataset::load_classification("susy", -1);

    auto [X_train, y_train, X_test, y_test] =
        Dataset::train_test_split(X, y, 0.7);

    cout << "Training set size: " << X_train.size() << endl;
    cout << "Test set size: " << X_test.size() << endl;
    // vector<vector<double> > X_train;
    // vector<int> y_train;

    // generateTestData(X_train, y_train, 5, 5, 3);
    // printData(X_train, y_train);
    cout << "Training start " << endl;
    RandomForestClassifier model({.n_trees = 1, .random_seed = 8});

    const auto start = std::chrono::steady_clock::now();
    model.fit(X_train, y_train);
    const auto end = std::chrono::steady_clock::now();

    cout << "Training end! :)" << endl;
    print_duration(end - start);
    cout << endl << endl;

    const double accuracy = model.evaluate(X_test, y_test);
    cout << "Accuracy: " << accuracy <<endl;

    timer.summary();
    return 0;
}
