#include <random>
#include <vector>
#include <fstream>
#include <iostream>

#include "include/Dataset.h"
#include "include/radix_sort_indices.h"
#include "include/RandomForestClassifier.h"
#include "include/Timer.h"
#include "utils/indicators.hpp"

using namespace std;

void generateTestData(vector<vector<double> > &features, vector<int> &labels, const int numSamples = 100,
                      const int numFeatures = 10, const int numLabels = 2) {
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

void printData(const vector<vector<double> > &features, const vector<int> &labels) {
    for (size_t i = 0; i < features.size(); ++i) {
        for (const double j: features[i]) {
            cout << fixed << setprecision(3) << setw(6) << j << " ";
        }
        cout << "| " << labels[i] << endl;
    }
}

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

// Genera matrice trasposta X[f][i]
std::vector<std::vector<double> > generateRandomMatrix(int rows, int cols) {
    std::vector<std::vector<double> > X(cols, std::vector<double>(rows));
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0f, 1.0f);

    for (int f = 0; f < cols; ++f)
        for (int i = 0; i < rows; ++i)
            X[f][i] = dist(rng);

    return X;
}

void bootstrap_sample(const int n_samples, vector<int> &indices) {
    indices.clear();
    indices.reserve(n_samples);

    std::srand(std::time(nullptr));

    for (size_t i = 0; i < n_samples; ++i) {
        const int idx = rand() % n_samples;

        indices.push_back(idx);
    }
}

void benchmark_pdqsort(int num_elements, int feature_index) {
    vector<int> indices;
    indices.reserve(num_elements);
    bootstrap_sample(num_elements, indices);

    auto X = generateRandomMatrix(num_elements, feature_index);

    const auto start = chrono::steady_clock::now();

    for (int f = 0; f < feature_index; ++f) {
        const auto start_f = chrono::steady_clock::now();
        RADIX_SORT_INDICES(indices, X, f);
        const auto end_f = chrono::steady_clock::now();
        print_duration(end_f - start_f);
        cout << endl;

        for (int i = 0; i < indices.size() - 1; ++i) {
            if (X[f][indices[i]] > X[f][indices[i + 1]]) {
                cerr << "Matrix indices not sorted";
                exit(1);
            }
        }

        ranges::shuffle(indices, std::mt19937(std::random_device()()));
    }
    cout << endl;

    const auto end = chrono::steady_clock::now();
    print_duration(end - start);
}



int main() {
    cout << "Loading dataset.." << endl;
    auto [X, y] = Dataset::load("susy", "../dataset");

    auto [X_train, y_train, X_test, y_test] =
            Dataset::train_test_split(X, y, 0.7);

    cout << "Training set size: " << X_train.size() << endl;
    cout << "Test set size: " << X_test.size() << endl;
    // cout << "Trainng size (bytes): " << (sizeof(double) * X_train.size() * X_train[0].size()) << endl;
    // vector<vector<double> > X_train;
    // vector<int> y_train;

    // generateTestData(X_train, y_train, 5, 5, 3);
    // printData(X_train, y_train);
    cout << "Training start " << endl;
    RandomForestClassifier model({.n_trees = 1, .random_seed = 8});

    const auto start = chrono::steady_clock::now();
    model.fit(X_train, y_train);
    const auto end = chrono::steady_clock::now();

    cout << "Training end! :)" << endl;
    print_duration(end - start);
    cout << endl << endl;

    const double accuracy = model.evaluate(X_test, y_test);
    cout << "Accuracy: " << accuracy << endl;

    timer.summary();
    return 0;
}
