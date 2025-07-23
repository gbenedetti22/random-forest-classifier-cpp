#include <vector>
#include <fstream>
#include <iostream>

#include "include/Dataset.h"
#include "include/RandomForestClassifier.h"
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

int main() {
    cout << "Loading dataset.." << endl;
    auto [X, y] = Dataset::load_classification("iris");
    auto [X_train, y_train, X_test, y_test] =
        Dataset::train_test_split(X, y, 0.7);

    cout << "Training set size: " << X_train.size() << endl;
    cout << "Test set size: " << X_test.size() << endl;
    // vector<vector<double> > X_train;
    // vector<int> y_train;

    // generateTestData(X_train, y_train, 5, 5, 3);
    // printData(X_train, y_train);

    RandomForestClassifier model({.n_trees = 1, .random_seed = 8});
    model.fit(X_train, y_train);

    cout << "Training end! :)" << endl;

    const double accuracy = model.evaluate(X_test, y_test);
    cout << "Accuracy: " << accuracy <<endl;

    return 0;
}
