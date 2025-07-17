#include <vector>
#include <fstream>
#include <iostream>
#include <numeric>

#include "include/Dataset.h"
#include "include/RandomForestClassifier.h"
#include "utils/indicators.hpp"

using namespace std;

int main() {
    cout << "Loading dataset.." << endl;
    auto [X, y] = Dataset::load_classification("iris");
    auto [X_train, y_train, X_test, y_test] =
        Dataset::train_test_split(X, y, 0.7);

    cout << "Training set size: " << X_train.size() << endl;
    cout << "Test set size: " << X_test.size() << endl;

    cout << "Training start.." << endl;
    RandomForestClassifier model(1);
    model.fit(X_train, y_train);
    cout << "Training end! :)" << endl;

    const double accuracy = model.evaluate(X_test, y_test);

    cout << "Accuracy: " << accuracy <<endl;

    return 0;
}
