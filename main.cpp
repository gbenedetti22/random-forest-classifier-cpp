#include <vector>
#include <fstream>

#include "include/Dataset.h"
#include "include/RandomForestClassifier.h"
#include "utils/indicators.hpp"

struct Sample;
using namespace std;

int main() {
    cout << "Loading dataset.." << endl;
    const vector<Sample> dataset = Dataset::load_classification("susy");
    auto [train, test] = Dataset::train_test_split(dataset);
    cout << "Training set size: " << train.size() << endl;
    cout << "Test set size: " << test.size() << endl;

    cout << "Training start.." << endl;
    RandomForestClassifier model(10);
    model.fit(train);
    cout << "Training end! :)" << endl;

    model.score(test);

    return 0;
}
