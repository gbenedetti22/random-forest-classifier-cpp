//
// Created by gabriele on 13/07/25.
//

#include "../include/RandomForestClassifier.h"

#include <iostream>

#include "indicators.hpp"
using namespace std;

RandomForestClassifier::RandomForestClassifier(const int n_tress) {
    trees.reserve(n_tress);
}

void RandomForestClassifier::fit(const std::vector<Sample> &data) {
    if (data.empty()) {
        std::cerr << "Cannot build the tree on dataset" << std::endl;
        return;
    }
    int num_trees = trees.capacity();

    indicators::ProgressBar bar{
        indicators::option::BarWidth{50},
        indicators::option::Fill{"■"},
        indicators::option::Lead{"■"},
        indicators::option::PostfixText{"Training RandomForest"},
        indicators::option::ShowPercentage{true},
        indicators::option::ShowElapsedTime{true},
        indicators::option::MaxProgress(num_trees),
        indicators::option::FontStyles{std::vector{indicators::FontStyle::bold}}
    };

    bar.print_progress();

    for (int i = 0; i < num_trees; i++) {
        vector<Sample> sample = bootstrap_sample(data);
        DecisionTreeClassifier tree("Decision Tree n. " + i);
        tree.train(sample);
        trees.push_back(tree);

        bar.tick();
    }

    if (!bar.is_completed()) {
        bar.mark_as_completed();
    }
}

int RandomForestClassifier::predict(const Sample &s) const {
    unordered_map<int, int> vote_counts;
    for (const auto& tree : trees) {
        int pred = tree.predict(s);
        vote_counts[pred]++;
    }

    int majority_class = -1, max_votes = -1;
    for (const auto& [label, count] : vote_counts) {
        if (count > max_votes) {
            max_votes = count;
            majority_class = label;
        }
    }
    return majority_class;
}

void RandomForestClassifier::score(const std::vector<Sample> &data) const {
    int classfied = 0;

    for (const Sample& sample : data) {
        if (predict(sample) == sample.label) {
            classfied++;
        }
    }

    cout << "Accuracy: " << static_cast<double>(classfied) / data.size() << endl;
}

std::vector<Sample> RandomForestClassifier::bootstrap_sample(const std::vector<Sample> &data) {
    vector<Sample> sample;
    for (size_t i = 0; i < data.size(); ++i) {
        const int idx = rand() % data.size();
        sample.push_back(data[idx]);
    }
    return sample;
}
