//
// Created by gabriele on 13/07/25.
//

#ifndef RANDOMFORESTCLASSIFIER_H
#define RANDOMFORESTCLASSIFIER_H
#include <optional>
#include <variant>
#include <vector>
#include "DecisionTreeClassifier.h"

class RandomForestClassifier {
public:
    struct RandomForestParams {
        int n_trees = 10;
        std::string split_criteria = "gini";
        int min_samples_split = 2;
        const std::variant<int, std::string> max_features = "sqrt";
        bool bootstrap = true;
        const std::optional<int> random_seed = std::nullopt;
    };

    explicit RandomForestClassifier(const RandomForestParams &params)
        : params(params) {
        trees.reserve(params.n_trees);
    }

    void fit(std::vector<std::vector<double>> &X, const std::vector<int> &y);

    [[nodiscard]] int predict(const std::vector<double> &x) const;

    [[nodiscard]] double evaluate(const std::vector<std::vector<double> > &X, const std::vector<int> &y) const;

private:
    RandomForestParams params;
    std::vector<DecisionTreeClassifier> trees;

    void bootstrap_sample(int n_samples, std::vector<int> &indices) const;
};


#endif //RANDOMFORESTCLASSIFIER_H
