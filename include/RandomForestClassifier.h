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
        float min_samples_ratio = 0.2f;
        int max_depth = INT_MAX;
        size_t max_leaf_nodes = SIZE_MAX;
        const std::variant<size_t, float> max_samples = -1.0F;
        int njobs = 1;
        int nworkers = 1;
        mutable bool mpi = false;
    };

    explicit RandomForestClassifier(const RandomForestParams &params)
        : params(params) {
        trees.reserve(params.n_trees);
    }

    void fit(const std::vector<std::vector<float>> &X, const std::vector<int> &y);

    void fit(const std::vector<float> &X, const std::vector<int> &y, const std::pair<unsigned long, unsigned long> &shape);

    [[nodiscard]] int predict(const std::vector<float> &x) const;

    [[nodiscard]] std::pair<float, float> evaluate(const std::vector<std::vector<float> > &X, const std::vector<int> &y) const;


private:
    const RandomForestParams params;
    int num_classes = 0;
    std::vector<DecisionTreeClassifier> trees;

    void bootstrap_sample(size_t n_samples, size_t total_features, std::vector<int> &indices) const;

    [[nodiscard]] static float f1_score(const std::vector<int> &y, const std::vector<int> &y_pred) ;
};


#endif //RANDOMFORESTCLASSIFIER_H
