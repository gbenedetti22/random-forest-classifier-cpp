//
// Created by gabriele on 13/07/25.
//

#ifndef RANDOMFORESTCLASSIFIER_H
#define RANDOMFORESTCLASSIFIER_H
#include <optional>
#include <variant>
#include <vector>
#include <DecisionTreeClassifier.h>

// Ensemble classifier using Bagging (Bootstrap Aggregating) of Decision Trees.
class RandomForestClassifier {
public:
    // Hyperparameters for the Random Forest.
    struct RandomForestParams {
        // Number of trees in the forest.
        int n_trees = 10;
        std::string split_criteria = "gini";
        int min_samples_split = 2;
        const std::variant<int, std::string> max_features = "sqrt";
        bool bootstrap = true;
        std::optional<int> random_seed = std::nullopt;
        float min_samples_ratio = 0.2f;
        int max_depth = INT_MAX;
        size_t max_leaf_nodes = SIZE_MAX;
        const std::variant<size_t, float> max_samples = -1.0F;
        int njobs = 1;
        int nworkers = 1;
    };

    explicit RandomForestClassifier(const RandomForestParams &params)
        : params(params) {
        trees.reserve(params.n_trees);
    }

    // Trains the Random Forest on the provided dataset.
    // Can optionally run in parallel using threads or other backends.
    void fit(const std::vector<std::vector<float>> &X, const std::vector<int> &y, bool transposed = false);

    void fit(std::vector<float> &X, const std::vector<int> &y, std::pair<unsigned long, unsigned long> &shape, bool transposed =
                     false);

    // Predicts the class for new samples by aggregating votes from all trees.
    [[nodiscard]] std::vector<int> predict(const std::vector<float> &X,
                                           const std::pair<size_t, size_t> &shape) const;

    [[nodiscard]] std::pair<float, float> score(const std::vector<std::vector<float>> &X, const std::vector<int> &y) const;

    [[nodiscard]] std::pair<float, float> score(const std::vector<float> &X, const std::vector<int> &y, const std::pair<size_t, size_t> &shape) const;


private:
    const RandomForestParams params;
    int num_classes = 0;
    std::vector<DecisionTreeClassifier> trees;

    // Generates a bootstrap sample (random sampling with replacement) for training a single tree.
    void bootstrap_sample(size_t n_samples, size_t total_features, std::vector<int> &indices) const;

};


#endif //RANDOMFORESTCLASSIFIER_H
