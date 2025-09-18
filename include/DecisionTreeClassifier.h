//
// Created by gabriele on 13/07/25.
//

#ifndef DECISIONTREECLASSIFIER_H
#define DECISIONTREECLASSIFIER_H
#include <unordered_map>
#include <random>
#include <vector>
#include <string>
#include <variant>

#include "TrainMatrix.hpp"

using SplitResult = std::tuple<std::vector<int>, std::vector<int>>;
struct TreeNode {
    bool is_leaf;
    int predicted_class;
    int feature_index;
    float threshold;
    TreeNode* left;
    TreeNode* right;

    TreeNode() : is_leaf(true), predicted_class(0), feature_index(-1),
                 threshold(0.0), left(nullptr), right(nullptr) {}
};

class DecisionTreeClassifier {
    TreeNode* root;
    std::mt19937 rng;

    void build_tree(TrainMatrix &X, std::vector<int> &y);

    static size_t split_left_right(TrainMatrix &X, std::vector<int> &y, size_t start, size_t end, float th, size_t f);

    [[nodiscard]] std::tuple<float, float, size_t> compute_threshold(const TrainMatrix &X,
                                                              const std::vector<int> &y, size_t start, size_t end,
                                                              int f,
                                                              const std::unordered_map<int, int> &label_counts, int
                                                              num_classes) const;
    static int compute_majority_class(const std::unordered_map<int, int> &counts);

    static float gini(const std::vector<int> &counts, size_t total);

    static float entropy(const std::vector<int> &counts, size_t total);

    [[nodiscard]] float get_impurity(const std::vector<int> &counts, size_t total) const;

    std::vector<int> sample_features(size_t total_features, size_t n_features);

public:
    const std::string &split_criteria;
    int min_samples_split;
    const std::variant<int, std::string>& max_features;
    float min_samples_ratio;
    int nworkers;

    DecisionTreeClassifier(const std::string &split_criteria, const int min_samples_split, const std::variant<int, std::string> &max_features,
            const unsigned int random_seed, const float min_samples_ratio, const int nworkers)
            : root(nullptr), split_criteria(split_criteria),
              min_samples_split(min_samples_split),
              max_features(max_features), min_samples_ratio(min_samples_ratio), nworkers(nworkers) {
        rng = std::mt19937(random_seed);
    }

    void train(const std::vector<float> &X, const std::pair<long, long> &shape, const std::vector<int> &y, std::vector<int> &indices);
    [[nodiscard]] int predict(const std::vector<float>& x) const;
};



#endif //DECISIONTREECLASSIFIER_H
