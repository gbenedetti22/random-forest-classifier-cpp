//
// Created by gabriele on 13/07/25.
//

#ifndef DECISIONTREECLASSIFIER_H
#define DECISIONTREECLASSIFIER_H
#include <unordered_map>
#include <optional>
#include <random>
#include <utility>
#include <vector>
#include <string>
#include <variant>

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

    void build_tree(const std::vector<std::vector<float>> &X, const std::vector<int> &y, std::vector<int> &samples);

    static std::tuple<std::vector<int>, std::vector<int>> split_left_right(const std::vector<std::vector<float>> &X, const std::vector<int> &indices, float th, int f);
    std::pair<float, float> compute_threshold(const std::vector<std::vector<float>> &X,
                                              const std::vector<int> &y, std::vector<int> &indices, int f, const std::unordered_map<int, int> &label_counts, int
                                              num_classes) const;
    static int compute_majority_class(const std::unordered_map<int, int> &counts);

    static float gini(const std::vector<int> &counts, int total);

    static float entropy(const std::vector<int> &counts, int total);

    [[nodiscard]] float get_impurity(const std::vector<int> &counts, int total) const;

    std::vector<int> sample_features(int total_features, int n_features);

public:
    const std::string &split_criteria;
    int min_samples_split;
    const std::variant<int, std::string>& max_features;
    const std::optional<int> random_seed;
    float min_samples_ratio;
    int nworkers;

DecisionTreeClassifier(const std::string &split_criteria, const int min_samples_split, const std::variant<int, std::string> &max_features,
        const std::optional<int> &random_seed, const float min_samples_ratio, const int nworkers)
        : root(nullptr), split_criteria(split_criteria),
          min_samples_split(min_samples_split),
          max_features(max_features),
          random_seed(random_seed), min_samples_ratio(min_samples_ratio), nworkers(nworkers) {
        if (random_seed.has_value()) {
            rng = std::mt19937(random_seed.value());
        }
    }

    void train(const std::vector<std::vector<float>> &X, const std::vector<int> &y, std::vector<int> &samples);
    [[nodiscard]] int predict(const std::vector<float>& x) const;
};



#endif //DECISIONTREECLASSIFIER_H
