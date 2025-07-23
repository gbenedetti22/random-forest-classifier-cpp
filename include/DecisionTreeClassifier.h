//
// Created by gabriele on 13/07/25.
//

#ifndef DECISIONTREECLASSIFIER_H
#define DECISIONTREECLASSIFIER_H
#include <map>
#include <optional>
#include <random>
#include <utility>
#include <vector>
#include <string>
#include <unordered_map>
#include <variant>

using SplitResult = std::tuple<std::vector<int>, std::vector<int>>;
struct TreeNode {
    bool is_leaf;
    int predicted_class;
    int feature_index;
    double threshold;
    TreeNode* left;
    TreeNode* right;

    TreeNode() : is_leaf(true), predicted_class(0), feature_index(-1),
                 threshold(0.0), left(nullptr), right(nullptr) {}
};

class DecisionTreeClassifier {
    TreeNode* root;
    std::mt19937 rng;

    void build_tree(const std::vector<std::vector<double>> &X, const std::vector<int> &y, std::vector<int> &samples, std::vector<std::
                    vector<int>> &labels_mapping);

    static auto split_left_right(const std::vector<std::vector<double>> &X, const std::vector<int> &y, std::vector<int> &indices, double th, int f)->SplitResult;
    std::pair<double, double> compute_treshold(const std::vector<std::vector<double>> &X,
                                               const std::vector<int> &y, std::vector<int> &indices, int f);
    static int compute_majority_class(const std::map<int, int>& counts);
    static int compute_error(const std::map<int, int>& counts, const std::vector<int>& y_test);

    static double gini(const std::map<int, int> &counts, int total);

    static double entropy(const std::map<int, int> &counts, int total);

    double get_impurity(const std::map<int, int> &counts, int total);

    std::vector<int> sample_features(int total_features, int n_features);

public:
    const std::string &split_criteria;
    int min_samples_split;
    const std::variant<int, std::string>& max_features;
    const std::optional<int> random_seed;

DecisionTreeClassifier(const std::string &split_criteria, const int min_samples_split, const std::variant<int, std::string> &max_features,
        const std::optional<int> &random_seed)
        : root(nullptr), split_criteria(split_criteria),
          min_samples_split(min_samples_split),
          max_features(max_features),
          random_seed(random_seed) {
    if (random_seed.has_value()) {
        rng = std::mt19937(random_seed.value());
    }
    }

    void train(const std::vector<std::vector<double>> &X, const std::vector<int> &y, std::vector<int> &samples, std::vector<std::
               vector<int>> &labels_mapping);
    int predict(const std::vector<double>& x) const;
};



#endif //DECISIONTREECLASSIFIER_H
