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
#include <TrainMatrix.hpp>
#include <variant>

#include "utils.h"

struct TreeNode {
    bool is_leaf;
    int predicted_class;
    int feature_index;
    float threshold;
    TreeNode* left;
    TreeNode* right;

    TreeNode() : is_leaf(false), predicted_class(0), feature_index(-1),
                 threshold(0.0), left(nullptr), right(nullptr) {}
};

struct DTreeParams {
    const std::string& split_criteria;
    int min_samples_split;
    const std::variant<int, std::string>& max_features;
    float min_samples_ratio;
    int nworkers;
    int max_depth;
    size_t max_leaf_nodes;
};

class DecisionTreeClassifier {
    TreeNode* root;
    std::mt19937 rng;
    DTreeParams params;

    void build_tree(const TrainMatrix &X, const std::vector<int> &y, std::vector<int> &samples);

    void debug_tree() const;

    static size_t split_left_right(
        const TrainMatrix &X, std::vector<int> &indices, size_t start, size_t end, float th, int f);

    std::tuple<float, float, unsigned long> compute_threshold(const TrainMatrix &X,
                                                              const std::vector<int> &y, const std::vector<int> &indices, size_t start, size_t end, int f, const std::unordered_map<
                                                              int, int> &label_counts, int
                                                              num_classes) const;
    static int compute_majority_class(const std::unordered_map<int, int> &counts);

    static float gini(const std::vector<int> &counts, int total);

    static float entropy(const std::vector<int> &counts, int total);

    [[nodiscard]] float get_impurity(const std::vector<int> &counts, int total) const;

    std::vector<int> sample_features(int total_features, int n_features);

public:
    DecisionTreeClassifier(const DecisionTreeClassifier&) = delete;
    DecisionTreeClassifier& operator=(const DecisionTreeClassifier&) = delete;

    DecisionTreeClassifier(DecisionTreeClassifier&& other) noexcept
            : root(other.root), rng(other.rng),
              params(other.params) {
        other.root = nullptr;
    }

    DecisionTreeClassifier& operator=(DecisionTreeClassifier&& other) noexcept {
        if (this != &other) {
            this->~DecisionTreeClassifier();
            root = other.root;
            rng = other.rng;
            params.min_samples_split = other.params.min_samples_split;
            params.min_samples_ratio = other.params.min_samples_ratio;
            params.nworkers = other.params.nworkers;
            other.root = nullptr;
        }
        return *this;
    }

DecisionTreeClassifier(const DTreeParams &params, const uint32_t &random_seed)
        : root(nullptr), params(params){
        rng = std::mt19937(random_seed);
    }

    ~DecisionTreeClassifier();
    void train(const std::vector<float> &X, const std::pair<unsigned long, unsigned long> &shape, const std::vector<int> &y, std::
               vector<int> &samples);
    [[nodiscard]] int predict(const std::vector<float>& x) const;
};



#endif //DECISIONTREECLASSIFIER_H
