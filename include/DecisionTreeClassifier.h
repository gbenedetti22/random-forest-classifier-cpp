//
// Created by gabriele on 13/07/25.
//

#ifndef DECISIONTREECLASSIFIER_H
#define DECISIONTREECLASSIFIER_H
#include <map>
#include <utility>
#include <vector>

#include "Dataset.h"

using SplitResult = std::tuple<std::vector<Sample>, std::map<int, int>, std::vector<Sample>, std::map<int, int>>;
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
    void build_tree(const std::vector<Sample>& dataset);
    static auto split_left_right(const std::vector<Sample> &data, double th, int f)->SplitResult;
    static double compute_treshold(const std::vector<Sample>& data, int i);
    static int compute_majority_class(const std::map<int, int>& counts);
    static int compute_error(const std::map<int, int>& counts, const std::vector<Sample>& test);

public:
    std::string id;
    DecisionTreeClassifier() : root(nullptr){}
    explicit DecisionTreeClassifier(std::string id) : root(nullptr), id(std::move(id)) {}
    void train(const std::vector<Sample>& data);
    int predict(const Sample& s) const;
};



#endif //DECISIONTREECLASSIFIER_H
