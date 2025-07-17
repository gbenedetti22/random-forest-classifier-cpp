//
// Created by gabriele on 13/07/25.
//

#ifndef DECISIONTREECLASSIFIER_H
#define DECISIONTREECLASSIFIER_H
#include <map>
#include <utility>
#include <vector>
#include <string>
#include <unordered_map>

using SplitResult = std::tuple<std::vector<std::vector<double>>, std::map<int, int>, std::vector<std::vector<double>>, std::map<int, int>, std::vector<int>, std::vector<int>>;
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

    void build_tree(const std::vector<std::vector<double>>& X, const std::vector<int>& y);

    static auto split_left_right(const std::vector<std::vector<double>> &X, const std::vector<int> &y, double th, int f)->SplitResult;
    static std::pair<double, double> compute_treshold(const std::vector<std::vector<double> > &X,
                                                      const std::vector<int> &y, int f);
    static int compute_majority_class(const std::map<int, int>& counts);
    static int compute_error(const std::map<int, int>& counts, const std::vector<int>& y_test);

    static double gini_index(const std::map<int, int> &counts, int total);

public:
    std::string id;
    DecisionTreeClassifier() : root(nullptr){}
    explicit DecisionTreeClassifier(std::string id) : root(nullptr), id(std::move(id)) {}
    void train(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
    int predict(const std::vector<double>& x) const;
    void set_class_weights(const std::unordered_map<int, double>& weights);
};



#endif //DECISIONTREECLASSIFIER_H
