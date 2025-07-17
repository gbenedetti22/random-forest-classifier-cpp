//
// Created by gabriele on 13/07/25.
//

#ifndef RANDOMFORESTCLASSIFIER_H
#define RANDOMFORESTCLASSIFIER_H
#include <vector>
#include "Dataset.h"
#include "DecisionTreeClassifier.h"

class RandomForestClassifier {
public:
    explicit RandomForestClassifier(int n_tress = 10);
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
    int predict(const std::vector<double>& x) const;
    double evaluate(const std::vector<std::vector<double>>& X, const std::vector<int>& y) const;
private:
    std::vector<DecisionTreeClassifier> trees;
    static void bootstrap_sample(const std::vector<std::vector<double>>& X, const std::vector<int>& y, std::vector<std::vector<double>>& X_sample, std::vector<int>& y_sample);
};



#endif //RANDOMFORESTCLASSIFIER_H
