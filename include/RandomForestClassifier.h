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
    void fit(const std::vector<Sample>& data);
    int predict(const Sample& s) const;
    void score(const std::vector<Sample>& data) const;
private:
    std::vector<DecisionTreeClassifier> trees;

    static std::vector<Sample> bootstrap_sample(const std::vector<Sample>& data);
};



#endif //RANDOMFORESTCLASSIFIER_H
