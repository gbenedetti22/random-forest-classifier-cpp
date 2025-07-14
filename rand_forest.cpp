#include <iostream>
#include <vector>
#include <map>
#include <limits>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <unordered_map>
#include <sstream>

using namespace std;

struct Sample {
    vector<double> features;
    int label;
};

// Nodo dell’albero decisionale
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

// Albero decisionale ricorsivo
class DecisionTree {
    TreeNode* root;
    int max_depth;

public:
    DecisionTree(int depth = 5) : root(nullptr), max_depth(depth) {}

    void train(const vector<Sample>& data) {
        root = build_tree(data, 0);
    }

    int predict(const Sample& s) const {
        TreeNode* node = root;
        while (!node->is_leaf) {
            if (s.features[node->feature_index] < node->threshold)
                node = node->left;
            else
                node = node->right;
        }
        return node->predicted_class;
    }

private:
    TreeNode* build_tree(const vector<Sample>& data, int depth) {
        TreeNode* node = new TreeNode();

        map<int, int> label_counts;
        for (const auto& s : data) label_counts[s.label]++;

        if (label_counts.size() == 1 || depth >= max_depth) {
            node->is_leaf = true;
            node->predicted_class = get_majority_class(label_counts);
            return node;
        }

        int best_feature = -1;
        double best_threshold = 0.0;
        int best_error = numeric_limits<int>::max();
        vector<Sample> best_left, best_right;
        int num_features = data[0].features.size();

        for (int f = 0; f < num_features; ++f) {
            double sum = 0.0;
            for (const auto& s : data) {
                sum += s.features[f];
            }
            double th = sum / data.size();

            vector<Sample> left, right;
            map<int, int> left_counts, right_counts;
            for (const auto& s : data) {
                if (s.features[f] < th) {
                    left.push_back(s);
                    left_counts[s.label]++;
                } else {
                    right.push_back(s);
                    right_counts[s.label]++;
                }
            }

            int error = compute_misclassification(left_counts, left)
                      + compute_misclassification(right_counts, right);

            if (error < best_error && !left.empty() && !right.empty()) {
                best_error = error;
                best_feature = f;
                best_threshold = th;
                best_left = left;
                best_right = right;
            }
        }

        if (best_feature == -1) {
            node->is_leaf = true;
            node->predicted_class = get_majority_class(label_counts);
            return node;
        }

        node->is_leaf = false;
        node->feature_index = best_feature;
        node->threshold = best_threshold;
        node->left = build_tree(best_left, depth + 1);
        node->right = build_tree(best_right, depth + 1);
        return node;
    }

    int get_majority_class(const map<int, int>& counts) const {
        int majority_class = -1, max_count = -1;
        for (const auto& [label, count] : counts) {
            if (count > max_count) {
                max_count = count;
                majority_class = label;
            }
        }
        return majority_class;
    }

    int compute_misclassification(const map<int, int>& counts, const vector<Sample>& subset) const {
        if (subset.empty()) return 0;
        int majority = get_majority_class(counts);
        int error = 0;
        for (const auto& s : subset)
            if (s.label != majority) error++;
        return error;
    }
};

// Random Forest
class RandomForest {
    vector<DecisionTree> trees;
    int num_trees;
    int max_depth;

public:
    RandomForest(int n = 10, int depth = 5) : num_trees(n), max_depth(depth) {
        srand(time(nullptr));
    }

    void train(const vector<Sample>& data) {
        for (int i = 0; i < num_trees; ++i) {
            vector<Sample> sample = bootstrap_sample(data);
            DecisionTree tree(max_depth);
            tree.train(sample);
            trees.push_back(tree);
        }
    }

    int predict(const Sample& s) const {
        map<int, int> vote_counts;
        for (const auto& tree : trees) {
            int pred = tree.predict(s);
            vote_counts[pred]++;
        }

        int majority_class = -1, max_votes = -1;
        for (const auto& [label, count] : vote_counts) {
            if (count > max_votes) {
                max_votes = count;
                majority_class = label;
            }
        }
        return majority_class;
    }

private:
    vector<Sample> bootstrap_sample(const vector<Sample>& data) {
        vector<Sample> sample;
        for (size_t i = 0; i < data.size(); ++i) {
            int idx = rand() % data.size();
            sample.push_back(data[idx]);
        }
        return sample;
    }
};

// === MAIN ===

vector<Sample> loadDataset(const string& filename) {
    ifstream file(filename);
    vector<Sample> dataset;
    unordered_map<string, int> label_to_int;
    int label_counter = 0;

    string line;
    int row = 0;
    while (getline(file, line)) {
        if (line.empty()) continue;

        stringstream ss(line);
        string field;
        vector<double> features;
        int col = 0;
        string label_str;

        while (getline(ss, field, ',')) {
            if (col < 4) {
                features.push_back(stod(field));
            } else {
                label_str = field;
            }
            col++;
        }

        if (label_to_int.find(label_str) == label_to_int.end()) {
            label_to_int[label_str] = label_counter++;
        }

        int numeric_label = label_to_int[label_str];
        dataset.push_back({features, numeric_label});
        row++;

    }

    return dataset;
}

int main() {
    const vector<Sample> dataset = loadDataset("dataset.data");

    RandomForest forest(10, 3); // 10 alberi, profondità max = 3
    forest.train(dataset);

    const vector<Sample> test_samples = {
        {dataset.front().features}, {dataset.back().features}
    };

    for (const auto& s : test_samples) {
        int pred = forest.predict(s);
        cout << "Predizione: classe " << pred << endl;
    }

    return 0;
}
