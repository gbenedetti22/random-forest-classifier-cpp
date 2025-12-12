
//
// Created by gabriele on 29/09/25.
//

#ifndef DECISION_TREE_PIPELINEFF_HPP
#define DECISION_TREE_PIPELINEFF_HPP
#include <vector>

#include "../DecisionTreeClassifier.h"
#include "../ff/ff.hpp"

// Parallel prediction using FastFlow's Pipeline pattern.
// Stages process samples sequentially: Input -> Tree Predictions -> Aggregation.
class PipelineFF {
    struct SampleData {
        const std::vector<float>& sample;
        int sample_id;
        std::vector<int> predictions;

    };

    // First stage: Feeds samples into the pipeline one by one.
    struct InputStage final : ff::ff_node_t<SampleData> {
        const std::vector<std::vector<float>>& samples;
        int current_sample;
        int num_classes;

        explicit InputStage(const std::vector<std::vector<float> > &s, const int num_classes) : samples(s), current_sample(0),
            num_classes(num_classes) {
        }

        SampleData* svc(SampleData*) override {
            if(current_sample >= samples.size()) {
                return EOS;
            }

            const auto data = new SampleData(samples[current_sample], current_sample);
            data->predictions.assign(num_classes, 0);
            current_sample++;
            return data;
        }
    };

    // Intermediate stage: Represents a single decision tree.
    // Updates the prediction vector for the passing sample.
    struct TreeStage final : ff::ff_node_t<SampleData> {
        const std::vector<DecisionTreeClassifier> &trees;
        const int tree_id;

        TreeStage(const std::vector<DecisionTreeClassifier> &trees, const int tree_id)
            : trees(trees),
              tree_id(tree_id) {
        }

        SampleData* svc(SampleData* data) override {
            const int prediction = trees[tree_id].predict(data->sample.data());
            data->predictions[prediction]++;

            return data;
        }
    };

    // Final stage: Aggregates votes and determines the final class label.
    struct AggregationStage final : ff::ff_node_t<SampleData> {
        std::vector<int>& final_results;
        explicit AggregationStage(std::vector<int>& results) : final_results(results) {}

        SampleData *svc(SampleData *data) override {
            int max_votes = -1;
            int majority_class = -1;

            for (int label = 0; label < data->predictions.size(); label++) {
                if (data->predictions[label] > max_votes) {
                    max_votes = data->predictions[label];
                    majority_class = label;
                }
            }
            final_results[data->sample_id] = majority_class;

            delete data;
            return GO_ON;
        }
    };

public:
    static std::vector<int> getVotes(
    const std::vector<std::vector<float>>& samples,
    const std::vector<DecisionTreeClassifier>& trees, const int num_classes) {

        std::vector<int> results(samples.size());

        ff::ff_pipeline pipe;

        pipe.add_stage(new InputStage(samples, num_classes));

        for(int i = 0; i < trees.size(); i++) {
            pipe.add_stage(new TreeStage(trees, i));
        }

        pipe.add_stage(new AggregationStage(results));

        if (pipe.run_and_wait_end() < 0) {
            perror("running pipeline\n");
            return {};
        }

        return results;
    }
};
#endif //DECISION_TREE_PIPELINEFF_HPP