//
// Created by gabriele on 28/09/25.
//

#ifndef DECISION_TREE_FARMFF_HPP
#define DECISION_TREE_FARMFF_HPP
#include <../ff/ff.hpp>

#include "../DecisionTreeClassifier.h"

class FarmFF {
    struct Task {
        size_t id;
        const std::vector<float>& samples;
    };

    struct SamplePrediction {
        size_t sample_id;
        int prediction;
    };

    struct Emitter final : ff::ff_node_t<Task> {
        const std::vector<std::vector<float> > &X;
        size_t currentSample = 0;

        explicit Emitter(const std::vector<std::vector<float> > &x)
            : X(x) {
        }

        Task* svc(Task *) override {
            if (currentSample >= X.size()) {
                return EOS;
            }

            ff_send_out(new Task(currentSample, X[currentSample]));
            currentSample++;
            return GO_ON;
        }

    };

    struct Worker final : ff::ff_node_t<Task, SamplePrediction> {
        const std::vector<DecisionTreeClassifier>& trees;

        explicit Worker(const std::vector<DecisionTreeClassifier> &trees)
            : trees(trees) {
        }

        SamplePrediction* svc(Task* task) override {
            std::unordered_map<int, int> votes;

            for (const auto& tree : trees) {
                int pred = tree.predict(task->samples.data());
                votes[pred]++;
            }

            int best_class = -1;
            int max_votes = 0;
            for (const auto& [label, count] : votes) {
                if (count > max_votes) {
                    max_votes = count;
                    best_class = label;
                }
            }

            const size_t sample_id = task->id;
            ff_send_out(new SamplePrediction(sample_id, best_class));
            delete task;

            return GO_ON;
        }
    };

    struct Collector final : ff::ff_node_t<SamplePrediction> {
        SamplePrediction *svc(SamplePrediction* pred) override {
            totalVotes[pred->sample_id] = pred->prediction;

            delete pred;
            return GO_ON;
        }

        std::unordered_map<size_t, int>& getVotes() {
            return totalVotes;
        }

    private:
        std::unordered_map<size_t, int> totalVotes;
    };


public:
    static std::unordered_map<size_t, int>& getVotes(const std::vector<DecisionTreeClassifier> &trees,
                                                     const std::vector<std::vector<float> > &samples, const int nworkers) {
        ff::ff_farm farm;

        farm.add_emitter(new Emitter(samples));

        std::vector<ff::ff_node*> workers;
        workers.reserve(nworkers);
        for (int i = 0; i < nworkers; i++) {
            workers.push_back(new Worker(trees));
        }
        farm.add_workers(workers);

        auto collector = new Collector();
        farm.add_collector(collector);

        if (farm.run_and_wait_end() < 0) {
            perror("Error while running farm\n");
            exit(EXIT_FAILURE);
        }

        return collector->getVotes();
    }
};


#endif //DECISION_TREE_FARMFF_HPP