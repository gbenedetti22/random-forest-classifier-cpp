//
// Created by gabriele on 28/09/25.
//

#ifndef DECISION_TREE_FARMFF_HPP
#define DECISION_TREE_FARMFF_HPP
#include <../ff/ff.hpp>

#include "../DecisionTreeClassifier.h"

class FarmFF {
    struct Task {
        size_t sample_start;
        size_t sample_end;
        size_t tree_start;
        size_t tree_end;
    };

    struct SamplePrediction {
        size_t sample_id;
        std::vector<int> predictions;
    };

    struct Emitter final : ff::ff_node_t<Task> {
        const std::vector<float>& X;
        const std::pair<size_t, size_t>& shape;
        const size_t num_trees;
        const int nworkers;
        const size_t samples_per_task;
        const size_t trees_per_task;
        size_t tasks_sent = 0;
        size_t total_tasks = 0;

        explicit Emitter(const std::vector<float>& x, const std::pair<size_t, size_t> &shape, const size_t num_trees, const int nworkers)
            : X(x), shape(shape), num_trees(num_trees), nworkers(nworkers),
              samples_per_task(std::max(1UL, shape.first / nworkers)),
              trees_per_task(std::max(1UL, num_trees / nworkers)) {

            const size_t num_sample_chunks = (shape.first + samples_per_task - 1) / samples_per_task;
            const size_t num_tree_chunks = (num_trees + trees_per_task - 1) / trees_per_task;
            total_tasks = num_sample_chunks * num_tree_chunks;
        }

        Task* svc(Task*) override {
            if (tasks_sent >= total_tasks) {
                return EOS;
            }

            // Calcola chunk di campioni e alberi per questo task
            const size_t sample_chunk_idx = tasks_sent / ((num_trees + trees_per_task - 1) / trees_per_task);
            const size_t tree_chunk_idx = tasks_sent % ((num_trees + trees_per_task - 1) / trees_per_task);

            const size_t sample_start = sample_chunk_idx * samples_per_task;
            const size_t sample_end = std::min(sample_start + samples_per_task, shape.first);

            const size_t tree_start = tree_chunk_idx * trees_per_task;
            const size_t tree_end = std::min(tree_start + trees_per_task, num_trees);

            ff_send_out(new Task{sample_start, sample_end, tree_start, tree_end});
            tasks_sent++;

            return GO_ON;
        }
    };

    struct Worker final : ff::ff_node_t<Task, SamplePrediction> {
        const std::vector<DecisionTreeClassifier>& trees;
        const std::vector<float>& X;
        const std::pair<size_t, size_t> shape;

        explicit Worker(const std::vector<DecisionTreeClassifier>& trees,
                       const std::vector<float>& x,
                       const std::pair<size_t, size_t> &shape)
            : trees(trees), X(x), shape(shape) {
        }

        SamplePrediction* svc(Task* task) override {
            for (size_t sample_idx = task->sample_start; sample_idx < task->sample_end; sample_idx++) {
                std::vector<int> predictions;
                predictions.reserve(task->tree_end - task->tree_start);

                const float* sample_ptr = X.data() + sample_idx * shape.second;

                for (size_t tree_idx = task->tree_start; tree_idx < task->tree_end; tree_idx++) {
                    int pred = trees[tree_idx].predict(sample_ptr);
                    predictions.push_back(pred);
                }

                ff_send_out(new SamplePrediction{sample_idx, std::move(predictions)});
            }

            delete task;
            return GO_ON;
        }
    };

    struct Collector final : ff::ff_node_t<SamplePrediction> {
        const size_t num_samples;
        const size_t num_trees;
        std::unordered_map<size_t, std::unordered_map<int, int>> sample_votes;
        std::unordered_map<size_t, size_t> predictions_received;
        std::vector<int> final_predictions;

        explicit Collector(const size_t num_samples, const size_t num_trees)
            : num_samples(num_samples), num_trees(num_trees) {
            final_predictions.resize(num_samples, 0);
        }

        SamplePrediction* svc(SamplePrediction* pred) override {
            const size_t sample_id = pred->sample_id;

            for (int prediction : pred->predictions) {
                sample_votes[sample_id][prediction]++;
            }

            predictions_received[sample_id] += pred->predictions.size();

            if (predictions_received[sample_id] == num_trees) {
                int best_class = -1;
                int max_votes = 0;
                for (const auto& [label, count] : sample_votes[sample_id]) {
                    if (count > max_votes) {
                        max_votes = count;
                        best_class = label;
                    }
                }
                final_predictions[sample_id] = best_class;

                sample_votes.erase(sample_id);
                predictions_received.erase(sample_id);
            }

            delete pred;
            return GO_ON;
        }

        std::vector<int>& getPredictions() {
            return final_predictions;
        }
    };

public:
    static std::vector<int>& getVotes(const std::vector<float>& samples, std::pair<size_t, size_t> shape,
                                      const std::vector<DecisionTreeClassifier>& trees, int nworkers) {
        ff::ff_farm farm;

        farm.add_emitter(new Emitter(samples, shape, trees.size(), nworkers));

        std::vector<ff::ff_node*> workers;
        workers.reserve(nworkers);
        for (int i = 0; i < nworkers; i++) {
            workers.push_back(new Worker(trees, samples, shape));
        }
        farm.add_workers(workers);

        auto collector = new Collector(shape.first, trees.size());
        farm.add_collector(collector);

        farm.no_mapping();
        farm.blocking_mode(false);

        if (farm.run_and_wait_end() < 0) {
            perror("Error while running farm\n");
            exit(EXIT_FAILURE);
        }

        return collector->getPredictions();
    }
};

#endif //DECISION_TREE_FARMFF_HPP