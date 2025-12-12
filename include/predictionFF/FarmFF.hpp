//
// Created by gabriele on 28/09/25.
//

#ifndef DECISION_TREE_FARMFF_HPP
#define DECISION_TREE_FARMFF_HPP
#include <../ff/ff.hpp>
#include <chrono>

#include "../DecisionTreeClassifier.h"

// Parallel prediction using FastFlow's Farm pattern.
// Distributes prediction tasks (chunks of samples/trees) to workers.
class FarmFF {
    struct SampleVotes {
        size_t sample_id;
        std::vector<int> votes;  // votes[label] = count. Aggregated votes for a sample.
    };

    // A task represents a subset of samples to be predicted by a subset of trees.
    struct Task {
        size_t sample_start;
        size_t sample_end;
        size_t tree_start;
        size_t tree_end;
        bool done; // Flag to indicate task completion/result return.
        std::vector<SampleVotes> results;
    };

    // Emitter node: Generates tasks by slicing samples and trees into chunks.
    // Schedules tasks to workers.
    struct Emitter final : ff::ff_node_t<Task> {
        const std::vector<float>& X;
        const std::pair<size_t, size_t>& shape;
        const size_t num_trees;
        const int nworkers;
        const std::vector<DecisionTreeClassifier>& trees;
        const int num_classes;
        const size_t samples_per_task;
        const size_t trees_per_task;
        size_t total_tasks = 0;
        size_t num_tree_chunks = 0;

        explicit Emitter(const std::vector<float>& x, const std::pair<size_t, size_t> &shape,
                         const size_t num_trees, const int nworkers,
                         const std::vector<DecisionTreeClassifier>& trees, const int num_classes)
            : X(x), shape(shape), num_trees(num_trees), nworkers(nworkers),
              trees(trees), num_classes(num_classes),
              samples_per_task(std::max(1UL, shape.first / (nworkers + 1))), // Diviso nworkers+1
              trees_per_task(std::max(1UL, num_trees / (nworkers + 1))) { // Diviso nworkers+1

            const size_t num_sample_chunks = (shape.first + samples_per_task - 1) / samples_per_task;
            num_tree_chunks = (num_trees + trees_per_task - 1) / trees_per_task;
            total_tasks = num_sample_chunks * num_tree_chunks;
        }

        Task* svc(Task*) override {
            const size_t tasks_for_emitter = total_tasks / (nworkers + 1);

            std::vector<Task*> tasks;
            tasks.reserve(tasks_for_emitter);

            for (size_t task_idx = 0; task_idx < total_tasks; ++task_idx) {
                const size_t sample_chunk_idx = task_idx / num_tree_chunks;
                const size_t tree_chunk_idx = task_idx % num_tree_chunks;

                const size_t sample_start = sample_chunk_idx * samples_per_task;
                const size_t sample_end = std::min(sample_start + samples_per_task, shape.first);

                const size_t tree_start = tree_chunk_idx * trees_per_task;
                const size_t tree_end = std::min(tree_start + trees_per_task, num_trees);

                auto* task = new Task{sample_start, sample_end, tree_start, tree_end, false, {}};

                if (task_idx >= total_tasks - tasks_for_emitter) {
                    tasks.push_back(task);
                    continue;
                }

                ff_send_out(task);
            }

            for (const auto& task : tasks) {
                task->done = true;
                task->results.reserve(task->sample_end - task->sample_start);

                for (size_t sample_idx = task->sample_start; sample_idx < task->sample_end; sample_idx++) {
                    std::vector<int> predictions;
                    predictions.reserve(task->tree_end - task->tree_start);

                    const float* sample_ptr = X.data() + sample_idx * shape.second;

                    for (size_t tree_idx = task->tree_start; tree_idx < task->tree_end; tree_idx++) {
                        int pred = trees[tree_idx].predict(sample_ptr);
                        predictions.push_back(pred);
                    }

                    std::vector local_votes(num_classes, 0);
                    for (const int prediction : predictions) {
                        local_votes[prediction]++;
                    }
                    task->results.push_back(SampleVotes{sample_idx, std::move(local_votes)});
                }

                ff_send_out(task);
            }

            return EOS;
        }
    };

    // Worker node: Executes the prediction for the assigned task.
    // Computes votes for the given sample range and tree range.
    struct Worker final : ff::ff_node_t<Task, SampleVotes> {
        const std::vector<DecisionTreeClassifier>& trees;
        const std::vector<float>& X;
        const std::pair<size_t, size_t> shape;
        const int num_classes;

        std::chrono::duration<double, std::milli> total_time{0};
        size_t svc_calls = 0;

        explicit Worker(const std::vector<DecisionTreeClassifier>& trees,
                       const std::vector<float>& x,
                       const std::pair<size_t, size_t> &shape, const int num_classes)
            : trees(trees), X(x), shape(shape), num_classes(num_classes) {
        }

        SampleVotes* svc(Task* task) override {
            if (task->done) {
                for (auto&[sample_id, votes] : task->results) {
                    ff_send_out(new SampleVotes{sample_id, std::move(votes)});
                }
            } else {
                for (size_t sample_idx = task->sample_start; sample_idx < task->sample_end; sample_idx++) {
                    std::vector<int> predictions;
                    predictions.reserve(task->tree_end - task->tree_start);

                    const float* sample_ptr = X.data() + sample_idx * shape.second;

                    for (size_t tree_idx = task->tree_start; tree_idx < task->tree_end; tree_idx++) {
                        int pred = trees[tree_idx].predict(sample_ptr);
                        predictions.push_back(pred);
                    }

                    std::vector local_votes(num_classes, 0);
                    for (const int prediction : predictions) {
                        local_votes[prediction]++;
                    }

                    ff_send_out(new SampleVotes{sample_idx, std::move(local_votes)});
                }
            }

            delete task;
            return GO_ON;
        }
    };

    // Collector node: Aggregates votes from all workers for each sample.
    // Final classification is done after collecting all votes.
    struct Collector final : ff::ff_node_t<SampleVotes> {
        const size_t num_samples;
        const size_t num_trees;
        std::vector<std::vector<int>> sample_votes;

        explicit Collector(const size_t num_samples, const size_t num_trees, const int num_classes)
            : num_samples(num_samples), num_trees(num_trees) {
            sample_votes.assign(num_samples, std::vector(num_classes, 0));
        }

        SampleVotes* svc(SampleVotes* votes) override {
            const size_t sample_id = votes->sample_id;

            for (int label = 0; label < votes->votes.size(); label++) {
                sample_votes[sample_id][label] += votes->votes[label];
            }

            delete votes;

            return GO_ON;
        }
    };

public:
    static std::vector<int> getVotes(const std::vector<float>& samples, std::pair<size_t, size_t> shape,
                                      const std::vector<DecisionTreeClassifier>& trees, int nworkers, int num_classes) {
        ff::ff_farm farm;

        auto emitter = new Emitter(samples, shape, trees.size(), nworkers, trees, num_classes);
        farm.add_emitter(emitter);

        std::vector<ff::ff_node*> workers;
        workers.reserve(nworkers);
        for (int i = 0; i < nworkers; i++) {
            auto w = new Worker(trees, samples, shape, num_classes);
            workers.push_back(w);
        }
        farm.add_workers(workers);

        auto collector = new Collector(shape.first, trees.size(), num_classes);
        farm.add_collector(collector);

        farm.no_mapping();
        farm.blocking_mode(false);

        if (farm.run_and_wait_end() < 0) {
            perror("Error while running farm\n");
            exit(EXIT_FAILURE);
        }

        std::vector<int> final_predictions(shape.first);
        auto& votes = collector->sample_votes;

        #pragma omp parallel for num_threads(nworkers)
        for (size_t sample_id = 0; sample_id < shape.first; sample_id++) {
            int best_class = -1;
            int max_votes = 0;
            for (int label = 0; label < votes[sample_id].size(); label++) {
                const int count = votes[sample_id][label];
                if (count > max_votes) {
                    max_votes = count;
                    best_class = label;
                }
            }
            final_predictions[sample_id] = best_class;
        }

        return final_predictions;
    }
};

#endif //DECISION_TREE_FARMFF_HPP