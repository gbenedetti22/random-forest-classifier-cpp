#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include <thread>

#include "include/Dataset.h"
#include "include/RandomForestClassifier.h"
#include "include/Timer.h"

using namespace std;

void print_duration(chrono::steady_clock::duration duration) {
    const auto total_ms = chrono::duration_cast<chrono::milliseconds>(duration).count();

    const long hours = total_ms / (1000 * 60 * 60);
    const long minutes = (total_ms % (1000 * 60 * 60)) / (1000 * 60);
    const long seconds = (total_ms % (1000 * 60)) / 1000;
    const long ms = total_ms % 1000;

    cout << "Time: ";

    bool printed_something = false;

    if (hours > 0) {
        cout << hours << "h ";
        printed_something = true;
    }

    if (minutes > 0 || printed_something) {
        cout << minutes << "m ";
    }

    if (ms > 0) {
        cout << seconds << "." << setfill('0') << setw(3) << ms << "s";
    } else {
        cout << seconds << "s";
    }
}

void serialize(const vector<double>& v, const string& filename) {
    ofstream out(filename, ios::binary);
    if (!out) throw runtime_error("Impossibile aprire file in scrittura");

    const size_t size = v.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));

    out.write(reinterpret_cast<const char*>(v.data()), size * sizeof(double));
}

vector<double> deserialize(const string& filename) {
    ifstream in(filename, ios::binary);
    if (!in) throw runtime_error("Impossibile aprire file in lettura");

    size_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));

    vector<double> v(size);
    in.read(reinterpret_cast<char*>(v.data()), size * sizeof(double));

    return v;
}
vector<pair<int,int>> threads_ratio_1 = {
    {2,1}, {4,1}, {8,1}, {16,1}, {24,1}, {32,1}, {40,1}
};


vector<pair<int,int>> threads_ratio_2 = {
    // nworkers = 2
    {2,2}, {4,2}, {8,2}, {16,2}, {20,2},

    // nworkers = 3
    {3,3}, {6,3}, {9,3}, {13,3},

    // nworkers = 4
    {4,4}, {8,4}, {10,4}
};

// int main() {
//     cout << "Num Threads: " << omp_get_max_threads() << endl << endl;
//     cout << "Loading dataset.." << endl;
//     auto [X, y] = Dataset::load("susy", "../dataset", 650000);
//
//     auto [X_train, y_train, X_test, y_test] =
//         Dataset::train_test_split(X, y, 0.7);
//
//     cout << "Training set size: " << X_train.size() << endl;
//     cout << "Test set size: " << X_test.size() << endl << endl;
//     constexpr int n_tress = 50;
//
//     vector<double> times_ratio_1, times_ratio_2;
//     for (auto [njobs, nworkers] : threads_ratio_1) {
//         RandomForestClassifier model_ratio_1({.n_trees = n_tress, .random_seed = 8, .njobs = njobs, .nworkers = nworkers});
//
//         const double start_seq = now();
//         model_ratio_1.fit(X_train, y_train);
//         const double stop_seq = now();
//         times_ratio_1.push_back(stop_seq - start_seq);
//     }
//
//     for (auto [njobs, nworkers] : threads_ratio_2) {
//         RandomForestClassifier model_ratio_2({.n_trees = n_tress, .random_seed = 8, .njobs = njobs, .nworkers = nworkers});
//
//         const double start_seq = now();
//         model_ratio_2.fit(X_train, y_train);
//         const double stop_seq = now();
//         times_ratio_2.push_back(stop_seq - start_seq);
//     }
//
//     serialize(times_ratio_1, "50_trees_times_ratio_1.txt");
//     serialize(times_ratio_2, "50_trees_times_ratio_2_plus.txt");
// }

// int main() {
//     cout << "Num Threads: " << omp_get_max_threads() << endl << endl;
//     cout << "Loading dataset.." << endl;
//     auto [X, y] = Dataset::load("susy", "../dataset");
//
//     auto [X_train, y_train, X_test, y_test] =
//         Dataset::train_test_split(X, y, 0.7);
//
//     cout << "Training set size: " << X_train.size() << endl;
//     cout << "Test set size: " << X_test.size() << endl << endl;
//     const vector n_tress_total = {5, 10, 30, 50, 80, 100};
//
//     vector<double> times_seq, times_mp_ff;
//     for (int n_trees : n_tress_total) {
//         RandomForestClassifier model_seq({.n_trees = n_trees, .random_seed = 8, .njobs = -1});
//
//         const double start_seq = now();
//         model_seq.fit(X_train, y_train);
//         const double stop_seq = now();
//         times_seq.push_back(stop_seq - start_seq);
//     }
//
//     for (int n_trees : n_tress_total) {
//         RandomForestClassifier model_mp_ff({.n_trees = n_trees, .random_seed = 8, .njobs = 10, .nworkers = 4});
//
//         const double start_seq = now();
//         model_mp_ff.fit(X_train, y_train);
//         const double stop_seq = now();
//         times_mp_ff.push_back(stop_seq - start_seq);
//     }
//
//     serialize(times_seq, "times_seq.txt");
//     serialize(times_mp_ff, "times_openmp_plus_ff.txt");
// }

#include <matplot/matplot.h>
// int main() {
//     vector<double> times_ratio_1 = deserialize("../times_seq.txt");
//     vector<double> times_ratio_2 = deserialize("../times_openmp_plus_ff.txt");
//
//     vector<double> ratio_1(threads_ratio_1.size());
//     ranges::transform(threads_ratio_1, ratio_1.begin(),
//                       [](const pair<int, int> &p) -> double { return static_cast<double>(p.first) * p.second; });
//
//     vector<double> ratio_2(threads_ratio_2.size());
//     ranges::transform(threads_ratio_2, ratio_2.begin(),
//                       [](const pair<int, int> &p) -> double { return static_cast<double>(p.first) * p.second; });
//     for (int i = 0; i < ratio_2.size(); ++i) {
//         cout << ratio_2[i] << " - Times: " << times_ratio_2[i] << endl;
//     }
//
//     vector<pair<double, double>> ratio_time_pairs;
//     for (size_t i = 0; i < ratio_2.size(); ++i) {
//         ratio_time_pairs.emplace_back(ratio_2[i], times_ratio_2[i]);
//     }
//
//     ranges::stable_sort(ratio_time_pairs, [](const auto& a, const auto& b) {
//         return a.first <= b.first;
//     });
//
//     vector<double> sorted_ratio_2;
//     vector<double> sorted_times_ratio_2;
//     for (const auto& pair : ratio_time_pairs) {
//         sorted_ratio_2.push_back(pair.first);
//         sorted_times_ratio_2.push_back(pair.second);
//     }
//
//     using namespace matplot;
//     hold(on);
//     plot(ratio_1, times_ratio_1, "-o")->color("b").display_name("OpenMP");
//     plot(sorted_ratio_2, sorted_times_ratio_2, "-o")->color({1.0, 0.5, 0.0}).display_name("OpenMP / FF");
//
//     xlabel("N. of Trees");
//     ylabel("Training time (s)");
//     title("Random Forest Classsifier: OpenMP vs OpenMP / FF");
//     legend();
//     grid(true);
//
//     show();
//     return 0;
// }

// int main() {
//     const vector n_tress_total = {5, 10, 30, 50, 80, 100};
//     vector<double> x_data_1 = deserialize("../times_seq.txt");
//     vector<double> x_data_2 = deserialize("../times_openmp_plus_ff.txt");
//
//     cout << "Seq" << endl;
//     for (int i = 0; i < n_tress_total.size(); ++i) {
//         cout << "#" << n_tress_total[i] << ": " << x_data_1[i] << endl;
//     }
//     cout << endl;
//
//     cout << "OpenMP / FF" << endl;
//     for (int i = 0; i < n_tress_total.size(); ++i) {
//         cout << "#" << n_tress_total[i] << ": " << x_data_2[i] << endl;
//     }
//
//
//     using namespace matplot;
//     hold(on);
//     plot(n_tress_total, x_data_1, "-o")->color("b").display_name("Seq");
//     plot(n_tress_total, x_data_2, "-o")->color({1.0, 0.5, 0.0}).display_name("OpenMP / FF");
//
//     xlabel("N. of Trees");
//     ylabel("Training time (s)");
//     title("Random Forest Classsifier: OpenMP vs OpenMP / FF");
//     legend();
//     grid(true);
//
//     show();
//     return 0;
// }

int main() {
    cout << "Loading dataset.." << endl;
    auto [X, y] = Dataset::load("susy", "../dataset");
    
     auto [X_train, y_train, X_test, y_test] =
         Dataset::train_test_split(X, y, 0.7);
    
     cout << "Training set size: " << X_train.size() << endl;
     cout << "Test set size: " << X_test.size() << endl << endl;

     cout << "Training start " << endl;
     RandomForestClassifier model({.n_trees = 1, .random_seed = 8, .njobs = 1, .nworkers = 1});
    
     const auto start = chrono::steady_clock::now();
     model.fit(X_train, y_train);
     const auto end = chrono::steady_clock::now();
    
     cout << "Training end! :)" << endl;
     print_duration(end - start);
     cout << endl << endl;
    
     auto [accuracy, f1] = model.evaluate(X_test, y_test);
     cout << "Accuracy: " << accuracy << endl;
     cout << "F1 (Macro): " << f1 << endl;

     timer.summary();
    return 0;
}