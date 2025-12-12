//
// Created by gabriele on 13/07/25.
//

#include "../include/Dataset.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <ranges>
#include <tuple>
#include <unordered_map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "../include/Timer.h"
using namespace std;
namespace fs = filesystem;

Dataset::Dataset() = default;

void Dataset::process_line(vector<vector<float>>& X, vector<int>& y, const string& dataset_name, const string& line, unordered_map<string, int>& labels, int &label_id) {
    if (dataset_name == "iris.data") {
        if (line.empty()) return;

        vector<string> tokens = split(line);
        const string label = tokens.back();
        tokens.pop_back();

        vector<float> features;
        for (string &token: tokens) {
            features.push_back(std::stof(token));
        }

        if (!labels.contains(label)) {
            labels[label] = label_id++;
        }

        X.push_back(features);
        y.push_back(labels[label]);
    }else if (dataset_name == "SUSY.csv" || dataset_name == "HIGGS.csv") {
        if (line.empty()) return;

        size_t start = 0;
        size_t end = line.find(',');

        int label = 0;
        auto [_, e] = from_chars(line.data() + start, line.data() + end, label);

        if (e != errc()) {
            label = 0;
        }
        y.push_back(label);

        vector<float> features;
        features.reserve(18);

        while (end != string::npos) {
            start = end + 1;
            end = line.find(',', start);
            float val = 0.0f;
            auto [_, e] = from_chars(line.data() + start,
                                                line.data() + (end == string::npos ? line.size() : end),
                                                val);
            if (e != errc()) {
                val = 0.0f;
            }
            features.push_back(val);
        }

        X.push_back(std::move(features));
    }else if (dataset_name == "criteo.txt") {
        if (line.empty()) return;

        size_t start = 0;
        size_t end = line.find('\t');

        int label = 0;
        auto [_, e] = from_chars(line.data() + start, line.data() + end, label);
        if (e != errc()) {
            label = 0;
        }
        y.push_back(label);

        vector<float> features;
        features.reserve(13 + 26);

        for (int i = 0; i < 13 && end != string::npos; i++) {
            start = end + 1;
            end = line.find('\t', start);

            float val = 0.0f;
            auto [__, e2] = from_chars(line.data() + start,
                                       line.data() + (end == string::npos ? line.size() : end),
                                       val);
            if (e2 != errc()) {
                val = 0.0f;
            }
            features.push_back(val);
        }

        for (int i = 0; i < 26 && end != string::npos; i++) {
            start = end + 1;
            end = line.find('\t', start);

            string hash_str = line.substr(start, (end == string::npos ? line.size() : end) - start);

            if (!labels.contains(hash_str)) {
                labels[hash_str] = label_id++;
            }

            features.push_back(static_cast<float>(labels[hash_str]));
        }

        X.push_back(std::move(features));
    }

}

pair<vector<vector<float> >, vector<int>> Dataset::load(string filename,
                              const string& directory,
                              const size_t max_lines) {

    static const std::unordered_map<std::string, std::string> files = {
        {"iris", "iris.data"},
        {"susy", "SUSY.csv"},
        {"higgs", "HIGGS.csv"},
        {"criteo", "criteo.txt"}
    };

    if (files.contains(filename)) {
        filename = files.at(filename);
    }else {
        cerr << "Error: File name (" << filename << ") not correctly read." << endl;
        exit(EXIT_FAILURE);
    }

    const fs::path filepath = fs::path(directory) / filename;

    const int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) {
        perror("Error while opening file");
        exit(1);
    }

    struct stat sb{};
    if (fstat(fd, &sb) == -1) {
        perror("Errore stat");
        close(fd);
        exit(1);
    }

    if (sb.st_size == 0) {
        cerr << "Empty file\n";
        close(fd);
        exit(1);
    }

    const size_t filesize = sb.st_size;

    void* file_data = mmap(nullptr, filesize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file_data == MAP_FAILED) {
        perror("Errore mmap");
        close(fd);
        return {};
    }
    madvise(file_data, filesize, MADV_SEQUENTIAL | MADV_WILLNEED);

    const char* data = static_cast<char*>(file_data);
    size_t line_start = 0;
    size_t line_count = 0;

    vector<vector<float>> features;
    vector<int> labels;
    unordered_map<string, int> labels_mapping;
    int label_id = 0;

    for (size_t i = 0; i < filesize && line_count < max_lines; ++i) {
        if (data[i] == '\n' || i == filesize - 1) {
            const size_t line_end = data[i] == '\n' ? i : i + 1;
            string line(data + line_start, line_end - line_start);
            process_line(features, labels, filename, line, labels_mapping, label_id);

            ++line_count;
            line_start = i + 1;
        }
    }

    munmap(file_data, filesize);
    close(fd);
    return make_pair(features, labels);
}

tuple<vector<vector<float> >, vector<int>, vector<vector<float> >, vector<int> >
Dataset::train_test_split(vector<vector<float> > &X,
                          vector<int> &y,
                          const float train_perc, const bool stratified) {
    const int n_samples = X.size();

    vector<vector<float> > X_train, X_test;
    vector<int> y_train, y_test;

    if (!stratified) {
        shuffle_data(X, y);

        const int train_size = static_cast<int>(n_samples * train_perc);

        X_train.reserve(train_size);
        y_train.reserve(train_size);
        X_test.reserve(n_samples - train_size);
        y_test.reserve(n_samples - train_size);

        for (int i = 0; i < n_samples; i++) {
            if (i < train_size) {
                X_train.push_back(X[i]);
                y_train.push_back(y[i]);
            } else {
                X_test.push_back(X[i]);
                y_test.push_back(y[i]);
            }
        }
    } else {
        unordered_map<int, vector<int> > class_indices;
        for (int i = 0; i < y.size(); ++i) {
            class_indices[y[i]].push_back(i);
        }

        random_device rd;
        mt19937 gen(rd());

        for (auto &indices: class_indices | views::values) {
            ranges::shuffle(indices, gen);
            const int train_count = static_cast<int>(indices.size() * train_perc);

            for (int i = 0; i < indices.size(); ++i) {
                const int idx = indices[i];
                if (i < train_count) {
                    X_train.push_back(X[idx]);
                    y_train.push_back(y[idx]);
                } else {
                    X_test.push_back(X[idx]);
                    y_test.push_back(y[idx]);
                }
            }
        }
    }

    return make_tuple(X_train, y_train, X_test, y_test);
}

vector<string> Dataset::split(const string &s, const char delim) {
    stringstream ss(s);
    vector<string> tokens;
    string value;
    while (getline(ss, value, delim)) {
        tokens.push_back(value);
    }
    return tokens;
}

void Dataset::shuffle_data(vector<vector<float> > &X, vector<int> &y) {
    for (int i = y.size() - 1; i >= 0; i--) {
        const int j = rand() % (i + 1);
        swap(X[i], X[j]);
        swap(y[i], y[j]);
    }
}
