//
// Created by gabriele on 17/07/25.
//

#ifndef FEATURESAMPLER_H
#define FEATURESAMPLER_H



#include <vector>
#include <algorithm>
#include <random>

using namespace std;

class FeatureSampler {
    mt19937 rng;
    
public:
    explicit FeatureSampler(const unsigned seed = random_device{}()) : rng(seed) {}
    
    vector<int> sample_features(const int total_features, const int n_features) {
        vector<int> all_features(total_features);
        iota(all_features.begin(), all_features.end(), 0);
        
        for (int i = 0; i < n_features; i++) {
            const int j = uniform_int_distribution(i, total_features - 1)(rng);
            swap(all_features[i], all_features[j]);
        }
        
        all_features.resize(n_features);
        return all_features;
    }
};



#endif //FEATURESAMPLER_H
