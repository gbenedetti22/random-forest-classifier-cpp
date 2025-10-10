# A (very) fast Random Forest Classifier (C++)

## Description

This project implements an **highly optimized Random Forest Classifier in C++**, designed to deliver **state-of-the-art training speed and accuracy**, while maintaining the **same user-friendly interface** as scikit-learn.

Our implementation is engineered for:

* Efficient multithreading and memory usage
* Fast tree building with optimized split strategies
* **Better or comparable accuracy** with significantly **lower training time** compared to Python-based solutions.
* **No custom datatypes or prerequisites**: you can just load your dataset in a standard vector and enjoy!

---

## Performance

Benchmarked my implementation against scikit-learn on a dataset with:

* **3,500,000 samples**
* **100 trees**
* **8 threads**

| Library                          | Training Time     |
| -------------------------------- | ----------------- |
| C++ Random Forest (this project) | 🟢 **1m 11.763s** |
| scikit-learn        | 🔸 ~ **14m 35s**  |

With a constant memory usage (approx 1.4Gb)
<img width="800" height="600" alt="massif" src="https://github.com/user-attachments/assets/9f4ea098-b39b-46ec-bc7d-cb0a4934dd7e" />

---

## 🧪 Implementation Notes

* Memory usage is kept minimal through careful allocations and the use of quantization
* The implementation use an histogram based technique for a fast threshold computation
* For learn more about the implementation, check the [report.pdf](report.pdf) file

---

## Installation

Just clone this repository and copy all the content from src/ and include/ directory into your project, no external dependences are needed.

---

## How to Use

The interface mimics sklearn’s `RandomForestClassifier`:

```cpp
#include "RandomForestClassifier.h"

int main() {
    // 1. Load your matrix X as a simple std::vector<std::vector<float>>
    // 2. Load your label vector as std::vector<int>
    // 3. (optional) split into X_train/y_train and X_test/y_test

    RandomForestClassifier model({
        .n_trees = 100,
        .random_seed = 24,
        .njobs = -1
    });

    model.fit(X_train, y_train);
    auto [accuracy, f1] = model.score(X_test, y_test);

    //..or auto y_pred = model.predict(X_test);

    std::cout << "Accuracy: " << accuracy << std::endl;
    std::cout << "F1 Score: " << f1 << std::endl;

    return 0;
}
```

This model works also for very large matrices (e.g. 30Gb of dataset). In this case i suggest to directly load the train set in column-major format and set "transposed" to true, to avoid intermediate copies.

```cpp
#include "RandomForestClassifier.h"

int main() {
    // 1. Load your matrix X as a flatten std::vector<float> with shape=(rows, cols) in column-major format
    // 2. Load your label vector as std::vector<int>
    // 3. (optional) split into X_train/y_train and X_test/y_test

    RandomForestClassifier model({
        .n_trees = 100,
        .random_seed = 24,
        .njobs = -1
    });

    model.fit(X_train, y_train, shape, true); // this will avoid any intermediate copies
    auto [accuracy, f1] = model.score(X_test, y_test);

    std::cout << "Accuracy: " << accuracy << std::endl;
    std::cout << "F1 Score: " << f1 << std::endl;

    return 0;
}
```

---

## Available Parameters

All hyperparameters match sklearn’s naming and behavior as closely as possible, with sensible defaults:

| Parameter           | Type                       | Default    | Description                                                  |
| ------------------- | -------------------------- | ---------- | ------------------------------------------------------------ |
| `n_trees`           | `int`                      | `10`       | Number of trees in the forest                                |
| `split_criteria`    | `std::string`              | `"gini"`   | Splitting criterion (`"gini"` or `"entropy"`)                |
| `min_samples_split` | `int`                      | `2`        | Minimum number of samples required to split an internal node |
| `max_features`      | `variant<int,std::string>` | `"sqrt"`   | Number of features to consider at each split                 |
| `bootstrap`         | `bool`                     | `true`     | Whether bootstrap samples are used                           |
| `random_seed`       | `optional<int>`            | `nullopt`  | Random seed for reproducibility                              |
| `min_samples_ratio` | `float`                    | `0.2f`     | Ratio of minimum samples to split                            |
| `max_depth`         | `int`                      | `INT_MAX`  | Maximum depth of the trees                                   |
| `max_leaf_nodes`    | `size_t`                   | `SIZE_MAX` | Maximum number of leaf nodes                                 |
| `max_samples`       | `variant<size_t,float>`    | `-1.0F`    | Number or fraction of samples to draw for training each tree |
| `njobs`             | `int`                      | `1`        | Number of threads used for training (`-1` means use all)     |
| `nworkers`          | `int`                      | `1`        | If you have more threads than trees to construct, setting this can speedup a lot   |

## License

This project is released under the **MIT License**.
