#include <iostream>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <iomanip>
#include <random>

#include "Dataset.h"
#include "DecisionTreeClassifier.h"
#include "Timer.h"
#include "TrainMatrix.hpp"

using namespace std;

void bytesToHR(long long bytes) {
    // Costanti per le conversioni
    const double KB = 1024.0;
    const double MB = KB * 1024.0;
    const double GB = MB * 1024.0;

    // Conversioni
    double kb = bytes / KB;
    double mb = bytes / MB;
    double gb = bytes / GB;

    // Stampa dei risultati con 2 decimali
    std::cout << std::fixed << std::setprecision(2);
    std::cout << bytes << " bytes equivalgono a:" << std::endl;
    std::cout << kb << " KB" << std::endl;
    std::cout << mb << " MB" << std::endl;
    std::cout << gb << " GB" << std::endl;
}

int main(int argc, char **argv) {
    cout << argv[2] << endl;

    constexpr float GB = 7;
    constexpr long long MU = GB * 1024 * 1024 * 1024;

    constexpr long N_THREADS = 40;
    constexpr long COLS = 39;
    constexpr long ROWS = 45840617 + 6042135;  // + 6042135
    // constexpr int ROWS = MU / (sizeof(float) * COLS);

    constexpr long long mu_matrix = ROWS * COLS * sizeof(float);
    constexpr long long mu_indices = ROWS * sizeof(int) * N_THREADS;
    constexpr long long mu_no_indices = ROWS * COLS * sizeof(uint8_t) * N_THREADS;
    constexpr long long trees_size = 60279850 * sizeof(TreeNode);

    cout << ROWS << endl;
    cout << "Matrice principale" << endl;
    bytesToHR(mu_matrix);
    cout << endl;

    cout << "Matrice con indici" << endl;
    bytesToHR(mu_indices);
    cout << endl;
    //
    // cout << "Matrice senza indici" << endl;
    // bytesToHR(mu_no_indices);
    // cout << endl;
    //
    cout << "Peso alberi" << endl;
    bytesToHR(trees_size);
    cout << endl;

    return 0;
}
