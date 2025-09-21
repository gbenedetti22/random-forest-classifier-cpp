//
// Created by gabriele on 07/08/25.
//

#ifndef RADIX_SORT_INDICES_H
#define RADIX_SORT_INDICES_H

#pragma once

#include <vector>
#include <cstdint>

namespace radix_sort_detail {
    union FloatToUint32 {
        float f;
        uint32_t u;
    };


    inline uint32_t float_to_sortable_uint32(const float value) {
        FloatToUint32 converter{};
        converter.f = value;

        uint32_t result = converter.u;
        if (result & 0x80000000U) {
            result = ~result; // Numero negativo: inverti tutti i bit
        } else {
            result |= 0x80000000U; // Numero positivo: imposta bit segno a 1
        }

        return result;
    }

    struct IndexKeyPair {
        int index;
        uint64_t key;
    };

    template<int BYTE_INDEX>
    inline void counting_sort_byte(std::vector<IndexKeyPair> &pairs,
                                   std::vector<IndexKeyPair> &temp_pairs) {
        constexpr int RADIX = 256;
        int counts[RADIX] = {0};

        // Conta le occorrenze
        const int n = pairs.size();
        for (int i = 0; i < n; ++i) {
            uint8_t byte = (pairs[i].key >> (BYTE_INDEX * 8)) & 0xFF;
            ++counts[byte];
        }

        // Calcola posizioni cumulative
        int pos[RADIX];
        pos[0] = 0;
        for (int i = 1; i < RADIX; ++i) {
            pos[i] = pos[i - 1] + counts[i - 1];
        }

        // Distribuisci gli elementi
        for (int i = 0; i < n; ++i) {
            uint8_t byte = (pairs[i].key >> (BYTE_INDEX * 8)) & 0xFF;
            temp_pairs[pos[byte]++] = pairs[i];
        }

        // Copia risultato
        std::swap(pairs, temp_pairs);
    }

    // Radix sort ottimizzato per indici basato su valori double
    template<typename Container, typename ValueExtractor>
    void radix_sort_indices_impl(Container &indices, ValueExtractor extract_value) {
        const int n = indices.size();
        if (n <= 1) return;

        // Crea coppie indice-chiave
        std::vector<IndexKeyPair> pairs(n);
        for (int i = 0; i < n; ++i) {
            pairs[i].index = indices[i];
            pairs[i].key = float_to_sortable_uint32(extract_value(indices[i]));
        }

        // Buffer temporaneo per il counting sort
        std::vector<IndexKeyPair> temp_pairs(n);

        // Esegui counting sort per ogni byte (dal meno al più significativo)
        counting_sort_byte<0>(pairs, temp_pairs);
        counting_sort_byte<1>(pairs, temp_pairs);
        counting_sort_byte<2>(pairs, temp_pairs);
        counting_sort_byte<3>(pairs, temp_pairs);

        // Ricopia gli indici ordinati
        for (int i = 0; i < n; ++i) {
            indices[i] = pairs[i].index;
        }
    }
} // namespace radix_sort_detail

// Funzione principale: drop-in replacement per pdqsort con lambda
template<typename Container, typename ValueExtractor>
void radix_sort_indices(Container &indices, ValueExtractor extract_value) {
    radix_sort_detail::radix_sort_indices_impl(indices, extract_value);
}

// Versione semplificata per il caso comune: ordinare indici 0..n-1
template<typename ValueContainer>
std::vector<int> radix_sort_create_indices(const ValueContainer &values) {
    const int n = values.size();
    std::vector<int> indices(n);

    // Inizializza indici 0, 1, 2, ..., n-1
    for (int i = 0; i < n; ++i) {
        indices[i] = i;
    }

    // Ordina usando radix sort
    radix_sort_indices(indices, [&values](int idx) { return values[idx]; });

    return indices;
}

#define RADIX_SORT_INDICES(indices, X, f) \
    radix_sort_indices(indices, [&X, f](int idx) { return X(idx, f); })

#endif //RADIX_SORT_INDICES_H
