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

    struct PairWithKey {
        std::pair<float, int> pair;
        uint32_t key;
    };

    template<int BYTE_INDEX>
    void counting_sort_byte(std::vector<PairWithKey> &pairs,
                            std::vector<PairWithKey> &temp_pairs) {
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

    // Radix sort ottimizzato per vector<pair<float, int>>
    void radix_sort_pairs_impl(std::vector<std::pair<float, int>> &values) {
        const int n = values.size();
        if (n <= 1) return;

        // Crea strutture con chiavi di ordinamento
        std::vector<PairWithKey> pairs(n);
        for (int i = 0; i < n; ++i) {
            pairs[i].pair = values[i];
            pairs[i].key = float_to_sortable_uint32(values[i].first);
        }

        // Buffer temporaneo per il counting sort
        std::vector<PairWithKey> temp_pairs(n);

        // Esegui counting sort per ogni byte (dal meno al più significativo)
        counting_sort_byte<0>(pairs, temp_pairs);
        counting_sort_byte<1>(pairs, temp_pairs);
        counting_sort_byte<2>(pairs, temp_pairs);
        counting_sort_byte<3>(pairs, temp_pairs);

        // Ricopia le coppie ordinate
        for (int i = 0; i < n; ++i) {
            values[i] = pairs[i].pair;
        }
    }
} // namespace radix_sort_detail

// Funzione principale per ordinare vector<pair<float, int>> per il valore float
inline void radix_sort_pairs(std::vector<std::pair<float, int>> &values) {
    radix_sort_detail::radix_sort_pairs_impl(values);
}


#define RADIX_SORT_PAIRS(pairs) \
    radix_sort_pairs(pairs)

#endif //RADIX_SORT_PAIRS_H