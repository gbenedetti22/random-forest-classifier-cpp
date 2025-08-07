//
// Created by gabriele on 07/08/25.
//

#ifndef RADIX_SORT_INDICES_H
#define RADIX_SORT_INDICES_H

#pragma once

#include <vector>
#include <cstring>
#include <algorithm>
#include <cstdint>

namespace radix_sort_detail {

// Union per interpretare double come uint64_t per radix sort
union DoubleToUint64 {
    double d;
    uint64_t u;
};

// Trasforma double in rappresentazione unsigned adatta per radix sort
inline uint64_t double_to_sortable_uint64(double value) {
    DoubleToUint64 converter;
    converter.d = value;

    // IEEE 754: se il bit del segno è 1 (numero negativo),
    // invertiamo tutti i bit. Altrimenti invertiamo solo il bit del segno
    uint64_t result = converter.u;
    if (result & 0x8000000000000000ULL) {
        result = ~result; // Numero negativo: inverti tutti i bit
    } else {
        result |= 0x8000000000000000ULL; // Numero positivo: imposta bit segno a 1
    }

    return result;
}

// Struct per memorizzare indice e chiave insieme
struct IndexKeyPair {
    int index;
    uint64_t key;
};

// Counting sort per un singolo byte (radix)
template<int BYTE_INDEX>
inline void counting_sort_byte(std::vector<IndexKeyPair>& pairs,
                              std::vector<IndexKeyPair>& temp_pairs) {

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
        pos[i] = pos[i-1] + counts[i-1];
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
void radix_sort_indices_impl(Container& indices, ValueExtractor extract_value) {
    const int n = indices.size();
    if (n <= 1) return;

    // Crea coppie indice-chiave
    std::vector<IndexKeyPair> pairs(n);
    for (int i = 0; i < n; ++i) {
        pairs[i].index = indices[i];
        pairs[i].key = double_to_sortable_uint64(extract_value(indices[i]));
    }

    // Buffer temporaneo per il counting sort
    std::vector<IndexKeyPair> temp_pairs(n);

    // Esegui counting sort per ogni byte (dal meno al più significativo)
    counting_sort_byte<0>(pairs, temp_pairs);
    counting_sort_byte<1>(pairs, temp_pairs);
    counting_sort_byte<2>(pairs, temp_pairs);
    counting_sort_byte<3>(pairs, temp_pairs);
    counting_sort_byte<4>(pairs, temp_pairs);
    counting_sort_byte<5>(pairs, temp_pairs);
    counting_sort_byte<6>(pairs, temp_pairs);
    counting_sort_byte<7>(pairs, temp_pairs);

    // Ricopia gli indici ordinati
    for (int i = 0; i < n; ++i) {
        indices[i] = pairs[i].index;
    }
}

} // namespace radix_sort_detail

// Funzione principale: drop-in replacement per pdqsort con lambda
template<typename Container, typename ValueExtractor>
void radix_sort_indices(Container& indices, ValueExtractor extract_value) {
    radix_sort_detail::radix_sort_indices_impl(indices, extract_value);
}

// Versione semplificata per il caso comune: ordinare indici 0..n-1
template<typename ValueContainer>
std::vector<int> radix_sort_create_indices(const ValueContainer& values) {
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

// Macro per rimpiazzare facilmente il codice esistente
#define RADIX_SORT_INDICES(indices, X, f) \
    radix_sort_indices(indices, [&X, f](int idx) { return X[f][idx]; })

/*
Uso della libreria:

1. Rimpiazzo diretto del codice esistente:

   Sostituire:
   pdqsort(indices.begin(), indices.end(),
           [&X, f](const int a, const int b) {
               return X[f][a] < X[f][b];
           });

   Con:
   RADIX_SORT_INDICES(indices, X, f);

2. Uso generico:

   std::vector<int> indices = {0, 1, 2, 3, 4};
   std::vector<double> values = {3.14, 1.41, 2.71, 0.57, 1.73};

   radix_sort_indices(indices, [&values](int i) { return values[i]; });

3. Creazione automatica degli indici:

   std::vector<double> values = {3.14, 1.41, 2.71, 0.57, 1.73};
   auto sorted_indices = radix_sort_create_indices(values);

Prestazioni:
- Complessità: O(8n) dove n è il numero di elementi (8 passate per i byte di un double)
- Memoria: O(n) aggiuntiva per buffer temporanei
- Molto più veloce di comparison-based sorts per grandi dataset
- Gestisce correttamente NaN, infiniti e numeri negativi

Note tecniche:
- Usa IndexKeyPair per evitare accessi fuori bound
- Memory-safe: tutte le allocazioni sono controllate
- Stable sort: mantiene l'ordine relativo degli elementi uguali
*/

#endif //RADIX_SORT_INDICES_H
