// timer.h

#ifndef TIMER_H
#define TIMER_H

#include <iostream>
#include <chrono>
#include <unordered_map>
#include <map>
#include <string>
#include <iomanip>
#include <stdexcept>
#include "tabulate.hpp"

class Timer {
public:
    void start(const std::string_view &name) {
        if (!active) return;

        start_times_[name] = std::chrono::steady_clock::now();
    }

    void stop(const std::string_view &name) {
        if (!active) return;

        if (!start_times_.contains(name)) {
            throw std::runtime_error("Stop chiamato senza start per il timer: " + std::string(name));
        }

        const auto end_time = std::chrono::steady_clock::now();
        const auto start_time = start_times_.at(name);
        const double duration = std::chrono::duration<double>(end_time - start_time).count();

        durations_[name] += duration;

        counts_[name]++;

        if (duration > max_durations_[name]) {
            max_durations_[name] = duration;
        }

        start_times_.erase(name);
    }

    void summary() const {
        if (!active) return;

        using namespace tabulate;

        Table table;

        table.add_row({"Segmento", "Chiamate", "Ore", "Minuti", "Secondi", "T. Singolo Max (s)"});

        table[0].format()
                .font_color(Color::yellow)
                .font_style({FontStyle::bold});

        for (const auto &[name, duration]: durations_) {
            const int count = counts_.at(name);
            const double max_d = max_durations_.at(name);

            const long long total_seconds_int = static_cast<long long>(duration);
            const long long hours = total_seconds_int / 3600;
            const long long minutes = (total_seconds_int % 3600) / 60;
            const double seconds = duration - (hours * 3600) - (minutes * 60);

            std::ostringstream seconds_stream, max_stream;
            seconds_stream << std::fixed << std::setprecision(6) << seconds;
            max_stream << std::fixed << std::setprecision(6) << max_d;

            table.add_row({
                name,
                std::to_string(count),
                std::to_string(hours),
                std::to_string(minutes),
                seconds_stream.str(),
                max_stream.str()
            });
        }

        table.column(0).format().font_align(FontAlign::left); // Segmento
        table.column(1).format().font_align(FontAlign::right); // Chiamate
        table.column(2).format().font_align(FontAlign::right); // Ore
        table.column(3).format().font_align(FontAlign::right); // Minuti
        table.column(4).format().font_align(FontAlign::right); // Secondi
        table.column(5).format().font_align(FontAlign::right); // T. Singolo Max

        table.format()
                .font_style({FontStyle::bold})
                .border_top("─")
                .border_bottom("─")
                .border_left("│")
                .border_right("│")
                .corner("┼")
                .multi_byte_characters(true);

        for (size_t i = 1; i < table.size() - 1; ++i) {
            table[i].format().hide_border_top();
        }

        table[0].format().border_bottom("─");

        std::cout << std::endl << "--- Sommario Timer ---" << std::endl;
        std::cout << table << std::endl;
    }

    void set_active(const bool active) {
        Timer::active = active;
    }

    void reset() {
        start_times_.clear();
        durations_.clear();
        counts_.clear();
        max_durations_.clear(); // Azzera anche le durate massime
    }

private:
    bool active = true;
    std::unordered_map<std::string_view, std::chrono::steady_clock::time_point> start_times_;
    std::unordered_map<std::string_view, double> durations_;
    std::unordered_map<std::string_view, int> counts_;
    std::unordered_map<std::string_view, double> max_durations_;
};

// Dichiarazione del timer globale
extern Timer timer;

#endif //TIMER_H
