#ifndef DECISION_TREE_LOGGER_HPP
#define DECISION_TREE_LOGGER_HPP
#pragma once

#include <spdlog/spdlog.h>
#include <string>
#include <atomic>

class Logger {
public:
    // Inizializza il logger con il pattern di default
    static void init() {
        if (!s_initialized) {
            spdlog::set_pattern("[%t %H:%M:%S] [%^%l%$] %v");
            s_enabled = true;
            s_initialized = true;
        }
    }

    // Abilita/disabilita l'output
    static void set_enable(const bool enabled) {
        ensureInitialized();
        s_enabled = enabled;
    }

    // Imposta un pattern personalizzato
    static void setPattern(const std::string& pattern) {
        ensureInitialized();
        spdlog::set_pattern(pattern);
    }

    // Metodi di logging
    template<typename... Args>
    static void trace(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        ensureInitialized();
        if (s_enabled) {
            spdlog::trace(fmt, std::forward<Args>(args)...);
        }
    }

    template<typename... Args>
    static void debug(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        ensureInitialized();
        if (s_enabled) {
            spdlog::debug(fmt, std::forward<Args>(args)...);
        }
    }

    template<typename... Args>
    static void info(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        ensureInitialized();
        if (s_enabled) {
            spdlog::info(fmt, std::forward<Args>(args)...);
        }
    }

    template<typename... Args>
    static void warn(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        ensureInitialized();
        if (s_enabled) {
            spdlog::warn(fmt, std::forward<Args>(args)...);
        }
    }

    template<typename... Args>
    static void error(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        ensureInitialized();
        if (s_enabled) {
            spdlog::error(fmt, std::forward<Args>(args)...);
        }
    }

    template<typename... Args>
    static void critical(spdlog::format_string_t<Args...> fmt, Args&&... args) {
        ensureInitialized();
        if (s_enabled) {
            spdlog::critical(fmt, std::forward<Args>(args)...);
        }
    }

private:
    static void ensureInitialized() {
        if (!s_initialized) {
            init();
        }
    }

    // Usa una funzione static per ottenere le variabili
    static bool& getInitialized() {
        static bool initialized = false;
        return initialized;
    }

    static bool& getEnabled() {
        static bool enabled = true;
        return enabled;
    }

    static bool& s_initialized;
    static bool& s_enabled;
};

// Definizioni che puntano alle funzioni
inline bool& Logger::s_initialized = getInitialized();
inline bool& Logger::s_enabled = getEnabled();

#endif //DECISION_TREE_LOGGER_HPP