//
// Created by gabriele on 25/09/25.
//

#ifndef DECISION_TREE_LOGGER_HPP
#define DECISION_TREE_LOGGER_HPP
#include <spdlog/spdlog.h>
class Logger {
    const int level;

public:
    explicit Logger(const int level = 1)
        : level(level) {
    }

    void debug(const std::string &msg) {
        spdlog::debug(msg);
    }
};
#endif //DECISION_TREE_LOGGER_HPP