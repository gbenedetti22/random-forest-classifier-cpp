cmake_minimum_required(VERSION 3.31)
project(dicision_tree)
set(CMAKE_CXX_STANDARD 26)

include_directories(utils)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

add_executable(dicision_tree main.cpp
        src/Dataset.cpp
        include/tabulate.hpp
        src/RandomForestClassifier.cpp
        include/RandomForestClassifier.h
        src/DecisionTreeClassifier.cpp
        include/DecisionTreeClassifier.h
        include/Timer.h
        include/pdqsort.h
        src/Timer.cpp
        include/radix_sort_indices.h
        include/BS_thread_pool.hpp
)
