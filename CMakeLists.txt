cmake_minimum_required(VERSION 3.31)
project(dicision_tree)
set(CMAKE_CXX_STANDARD 26)

include_directories(utils)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

add_executable(dicision_tree main.cpp
        src/Dataset.cpp
        include/csv.hpp
        include/tabulate.hpp
        src/RandomForestClassifier.cpp
        include/RandomForestClassifier.h
        src/DecisionTreeClassifier.cpp
        include/DecisionTreeClassifier.h
        include/Timer.h
        include/pdqsort.h
        include/RadixSort.h
        src/Timer.cpp
        include/RadixSort.h
)
