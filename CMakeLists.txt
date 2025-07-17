cmake_minimum_required(VERSION 3.31)
project(dicision_tree)
set(CMAKE_CXX_STANDARD 26)

include_directories(utils)

add_executable(dicision_tree src/Dataset.cpp main.cpp
        src/RandomForestClassifier.cpp
        include/RandomForestClassifier.h
        src/DecisionTreeClassifier.cpp
        include/DecisionTreeClassifier.h
        include/FeatureSampler.h)
