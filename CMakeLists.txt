cmake_minimum_required(VERSION 3.26.5)
project(dicision_tree)
set(CMAKE_CXX_STANDARD 23)

include_directories(utils)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
find_package(OpenMP REQUIRED)

add_executable(dicision_tree main.cpp
        src/Dataset.cpp
        include/tabulate.hpp
        src/RandomForestClassifier.cpp
        include/RandomForestClassifier.h
        src/DecisionTreeClassifier.cpp
        include/DecisionTreeClassifier.h
        include/Timer.h
        src/Timer.cpp
        include/radix_sort_indices.h
        include/splitters/BaseSplitter.hpp
        include/splitters/SplitterFF.hpp
        include/splitters/SequentialSplitter.hpp
)

target_link_libraries(dicision_tree OpenMP::OpenMP_CXX)
target_include_directories(dicision_tree PRIVATE ${CMAKE_SOURCE_DIR}/include)