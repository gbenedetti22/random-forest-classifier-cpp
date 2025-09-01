cmake_minimum_required(VERSION 3.26.5)
project(decision_tree)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -funroll-loops -ffast-math")

find_package(OpenMP REQUIRED)
find_package(MPI QUIET)

add_library(libs
        src/Dataset.cpp
        src/RandomForestClassifier.cpp
        src/DecisionTreeClassifier.cpp
        src/Timer.cpp
        include/splitters/SplitterMP.hpp
        include/splitters/SplitterFF.hpp
)

target_include_directories(libs PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/eigen
)
add_subdirectory(external/matplotplusplus)

target_link_libraries(libs PUBLIC OpenMP::OpenMP_CXX)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE libs)
target_link_libraries(main PRIVATE matplot)

add_executable(mpi main_mpi.cpp)
target_link_libraries(mpi PRIVATE libs)
target_link_libraries(mpi PRIVATE matplot)

add_executable(test test.cpp)
target_link_libraries(test PRIVATE libs)

if(MPI_FOUND)
    message(STATUS "MPI found: ${MPI_C_COMPILER}")
    target_link_libraries(mpi PRIVATE MPI::MPI_CXX)
    target_compile_definitions(mpi PRIVATE MPI_AVAILABLE)
    target_link_libraries(libs PUBLIC MPI::MPI_CXX)
    target_compile_definitions(libs PUBLIC MPI_AVAILABLE)
else()
    message(STATUS "MPI not found, compiling mpi target without MPI support")
endif()
