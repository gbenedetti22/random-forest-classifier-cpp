# =====================================================
# Benedetti Gabriele - 602202
# =====================================================
BUILD_DIR := ./cmake-build-spmcluster_release
SM_BIN := $(BUILD_DIR)/main
MPI_BIN := $(BUILD_DIR)/mpi

# Default parameters for run command
dataset ?= iris
trees ?= 100
jobs ?= -1
nodes ?= 2  # MPI only

.PHONY: configure all build build-mpi clean tests-all tests tests-mpi run run-mpi help

# =====================================================
# Build
# =====================================================
configure:
	cmake -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" -S . -B $(BUILD_DIR)

all: build build-mpi

build: configure ## Build the shared-memory version
	@echo ">> Building shared memory version..."
	cmake --build $(BUILD_DIR) --target main -- -j 6 --no-print-directory

build-mpi: configure ## Build the MPI version
	@echo ">> Building MPI version..."
	cmake --build $(BUILD_DIR) --target mpi -- -j 6 --no-print-directory

clean: ## Clean built files
	@echo ">> Cleaning..."
	cmake --build $(BUILD_DIR) --target clean -- -j 6 --no-print-directory

# =====================================================
# Testing - Shared Memory
# =====================================================

test-iris-sm: ## Run IRIS (100 trees, 1 thread)
	cd $(BUILD_DIR) && ./main --dataset iris --trees 100 --seed 42 --njobs 1

test-iris-sm-par: ## Run IRIS (100 trees, all threads)
	cd $(BUILD_DIR) && ./main --dataset iris --trees 100 --seed 42 --njobs -1

test-iris-xl: ## Run IRIS (1000 trees, 1 thread)
	cd $(BUILD_DIR) && ./main --dataset iris --trees 1000 --seed 42 --njobs 1

test-iris-xl-par: ## Run IRIS (1000 trees, all threads)
	cd $(BUILD_DIR) && ./main --dataset iris --trees 1000 --seed 42 --njobs -1

# -----------------------------------------------------

test-susy-sm: ## Run SUSY (100 trees, 1 thread)
	cd $(BUILD_DIR) && ./main --dataset susy --trees 5 --seed 24 --njobs 1

test-susy-sm-par: ## Run SUSY (100 trees, all threads)
	cd $(BUILD_DIR) && ./main --dataset susy --trees 100 --seed 24 --njobs -1

test-susy-xl: ## Run SUSY (1000 trees, 1 thread)
	cd $(BUILD_DIR) && ./main --dataset susy --trees 10 --seed 24 --njobs 1

test-susy-xl-par: ## Run SUSY (1000 trees, all threads)
	cd $(BUILD_DIR) && ./main --dataset susy --trees 1000 --seed 24 --njobs -1

test-susy-hybrid: ## Run SUSY (1000 trees, njobs=20, nworkers=4)
	cd $(BUILD_DIR) && ./main --dataset susy --trees 1000 --seed 24 --njobs 20 --nworkers 4

# =====================================================
# Testing - MPI
# =====================================================

test-mpi-susy-2nodes: ## Run MPI on SUSY (1000 trees, 2 nodes)
	cd $(BUILD_DIR) && srun --mpi=pmix -N 2 -n 2 --ntasks-per-node=1 ./mpi --dataset susy --trees 1000 --seed 24 --njobs -1

test-mpi-susy-4nodes: ## Run MPI on SUSY (1000 trees, 4 nodes)
	cd $(BUILD_DIR) && srun --mpi=pmix -N 4 -n 4 --ntasks-per-node=1 ./mpi --dataset susy --trees 1000 --seed 24 --njobs -1

# =====================================================
# Target aggregati test
# =====================================================

tests: build test-iris-sm test-iris-sm-par test-susy-sm test-susy-sm-par ## Run all shared-memory tests
tests-mpi: build test-mpi-susy-2nodes test-mpi-susy-4nodes ## Run all MPI tests
tests-all: tests tests-mpi ## Run all tests (SM + MPI)

# =====================================================
# Custom Run
# =====================================================

run: build ## Run shared-memory version with custom parameters (eg. "make run dataset=iris trees=10 jobs=1")
	cd $(BUILD_DIR) && ./main --dataset $(dataset) --trees $(trees) --seed 27 --njobs $(jobs) -v

run-mpi: build-mpi ## Run MPI version with custom parameters (eg. "make run-mpi nodes=2 dataset=iris trees=10 jobs=1")
	cd $(BUILD_DIR) && srun --mpi=pmix -N $(nodes) -n $(nodes) --ntasks-per-node=1 ./mpi --dataset $(dataset) --trees $(trees) --seed 27 --njobs $(jobs) -v

# =====================================================
# Help
# =====================================================

help: ## Show this help message
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'
