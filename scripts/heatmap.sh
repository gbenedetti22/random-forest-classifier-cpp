#!/bin/bash
#SBATCH --job-name=pippo_strong
#SBATCH --output=logs/strong_%A_%a.out
#SBATCH --error=logs/strong_%A_%a.err
#SBATCH --nodes=8
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --array=0-19

APP_DIR="/home/g.benedetti2/home/g.benedetti2/spm/decision_tree/cmake-build-spmcluster_release"
APP="./mpi"

# Strong scaling su 8 nodi:
# 1 processo MPI per nodo, vario solo il numero di thread OpenMP
# Numero di alberi costante (1000)

CONFIGS=(
    "2 2"
    "2 4"
    "2 8"
    "2 16"
    "2 32"
    "4 2"
    "4 4"
    "4 8"
    "4 16"
    "4 32"
    "6 2"
    "6 4"
    "6 8"
    "6 16"
    "6 32"
    "8 2"
    "8 4"
    "8 8"
    "8 16"
    "8 32"
)

# Leggi la configurazione corrente
# shellcheck disable=SC2206
config=(${CONFIGS[$SLURM_ARRAY_TASK_ID]})
nodes=${config[0]}
threads=${config[1]}
total_cores=$((nodes * threads))

echo "=== Strong Scaling Test ==="
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Nodi: ${nodes}"
echo "Processi MPI: ${nodes} (1 per nodo)"
echo "Thread OpenMP per processo: ${threads}"
echo "Core totali: ${total_cores}"
echo "Numero di alberi totali: ${TOTAL_TREES}"
echo "==========================="

LOGFILE="${APP_DIR}/logs/strong_N${nodes}_t${threads}_C${total_cores}.log"

cd ${APP_DIR} && srun --mpi=pmix \
    -N "${nodes}" \
    -n "${nodes}" \
    --ntasks-per-node=1 \
    ${APP} --dataset susy --trees 1000 --seed 24 --njobs "${threads}" > "${LOGFILE}" 2>&1

echo "Completato con exit code: $?"
