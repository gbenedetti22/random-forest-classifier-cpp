#!/bin/bash
#SBATCH --job-name=pippo_strong
#SBATCH --output=logs/strong_%A_%a.out
#SBATCH --error=logs/strong_%A_%a.err
#SBATCH --nodes=6
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --array=0-9

APP_DIR="/home/g.benedetti2/home/g.benedetti2/spm/decision_tree/cmake-build-spmcluster_release"
APP="./mpi"

# Strong scaling su 8 nodi:
# 1 processo MPI per nodo, vario solo il numero di thread OpenMP
# Numero di alberi costante (1000)

CONFIGS=(
  # "1 1"      # 1 core
  "1 4"      # 4 core
  "2 4"      # 8 core
  "4 5"      # 20 core  ← probabilmente il miglior tradeoff
  "5 5"      # 25 core  ← o questo
  "8 5"      # 40 core
  "10 5"     # 50 core
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
