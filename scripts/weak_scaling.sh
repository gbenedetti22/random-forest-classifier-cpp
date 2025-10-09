#!/bin/bash
#SBATCH --job-name=pippo
#SBATCH --output=logs/weak_%A_%a.out
#SBATCH --error=logs/weak_%A_%a.err
#SBATCH --nodes=6
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --array=0-9

APP_DIR="/home/g.benedetti2/home/g.benedetti2/spm/decision_tree/cmake-build-spmcluster_release"
APP="./mpi"

# Weak scaling:
# Mantieni costante il lavoro per core, aumentando proporzionalmente gli alberi al crescere dei core totali.
# 1 processo MPI per nodo, thread OpenMP variabili

CONFIGS=(
    "1 32"     # 32 core
    "2 32"     # 64 core
    "3 32"     # 96 core
    "4 32"     # 128 core
    "5 32"     # 160 core
    "6 32"     # 192 core
)

# Numero di alberi per core nella configurazione base
BASE_TREES_PER_CORE=100

# Leggi la configurazione corrente
# shellcheck disable=SC2206
config=(${CONFIGS[$SLURM_ARRAY_TASK_ID]})
nodes=${config[0]}
threads=${config[1]}
total_cores=$((nodes * threads))

# Calcola il numero totale di alberi proporzionale al numero di core
trees=$((BASE_TREES_PER_CORE * total_cores))

echo "=== Weak Scaling Test ==="
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Nodi: ${nodes}"
echo "Processi MPI: ${nodes} (1 per nodo)"
echo "Thread OpenMP per processo: ${threads}"
echo "Core totali: ${total_cores}"
echo "Numero di alberi totali: ${trees}"
echo "=========================="

LOGFILE="${APP_DIR}/logs/weak_N${nodes}_t${threads}_C${total_cores}.log"

cd ${APP_DIR} && srun --mpi=pmix \
    -N "${nodes}" \
    -n "${nodes}" \
    --ntasks-per-node=1 \
    ${APP} --dataset susy --trees "${trees}" --seed 24 --njobs "${threads}" > "${LOGFILE}" 2>&1

echo "Completato con exit code: $?"

