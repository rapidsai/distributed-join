#!/bin/bash

set -eux

env

[ ! -z ${WEBHOOK_URL} ] || (echo "Missing WEBHOOK_URL" && false)

# Directory in which to do work should be done.
CI_DIR="${CI_DIR:-$PWD}"

# Unique ID for each run so that they don't clobber each other.
CI_ID="$(date +%s%N)"

function cleanup
{
    set +ue
    if [ -z "${SUCCESS+x}" ]; then
        curl -X POST \
             -H 'Content-type: application/json' \
             --data '{"text":"<!here> Latest CI errored"}' "${WEBHOOK_URL}"
    fi
    [ ! -z ${SUCCESS}EBHOOK_URL} ] || (echo "Missing WEBHOOK_URL" && false)
    conda deactivate
    conda env remove -y --name "join-ci-$CI_ID"
    rm -rf "${CI_DIR}/${CI_ID}"
}
trap cleanup EXIT

mkdir "${CI_DIR}/${CI_ID}"
source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
conda create -y --name "join-ci-${CI_ID}"
set +u && conda activate "join-ci-${CI_ID}" && set -u

module load cuda/11.0.3

set +u && conda install -y -c rapidsai-nightly -c nvidia -c conda-forge -c defaults cudf=0.17 python=3.8 cudatoolkit=11.0 && set -u
set +u && conda install -y -c rapidsai-nightly -c nvidia -c conda-forge -c defaults ucx ucx-proc=*=gpu nccl && set -u

OPENMPI_VERSION=4.0.5
wget -O "${CI_DIR}/${CI_ID}/openmpi-${OPENMPI_VERSION}.tar.gz" "https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-${OPENMPI_VERSION}.tar.gz"
pushd "${CI_DIR}/${CI_ID}"
mkdir .local
tar xzvf openmpi-${OPENMPI_VERSION}.tar.gz
pushd openmpi-${OPENMPI_VERSION}
./configure --prefix "${CI_DIR}/${CI_ID}/.local"
make -j
make install
popd
popd

export MPI_HOME="${CI_DIR}/${CI_ID}/.local"
export PATH="${MPI_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MPI_HOME}/lib:${LD_LIBRARY_PATH}"

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDF_HOME=$CONDA_PREFIX
export UCX_HOME=$CUDF_HOME
export NCCL_HOME=$CUDF_HOME
export MPI_HOME=$(dirname $(dirname $(which mpirun)))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDF_HOME/lib

pushd "${CI_DIR}/${CI_ID}"
git clone https://github.com/rapidsai/distributed-join
pushd distributed-join
make -j
export GPU_COUNT=4
export ELAPSED_TIME=$(UCX_MEMTYPE_CACHE=n \
                      UCX_TLS=sm,cuda_copy,cuda_ipc \
                      mpirun \
                      -n $GPU_COUNT \
                      --cpus-per-rank 2 \
                      benchmark/distributed_join | grep "Elasped time")
popd
popd

if [ -z "${ELAPSED_TIME}" ]; then
    curl -X POST \
         -H 'Content-type: application/json' \
         --data '{"text":"<!here> Latest CI failed"}' "${WEBHOOK_URL}"
else
    curl -X POST \
         -H 'Content-type: application/json' \
         --data "{\"text\":\"Latest result: ${ELAPSED_TIME}\"}" "${WEBHOOK_URL}"
fi

export SUCCESS=1
echo done
