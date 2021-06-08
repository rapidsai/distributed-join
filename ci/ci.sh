#!/bin/bash

set -eux

[ ! -z ${WEBHOOK_URL} ] || (echo "Missing WEBHOOK_URL" && false)
[ ! -z ${AWS_ACCESS_KEY_ID} ] || (echo "Missing AWS_ACCESS_KEY_ID" && false)
[ ! -z ${AWS_SECRET_ACCESS_KEY} ] || (echo "Missing AWS_SECRET_ACCESS_KEY" && false)

# Directory in which to do work should be done.
CI_DIR="${CI_DIR:-$PWD}"

# Unique ID for each run so that they don't clobber each other.
CI_ID="$(date +%s%N)"

NOW="$(date -Iseconds)"

function cleanup
{
    set +ue
    exec 2>&4 1>&3
    if [ -z "${SUCCESS+x}" ]; then
        MESSAGE="<!here> Latest CI errored"
    elif [ -z "${ELAPSED_TIME}" ]; then
        MESSAGE="<!here> Latest CI failed"
    else
        MESSAGE="Latest result: ${ELAPSED_TIME}"
    fi
    curl -X POST \
         -H 'Content-type: application/json' \
         --data "{\"blocks\":[{\"type\":\"section\",\"text\":{\"type\":\"mrkdwn\",\"text\":\"${MESSAGE}\"}},{\"type\":\"section\",\"text\":{\"type\":\"mrkdwn\",\"text\":\"distributed-join CI log\"},\"accessory\":{\"type\":\"button\",\"text\":{\"type\":\"plain_text\",\"text\":\"View\",\"emoji\":true},\"url\":\"https://dtcomp-data-analytics-ci.s3.amazonaws.com/ci/distributed-join-ci-$NOW.txt\",\"action_id\":\"button-action\"}}]}" "${WEBHOOK_URL}"
    aws s3 cp "${CI_DIR}/${CI_ID}/distributed-join-ci-${NOW}.txt" s3://dtcomp-data-analytics-ci/ci/
    conda deactivate
    conda env remove -y --name "join-ci-$CI_ID"
    rm -rf "${CI_DIR}/${CI_ID}"
}
trap cleanup EXIT

source "${CONDA_PREFIX}/etc/profile.d/conda.sh"

mkdir "${CI_DIR}/${CI_ID}"
# Logging as suggested here: https://serverfault.com/a/103569
exec 3>&1 4>&2
exec 1>"${CI_DIR}/${CI_ID}/distributed-join-ci-$NOW.txt" 2>&1

conda create -y --name "join-ci-${CI_ID}"
set +u && conda activate "join-ci-${CI_ID}" && set -u

module load cuda/11.0.3

set +u && conda install -c conda-forge awscli && set -u
set +u && conda install -y -c rapidsai-nightly -c nvidia -c conda-forge cudf=0.17 python=3.8 cudatoolkit=11.0 && set -u
set +u && conda install -y -c rapidsai-nightly -c nvidia -c conda-forge ucx ucx-proc=*=gpu nccl && set -u

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

export SUCCESS=1
echo done

exit 0
