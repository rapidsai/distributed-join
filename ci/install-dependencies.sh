#!/bin/bash

set -exo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update -y && apt-get install -y build-essential wget cmake git vim
wget -O /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh /miniconda.sh -b -p /conda && /conda/bin/conda update -y -n base conda
export PATH=${PATH}:/conda/bin

cd /root
conda create -y --name join
source activate join
conda install -y -c rapidsai -c nvidia -c conda-forge -c defaults cudf=0.18 python=3.8 cudatoolkit=11.0
conda install -y -c rapidsai -c nvidia -c conda-forge -c defaults ucx ucx-proc=*=gpu nccl openmpi
conda install -y -c conda-forge -c defaults cmake

cd /root
git clone https://github.com/NVIDIA/nvcomp && cd nvcomp
mkdir -p build && cd build && cmake ..
make -j
