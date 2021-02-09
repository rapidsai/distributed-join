FROM nvidia/cuda:11.0-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_ROOT=/usr/local/cuda

RUN apt-get update -y && apt-get install -y build-essential wget git vim

# Install conda
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh /miniconda.sh
RUN sh /miniconda.sh -b -p /conda && /conda/bin/conda update -n base conda
ENV PATH=${PATH}:/conda/bin
# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

# Setup cuDF and NCCL
WORKDIR /root
RUN conda create --name cudf_release \
    && source activate cudf_release \
    && conda install -c rapidsai -c nvidia -c conda-forge -c defaults -y \
        cudf=0.17 \
        python=3.8 \
        cudatoolkit=11.0 \
        nccl \
        openmpi \
        cmake \
    && conda clean -a -y
ENV CUDF_ROOT=/conda/envs/cudf_release
ENV NCCL_ROOT=${CUDF_ROOT}
ENV LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${CUDF_ROOT}/lib:${LD_LIBRARY_PATH}
ENV PATH=${PATH}:/conda/envs/cudf_release/bin

# Setup Mellanox OFED
WORKDIR /root
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        wget
RUN wget -qO - https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | apt-key add - && \
    mkdir -p /etc/apt/sources.list.d && wget -q -nc --no-check-certificate -P /etc/apt/sources.list.d https://linux.mellanox.com/public/repo/mlnx_ofed/5.1-2.5.8.0/ubuntu18.04/mellanox_mlnx_ofed.list && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ibverbs-providers \
        ibverbs-utils \
        libibmad-dev \
        libibmad5 \
        libibumad-dev \
        libibumad3 \
        libibverbs-dev \
        libibverbs1 \
        librdmacm-dev \
        librdmacm1

# Setup UCX
WORKDIR /root
ADD https://github.com/openucx/ucx/releases/download/v1.9.0/ucx-1.9.0.tar.gz .
RUN apt-get install -y numactl libnuma-dev file pkg-config binutils binutils-dev \
    && tar -zxf ucx-1.9.0.tar.gz && cd ucx-1.9.0 \
    && ./contrib/configure-release --enable-mt --with-cuda=/usr/local/cuda --with-rdmacm --with-verbs \
    && make -j \
    && make install \
    && cd /root && rm -rf ucx-1.9.0 && rm ucx-1.9.0.tar.gz
ENV UCX_ROOT=/usr

# Setup nvcomp
WORKDIR /root
RUN git clone https://github.com/NVIDIA/nvcomp && cd nvcomp && mkdir -p build && cd build \
    && /conda/envs/cudf_release/bin/cmake .. && make -j
ENV NVCOMP_ROOT=/root/nvcomp/build
ENV LD_LIBRARY_PATH=${NVCOMP_ROOT}/lib:${LD_LIBRARY_PATH}
