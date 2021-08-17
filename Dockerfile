FROM nvidia/cuda:11.0-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_ROOT=/usr/local/cuda
WORKDIR /

RUN apt-get update -y && apt-get install -y build-essential wget git vim libpciaccess-dev pciutils

# Install conda
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh /miniconda.sh
RUN sh /miniconda.sh -b -p /conda && /conda/bin/conda update -n base conda
ENV PATH=${PATH}:/conda/bin
# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

# Setup cuDF
RUN conda create --name cudf_release \
    && source activate cudf_release \
    && conda install -c rapidsai -c nvidia -c conda-forge -y \
        cudf=0.19 \
        python=3.8 \
        cudatoolkit=11.0 \
        openmpi \
        cmake \
    && conda clean -a -y
ENV CUDF_ROOT=/conda/envs/cudf_release
ENV LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${CUDF_ROOT}/lib:${LD_LIBRARY_PATH}
ENV PATH=${PATH}:${CUDF_ROOT}/bin

# Setup Mellanox OFED
RUN apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        wget
RUN wget -qO - https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | apt-key add - && \
    mkdir -p /etc/apt/sources.list.d && wget -q -nc --no-check-certificate -P /etc/apt/sources.list.d https://linux.mellanox.com/public/repo/mlnx_ofed/5.2-1.0.4.0/ubuntu20.04/mellanox_mlnx_ofed.list && \
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
ADD https://github.com/openucx/ucx/releases/download/v1.9.0/ucx-1.9.0.tar.gz .
RUN apt-get install -y numactl libnuma-dev file pkg-config binutils binutils-dev \
    && tar -zxf ucx-1.9.0.tar.gz && cd ucx-1.9.0 \
    && ./contrib/configure-release --enable-mt --with-cuda=/usr/local/cuda --with-rdmacm --with-verbs \
    && make -j \
    && make install \
    && cd / && rm -rf ucx-1.9.0 && rm ucx-1.9.0.tar.gz
ENV UCX_ROOT=/usr

# Setup nvcomp
RUN git clone https://github.com/NVIDIA/nvcomp && cd nvcomp && git checkout branch-2.0 && mkdir -p build && cd build \
    && ${CUDF_ROOT}/bin/cmake .. && make -j
ENV NVCOMP_ROOT=/nvcomp/build
ENV LD_LIBRARY_PATH=${NVCOMP_ROOT}/lib:${LD_LIBRARY_PATH}

# Setup NCCL
RUN git clone https://github.com/NVIDIA/nccl && cd nccl \
    && make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80"
ENV NCCL_ROOT=/nccl/build
ENV LD_LIBRARY_PATH=${NCCL_ROOT}/lib:${LD_LIBRARY_PATH}
