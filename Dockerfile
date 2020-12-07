FROM nvidia/cuda:11.0-devel-ubuntu18.04
ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update -y && apt-get install -y build-essential wget cmake git vim

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
    && conda install -c rapidsai-nightly -c nvidia -c conda-forge -c defaults \
        cudf=0.17 \
        python=3.8 \
        cudatoolkit=11.0 \
        nccl \
    && conda clean --all
ENV CUDF_HOME=/conda/envs/cudf_release
ENV NCCL_HOME=${CUDF_HOME}
ENV LD_LIBRARY_PATH=${CUDF_HOME}/lib:${LD_LIBRARY_PATH}

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
ENV UCX_HOME=/usr

# Setup MPI
WORKDIR /root
ADD https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.5.tar.gz .
RUN apt-get install -y numactl libnuma-dev \
    && tar -zxf openmpi-4.0.5.tar.gz \
    && cd openmpi-4.0.5 && ./configure --prefix=/opt/openmpi-4.0.5 && make -j && make install \
    && cd /root && rm -rf openmpi-4.0.5 && rm openmpi-4.0.5.tar.gz
ENV MPI_HOME=/opt/openmpi-4.0.5
ENV PATH=${MPI_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${MPI_HOME}/lib:${LD_LIBRARY_PATH}
