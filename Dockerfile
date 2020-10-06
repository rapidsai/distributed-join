FROM nvidia/cuda:11.0-devel-ubuntu18.04
ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get -y update && apt-get install -y build-essential wget cmake git

# Install conda
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh /miniconda.sh
RUN sh /miniconda.sh -b -p /conda && /conda/bin/conda update -n base conda
ENV PATH=${PATH}:/conda/bin
# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

# Setup cuDF
WORKDIR /root/cudf
RUN git clone https://github.com/gaohao95/cudf.git /root/cudf \
    && git checkout select-hash-functions \
    && git submodule update --init --remote --recursive \
    && conda env create --name cudf_dev --file conda/environments/cudf_dev_cuda11.0.yml \
    && source activate cudf_dev \
    && conda install -c rapidsai -c nvidia -c conda-forge -c defaults -y nccl \
    && mkdir -p cpp/build \
    && cd cpp/build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -DGPU_ARCHS="70;80" \
    && make -j install \
    && conda clean -a -y
ENV CUDF_HOME=/conda/envs/cudf_dev
ENV NCCL_HOME=${CUDF_HOME}
ENV LD_LIBRARY_PATH=${CUDF_HOME}/lib:${LD_LIBRARY_PATH}

# Setup Mellanox OFED
WORKDIR /root
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        wget
RUN wget -qO - https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | apt-key add - && \
    mkdir -p /etc/apt/sources.list.d && wget -q -nc --no-check-certificate -P /etc/apt/sources.list.d https://linux.mellanox.com/public/repo/mlnx_ofed/5.0-2.1.8.0/ubuntu18.04/mellanox_mlnx_ofed.list && \
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
ADD https://github.com/openucx/ucx/releases/download/v1.9.0/ucx-v1.9.0-ubuntu18.04-mofed5.0-1.0.0.0-cuda11.0.deb .
RUN apt-get install ./ucx-v1.9.0-ubuntu18.04-mofed5.0-1.0.0.0-cuda11.0.deb -y \
    && rm ucx-v1.9.0-ubuntu18.04-mofed5.0-1.0.0.0-cuda11.0.deb
ENV UCX_HOME=/usr

# Setup MPI
RUN apt-get install -y openmpi-bin libopenmpi-dev numactl
ENV MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
