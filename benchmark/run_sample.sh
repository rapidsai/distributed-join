#!/bin/bash

#lrank=$SLURM_LOCALID
lrank=$OMPI_COMM_WORLD_LOCAL_RANK

export CUDA_DEVICE_MAX_CONNECTIONS=1

# APP="benchmark/all_to_all --repeat 1"
APP="benchmark/distributed_join"

# this is the list of GPUs we have
GPUS=(0 1 2 3 4 5 6 7)

# This is the list of NICs we should use for each GPU
# e.g., associate GPU0,1 with MLX0, GPU2,3 with MLX1, GPU4,5 with MLX2 and GPU6,7 with MLX3
NICS=(mlx5_0 mlx5_0 mlx5_1 mlx5_1 mlx5_2 mlx5_2 mlx5_3 mlx5_3)

# This is the list of CPU cores we should use for each GPU
# e.g., 2x20 core CPUs split into 4 threads per process with correct NUMA assignment
CPUS=(1-4 5-8 10-13 15-18 21-24 25-28 30-33 35-38)

# this is the order we want the GPUs to be assigned in (e.g. for NVLink connectivity)
REORDER=(0 1 2 3 4 5 6 7)

# now given the REORDER array, we set CUDA_VISIBLE_DEVICES, NIC_REORDER and CPU_REORDER to for this mapping
export CUDA_VISIBLE_DEVICES="${GPUS[${REORDER[0]}]},${GPUS[${REORDER[1]}]},${GPUS[${REORDER[2]}]},${GPUS[${REORDER[3]}]},${GPUS[${REORDER[4]}]},${GPUS[${REORDER[5]}]},${GPUS[${REORDER[6]}]},${GPUS[${REORDER[7]}]}"
NIC_REORDER=(${NICS[${REORDER[0]}]} ${NICS[${REORDER[1]}]} ${NICS[${REORDER[2]}]} ${NICS[${REORDER[3]}]} ${NICS[${REORDER[4]}]} ${NICS[${REORDER[5]}]} ${NICS[${REORDER[6]}]} ${NICS[${REORDER[7]}]})
CPU_REORDER=(${CPUS[${REORDER[0]}]} ${CPUS[${REORDER[1]}]} ${CPUS[${REORDER[2]}]} ${CPUS[${REORDER[3]}]} ${CPUS[${REORDER[4]}]} ${CPUS[${REORDER[5]}]} ${CPUS[${REORDER[6]}]} ${CPUS[${REORDER[7]}]})


export UCX_NET_DEVICES=${NIC_REORDER[lrank]}:1
export UCX_MEMTYPE_CACHE=n
export UCX_TLS=rc,cuda_copy,cuda_ipc
export UCX_WARN_UNUSED_ENV_VARS=n
#export UCX_IB_GPU_DIRECT_RDMA=no
#export UCX_IB_REG_METHODS=rcache
#export UCX_RNDV_THRESH=8192
export UCX_RNDV_SCHEME=put_zcopy

echo "rank" $lrank "gpu list" $CUDA_VISIBLE_DEVICES "cpu bind" ${CPU_REORDER[$lrank]} "ndev" $UCX_NET_DEVICES
numactl --physcpubind=${CPU_REORDER[$lrank]} $APP
