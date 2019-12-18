#!/usr/bin/env python
import os
import subprocess

GPUS=[0, 1, 2, 3, 4, 5, 6, 7]

APP="benchmark/distributed_join"

# This is the list of NICs we should use for each GPU
NICS=["mlx5_0", "mlx5_0", "mlx5_1", "mlx5_1", "mlx5_2", "mlx5_2", "mlx5_3", "mlx5_3"]

# This is the list of CPU cores we should use for each GPU
# e.g., 2x20 core CPUs split into 4 threads per process with correct NUMA assignment
CPUS=["1-4", "5-8", "10-13", "15-18", "21-24", "25-28", "30-33", "35-38"]

CUDA_VISIBLE_DEVICES = ""
for igpu in GPUS[:-1]:
    CUDA_VISIBLE_DEVICES += str(igpu)
    CUDA_VISIBLE_DEVICES += ","
CUDA_VISIBLE_DEVICES += str(GPUS[-1])

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# lrank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
lrank = int(os.environ["SLURM_LOCALID"])
os.environ["UCX_NET_DEVICES"] = NICS[GPUS[lrank]] + ":1"

os.environ["UCX_MEMTYPE_CACHE"] = "n"
os.environ["UCX_RNDV_SCHEME"] = "get_zcopy"
os.environ["UCX_TLS"] = "rc,cuda_copy,cuda_ipc"
os.environ["UCX_IB_REG_METHODS"] = "rcache"

# os.execlpe("numactl", "--physcpubind=" + str(CPUS[GPUS[lrank]]), APP, os.environ)
subprocess.run("numactl --physcpubind=" + str(CPUS[GPUS[lrank]]) + " " + APP, shell=True)
