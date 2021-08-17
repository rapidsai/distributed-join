# Distributed Join Project

## Overview

This proof-of-concept repo implements the distributed repartitioned join algorithm. The algorithm consists of three steps:
1. Hash partition: reorder input tables into partitions based on the hash values of the key columns.
2. All-to-all communication: send each partition to the corresponding MPI rank so that rows with the same hash values end up in the same rank.
3. Local join: each MPI rank performs local join independently.

For more information about the algorithm used and optimizations, please refer to [the ADMS'21 paper](http://www.adms-conf.org/2021-camera-ready/gao_adms21.pdf) and [the presentatiton](http://www.adms-conf.org/2021-camera-ready/gao_presentation.pdf).

For production-quality distributed join implementation, checkout [cuDF's Dask integration](https://rapids.ai/dask.html).

The following plot shows the weak-scaling performance when joining the `l_orderkey` column from lineitem table with the `o_orderkey` and the `o_orderpriority` columns from the orders table on TPC-H dataset with SF100k.

![weak scaling performance](/doc/tpch_perf.svg)

## Compilation

This project depends on CUDA, UCX, NCCL, MPI, cuDF 0.19 and nvcomp 2.0.

To compile, make sure the variables `CUDA_ROOT`, `CUDF_ROOT`, `MPI_ROOT`, `UCX_ROOT`, `NCCL_ROOT` and `NVCOMP_ROOT` are pointing to the installation path of CUDA, cuDF, MPI, UCX, NCCL and nvcomp repectively.

[The wiki page](https://github.com/rapidsai/distributed-join/wiki/How-to-compile-and-run-the-code) contains step-by-step instructions for setting up the environment.

To compile, run
```bash
mkdir build && cd build
cmake ..
make -j
```

## Running

To run on systems not needing Infiniband (e.g. single-node DGX-2):

```bash
UCX_MEMTYPE_CACHE=n UCX_TLS=sm,cuda_copy,cuda_ipc mpirun -n 16 --cpus-per-rank 3 bin/benchmark/distributed_join
```

On systems needing Infiniband communication (e.g. single or multi-node DGX-1Vs):

* GPU-NIC affinity is critical on systems with multiple GPUs and NICs, please refer to [this page from QUDA](https://github.com/lattice/quda/wiki/Multi-GPU-Support#maximizing-gdr-performance) for more detailed info. Also, you could modify run script included in the benchmark folder.
* Depending on whether you're running with `srun` or `mpirun`, update `run_sample.sh` to set `lrank` to `$SLURM_LOCALID` or `$OMPI_COMM_WORLD_LOCAL_RANK` correspondingly.

Example run on a single DGX-1V (all 8 GPUs):
```bash
$ mpirun -n 8 --bind-to none --mca btl ^openib,smcuda benchmark/run_sample.sh
rank 0 gpu list 0,1,2,3,4,5,6,7 cpu bind 1-4 ndev mlx5_0:1
rank 1 gpu list 0,1,2,3,4,5,6,7 cpu bind 5-8 ndev mlx5_0:1
rank 2 gpu list 0,1,2,3,4,5,6,7 cpu bind 10-13 ndev mlx5_1:1
rank 3 gpu list 0,1,2,3,4,5,6,7 cpu bind 15-18 ndev mlx5_1:1
rank 4 gpu list 0,1,2,3,4,5,6,7 cpu bind 21-24 ndev mlx5_2:1
rank 6 gpu list 0,1,2,3,4,5,6,7 cpu bind 30-33 ndev mlx5_3:1
rank 7 gpu list 0,1,2,3,4,5,6,7 cpu bind 35-38 ndev mlx5_3:1
rank 5 gpu list 0,1,2,3,4,5,6,7 cpu bind 25-28 ndev mlx5_2:1
Device count: 8
Rank 4 select 4/8 GPU
Device count: 8
Rank 5 select 5/8 GPU
Device count: 8
Rank 3 select 3/8 GPU
Device count: 8
Rank 7 select 7/8 GPU
Device count: 8
Rank 0 select 0/8 GPU
Device count: 8
Rank 1 select 1/8 GPU
Device count: 8
Rank 2 select 2/8 GPU
Device count: 8
Rank 6 select 6/8 GPU
========== Parameters ==========
Key type: int64_t
Payload type: int64_t
Number of rows in the build table: 800 million
Number of rows in the probe table: 800 million
Selectivity: 0.3
Keys in build table are unique: true
Over-decomposition factor: 1
Communicator: UCX
Registration method: preregistered
Compression: false
================================
Elasped time (s) 0.392133
```

For the arguments accepted by each benchmark, please refer to the source files in the `benchmark` folder.

## Code formatting

This repo uses `clang-format` for code formatting. To format the code, make sure `clang-format` is installed and run
```bash
./run-clang-format.py -p <path to clang-format>
```
