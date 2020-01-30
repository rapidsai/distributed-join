# Distributed Join Project

## Compilation

This project depends on CUDA, UCX, MPI and cuDF.

The Makefile uses `pkg-config` to determine the installation path of UCX, so make sure `ucx.pc` is in `PKG_CONFIG_PATH`.

To query the size of cuDF dtype, this project uses `gdf_dtype_size`, which is removed in cuDF upstream recently. For now, you could use [my custom branch](https://github.com/gaohao95/cudf/tree/improve-join-perf-with-size), which brings `gdf_dtype_size` back. In the long term, the dtype size should be queried using [Jake's column redesign](https://github.com/rapidsai/cudf/pull/2207).

To compile, make sure the variables `CUDA_HOME`, `CUDF_HOME`, `MPI_HOME` are pointing to the installation path of CUDA, cuDF and MPI, repectively, and then run

```bash
make -j
```

## Running

To run on systems not needing Infiniband (e.g. single-node DGX-2):

Make sure you are using `UCXCommunicator` for communication and then run

```bash
UCX_MEMTYPE_CACHE=n UCX_RNDV_SCHEME=get_zcopy UCX_TLS=sm,cuda_copy,cuda_ipc mpirun -n 16 --cpus-per-rank 3 benchmark/distributed_join
```

On systems needing Infiniband communication (e.g. single or multi-node DGX-1Vs):

Make sure you are using `UCXBufferCommunicator` for reusing communication buffer.

GPU-NIC affinity is critical on systems with multiple GPUs and NICs, please refer to [this page from QUDA](https://github.com/lattice/quda/wiki/Multi-GPU-Support#maximizing-gdr-performance) for more detailed info. Also, you could modify run script included in the benchmark folder.

Example run on a single DGX-1V (all 8 GPUs):
```bash
$ mpirun -n 8 --bind-to none --mca pml ucx --mca btl ^openib,smcuda benchmark/run_sample.sh
rank 0 gpu list 0,1,2,3,4,5,6,7 cpu bind 1-4 ndev mlx5_0:1
rank 1 gpu list 0,1,2,3,4,5,6,7 cpu bind 5-8 ndev mlx5_0:1
rank 2 gpu list 0,1,2,3,4,5,6,7 cpu bind 10-13 ndev mlx5_1:1
rank 3 gpu list 0,1,2,3,4,5,6,7 cpu bind 15-18 ndev mlx5_1:1
rank 4 gpu list 0,1,2,3,4,5,6,7 cpu bind 21-24 ndev mlx5_2:1
rank 5 gpu list 0,1,2,3,4,5,6,7 cpu bind 25-28 ndev mlx5_2:1
rank 6 gpu list 0,1,2,3,4,5,6,7 cpu bind 30-33 ndev mlx5_3:1
rank 7 gpu list 0,1,2,3,4,5,6,7 cpu bind 35-38 ndev mlx5_3:1
Device count: 8
Rank 2 select 2/8 GPU
Device count: 8
Rank 3 select 3/8 GPU
Device count: 8
Rank 1 select 1/8 GPU
Device count: 8
Rank 0 select 0/8 GPU
Device count: 8
Rank 7 select 7/8 GPU
Device count: 8
Rank 4 select 4/8 GPU
Device count: 8
Rank 5 select 5/8 GPU
Device count: 8
Rank 6 select 6/8 GPU
Elasped time (s) 0.490553
```

Example run on multiple DGX-1Vs:

```bash
srun --time=0:30:00 --nodes=4 --ntasks-per-node=8 benchmark/run.py
```

## File Structure

```
benchmark/
    all_to_all.cu               Benchmark the throughput of all-to-all communications
    distributed_join.cu         Benchmark the throughput of distributed-join
src/
    comm.cuh                    Communication related helper functions.
    communicator.cpp            Different implementations for the common send/recv interface definced in the header file.
    cudf_helper.cuh             cuDF related helper functions.
    distributed.cuh             Distributed algorithms. E.g. distributed table generations, distributed join, all-to-all communication etc.
    error.cuh                   Error checking macros.
test/
    buffer_communicator.cu      Test the correctness of the buffer communicator.
    compare_against_shared.cu   Test the correctness of the distributed-join compared to shared-memory implementation on random tables.
    prebuild.cu                 Test the correctness of the distributed-join compared to known solution.
```
