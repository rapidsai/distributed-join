# Distributed Join Project

## Compilation

This project depends on CUDA, UCX, MPI and cuDF.

The Makefile uses `pkg-config` to determine the installation path of UCX, so make sure `ucx.pc` is in `PKG_CONFIG_PATH`.

To query the size of cuDF dtype, this project uses `gdf_dtype_size`, which is removed in cuDF upstream recently. For now, you could use [my custom branch](https://github.com/gaohao95/cudf/tree/improve-join-perf-with-size), which brings `gdf_dtype_size` back. In the long term, the dtype size should be queried using [Jake's column redesign](https://github.com/rapidsai/cudf/pull/2207).

To compile, modify the variable `CUDA_HOME`, `CUDF_HOME`, `MPI_HOME` in *Makefile* to the installation path of CUDA, cuDF and MPI, repectively, and then run

```bash
make -j
```

## Running

To run on systems not needing Infiniband (e.g. single-node DGX-2):

Make sure you are using `UCXCommunicator` for communication and then run

```bash
UCX_MEMTYPE_CACHE=n UCX_RNDV_SCHEME=get_zcopy UCX_TLS=sm,cuda_copy,cuda_ipc mpirun -n 16 --cpus-per-rank 3 benchmark/distributed_join
```

On systems needing Infiniband communication (e.g. multi-node DGX-1Vs):

Make sure you are using `UCXBufferCommunicator` for reusing communication buffer.

GPU-NIC affinity is critical on systems with multiple GPUs and NICs, please refer to [this page from QUDA](https://github.com/lattice/quda/wiki/Multi-GPU-Support#maximizing-gdr-performance) for more detailed info. Also, you could modify run script included in the benchmark folder.

Then run distributed join with

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
