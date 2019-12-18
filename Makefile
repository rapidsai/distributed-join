CC=nvcc

CUDA_HOME=/gpfs/sw/software/CUDA/10.1.105
CUDF_HOME=/gpfs/fs1/haog/miniconda3/envs/cudf
MPI_HOME=/gpfs/sw/software/OpenMPI/3.1.3-GCC-7.3.0-2.30-CUDA-10.1.105

CUDF_CFLAGS=-I${CUDF_HOME}/include
CUDF_LIBS=-L${CUDF_HOME}/lib -lcudf -lrmm
MPI_CFLAGS=-I${MPI_HOME}/include
MPI_LIBS=-L${MPI_HOME}/lib -lmpi -lmpi_cxx
UCX_CFLAGS=`pkg-config --cflags ucx`
UCX_LIBS=`pkg-config --libs ucx`
CUDA_CFLAGS=-I${CUDA_HOME}/include -arch=sm_70 --expt-extended-lambda --default-stream per-thread
CUDA_LIBS=-L${CUDA_HOME}/lib64 -lcuda -lcudart

CFLAGS=-g -std=c++14 ${MPI_CFLAGS} ${CUDA_CFLAGS} ${UCX_CFLAGS} ${CUDF_CFLAGS}
LDFLAGS=${MPI_LIBS} ${CUDA_LIBS} ${UCX_LIBS} ${CUDF_LIBS}

generate_dataset=generate_dataset/generate_dataset.cuh generate_dataset/nvtx_helper.cuh
src=src/cudf_helper.cuh src/comm.cuh src/error.cuh src/distributed.cuh src/communicator.o


all: benchmark/distributed_join benchmark/all_to_all test/compare_against_shared test/prebuild test/buffer_communicator

benchmark/distributed_join: benchmark/distributed_join.cu $(generate_dataset) $(src)
	$(CC) $(CFLAGS) $(LDFLAGS) -o benchmark/distributed_join benchmark/distributed_join.cu src/communicator.o

benchmark/all_to_all: benchmark/all_to_all.cu $(generate_dataset) $(src)
	$(CC) $(CFLAGS) $(LDFLAGS) -o benchmark/all_to_all benchmark/all_to_all.cu src/communicator.o

test/compare_against_shared: test/compare_against_shared.cu $(generate_dataset) $(src)
	$(CC) $(CFLAGS) $(LDFLAGS) -o test/compare_against_shared test/compare_against_shared.cu src/communicator.o

test/prebuild: test/prebuild.cu $(generate_dataset) $(src)
	$(CC) $(CFLAGS) $(LDFLAGS) -o test/prebuild test/prebuild.cu src/communicator.o

test/buffer_communicator: test/buffer_communicator.cu $(src)
	$(CC) $(CFLAGS) $(LDFLAGS) -o test/buffer_communicator test/buffer_communicator.cu src/communicator.o

src/communicator.o: src/communicator.h src/communicator.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) -o src/communicator.o -c src/communicator.cpp

clean:
	rm -f benchmark/distributed_join benchmark/all_to_all test/compare_against_shared test/prebuild src/communicator.o test/buffer_communicator
