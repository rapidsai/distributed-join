CC=nvcc

CUDA_HOME=/cm/extra/apps/CUDA.linux86-64/10.1.150_418.39
CUDF_HOME=/home/hgao/miniconda3/envs/cudf
THIRD_PARTY_HOME=/home/hgao/thirdparty-freestanding
CUB_HOME=/home/hgao/cudf/thirdparty/cub
MPI_HOME=/home/hgao/openmpi_install

CUDF_CFLAGS=-I${CUDF_HOME}/include -I${THIRD_PARTY_HOME}/include -I${CUB_HOME}
CUDF_LIBS=-L${CUDF_HOME}/lib -lcudf -lrmm
MPI_CFLAGS=-I${MPI_HOME}/include
MPI_LIBS=-L${MPI_HOME}/lib -lmpi
UCX_CFLAGS=`pkg-config --cflags ucx`
UCX_LIBS=`pkg-config --libs ucx`
CUDA_CFLAGS=-I${CUDA_HOME}/include -arch=sm_70 --expt-extended-lambda --default-stream per-thread
CUDA_LIBS=-L${CUDA_HOME}/lib64 -lcuda -lcudart

CFLAGS=-g -std=c++14 ${MPI_CFLAGS} ${CUDA_CFLAGS} ${UCX_CFLAGS} ${CUDF_CFLAGS}
LDFLAGS=${MPI_LIBS} ${CUDA_LIBS} ${UCX_LIBS} ${CUDF_LIBS}

generate_dataset=generate_dataset/generate_dataset.cuh generate_dataset/nvtx_helper.cuh
src=src/comm.cuh src/error.cuh src/distribute_table.cuh src/distributed_join.cuh src/generate_table.cuh src/communicator.o

all: benchmark/distributed_join benchmark/all_to_all test/compare_against_shared test/prebuild test/buffer_communicator

benchmark/distributed_join: benchmark/distributed_join.cu $(generate_dataset) $(src)
	$(CC) $(CFLAGS) $(LDFLAGS) -o benchmark/distributed_join benchmark/distributed_join.cu src/communicator.o

benchmark/all_to_all: benchmark/all_to_all.cu $(src)
	$(CC) $(CFLAGS) $(LDFLAGS) -o benchmark/all_to_all benchmark/all_to_all.cu src/communicator.o

test/compare_against_shared: test/compare_against_shared.cu $(generate_dataset) $(src)
	$(CC) $(CFLAGS) $(LDFLAGS) -o test/compare_against_shared test/compare_against_shared.cu src/communicator.o

test/prebuild: test/prebuild.cu $(src)
	$(CC) $(CFLAGS) $(LDFLAGS) -o test/prebuild test/prebuild.cu src/communicator.o

test/buffer_communicator: test/buffer_communicator.cu $(src)
	$(CC) $(CFLAGS) $(LDFLAGS) -o test/buffer_communicator test/buffer_communicator.cu src/communicator.o

src/communicator.o: src/communicator.h src/communicator.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) -o src/communicator.o -c src/communicator.cpp

clean:
	rm -f benchmark/distributed_join benchmark/all_to_all test/compare_against_shared test/prebuild src/communicator.o test/buffer_communicator
