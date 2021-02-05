CC=${CUDA_HOME}/bin/nvcc

CUDF_CFLAGS=-I${CUDF_HOME}/include -I${CUDF_HOME}/include/libcudf/libcudacxx
CUDF_LIBS=-L${CUDF_HOME}/lib -Xcompiler \"-Wl,-rpath-link,${CUDF_HOME}/lib\" -lcudf -lcudf_base -lcudf_join -lcudf_hash -lcudf_partitioning -lcudf_io
MPI_CFLAGS=-I${MPI_HOME}/include
MPI_LIBS=-L${MPI_HOME}/lib -lmpi
UCX_CFLAGS=-I${UCX_HOME}/include
UCX_LIBS=-L${UCX_HOME}/lib -lucs -luct -lucp
CUDA_CFLAGS=-I${CUDA_HOME}/include -arch=sm_70 --expt-extended-lambda --default-stream per-thread
CUDA_LIBS=-L${CUDA_HOME}/lib64 -lcuda -lcudart
NCCL_CFLAGS=-I${NCCL_HOME}/include
NCCL_LIBS=-L${NCCL_HOME}/lib -lnccl
NVCOMP_CFLAGS=-I${NVCOMP_HOME}/include
NVCOMP_LIBS=-L${NVCOMP_HOME}/lib -lnvcomp

CFLAGS=-g -std=c++14 ${NCCL_CFLAGS} ${MPI_CFLAGS} ${CUDA_CFLAGS} ${UCX_CFLAGS} ${CUDF_CFLAGS} ${NVCOMP_CFLAGS}
LDFLAGS=${NCCL_LIBS} ${MPI_LIBS} ${CUDA_LIBS} ${UCX_LIBS} ${CUDF_LIBS} ${NVCOMP_LIBS}

generate_dataset=generate_dataset/generate_dataset.cuh generate_dataset/nvtx_helper.cuh
src=src/comm.cuh src/error.cuh src/distribute_table.cuh src/distributed_join.cuh src/generate_table.cuh src/communicator.o src/registered_memory_resource.hpp src/strings_column.cuh

all: benchmark/distributed_join benchmark/all_to_all benchmark/tpch test/compare_against_single_gpu test/prebuild test/buffer_communicator test/string_payload

benchmark/distributed_join: benchmark/distributed_join.cu $(generate_dataset) $(src)
	$(CC) $(CFLAGS) $(LDFLAGS) -o benchmark/distributed_join benchmark/distributed_join.cu src/communicator.o

benchmark/all_to_all: benchmark/all_to_all.cu $(src)
	$(CC) $(CFLAGS) $(LDFLAGS) -o benchmark/all_to_all benchmark/all_to_all.cu src/communicator.o

benchmark/tpch: benchmark/tpch.cu $(src)
	$(CC) $(CFLAGS) $(LDFLAGS) -o benchmark/tpch benchmark/tpch.cu src/communicator.o

test/compare_against_single_gpu: test/compare_against_single_gpu.cu $(generate_dataset) $(src)
	$(CC) $(CFLAGS) $(LDFLAGS) -o test/compare_against_single_gpu test/compare_against_single_gpu.cu src/communicator.o

test/prebuild: test/prebuild.cu $(src)
	$(CC) $(CFLAGS) $(LDFLAGS) -o test/prebuild test/prebuild.cu src/communicator.o

test/buffer_communicator: test/buffer_communicator.cu $(src)
	$(CC) $(CFLAGS) $(LDFLAGS) -o test/buffer_communicator test/buffer_communicator.cu src/communicator.o

test/string_payload: test/string_payload.cu $(src)
	$(CC) $(CFLAGS) $(LDFLAGS) -o test/string_payload test/string_payload.cu src/communicator.o

src/communicator.o: src/communicator.h src/communicator.cu
	$(CC) $(CFLAGS) $(LDFLAGS) -o src/communicator.o -c src/communicator.cu

clean:
	rm -f benchmark/distributed_join benchmark/all_to_all benchmark/tpch test/compare_against_single_gpu test/prebuild src/communicator.o test/buffer_communicator test/string_payload
