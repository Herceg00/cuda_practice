#include <iostream>
#include "cuda_runtime.h"
#include "stdio.h"
//#include "/usr/local/cuda-10.1/include/cuda_runtime.h"
#include "assert.h"

template<class T>
class ArrayHost {
    T *values;
    long elem_numb;
public:
    explicit ArrayHost(long n) : elem_numb(n) {
        cudaMallocHost((T**)&values,elem_numb*sizeof(T));
    }
    void Init(void) {
        time_t t;
        srand((unsigned) time(&t));
        for (int i = 0; i < elem_numb; i++) {
            values[i] = (T) (rand() & 0xFF) / 10.0f;
        }
    }

    T *get_values(void) {
        return values;
    }

    long get_size() {
        return sizeof(T) * elem_numb;
    }

    ~ArrayHost() {
        cudaFreeHost(values);
    }
};


template<class T>
class ArrayDevice {
    T *values;
    long elem_numb;
public:
    explicit ArrayDevice(long n) : elem_numb(n) {
        cudaMalloc((T**)&values,elem_numb*sizeof(T));
    };

    T *get_values(void) {
        return values;
    }

    ~ArrayDevice() {
        cudaFree(values);
    }
};

template <typename type>
void sum_on_host(type *A,type *B, type *C,long N){
    for(long i =0;i<N;i++){
        C[i] = A[i] + B[i];
    }
}

template <typename type>
void checkResult(type *host_array,type *device_array,const long array_size){
    double epsilon  = 1;
    for (long i=0;i<array_size;i++){
        if(abs(device_array[i] - host_array[i]) > epsilon){
            std::cout<<"ERROR in %ld position"<<i;
            break;
        }
    }
    std::cout<<"SUCCESS\n";
}

template <typename type>
__global__ void sum_on_device(type *A, type *B, type *C, long array_size,long offset) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < array_size) {
        C[i+offset] = A[i+offset] + B[i+offset];
    }
}

int main(int argc, char **argv) {

    long array_size = atoi(argv[1]);
    long n_streams = atoi(argv[2]);
    long block_for_stream = array_size / n_streams;

    ArrayHost <float>A_h(array_size);
    ArrayHost <float>B_h(array_size);
    ArrayHost <float>C_h(array_size);
    ArrayHost <float>C_from_d(array_size);

    A_h.Init();
    B_h.Init();
    sum_on_host<float>(A_h.get_values(),B_h.get_values(),C_h.get_values(),array_size);

    ArrayDevice <float>A_d(array_size) ;
    ArrayDevice <float>B_d(array_size);
    ArrayDevice <float>C_d(array_size);
    cudaStream_t *stream_array = (cudaStream_t *)malloc(n_streams * sizeof(cudaStream_t));


    for (int i = 0; i < n_streams; i++) {
        cudaStreamCreate(&stream_array[i]);
    }

    int max_threads_for_block = 1024;
    dim3 block(max_threads_for_block);
    dim3 grid ((array_size + max_threads_for_block - 1)/ max_threads_for_block);

    for (int i = 0; i < n_streams; i++) {
        long offset = i * block_for_stream;
        cudaMemcpyAsync(&((A_d.get_values())[offset]),&((A_h.get_values())[offset]),block_for_stream* sizeof(float),cudaMemcpyHostToDevice,stream_array[i]);
        cudaMemcpyAsync(&((B_d.get_values())[offset]),&((B_h.get_values())[offset]),block_for_stream* sizeof(float),cudaMemcpyHostToDevice,stream_array[i]);
        sum_on_device<float><<<grid,block>>>(A_d.get_values(),B_d.get_values(),C_d.get_values(),array_size,offset);
        cudaMemcpyAsync(&((C_from_d.get_values())[offset]),&((C_d.get_values())[offset]),block_for_stream* sizeof(float),cudaMemcpyDeviceToHost,stream_array[i]);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < n_streams; i++) {
        cudaStreamDestroy(stream_array[i]);
    }

    free(stream_array);
    checkResult(C_h.get_values(),C_from_d.get_values(),array_size);

    return 0;
}