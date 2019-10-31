#include <iostream>
#include "cuda_runtime.h"

#define MAX_THREADS_FOR_BLOCK 32


template<class T>
class ArrayHost {
    T *values;
    long matrix_size;
public:
    explicit ArrayHost(long n) : matrix_size(n) {
        time_t t;
        srand((unsigned) time(&t));
        values = new T[matrix_size * matrix_size* sizeof(T)];
        for (int j = 0; j < matrix_size*matrix_size; j++) {
            values[j] = (T) (rand() & 0xFF) / 10.0f;
        }
    }

    T * get_values(void) {
        return values;
    }

    long get_size() {
        return sizeof(T) * matrix_size * matrix_size;
    }

    ~ArrayHost() {
        delete []values;
    }
};


template<class T>
class ArrayDevice {
    T *values;
    long matrix_size;
public:
    explicit ArrayDevice(long n) : matrix_size(n) {
        cudaMalloc((T **) &values, matrix_size* matrix_size * sizeof(T));
    };

    T *get_values(void) {
        return values;
    }

    ~ArrayDevice() {
        cudaFree(values);
    }
};


template<typename type>
__global__ void transpose_on_device(type *A, long matrix_size) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x > y) {
        type t = A[y*matrix_size+x];
        A[y*matrix_size+x ] = A[x*matrix_size+y];
        A[x*matrix_size+y] = t;
    }
}

template<typename type>
void transpose_on_host(type *A, long matrix_size) {
    for (long i = 0; i < matrix_size; i++) {
        for (long j = i + 1; j < matrix_size; j++) {
            type t = A[j*matrix_size+i];
            A[j*matrix_size+i ] = A[i*matrix_size+j];
            A[i*matrix_size+j] = t;
        }

    }

}

template<typename type>
void checkResult(type *host_array, type *device_array, const long array_size) {
    double epsilon = 0.1;
    for (long i = 0; i < array_size*array_size; i++) {
            if (abs(device_array[i] - host_array[i]) > epsilon) {
                std::cout << "ERROR in"<<i<< "position";
                break;
            }

    }
    std::cout << "SUCCESS\n";
}


int main(void) {

    int matrix_size = 64;

    cudaEvent_t start_host,stop_host,start_device,stop_device;
    float time_host,time_device;


    cudaEventCreate(&start_host);
    cudaEventCreate(&stop_host);
    cudaEventCreate(&start_device);
    cudaEventCreate(&stop_device);

    ArrayHost<int> A_h(matrix_size);
    ArrayDevice<int> A_d(matrix_size);
    ArrayHost<int> A_from_d(matrix_size);


    cudaMemcpy(A_d.get_values(), A_h.get_values(), A_h.get_size(), cudaMemcpyHostToDevice);


    cudaEventRecord(start_host);

    transpose_on_host<int>(A_h.get_values(), matrix_size);

    cudaEventRecord(stop_host);
    cudaEventSynchronize(stop_host);

    cudaEventElapsedTime(&time_host,start_host,stop_host);



    dim3 block(MAX_THREADS_FOR_BLOCK, MAX_THREADS_FOR_BLOCK);
    dim3 grid((matrix_size + block.x - 1) / block.x, (matrix_size + block.y - 1) / block.y);

    cudaEventRecord(start_device);
    transpose_on_device<int> << < grid, block >> > (A_d.get_values(), matrix_size);
    cudaEventRecord(stop_device);
    cudaEventSynchronize(stop_device);
    cudaEventElapsedTime(&time_device,start_device,stop_device);

    cudaMemcpy(A_from_d.get_values(), A_d.get_values(), A_h.get_size(), cudaMemcpyDeviceToHost);

    checkResult<int>(A_from_d.get_values(), A_h.get_values(), matrix_size);



    std::cout<<"TIME FOR HOST TRANSPOSE " << time_host<<" sec\n";
    std::cout<<"TIME FOR DEVICE TRANSPOSE " << time_device<<" sec\n";


}
