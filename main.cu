#include <iostream>
#include "cuda_runtime.h"

#define MAX_THREADS_FOR_BLOCK 32 //For one dimension


template<class T>
class ArrayHost {
    T *values;
    long matrix_size;
public:
    explicit ArrayHost(long n) : matrix_size(n) {
        values = new T[matrix_size * matrix_size* sizeof(T)];
    }
    void Init(void){
        time_t t;
        srand((unsigned) time(&t));
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
__global__ void transpose_on_device1(type *A,type *B, long matrix_size) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    B[x*matrix_size + y] = A[y*matrix_size+x];
}

template<typename type>
__global__ void transpose_on_device2(type *A, long matrix_size) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x > y) {
        type t = A[y*matrix_size+x];
        A[y*matrix_size+x ] = A[x*matrix_size+y];
        A[x*matrix_size+y] = t;
    }
}

template<typename type>
__global__ void transpose_on_device3(type *A,type *B, long matrix_size) {
    __shared__ type submatrix[MAX_THREADS_FOR_BLOCK][MAX_THREADS_FOR_BLOCK];
    int x = blockDim.x * blockIdx.x;
    int y = blockDim.y * blockIdx.y;
    submatrix[threadIdx.y][threadIdx.x] = A[(y+threadIdx.y)*matrix_size+(x+threadIdx.x)];
    __syncthreads();
    B[ (x+threadIdx.y) * matrix_size + (y+threadIdx.x) ] = submatrix[ threadIdx.x ][threadIdx.y ];
}




template<typename type>
void transpose_on_host(type *A, type* B, long matrix_size) {
    for (long i = 0; i < matrix_size; i++) {
        for (long j = 0 ; j < matrix_size; j++) {
           B[j*matrix_size + i] = A[i*matrix_size+j];
        }

    }
}

template<typename type>
void checkResult(type *host_array, type *device_array, const long array_size) {
    double epsilon = 0.1;
    for (long i = 0; i < array_size*array_size; i++) {
            if (abs(device_array[i] - host_array[i]) > epsilon) {
                std::cout << "ERROR in "<<i<< " position";
                //break;
            }

    }
    std::cout << "SUCCESS\n";
}


int main(int argc,char** argv) {

    int matrix_size = atoi(argv[1]);

    cudaEvent_t start_host,stop_host,start_device,stop_device;
    float time_host,time_device;


    cudaEventCreate(&start_host);
    cudaEventCreate(&stop_host);
    cudaEventCreate(&start_device);
    cudaEventCreate(&stop_device);

    ArrayHost<int> A_h(matrix_size);
    A_h.Init();
    ArrayHost<int> B_h(matrix_size);

    ArrayDevice<int> A_d(matrix_size); //Place to copy
    ArrayDevice<int> B_d(matrix_size); //Result on device


    ArrayHost<int> A_from_d(matrix_size); // Reply from device



    cudaMemcpy(A_d.get_values(), A_h.get_values(), A_h.get_size(), cudaMemcpyHostToDevice);

    cudaEventRecord(start_host);
    transpose_on_host<int>(A_h.get_values(), B_h.get_values(),matrix_size);
    cudaEventRecord(stop_host);
    cudaEventSynchronize(stop_host);
    cudaEventElapsedTime(&time_host,start_host,stop_host);



    dim3 block(MAX_THREADS_FOR_BLOCK, MAX_THREADS_FOR_BLOCK);
    dim3 grid((matrix_size + block.x - 1) / block.x, (matrix_size + block.y - 1) / block.y);

    cudaEventRecord(start_device);

    transpose_on_device1<int> << < grid, block >> > (A_d.get_values(),B_d.get_values(), matrix_size);
    //transpose_on_device2<int> << < grid, block >> > (A_d.get_values(),matrix_size);
    //transpose_on_device3<int> << < grid, block >> > (A_d.get_values(),B_d.get_values(), matrix_size);
    cudaEventRecord(stop_device);
    cudaEventSynchronize(stop_device);
    cudaEventElapsedTime(&time_device,start_device,stop_device);

    cudaMemcpy(A_from_d.get_values(), B_d.get_values(), A_h.get_size(), cudaMemcpyDeviceToHost);
    //cudaMemcpy(A_from_d.get_values(), A_d.get_values(), A_h.get_size(), cudaMemcpyDeviceToHost);
    //cudaMemcpy(A_from_d.get_values(), B_d.get_values(), A_h.get_size(), cudaMemcpyDeviceToHost);

    checkResult<int>(A_from_d.get_values(), B_h.get_values(), matrix_size);

    std::cout<<"GPU BANDWIDTH: " << (A_h.get_size()*2)/(time_device*1000000)<<" GB/s\n";
    std::cout<<"Matrix size: " << A_h.get_size() <<" B\n";
    std::cout<<"TIME FOR HOST TRANSPOSE " << time_host<<" msec\n";
    std::cout<<"TIME FOR DEVICE TRANSPOSE " << time_device<<" msec\n";


}