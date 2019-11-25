#include <iostream>
#include "cuda_runtime.h"
#include <cublas_v2.h>
#define BLOCK_DIM 32
#define TILE_DIM 32
#define BLOCK_ROWS 8


template<class T>
class MatrixHost {
    T *values;
    long matrix_size;
public:
    explicit MatrixHost(long n) : matrix_size(n) {
        values = new T[matrix_size * matrix_size * sizeof(T)];
    }

    void Initialize_Matrix(void) {
        time_t t;
        srand((unsigned) time(&t));
        for (int j = 0; j < matrix_size * matrix_size; j++) {
            values[j] = (T) (rand() & 0xFF) / 10.0f;
        }
    }

    T *get_values(void) {
        return values;
    }

    long get_size() {
        return sizeof(T) * matrix_size * matrix_size;
    }

    ~MatrixHost() {
        delete[]values;
    }
};


template<class T>
class MatrixDevice {
    T *values;
    long matrix_size;
public:
    explicit MatrixDevice(long n) : matrix_size(n) {
        cudaMalloc((T **) &values, matrix_size * matrix_size * sizeof(T));
    };

    T *get_values(void) {
        return values;
    }

    ~MatrixDevice() {
        cudaFree(values);
    }
};

template<typename type>
__global__ void IJK(const type *A, const type *B, type *C, int matrix_size) {
    int i = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int j = blockIdx.y * BLOCK_DIM + threadIdx.y;
    float temp = 0;
    for (int k = 0; k < matrix_size; k++) {
        temp += A[i * matrix_size + k] * B[k * matrix_size + j];
    }
    C[i * matrix_size + j] = temp;
}

template<typename type>
__global__ void IKJ(const type *A, const type *B, type *C, int matrix_size) {
    int i = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int k = blockIdx.y * BLOCK_DIM + threadIdx.y;
    float reg = A[i * matrix_size + k];
    for (int j = 0; j < matrix_size; j++) {
        atomicAdd(&C[i * matrix_size + j], reg * B[k * matrix_size + j]);
    }
}


template<typename type>
__global__ void KIJ(const type *A, const type *B, type *C, int matrix_size) {
    int k = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int i = blockIdx.y * BLOCK_DIM + threadIdx.y;
    float reg = A[i * matrix_size + k];
    for (int j = 0; j < matrix_size; j++) {
        atomicAdd(&C[i * matrix_size + j], reg * B[k * matrix_size + j]);
    }
}


template<typename type>
__global__ void JIK(const type *A, const type *B, type *C, int matrix_size) {
    int j = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int i = blockIdx.y * BLOCK_DIM + threadIdx.y;
    float temp = 0;
    for (int k = 0; k < matrix_size; k++) {
        temp += A[i * matrix_size + k] * B[k * matrix_size + j];
    }
    C[i * matrix_size + j] = temp;
}


template<typename type>
__global__ void JKI(const type *A, const type *B, type *C, int matrix_size) {
    int j = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int k = blockIdx.y * BLOCK_DIM + threadIdx.y;
    float reg = B[k * matrix_size + j];
    for (int i = 0; i < matrix_size; i++) {
        atomicAdd(&C[i * matrix_size + j], A[i * matrix_size + k] * reg);
    }
}

template<typename type>
__global__ void KJI(const type *A, const type *B, type *C, int matrix_size) {
    int k = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int j = blockIdx.y * BLOCK_DIM + threadIdx.y;
    float reg = B[k * matrix_size + j];
    for (int i = 0; i < matrix_size; i++) {
        atomicAdd(&C[i * matrix_size + j], A[i * matrix_size + k] * reg);
    }

}


template<typename type>
void multiply_on_host(const type *A, const type *B, type *C, long matrix_size) { //JIK mult
    for (int j = 0; j < matrix_size; j++) {
        for (int i = 0; i < matrix_size; i++) {
            type r = 0;
            for (int k = 0; k < matrix_size; k++) {
                r += A[(i*matrix_size) + k] * B[(k * matrix_size) + j];
            }
            C[(i* matrix_size) + j] = r;
        }
    }
}


template<typename type>
void checkResult(type *host_array, type *device_array, long matrix_size) {
    double epsilon = 0.5;
    int pos = 0;
    for (long i = 0; i < matrix_size; i++) {
        if (abs(device_array[i] - host_array[i]) > epsilon) {
            std::cout << "ERROR " << i<< "  " << host_array[i] <<" "<<device_array[i];
            return;
        }
    }
    std::cout << "SUCCESS\n";
}


template<typename type>
void cublas_multiply(const type *A, const type *B, type *C, long matrix_size) {
    //int dima = matrix_size,dimB =matrix_size , dimC =matrix_size;
    cudaEvent_t start_multiply, stop_multiply;
    float time_multiply;
    cudaEventCreate(&start_multiply);
    cudaEventCreate(&stop_multiply);
    float alpha = 1.0;
    float betta = 0.0;
    float *alph_point = &alpha;
    float *betta_point = &betta;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEventRecord(start_multiply);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size, matrix_size, matrix_size, alph_point, A, matrix_size, B,
                matrix_size, betta_point, C, matrix_size);
    cudaEventRecord(stop_multiply);
    cudaEventSynchronize(stop_multiply);
    cudaEventElapsedTime(&time_multiply, start_multiply, stop_multiply);
    cublasDestroy(handle);
    std::cout << "GPU PERFORMANCE: " << (matrix_size * matrix_size * matrix_size * 1000) / ((double) time_multiply)
              << "FLOPS\n";
}


template<typename type>
__global__ void transposeNaive(type *odata, const type *idata) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        odata[x * width + (y + j)] = idata[(y + j) * width + x];
    }
}


int main(int argc, char **argv) {

    long matrix_size = atol(argv[1]);
    MatrixHost<float> A_h(matrix_size);
    MatrixHost<float> B_h(matrix_size);
    MatrixHost<float> C_h(matrix_size);
    MatrixHost<float> C_from_cublas(matrix_size);


    A_h.Initialize_Matrix();
    B_h.Initialize_Matrix();

    //multiply_on_host<float>(A_h.get_values(), B_h.get_values(), C_h.get_values(), matrix_size);

    MatrixDevice<float> A_d(matrix_size);
    MatrixDevice<float> B_d(matrix_size);
    MatrixDevice<float> C_d(matrix_size);
    MatrixDevice<float> A_dt(matrix_size);


    cudaMemcpy(A_d.get_values(), A_h.get_values(), A_h.get_size(), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d.get_values(), B_h.get_values(), A_h.get_size(), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((matrix_size + block.x - 1) / block.x, (matrix_size + block.y - 1) / block.y);

    //transposeNaive<float> <<<grid,block>>>(A_dt.get_values(),A_d.get_values());

    cublas_multiply<float>(A_d.get_values(), B_d.get_values(), C_d.get_values(), matrix_size);
    //cudaEventRecord(start_device);
    cublas_multiply<float>(A_d.get_values(), B_d.get_values(), C_d.get_values(), matrix_size);
    //cudaEventRecord(stop_device);
    //cudaEventSynchronize(stop_device);
    //cudaEventElapsedTime(&time_device,start_device,stop_device);


    cudaMemcpy(C_from_cublas.get_values(), C_d.get_values(), A_h.get_size(), cudaMemcpyDeviceToHost);


    //checkResult<float>(C_h.get_values(), C_from_cublas.get_values(), matrix_size * matrix_size);
}